"""Full-card Supremus-style HUNL river DataGenerator (V1 reconstruction).

This module intentionally stores the expensive solver target in **raw chips**.
That makes the dataset independent of the unresolved network-normalization
convention (the papers say "fraction of pot size", while the released
DeepStack-Leduc generator uses an equal-per-player committed variable named
``pot_size``).  Both common normalized views can be derived losslessly from
raw targets and are exposed as convenience properties.

Evidence classes
----------------
PAPER_EXPLICIT (Zarick et al. 2020 / DeepStack supplement):
  * Supremus random subgames are generated in a manner identical to DeepStack;
  * river network is trained first;
  * 50M river samples;
  * each random subgame solved with 4,000 iterations per player of DCFR+;
  * DCFR+ delayed linear average weight max(0,t-100);
  * Table-2 Supremus action fractions;
  * input/output are bucketed to 1000 values per player for the network.

SOURCE/RELEASE-ANCHORED reconstruction choices:
  * DeepStack R(S,p) recursive range generator as implemented by
    ``author_range_v2``;
  * THRandom/Torch7 MT19937 and rejection sampling pattern from released
    DeepStack-Leduc;
  * the DeepStack supplement's printed pot-category distribution.

PROJECT-EXPLICIT (not private Supremus source):
  * equal-strength hand-id tie break;
  * anomalous [100,100) pot interval interpreted as the singleton {100};
  * nearest-half-up quantization for non-integral Supremus Table-2 chip sizes;
  * master/shard seed schedule and serialization ordering.

``AUTHOR_STRICT`` refuses to emit a sample instead of hiding those gaps.
"""
from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass

import numpy as np

from hunl.cards import HAND_COUNT, possible_hands_mask
from hunl.river_dcfr_plus import DcfrPlusSpec, RiverDcfrPlusEngine
from hunl.supremus_config import ChipQuantization, SupremusRiverConfig
from .source_contract_v2 import UnresolvedSourceAmbiguity
from .turn_datagen_v2 import (
    CountingTHRandom,
    DatagenMode,
    _Float32FloorRangeGenerator,
    sample_pot_half,
)


class RiverDatagenMode(str, enum.Enum):
    AUTHOR_STRICT = "AUTHOR_STRICT"
    PAPER_RECONSTRUCTION = "PAPER_RECONSTRUCTION"


@dataclass(frozen=True)
class RiverDatagenV1Config:
    mode: RiverDatagenMode = RiverDatagenMode.PAPER_RECONSTRUCTION
    batch_size: int = 1
    dcfr_iterations: int = 4_000
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.0
    dcfr_delay: int = 100
    chip_quantization: ChipQuantization = ChipQuantization.NEAREST_HALF_UP
    master_seed: int = 20260816
    solver_backend: str = "numba_flat"  # exact-regression against Python backend
    schema: str = "HUNL_RIVER_FULLCARD_V1_SUPREMUS_PAPER_RECONSTRUCTION"

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.dcfr_iterations <= 0:
            raise ValueError("dcfr_iterations must be positive")
        if self.mode == RiverDatagenMode.AUTHOR_STRICT:
            strict_readiness_check()


@dataclass(frozen=True)
class RiverBatchInputs:
    board: tuple[int, int, int, int, int]
    ranges: np.ndarray       # [2,batch,1326], float32
    pot_half: np.ndarray     # [batch], int32; equal committed amount/player
    masks: np.ndarray        # [batch,1326], uint8
    rng_draws_after: int
    board_rejections: int
    boundary_ties: int

    @property
    def total_pot(self) -> np.ndarray:
        return 2 * self.pot_half.astype(np.int64)


@dataclass(frozen=True)
class RiverSolvedBatch:
    inputs: RiverBatchInputs
    targets_chips: np.ndarray        # [batch,2,1326], float32
    expected_utility_residuals: np.ndarray  # [batch], float64 chips
    decision_nodes: np.ndarray       # [batch], int32
    terminal_nodes: np.ndarray       # [batch], int32

    @property
    def targets_per_pot_half(self) -> np.ndarray:
        d = self.inputs.pot_half.astype(np.float32)[:, None, None]
        return self.targets_chips / d

    @property
    def targets_per_total_pot(self) -> np.ndarray:
        d = self.inputs.total_pot.astype(np.float32)[:, None, None]
        return self.targets_chips / d


def strict_readiness_check() -> None:
    raise UnresolvedSourceAmbiguity(
        "AUTHOR_STRICT Supremus river generation is blocked: private integer-"
        "chip quantization of decimal Table-2 actions is unpublished; private "
        "RNG/seed schedule is unpublished; equal-strength R(S,p) tie order is "
        "unpublished; DeepStack [100,100) pot category is ambiguous"
    )


def shard_seed(master_seed: int, shard_idx: int) -> int:
    """PROJECT-CANONICAL deterministic seed derivation."""
    b = f"hunl-river-v1:{master_seed}:shard:{shard_idx:05d}".encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:4], "big")


def sample_river_board(rng: CountingTHRandom) -> tuple[tuple[int, ...], int]:
    """Five distinct public cards via released DeepStack-Leduc rejection style."""
    used = np.zeros(52, dtype=np.uint8)
    cards: list[int] = []
    rejects = 0
    while len(cards) < 5:
        c = rng.random_range(1, 52) - 1
        if used[c]:
            rejects += 1
            continue
        used[c] = 1
        cards.append(c)
    return tuple(cards), rejects


class RiverDataGeneratorV1:
    def __init__(self, cfg: RiverDatagenV1Config = RiverDatagenV1Config()):
        self.cfg = cfg
        if cfg.mode == RiverDatagenMode.AUTHOR_STRICT:
            strict_readiness_check()
        self.game_cfg = SupremusRiverConfig(rounding=cfg.chip_quantization)
        self.dcfr = DcfrPlusSpec(
            iterations=cfg.dcfr_iterations,
            alpha=cfg.dcfr_alpha,
            beta=cfg.dcfr_beta,
            delay=cfg.dcfr_delay,
            simultaneous=True,
        )

    def make_batch_inputs(self, rng: CountingTHRandom) -> RiverBatchInputs:
        board, rejects = sample_river_board(rng)
        rg = _Float32FloorRangeGenerator(board)
        ranges = np.empty((2, self.cfg.batch_size, HAND_COUNT), dtype=np.float32)
        ranges[0] = rg.generate(self.cfg.batch_size, rng)
        ranges[1] = rg.generate(self.cfg.batch_size, rng)
        pots = np.empty(self.cfg.batch_size, dtype=np.int32)
        for i in range(self.cfg.batch_size):
            pots[i] = sample_pot_half(rng, DatagenMode.RELEASED_CODE_ANCHORED)
        mask = possible_hands_mask(board).astype(np.uint8)
        masks = np.broadcast_to(mask, (self.cfg.batch_size, HAND_COUNT)).copy()
        return RiverBatchInputs(
            board=tuple(int(c) for c in board),
            ranges=ranges,
            pot_half=pots,
            masks=masks,
            rng_draws_after=rng.draws,
            board_rejections=rejects,
            boundary_ties=rg.boundary_ties,
        )

    def solve_batch(self, inputs: RiverBatchInputs) -> RiverSolvedBatch:
        b = self.cfg.batch_size
        raw = np.empty((b, 2, HAND_COUNT), dtype=np.float32)
        residuals = np.empty(b, dtype=np.float64)
        dn = np.empty(b, dtype=np.int32)
        tn = np.empty(b, dtype=np.int32)
        for i in range(b):
            engine = RiverDcfrPlusEngine(
                inputs.board,
                int(inputs.pot_half[i]),
                game_cfg=self.game_cfg,
                dcfr=self.dcfr,
            )
            r0 = inputs.ranges[0, i].astype(np.float64)
            r1 = inputs.ranges[1, i].astype(np.float64)
            if self.cfg.solver_backend == "numba_flat":
                result = engine.solve_numba_flat(r0, r1)
            elif self.cfg.solver_backend == "python":
                result = engine.solve(r0, r1)
            else:
                raise ValueError(f"unknown solver_backend {self.cfg.solver_backend!r}")
            raw[i] = result.root_cfvs.astype(np.float32)
            residuals[i] = result.weighted_zero_sum_residual
            dn[i] = result.decision_nodes
            tn[i] = result.terminal_nodes
        return RiverSolvedBatch(inputs, raw, residuals, dn, tn)


__all__ = [
    "RiverDatagenMode", "RiverDatagenV1Config", "RiverBatchInputs",
    "RiverSolvedBatch", "RiverDataGeneratorV1", "strict_readiness_check",
    "shard_seed", "sample_river_board",
]
