"""Source-constrained DeepStack HUNL TURN DataGenerator V2.

This is the production replacement for the historical V1 pilot.  It separates
three evidence classes rather than silently collapsing them:

AUTHOR_EXPLICIT
  * turn situations are sampled after the turn card;
  * pot category distribution printed in the DeepStack supplement;
  * R(S,p): p1 uniform, floor(|S|/2), current-public-state hand strength;
  * 1,000 CFR+ iterations; actions F/C/P/A; no card abstraction;
  * targets are root counterfactual values for both players.

RELEASED_CODE_ANCHORED (same Schmid/Moravcik DeepStack-Leduc/Torch7 family)
  * Torch7 THRandom MT19937 stream;
  * rejection sampling without replacement for public cards;
  * batch size 10 and one public board shared by a generated batch;
  * FloatTensor (float32) range-recursion arithmetic;
  * resolve_first_node -> get_root_cfv_both_players;
  * uniform post-skip averaging. Martin Schmid independently recalled that
    skip iterations were always used; the exact private-HUNL skip count was
    not stated in that email. The numerical 1000/500 schedule remains
    RELEASED-CODE-ANCHORED to DeepStack-Leduc.
  * root CFVs divided by the equal per-player committed amount.

PROJECT_CANONICAL (not author-confirmed for private HUNL generator)
  * printed [100,100) category interpreted as singleton {100};
  * equal-strength ties ordered by frozen 1326 hand id;
  * frozen 1326 lexicographic serialization ordering;
  * master/per-shard seed derivation and NumPy file format.

AUTHOR_STRICT mode intentionally refuses to produce samples until the private
ambiguities (at minimum [100,100), tie order, RNG/seed schedule) are resolved.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

import numpy as np

from datagen.th_random import THRandom
from hunl.cards import HAND_COUNT, possible_hands_mask
from hunl.config import DEFAULT_CONFIG
from hunl.turn_engine import TurnEngine
from .author_range_v2 import source_order_with_project_tiebreak
from .source_contract_v2 import (
    UnresolvedSourceAmbiguity,
    literal_integer_values,
    reconstruction_v1_integer_values,
)


class DatagenMode(str, enum.Enum):
    AUTHOR_STRICT = "AUTHOR_STRICT"
    RELEASED_CODE_ANCHORED = "RELEASED_CODE_ANCHORED"


@dataclass(frozen=True)
class TurnDatagenV2Config:
    mode: DatagenMode = DatagenMode.RELEASED_CODE_ANCHORED
    batch_size: int = 10
    cfr_iters: int = 1000
    cfr_skip_iters: int = 500
    terminal_backend: str = "rank_numba"
    master_seed: int = 20260816
    schema: str = "HUNL_TURN_DATASET_V2_SOURCE_CONSTRAINED"

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0 <= self.cfr_skip_iters < self.cfr_iters):
            raise ValueError("need 0 <= cfr_skip_iters < cfr_iters")


class CountingTHRandom(THRandom):
    """Certified Torch7 stream plus an exact state-draw ledger."""
    def __init__(self, seed: int):
        super().__init__(seed)
        self.draws = 0

    def random_u32(self) -> int:
        self.draws += 1
        return super().random_u32()

    def uniform01(self, n: int) -> np.ndarray:
        # Released range_generator.lua uses torch.rand(FloatTensor), hence
        # FloatTensor quantization is part of RELEASED_CODE_ANCHORED mode.
        x = self.rand_float(n)
        if np.any(x <= 0.0):
            # Paper says open interval (0,p). Torch7 can theoretically emit
            # exactly zero. Do not silently change an observed zero draw.
            raise UnresolvedSourceAmbiguity(
                "Torch7 emitted exactly 0 although HUNL R(S,p) specifies (0,p)"
            )
        return x


@dataclass(frozen=True)
class BatchInputs:
    board: tuple[int, int, int, int]
    ranges: np.ndarray            # [2,batch,1326], f32
    pots: np.ndarray              # [batch], int32, equal committed amount/player
    masks: np.ndarray             # [batch,1326], uint8
    rng_draws_after: int
    boundary_ties: int


@dataclass(frozen=True)
class SolvedBatch:
    inputs: BatchInputs
    targets: np.ndarray           # [batch,2,1326], f32, root CFVs / pot_half
    expected_utility_residuals: np.ndarray  # [batch], f64 chips


class _Float32FloorRangeGenerator:
    """HUNL paper split rule + released Torch7 FloatTensor arithmetic.

    The HUNL paper explicitly requires floor(|S|/2), so we intentionally do
    *not* copy DeepStack-Leduc's randomized odd-card split. Equal-strength tie
    order remains PROJECT_CANONICAL because the private HUNL code is absent.
    """
    def __init__(self, board, *, tiebreak: str = "HAND_ID_ASC"):
        self.board = tuple(int(c) for c in board)
        self.tiebreak = tiebreak
        self.order, self.strengths, self.boundary_ties = (
            source_order_with_project_tiebreak(self.board, tiebreak=tiebreak)
        )
        self.mask = possible_hands_mask(self.board)

    def generate(self, batch: int, rng: CountingTHRandom) -> np.ndarray:
        sorted_ranges = np.empty((batch, len(self.order)), dtype=np.float32)

        def rec(lo: int, hi: int, mass: np.ndarray) -> None:
            n = hi - lo
            if n == 1:
                sorted_ranges[:, lo] = mass
                return
            u = rng.uniform01(batch)  # f32, exactly as released Torch7 path
            m1 = np.multiply(mass, u, dtype=np.float32)
            m2 = np.subtract(mass, m1, dtype=np.float32)
            mid = lo + n // 2        # AUTHOR_EXPLICIT floor split
            rec(lo, mid, m1)
            rec(mid, hi, m2)

        rec(0, len(self.order), np.ones(batch, dtype=np.float32))
        out = np.zeros((batch, HAND_COUNT), dtype=np.float32)
        out[:, self.order] = sorted_ranges
        return out


def strict_readiness_check() -> None:
    """Refuse to claim an original/private HUNL sample can be reproduced."""
    blockers = [
        "published pot category [100,100) has no literal integer",
        "equal-strength hand tie ordering was not published",
        "private HUNL RNG/seed/job schedule was not published",
        "private HUNL serialization/1326 ordering was not published",
    ]
    raise UnresolvedSourceAmbiguity(
        "AUTHOR_STRICT HUNL generation is blocked: " + "; ".join(blockers)
    )


def shard_seed(master_seed: int, shard_idx: int) -> int:
    """PROJECT_CANONICAL deterministic seed derivation, never author-claimed."""
    b = f"hunl-turn-v2:{master_seed}:shard:{shard_idx:05d}".encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:4], "big")


def sample_turn_board(rng: CountingTHRandom) -> tuple[tuple[int, int, int, int], int]:
    """Released DeepStack-Leduc random_card_generator.lua pattern.

    Preserve draw order (flop cards then turn card) rather than sorting it.
    Marginally, every four-card subset is uniform; exact private HUNL sampler
    remains unpublished.
    """
    used = np.zeros(52, dtype=np.uint8)
    cards: list[int] = []
    rejects = 0
    while len(cards) < 4:
        c = rng.random_range(1, 52) - 1
        if used[c]:
            rejects += 1
            continue
        used[c] = 1
        cards.append(c)
    return (cards[0], cards[1], cards[2], cards[3]), rejects


def sample_pot_half(rng: CountingTHRandom, mode: DatagenMode) -> int:
    """Sample one printed HUNL pot category then one integer within it."""
    if mode == DatagenMode.AUTHOR_STRICT:
        strict_readiness_check()
    cat = rng.random_range(0, 4)
    if cat == 0:
        # PROJECT_CANONICAL: degenerate category. Still consume the integer
        # draw, matching the published two-stage description and Torch random
        # call shape (random_range(100,100) consumes one MT draw).
        vals = reconstruction_v1_integer_values(0)
        assert vals == (100,)
        return rng.random_range(100, 100)
    vals = literal_integer_values(cat)
    return rng.random_range(vals[0], vals[-1])


def _fcp_a_config():
    # Training targets are explicitly F/C/P/A only, unlike the online turn
    # Table-4 first action which also contains half-pot.
    return dataclasses.replace(
        DEFAULT_CONFIG,
        turn_menus=((Fraction(1),), (Fraction(1),), (Fraction(1),)),
        turn_allin=True,
    )


class TurnDataGeneratorV2:
    def __init__(self, cfg: TurnDatagenV2Config = TurnDatagenV2Config()):
        self.cfg = cfg
        if cfg.mode == DatagenMode.AUTHOR_STRICT:
            strict_readiness_check()
        self.solver_cfg = _fcp_a_config()

    def make_batch_inputs(self, rng: CountingTHRandom) -> BatchInputs:
        board, _rejects = sample_turn_board(rng)
        rg = _Float32FloorRangeGenerator(board)
        ranges = np.empty((2, self.cfg.batch_size, HAND_COUNT), dtype=np.float32)
        # Released data_generation.lua generates the full P1 batch followed by
        # the full P2 batch, preserving this RNG call order.
        ranges[0] = rg.generate(self.cfg.batch_size, rng)
        ranges[1] = rg.generate(self.cfg.batch_size, rng)
        pots = np.empty(self.cfg.batch_size, dtype=np.int32)
        for i in range(self.cfg.batch_size):
            pots[i] = sample_pot_half(rng, self.cfg.mode)
        mask = possible_hands_mask(board).astype(np.uint8)
        masks = np.broadcast_to(mask, (self.cfg.batch_size, HAND_COUNT)).copy()
        return BatchInputs(
            board=board,
            ranges=ranges,
            pots=pots,
            masks=masks,
            rng_draws_after=rng.draws,
            boundary_ties=rg.boundary_ties,
        )

    def solve_batch(self, inputs: BatchInputs) -> SolvedBatch:
        b = self.cfg.batch_size
        targets = np.empty((b, 2, HAND_COUNT), dtype=np.float32)
        residuals = np.empty(b, dtype=np.float64)
        for i in range(b):
            pot = int(inputs.pots[i])
            engine = TurnEngine(
                inputs.board,
                pot,
                self.solver_cfg,
                cfr_iters=self.cfg.cfr_iters,
                cfr_skip_iters=self.cfg.cfr_skip_iters,
                terminal_backend=self.cfg.terminal_backend,
            )
            r1 = inputs.ranges[0, i].astype(np.float64)
            r2 = inputs.ranges[1, i].astype(np.float64)
            cfvs = engine.resolve_first_node(r1, r2)
            # Same-author released DataGeneration path divides the returned
            # root CFVs by the equal committed amount used as node.bets.
            targets[i] = (cfvs / float(pot)).astype(np.float32)
            u1 = float(np.dot(r1, cfvs[0]))
            u2 = float(np.dot(r2, cfvs[1]))
            residuals[i] = u1 + u2
        return SolvedBatch(inputs, targets, residuals)


__all__ = [
    "DatagenMode", "TurnDatagenV2Config", "CountingTHRandom",
    "BatchInputs", "SolvedBatch", "TurnDataGeneratorV2",
    "strict_readiness_check", "shard_seed", "sample_turn_board",
    "sample_pot_half",
]
