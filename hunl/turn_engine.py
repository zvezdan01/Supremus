"""HUNL full-card exact-to-end turn engine for OFFLINE DATAGEN / ORACLES.

This engine traverses turn betting -> river chance -> river betting -> terminals
in the full 1326-hand card space, with no card abstraction and no NN.  That is
the required shape for the published DeepStack TURN TRAINING TARGET games
(F/C/P/A, no card abstraction).

IMPORTANT PROVENANCE BOUNDARY (Schmid follow-up evidence, 2020): the original
DeepStack PLAY-TIME turn resolver did *not* use this full-card river layer; it
solved to game end using an unpublished bucketed abstraction for all river
actions.  Therefore this class is a source-constrained datagen/math oracle, not
a claim of exact original online-turn implementation.

Solver semantics are the certified DeepStack scheme, transcribed
operation-for-operation from the certified sources (line references):
  - regret matching+: current strategy = clip(cum_regrets, eps, 999999) /
    sum         [golden lookahead.py:_compute_current_strategies:125-143]
  - cumulative regrets updated then clipped to [0, 999999]
                [golden lookahead.py:_compute_regrets:268-292]
  - simultaneous updates, both players per iteration   [supplement p.22]
  - uniform post-omit averaging of strategies and root CFVs
                [golden lookahead.py:169-171, 245-259]
  - CFR-D gadget: the golden `CFRDGadget` class imported UNCHANGED
    (dimension-generic; certified at 1326 dims in Gate G1.7)
  - chance: masked counterfactual reach down, single x(1/44) on value
    aggregation up — the certified transition algebra (gate 43020db)
  - terminals: certified G1.2/G1.3 matrices, orientation as certified in
    the G1-turn BR audit (validated against the frozen river engine to
    0.003% pot).

dtype: float64 throughout the engine (documented; the frozen f32 river
engine remains the river-resolve Golden path — this engine is the
full-game turn path and is certified NUMERIC against independent
references, never claimed author-BIT_EXACT).
"""
from __future__ import annotations

import numpy as np

from .blockers import legal_pairs_mask
from .cards import HAND_CARDS, HAND_COUNT, possible_hands_mask
from .chance import CHANCE_FACTOR
from .config import DEFAULT_CONFIG, HunlConfig
from .showdown import showdown_matrix
from .river_terminal_fast import (
    NUMBA_AVAILABLE, RiverTerminalFastKernel, _native_arrays,
    _numba_showdown_batch, _numba_fold_batch,
)
from .turn_tree import TurnGameTreeBuilder, TurnNode


EPS = 1e-9
CAP = 999999.0


class TurnEngine:
    """Exact full-game turn solver for one (board4, pot_half) state."""

    def __init__(self, board4, pot_half: int,
                 cfg: HunlConfig = DEFAULT_CONFIG,
                 cfr_iters: int | None = None,
                 cfr_skip_iters: int | None = None,
                 terminal_backend: str = "dense") -> None:
        self.cfg = cfg
        self.board = tuple(int(c) for c in board4)
        self.pot_half = int(pot_half)
        self.iters = cfr_iters if cfr_iters is not None else cfg.turn_cfr_iters
        self.skip = (cfr_skip_iters if cfr_skip_iters is not None
                     else cfg.turn_cfr_omit)
        if terminal_backend not in ("dense", "rank_numba"):
            raise ValueError("terminal_backend must be dense or rank_numba")
        if terminal_backend == "rank_numba" and not NUMBA_AVAILABLE:
            raise RuntimeError("rank_numba terminal backend requested but numba is unavailable")
        self.terminal_backend = terminal_backend
        self.tree = TurnGameTreeBuilder(cfg).build(self.board, self.pot_half)
        self.pm4 = possible_hands_mask(self.board)
        self._index()
        self._build_matrices()

    # ------------------------------------------------------- indexing --
    def _index(self) -> None:
        self.nodes: list[TurnNode] = []
        self.parent: list[int] = []
        self.decision: list[int] = []
        self.terminals_by_board: dict[tuple, dict[str, list[int]]] = {}

        def walk(n: TurnNode, parent_id: int) -> None:
            nid = len(self.nodes)
            self.nodes.append(n)
            self.parent.append(parent_id)
            n._id = nid  # type: ignore
            if n.terminal is not None:
                key = n.board
                g = self.terminals_by_board.setdefault(
                    key, {"fold": [], "showdown": []})
                g[n.terminal].append(nid)
                return
            if n.street != "chance":
                self.decision.append(nid)
                k = len(n.children)
                n._reg = np.zeros((k, HAND_COUNT))       # type: ignore
                n._avg = np.zeros((k, HAND_COUNT))       # type: ignore
                n._strat = np.full((k, HAND_COUNT), 1.0 / k)  # type: ignore
            for c in n.children:
                walk(c, nid)

        import sys
        sys.setrecursionlimit(200000)
        walk(self.tree, -1)
        self.reach = np.zeros((len(self.nodes), 2, HAND_COUNT))
        self.value = np.zeros((len(self.nodes), 2, HAND_COUNT))

    def _build_matrices(self) -> None:
        self.mats: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.fast_terminal: dict[tuple, tuple] = {}
        for key in self.terminals_by_board:
            if len(key) == 5 and self.terminal_backend == "rank_numba":
                # Keep the full certified dense matrices OUT of the hot path.
                # The rank/blocker kernel is algebraically the same operator
                # and is independently regression-tested against them.
                kernel = RiverTerminalFastKernel.build(key)
                ranks, gids, offsets, ids = _native_arrays(kernel)
                self.fast_terminal[key] = (
                    kernel, ranks, gids, offsets, ids,
                    kernel.legal_mask.astype(np.bool_),
                )
                self.mats[key] = (None, None)
            elif len(key) == 5:
                m, legal, _ = showdown_matrix(key)
                self.mats[key] = (m.astype(np.float64),
                                  legal.astype(np.float64))
            else:
                # Only the single turn board uses this path; keep the frozen
                # dense fold oracle here because it is not the performance
                # bottleneck and preserves the existing golden semantics.
                self.mats[key] = (None,
                                  legal_pairs_mask(key).astype(np.float64))

    # --------------------------------------------------------- solve --
    def _iterate(self, r1: np.ndarray, gadget, r2_fixed) -> None:
        nodes, reach, value = self.nodes, self.reach, self.value
        root_avg = self.root_avg
        for it in range(1, self.iters + 1):
            if gadget is not None:
                opp = gadget.compute_opponent_range(
                    self._last_root_cfvs[1].astype(np.float32), it)
                r2 = opp.astype(np.float64)
            else:
                r2 = r2_fixed
            reach[0, 0] = r1
            reach[0, 1] = r2
            # top-down reach
            for nid, n in enumerate(nodes):
                if n.terminal is not None:
                    continue
                if n.street == "chance":
                    for k, c in enumerate(n.children):
                        cid = c._id  # type: ignore
                        reach[cid] = reach[nid] * n.chance_masks[k]
                else:
                    p = n.player
                    st = n._strat  # type: ignore
                    for a, c in enumerate(n.children):
                        cid = c._id  # type: ignore
                        reach[cid] = reach[nid]
                        reach[cid, p] = reach[nid, p] * st[a]
            # terminals (batched per board)
            for key, groups in self.terminals_by_board.items():
                m, fmask = self.mats[key]
                fast = self.fast_terminal.get(key)
                sd = groups["showdown"]
                if sd:
                    R1 = reach[sd, 1]
                    R0 = reach[sd, 0]
                    if fast is not None:
                        _kernel, ranks, gids, offsets, ids, legal_mask = fast
                        U0 = _numba_showdown_batch(
                            R1, ranks, gids, offsets, ids, legal_mask)
                        # -(R0 @ M) == R0 @ M.T because M is antisymmetric.
                        U1 = _numba_showdown_batch(
                            R0, ranks, gids, offsets, ids, legal_mask)
                    else:
                        U0 = R1 @ m.T
                        U1 = -(R0 @ m)
                    b = np.array([float(min(nodes[i].spent))
                                  for i in sd])[:, None]
                    value[sd, 0] = b * U0
                    value[sd, 1] = b * U1
                fl = groups["fold"]
                if fl:
                    R1 = reach[fl, 1]
                    R0 = reach[fl, 0]
                    if fast is not None:
                        _kernel, _ranks, _gids, offsets, ids, legal_mask = fast
                        U0 = _numba_fold_batch(
                            R1, offsets, ids, legal_mask, HAND_CARDS.astype(np.int16))
                        U1 = _numba_fold_batch(
                            R0, offsets, ids, legal_mask, HAND_CARDS.astype(np.int16))
                    else:
                        U0 = R1 @ fmask
                        U1 = R0 @ fmask
                    for row, i in enumerate(fl):
                        n = nodes[i]
                        b = float(min(n.spent))
                        sgn = -1.0 if n.folder == 0 else 1.0
                        value[i, 0] = sgn * b * U0[row]
                        value[i, 1] = -sgn * b * U1[row]
            # bottom-up values + regrets
            for nid in range(len(nodes) - 1, -1, -1):
                n = nodes[nid]
                if n.terminal is not None:
                    continue
                if n.street == "chance":
                    acc = np.zeros((2, HAND_COUNT))
                    for k, c in enumerate(n.children):
                        acc += n.chance_masks[k] * value[c._id]  # type: ignore
                    value[nid] = CHANCE_FACTOR * acc
                else:
                    p = n.player
                    st = n._strat  # type: ignore
                    u = np.zeros((2, HAND_COUNT))
                    k = len(n.children)
                    avals = np.empty((k, HAND_COUNT))
                    for a, c in enumerate(n.children):
                        cv = value[c._id]  # type: ignore
                        avals[a] = cv[p]
                        u[p] += st[a] * cv[p]
                        u[1 - p] += cv[1 - p]
                    value[nid] = u
                    reg = n._reg  # type: ignore
                    reg += avals - u[p][None, :]
                    np.clip(reg, 0.0, CAP, out=reg)
            self._last_root_cfvs = value[0].copy()
            # next strategies + averaging
            for nid in self.decision:
                n = nodes[nid]
                pos = np.clip(n._reg, EPS, CAP)  # type: ignore
                n._strat = pos / pos.sum(axis=0, keepdims=True)  # type: ignore
                if it > self.skip:
                    n._avg += n._strat  # type: ignore
            if it > self.skip:
                root_avg += value[0]

    def _finish(self):
        denom = self.iters - self.skip
        self.root_cfvs = self.root_avg / denom
        for nid in self.decision:
            n = self.nodes[nid]
            s = n._avg.sum(axis=0, keepdims=True)  # type: ignore
            with np.errstate(divide="ignore", invalid="ignore"):
                n._navg = np.where(  # type: ignore
                    s > 0, n._avg / s, 1.0 / len(n.children))
        root = self.tree
        self.root_strategy = root._navg.copy()  # type: ignore
        return self.root_cfvs

    def resolve_first_node(self, r1: np.ndarray, r2: np.ndarray):
        r1 = np.asarray(r1, dtype=np.float64)
        r2 = np.asarray(r2, dtype=np.float64)
        assert (r1[~self.pm4] == 0).all() and (r2[~self.pm4] == 0).all()
        self.root_avg = np.zeros((2, HAND_COUNT))
        self._last_root_cfvs = np.zeros((2, HAND_COUNT))
        self._iterate(r1, None, r2)
        return self._finish()

    def resolve(self, r1: np.ndarray, opponent_cfvs: np.ndarray):
        """CFR-D re-solve: golden gadget, opponent-optimal constraints."""
        r1 = np.asarray(r1, dtype=np.float64)
        assert (r1[~self.pm4] == 0).all()
        # Lazy imports keep the first-node/offline datagen path independent
        # of the legacy Leduc BLAS runtime. This does not change resolve()
        # semantics; it only defers loading CFR-D dependencies until needed.
        from deepstack_leduc.cfrd_gadget import CFRDGadget
        from .river_resolver import hunl_river_config

        gadget = CFRDGadget(self.board,
                            r1.astype(np.float32),
                            np.asarray(opponent_cfvs, dtype=np.float32),
                            hunl_river_config(self.cfg))
        gadget.range_mask = self.pm4.astype(np.float32)
        self.gadget = gadget
        self.root_avg = np.zeros((2, HAND_COUNT))
        self._last_root_cfvs = np.zeros((2, HAND_COUNT))
        self._iterate(r1, gadget, None)
        return self._finish()

    # ---------------------------------------------------- exact BR ----
    def best_response_value(self, brp: int, reach_opp: np.ndarray
                            ) -> np.ndarray:
        """Exact BR value vector for player brp vs the average profile."""
        def br(n: TurnNode, ro: np.ndarray) -> np.ndarray:
            if n.terminal is not None:
                m, fmask = self.mats[n.board]
                b = float(min(n.spent))
                if n.terminal == "fold":
                    sgn = -1.0 if n.folder == 0 else 1.0
                    sgn = sgn if brp == 0 else -sgn
                    return sgn * b * (fmask @ ro)
                mat = m if brp == 0 else (-m.T)
                return b * (mat @ ro)
            if n.street == "chance":
                acc = np.zeros(HAND_COUNT)
                for k, c in enumerate(n.children):
                    acc += n.chance_masks[k] * br(c, ro * n.chance_masks[k])
                return CHANCE_FACTOR * acc
            if n.player == brp:
                return np.max(np.stack([br(c, ro) for c in n.children]),
                              axis=0)
            out = np.zeros(HAND_COUNT)
            for a, c in enumerate(n.children):
                out += br(c, ro * n._navg[a])  # type: ignore
            return out
        return br(self.tree, np.asarray(reach_opp, dtype=np.float64))
