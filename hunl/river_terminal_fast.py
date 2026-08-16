"""Fast exact-semantics river terminal kernels for full 1326-hand HUNL.

The frozen/G1 river layer materializes dense 1326x1326 showdown and fold
matrices.  That is excellent as an oracle but extremely expensive inside the
turn DataGenerator: hundreds of terminal matrix-vector products are repeated
on every CFR iteration.

This module computes the SAME mathematical linear operators without dense
pair matrices:

Fold operator
-------------
For a legal hero hand h=(c0,c1), legal opponent mass is

  total_board_legal_mass - mass(hands containing c0)
                         - mass(hands containing c1)
                         + mass(h)

because the only opponent hand containing both c0 and c1 is h itself.

Showdown operator
-----------------
Ignoring private-card overlap, value against a range is total mass on lower
hand ranks minus total mass on higher hand ranks.  We compute this from
rank-group prefix sums.  Then subtract the same rank-comparison contribution
from hands containing c0 and c1.  Their intersection is h itself, whose
showdown sign against itself is zero, so no intersection correction is needed.

Board-blocked hero hands are exactly zero.  This is algebraically equivalent
to the certified dense ``showdown_matrix`` / ``legal_pairs_mask`` operators.
Floating-point addition order differs, so the accelerated path is certified
NUMERIC/SEMANTIC, not bit-identical to BLAS matrix multiplication.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cards import CARD_COUNT, HAND_CARDS, HAND_COUNT, possible_hands_mask
from .evaluator import rank_board_hands


@dataclass(frozen=True)
class _RankPlan:
    ids: np.ndarray              # global hand ids, sorted by rank
    group_starts: np.ndarray     # starts into ids
    group_ids: np.ndarray        # per sorted position -> group index


@dataclass(frozen=True)
class RiverTerminalFastKernel:
    board: tuple[int, int, int, int, int]
    legal_ids: np.ndarray
    legal_mask: np.ndarray
    global_plan: _RankPlan
    card_plans: tuple[_RankPlan | None, ...]
    hands_by_card: tuple[np.ndarray, ...]

    @staticmethod
    def _plan(ids: np.ndarray, ranks: np.ndarray) -> _RankPlan:
        ids = np.asarray(ids, dtype=np.int32)
        if ids.size == 0:
            return _RankPlan(ids, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))
        order = np.lexsort((ids, ranks[ids]))
        sids = ids[order].astype(np.int32, copy=False)
        sr = ranks[sids]
        starts = np.flatnonzero(np.r_[True, sr[1:] != sr[:-1]]).astype(np.int32)
        gids = np.empty(len(sids), dtype=np.int32)
        for g, st in enumerate(starts.tolist()):
            en = int(starts[g + 1]) if g + 1 < len(starts) else len(sids)
            gids[st:en] = g
        return _RankPlan(sids, starts, gids)

    @classmethod
    def build(cls, board5) -> "RiverTerminalFastKernel":
        b = tuple(int(c) for c in board5)
        if len(b) != 5 or len(set(b)) != 5:
            raise ValueError("river board must have 5 distinct cards")
        possible = possible_hands_mask(b)
        ids = np.flatnonzero(possible).astype(np.int32)
        if ids.size != 1081:  # C(47,2)
            raise AssertionError("river board must leave 1081 legal private hands")
        ranks = rank_board_hands(b)
        gp = cls._plan(ids, ranks)
        plans: list[_RankPlan | None] = []
        hbc: list[np.ndarray] = []
        for c in range(CARD_COUNT):
            if c in b:
                plans.append(None)
                hbc.append(np.zeros(0, dtype=np.int32))
                continue
            cid = ids[(HAND_CARDS[ids, 0] == c) | (HAND_CARDS[ids, 1] == c)]
            if cid.size != 46:
                raise AssertionError(f"unblocked card {c} must occur in 46 river-legal hands")
            cid = cid.astype(np.int32, copy=False)
            plans.append(cls._plan(cid, ranks))
            hbc.append(cid)
        return cls(b, ids, possible, gp, tuple(plans), tuple(hbc))

    @staticmethod
    def _rank_compare(range_batch: np.ndarray, plan: _RankPlan) -> np.ndarray:
        """Values at plan.ids: mass(lower rank) - mass(higher rank)."""
        if plan.ids.size == 0:
            return np.zeros((range_batch.shape[0], 0), dtype=np.float64)
        w = range_batch[:, plan.ids]
        group_sums = np.add.reduceat(w, plan.group_starts, axis=1)
        prefix = np.cumsum(group_sums, axis=1)
        lower = prefix - group_sums
        total = prefix[:, -1:]
        group_values = 2.0 * lower + group_sums - total
        return group_values[:, plan.group_ids]

    def showdown(self, ranges: np.ndarray) -> np.ndarray:
        """Return dense-operator equivalent ``ranges @ M.T``.

        ranges may be shape (1326,) or (B,1326).  Output has matching leading
        batch shape and 1326 hero-hand columns.
        """
        r = np.asarray(ranges, dtype=np.float64)
        one = r.ndim == 1
        if one:
            r = r[None, :]
        if r.ndim != 2 or r.shape[1] != HAND_COUNT:
            raise ValueError("ranges must have shape (1326,) or (B,1326)")
        out = np.zeros((r.shape[0], HAND_COUNT), dtype=np.float64)
        global_vals = self._rank_compare(r, self.global_plan)
        out[:, self.global_plan.ids] = global_vals
        # Remove all opponent hands sharing either private card with hero.
        for c, plan in enumerate(self.card_plans):
            if plan is None:
                continue
            vals = self._rank_compare(r, plan)
            out[:, plan.ids] -= vals
        out[:, ~self.legal_mask] = 0.0
        return out[0] if one else out

    def fold(self, ranges: np.ndarray) -> np.ndarray:
        """Return dense-operator equivalent ``ranges @ legal_pairs_mask``."""
        r = np.asarray(ranges, dtype=np.float64)
        one = r.ndim == 1
        if one:
            r = r[None, :]
        if r.ndim != 2 or r.shape[1] != HAND_COUNT:
            raise ValueError("ranges must have shape (1326,) or (B,1326)")
        rr = r.copy()
        rr[:, ~self.legal_mask] = 0.0
        total = rr.sum(axis=1, keepdims=True)
        card_mass = np.zeros((rr.shape[0], CARD_COUNT), dtype=np.float64)
        # Only 47 unblocked cards; each has 46 legal hands.
        for c, ids in enumerate(self.hands_by_card):
            if ids.size:
                card_mass[:, c] = rr[:, ids].sum(axis=1)
        h0 = HAND_CARDS[:, 0].astype(np.intp)
        h1 = HAND_CARDS[:, 1].astype(np.intp)
        out = total - card_mass[:, h0] - card_mass[:, h1] + rr
        out[:, ~self.legal_mask] = 0.0
        return out[0] if one else out


__all__ = ["RiverTerminalFastKernel"]

# Optional native JIT backend.  Kept separate from the pure-NumPy oracle above
# so environments without numba still import the project.
try:
    from numba import njit

    @njit(cache=True)
    def _numba_showdown_batch(ranges, ranks, global_ids, card_offsets,
                              card_ids, legal_mask):
        B = ranges.shape[0]
        out = np.zeros((B, HAND_COUNT), dtype=np.float64)
        for b in range(B):
            # Global board-legal rank comparison.
            total = 0.0
            for q in range(global_ids.shape[0]):
                total += ranges[b, global_ids[q]]
            lower = 0.0
            p = 0
            while p < global_ids.shape[0]:
                rr = ranks[global_ids[p]]
                q = p
                gs = 0.0
                while q < global_ids.shape[0] and ranks[global_ids[q]] == rr:
                    gs += ranges[b, global_ids[q]]
                    q += 1
                v = lower - (total - lower - gs)
                for z in range(p, q):
                    out[b, global_ids[z]] = v
                lower += gs
                p = q

            # Remove rank-comparison contribution from every private-card
            # overlap set.  Each legal hero occurs in exactly two segments.
            for c in range(CARD_COUNT):
                lo = card_offsets[c]
                hi = card_offsets[c + 1]
                if lo == hi:
                    continue
                totalc = 0.0
                for z in range(lo, hi):
                    totalc += ranges[b, card_ids[z]]
                lowerc = 0.0
                p = lo
                while p < hi:
                    rr = ranks[card_ids[p]]
                    q = p
                    gs = 0.0
                    while q < hi and ranks[card_ids[q]] == rr:
                        gs += ranges[b, card_ids[q]]
                        q += 1
                    v = lowerc - (totalc - lowerc - gs)
                    for z in range(p, q):
                        out[b, card_ids[z]] -= v
                    lowerc += gs
                    p = q

            for h in range(HAND_COUNT):
                if not legal_mask[h]:
                    out[b, h] = 0.0
        return out

    @njit(cache=True)
    def _numba_showdown_batch_into(ranges, ranks, global_ids, card_offsets,
                                   card_ids, legal_mask, out):
        B = ranges.shape[0]
        out[:] = 0.0
        for b in range(B):
            total = 0.0
            for q in range(global_ids.shape[0]):
                total += ranges[b, global_ids[q]]
            lower = 0.0
            p = 0
            while p < global_ids.shape[0]:
                rr = ranks[global_ids[p]]
                q = p
                gs = 0.0
                while q < global_ids.shape[0] and ranks[global_ids[q]] == rr:
                    gs += ranges[b, global_ids[q]]
                    q += 1
                v = lower - (total - lower - gs)
                for z in range(p, q):
                    out[b, global_ids[z]] = v
                lower += gs
                p = q
            for c in range(CARD_COUNT):
                lo = card_offsets[c]
                hi = card_offsets[c + 1]
                if lo == hi:
                    continue
                totalc = 0.0
                for z in range(lo, hi):
                    totalc += ranges[b, card_ids[z]]
                lowerc = 0.0
                p = lo
                while p < hi:
                    rr = ranks[card_ids[p]]
                    q = p
                    gs = 0.0
                    while q < hi and ranks[card_ids[q]] == rr:
                        gs += ranges[b, card_ids[q]]
                        q += 1
                    v = lowerc - (totalc - lowerc - gs)
                    for z in range(p, q):
                        out[b, card_ids[z]] -= v
                    lowerc += gs
                    p = q
            for h in range(HAND_COUNT):
                if not legal_mask[h]:
                    out[b, h] = 0.0

    @njit(cache=True)
    def _numba_fold_batch_into(ranges, card_offsets, card_ids, legal_mask,
                               hand_cards, out):
        B = ranges.shape[0]
        out[:] = 0.0
        for b in range(B):
            total = 0.0
            for h in range(HAND_COUNT):
                if legal_mask[h]:
                    total += ranges[b, h]
            cm = np.zeros(CARD_COUNT, dtype=np.float64)
            for c in range(CARD_COUNT):
                lo = card_offsets[c]
                hi = card_offsets[c + 1]
                ss = 0.0
                for z in range(lo, hi):
                    ss += ranges[b, card_ids[z]]
                cm[c] = ss
            for h in range(HAND_COUNT):
                if legal_mask[h]:
                    c0 = hand_cards[h, 0]
                    c1 = hand_cards[h, 1]
                    out[b, h] = total - cm[c0] - cm[c1] + ranges[b, h]

    @njit(cache=True)
    def _numba_fold_batch(ranges, card_offsets, card_ids, legal_mask,
                          hand_cards):
        B = ranges.shape[0]
        out = np.zeros((B, HAND_COUNT), dtype=np.float64)
        for b in range(B):
            total = 0.0
            for h in range(HAND_COUNT):
                if legal_mask[h]:
                    total += ranges[b, h]
            cm = np.zeros(CARD_COUNT, dtype=np.float64)
            for c in range(CARD_COUNT):
                lo = card_offsets[c]
                hi = card_offsets[c + 1]
                s = 0.0
                for z in range(lo, hi):
                    s += ranges[b, card_ids[z]]
                cm[c] = s
            for h in range(HAND_COUNT):
                if legal_mask[h]:
                    c0 = hand_cards[h, 0]
                    c1 = hand_cards[h, 1]
                    out[b, h] = total - cm[c0] - cm[c1] + ranges[b, h]
        return out

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional accelerator
    NUMBA_AVAILABLE = False


def _native_arrays(kernel: RiverTerminalFastKernel):
    """Build compact arrays consumed by the native/JIT backend."""
    ranks = rank_board_hands(kernel.board).astype(np.int64)
    gids = kernel.global_plan.ids.astype(np.int32, copy=False)
    offsets = np.zeros(CARD_COUNT + 1, dtype=np.int32)
    parts = []
    n = 0
    for c, plan in enumerate(kernel.card_plans):
        offsets[c] = n
        if plan is not None:
            parts.append(plan.ids.astype(np.int32, copy=False))
            n += len(plan.ids)
    offsets[CARD_COUNT] = n
    ids = np.concatenate(parts).astype(np.int32, copy=False) if parts else np.zeros(0, dtype=np.int32)
    return ranks, gids, offsets, ids


def showdown_numba(kernel: RiverTerminalFastKernel, ranges: np.ndarray) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba backend is unavailable")
    r = np.asarray(ranges, dtype=np.float64)
    one = r.ndim == 1
    if one:
        r = r[None, :]
    ranks, gids, offsets, ids = _native_arrays(kernel)
    out = _numba_showdown_batch(r, ranks, gids, offsets, ids,
                                kernel.legal_mask.astype(np.bool_))
    return out[0] if one else out


def fold_numba(kernel: RiverTerminalFastKernel, ranges: np.ndarray) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba backend is unavailable")
    r = np.asarray(ranges, dtype=np.float64)
    one = r.ndim == 1
    if one:
        r = r[None, :]
    _, _, offsets, ids = _native_arrays(kernel)
    out = _numba_fold_batch(r, offsets, ids,
                            kernel.legal_mask.astype(np.bool_),
                            HAND_CARDS.astype(np.int16))
    return out[0] if one else out

__all__ += ["NUMBA_AVAILABLE", "showdown_numba", "fold_numba"]
