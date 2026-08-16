"""Source-constrained HUNL training-range primitives (V2).

Primary HUNL authority: DeepStack supplementary material, Algorithm prose for
R(S,p):
  * recursively split probability mass with p1 uniform in (0,p);
  * |S1| = floor(|S|/2);
  * every hand in S1 has hand strength <= every hand in S2;
  * hand strength is the probability of beating a uniformly selected random
    hand from the current public state.

This module implements the parts that are mathematically determined by that
text.  It does NOT claim the private DeepStack RNG or tie ordering.

Important correction versus the historical project pilot:
``all-in equity after future runout`` is not used as the sorting metric.  The
hand-strength definition is evaluated on the *current* public board, matching
the literal supplement definition and the released DeepStack-Leduc pattern of
sorting by the current-board evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hunl.blockers import blocker_matrix
from hunl.cards import HAND_CARDS, HAND_COUNT, possible_hands_mask
from hunl.evaluator import rank_suit_masks, suit_masks


class Uniform01Source(Protocol):
    """Injected random source; exact private HUNL RNG remains unresolved."""

    def uniform01(self, n: int) -> np.ndarray: ...


@dataclass(frozen=True)
class HandStrengthResult:
    board: tuple[int, ...]
    legal_hands: np.ndarray
    strengths: np.ndarray
    opponent_counts: np.ndarray


def current_public_hand_strength(board) -> HandStrengthResult:
    """Exact P(current hand strictly beats uniform legal opponent hand).

    No future public cards are dealt.  On flop/turn/river this ranks the
    5/6/7-card current holdings using the certified ACPC ``rankCardset``
    evaluator semantics.  On a river board each legal hand therefore compares
    against exactly C(45,2)=990 compatible opponent hands.  Ties are not wins,
    because the source says
    "probability ... beating" rather than showdown equity.

    Returned ``strengths`` has shape [1326], with NaN on board-blocked hands.
    """
    b = tuple(int(c) for c in board)
    if len(b) not in (3, 4, 5):
        raise ValueError("range-generation public board must be flop, turn, or river")
    if len(set(b)) != len(b):
        raise ValueError("public board cards must be distinct")

    possible = possible_hands_mask(b)
    ids = np.flatnonzero(possible)
    cards = np.empty((ids.size, len(b) + 2), dtype=np.int8)
    cards[:, :2] = HAND_CARDS[ids]
    cards[:, 2:] = np.asarray(b, dtype=np.int8)
    ranks = rank_suit_masks(suit_masks(cards)).astype(np.int32, copy=False)

    compat = blocker_matrix()[np.ix_(ids, ids)] == 0
    counts = compat.sum(axis=1).astype(np.int32)
    wins = ((ranks[:, None] > ranks[None, :]) & compat).sum(axis=1)

    # Fixed-card combinatorics provides a useful invariant.
    unseen_after_board_and_own = 52 - len(b) - 2
    expected_opp = unseen_after_board_and_own * (unseen_after_board_and_own - 1) // 2
    if not np.all(counts == expected_opp):
        raise AssertionError("uniform-opponent denominator is not constant")

    out = np.full(HAND_COUNT, np.nan, dtype=np.float64)
    out[ids] = wins.astype(np.float64) / counts.astype(np.float64)
    return HandStrengthResult(b, ids, out, counts)


def source_order_with_project_tiebreak(board) -> tuple[np.ndarray, np.ndarray, int]:
    """Weak→strong order satisfying the source constraint.

    Equal-strength hands are ordered by frozen hand id as an explicit
    PROJECT-CANONICAL tie-break.  The supplement does not state how equal
    strengths were ordered.  ``boundary_tie_count`` counts recursive split
    nodes at which an equal-strength class crosses the floor split boundary;
    these are exactly the places where the unpublished tie convention can
    change individual generated ranges.
    """
    hs = current_public_hand_strength(board)
    ids = hs.legal_hands
    vals = hs.strengths[ids]
    # np.lexsort uses the final key as primary: strength primary, hand id tie.
    order_pos = np.lexsort((ids, vals))
    ordered_ids = ids[order_pos]
    ordered_strengths = vals[order_pos]

    ties = 0
    stack = [(0, len(ordered_ids))]
    while stack:
        lo, hi = stack.pop()
        n = hi - lo
        if n <= 1:
            continue
        half = n // 2  # AUTHOR-EXPLICIT floor(|S|/2)
        mid = lo + half
        if ordered_strengths[mid - 1] == ordered_strengths[mid]:
            ties += 1
        stack.append((lo, mid))
        stack.append((mid, hi))
    return ordered_ids, ordered_strengths, ties


class AuthorRangeGeneratorV2:
    """R(S,1) with the HUNL paper's floor split and injected uniform draws.

    The source leaves RNG implementation and equal-strength ordering
    unpublished.  This class therefore takes an RNG explicitly and uses the
    documented project-canonical hand-id tie-break from
    ``source_order_with_project_tiebreak``.
    """

    def __init__(self, board) -> None:
        self.board = tuple(int(c) for c in board)
        self.order, self.ordered_strengths, self.boundary_tie_count = (
            source_order_with_project_tiebreak(self.board)
        )
        self.possible = possible_hands_mask(self.board)

    def generate(self, batch: int, rng: Uniform01Source) -> np.ndarray:
        if batch <= 0:
            raise ValueError("batch must be positive")
        sorted_ranges = np.empty((batch, len(self.order)), dtype=np.float64)

        def rec(lo: int, hi: int, mass: np.ndarray) -> None:
            n = hi - lo
            if n == 1:
                sorted_ranges[:, lo] = mass
                return
            u = np.asarray(rng.uniform01(batch), dtype=np.float64)
            if u.shape != (batch,):
                raise ValueError("uniform01 source returned wrong shape")
            if np.any(u <= 0.0) or np.any(u >= 1.0):
                raise ValueError("R(S,p) requires draws strictly inside (0,1)")
            m1 = mass * u
            m2 = mass - m1
            mid = lo + n // 2  # AUTHOR-EXPLICIT, no random odd split
            rec(lo, mid, m1)
            rec(mid, hi, m2)

        rec(0, len(self.order), np.ones(batch, dtype=np.float64))
        out = np.zeros((batch, HAND_COUNT), dtype=np.float64)
        out[:, self.order] = sorted_ranges
        return out


__all__ = [
    "Uniform01Source",
    "HandStrengthResult",
    "current_public_hand_strength",
    "source_order_with_project_tiebreak",
    "AuthorRangeGeneratorV2",
]
