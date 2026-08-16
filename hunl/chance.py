"""HUNL turn -> river chance layer (Gate G1 transition milestone).

CHANCE-WEIGHT ALGEBRA (HIGH-RISK part — derivation documented per
operation; the general construction is the game-independent author
treatment: Leduc chance strategy = possible_mask(child board) / 4 with
4 = card_count - both players' private cards, tree.py:_fill_uniform and
NextRoundValue's 1/(board_count-2) — generalized to HUNL numbers):

Definitions on a fixed 4-card turn board B4:
  - deck remainder D(B4) = 52 - 4 = 48 cards; river card c ∈ D(B4).
  - legal private hands on B4: 1,128 = C(48,2); on B4+c: 1,081 = C(47,2).

Distinguished probability objects (NEVER silently interchanged):
  1. PUBLIC-CARD PRIOR  P(c | B4) = 1/48
       numerator 1, denominator |D(B4)| = 48; conditioning: public board
       only, no private information. Used ONLY for public-observer
       statistics; NEVER in CFV aggregation.
  2. HAND-CONDITIONAL CHANCE  P(c | B4, h1, h2) = 1/44
       numerator 1, denominator 52 - 4(board) - 2(h1) - 2(h2) = 44;
       conditioning: both private hands fixed, c ∉ B4 ∪ h1 ∪ h2.
       This is THE chance factor of the game tree.
  3. RANGE-CONDITIONAL (counterfactual-reach) PROPAGATION
       child reach of player p on river c:
           reach_p^c(h) = reach_p(h) * mask_{B4+c}(h)        [f32/f64]
       i.e. reach vectors are MASKED, NOT renormalized (counterfactual
       reach convention of the certified engine). The factor 1/44 is
       applied EXACTLY ONCE per CFV aggregation (not once per player):
           u_p^{turn}(h) = (1/44) * sum_c mask_{B4+c}(h) * u_p^{river,c}(h)
       where u_p^{river,c} is computed from the OPPONENT's masked reach.
       Blocker correction: c ∈ h2 cases contribute zero via the opponent
       range mask on river c — the denominator stays 44 (cards unseen by
       the PAIR), and the identity below guarantees exact mass.
  4. NORMALIZED CONDITIONAL RANGE (helper for consumers that need a
     probability distribution): range_p^c = reach_p^c / sum(reach_p^c),
     defined only when the mass is positive. Not used in CFV math.

EXACT MASS IDENTITY (certified exhaustively in the harness): for every
legal disjoint pair (h1, h2) on B4:
    |{c ∈ D(B4) : c ∉ h1 ∪ h2}| = 44   =>   sum_c P(c|B4,h1,h2) = 1.

dtypes: masks bool/uint8 exact; chance factor and aggregation float64
(exact for 1/44-scaled sums up to f64 rounding); river-solver inputs are
cast to float32 by the certified engine (its frozen contract).
"""
from __future__ import annotations

import numpy as np

from .cards import CARD_COUNT, HAND_COUNT, possible_hands_mask
from .blockers import card_hand_membership

RIVER_DECK = 48            # cards not on a 4-card board
PAIR_UNSEEN = 44           # 52 - 4 board - 2 - 2 private
TURN_LEGAL_HANDS = 1128    # C(48,2)
RIVER_LEGAL_HANDS = 1081   # C(47,2)
CHANCE_FACTOR = 1.0 / PAIR_UNSEEN          # float64 exactly representable?
# 1/44 is not a dyadic rational; stored as the correctly rounded f64.


def river_cards(board4) -> list[int]:
    """All 48 possible river cards, ascending (deterministic order)."""
    board = {int(c) for c in board4}
    assert len(board) == 4, "turn board must have 4 distinct cards"
    return [c for c in range(CARD_COUNT) if c not in board]


def river_masks(board4) -> tuple[list[int], np.ndarray]:
    """(rivers, masks[48, 1326] bool): hand possibility on each B4+c."""
    rivers = river_cards(board4)
    board = tuple(int(c) for c in board4)
    masks = np.zeros((len(rivers), HAND_COUNT), dtype=bool)
    for k, c in enumerate(rivers):
        masks[k] = possible_hands_mask(board + (c,))
    return rivers, masks


def legal_river_counts_per_hand(board4) -> np.ndarray:
    """(1326,) int: for each hand legal on B4, the number of rivers on
    which it remains legal (must be 46 = 48 - 2); 0 for blocked hands."""
    rivers, masks = river_masks(board4)
    counts = masks.sum(axis=0)
    counts[~possible_hands_mask(board4)] = 0
    return counts.astype(np.int64)


def pair_legal_river_count(board4, h1: int, h2: int) -> int:
    """|{c : c ∉ B4 ∪ h1 ∪ h2}| — exact discrete recount (bitset-free
    reference form; the harness also certifies a bitset variant)."""
    CH = card_hand_membership()
    rivers = river_cards(board4)
    return sum(1 for c in rivers if not (CH[c, h1] or CH[c, h2]))
