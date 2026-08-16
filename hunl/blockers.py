"""HUNL blocker / hand-compatibility layer (Gate G1.2).

Uses the FROZEN 1326 hand ordering contract from hunl.cards
(lexicographic (c0, c1), c0 < c1 over ACPC card ids 0..51).

Layout contracts (FROZEN):
  - blocker matrix  B: uint8 (1326, 1326), row-major; B[i, j] = 1 iff hands
    i and j share at least one card. Diagonal is all 1 (self-overlap).
  - compatible matrix = 1 - B (uint8): hands can coexist.
  - per-board possible mask: bool (1326,), True iff neither hole card is on
    the board (any board size 0/3/4/5).
  - legal pair mask for a board: possible_i AND possible_j AND compatible.

No approximations anywhere; every structure is exact combinatorics.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .cards import CARD_COUNT, HAND_CARDS, HAND_COUNT, possible_hands_mask

_BLOCKER: np.ndarray | None = None
_CARD_HAND: np.ndarray | None = None


def blocker_matrix() -> np.ndarray:
    """uint8 (1326, 1326): 1 iff the two hands share a card (diag = 1)."""
    global _BLOCKER
    if _BLOCKER is None:
        a0 = HAND_CARDS[:, 0][:, None]
        a1 = HAND_CARDS[:, 1][:, None]
        b0 = HAND_CARDS[:, 0][None, :]
        b1 = HAND_CARDS[:, 1][None, :]
        share = (a0 == b0) | (a0 == b1) | (a1 == b0) | (a1 == b1)
        _BLOCKER = share.astype(np.uint8)
        _BLOCKER.setflags(write=False)
    return _BLOCKER


def compatible_matrix() -> np.ndarray:
    """uint8 (1326, 1326): 1 iff the two hands are disjoint."""
    out = (1 - blocker_matrix()).astype(np.uint8)
    return out


def card_hand_membership() -> np.ndarray:
    """bool (52, 1326): [c, h] True iff card c is one of hand h's cards."""
    global _CARD_HAND
    if _CARD_HAND is None:
        m = np.zeros((CARD_COUNT, HAND_COUNT), dtype=bool)
        m[HAND_CARDS[:, 0], np.arange(HAND_COUNT)] = True
        m[HAND_CARDS[:, 1], np.arange(HAND_COUNT)] = True
        m.setflags(write=False)
        _CARD_HAND = m
    return _CARD_HAND


def legal_pairs_mask(board) -> np.ndarray:
    """bool (1326, 1326): both hands possible on the board AND disjoint."""
    p = possible_hands_mask(board)
    return (p[:, None] & p[None, :]) & (blocker_matrix() == 0)


def blocker_sha256() -> str:
    """Determinism anchor over the frozen row-major uint8 matrix bytes."""
    return hashlib.sha256(blocker_matrix().tobytes()).hexdigest()
