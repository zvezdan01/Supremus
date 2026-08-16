"""HUNL exact river showdown layer (Gate G1.3), built strictly on the
G1.1-certified evaluator and the G1.2 blocker layer.

Contracts (FROZEN):
  - showdown matrix M: int8 (1326, 1326), row-major, over the frozen hand
    ordering. For a LEGAL pair (both hands possible on the 5-card board and
    mutually disjoint):
        M[i, j] = +1  if hand i beats hand j at showdown
        M[i, j] =  0  if tie
        M[i, j] = -1  if hand i loses to hand j
    Every ILLEGAL entry (either hand blocked by the board, hands sharing a
    card, or i == j) is EXACTLY 0; legality is carried separately by the
    uint8 legal mask — consumers must multiply/select by it, never infer
    legality from M values (a legal tie is also 0).
  - antisymmetry holds exactly over the whole matrix: M == -M.T
    (illegal entries are 0 on both sides).
  - ranks: uint32 vector from evaluator.rank_board_hands (blocked =
    0xFFFFFFFF sentinel).

Terminal utility reference (river showdown-only; used by the G1.3 CFV
identity test and later by exact terminal utilities):
    u1 = (M @ r2),  u2 = (-M.T @ r1) = (M @ r1) by antisymmetry-of-roles;
    zero-sum closure: r1 . u1 + r2 . u2 == 0 (exact in exact arithmetic).
"""
from __future__ import annotations

import hashlib

import numpy as np

from .blockers import legal_pairs_mask
from .cards import HAND_COUNT
from .evaluator import BLOCKED_SENTINEL, rank_board_hands


def showdown_matrix(board5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-card board -> (M int8 (1326,1326), legal uint8 (1326,1326),
    ranks uint32 (1326,))."""
    ranks = rank_board_hands(board5)
    legal = legal_pairs_mask(board5)
    r = ranks.astype(np.int64)
    gt = (r[:, None] > r[None, :])
    lt = (r[:, None] < r[None, :])
    m = gt.astype(np.int8) - lt.astype(np.int8)
    m[~legal] = 0
    return m, legal.astype(np.uint8), ranks


def showdown_sha256(board5) -> str:
    m, legal, ranks = showdown_matrix(board5)
    h = hashlib.sha256()
    h.update(m.tobytes())
    h.update(legal.tobytes())
    h.update(ranks.tobytes())
    return h.hexdigest()


def river_call_values(m: np.ndarray, ranges: np.ndarray,
                      dtype=np.float64) -> np.ndarray:
    """Showdown-only terminal values for both players.

    ranges: (2, 1326) reach masses (already zero on illegal hands).
    Returns (2, 1326) in `dtype`:
        out[0] = M       @ ranges[1]   (P1 values vs P2 range)
        out[1] = (-M.T)  @ ranges[0]   (P2 values vs P1 range)
    """
    r = np.asarray(ranges, dtype=dtype)
    md = m.astype(dtype)
    out = np.empty_like(r)
    out[0] = md @ r[1]
    out[1] = (-md.T) @ r[0]
    return out


__all__ = ["showdown_matrix", "showdown_sha256", "river_call_values",
           "BLOCKED_SENTINEL", "HAND_COUNT"]
