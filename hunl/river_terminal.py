"""HUNL river terminal-equity object for the certified Golden Baseline
lookahead (Gate G1.8).

Provides exactly the attribute interface the golden vectorized lookahead
consumes on a street-2 (terminal-street) tree
(`deepstack_leduc/lookahead.py::_compute_terminal_equities_terminal_equity`):
  .call_matrix  (K, K) float32
  .fold_matrix  (K, K) float32
with K = 1326 and the LEDUC ORIENTATION CONVENTION:
  - Leduc call matrix: entry [i, j] = +1 iff row hand LOSES to column hand
    (terminal.py builds sign(strength_i - strength_j) with lower strength =
    stronger). The lookahead computes r_p @ call_matrix, i.e. column hand
    values against the row player's range — certified end-to-end vs Lua.
  - HUNL equivalent: [i, j] = +1 iff rank_i < rank_j, which is exactly the
    TRANSPOSE of the G1.3 showdown matrix (M[i,j] = +1 iff i beats j),
    with every illegal pair (board-blocked or card-sharing) already 0.
  - Leduc fold matrix: ones minus diagonal, masked by board possibility =
    "opponent may hold j while I hold i" indicator. HUNL equivalent: the
    G1.2 legal-pairs mask.

Both matrices are derived from the G1.1/G1.2/G1.3-certified layers; no new
card math is introduced here.
"""
from __future__ import annotations

import numpy as np

from .blockers import legal_pairs_mask
from .showdown import showdown_matrix


class HunlRiverTerminalEquity:
    """Drop-in terminal-equity object for the golden lookahead (river)."""

    def __init__(self, board5) -> None:
        self.board = tuple(int(c) for c in board5)
        m, legal, ranks = showdown_matrix(self.board)
        # Leduc orientation: +1 where the ROW hand loses to the COLUMN hand.
        self.call_matrix = np.ascontiguousarray(m.T, dtype=np.float32)
        self.fold_matrix = np.ascontiguousarray(
            legal_pairs_mask(self.board), dtype=np.float32)
        self.showdown_ranks = ranks
