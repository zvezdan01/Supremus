"""Independent sequence-form LP oracle for restricted-support Supremus river trees.

No CFR/DCFR code is used here.  The only shared game object is the already
ACPC-certified sparse river tree, parameterized by the explicit Supremus
Table-2 action configuration.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from hunl.blockers import blocker_matrix
from hunl.cards import possible_hands_mask
from hunl.evaluator import rank_board_hands, BLOCKED_SENTINEL
from hunl.tree import RiverTreeBuilder
from hunl.supremus_config import SupremusRiverConfig


def solve_lp(board, pot_half, support0, support1, r0, r1,
             cfg: SupremusRiverConfig):
    tree = RiverTreeBuilder(cfg).build(int(pot_half))
    ranks = rank_board_hands(board)
    pm = possible_hands_mask(board)
    B = blocker_matrix()
    s0 = [int(h) for h in support0 if pm[int(h)]]
    s1 = [int(h) for h in support1 if pm[int(h)]]
    assert s0 and s1

    seqs = [{}, {}]
    n_seq = [1, 1]
    infosets = [[], []]

    def index_sequences(node, parent_seq, hand, player):
        if node.terminal is not None:
            return
        p = node.player
        if p == player:
            infosets[player].append((hand, node, parent_seq))
            for ai in range(len(node.children)):
                key = (hand, id(node), ai)
                seqs[player][key] = n_seq[player]
                n_seq[player] += 1
                index_sequences(node.children[ai], seqs[player][key], hand, player)
        else:
            for child in node.children:
                index_sequences(child, parent_seq, hand, player)

    for h in s0:
        index_sequences(tree, 0, h, 0)
    for h in s1:
        index_sequences(tree, 0, h, 1)

    A = lil_matrix((n_seq[0], n_seq[1]))

    def walk(node, seq0, seq1):
        if node.terminal is not None:
            b = float(min(node.spent))
            for i, si in seq0.items():
                for j, sj in seq1.items():
                    if B[i, j]:
                        continue
                    w = float(r0[i]) * float(r1[j])
                    if w == 0.0:
                        continue
                    if node.terminal == "fold":
                        u0 = -b if node.folder == 0 else b
                    else:
                        ri, rj = int(ranks[i]), int(ranks[j])
                        assert ri != int(BLOCKED_SENTINEL) and rj != int(BLOCKED_SENTINEL)
                        u0 = b * ((ri > rj) - (ri < rj))
                    A[si, sj] += w * u0
            return
        p = node.player
        for ai, child in enumerate(node.children):
            if p == 0:
                nxt = {h: seqs[0].get((h, id(node), ai), s)
                       for h, s in seq0.items()}
                walk(child, nxt, seq1)
            else:
                nxt = {h: seqs[1].get((h, id(node), ai), s)
                       for h, s in seq1.items()}
                walk(child, seq0, nxt)

    walk(tree, {h: 0 for h in s0}, {h: 0 for h in s1})
    A = A.toarray()

    def flow(player):
        E = np.zeros((1 + len(infosets[player]), n_seq[player]))
        e = np.zeros(E.shape[0])
        E[0, 0] = 1.0
        e[0] = 1.0
        for k, (hand, node, parent) in enumerate(infosets[player]):
            E[1+k, parent] = -1.0
            for ai in range(len(node.children)):
                E[1+k, seqs[player][(hand, id(node), ai)]] = 1.0
        return E, e

    E, e = flow(0)
    F, f = flow(1)
    nx, npv = n_seq[0], F.shape[0]
    c = np.concatenate([np.zeros(nx), -f])
    A_ub = np.hstack([-A.T, F.T])
    b_ub = np.zeros(A.shape[1])
    A_eq = np.hstack([E, np.zeros((E.shape[0], npv))])
    bounds = [(0, None)] * nx + [(None, None)] * npv
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=e,
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(-res.fun)
