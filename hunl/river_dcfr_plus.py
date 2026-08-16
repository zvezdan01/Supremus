"""Full-card HUNL river solver using the Supremus DCFR+ paper contract.

This is a SOURCE-CONSTRAINED reconstruction, not original Supremus source.

Primary algorithmic constraints:
  * DCFR is Brown & Sandholm DCFR(1.5, 0, 2) by default.
  * Supremus DCFR+ keeps DCFR regret discounting but replaces quadratic
    average-policy weighting by max(0, t-d), with d=100.
  * Supremus reports simultaneous updates as faster when CFVnets are used and
    solves generated subgames with 4,000 iterations per player.

River training games contain no future CFVnet leaf.  We nevertheless expose
``simultaneous=True`` as the paper-profile default and record it in manifests;
the private implementation is unavailable.

Unlike the older DeepStack tensor lookahead, this class evaluates the actual
reach-weighted average behavior strategy after solving.  Terminal operators are
exact full-card blocker/showdown kernels already certified elsewhere in this
project.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

from .cards import HAND_CARDS, HAND_COUNT, possible_hands_mask
from .river_terminal_fast import (
    NUMBA_AVAILABLE,
    RiverTerminalFastKernel,
    _native_arrays,
    _numba_fold_batch,
    _numba_showdown_batch,
    _numba_fold_batch_into,
    _numba_showdown_batch_into,
)
from .tree import RiverNode, RiverTreeBuilder
from .supremus_config import SupremusRiverConfig

_FLOAT64_TINY = np.finfo(np.float64).tiny



if njit is not None:
    @njit(cache=True)
    def _flat_dcfr_solve(iterations, alpha, beta, delay,
                         player, edge_start, children,
                         show_ix, fold_ix, show_b, fold_b, fold_sgn,
                         ranks, gids, card_offsets, card_ids, legal,
                         hand_cards, r0, r1):
        N = player.shape[0]
        E = children.shape[0]
        H = r0.shape[0]
        ns = show_ix.shape[0]
        nf = fold_ix.shape[0]
        reg = np.zeros((E,H), dtype=np.float64)
        strat = np.zeros((E,H), dtype=np.float64)
        avg_num = np.zeros((E,H), dtype=np.float64)
        avg_den = np.zeros((N,H), dtype=np.float64)
        reach = np.zeros((N,2,H), dtype=np.float64)
        self_reach = np.zeros((N,2,H), dtype=np.float64)
        value = np.zeros((N,2,H), dtype=np.float64)
        inst = np.zeros((E,H), dtype=np.float64)
        show_r0 = np.zeros((ns,H), dtype=np.float64)
        show_r1 = np.zeros((ns,H), dtype=np.float64)
        show_u0 = np.zeros((ns,H), dtype=np.float64)
        show_u1 = np.zeros((ns,H), dtype=np.float64)
        fold_r0 = np.zeros((nf,H), dtype=np.float64)
        fold_r1 = np.zeros((nf,H), dtype=np.float64)
        fold_u0 = np.zeros((nf,H), dtype=np.float64)
        fold_u1 = np.zeros((nf,H), dtype=np.float64)

        def make_strategy():
            for n in range(N):
                p = player[n]
                if p < 0:
                    continue
                lo = edge_start[n]
                hi = edge_start[n+1]
                k = hi-lo
                for h in range(H):
                    den = 0.0
                    for e in range(lo,hi):
                        x = reg[e,h]
                        if x > 0.0:
                            den += x
                    if den > 0.0:
                        for e in range(lo,hi):
                            x = reg[e,h]
                            strat[e,h] = x/den if x > 0.0 else 0.0
                    else:
                        q = 1.0/k
                        for e in range(lo,hi):
                            strat[e,h] = q

        def propagate():
            reach[:] = 0.0
            self_reach[:] = 0.0
            for h in range(H):
                reach[0,0,h] = r0[h]
                reach[0,1,h] = r1[h]
                z = 1.0 if legal[h] else 0.0
                self_reach[0,0,h] = z
                self_reach[0,1,h] = z
            for n in range(N):
                p = player[n]
                if p < 0:
                    continue
                lo=edge_start[n]; hi=edge_start[n+1]
                for e in range(lo,hi):
                    c=children[e]
                    for h in range(H):
                        reach[c,0,h] = reach[n,0,h]
                        reach[c,1,h] = reach[n,1,h]
                        self_reach[c,0,h] = self_reach[n,0,h]
                        self_reach[c,1,h] = self_reach[n,1,h]
                        reach[c,p,h] *= strat[e,h]
                        self_reach[c,p,h] *= strat[e,h]

        def terminals():
            for j in range(ns):
                n=show_ix[j]
                for h in range(H):
                    show_r0[j,h]=reach[n,0,h]
                    show_r1[j,h]=reach[n,1,h]
            _numba_showdown_batch_into(show_r1, ranks, gids, card_offsets, card_ids, legal, show_u0)
            _numba_showdown_batch_into(show_r0, ranks, gids, card_offsets, card_ids, legal, show_u1)
            for j in range(ns):
                n=show_ix[j]; bb=show_b[j]
                for h in range(H):
                    value[n,0,h]=bb*show_u0[j,h]
                    value[n,1,h]=bb*show_u1[j,h]
            for j in range(nf):
                n=fold_ix[j]
                for h in range(H):
                    fold_r0[j,h]=reach[n,0,h]
                    fold_r1[j,h]=reach[n,1,h]
            _numba_fold_batch_into(fold_r1, card_offsets, card_ids, legal, hand_cards, fold_u0)
            _numba_fold_batch_into(fold_r0, card_offsets, card_ids, legal, hand_cards, fold_u1)
            for j in range(nf):
                n=fold_ix[j]; bb=fold_b[j]; sg=fold_sgn[j]
                for h in range(H):
                    value[n,0,h]=sg*bb*fold_u0[j,h]
                    value[n,1,h]=-sg*bb*fold_u1[j,h]

        def bottom(make_inst):
            for n in range(N-1,-1,-1):
                p=player[n]
                if p < 0:
                    continue
                op=1-p
                lo=edge_start[n]; hi=edge_start[n+1]
                for h in range(H):
                    value[n,0,h]=0.0
                    value[n,1,h]=0.0
                for e in range(lo,hi):
                    c=children[e]
                    for h in range(H):
                        if make_inst:
                            inst[e,h]=value[c,p,h]
                        value[n,p,h] += strat[e,h]*value[c,p,h]
                        value[n,op,h] += value[c,op,h]
                if make_inst:
                    for e in range(lo,hi):
                        for h in range(H):
                            inst[e,h] -= value[n,p,h]

        for t in range(1,iterations+1):
            make_strategy()
            propagate()
            value[:] = 0.0
            terminals()
            bottom(True)
            w = t-delay
            if w > 0:
                wf=float(w)
                for n in range(N):
                    p=player[n]
                    if p < 0:
                        continue
                    lo=edge_start[n]; hi=edge_start[n+1]
                    for h in range(H):
                        c=wf*self_reach[n,p,h]
                        avg_den[n,h] += c
                        for e in range(lo,hi):
                            avg_num[e,h] += c*strat[e,h]
            pa = float(t)**alpha
            pf = pa/(pa+1.0)
            if beta == 0.0:
                nfct=0.5
            else:
                pb=float(t)**beta
                nfct=pb/(pb+1.0)
            for e in range(E):
                for h in range(H):
                    x=reg[e,h]+inst[e,h]
                    y = x*pf if x > 0.0 else (x*nfct if x < 0.0 else 0.0)
                    if y != 0.0 and abs(y) < 2.2250738585072014e-308:
                        y = 0.0
                    reg[e,h] = y

        # Final average strategy into strat; fallback to RM(reg) where never reached.
        for n in range(N):
            p=player[n]
            if p < 0:
                continue
            lo=edge_start[n]; hi=edge_start[n+1]; k=hi-lo
            for h in range(H):
                den=avg_den[n,h]
                if den > 0.0:
                    for e in range(lo,hi):
                        strat[e,h]=avg_num[e,h]/den
                else:
                    rr=0.0
                    for e in range(lo,hi):
                        if reg[e,h] > 0.0:
                            rr += reg[e,h]
                    if rr > 0.0:
                        for e in range(lo,hi):
                            x=reg[e,h]
                            strat[e,h]=x/rr if x > 0.0 else 0.0
                    else:
                        q=1.0/k
                        for e in range(lo,hi):
                            strat[e,h]=q
        propagate()
        value[:] = 0.0
        terminals()
        bottom(False)
        root_lo=edge_start[0]; root_hi=edge_start[1]
        return value[0].copy(), strat[root_lo:root_hi].copy()

@dataclass(frozen=True)
class DcfrPlusSpec:
    iterations: int = 4_000
    alpha: float = 1.5
    beta: float = 0.0
    delay: int = 100
    simultaneous: bool = True

    def average_weight(self, t: int) -> float:
        return float(max(0, t - self.delay))

    def positive_discount(self, t: int) -> float:
        z = float(t) ** self.alpha
        return z / (z + 1.0)

    def negative_discount(self, t: int) -> float:
        if math.isinf(self.beta) and self.beta < 0:
            return 0.0
        z = float(t) ** self.beta
        return z / (z + 1.0)


@dataclass(frozen=True)
class RiverSolveResult:
    root_cfvs: np.ndarray                 # [2,1326], chips
    root_strategy: np.ndarray             # [actions,1326]
    weighted_zero_sum_residual: float
    iterations: int
    decision_nodes: int
    terminal_nodes: int


class RiverDcfrPlusEngine:
    """Sparse full-card river DCFR+ solve for one public state."""

    def __init__(self, board5, pot_half: int,
                 game_cfg: SupremusRiverConfig = SupremusRiverConfig(),
                 dcfr: DcfrPlusSpec = DcfrPlusSpec()) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError("numba exact terminal backend is required")
        self.board = tuple(int(c) for c in board5)
        if len(self.board) != 5 or len(set(self.board)) != 5:
            raise ValueError("board must contain five distinct cards")
        self.pot_half = int(pot_half)
        self.game_cfg = game_cfg
        self.dcfr = dcfr
        self.pm = possible_hands_mask(self.board)
        self.tree = RiverTreeBuilder(game_cfg).build(self.pot_half)
        self._index_tree()
        self._terminal_kernel = RiverTerminalFastKernel.build(self.board)
        self._native = _native_arrays(self._terminal_kernel)
        self._init_terminal_workspaces()
        self._init_flat_tree()

    def _index_tree(self) -> None:
        self.nodes: list[RiverNode] = []
        self.decision: list[int] = []
        self.fold_terminals: list[int] = []
        self.showdown_terminals: list[int] = []

        def walk(n: RiverNode) -> None:
            nid = len(self.nodes)
            self.nodes.append(n)
            n._id = nid  # type: ignore[attr-defined]
            if n.terminal is not None:
                if n.terminal == "fold":
                    self.fold_terminals.append(nid)
                elif n.terminal == "showdown":
                    self.showdown_terminals.append(nid)
                else:
                    raise AssertionError(n.terminal)
                return
            self.decision.append(nid)
            k = len(n.children)
            n._reg = np.zeros((k, HAND_COUNT), dtype=np.float64)  # type: ignore[attr-defined]
            n._strat = np.full((k, HAND_COUNT), 1.0 / k, dtype=np.float64)  # type: ignore[attr-defined]
            n._avg_num = np.zeros((k, HAND_COUNT), dtype=np.float64)  # type: ignore[attr-defined]
            n._avg_den = np.zeros(HAND_COUNT, dtype=np.float64)  # type: ignore[attr-defined]
            n._inst = np.zeros((k, HAND_COUNT), dtype=np.float64)  # type: ignore[attr-defined]
            for c in n.children:
                walk(c)

        walk(self.tree)
        N = len(self.nodes)
        self.reach = np.zeros((N, 2, HAND_COUNT), dtype=np.float64)
        self.self_reach = np.zeros((N, 2, HAND_COUNT), dtype=np.float64)
        self.value = np.zeros((N, 2, HAND_COUNT), dtype=np.float64)

    def _init_terminal_workspaces(self) -> None:
        self._show_ix = np.asarray(self.showdown_terminals, dtype=np.int64)
        self._fold_ix = np.asarray(self.fold_terminals, dtype=np.int64)
        ns, nf = len(self._show_ix), len(self._fold_ix)
        self._show_r0 = np.empty((ns, HAND_COUNT), dtype=np.float64)
        self._show_r1 = np.empty((ns, HAND_COUNT), dtype=np.float64)
        self._show_u0 = np.empty((ns, HAND_COUNT), dtype=np.float64)
        self._show_u1 = np.empty((ns, HAND_COUNT), dtype=np.float64)
        self._fold_r0 = np.empty((nf, HAND_COUNT), dtype=np.float64)
        self._fold_r1 = np.empty((nf, HAND_COUNT), dtype=np.float64)
        self._fold_u0 = np.empty((nf, HAND_COUNT), dtype=np.float64)
        self._fold_u1 = np.empty((nf, HAND_COUNT), dtype=np.float64)
        self._show_b = np.asarray([float(min(self.nodes[int(i)].spent)) for i in self._show_ix])[:, None]
        self._fold_b = np.asarray([float(min(self.nodes[int(i)].spent)) for i in self._fold_ix])[:, None]
        self._fold_sgn = np.asarray([
            -1.0 if self.nodes[int(i)].folder == 0 else 1.0 for i in self._fold_ix
        ])[:, None]

    def _init_flat_tree(self) -> None:
        N=len(self.nodes)
        player=np.full(N,-1,dtype=np.int8)
        edge_start=np.zeros(N+1,dtype=np.int32)
        children=[]
        for n in self.nodes:
            nid=n._id  # type: ignore[attr-defined]
            edge_start[nid]=len(children)
            if n.terminal is None:
                player[nid]=n.player
                children.extend(c._id for c in n.children)  # type: ignore[attr-defined]
        edge_start[N]=len(children)
        self._flat_player=player
        self._flat_edge_start=edge_start
        self._flat_children=np.asarray(children,dtype=np.int32)

    def solve_numba_flat(self, player0_range: np.ndarray, player1_range: np.ndarray) -> RiverSolveResult:
        if njit is None:
            raise RuntimeError("numba unavailable")
        if not self.dcfr.simultaneous:
            raise NotImplementedError("flat backend currently implements simultaneous updates")
        r0=np.asarray(player0_range,dtype=np.float64)
        r1=np.asarray(player1_range,dtype=np.float64)
        if r0.shape != (HAND_COUNT,) or r1.shape != (HAND_COUNT,):
            raise ValueError("ranges must each have shape [1326]")
        if np.any(r0[~self.pm] != 0) or np.any(r1[~self.pm] != 0):
            raise ValueError("board-blocked hands must have zero probability")
        if not np.isclose(r0.sum(),1.0,atol=1e-6) or not np.isclose(r1.sum(),1.0,atol=1e-6):
            raise ValueError("ranges must each sum to one")
        ranks,gids,offsets,ids=self._native
        legal=self._terminal_kernel.legal_mask.astype(np.bool_)
        root_cfvs,root_strategy=_flat_dcfr_solve(
            self.dcfr.iterations,self.dcfr.alpha,self.dcfr.beta,self.dcfr.delay,
            self._flat_player,self._flat_edge_start,self._flat_children,
            self._show_ix,self._fold_ix,self._show_b[:,0],self._fold_b[:,0],self._fold_sgn[:,0],
            ranks,gids,offsets,ids,legal,HAND_CARDS.astype(np.int16),r0,r1)
        u0=float(np.dot(r0,root_cfvs[0])); u1=float(np.dot(r1,root_cfvs[1]))
        return RiverSolveResult(root_cfvs,root_strategy,u0+u1,self.dcfr.iterations,
                                len(self.decision),len(self.fold_terminals)+len(self.showdown_terminals))

    @staticmethod
    def _regret_matching_into(reg: np.ndarray, out: np.ndarray) -> None:
        np.maximum(reg, 0.0, out=out)
        denom = out.sum(axis=0)
        good = denom > 0
        if np.any(good):
            out[:, good] /= denom[good][None, :]
        bad = ~good
        if np.any(bad):
            out[:, bad] = 1.0 / reg.shape[0]

    @staticmethod
    def _regret_matching(reg: np.ndarray) -> np.ndarray:
        pos = np.maximum(reg, 0.0)
        denom = pos.sum(axis=0, keepdims=True)
        k = reg.shape[0]
        out = np.empty_like(reg)
        np.divide(pos, denom, out=out, where=denom > 0)
        no_pos = (denom[0] <= 0)
        if np.any(no_pos):
            out[:, no_pos] = 1.0 / k
        return out

    def _propagate_reach(self, r1: np.ndarray, r2: np.ndarray) -> None:
        self.reach.fill(0.0)
        self.self_reach.fill(0.0)
        self.reach[0, 0] = r1
        self.reach[0, 1] = r2
        # Player-own action reach starts at 1 for each feasible private hand;
        # private-card prior probabilities are chance, not player reach.
        legal = self.pm.astype(np.float64)
        self.self_reach[0, 0] = legal
        self.self_reach[0, 1] = legal
        for n in self.nodes:
            if n.terminal is not None:
                continue
            nid = n._id  # type: ignore[attr-defined]
            p = n.player
            st = n._strat  # type: ignore[attr-defined]
            for a, c in enumerate(n.children):
                cid = c._id  # type: ignore[attr-defined]
                self.reach[cid] = self.reach[nid]
                self.self_reach[cid] = self.self_reach[nid]
                self.reach[cid, p] = self.reach[nid, p] * st[a]
                self.self_reach[cid, p] = self.self_reach[nid, p] * st[a]

    def _terminal_values(self) -> None:
        ranks, gids, offsets, ids = self._native
        legal = self._terminal_kernel.legal_mask.astype(np.bool_)
        hc = HAND_CARDS.astype(np.int16)
        if self._show_ix.size:
            np.take(self.reach[:, 1], self._show_ix, axis=0, out=self._show_r1)
            np.take(self.reach[:, 0], self._show_ix, axis=0, out=self._show_r0)
            _numba_showdown_batch_into(self._show_r1, ranks, gids, offsets, ids, legal, self._show_u0)
            _numba_showdown_batch_into(self._show_r0, ranks, gids, offsets, ids, legal, self._show_u1)
            self.value[self._show_ix, 0] = self._show_b * self._show_u0
            self.value[self._show_ix, 1] = self._show_b * self._show_u1
        if self._fold_ix.size:
            np.take(self.reach[:, 1], self._fold_ix, axis=0, out=self._fold_r1)
            np.take(self.reach[:, 0], self._fold_ix, axis=0, out=self._fold_r0)
            _numba_fold_batch_into(self._fold_r1, offsets, ids, legal, hc, self._fold_u0)
            _numba_fold_batch_into(self._fold_r0, offsets, ids, legal, hc, self._fold_u1)
            self.value[self._fold_ix, 0] = self._fold_sgn * self._fold_b * self._fold_u0
            self.value[self._fold_ix, 1] = -self._fold_sgn * self._fold_b * self._fold_u1

    def _bottom_up_and_instant_regrets(self) -> None:
        for nid in range(len(self.nodes) - 1, -1, -1):
            n = self.nodes[nid]
            if n.terminal is not None:
                continue
            p = n.player
            st = n._strat  # type: ignore[attr-defined]
            avals = n._inst  # type: ignore[attr-defined]
            u = self.value[nid]
            u.fill(0.0)
            for a, c in enumerate(n.children):
                cv = self.value[c._id]  # type: ignore[attr-defined]
                avals[a] = cv[p]
                u[p] += st[a] * cv[p]
                u[1 - p] += cv[1 - p]
            avals -= u[p][None, :]

    def _discount_and_update_regrets(self, t: int) -> None:
        pos_factor = self.dcfr.positive_discount(t)
        neg_factor = self.dcfr.negative_discount(t)
        for nid in self.decision:
            n = self.nodes[nid]
            reg = n._reg  # type: ignore[attr-defined]
            # Brown--Sandholm DCFR discounts the cumulative regret on the
            # current iteration.  Add the instantaneous regret first, then
            # apply the sign-dependent iteration-t discount.  This ordering
            # is also what makes the LCFR special case reproduce linear
            # iteration weights.
            reg += n._inst  # type: ignore[attr-defined]
            pos = reg > 0
            neg = reg < 0
            reg[pos] *= pos_factor
            reg[neg] *= neg_factor
            # Prevent pathological CPU slowdowns on IEEE-754 subnormals.
            # This only maps magnitudes below the smallest *normal* float64
            # (~2.225e-308) to exact zero; it is an explicit numerical
            # engineering seam, not a claimed Supremus CUDA detail.
            tiny = (reg != 0.0) & (np.abs(reg) < _FLOAT64_TINY)
            if np.any(tiny):
                reg[tiny] = 0.0

    def _accumulate_average_strategy(self, t: int) -> None:
        w = self.dcfr.average_weight(t)
        if w <= 0:
            return
        for nid in self.decision:
            n = self.nodes[nid]
            p = n.player
            own = self.self_reach[nid, p]
            st = n._strat  # type: ignore[attr-defined]
            contrib = w * own
            n._avg_num += st * contrib[None, :]  # type: ignore[attr-defined]
            n._avg_den += contrib  # type: ignore[attr-defined]

    def _finalize_average_strategy(self) -> None:
        for nid in self.decision:
            n = self.nodes[nid]
            den = n._avg_den  # type: ignore[attr-defined]
            num = n._avg_num  # type: ignore[attr-defined]
            avg = np.empty_like(num)
            np.divide(num, den[None, :], out=avg, where=den[None, :] > 0)
            missing = den <= 0
            if np.any(missing):
                avg[:, missing] = self._regret_matching(n._reg)[:, missing]  # type: ignore[attr-defined]
            n._avg = avg  # type: ignore[attr-defined]

    def _evaluate_profile(self, r1: np.ndarray, r2: np.ndarray,
                          use_average: bool) -> np.ndarray:
        # Temporarily swap current strategies to average strategies.
        saved = []
        if use_average:
            for nid in self.decision:
                n = self.nodes[nid]
                saved.append(n._strat)  # type: ignore[attr-defined]
                n._strat = n._avg  # type: ignore[attr-defined]
        try:
            self._propagate_reach(r1, r2)
            self.value.fill(0.0)
            self._terminal_values()
            self._bottom_up_and_instant_regrets()
            return self.value[0].copy()
        finally:
            if use_average:
                for nid, st in zip(self.decision, saved):
                    self.nodes[nid]._strat = st  # type: ignore[attr-defined]

    def solve(self, player0_range: np.ndarray,
              player1_range: np.ndarray) -> RiverSolveResult:
        if not self.dcfr.simultaneous:
            raise NotImplementedError(
                "this paper-profile river solver currently implements only "
                "simultaneous updates"
            )
        r0 = np.asarray(player0_range, dtype=np.float64)
        r1 = np.asarray(player1_range, dtype=np.float64)
        if r0.shape != (HAND_COUNT,) or r1.shape != (HAND_COUNT,):
            raise ValueError("ranges must each have shape [1326]")
        if np.any(r0[~self.pm] != 0) or np.any(r1[~self.pm] != 0):
            raise ValueError("board-blocked hands must have zero probability")
        if not np.isclose(r0.sum(), 1.0, atol=1e-6) or not np.isclose(r1.sum(), 1.0, atol=1e-6):
            raise ValueError("ranges must each sum to one")

        for t in range(1, self.dcfr.iterations + 1):
            # Strategies are a function of regrets at the START of iteration.
            for nid in self.decision:
                n = self.nodes[nid]
                self._regret_matching_into(n._reg, n._strat)  # type: ignore[attr-defined]
            self._propagate_reach(r0, r1)
            self.value.fill(0.0)
            self._terminal_values()
            self._bottom_up_and_instant_regrets()
            self._accumulate_average_strategy(t)
            self._discount_and_update_regrets(t)

        self._finalize_average_strategy()
        root_cfvs = self._evaluate_profile(r0, r1, use_average=True)
        u0 = float(np.dot(r0, root_cfvs[0]))
        u1 = float(np.dot(r1, root_cfvs[1]))
        return RiverSolveResult(
            root_cfvs=root_cfvs,
            root_strategy=self.tree._avg.copy(),  # type: ignore[attr-defined]
            weighted_zero_sum_residual=u0 + u1,
            iterations=self.dcfr.iterations,
            decision_nodes=len(self.decision),
            terminal_nodes=len(self.fold_terminals) + len(self.showdown_terminals),
        )


__all__ = ["DcfrPlusSpec", "RiverSolveResult", "RiverDcfrPlusEngine"]
