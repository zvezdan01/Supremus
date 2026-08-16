"""HUNL full turn game tree: turn betting -> river chance -> river betting
-> showdown/fold (exact, full card space; Gate "exact turn engine").

Betting rules are the SAME certified ACPC mirror used by the frozen river
tree — the rule predicates of `hunl.tree.RiverTreeBuilder` are reused by
composition (duck-typed nodes), never re-implemented, and every tree is
replayed against the untouched game.c oracle by the harness (including
the turn->river round transition and the cross-street min-raise reset
min_raise_to = max_spent + big blind, game.c doAction:1003-1010).

Action menus: Table-4 turn row indexed by LOOKAHEAD-GLOBAL action depth
(`depth_global`: first action = 0 at the turn root; >= 2 = "remaining",
which also governs all river betting inside the turn resolve — the turn
lookahead is solved to the end of the game, supplement p.22-23).
Round closure uses ROUND-LOCAL depth (`round_depth`), reset to 0 on the
river chance transition — exactly the ACPC numActions-per-round rule.
Fold children only where ACPC-legal (dominated-fold lookahead convention
of the golden layered engine is NOT needed by the turn engine and is not
used; documented, value-inert — river gate measured ~0 mass).

Chance nodes: 48 river children ascending, each with the certified
possibility mask; chance algebra = certified transition semantics
(hunl/chance.py, gate 43020db): masked counterfactual reach down,
single x(1/44) on value aggregation up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .chance import river_masks
from .config import DEFAULT_CONFIG, HunlConfig
from .tree import RiverTreeBuilder

Action = tuple[str, int]


@dataclass
class TurnNode:
    street: str                     # "turn" | "river" | "chance"
    player: int                     # acting seat (-1 on chance)
    spent: tuple[int, int]
    max_spent: int
    min_raise_to: int
    depth_global: int               # lookahead-global action depth (menus)
    round_depth: int                # actions taken this betting round
    board: tuple[int, ...]          # 4 cards (turn/chance) or 5 (river)
    terminal: str | None = None     # None | "fold" | "showdown"
    folder: int | None = None
    river_card: int | None = None
    actions: list[Action] = field(default_factory=list)
    children: list["TurnNode"] = field(default_factory=list)
    chance_masks: np.ndarray | None = None


class TurnGameTreeBuilder:
    def __init__(self, cfg: HunlConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self._rules = RiverTreeBuilder(cfg)   # certified ACPC rule mirror

    def _menu(self, depth_global: int) -> tuple[Fraction, ...]:
        menus = self.cfg.turn_menus
        return menus[min(depth_global, len(menus) - 1)]

    def build(self, board4, pot_half: int) -> TurnNode:
        cfg = self.cfg
        board = tuple(int(c) for c in board4)
        assert len(board) == 4
        assert pot_half == cfg.big_blind or \
            2 * cfg.big_blind <= pot_half < cfg.stack, \
            f"unreachable turn pot half {pot_half}"
        root = TurnNode(
            street="turn", player=cfg.first_player[2],
            spent=(pot_half, pot_half), max_spent=pot_half,
            min_raise_to=pot_half + cfg.big_blind,
            depth_global=0, round_depth=0, board=board)
        self._expand(root)
        return root

    def _expand(self, node: TurnNode) -> None:
        cfg = self.cfg
        p = node.player
        opp = 1 - p

        if self._rules.fold_valid(node):
            child = TurnNode(node.street, opp, node.spent, node.max_spent,
                             node.min_raise_to, node.depth_global + 1,
                             node.round_depth + 1, node.board,
                             terminal="fold", folder=p,
                             river_card=node.river_card)
            node.actions.append(("fold", 0))
            node.children.append(child)

        call_to = min(node.max_spent, cfg.stack)
        spent = list(node.spent)
        spent[p] = call_to
        closes = node.round_depth >= 1
        if closes and node.street == "turn":
            child = self._make_chance(tuple(spent), node)
        elif closes:
            child = TurnNode("river", opp, tuple(spent), node.max_spent,
                             node.min_raise_to, node.depth_global + 1,
                             node.round_depth + 1, node.board,
                             terminal="showdown", river_card=node.river_card)
        else:
            child = TurnNode(node.street, opp, tuple(spent), node.max_spent,
                             node.min_raise_to, node.depth_global + 1,
                             node.round_depth + 1, node.board,
                             river_card=node.river_card)
            self._expand(child)
        node.actions.append(("call", 0))
        node.children.append(child)

        window = self._rules.raise_window(node)
        if window is not None:
            mn, mx = window
            allin = (cfg.turn_allin if node.street == "turn"
                     else cfg.river_allin)
            cands = set()
            pot = 2 * node.max_spent
            for f in self._menu(node.depth_global):
                delta = f * pot
                assert delta.denominator == 1, "non-integer chip bet"
                r = node.max_spent + int(delta)
                if mn <= r <= mx:
                    cands.add(r)
            if allin:
                cands.add(mx)
            for r in sorted(cands):
                spent = list(node.spent)
                spent[p] = r
                child = TurnNode(node.street, opp, tuple(spent), r,
                                 max(node.min_raise_to,
                                     2 * r - node.max_spent),
                                 node.depth_global + 1,
                                 node.round_depth + 1, node.board,
                                 river_card=node.river_card)
                node.actions.append(("raise", r))
                node.children.append(child)
                self._expand(child)
        assert node.actions, "decision node with no legal action"

    def _make_chance(self, spent: tuple[int, int],
                     parent: TurnNode) -> TurnNode:
        cfg = self.cfg
        ms = max(spent)
        node = TurnNode("chance", -1, spent, ms, ms + cfg.big_blind,
                        parent.depth_global + 1, 0, parent.board)
        rivers, masks = river_masks(parent.board)
        node.chance_masks = masks
        all_in = ms >= cfg.stack
        for c in rivers:
            board5 = parent.board + (int(c),)
            if all_in:
                child = TurnNode("river", -1, spent, ms, node.min_raise_to,
                                 node.depth_global, 0, board5,
                                 terminal="showdown", river_card=int(c))
            else:
                child = TurnNode("river", cfg.first_player[3], spent, ms,
                                 node.min_raise_to, node.depth_global, 0,
                                 board5, river_card=int(c))
                self._expand(child)
            node.children.append(child)
        return node


def count_turn_nodes(node: TurnNode) -> tuple[int, int, int, int]:
    """(total, decision, terminal, chance)."""
    if node.terminal is not None:
        return (1, 0, 1, 0)
    if node.street == "chance":
        tot, dec, term, ch = 1, 0, 0, 1
    else:
        tot, dec, term, ch = 1, 1, 0, 0
    for c in node.children:
        a, b, d, e = count_turn_nodes(c)
        tot += a; dec += b; term += d; ch += e
    return (tot, dec, term, ch)
