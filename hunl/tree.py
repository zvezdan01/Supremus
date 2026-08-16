"""HUNL river betting tree (Gate G1 tree layer).

Builds the complete sparse betting tree for a river re-solve situation:
both players entered the river having committed `pot_half` chips each,
seat 0 (big blind) acts first (ACPC firstPlayer round 4), and the action
menus follow the VERIFIED Table-4 river configuration in
`hunl.config.HunlConfig`.

Betting-rule model (mirrors untouched ACPC game.c; the certification
harness verifies EVERY node, window and candidate against the compiled
original — game.c is the authority, this file only mirrors it):
  - fold valid iff the player still faces chips (spent < max_spent)
    [game.c isValidAction:889-897]
  - call always valid; spends min(max_spent, stack) [doAction:936-946]
  - raise-to window: min = min_raise_to, max = own stack; if
    min > stack: no raise when max_spent >= stack, else all-in-only
    (min = max) [raiseIsValid:788-843]; no raise when the opponent is
    all-in (numActingPlayers <= 1) [raiseIsValid:807-812]
  - after a raise to R: min_raise_to' = max(min_raise_to, 2R - max_spent)
    [doAction:957-962]; a short all-in raise does NOT increase it
  - entering a post-flop round: min_raise_to = max_spent + big blind
    [doAction:1003-1010]
  - round closes when all acting players have called; on the river the
    game is then finished (showdown) [doAction:988-1022]

State/terminal contracts (FROZEN):
  - spent[i] = TOTAL chips committed by seat i (all rounds); remaining
    stack = stack - spent[i]; pot = spent[0] + spent[1]
  - fold terminal: `folder` loses spent[folder] to the other seat
  - showdown terminal: spent[0] == spent[1]; G1.3 showdown matrix applies
  - action encoding: ("fold", 0), ("call", 0), ("raise", raise_to_total)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .config import DEFAULT_CONFIG, HunlConfig

Action = tuple[str, int]


@dataclass
class RiverNode:
    player: int                      # seat to act (decision nodes)
    spent: tuple[int, int]
    max_spent: int
    min_raise_to: int
    action_depth: int                # actions taken this round
    terminal: str | None = None      # None | "fold" | "showdown"
    folder: int | None = None
    actions: list[Action] = field(default_factory=list)
    children: list["RiverNode"] = field(default_factory=list)


class RiverTreeBuilder:
    def __init__(self, cfg: HunlConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    # -- rule mirror (authority: game.c via the certification harness) --
    def fold_valid(self, node: RiverNode) -> bool:
        p = node.player
        return (node.spent[p] != node.max_spent
                and node.spent[p] != self.cfg.stack)

    def raise_window(self, node: RiverNode) -> tuple[int, int] | None:
        p = node.player
        opp = 1 - p
        if node.spent[opp] >= self.cfg.stack or node.spent[p] >= self.cfg.stack:
            return None                       # <=1 acting player
        mn, mx = node.min_raise_to, self.cfg.stack
        if mn > mx:
            if node.max_spent >= mx:
                return None
            return (mx, mx)                   # all-in-only (short) raise
        return (mn, mx)

    # ------------------------------------------------------- builder --
    def build(self, pot_half: int) -> RiverNode:
        cfg = self.cfg
        assert pot_half == cfg.big_blind or \
            2 * cfg.big_blind <= pot_half < cfg.stack, \
            f"unreachable river pot half {pot_half}"
        root = RiverNode(
            player=cfg.first_player[3],
            spent=(pot_half, pot_half),
            max_spent=pot_half,
            min_raise_to=pot_half + cfg.big_blind,
            action_depth=0,
        )
        self._expand(root)
        return root

    def _expand(self, node: RiverNode) -> None:
        if node.terminal is not None:
            return
        cfg = self.cfg
        p = node.player
        opp = 1 - p

        if self.fold_valid(node):
            child = RiverNode(opp, node.spent, node.max_spent,
                              node.min_raise_to, node.action_depth + 1,
                              terminal="fold", folder=p)
            node.actions.append(("fold", 0))
            node.children.append(child)

        # call / check
        call_to = min(node.max_spent, cfg.stack)
        spent = list(node.spent)
        spent[p] = call_to
        closes = node.action_depth >= 1
        child = RiverNode(opp, tuple(spent), node.max_spent,
                          node.min_raise_to, node.action_depth + 1,
                          terminal="showdown" if closes else None)
        node.actions.append(("call", 0))
        node.children.append(child)
        if not closes:
            self._expand(child)

        window = self.raise_window(node)
        if window is not None:
            mn, mx = window
            cands = set()
            for r in cfg.raise_to_candidates(node.max_spent, node.action_depth):
                if mn <= r <= mx:
                    cands.add(r)
            if cfg.river_allin:
                cands.add(mx)                 # all-in (== short raise if mn==mx)
            for r in sorted(cands):
                spent = list(node.spent)
                spent[p] = r
                child = RiverNode(
                    opp, tuple(spent), r,
                    max(node.min_raise_to, 2 * r - node.max_spent),
                    node.action_depth + 1)
                node.actions.append(("raise", r))
                node.children.append(child)
                self._expand(child)

        assert node.actions, "decision node with no legal action"

    # ------------------------------------------------------ manifest --
    @staticmethod
    def manifest_lines(root: RiverNode) -> list[str]:
        out: list[str] = []

        def walk(node: RiverNode, path: str) -> None:
            term = node.terminal or "-"
            folder = node.folder if node.folder is not None else "-"
            acts = ",".join(f"{a}{'' if a != 'raise' else ':' + str(s)}"
                            for a, s in node.actions)
            out.append(f"{path}|p{node.player}|{node.spent[0]},{node.spent[1]}"
                       f"|M{node.max_spent}|mr{node.min_raise_to}"
                       f"|d{node.action_depth}|{term}|{folder}|{acts}")
            for (a, s), c in zip(node.actions, node.children):
                walk(c, path + "/" + (a[0] if a != "raise" else f"r{s}"))

        walk(root, "root")
        return out

    @classmethod
    def manifest_sha256(cls, root: RiverNode) -> str:
        return hashlib.sha256(
            "\n".join(cls.manifest_lines(root)).encode()).hexdigest()


def count_nodes(node: RiverNode) -> int:
    return 1 + sum(count_nodes(c) for c in node.children)
