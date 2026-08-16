"""HUNL game + river-resolving configuration.

Every constant is anchored to a VERIFIED row of HUNL_RECONSTRUCTION_SPEC v2
(spec freeze f81d08c):
  - game constants: ACPC `holdem.nolimit.2p.reverse_blinds.game`
      numPlayers=2, stack=20000 20000, blind=100 50 (seat0=big blind),
      firstPlayer=2 1 1 1 (1-based; 0-based: seat1 preflop, seat0 after)
  - river action menus + iterations: DeepStack supplement Table 4
      (arXiv 1701.01724v3 p.22):
        river 1st action  {F, C, 1/2P, P, 2P, A}
        river 2nd action  {F, C, 1/2P, P, 2P, A}
        river remaining   {F, C, P, A}
        river re-solve: 2000 CFR iterations, first 1000 omitted
  - bet-size chip semantics: fraction-of-pot with pot = 2*max_spent
    (call amount included), raise-to = max_spent + fraction*pot — the
    author bet-sizing convention (Leduc bet_sizing.lua) applied to the
    Table-4 fractions; with fractions {1/2, 1, 2} every candidate is an
    exact integer chip amount (delta = max_spent, 2*max_spent,
    4*max_spent). Legality is decided by ACPC game.c, never by this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


@dataclass(frozen=True)
class HunlConfig:
    stack: int = 20000
    blinds: tuple[int, int] = (100, 50)      # seat0 = big blind, seat1 = small
    num_rounds: int = 4
    # 0-based seat acting first in each round (ACPC firstPlayer = 2 1 1 1)
    first_player: tuple[int, int, int, int] = (1, 0, 0, 0)

    # Table 4, river row: menus by action depth (0, 1, >=2); fractions of pot
    river_menus: tuple[tuple[Fraction, ...], ...] = (
        (Fraction(1, 2), Fraction(1), Fraction(2)),
        (Fraction(1, 2), Fraction(1), Fraction(2)),
        (Fraction(1),),
    )
    river_allin: bool = True                  # A present at every depth
    river_cfr_iters: int = 2000               # Table 4 (not used by the tree)
    river_cfr_omit: int = 1000                # Table 4 (not used by the tree)

    # Table 4, pre-flop row (VERIFIED, DeepStack supplement p.22):
    # first {F,C,1/2P,P,A}, second {F,C,1/2P,P,2P,A}, remaining {F,C,P,A}.
    preflop_menus: tuple[tuple[Fraction, ...], ...] = (
        (Fraction(1, 2), Fraction(1)),
        (Fraction(1, 2), Fraction(1), Fraction(2)),
        (Fraction(1),),
    )
    preflop_allin: bool = True
    preflop_cfr_iters: int = 1000
    preflop_cfr_omit: int = 980

    # Table 4, flop row: first {F,C,1/2P,P,A}, second and remaining
    # {F,C,P,A}.  The lookahead stops at the turn boundary (turn NN).
    flop_menus: tuple[tuple[Fraction, ...], ...] = (
        (Fraction(1, 2), Fraction(1)),
        (Fraction(1),),
        (Fraction(1),),
    )
    flop_allin: bool = True
    flop_cfr_iters: int = 1000
    flop_cfr_omit: int = 500

    # Table 4, turn row (VERIFIED, spec f81d08c): lookahead-global action
    # depths — first {F,C,1/2P,P,A}, second {F,C,P,A}, remaining {F,C,P,A}
    # (the "remaining" menu also governs river betting inside a turn
    # resolve, which is solved to the end of the game with no NN).
    turn_menus: tuple[tuple[Fraction, ...], ...] = (
        (Fraction(1, 2), Fraction(1)),
        (Fraction(1),),
        (Fraction(1),),
    )
    turn_allin: bool = True
    turn_cfr_iters: int = 1000                # Table 4 turn schedule
    turn_cfr_omit: int = 500

    def menu_for_street(self, street: str, depth: int) -> tuple[Fraction, ...]:
        """Table-4 fraction menu for one resolving street.

        `depth` is the lookahead action depth.  For pre-flop/flop, the
        lookahead ends at the next-street boundary, so this is also the
        current-round action depth.  Turn uses its existing global-depth
        convention across the turn->river exact-to-end lookahead.
        """
        menus = {
            "preflop": self.preflop_menus,
            "flop": self.flop_menus,
            "turn": self.turn_menus,
            "river": self.river_menus,
        }[street]
        return menus[min(depth, len(menus) - 1)]

    def menu_for_depth(self, depth: int) -> tuple[Fraction, ...]:
        # Backward-compatible river helper used by the frozen river tree.
        return self.menu_for_street("river", depth)

    @property
    def big_blind(self) -> int:
        return max(self.blinds)

    def raise_to_candidates(self, max_spent: int, depth: int) -> list[int]:
        """Exact chip raise-to targets from the Table-4 fraction menu.
        pot = 2*max_spent (call included); raise-to = max_spent + f*pot.
        All-in / legality filtering happens in the tree builder."""
        out = []
        pot = 2 * max_spent
        for f in self.menu_for_depth(depth):
            delta = f * pot
            assert delta.denominator == 1, "non-integer chip bet"
            out.append(max_spent + int(delta))
        return out


DEFAULT_CONFIG = HunlConfig()
