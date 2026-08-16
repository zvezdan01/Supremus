"""Supremus paper action-abstraction contract for HUNL river solving.

Primary source: Zarick et al., arXiv:2007.10442v1, Table 2.

The paper specifies pot fractions but does not specify how non-integral chip
amounts are quantized in the custom CUDA implementation.  This module makes
that missing detail explicit instead of silently hiding it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from .config import HunlConfig


class ChipQuantization(str, Enum):
    AUTHOR_STRICT = "AUTHOR_STRICT"
    NEAREST_HALF_UP = "NEAREST_HALF_UP"
    FLOOR = "FLOOR"


class UnpublishedSupremusDetail(RuntimeError):
    pass


def _quantize_fraction(x: Fraction, policy: ChipQuantization) -> int:
    if x.denominator == 1:
        return x.numerator
    if policy == ChipQuantization.AUTHOR_STRICT:
        raise UnpublishedSupremusDetail(
            "Supremus Table-2 pot fraction produces a non-integral chip "
            "amount; the paper does not publish the integer-chip rounding rule"
        )
    if policy == ChipQuantization.FLOOR:
        return x.numerator // x.denominator
    if policy == ChipQuantization.NEAREST_HALF_UP:
        q, r = divmod(x.numerator, x.denominator)
        return q + (1 if 2 * r >= x.denominator else 0)
    raise AssertionError(policy)


@dataclass(frozen=True)
class SupremusRiverConfig:
    """Duck-compatible configuration for ``RiverTreeBuilder``.

    Table 2 (Supremus):
      first     F,C,.33,.5,.75,1,1.25,2,A
      second    F,C,.25,.5,1,A
      third     F,C,.25,A
      remaining F,C,1,A

    ``rounding`` is PROJECT-EXPLICIT because the paper does not state the
    integer-chip quantization convention of the private CUDA code.
    """

    stack: int = 20_000
    blinds: tuple[int, int] = (100, 50)
    num_rounds: int = 4
    first_player: tuple[int, int, int, int] = (1, 0, 0, 0)
    river_allin: bool = True
    rounding: ChipQuantization = ChipQuantization.NEAREST_HALF_UP

    river_menus: tuple[tuple[Fraction, ...], ...] = (
        (Fraction(33,100), Fraction(1,2), Fraction(3,4), Fraction(1),
         Fraction(5,4), Fraction(2)),
        (Fraction(1,4), Fraction(1,2), Fraction(1)),
        (Fraction(1,4),),
        (Fraction(1),),
    )

    @property
    def big_blind(self) -> int:
        return max(self.blinds)

    def menu_for_depth(self, depth: int) -> tuple[Fraction, ...]:
        return self.river_menus[min(depth, len(self.river_menus) - 1)]

    def raise_to_candidates(self, max_spent: int, depth: int) -> list[int]:
        # Same DeepStack/Leduc sizing seam used elsewhere in this project:
        # after matching the opponent, bet f * current total pot.
        pot = 2 * int(max_spent)
        out: list[int] = []
        for f in self.menu_for_depth(depth):
            delta = _quantize_fraction(f * pot, self.rounding)
            out.append(int(max_spent) + int(delta))
        return out


__all__ = [
    "ChipQuantization", "UnpublishedSupremusDetail", "SupremusRiverConfig",
]
