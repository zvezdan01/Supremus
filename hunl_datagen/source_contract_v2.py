"""Literal primary-source contract for HUNL DeepStack training situations.

This module intentionally exposes unresolved publication ambiguities instead of
silently choosing an answer.
"""
from __future__ import annotations

from dataclasses import dataclass


# DeepStack supplement v1/v2/v3 all print these same interval endpoints.
PUBLISHED_POT_INTERVALS = (
    (100, 100, "[)"),
    (200, 400, "[)"),
    (400, 2000, "[)"),
    (2000, 6000, "[)"),
    (6000, 19950, "[]"),
)


class UnresolvedSourceAmbiguity(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratorSourceStatus:
    turn_situations: int = 10_000_000
    cfr_iterations: int = 1_000
    skip_iterations_used: bool = True  # AUTHOR-CONFIRMED recollection (Schmid, 2026-08-16)
    exact_skip_count_published: bool = False
    actions: tuple[str, ...] = ("fold", "call", "pot", "allin")
    card_abstraction: bool = False
    cpu_cores: int = 6_144
    core_years_lower_bound: int = 175


def literal_integer_values(interval_index: int) -> tuple[int, ...]:
    """Return integers literally contained in a published interval.

    The first printed interval ``[100,100)`` is empty under standard interval
    notation.  Raising here is deliberate: no private-source correction has
    been recovered, so AUTHOR_STRICT mode must not turn it into ``{100}``.
    """
    lo, hi, kind = PUBLISHED_POT_INTERVALS[interval_index]
    if kind == "[)":
        vals = tuple(range(lo, hi))
    elif kind == "[]":
        vals = tuple(range(lo, hi + 1))
    else:
        raise AssertionError(kind)
    if not vals:
        raise UnresolvedSourceAmbiguity(
            "published DeepStack pot interval [100,100) contains no integer; "
            "the intended correction is not author-confirmed"
        )
    return vals


def reconstruction_v1_integer_values(interval_index: int) -> tuple[int, ...]:
    """Explicit project reconstruction used only when AUTHOR_STRICT cannot run.

    The project-canonical choice interprets the anomalous first category as the
    point mass {100}.  This is *not* author-confirmed.
    """
    if interval_index == 0:
        return (100,)
    return literal_integer_values(interval_index)


__all__ = [
    "PUBLISHED_POT_INTERVALS",
    "UnresolvedSourceAmbiguity",
    "GeneratorSourceStatus",
    "literal_integer_values",
    "reconstruction_v1_integer_values",
]
