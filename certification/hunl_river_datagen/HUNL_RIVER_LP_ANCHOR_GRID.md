# The engine's external cross-check, widened 10x

Date: 2026-08-16 · Harness: `run_lp_anchor_grid.py` · Certificate:
`HUNL_RIVER_LP_ANCHOR_GRID_V1.json`

## Why this was run

Almost every certificate in this project is *internal*: determinism, byte
reproducibility, zero-sum residuals, agreement between a fast path and a
reference path. Those catch regressions and prove reproducibility. None of them
would notice if the solver were consistently, reproducibly wrong.

Exactly one check is external — a sequence-form linear program that computes the
value of the same game by a completely different route, sharing no CFR code
with the solver. It was run at a **support of 6 hands per player**. For a
solver used on 1081-hand games, that is a thin basis.

Widening it was the one verification gap this project could close on its own,
without waiting on anybody.

## Result

The DCFR+ side always solves the **full 1326-hand game** with ranges zeroed
outside the support, so the engine under test is the production one; only the
LP is restricted. Three independent random supports per size.

**Convergence, at the widest support (64 hands per player):** worst gap across
the three draws.

| DCFR+ iterations | worst gap, % of pot |
|---|---|
| 500 | 1.1598 |
| 1,000 | 0.3207 |
| 2,000 | 0.1606 |
| **4,000 (production)** | **0.0665** |

The gap roughly halves with each doubling of iterations. That is the signature
of convergence error, not of a disagreement about what the game is: a
structural discrepancy would not shrink.

**Support, at a fixed 1,000 iterations:** worst gap across three draws.

| support / player | worst gap, % of pot |
|---|---|
| 6 | 0.9775 |
| 12 | 0.2874 |
| 24 | 0.5011 |
| 48 | 0.8239 |
| 64 | 0.3207 |

No trend. The residual at a fixed iteration count depends on which game was
drawn, not on how wide the support is — the worst case here is at support 6,
the *narrowest*. Richer restricted games do not degrade the agreement.

## What this replaces

| | previous certificate | now |
|---|---|---|
| support per player | 6 | **64** |
| iterations | 500 | **4,000** (the production setting) |
| random draws | 1 | **3 per configuration** |
| gap | 0.121% of pot | **0.0665% of pot** |

The engine now agrees with an independent oracle to **0.067% of the pot** on
games ten times wider than before, at the iteration count the data generator
actually uses, with the residual behaving as convergence error.

Every other sensitivity figure in this project is quoted against the old 0.121%
threshold. Those comparisons remain valid — 0.121% was and is a conservative
bar — but the engine is better than that number suggested.

## Scope and limits

- **Restricted support, not full 1326.** The LP is quadratic in the support and
  a single solve already takes 190 s at 96 hands, so the full game is out of
  reach by this route. 64 of 1081 hands is a sample of the game, not the game.
- **One board, one pot.** Board `(0,5,10,15,20)`, `pot_half = 100`, inherited
  from the V1 certificate so the numbers stay comparable. A sweep over boards
  would test the terminal kernels harder; those are separately certified
  against exhaustive recounts.
- **Numba backend.** The grid uses `solve_numba_flat`, which is
  regression-checked against the Python reference backend rather than being the
  reference itself.

## What would strengthen it further

Extending the LP to a second board type — a coordinated board, where hand
strengths collapse into few classes — would exercise the showdown and blocker
paths under the conditions that make the bucketing degenerate. Cost is the same
as this run.
