# The remaining two invented choices, measured

Date: 2026-08-16 · Harness: `run_generator_fork_sensitivity.py` · Certificate:
`HUNL_RIVER_GENERATOR_FORK_SENSITIVITY_V1.json`

After the equal-strength tie-break was measured and downgraded, two
`PROJECT_CANONICAL` decisions were left with no number attached. Both are now
settled, and one of them is settled outright rather than merely bounded.

## 1. Chip rounding — NEGLIGIBLE

Table 2 gives bet sizes as fractions of the pot, which are usually not whole
chips, and the paper never states the quantization. This project uses
`NEAREST_HALF_UP`; `FLOOR` is the obvious alternative.

Twelve production-seed samples solved at the full 4,000 iterations under both
rules. Inputs are untouched by the choice, so only the solved game differs and
the comparison is exact per sample.

| quantity | min | mean | max |
|---|---|---|---|
| subgame EV delta, % of pot | 0.0 | 0.0011 | **0.0042** |

The worst case is **0.0042% of the pot — thirty times below** the 0.121%
agreement between the solver and the independent LP, and sixteen times below
the 0.067% the widened LP grid now reports at production settings. The betting
trees were structurally identical on every sample (same decision and terminal
node counts), so the rule shifts a few chip amounts without changing the shape
of the game.

One sample came out exactly zero: at `pot_half = 100` the Table-2 fractions
against a 200-chip pot land on whole chips for every action, so there is
nothing to round.

**Verdict: closed for practical purposes.** The choice remains unpublished and
is still labelled as such, but it cannot move a target by an amount this
project can even measure reliably.

## 2. The `[100,100)` pot category — RESOLVED BY GAME RULES

The published first interval is empty under standard interval notation. The
project reads it as the point mass {100}; the obvious competing reading is that
it is a typo for `[100,200)`, which would also make the five printed intervals
a contiguous partition — an argument that looked, on its face, to favour the
alternative.

It does not, and the reason is not statistical.

`pot_half` is the amount **each** player has committed, so the pot is symmetric
by construction. In HUNL with blinds (50,100):

- both players can be at 100, having merely posted the big blind;
- to be symmetric above that, someone must raise and be called, and the
  smallest legal raise goes from 100 to 200;
- so nothing between 101 and 199 exists.

Testing every integer in `[100,200)` against the certified tree builder — whose
betting rules mirror ACPC `game.c` — gives exactly one reachable value:

```
reachable pot halves in [100,200) = [100]
```

The gap between the first two printed intervals is a **gap in the game, not a
hole in the partition**. `[100,100)` is awkward notation for a set that
genuinely contains one element, and the project's reading is the only
admissible one.

**Verdict: resolved.** This closes a blocker that has been carried as
`UNRESOLVED` since the source audit, and it closes it by derivation from
certified betting rules rather than by author confirmation — which is a
stronger form of closure, because it cannot be withdrawn.

## Effect on the open-questions list

Question 9 to Schmid (`[100,100)`) is **withdrawn**. Asking would waste one of
a small number of chances on something already answerable from the rules.

Question 3 (chip rounding) drops to the bottom: still worth a line if he is
answering anyway, but the measured impact does not justify spending attention
on it.

## Standing after all three sensitivity runs

| invented choice | status |
|---|---|
| equal-strength tie ordering | measured — changes the sample, not its value (EV delta 0.106% mean) |
| chip rounding | measured — **negligible** (EV delta 0.0042% worst) |
| `[100,100)` pot category | **resolved by game rules** |
| seed derivation | affects bit-identity only, which is unreachable regardless |
| **1000-bucket artifact** | **still open, still the leading uncertainty** |

Every convention in the generator has now either been measured or derived. What
remains is the one item that is an artifact rather than a convention.
