# Equal-strength tie-break: measured, and downgraded

Date: 2026-08-16 · Harness: `run_tiebreak_sensitivity.py` · Certificate:
`HUNL_RIVER_TIEBREAK_SENSITIVITY_V1.json`

## Why this was run

The range generator sorts hands by current-board strength and recursively
splits the sorted vector. Counting distinct strengths on real river boards
gave an alarming picture:

| board | legal hands | distinct strengths | split boundaries inside a tie class |
|---|---|---|---|
| (6,1,46,50,0) | 1081 | 76 | **93.1%** |
| (22,12,17,3,34) | 1081 | 81 | **92.6%** |
| (30,27,50,25,47) | 1081 | 27 | **97.6%** |

Largest single equal-strength class: 240 hands. So almost every split boundary
is resolved by a convention the HUNL supplement never states, and which this
project fixed as ascending hand id, labelled `PROJECT_CANONICAL`.

On that count alone the tie-break looked like it might be the single largest
uncertainty in the generator — plausibly ahead of the bucketing artifact. This
harness was written to check that rather than assume it.

## Design

Four sample indices from the production seed schedule, each generated and
solved at the full 4,000 DCFR+ iterations under four tie-break rules:
`HAND_ID_ASC` (baseline), `HAND_ID_DESC`, `SHUFFLE_1`, `SHUFFLE_2`.

The comparison is controlled by construction. The recursion draws one uniform
per internal node and the node count depends only on the number of legal hands,
which no reordering changes. Verified rather than assumed: every variant
produced the identical board, identical pot and identical `rng_draws = 2167`.
Strength monotonicity was also checked — no tie-break ever reorders distinct
strengths, only hands inside a tie class.

The variants really are different: only 47–79 of 1081 positions coincide with
the baseline ordering.

## Result

| quantity | min | mean | max |
|---|---|---|---|
| range total-variation distance | 0.53 | 0.66 | 0.77 |
| per-hand \|Δtarget\|, mean, % of pot | 1.49 | 2.02 | 2.82 |
| per-hand \|Δtarget\|, max, % of pot | 13.1 | 25.6 | 39.7 |
| **subgame EV delta, % of pot** | **0.001** | **0.106** | **0.229** |

Reference point: the project's independent sequence-form LP anchor agrees with
the DCFR+ solver to **0.121% of the pot**.

## Reading

The two headline numbers point in opposite directions, and both are real.

**The tie-break completely changes which subgame you get.** A total-variation
distance of 0.66 means two thirds of the probability mass sits on different
hands. Individual hand targets move by 2% of the pot on average and by up to
40% in the tail. These are not the same training samples.

**But the subgame's value is nearly invariant.** The EV delta averages 0.106%
of the pot — below the 0.121% gap between our solver and an independent LP.
The tie-break moves the sample within its own family without changing what the
family is worth.

That resolves the worry. Swapping strength-equivalent hands yields a
*different but equally valid* draw from the same distribution: the sort by
strength is preserved, so the range's strength profile is preserved, and only
the identity of individual hands inside each tie class changes. A network
trained on either dataset is learning the same function — counterfactual value
as a function of the input range — sampled at different points of the same
input space.

**Verdict: downgraded.** This is a fidelity gap, not a correctness or quality
problem:

- *Is the generated data valid?* Yes. Every sample is a correct (range, target)
  pair for a legitimately sampled subgame.
- *Does it match the dataset DeepStack and Supremus generated?* No — but that
  was already impossible, since the private RNG and seed schedule are gone.
  This adds nothing new to that verdict.
- *Does it threaten the trained network?* On this evidence, no. It changes
  which points of the input space get sampled, not the function being learned.

The leading uncertainty in this project remains the **bucketing artifact**,
which is an artifact rather than a convention and therefore cannot be derived,
measured around, or averaged out.

## What this does not prove

Four subgames. The EV invariance is well-supported by them but they are not a
distribution-level argument.

More importantly, the fixed convention biases *systematically* rather than
randomly. Ordering by hand id prefers particular card indices inside each tie
class, and card index encodes rank and suit — so the rule is **not
suit-symmetric**, while the game is. Across randomly drawn boards this should
wash out, but it is an asymmetry the original may not have had, and the
argument here is a plausibility argument rather than a measurement.

A stronger test, if it ever matters: train two networks on corpora generated
under different tie-breaks and compare validation loss on a third, held-out
corpus. That is only worth doing once a real corpus exists.

## Position in the question list

Question 8 to Schmid (equal-strength tie ordering) drops from the top tier back
to where it was. Worth asking while asking anyway; not worth waiting for.
