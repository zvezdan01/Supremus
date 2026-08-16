# Research notes — availability and adjacent work

Date: 2026-08-16

## Supremus source code: not public

Searched for a code release accompanying arXiv:2007.10442. **None found.** The
paper is indexed on arXiv, Semantic Scholar, ADS, ResearchGate and DeepAI with
no linked repository, and the authors' affiliations (Minimal AI, FAIR) show no
corresponding public release.

This corroborates rather than resolves the project's blockers. The five
unresolved items in `HUNL_RIVER_VALUE_V1_MILESTONE.md` — private 1000-bucket
river artifact, integer-chip rounding of the Table-2 fractions, private RNG and
dataset ordering, the river-training update mode, and original weights — remain
closed by absence of the private code.

Status: **NEGATIVE FINDING, recorded so it is not re-searched blind.** It is not
proof of destruction; it is proof that no public copy was locatable on this
date.

---

# DEVN — "predict expected values instead": EVALUATED, NOT ADOPTED

Jeremiasz Wołosiuk, Maciej Świechowski, Jacek Mańdziuk,
*Don't Predict Counterfactual Values, Predict Expected Values Instead*,
AAAI-23. Supplementary code and PDF at
`github.com/jwolosiuk/dont-predict-cfvs-predict-evs-instead`.

## Licence position

That repository contains **no LICENCE file**, and its README states the full
code version is used commercially. It is therefore all-rights-reserved. The
supplementary PDF is © 2023 AAAI, all rights reserved.

Consequently: **nothing from it is vendored, copied or derived in this
repository.** The published method is referenced and measured; the authors'
expression of it is not reused. `run_ev_factorization_analysis.py` was written
against this project's own primitives.

## The method

A counterfactual value factorizes exactly:

    CFV(h) = matchup(h) · EV(h)
    matchup(h) = Σ over opponent hands compatible with h of the opponent range

`matchup` is computable in closed form from the opponent range, which the
network already receives as input. So a network regressing CFV directly spends
capacity relearning a known quantity. DEVN regresses EV and multiplies by the
exact matchup afterwards.

The identity is real, and it holds on this project's data to machine precision —
**max |EV·matchup − CFV| = 4.5e-13 chips** across all three frozen
4,000-iteration subgames, both players (`run_ev_factorization_analysis.py`).

## Why it is not adopted

### 1. The paper's own comparable evidence is for the *no-abstraction* setting

The headline **3.37–8.39% relative improvement** comes from supplementary
settings VI and VII, both of which use **"identity bucketing" — no card
abstraction**, each hand in its own bucket, plus a 52-element one-hot board
encoding as extra input.

For the five settings that *do* use bucketing (supplementary Tables 1–5), the
only figures given are each method's loss **on its own target**. EV-loss and
CFV-loss are losses on different quantities at different scales, so those tables
do not support a cross-method comparison in either direction — including the
naive reading that CFV wins there.

This project uses a 1000-bucket abstraction, because Supremus and DeepStack do.
The regime where DEVN is demonstrated is not the regime this project is in.

### 2. It breaks bucket-space zero-sum enforcement

The supplementary is explicit: under DEVN the zero-sum property **cannot** be
enforced on bucketed EVs, because the outer network would need matchups for
bucketed ranges, which the authors call not feasible. It can only be imposed
after inverse bucketing.

This project's `DeepStackHUNLValueNet` enforces zero-sum as an outer layer in
**bucket space**, and `HUNL_RIVER_VALUE_V1_CERT` certifies it there
(`bucket_weighted_zero_sum_residual_after_training`, tolerance 5e-5). Adopting
DEVN would move that guarantee downstream and invalidate the existing
certificate. The authors state they did not verify the playing-strength impact
of not enforcing it, and left it as further study.

Trading a certified architectural invariant for an unquantified one is not a
trade this project should make.

### 3. On this project's river data the mechanism is nearly absent

Measured on the three real 4,000-iteration subgames:

| quantity | measured |
|---|---|
| mean matchup | **0.9158** = 990/1081 |
| coefficient of variation | 4.2% – 7.7% |
| max/min across hands | 1.30 – 1.88 |
| \|corr(\|CFV\|, matchup)\| | 0.077 – 0.458 |
| std(EV) / std(CFV) | **1.07 – 1.12** |

On a five-card river board, blockers remove only 91 of 1081 hands, so the reach
factor is nearly uniform. Dividing by a near-constant ≈0.92 makes the EV
target's spread slightly **larger** than the CFV target's — the opposite of the
simplification the method is meant to deliver.

The authors' own experiments are on 4-card (turn) boards, where more of the deck
is live and their generated ranges may be far more concentrated than the
DeepStack-style `R(S,p)` ranges used here. Their result is not contradicted;
it simply does not transfer to this setting on the evidence available.

## If it is ever revisited

The prerequisites, in order:

1. a river network trained on enough data that a few-percent difference is
   measurable at all — the current 3-sample checkpoint is deliberate
   overfitting and cannot resolve this;
2. a decision on where zero-sum is enforced, with a certificate for the new
   position;
3. the comparison run in **card space after inverse bucketing**, which is this
   project's loss convention and the only space where the two methods are
   directly comparable.

The frozen raw chip CFVs make step 3 cheap: an EV-target variant is derivable
from them without re-solving a single subgame. That is exactly the property the
V1 milestone was designed to preserve, so revisiting costs nothing but training
time.

Any such work belongs behind an explicit `EV_FACTORED` profile with its own
certificate, with the paper-faithful CFV path remaining the default — the same
lineage discipline that keeps Supremus out of the forensic DeepStack baseline in
`quant-trade`.

---

## Schmid correspondence

The direct email testimony from Martin Schmid (2026-08-16, recorded in
`quant-trade` at `FORENSICS/SCHMID_TESTIMONY_2026-08-16.md`) bears on the
DeepStack side — the existence of a skip/burn-in window in offline target
generation — and it was Schmid who recommended arXiv:2007.10442, the paper this
repository is built on. Its scope warning applies here too: the testimony does
not establish the private RNG, seed schedule, river bucketing, serialization, or
any exact numerical constant.
