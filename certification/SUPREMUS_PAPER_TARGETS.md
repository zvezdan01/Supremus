# Quantitative targets from the primary source

All figures transcribed from Zarick, Pellegrino, Brown, Banister,
*Unlocking the Potential of Deep Counterfactual Value Networks*,
arXiv:2007.10442v1 (vendored at `third_party/papers/`, sha256 `207887d4…9989f00`).

These are **performance and regression targets, not bit-exact oracles.** Nothing
here can certify a reconstruction; they say when one is in the right range.

## Table 1 — Huber loss (paper p.7)

| network | Supremus train | Supremus validation | DeepStack train | DeepStack validation |
|---|---|---|---|---|
| River | 0.010 | 0.015 | N/A | N/A |
| Turn | 0.008 | 0.010 | 0.016 | 0.026 |
| Flop | 0.0092 | 0.011 | 0.008 | 0.034 |
| Auxiliary | 0.000069 | 0.000070 | 0.000053 | 0.000055 |

DeepStack did not use a river network, so the river row has no baseline. The
paper's own reimplementation of DeepStack — a third, distinct column not in
Table 1 — reported validation errors of **0.016 turn, 0.028 flop, 0.000099
preflop auxiliary** (p.4).

### Comparability caveat — read before using these as a bar

The paper does not state the space in which its Huber loss is averaged. This
project's `masked_card_huber_loss` averages over **legal card-space hands after
inverse bucketing**, deliberately, so that no unpublished card-target →
bucket-target reduction has to be invented. A bucket-space average over 1,000
buckets and a card-space average over ~1,081 legal hands are not the same
quantity, and the paper's `beta`/delta for the Huber transition is also
unstated.

So: treat 0.010 / 0.015 as the **order of magnitude** the river network should
reach, not as a number to match. Any claim of "we hit the paper's loss" needs
the averaging convention stated alongside it.

## Training set sizes (p.7)

| network | Supremus | DeepStack |
|---|---|---|
| River | 50,000,000 subgames | — |
| Turn | 20,000,000 | 10,000,000 |
| Flop | 5,000,000 | 1,000,000 |
| Preflop auxiliary | 10,000,000 situations | 10,000,000 |

Each subgame solved with 4,000 iterations per player of DCFR+, random subgames
generated "in a manner identical to DeepStack".

Scale check for this repository: the current CPU river solve runs ~33.6 s per
4,000-iteration subgame. 50M subgames is therefore ~53,000 core-years on this
backend. The paper's implementation is end-to-end CUDA. **Matching the paper's
data volume is not reachable here**; the targets above are what a smaller
run should be measured against, with the sample count always reported next to
the loss.

## Table 2 — action abstraction (p.7)

| | first action | second | third | remaining |
|---|---|---|---|---|
| DeepStack | F, C, 0.5, 1.0, 2.0, A | F, C, 0.5, 1.0, 2.0, A | F, C, 1.0, A | F, C, 1.0, A |
| Supremus | F, C, 0.33, 0.5, 0.75, 1.0, 1.25, 2.0, A | F, C, 0.25, 0.5, 1.0, A | F, C, 0.25, A | F, C, 1.0, A |

**Verified against `hunl/supremus_config.py`: all four depths agree exactly**,
and depths beyond the third clamp to the "remaining" menu. The one thing Table 2
does not fix is how a non-integral pot fraction becomes an integer chip amount;
that stays fail-closed under `AUTHOR_STRICT`.

## Head-to-head and exploitability anchors

| matchup | result |
|---|---|
| Supremus vs Slumbot, 150,000 hands | **+176 ± 44 mbb/g** (2,637,277 chips) |
| DeepStack reimplementation vs Slumbot | **−63 ± 40 mbb/g** (−948,096 chips) |
| net effect of the Supremus improvements | +239 mbb/g |
| DeepStack (original) vs LBR, fold/call only | +428 ± 87 mbb/g |
| DeepStack reimplementation vs LBR | +536 ± 68 mbb/g |

The reimplementation *losing* to Slumbot is the paper's central motivating
result: a faithful DeepStack is not competitive with the 2018 benchmark, and the
gap is closed by the changes this repository reconstructs.

## What the paper anchors that was previously unstated here

- Networks trained with **Adam** and **Huber loss** as the evaluation metric
  (p.4). This repository's `train_smoke_steps` (Adam) and
  `masked_card_huber_loss` are therefore source-anchored in *kind*; the learning
  rate, schedule, batch size, epoch count and initialization remain unpublished
  and are project choices.
- The paper's DeepStack reimplementation **stops at the start of the river and
  substitutes the Supremus river network**, precisely because the original
  play-time river bucketing was never published (p.4). This is the same wall
  this project documents, reached independently by the paper's authors.
