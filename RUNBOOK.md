# Runbook — first real river training signal

Goal: replace "we can reproduce our artifacts" with "we know whether the
network generalizes". That needs one thing the project has never had — a
training set larger than three subgames.

Everything here is resumable and incremental. You do **not** have to decide the
final dataset size up front: generate some shards, run the curve, generate more,
run it again. Shards already built are never recomputed.

## 0. Setup

```
git clone https://github.com/zvezdan01/Supremus
cd Supremus
pip install numpy scipy numba torch
```

Verified on Python 3.11 with numpy 2.4.6, scipy 1.17.1, numba 0.67.0,
torch 2.13.0. The original snapshot was built under Python 3.13; both work.

Sanity check before spending hours — this rebuilds the frozen bucket artifact
and must print 21 PASS lines:

```
python -m certification.hunl_river_value.run_river_bucket_reproduction_cert
```

## 1. Generate subgames

One subgame = one 4,000-iteration DCFR+ river solve ≈ **30 s on one core**.
Each solve is single-threaded, so run one worker per core you want to spend and
leave one or two for the machine.

Shards are independent. Give each worker a disjoint stride so they never
collide:

```
# 6 workers, 10 subgames per shard, shards 0..599 => 6,000 subgames
for w in 0 1 2 3 4 5; do
  python -m certification.hunl_river_value.run_river_dataset_build \
    --shards $(python -c "print(','.join(str($w+6*i) for i in range(100)))") \
    --samples-per-shard 10 --out datagen_out/river_v1 > logs/gen_$w.log 2>&1 &
done
```

What that costs, at 30 s/subgame:

| subgames | 4 workers | 6 workers | 12 workers |
|---|---|---|---|
| 2,000 | 4.2 h | 2.8 h | 1.4 h |
| 6,000 | 12.5 h | 8.3 h | 4.2 h |
| 10,000 | 21 h | 14 h | 7 h |
| 100,000 | 8.7 d | 5.8 d | 2.9 d |

**Start with 2,000.** That is enough to see the curve's shape, and it is one
evening. Extend later by running higher shard numbers — nothing is redone.

Notes:

- The first solve in each worker includes a one-off numba compile (~10 s).
- Interrupting is safe. A shard is written to a `.partial.npz` and renamed only
  when complete, so a killed run never leaves a half shard that a later run
  would trust. Re-running the same command resumes.
- Storage is ~21 KB per subgame before compression: 100k ≈ 2 GB.
- On a laptop, watch thermals. Sustained all-core load will throttle, which
  stretches the estimates above.

## 2. Train and get the curve

```
python -m certification.hunl_river_value.run_river_training_curve \
  --data datagen_out/river_v1 --val-fraction 0.2 --epochs 60
```

It holds out 20% as validation, then trains from scratch at 100, 250, 500,
1000, 2500, ... training samples and reports the best validation loss at each
size. Output goes to `HUNL_RIVER_TRAINING_CURVE_V1.json`.

Before training it checks its own fast path against the project's reference
`loss_on_prepared_batch` and aborts if they disagree by more than 1e-6, so the
curve cannot silently be measuring something other than the certified loss.

Two flags exist for the forks that `certification/RELEASED_CODE_EVIDENCE.md`
opened, both defaulting to the current behaviour:

- `--pot-convention TOTAL_POT|POT_HALF` — the Supremus paper's literal reading
  versus released DeepStack-Leduc. Changes every loss by roughly a factor of
  three, so never compare across it.
- `--optimizer adam|adamw` — Adam at 1e-3 is the released setting and the
  default; AdamW at 3e-4 is the DEVN third-party choice.

Resource notes for larger runs. Bucketing is a Python loop with an exact
1326×1326 equity computation per subgame, about 25 ms each: **~40 minutes for
100k subgames**, once, before training starts. Held tensors are roughly 30 KB
per subgame, so 100k needs **~3 GB of RAM**. Evaluation is chunked, so the
split size does not add a second peak.

## 3. How to read it

Two reference points, both already measured in this repository:

| marker | value | meaning |
|---|---|---|
| encoding floor | **4.34e-04** | best possible given the 1000-bucket abstraction |
| paper's river validation | **1.5e-02** | Supremus on 50M subgames, averaging space unstated |

The loss here is card-space masked Huber after inverse bucketing — this
project's convention. The paper does not state its own, so treat 1.5e-02 as an
order of magnitude, not a target to hit.

What the shapes mean:

- **Validation falling steadily as samples grow** — working as intended. Keep
  generating; the curve tells you where the returns flatten.
- **Validation flat while training loss falls** — overfitting; the dataset is
  the constraint, not the model. Generate more before touching architecture.
- **Both flat and well above 4.34e-04** — an optimization problem, not a data
  one. Learning rate and schedule are project choices (the paper publishes only
  Adam + Huber), so they are the first things to vary.
- **Validation approaching 4.34e-04** — the bucketing has become the limit.
  That is the point at which a better river abstraction, not more data, is the
  next move.

## 4. What not to conclude

A loss figure without its sample count is meaningless here. 100k subgames is
0.2% of the paper's river training set; the curve tells you the trend and where
the effort should go next, not that the reconstruction matches Supremus.

Nothing in this run touches the unresolved blockers — private bucket artifact,
integer-chip rounding, private RNG and ordering, river-training update mode,
original weights. Those stay closed regardless of how the curve looks.
