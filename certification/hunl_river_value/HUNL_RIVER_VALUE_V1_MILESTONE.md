# HUNL River 1000-Bucket Projection + CFVnet — Milestone V1

## Verdict

This milestone adds a **reproducible replacement river bucketing artifact** and
connects real 4,000-iteration full-card river CFV targets to the published
2001→7×500 PReLU→2000 value-network contract.

It does **not** claim the private Supremus river bucket map, private Supremus
weights, or bit identity with the 2020 implementation.

## Primary-source constraints

Zarick et al. (2020) state that:

- both DeepStack and Supremus reduce 1326 hand probabilities/values by bucketing;
- network input is 1000 bucket probabilities/player plus one pot feature = 2001;
- network output is 1000 expected values/player = 2000;
- architecture is seven fully-connected hidden layers × 500 plus the external
  zero-sum correction;
- Supremus trains a river network first;
- river training uses 50 million random subgames;
- each random subgame is solved with 4,000 iterations/player of DCFR+;
- outputs are represented as a fraction of current pot size and the input pot
  feature is current pot as a fraction of starting chips.

The paper does **not** publish the actual 1000 river clusters or a generator for
them.

## Reconstruction bucket artifact

`hunl/river_bucket_reconstruction.py`

For a fixed five-card river board and legal hero hand, compute exact terminal
hand strength against a uniformly random compatible opponent hand:

    equity = wins + 0.5*ties

There are exactly C(45,2)=990 opponent hands.  The feature is stored exactly as
integer numerator `2*wins + ties` / `1980` before conversion to float.

A PROJECT reconstruction then fits 1000 scalar k-means centroids over these
features.  This is an engineering replacement because the private Supremus
river feature/centroid artifact is unpublished.

Frozen artifact:

- samples: 131,072 exact board+hand features;
- sampling: 256 random raw river boards × 512 legal hands/board;
- k=1000;
- k-means++ initialization;
- 10 Lloyd iterations;
- centroid order sorted by increasing equity (PROJECT canonical);
- 1000/1000 unique centroids;
- centroid SHA-256: `ea386b8e3b2bd4861b5e59711356e4a94b49d09e1c942b5673c0eeebcc61bbbf`;
- artifact SHA-256: `50f994efd17d191cf8b4434b66dc27892e7131b718621a389aca575ccbf2d617`;
- byte-for-byte deterministic rebuild: PASS.

## Projection algebra

For every board:

- blocked hands map to -1;
- legal hands map to one of 1000 bucket ids;
- card range → bucket range by summation;
- bucket values → card values by inverse scatter;
- range mass is conserved;
- dot-product duality between projected ranges and inverse values is conserved.

Certified representative-board errors:

- range-mass error: 0;
- range/value duality error: 2.78e-17;
- 64 global suit-permutation correspondence checks: PASS.

On the three real 4000-iteration anchors, one board uses only a subset of the
global bucket set (95, 91, and 28 buckets respectively).  This is recorded as a
property of this scalar-strength reconstruction, not evidence about the private
Supremus bucketizer.

## River training adapter

`prepare_river_training_batch()` was added without changing the existing turn
training API.

River branch convention follows the literal Supremus paper wording:

    pot_feature = total_current_pot / 20000
    target = raw_CFVs_in_chips / total_current_pot

Raw full-card targets remain the primary saved artifact.  Bucketing can be
replaced later without rerunning the expensive 4000-iteration solves.

The training graph is:

    1326 card ranges
      -> 1000 bucket distributions/player
      -> [2001] network input
      -> 7×500 PReLU
      -> [2000] bucket CFVs
      -> zero-sum correction
      -> inverse bucket scatter
      -> [2,1326] card CFVs
      -> masked Huber against raw full-card targets

This intentionally avoids inventing an unpublished card-target→bucket-target
reduction rule.

## Real 4000-iteration mini-dataset

Three independent full-card river subgames are frozen:

1. seed 123, board `(6,1,46,50,0)`, pot_half 301;
2. seed 124, board `(22,12,17,3,34)`, pot_half 1829;
3. seed 126, board `(30,27,50,25,47)`, pot_half 1932.

All use 4000 iterations of the current Supremus-paper reconstruction DCFR+
solver.  Raw targets remain full 1326-hand CFVs.

Combined raw miniset SHA-256:
`66ebd8a2dcb964bd64ed33748ba2d30d781736ff37e6f278c36f00d54590bc34`.

Seed 125 was a runtime outlier in this CPU sandbox and was not included; no
mathematical parameter was changed to force it through.

## Network training smoke

A full 3,506,007-parameter network was trained only as an engineering smoke on
the three frozen real subgames.

- 120 Adam steps;
- initial card-space Huber: 0.26161718;
- final card-space Huber: 0.00056989;
- final per-sample losses: 0.00102052, 0.00036454, 0.00032461;
- weighted zero-sum residuals: all < 1.1e-7 in normalized units.

This very low loss is **not a model-quality benchmark**: it is deliberate tiny-
dataset overfitting to prove the full training graph is differentiable and wired
correctly.  Supremus quality claims require orders of magnitude more data.

Checkpoint `HUNL_RIVER_CFVNET_3SAMPLE_SMOKE.pt` is therefore explicitly tagged
`ENGINEERING_SMOKE_NOT_PRODUCTION_MODEL`.

## Regression retained

After adding this milestone:

- river datagen certification: PASS;
- 25-iteration river target deterministic hash remains green;
- LP anchor remains within 0.121% total pot at 500 iterations;
- original HUNL value-network architecture contract: PASS;
- original turn DataGenerator V2 wiring/replay: PASS.

## Remaining blockers

For an original/bit-exact Supremus river network:

1. private 1000-bucket river artifact or generator;
2. private integer-chip rounding for decimal Table-2 bet sizes;
3. private RNG / seed / dataset ordering;
4. confirmation of the private river-training update mode;
5. original weights / optimizer state / initialization details.

For a strong reconstruction (not original-bit-exact), the next practical step
is to scale raw river generation and train a real river network, then use that
network as the turn depth boundary.  The current three-sample checkpoint is not
suitable for game play.
