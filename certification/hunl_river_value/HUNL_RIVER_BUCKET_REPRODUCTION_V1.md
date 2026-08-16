# River bucket + CFVnet V1 — independent reproduction

Date: 2026-08-16 · Repository: `zvezdan01/Supremus`

## Why this document exists

The V1 river bucketing and CFVnet milestone was built in the project owner's
working environment. The snapshot that reached this repository carried the
**certification artifacts but not the source modules**: the archive contained
`FORENSICS/`, `certification/` and `third_party/` only. Three things were
missing:

| missing | recorded sha256 of the original |
|---|---|
| `hunl/river_bucket_reconstruction.py` | `abc73044c7f577c6547aa2082e71f12807d6102d32c5af3dae33a0aa854c05dc` |
| `hunl.value_training.prepare_river_training_batch` | (inside `value_training.py`, `77ed9ad40ca2c0075556cdf58ec90a170f89a715406afe8a52404157c367e3ca`) |
| `hunl.value_training.train_smoke_steps` | same file |

Both modules were therefore rewritten here from the milestone prose, the frozen
manifest, and the call signatures visible in
`build_and_cert_river_value_v1.py`. **The rewrites do not reproduce the
original source bytes** — those two sha256 values above cannot be matched and
are recorded only so the divergence is explicit.

What *is* reproduced is the output.

## Result

`run_river_bucket_reproduction_cert.py` — **PASS, 21/21 checks.**

| anchor | frozen | reproduced |
|---|---|---|
| `board_stream_sha256` | `c24c7005…8da7f` | identical |
| `feature_sha256` | `db4c97a5…8a1a44` | identical |
| `centroid_sha256` | `ea386b8e…61bbbf` | identical |
| artifact `.npz` sha256 | `50f994ef…f2d617` | identical, byte for byte |
| objective history (10 values) | — | bit-exact, all ten |
| range-mass error | 0 | 0 |
| range/value duality error | 2.7755575615628914e-17 | identical |
| suit-permutation checks | 64 | 64 PASS |
| bucket occupancy on the three 4000-iteration anchors | 95, 91, 28 | 95, 91, 28 |

The serialized artifact matching byte for byte — centroids *and* embedded
manifest — means the milestone documentation is complete enough to rebuild the
V1 bucket artifact without the original code.

## The numeric contract that decides it

The one detail not stated in prose, recovered by bisecting against the frozen
objective history:

> Features are quantized to **float32** (`numerator / 1980`), and the k-means
> fit then runs in **float64 over those float32 values**. Centroids are
> narrowed back to float32 only when serialized.

Fitting float32 throughout, or fitting float64 directly on `numerator/1980`,
both give a *different* artifact. The first objective value matches under any
of these (it is determined by the k-means++ initialization alone); the
divergence appears at the second, after the first Lloyd update. This is now
stated explicitly in the module docstring so it cannot be lost again.

## CFVnet adapter

The **unmodified** `build_and_cert_river_value_v1.py` from the owner's snapshot
was then run against the rewritten modules. It completed with `status: PASS`
and reproduced 26 of its 29 comparable fields exactly, including:

- `loss_initial` = `0.5254964828491211` — identical to all printed digits;
- `network_parameters` = 3,506,007;
- `pot_feature` = `0.03009999915957451`;
- `centroid_sha256`, `range_mass_error`, `range_value_duality_error`,
  `suit_assignment_checks`, `unique_centroids`, `centroid_min`/`max`.

`loss_initial` agreeing to all 16 digits is the load-bearing check for the
adapter: it is a forward pass of a 3.5M-parameter network over the prepared
input, so it pins the bucket ranges, the pot-feature convention and the
target normalization simultaneously. A different reading of "fraction of
current pot size" would move it.

The three fields that differ are training outcomes only:

| field | frozen | rerun |
|---|---|---|
| `loss_final` | 0.005225134082138538 | 0.00522513035684824 |
| `loss_min` | 0.004436210263520479 | 0.0044362107291817665 |
| `bucket_weighted_zero_sum_residual` | 1.1362135410308838e-07 | 6.332993507385254e-08 |

These agree to roughly seven significant digits and sit far inside the
harness's own tolerance (`5e-5` for the zero-sum residual). The residue is
float reduction order in backward/Adam across PyTorch builds and thread counts;
no mathematical parameter differs. The re-run output is kept alongside the
owner's frozen certificate under `independent_rerun/` rather than replacing it.

## Environment

`numpy 2.4.6`, `scipy 1.17.1`, `numba 0.67.0`, `torch 2.13.0`, Python 3.11.15.
The original snapshot was produced under Python 3.13.

## What this does not establish

Nothing about the private Supremus bucketizer. The 1000 river clusters used by
Zarick et al. are unpublished, and this artifact remains a
`PROJECT_RECONSTRUCTION_NOT_ORIGINAL` scalar-equity k-means replacement. The
five blockers listed in `HUNL_RIVER_VALUE_V1_MILESTONE.md` are untouched by
this work.

## Still outstanding from the v6 upload

- `HUNL_RIVER_CFVNET_3SAMPLE_SMOKE.pt` (expected sha256
  `e73f22aba39a7dbb5d42c63c8b102ac0a7c6892ca59f072d17f37ce6823244f7`) arrived
  truncated: 12,070,493 of 12,949,798 bytes. It is not committed. It is an
  `ENGINEERING_SMOKE_NOT_PRODUCTION_MODEL` checkpoint, so nothing depends on
  it, but the multiboard smoke figures in
  `HUNL_RIVER_CFVNET_MULTIBOARD_SMOKE_V1.json` cannot be re-derived from it
  here.
- `river_4000_seed123.npz` is absent from the snapshot; `seed124` and `seed126`
  are present, and the seed-123 subgame is available as
  `certification/hunl_river_datagen/HUNL_RIVER_RANDOM_4000_ANCHOR.npz`.
