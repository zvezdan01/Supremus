#!/usr/bin/env python3
"""Independent reproduction certificate for the V1 river bucket artifact.

The V1 artifact was built elsewhere by a source module that never reached this
repository (see PROVENANCE.md).  Only the artifact, its manifest and the
milestone prose survived.  `hunl/river_bucket_reconstruction.py` was therefore
written from that prose alone, and this harness is the check that the rewrite
is not merely plausible but produces the *same* artifact, byte for byte.

Four independent anchors, in increasing strength:

1. `board_stream_sha256` — the RNG board draws, which pins the sampling scheme
   including the interleaving of board and hand draws;
2. `feature_sha256`      — the exact equity numerators and their float32
   quantization;
3. `centroid_sha256` plus the bit-exact squared-equity objective history —
   the k-means++ initialization and every Lloyd step;
4. the serialized `.npz` — full byte identity including the manifest.

Passing all four means the milestone documentation is complete enough to
rebuild the artifact without the original code.  It says nothing about the
private Supremus bucketizer, which remains unpublished.

Run from the repository root:  python -m certification.hunl_river_value.run_river_bucket_reproduction_cert
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from hunl.cards import HAND_CARDS, hand_index, possible_hands_mask
from hunl.river_bucket_reconstruction import (
    ReconstructedRiverBucketProvider,
    fit_river_bucket_reconstruction_v1,
    river_uniform_equity_numerators,
    sample_river_strength_features,
)

HERE = Path(__file__).resolve().parent
SEED = 20260816

# Frozen V1 values, transcribed from HUNL_RIVER_BUCKET_RECONSTRUCTION_V1_BUILD.json
# and HUNL_RIVER_VALUE_V1_MILESTONE.md.
FROZEN_BOARD_STREAM = "c24c70050d49b1584a3a15e0218df8792c15810f11806c863741f81b2ce8da7f"
FROZEN_FEATURES = "db4c97a5445e613f5e709484c10cdeb236f64795e63609e9170acae09a8a1a44"
FROZEN_CENTROIDS = "ea386b8e3b2bd4861b5e59711356e4a94b49d09e1c942b5673c0eeebcc61bbbf"
FROZEN_ARTIFACT = "50f994efd17d191cf8b4434b66dc27892e7131b718621a389aca575ccbf2d617"
FROZEN_OBJECTIVES = [
    0.00588212020348377, 0.003909913771877626, 0.003749302071070597,
    0.0037465372978269665, 0.0037465372978269665, 0.0037465372978269665,
    0.0037465372978269665, 0.0037465372978269665, 0.0037465372978269665,
    0.0037465372978269665,
]
# Milestone claim: bucket occupancy on the three real 4000-iteration anchors.
FROZEN_ANCHOR_BUCKETS = {
    (6, 1, 46, 50, 0): 95,
    (22, 12, 17, 3, 34): 91,
    (30, 27, 50, 25, 47): 28,
}

failures: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    if not ok:
        failures.append(f"{name}: got {got!r}, want {want!r}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


print("1. sampling + features")
t0 = time.perf_counter()
x, meta = sample_river_strength_features(seed=SEED, boards=256, hands_per_board=512)
feature_seconds = time.perf_counter() - t0
check("sample count", int(x.size), 131072)
check("board_stream_sha256", meta["board_stream_sha256"], FROZEN_BOARD_STREAM)
check("feature_sha256", meta["feature_sha256"], FROZEN_FEATURES)

print("2. k-means fit")
t0 = time.perf_counter()
art = fit_river_bucket_reconstruction_v1(x, k=1000, seed=SEED, iterations=10)
fit_seconds = time.perf_counter() - t0
check("centroid_sha256", art.manifest["centroid_sha256"], FROZEN_CENTROIDS)
check("objective history (bit-exact)",
      art.manifest["objective_history_squared_equity"], FROZEN_OBJECTIVES)
check("unique centroids", int(np.unique(art.centroids).size), 1000)
check("empty clusters", art.manifest["empty_clusters_last_lloyd_step"], 0)

print("3. deterministic rebuild + serialized byte identity")
art2 = fit_river_bucket_reconstruction_v1(x, k=1000, seed=SEED, iterations=10)
check("in-memory rebuild identical", bool(np.array_equal(art.centroids, art2.centroids)), True)
art.manifest["training_sample"] = meta
tmp = HERE / "_river_bucket_reproduction_tmp.npz"
art.save(tmp)
rebuilt = tmp.read_bytes()
tmp.unlink()
check("artifact sha256", hashlib.sha256(rebuilt).hexdigest(), FROZEN_ARTIFACT)
frozen_path = HERE / "HUNL_RIVER_BUCKET_RECONSTRUCTION_V1.npz"
if frozen_path.is_file():
    check("byte identical to frozen artifact", rebuilt == frozen_path.read_bytes(), True)
else:
    print("  [SKIP] frozen artifact not present")

print("4. projection algebra on the real anchor board")
provider = ReconstructedRiverBucketProvider(art)
board = (6, 1, 46, 50, 0)
nums = river_uniform_equity_numerators(board)
legal = nums >= 0
check("legal hands", int(legal.sum()), 1081)
check("numerators in range",
      bool(nums[legal].min() >= 0 and nums[legal].max() <= 1980), True)
bm = provider.for_board(board)
check("legal mask agrees", bool(np.array_equal(bm.legal_mask, possible_hands_mask(board))), True)
check("provider deterministic",
      bool(np.array_equal(bm.hand_to_bucket, provider.for_board(board).hand_to_bucket)), True)

rng = np.random.default_rng(7)
r = rng.random(1326)
r[~legal] = 0
r /= r.sum()
br = bm.range_to_buckets(r)
v = rng.normal(size=1000)
hv = bm.bucket_values_to_hands(v)
mass_err = abs(float(br.sum()) - 1.0)
dual_err = abs(float(np.dot(br, v)) - float(np.dot(r, hv)))
check("range mass preserved", mass_err == 0.0, True)
check("range/value duality < 1e-12", dual_err < 1e-12, True)

print("5. global suit-permutation correspondence")
perm = (1, 0, 3, 2)


def permute_card(c: int) -> int:
    return (c // 4) * 4 + perm[c % 4]


pbm = provider.for_board(tuple(permute_card(c) for c in board))
suit_checks = 0
for h in np.flatnonzero(legal)[::17]:
    c0, c1 = (int(c) for c in HAND_CARDS[h])
    ph = hand_index(permute_card(c0), permute_card(c1))
    if int(bm.hand_to_bucket[h]) != int(pbm.hand_to_bucket[ph]):
        failures.append(f"suit bucket mismatch at hand {h}")
        break
    suit_checks += 1
check("suit-permutation checks", suit_checks, 64)

print("6. milestone bucket-occupancy claim on the three 4000-iteration anchors")
occupancy = {}
for anchor, expected in FROZEN_ANCHOR_BUCKETS.items():
    m = provider.for_board(anchor)
    used = int(np.unique(m.hand_to_bucket[m.legal_mask]).size)
    occupancy["-".join(str(c) for c in anchor)] = used
    check(f"buckets used on {anchor}", used, expected)

status = "PASS" if not failures else "FAIL"
cert = {
    "schema": "HUNL_RIVER_BUCKET_REPRODUCTION_V1_CERT",
    "status": status,
    "claim": "V1 river bucket artifact independently reproduced from the milestone "
             "specification alone, byte for byte",
    "not_a_claim": "says nothing about the private Supremus bucketizer, which "
                   "remains unpublished",
    "reproduced_board_stream_sha256": meta["board_stream_sha256"],
    "reproduced_feature_sha256": meta["feature_sha256"],
    "reproduced_centroid_sha256": art.manifest["centroid_sha256"],
    "reproduced_artifact_sha256": hashlib.sha256(rebuilt).hexdigest(),
    "objective_history_bit_exact": art.manifest["objective_history_squared_equity"] == FROZEN_OBJECTIVES,
    "range_mass_error": mass_err,
    "range_value_duality_error": dual_err,
    "suit_assignment_checks": suit_checks,
    "anchor_bucket_occupancy": occupancy,
    "feature_generation_seconds": feature_seconds,
    "bucket_fit_seconds": fit_seconds,
    "numeric_contract": "features quantized to float32, k-means fitted in float64 "
                        "over those values, centroids narrowed to float32 on save",
    "failures": failures,
}
out = HERE / "HUNL_RIVER_BUCKET_REPRODUCTION_V1_CERT.json"
out.write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n")
print()
print(json.dumps(cert, indent=2, sort_keys=True))
print(f"\n{status} {out}")
sys.exit(0 if status == "PASS" else 1)
