# HUNL River DataGenerator V1 — Supremus paper reconstruction

Status: **GREEN as a source-constrained reconstruction; NOT bit-identical Supremus private code.**

## Primary paper constraints

From Zarick et al. (2020):
- River CFV network is trained first.
- 50,000,000 river subgames.
- Random subgames generated in a manner identical to DeepStack.
- Each generated subgame solved with 4,000 iterations per player of DCFR+.
- DCFR+ uses delayed linear average weight `max(0, t-100)`.
- Paper architecture uses 1,000 buckets per player and value outputs as fractions of pot size.

## Implemented full-card target layer

The expensive solver target is deliberately stored before bucketing:
- 5-card public river board
- two 1326-hand ranges
- integer chip pot state
- full-card sparse river betting tree
- exact fold/showdown terminal kernels
- DCFR+ reconstruction
- raw chip CFVs `[2,1326]`

Derived normalized views are provided for both `pot_half` and total-pot conventions.

## Certification

`HUNL_RIVER_DATAGEN_V1_CERT.json`:
- legal private hands on a river board: 1081
- compatible opponent hands per legal private hand: 990
- paper-private non-integral action rounding: fail-closed in AUTHOR_STRICT
- deterministic input-generation seed anchor
- deterministic 25-iteration target hash unchanged after numerical hardening
- zero-sum residual near machine precision
- independent restricted-support sequence-form LP anchor: <0.5% total-pot gap after 500 iterations

## 4,000-iteration production-shape random anchor

Artifact: `HUNL_RIVER_RANDOM_4000_ANCHOR.npz`

- seed: 123
- board: `(6,1,46,50,0)`
- pot_half: 301 (total pot 602)
- decision nodes: 104
- terminal nodes: 205
- iterations: 4000
- update mode: `SIMULTANEOUS_RECONSTRUCTION`
- zero-sum residual: `-1.4210854715202004e-14` chips
- target SHA-256: `f6301214462c3c41939164df77fc9c86b993911af33555b0b3a69928af2add4e`
- NPZ SHA-256: `f853aafa3d59d7b4db80acfb5a9a742204590cad2ce8f752e95b4af0fdfd6f89`
- observed completed solve time: ~33.6 s in this sandbox
- the target hash was reproduced in separate completed runs.

## Performance backend

A flattened Numba backend is used for 4,000-iteration generation.  It was regression-checked against the Python oracle on shorter iteration counts.  The river terminal kernels remain the exact full-card blocker/showdown implementation.

A numerical hardening seam maps float64 *subnormal* cumulative regrets (`0 < |r| < np.finfo(float64).tiny`) to exact zero. This prevents severe CPU slowdowns after thousands of beta=0 discount operations. It is explicitly an engineering seam, not a claim about Supremus CUDA floating-point mode.

## Unresolved private details

The following remain explicitly unresolved and therefore prevent a bit-identical-Supremus claim:
1. exact integer-chip rounding of decimal Table-2 action fractions;
2. private RNG / seed schedule;
3. equal-strength tie ordering inside `R(S,p)`;
4. intended private handling of DeepStack's printed `[100,100)` pot interval;
5. original river 1000-bucket mapping/centroids;
6. whether the private river-training DCFR+ solve used simultaneous updates (the paper establishes the 4000-iteration DCFR+ setup and reports simultaneous-update benefits in CFVnet experiments, but does not expose the private implementation).

## Next milestone

Do **not** regenerate expensive full-card targets for each bucketing experiment. Keep raw chip CFVs immutable, then build a separate, explicitly reconstructed river 1000-bucket projection and CFVnet training layer. The original private bucket artifact remains a missing provenance item.
