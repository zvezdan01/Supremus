# Supremus

Source-constrained HUNL reconstruction of the **Supremus / Deep CFV** line:

> Ryan Zarick, Bryan Pellegrino, Noam Brown, Caleb Banister,
> *Unlocking the Potential of Deep Counterfactual Value Networks*,
> arXiv:2007.10442 (2020). Vendored at `third_party/papers/`.

This is **not** original Supremus source code, and no claim of bit-identity with
the authors' private CUDA implementation is made anywhere in this repository.
Every reconstruction decision is either anchored in the paper or marked as
project-explicit and made to fail closed.

Split out of `zvezdan01/quant-trade` on 2026-08-16 so that Supremus' deliberate
algorithmic *changes* cannot leak into that repository's forensic DeepStack
baseline. See `PROVENANCE.md` for the extraction manifest and per-file SHA-256.

## What the paper fixes, and what it does not

Supremus reimplements DeepStack and then changes it. The changes this repository
targets:

| Supremus choice | source | status here |
|---|---|---|
| DCFR+ — DCFR regret discounting, average weight `max(0, t-100)` | paper | implemented, `hunl/river_dcfr_plus.py` |
| 4,000 iterations per player on generated subgames | paper | implemented, anchored |
| River value network, values at end of every round but the last | paper | **not yet built** |
| Larger action abstraction (Table 2) | paper | implemented, `hunl/supremus_config.py` |
| 50M river / 20M turn / 5M flop / 10M auxiliary training games | paper | datagen exists; scale is compute-bound |
| End-to-end CUDA implementation | paper | out of scope; Numba/CPU here |
| Subgames generated "in a manner identical to DeepStack" | paper | inherited from the DeepStack reconstruction |

Details the paper does **not** publish, which therefore stay fail-closed rather
than guessed:

1. integer-chip rounding of the decimal Table-2 pot fractions —
   `AUTHOR_STRICT` raises `UnpublishedSupremusDetail`; `NEAREST_HALF_UP` is the
   project-explicit default and is recorded in every manifest;
2. private RNG instance and seed schedule;
3. equal-strength tie ordering inside the range recursion `R(S,p)`;
4. intended handling of DeepStack's printed `[100,100)` pot interval;
5. the river 1000-bucket mapping and centroids;
6. whether the private river training solve used simultaneous updates.

## Layout

```
hunl/                          exact full-card HUNL primitives
  cards, evaluator, blockers,  frozen certified core — copied, read-only
  showdown, chance, tree,      (certified in quant-trade against untouched
  turn_tree, river_terminal*,   ACPC game.c, LP anchors, exhaustive recounts)
  river_resolver, turn_engine
  supremus_config.py           Table-2 action abstraction + chip quantization
  river_dcfr_plus.py           DCFR+ river solver, Numba backend
hunl_datagen/                  DeepStack-identical subgame generation
  river_datagen_v1.py          Supremus river training-sample generator
certification/hunl_river_datagen/
  run_river_datagen_cert.py    certification harness
  lp_supremus_river.py         independent sequence-form LP anchor
  HUNL_RIVER_*                 certificates and frozen anchors
third_party/papers/            primary source PDF + checksum
```

The `hunl/` core is deliberately **not** renamed or refactored: keeping the
import paths and file bytes identical is what makes the SHA-256 table in
`PROVENANCE.md` a verifiable link rather than a claim.

## Current certified state

From `certification/hunl_river_datagen/HUNL_RIVER_DATAGEN_V1_MILESTONE.md` —
**green as a source-constrained reconstruction, not bit-identical Supremus**:

- 1,081 legal private hands on a river board; 990 compatible opponent hands each;
- non-integral Table-2 rounding fails closed under `AUTHOR_STRICT`;
- deterministic 25-iteration target hash stable across numerical hardening;
- zero-sum residual at machine precision;
- independent restricted-support sequence-form LP anchor within 0.5% of total
  pot after 500 iterations;
- 4,000-iteration production-shape anchor (`seed 123`, board `(6,1,46,50,0)`,
  pot_half 301, 104 decision / 205 terminal nodes), target SHA-256
  `f6301214…2add4e`, reproduced across separate completed runs.

Raw chip CFVs `[2,1326]` are stored **before** bucketing, so a future bucketing
experiment never forces the expensive solves to be repeated.

## Next milestone

Build the river 1000-bucket projection and the CFVnet training layer as an
explicitly reconstructed component, keeping the raw full-card targets immutable.
The original private bucket artifact remains a missing provenance item and must
not be silently substituted.

## Running

Requires `numpy`, `scipy` and `numba`. From the repository root:

```
python -m certification.hunl_river_datagen.run_river_datagen_cert
```
