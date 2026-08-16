#!/usr/bin/env python3
"""Widen the independent LP anchor — the engine's only external cross-check.

Every other certificate in this project is internal: determinism, byte
reproducibility, zero-sum residuals, self-consistency. Exactly one check
compares the DCFR+ solver against something computed a completely different
way — a sequence-form linear program, no CFR code shared.

That check was run at a support of **6 hands per player**, which is a thin
basis for trusting a solver used on 1081-hand games. This harness widens it and
measures two axes separately:

- **support** — does the agreement survive richer restricted games? The DCFR
  side always solves the full 1326-hand game with ranges zeroed outside the
  support, so growing the support grows the part of the real engine under test.
- **iterations** — does the gap shrink as DCFR+ converges, as it must if the
  gap is convergence error rather than a discrepancy in the game itself?

Several random supports per size, so a favourable draw cannot flatter the
result.

Run from the repository root:
    python -m certification.hunl_river_datagen.run_lp_anchor_grid
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from certification.hunl_river_datagen.lp_supremus_river import solve_lp
from hunl.cards import HAND_COUNT, possible_hands_mask
from hunl.river_dcfr_plus import DcfrPlusSpec, RiverDcfrPlusEngine
from hunl.supremus_config import SupremusRiverConfig

HERE = Path(__file__).resolve().parent
BOARD = (0, 5, 10, 15, 20)   # same anchor board as the V1 certificate
POT_HALF = 100
TOTAL_POT = 2 * POT_HALF


def draw_case(support: int, seed: int):
    live = np.flatnonzero(possible_hands_mask(BOARD))
    rng = np.random.default_rng(seed)
    s0 = sorted(rng.choice(live, support, replace=False).tolist())
    s1 = sorted(rng.choice(live, support, replace=False).tolist())
    r0 = np.zeros(HAND_COUNT)
    r1 = np.zeros(HAND_COUNT)
    r0[s0] = rng.random(support) + 0.1
    r1[s1] = rng.random(support) + 0.1
    r0 /= r0.sum()
    r1 /= r1.sum()
    return s0, s1, r0, r1


def measure(support: int, seed: int, iterations: int, cfg, cache: dict) -> dict:
    s0, s1, r0, r1 = draw_case(support, seed)

    key = (support, seed)
    if key not in cache:
        t = time.perf_counter()
        cache[key] = (solve_lp(BOARD, POT_HALF, s0, s1, r0, r1, cfg),
                      time.perf_counter() - t)
    lp_value, lp_seconds = cache[key]

    t = time.perf_counter()
    solved = RiverDcfrPlusEngine(
        BOARD, POT_HALF, cfg, DcfrPlusSpec(iterations=iterations),
    ).solve_numba_flat(r0, r1)
    dcfr_seconds = time.perf_counter() - t

    dcfr_value = float(r0 @ solved.root_cfvs[0])
    gap = abs(dcfr_value - lp_value)
    return {
        "support_each_player": support,
        "seed": seed,
        "iterations": iterations,
        "lp_value_chips": lp_value,
        "dcfr_value_chips": dcfr_value,
        "abs_gap_chips": gap,
        "gap_pct_of_pot": 100.0 * gap / TOTAL_POT,
        "lp_seconds": lp_seconds,
        "dcfr_seconds": dcfr_seconds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supports", default="6,12,24,48,64")
    ap.add_argument("--seeds", default="7,11,13")
    ap.add_argument("--support-axis-iterations", type=int, default=1000)
    ap.add_argument("--convergence-iterations", default="500,1000,2000,4000")
    args = ap.parse_args()

    supports = [int(s) for s in args.supports.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    conv_iters = [int(s) for s in args.convergence_iterations.split(",")]
    cfg = SupremusRiverConfig()
    cache: dict = {}
    started = time.perf_counter()

    print("axis A — support, at "
          f"{args.support_axis_iterations} iterations", flush=True)
    support_axis = []
    for support in supports:
        for seed in seeds:
            row = measure(support, seed, args.support_axis_iterations, cfg, cache)
            support_axis.append(row)
            print(f"  support {support:3d} seed {seed:3d}: "
                  f"gap {row['gap_pct_of_pot']:8.4f}% of pot   "
                  f"LP {row['lp_seconds']:6.2f}s  DCFR {row['dcfr_seconds']:6.2f}s",
                  flush=True)

    widest = max(supports)
    print(f"\naxis B — convergence, at support {widest}", flush=True)
    convergence_axis = []
    for iterations in conv_iters:
        for seed in seeds:
            row = measure(widest, seed, iterations, cfg, cache)
            convergence_axis.append(row)
            print(f"  {iterations:5d} iters seed {seed:3d}: "
                  f"gap {row['gap_pct_of_pot']:8.4f}% of pot", flush=True)

    def worst(rows):
        return max(r["gap_pct_of_pot"] for r in rows)

    by_support = {s: worst([r for r in support_axis if r["support_each_player"] == s])
                  for s in supports}
    by_iterations = {i: worst([r for r in convergence_axis if r["iterations"] == i])
                     for i in conv_iters}

    converges = all(
        by_iterations[b] <= by_iterations[a] * 1.5
        for a, b in zip(conv_iters, conv_iters[1:])
    )
    result = {
        "schema": "HUNL_RIVER_LP_ANCHOR_GRID_V1",
        "claim": "the DCFR+ solver agrees with an independent sequence-form LP "
                 "across a range of restricted-support games, and the residual "
                 "gap behaves as convergence error",
        "board": list(BOARD),
        "pot_half": POT_HALF,
        "previous_certificate": {"support_each_player": 6, "iterations": 500,
                                 "gap_pct_of_pot": 0.121},
        "widest_support": widest,
        "worst_gap_pct_of_pot_by_support": by_support,
        "worst_gap_pct_of_pot_by_iterations": by_iterations,
        "gap_shrinks_with_iterations": converges,
        "support_axis": support_axis,
        "convergence_axis": convergence_axis,
        "seconds": time.perf_counter() - started,
        "note": "the DCFR side always solves the full 1326-hand game with "
                "ranges zeroed outside the support, so the engine under test "
                "is the production one; only the LP is restricted.",
        "backend": "solve_numba_flat, regression-checked against the Python "
                   "reference backend",
    }
    out = HERE / "HUNL_RIVER_LP_ANCHOR_GRID_V1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("\nworst gap, % of pot, by support:")
    for s in supports:
        print(f"  {s:3d} hands: {by_support[s]:8.4f}%")
    print("worst gap, % of pot, by iterations (widest support):")
    for i in conv_iters:
        print(f"  {i:5d}: {by_iterations[i]:8.4f}%")
    print(f"\ngap shrinks with iterations: {converges}")
    print(out)


if __name__ == "__main__":
    main()
