#!/usr/bin/env python3
"""Measure the two remaining invented choices in the generator.

After the tie-break was measured and downgraded, two PROJECT_CANONICAL
decisions were left unquantified. Both are invented, both are marked as such in
the code, and neither had a number attached.

**Chip rounding.** Table 2 gives bet sizes as fractions of the pot, which are
usually not whole chips. The paper never states the quantization. This project
uses `NEAREST_HALF_UP`; `FLOOR` is the obvious alternative. It changes the
betting tree, so inputs are untouched and only the solved game differs — the
comparison is exact per sample.

**The `[100,100)` pot category.** The published first interval is empty under
standard interval notation, and this project reads it as the point mass {100},
marked PROJECT_CANONICAL because no author correction was recovered. The
obvious alternative is that `[100,100)` is a typo for `[100,200)`, which would
also make the five printed intervals a contiguous partition.

That alternative turns out to be impossible, and this harness proves it rather
than measuring it. In HUNL with blinds (50,100) a *symmetric* pot — both
players committed equally, which is what `pot_half` denotes — can only be 100,
when both have merely posted the big blind, or at least 200, because the
smallest legal raise goes from 100 to 200. Nothing between is reachable. The
certified tree builder, whose betting rules mirror ACPC `game.c`, rejects
exactly those pots.

So the gap between 100 and 200 is not a hole in the printed partition; it is a
hole in the game. `[100,100)` is an awkward notation for a set that genuinely
contains one element, and the project's reading is the only admissible one.

Run from the repository root:
    python -m certification.hunl_river_value.run_generator_fork_sensitivity --samples 12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hunl.supremus_config import ChipQuantization
from hunl_datagen.river_datagen_v1 import (
    RiverDataGeneratorV1,
    RiverDatagenMode,
    RiverDatagenV1Config,
    shard_seed,
)
from hunl_datagen.turn_datagen_v2 import CountingTHRandom

HERE = Path(__file__).resolve().parent
MASTER_SEED = 20260816


def run(index: int, iterations: int, *, rounding=ChipQuantization.NEAREST_HALF_UP,
        bin0: str = "POINT_MASS_100"):
    gen = RiverDataGeneratorV1(RiverDatagenV1Config(
        mode=RiverDatagenMode.PAPER_RECONSTRUCTION, batch_size=1,
        dcfr_iterations=iterations, solver_backend="numba_flat",
        chip_quantization=rounding, pot_bin0=bin0,
    ))
    rng = CountingTHRandom(shard_seed(MASTER_SEED, index))
    inputs = gen.make_batch_inputs(rng)
    solved = gen.solve_batch(inputs)
    return {
        "board": tuple(int(c) for c in inputs.board),
        "pot_half": int(inputs.pot_half[0]),
        "ranges": inputs.ranges[:, 0, :].astype(np.float64),
        "targets": solved.targets_chips[0].astype(np.float64),
        "decision_nodes": int(solved.decision_nodes[0]),
        "terminal_nodes": int(solved.terminal_nodes[0]),
        "rng_draws": int(inputs.rng_draws_after),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--iterations", type=int, default=4000)
    args = ap.parse_args()

    started = time.perf_counter()
    rounding_rows = []

    for i in range(args.samples):
        base = run(i, args.iterations)

        # --- chip rounding: identical inputs, different betting tree ---
        floor = run(i, args.iterations, rounding=ChipQuantization.FLOOR)
        assert floor["pot_half"] == base["pot_half"] and floor["board"] == base["board"]
        total = 2.0 * base["pot_half"]
        live = base["ranges"] > 0
        d = np.abs(base["targets"] - floor["targets"]) / total
        ev = [float(np.dot(r["ranges"][0], r["targets"][0])) / total for r in (base, floor)]
        rounding_rows.append({
            "sample": i, "pot_half": base["pot_half"],
            "tree_nodes_baseline": [base["decision_nodes"], base["terminal_nodes"]],
            "tree_nodes_floor": [floor["decision_nodes"], floor["terminal_nodes"]],
            "tree_identical": (base["decision_nodes"] == floor["decision_nodes"]
                               and base["terminal_nodes"] == floor["terminal_nodes"]),
            "target_abs_delta_mean_pct_of_pot": 100.0 * float(d[live].mean()),
            "target_abs_delta_max_pct_of_pot": 100.0 * float(d[live].max()),
            "subgame_ev_abs_delta_pct_of_pot": 100.0 * abs(ev[0] - ev[1]),
        })

        print(json.dumps({"sample": i, "rounding": rounding_rows[-1]}), flush=True)

    # --- pot category 0: which integers are reachable at all? ---
    from hunl.supremus_config import SupremusRiverConfig
    from hunl.tree import RiverTreeBuilder
    cfg = SupremusRiverConfig()
    reachable = []
    for pot_half in range(100, 200):
        try:
            RiverTreeBuilder(cfg).build(pot_half)
            reachable.append(pot_half)
        except AssertionError:
            pass
    potbin_rows = {
        "candidate_reading": "[100,200)",
        "integers_tested": list(range(100, 200)),
        "reachable_pot_halves": reachable,
        "big_blind": cfg.big_blind,
        "resolved": reachable == [100],
    }

    def stat(rows, key):
        v = [r[key] for r in rows if key in r]
        return {"min": min(v), "mean": float(np.mean(v)), "max": max(v)} if v else None

    rounding_ev = stat(rounding_rows, "subgame_ev_abs_delta_pct_of_pot")

    result = {
        "schema": "HUNL_RIVER_GENERATOR_FORK_SENSITIVITY_V1",
        "samples": args.samples,
        "dcfr_iterations": args.iterations,
        "reference_lp_anchor_gap_pct_of_pot": 0.121,
        "chip_rounding": {
            "baseline": "NEAREST_HALF_UP", "variant": "FLOOR",
            "trees_identical_on_all_samples": all(r["tree_identical"] for r in rounding_rows),
            "subgame_ev_abs_delta_pct_of_pot": rounding_ev,
            "target_abs_delta_mean_pct_of_pot": stat(rounding_rows, "target_abs_delta_mean_pct_of_pot"),
            "target_abs_delta_max_pct_of_pot": stat(rounding_rows, "target_abs_delta_max_pct_of_pot"),
            "verdict": ("NEGLIGIBLE" if rounding_ev and rounding_ev["max"] < 0.05
                        else "MATERIAL" if rounding_ev and rounding_ev["max"] >= 0.5
                        else "NON-TRIVIAL"),
            "measurements": rounding_rows,
        },
        "pot_category_zero": {
            "baseline": "POINT_MASS_100",
            "rejected_alternative": "CONTIGUOUS_100_200",
            "verdict": ("RESOLVED BY GAME RULES — the point mass is the only "
                        "admissible reading" if potbin_rows["resolved"] else
                        "UNEXPECTED: more than one pot half is reachable below 200"),
            "reason": "a symmetric pot above the big blind requires a raise, and "
                      "the smallest legal raise goes from 100 to 200, so no "
                      "pot_half between 101 and 199 exists in HUNL. The gap "
                      "between the first two printed intervals is a gap in the "
                      "game, not a hole in the partition.",
            "authority": "RiverTreeBuilder betting rules, mirroring ACPC game.c",
            **potbin_rows,
        },
        "seconds": time.perf_counter() - started,
    }
    out = HERE / "HUNL_RIVER_GENERATOR_FORK_SENSITIVITY_V1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("\nchip rounding, NEAREST_HALF_UP vs FLOOR:")
    print(f"  subgame EV delta % of pot: {rounding_ev}")
    print(f"  verdict: {result['chip_rounding']['verdict']}")
    print(f"\npot category 0: reachable pot halves in [100,200) = "
          f"{potbin_rows['reachable_pot_halves']}")
    print(f"  verdict: {result['pot_category_zero']['verdict']}")
    print(f"\nLP anchor gap for comparison: 0.121%")
    print(out)


if __name__ == "__main__":
    main()
