#!/usr/bin/env python3
"""How much does the unpublished equal-strength tie-break actually matter?

The range generator sorts hands by current-board strength and recursively
splits the sorted vector. On a river board only 27-81 distinct strengths exist
among 1081 legal hands, so **93-98% of split boundaries fall inside a class of
equal-strength hands**, and which of those hands lands in the upper half is
decided by a convention the HUNL supplement never states. This project uses
ascending hand id and labels it PROJECT_CANONICAL.

That convention is fixed, so its effect does not average out across samples: a
given hand systematically receives more mass than a strength-identical sibling
on every subgame from that board. If the effect is a hundredth of a percent of
the pot, the item can be closed. If it is percent-scale, it is the largest
single uncertainty in the generator and belongs at the top of the questions for
the authors.

The comparison is clean by construction. The recursion consumes one uniform
draw per internal node and the node count depends only on the number of legal
hands, which no reordering changes — so every variant sees an identical RNG
stream and identical boards and pots. The only thing that differs is which
hand sits on which side of a tie boundary.

Cost is one full 4,000-iteration solve per sample per variant. Nothing is
reused between variants, because the whole point is that the inputs differ.

Run from the repository root:
    python -m certification.hunl_river_value.run_tiebreak_sensitivity --samples 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hunl_datagen.river_datagen_v1 import (
    RiverDataGeneratorV1,
    RiverDatagenMode,
    RiverDatagenV1Config,
    shard_seed,
)
from hunl_datagen.turn_datagen_v2 import CountingTHRandom

HERE = Path(__file__).resolve().parent
MASTER_SEED = 20260816
BASELINE = "HAND_ID_ASC"


def generate_and_solve(tiebreak: str, index: int, iterations: int):
    gen = RiverDataGeneratorV1(RiverDatagenV1Config(
        mode=RiverDatagenMode.PAPER_RECONSTRUCTION,
        batch_size=1,
        dcfr_iterations=iterations,
        solver_backend="numba_flat",
        range_tiebreak=tiebreak,
    ))
    rng = CountingTHRandom(shard_seed(MASTER_SEED, index))
    inputs = gen.make_batch_inputs(rng)
    solved = gen.solve_batch(inputs)
    return {
        "board": tuple(int(c) for c in inputs.board),
        "pot_half": int(inputs.pot_half[0]),
        "ranges": inputs.ranges[:, 0, :].astype(np.float64),
        "targets": solved.targets_chips[0].astype(np.float64),
        "rng_draws": int(inputs.rng_draws_after),
        "boundary_ties": int(inputs.boundary_ties),
    }


def compare(base: dict, other: dict) -> dict:
    if base["board"] != other["board"] or base["pot_half"] != other["pot_half"]:
        raise AssertionError("variants diverged on board or pot — not a controlled comparison")
    if base["rng_draws"] != other["rng_draws"]:
        raise AssertionError("variants consumed different RNG streams")
    total_pot = 2.0 * base["pot_half"]

    # Total-variation distance between the two generated ranges, per player.
    tv = [0.5 * float(np.abs(base["ranges"][p] - other["ranges"][p]).sum()) for p in (0, 1)]

    # Target movement, in fractions of the total pot — the same unit as the
    # LP anchor gap, so the two numbers can be read side by side.
    d = np.abs(base["targets"] - other["targets"]) / total_pot
    live = base["ranges"] > 0
    per_hand = d[live]

    # Scalar summary: the expected value of the subgame to player 0.
    ev = [float(np.dot(r["ranges"][0], r["targets"][0])) / total_pot for r in (base, other)]

    return {
        "range_total_variation": tv,
        "target_abs_delta_max_pct_of_pot": 100.0 * float(per_hand.max()),
        "target_abs_delta_mean_pct_of_pot": 100.0 * float(per_hand.mean()),
        "subgame_ev_baseline_pct_of_pot": 100.0 * ev[0],
        "subgame_ev_variant_pct_of_pot": 100.0 * ev[1],
        "subgame_ev_abs_delta_pct_of_pot": 100.0 * abs(ev[0] - ev[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--variants", default="HAND_ID_DESC,SHUFFLE_1,SHUFFLE_2",
                    help="comma-separated; compared against HAND_ID_ASC")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rows = []
    started = time.perf_counter()

    for i in range(args.samples):
        base = generate_and_solve(BASELINE, i, args.iterations)
        for variant in variants:
            other = generate_and_solve(variant, i, args.iterations)
            row = {"sample": i, "variant": variant,
                   "board": list(base["board"]), "pot_half": base["pot_half"],
                   "boundary_ties": base["boundary_ties"], **compare(base, other)}
            rows.append(row)
            print(json.dumps(row), flush=True)

    def stat(key):
        v = [r[key] for r in rows]
        return {"min": min(v), "mean": float(np.mean(v)), "max": max(v)}

    ev_delta = stat("subgame_ev_abs_delta_pct_of_pot")
    verdict = ("NEGLIGIBLE — closes the item" if ev_delta["max"] < 0.05 else
               "MATERIAL — the tie-break is a leading uncertainty"
               if ev_delta["max"] >= 0.5 else
               "NON-TRIVIAL — comparable to solver convergence error")

    result = {
        "schema": "HUNL_RIVER_TIEBREAK_SENSITIVITY_V1",
        "question": "how much do generated targets move when the unpublished "
                    "equal-strength tie-break changes?",
        "verdict": verdict,
        "baseline": BASELINE,
        "variants": variants,
        "samples": args.samples,
        "dcfr_iterations": args.iterations,
        "seconds": time.perf_counter() - started,
        "subgame_ev_abs_delta_pct_of_pot": ev_delta,
        "target_abs_delta_max_pct_of_pot": stat("target_abs_delta_max_pct_of_pot"),
        "target_abs_delta_mean_pct_of_pot": stat("target_abs_delta_mean_pct_of_pot"),
        "reference_lp_anchor_gap_pct_of_pot": 0.121,
        "measurements": rows,
        "note": "read the EV delta against the 0.121% LP anchor gap: a "
                "tie-break effect below that is inside the project's own "
                "solver-validation tolerance, one above it is not.",
        "caveat": "this measures the effect on individual subgames. A fixed "
                  "convention biases the training distribution systematically "
                  "rather than randomly, so a small per-sample delta does not "
                  "by itself prove the trained network is unaffected.",
    }
    out = HERE / "HUNL_RIVER_TIEBREAK_SENSITIVITY_V1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nsubgame EV delta, % of pot: {ev_delta}")
    print(f"LP anchor gap for comparison: 0.121%")
    print(f"\nverdict: {verdict}\n{out}")


if __name__ == "__main__":
    main()
