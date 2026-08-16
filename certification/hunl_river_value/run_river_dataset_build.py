#!/usr/bin/env python3
"""Build a real river training set, shard by shard, resumably.

The V1 milestone froze three 4,000-iteration subgames — enough to prove the
training graph is wired correctly, far too few to say anything about model
quality. This builds the dataset that makes the quality question answerable.

Design constraints that matter:

- **Deterministic.** Sample i everywhere derives from `shard_seed(master, i)`,
  so any shard can be rebuilt or audited in isolation and two machines
  generating disjoint ranges produce the same corpus as one machine doing all
  of it.
- **Resumable.** Each shard is written to a temporary file and renamed only
  once complete, so an interrupted run never leaves a half shard that a later
  run would trust. Completed shards are skipped, never recomputed.
- **Paper-faithful.** 4,000 DCFR+ iterations per subgame, PAPER_RECONSTRUCTION
  mode, `NEAREST_HALF_UP` chip quantization. No parameter is lowered to make
  the run finish faster; if a smaller corpus is all that fits, it is smaller,
  not cheaper per sample.

Raw chip CFVs are stored, never normalized targets, so a change of bucketing or
of the target convention never forces a re-solve.

Usage:
    python -m certification.hunl_river_value.run_river_dataset_build \
        --shards 0-39 --samples-per-shard 25 --out datagen_out/river_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from hunl.cards import HAND_COUNT
from hunl_datagen.river_datagen_v1 import (
    RiverDataGeneratorV1,
    RiverDatagenMode,
    RiverDatagenV1Config,
    shard_seed,
)
from hunl_datagen.turn_datagen_v2 import CountingTHRandom

MASTER_SEED = 20260816


def parse_shards(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def build_shard(gen: RiverDataGeneratorV1, shard: int, n: int, out_dir: Path) -> dict:
    final = out_dir / f"river_shard_{shard:05d}.npz"
    if final.exists():
        return {"shard": shard, "status": "skipped", "path": str(final)}

    boards = np.zeros((n, 5), dtype=np.int16)
    pots = np.zeros(n, dtype=np.int32)
    ranges = np.zeros((n, 2, HAND_COUNT), dtype=np.float32)
    targets = np.zeros((n, 2, HAND_COUNT), dtype=np.float32)
    residuals = np.zeros(n, dtype=np.float64)
    decisions = np.zeros(n, dtype=np.int32)
    terminals = np.zeros(n, dtype=np.int32)

    started = time.perf_counter()
    for i in range(n):
        index = shard * n + i
        rng = CountingTHRandom(shard_seed(MASTER_SEED, index))
        inputs = gen.make_batch_inputs(rng)
        solved = gen.solve_batch(inputs)
        boards[i] = np.asarray(inputs.board, dtype=np.int16)
        pots[i] = int(inputs.pot_half[0])
        ranges[i] = inputs.ranges[:, 0, :]
        targets[i] = solved.targets_chips[0]
        residuals[i] = float(solved.expected_utility_residuals[0])
        decisions[i] = int(solved.decision_nodes[0])
        terminals[i] = int(solved.terminal_nodes[0])
    seconds = time.perf_counter() - started

    # Must end in .npz: savez_compressed appends the suffix otherwise, and the
    # rename would then chase a filename that was never written.
    tmp = out_dir / f".river_shard_{shard:05d}.partial.npz"
    np.savez_compressed(
        tmp, boards=boards, pot_half=pots, ranges=ranges, targets_chips=targets,
        zero_sum_residuals=residuals, decision_nodes=decisions, terminal_nodes=terminals,
    )
    os.replace(tmp, final)

    return {
        "shard": shard,
        "status": "built",
        "samples": n,
        "seconds": seconds,
        "seconds_per_sample": seconds / n,
        "max_abs_zero_sum_residual_chips": float(np.abs(residuals).max()),
        "target_sha256": hashlib.sha256(targets.astype("<f4").tobytes()).hexdigest(),
        "path": str(final),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="e.g. 0-39 or 0,3,7")
    ap.add_argument("--samples-per-shard", type=int, default=25)
    ap.add_argument("--out", default="datagen_out/river_v1")
    ap.add_argument("--iterations", type=int, default=4000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = RiverDataGeneratorV1(RiverDatagenV1Config(
        mode=RiverDatagenMode.PAPER_RECONSTRUCTION,
        batch_size=1,
        dcfr_iterations=args.iterations,
        solver_backend="numba_flat",
    ))

    for shard in parse_shards(args.shards):
        row = build_shard(gen, shard, args.samples_per_shard, out_dir)
        print(json.dumps(row), flush=True)
        if row["status"] == "built":
            (out_dir / f"river_shard_{shard:05d}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
