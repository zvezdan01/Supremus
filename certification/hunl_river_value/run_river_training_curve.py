#!/usr/bin/env python3
"""Train the river CFVnet at increasing dataset sizes and plot the curve.

This is the project's first real feedback signal. Everything certified so far
measures reproducibility and internal consistency; nothing measures whether the
network generalizes. A validation-loss-versus-sample-count curve answers that,
and it answers it early: the shape is visible long before the paper's 50M
subgames are in hand.

Read the result against two reference points, both already measured:

- **encoding floor 4.34e-04** (`run_bucket_encoding_error.py`) — the best any
  network can do given the 1000-bucket abstraction. Approaching it means the
  bucketing, not the training, is now the limit.
- **paper river validation 0.015** (arXiv:2007.10442 Table 1) — reached on 50M
  subgames, and in an averaging space the paper never states, so it is an order
  of magnitude to aim at, not a number to match.

The loss here is this project's convention throughout: card-space masked Huber
after inverse bucketing, against raw solver CFVs normalized by total pot.

Usage:
    python -m certification.hunl_river_value.run_river_training_curve \
        --data datagen_out/river_v1 --val-fraction 0.2 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hunl.cards import HAND_COUNT
from hunl.river_bucket_reconstruction import (
    ReconstructedRiverBucketProvider,
    RiverReconstructionArtifact,
)
from hunl.value_bucketing import POSTFLOP_BUCKET_COUNT
from hunl.value_network import DeepStackHUNLValueNet, HUNLValueNetworkSpec
from hunl.value_training import loss_on_prepared_batch, prepare_river_training_batch

HERE = Path(__file__).resolve().parent
STACK = 20_000
SEED = 20260816


def load_shards(data_dir: Path):
    files = sorted(data_dir.glob("river_shard_*.npz"))
    if not files:
        raise SystemExit(f"no shards in {data_dir} — run run_river_dataset_build.py first")
    boards, pots, ranges, targets = [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        boards.append(z["boards"]); pots.append(z["pot_half"])
        ranges.append(z["ranges"]); targets.append(z["targets_chips"])
    return (np.concatenate(boards), np.concatenate(pots),
            np.concatenate(ranges), np.concatenate(targets), len(files))


def precompute(boards, pots, ranges, targets, provider, pot_convention="TOTAL_POT"):
    """Bucket every sample once; training then never touches numpy again.

    Equivalent to calling prepare_river_training_batch per row, but the
    bucket->card scatter is kept as an index tensor so the training loop can do
    it with a single gather instead of a Python loop over BoardBucketMap.
    """
    n = len(boards)
    inputs = torch.zeros(n, 2 * POSTFLOP_BUCKET_COUNT + 1)
    card_targets = torch.zeros(n, 2, HAND_COUNT)
    legal = torch.zeros(n, HAND_COUNT, dtype=torch.bool)
    bucket_ids = torch.zeros(n, HAND_COUNT, dtype=torch.int64)
    for i in range(n):
        batch = prepare_river_training_batch(
            boards=[boards[i]], ranges=ranges[i:i + 1], pot_halves=pots[i:i + 1],
            targets_chips=targets[i:i + 1], bucket_provider=provider, stack=STACK,
            pot_convention=pot_convention)
        inputs[i] = batch.inputs[0]
        card_targets[i] = batch.card_targets[0]
        legal[i] = batch.legal_mask[0]
        ids = batch.maps[0].hand_to_bucket.astype(np.int64)
        bucket_ids[i] = torch.as_tensor(np.where(ids >= 0, ids, 0))
    return inputs, card_targets, legal, bucket_ids


def card_values(model, inputs, bucket_ids):
    """Vectorized equivalent of inverse_bucket_outputs."""
    out = model(inputs).reshape(-1, 2, POSTFLOP_BUCKET_COUNT)
    idx = bucket_ids.unsqueeze(1).expand(-1, 2, -1)
    return torch.gather(out, 2, idx)


def masked_huber(values, targets, legal):
    mask = legal.unsqueeze(1).expand_as(values)
    return F.smooth_l1_loss(values[mask], targets[mask], beta=1.0, reduction="mean")


@torch.no_grad()
def eval_masked_huber(model, inputs, card_targets, legal, bucket_ids, chunk=512):
    """Masked Huber over a whole split, in chunks.

    A single forward pass over 100k rows would materialize [100k,2,1326] card
    values (~1 GB) plus the mask expansion, so evaluation is chunked and the
    per-element mean is reassembled from element counts rather than averaging
    the chunk means, which would misweight unequal chunks.
    """
    total = 0.0
    count = 0
    for lo in range(0, len(inputs), chunk):
        hi = min(len(inputs), lo + chunk)
        v = card_values(model, inputs[lo:hi], bucket_ids[lo:hi])
        m = legal[lo:hi].unsqueeze(1).expand_as(v)
        n = int(m.sum())
        if n:
            total += float(F.smooth_l1_loss(v[m], card_targets[lo:hi][m],
                                            beta=1.0, reduction="sum"))
            count += n
    return total / max(count, 1)


def verify_vectorized_path(model, boards, pots, ranges, targets, provider,
                           inputs, card_targets, legal, bucket_ids,
                           pot_convention="TOTAL_POT") -> float:
    """The fast path must agree with the project's reference implementation.

    The convention is threaded through, or a POT_HALF run would compare itself
    against a TOTAL_POT reference and fail for the wrong reason.
    """
    reference_batch = prepare_river_training_batch(
        boards=[boards[i] for i in range(min(4, len(boards)))],
        ranges=ranges[:4], pot_halves=pots[:4], targets_chips=targets[:4],
        bucket_provider=provider, stack=STACK, pot_convention=pot_convention)
    with torch.no_grad():
        ref = float(loss_on_prepared_batch(model, reference_batch))
        fast = float(masked_huber(card_values(model, inputs[:4], bucket_ids[:4]),
                                  card_targets[:4], legal[:4]))
    return abs(ref - fast)


def train_once(n_train, tr, va, epochs, batch_size, lr, seed, optimizer="adam"):
    inputs, card_targets, legal, bucket_ids = tr
    v_inputs, v_targets, v_legal, v_ids = va
    torch.manual_seed(seed)
    model = DeepStackHUNLValueNet(HUNLValueNetworkSpec())
    make = torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam
    opt = make(model.parameters(), lr=lr)
    best = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_train)
        for lo in range(0, n_train, batch_size):
            sel = order[lo:lo + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = masked_huber(card_values(model, inputs[sel], bucket_ids[sel]),
                                card_targets[sel], legal[sel])
            loss.backward()
            opt.step()
        model.eval()
        val = eval_masked_huber(model, v_inputs, v_targets, v_legal, v_ids)
        trn = eval_masked_huber(model, inputs[:n_train], card_targets[:n_train],
                                legal[:n_train], bucket_ids[:n_train])
        history.append({"epoch": epoch, "train": trn, "validation": val})
        best = min(best, val)
    return best, history[-1]["train"], history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datagen_out/river_v1")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    # Adam at 1e-3 is the released DeepStack-Leduc setting (train.lua:78,80 and
    # arguments.lua:54); AdamW at 3e-4 is the third-party DEVN choice. Default
    # to the anchored one.
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    ap.add_argument("--pot-convention", choices=("TOTAL_POT", "POT_HALF"),
                    default="TOTAL_POT",
                    help="TOTAL_POT reads the Supremus paper literally; "
                         "POT_HALF matches released DeepStack-Leduc")
    ap.add_argument("--threads", type=int, default=0, help="0 = leave torch default")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    boards, pots, ranges, targets, n_shards = load_shards(Path(args.data))
    n = len(boards)
    print(f"loaded {n} subgames from {n_shards} shards", flush=True)

    provider = ReconstructedRiverBucketProvider(
        RiverReconstructionArtifact.load(HERE / "HUNL_RIVER_BUCKET_RECONSTRUCTION_V1.npz"))

    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    n_val = max(1, int(round(args.val_fraction * n)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    print(f"split: {len(train_idx)} train / {len(val_idx)} validation", flush=True)

    t0 = time.perf_counter()
    tr = precompute(boards[train_idx], pots[train_idx], ranges[train_idx],
                    targets[train_idx], provider, args.pot_convention)
    va = precompute(boards[val_idx], pots[val_idx], ranges[val_idx],
                    targets[val_idx], provider, args.pot_convention)
    print(f"bucketed in {time.perf_counter()-t0:.1f}s", flush=True)

    torch.manual_seed(SEED)
    probe = DeepStackHUNLValueNet(HUNLValueNetworkSpec())
    delta = verify_vectorized_path(probe, boards[train_idx], pots[train_idx],
                                   ranges[train_idx], targets[train_idx], provider, *tr,
                                   pot_convention=args.pot_convention)
    print(f"fast path vs reference loss_on_prepared_batch: |delta| = {delta:.3e}", flush=True)
    if delta > 1e-6:
        raise SystemExit("vectorized training path disagrees with the reference implementation")

    sizes = [s for s in (100, 250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000)
             if s <= len(train_idx)]
    if not sizes or sizes[-1] != len(train_idx):
        sizes.append(len(train_idx))

    points = []
    for size in sizes:
        started = time.perf_counter()
        best, final_train, _ = train_once(size, tr, va, args.epochs, args.batch_size,
                                          args.lr, SEED, args.optimizer)
        points.append({
            "train_samples": size,
            "best_validation_huber": best,
            "final_train_huber": final_train,
            "seconds": time.perf_counter() - started,
        })
        print(json.dumps(points[-1]), flush=True)

    ENCODING_FLOOR = 4.343406267330793e-04
    PAPER_RIVER_VALIDATION = 1.5e-2
    result = {
        "schema": "HUNL_RIVER_TRAINING_CURVE_V1",
        "subgames_total": int(n),
        "validation_subgames": int(n_val),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "pot_convention": args.pot_convention,
        "loss_convention": "card-space masked Huber after inverse bucketing, "
                           "targets = raw chips / total pot",
        "reference_encoding_floor": ENCODING_FLOOR,
        "reference_paper_river_validation": PAPER_RIVER_VALIDATION,
        "curve": points,
        "best_validation_over_floor": (min(p["best_validation_huber"] for p in points)
                                       / ENCODING_FLOOR),
        "caveat": "learning rate, schedule, batch size and initialization are "
                  "project choices; the paper publishes only Adam + Huber. "
                  "Report the sample count with any loss figure.",
    }
    out = HERE / "HUNL_RIVER_TRAINING_CURVE_V1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("\n  samples   best val Huber   x floor   train Huber")
    for p in points:
        print(f"  {p['train_samples']:7d}   {p['best_validation_huber']:14.3e}"
              f"   {p['best_validation_huber']/ENCODING_FLOOR:7.2f}"
              f"   {p['final_train_huber']:11.3e}")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
