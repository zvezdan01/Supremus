"""Versioned RECONSTRUCTION (not original Supremus) of 1000 river buckets.

What the primary source constrains (Zarick et al., arXiv:2007.10442):
  * both DeepStack and Supremus reduce 1326 hand probabilities/values by
    bucketing;
  * the river network consumes 1000 bucket probabilities per player;
  * the river network is trained first.

What the private Supremus artifact does NOT publish:
  * the actual 1000 river clusters, or any generator for them;
  * the feature the clusters were fitted over;
  * RNG seed, restart count and bucket ordering.

This module therefore creates an explicitly PROJECT_RECONSTRUCTION artifact.
It is a reproducible replacement suitable for training a replacement river
value network; it must never be labelled the original Supremus buckets.

Feature: for a fixed five-card river board and a legal hero hand, terminal
equity against a uniformly random compatible opponent hand.  There are exactly
C(45,2)=990 such opponent hands, so ``2*wins + ties`` is an exact integer in
``[0, 1980]`` and the feature is that numerator over 1980.  Unlike the turn
feature there is no future chance, so this is exact, not a histogram.

Numeric contract (matters for byte-reproducibility of the artifact): features
are quantized to float32 first, and the k-means fit then runs in float64 over
those float32 values.  Centroids are narrowed back to float32 only when
serialized.  Fitting directly on float64 ``numerator/1980`` gives a *different*
artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .blockers import legal_pairs_mask
from .cards import CARD_COUNT, HAND_COUNT, possible_hands_mask
from .evaluator import rank_board_hands
from .value_bucketing import BoardBucketMap, POSTFLOP_BUCKET_COUNT

RIVER_EQUITY_DENOMINATOR = 1980  # 2 * C(45,2)


def river_uniform_equity_numerators(board5) -> np.ndarray:
    """Exact ``2*wins + ties`` per hand against a uniform legal opponent.

    Returns int32 ``[1326]``; board-blocked hands are -1 and every legal hand
    is in ``[0, 1980]``.  A legal hand always has exactly 990 opponents, so no
    per-hand normalization by a variable count is needed.
    """
    b = tuple(int(c) for c in board5)
    if len(b) != 5:
        raise ValueError("river equity requires a 5-card public board")
    ranks = rank_board_hands(b).astype(np.int64)
    legal = possible_hands_mask(b)
    pairs = legal_pairs_mask(b)
    # +2 when the hero hand wins, +1 on a tie, 0 when it loses: summing this
    # over legal opponents is exactly 2*wins + ties.
    val = (ranks[:, None] > ranks[None, :]).astype(np.int16) * 2
    val += (ranks[:, None] == ranks[None, :]).astype(np.int16)
    val[~pairs] = 0
    out = val.sum(axis=1, dtype=np.int32)
    out[~legal] = -1
    return out


def river_uniform_equity(board5) -> np.ndarray:
    """``river_uniform_equity_numerators`` as float32 equity; blocked -> nan."""
    nums = river_uniform_equity_numerators(board5)
    out = np.full(HAND_COUNT, np.nan, dtype=np.float32)
    legal = nums >= 0
    out[legal] = nums[legal].astype(np.float32) / np.float32(RIVER_EQUITY_DENOMINATOR)
    return out


@dataclass(frozen=True)
class RiverReconstructionArtifact:
    centroids: np.ndarray  # [1000] scalar equities, ascending
    manifest: dict

    def save(self, path: str | Path) -> None:
        p = Path(path)
        m = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"))
        np.savez_compressed(p, centroids=self.centroids.astype(np.float32),
                            manifest_json=np.asarray(m))

    @classmethod
    def load(cls, path: str | Path) -> "RiverReconstructionArtifact":
        z = np.load(path, allow_pickle=False)
        cent = np.asarray(z["centroids"], dtype=np.float32)
        manifest = json.loads(str(z["manifest_json"].item()))
        return cls(cent, manifest)


def sample_river_strength_features(
    *, seed: int, boards: int, hands_per_board: int,
) -> tuple[np.ndarray, dict]:
    """PROJECT sampling scheme: uniform raw river boards, uniform legal hands.

    Board draws and hand draws share one generator and are interleaved, so the
    board stream is only reproducible if the hand draw is replayed too.
    """
    rng = np.random.default_rng(int(seed))
    feats = []
    board_log = []
    for _ in range(int(boards)):
        board = tuple(int(x) for x in rng.choice(CARD_COUNT, size=5, replace=False))
        nums = river_uniform_equity_numerators(board)
        ids = np.flatnonzero(possible_hands_mask(board))
        take = min(int(hands_per_board), len(ids))
        pick = rng.choice(ids, size=take, replace=False)
        feats.append(nums[pick].astype(np.float32)
                     / np.float32(RIVER_EQUITY_DENOMINATOR))
        board_log.append(board)
    x = np.concatenate(feats, axis=0)
    meta = {
        "seed": int(seed),
        "boards": int(boards),
        "hands_per_board": int(hands_per_board),
        "samples": int(x.shape[0]),
        "board_stream_sha256": hashlib.sha256(
            np.asarray(board_log, dtype='<i2').tobytes()).hexdigest(),
        "feature_sha256": hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
    }
    return x, meta


def _kmeanspp_scalar(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Standard D^2 k-means++ initialization under scalar |dx| distance."""
    n = x.shape[0]
    if k > n:
        raise ValueError("k cannot exceed number of training features")
    chosen = np.empty(k, dtype=np.int32)
    first = int(rng.integers(0, n))
    chosen[0] = first
    min_d = np.abs(x - x[first])
    min_d[first] = 0.0
    used = np.zeros(n, dtype=bool)
    used[first] = True
    for i in range(1, k):
        w = min_d.astype(np.float64) ** 2
        w[used] = 0.0
        total = float(w.sum())
        if total <= 0.0:
            # Degenerate duplicate feature set: deterministic first unused.
            idx = int(np.flatnonzero(~used)[0])
        else:
            t = float(rng.random()) * total
            idx = int(np.searchsorted(np.cumsum(w), t, side="right"))
            if idx >= n or used[idx]:
                idx = int(np.flatnonzero(~used)[0])
        chosen[i] = idx
        used[idx] = True
        np.minimum(min_d, np.abs(x - x[idx]), out=min_d)
    return x[chosen].copy()


def _assign_scalar(x: np.ndarray, cent: np.ndarray, chunk: int = 8192):
    """Nearest centroid by |dx|; ties resolve to the lowest centroid index."""
    n = x.shape[0]
    labels = np.empty(n, dtype=np.int32)
    dmin = np.empty(n, dtype=cent.dtype)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        d = np.abs(x[lo:hi, None] - cent[None, :])
        lab = np.argmin(d, axis=1)
        labels[lo:hi] = lab.astype(np.int32)
        dmin[lo:hi] = d[np.arange(hi - lo), lab]
    return labels, dmin


def fit_river_bucket_reconstruction_v1(
    x: np.ndarray,
    *,
    k: int = POSTFLOP_BUCKET_COUNT,
    seed: int = 20260816,
    iterations: int = 10,
) -> RiverReconstructionArtifact:
    """Deterministic Lloyd k-means over exact scalar river equity.

    Assignment: nearest centroid by absolute equity distance.
    Update: arithmetic mean of assigned samples; empty clusters keep their
    previous centroid, which is deterministic and explicit.

    The fit runs in float64 over float32-quantized features (see module
    docstring); the returned centroids are float32 and sorted ascending, which
    is a PROJECT canonical ordering, not a Supremus-published one.
    """
    xf = np.asarray(x, dtype=np.float32).astype(np.float64)
    if xf.ndim != 1:
        raise ValueError("expected a 1-D vector of scalar equity features")
    if float(xf.min()) < 0.0 or float(xf.max()) > 1.0:
        raise ValueError("equity features must lie in [0,1]")
    rng = np.random.default_rng(int(seed))
    cent = _kmeanspp_scalar(xf, int(k), rng)
    objectives = []
    populations = None
    for _ in range(int(iterations)):
        labels, dmin = _assign_scalar(xf, cent)
        objectives.append(float((dmin.astype(np.float64) ** 2).sum()))
        populations = np.bincount(labels, minlength=k).astype(np.int32)
        sums = np.bincount(labels, weights=xf, minlength=k)
        nonempty = populations > 0
        cent[nonempty] = sums[nonempty] / populations[nonempty]
    cent = np.sort(cent).astype(np.float32)
    manifest = {
        "schema": "HUNL_RIVER_BUCKET_RECONSTRUCTION_V1",
        "status": "PROJECT_RECONSTRUCTION_NOT_ORIGINAL",
        "bucket_count": int(k),
        "feature": "exact river hand strength = equity vs uniform legal opponent",
        "equity": "win + 0.5*tie; exact numerator denominator 1980",
        "distance": "absolute scalar equity distance; squared objective for k-means",
        "initialization": "standard k-means++ D^2",
        "centroid_update": "arithmetic mean scalar equity",
        "bucket_order": "centroids sorted ascending after fit (PROJECT canonical)",
        "restarts": 1,
        "iterations": int(iterations),
        "seed": int(seed),
        "objective_history_squared_equity": objectives,
        "empty_clusters_last_lloyd_step": int(np.count_nonzero(populations == 0)),
        "centroid_sha256": hashlib.sha256(cent.astype('<f4').tobytes()).hexdigest(),
    }
    return RiverReconstructionArtifact(cent, manifest)


class ReconstructedRiverBucketProvider:
    """Board-specific 1326->1000 assignment from a versioned centroid file."""

    bucket_count = POSTFLOP_BUCKET_COUNT

    def __init__(self, artifact: RiverReconstructionArtifact):
        c = np.asarray(artifact.centroids, dtype=np.float32)
        if c.shape != (POSTFLOP_BUCKET_COUNT,):
            raise ValueError("river centroid artifact must be [1000]")
        self.artifact = artifact
        self.centroids = c
        self._centroids64 = c.astype(np.float64)

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(RiverReconstructionArtifact.load(path))

    def for_board(self, board: tuple[int, ...]) -> BoardBucketMap:
        b = tuple(int(c) for c in board)
        if len(b) != 5:
            raise ValueError("river bucket provider requires a 5-card public board")
        nums = river_uniform_equity_numerators(b)
        ids = np.flatnonzero(nums >= 0)
        feat = (nums[ids].astype(np.float32)
                / np.float32(RIVER_EQUITY_DENOMINATOR)).astype(np.float64)
        labels, _ = _assign_scalar(feat, self._centroids64, chunk=4096)
        mapping = np.full(HAND_COUNT, -1, dtype=np.int16)
        mapping[ids] = labels.astype(np.int16)
        return BoardBucketMap(b, mapping, POSTFLOP_BUCKET_COUNT)


__all__ = [
    "RIVER_EQUITY_DENOMINATOR",
    "river_uniform_equity_numerators", "river_uniform_equity",
    "RiverReconstructionArtifact", "sample_river_strength_features",
    "fit_river_bucket_reconstruction_v1", "ReconstructedRiverBucketProvider",
]
