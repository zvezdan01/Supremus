"""HUNL value-network bucketing contract.

DeepStack's HUNL paper specifies 1,000 postflop hand clusters, produced by
k-means with earth mover's distance over hand-strength-like features.  The
original cluster artifacts / exact feature construction were not released.

This module therefore defines the *interface and algebra only*.  It does NOT
silently invent a replacement clustering.  A concrete ``BucketProvider`` must
supply a board-specific mapping from the frozen 1326-card-space ordering to
bucket ids.  This keeps the unresolved author artifact isolated from the rest
of the value-network / lookahead machinery.

Card-space ordering is the frozen HUNL contract in ``hunl.cards``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cards import HAND_COUNT, possible_hands_mask

POSTFLOP_BUCKET_COUNT = 1000


@dataclass(frozen=True)
class BoardBucketMap:
    """Bucket mapping for one public board.

    ``hand_to_bucket[h]`` is -1 for a board-blocked hand and otherwise an
    integer in ``[0, bucket_count)``.  Multiple legal hands may share a bucket.
    Empty bucket ids are permitted because a 1,000-cluster model can have
    board-specific buckets with no legal member after blockers.
    """

    board: tuple[int, ...]
    hand_to_bucket: np.ndarray
    bucket_count: int = POSTFLOP_BUCKET_COUNT

    def __post_init__(self) -> None:
        board = tuple(int(c) for c in self.board)
        mapping = np.asarray(self.hand_to_bucket)
        if mapping.shape != (HAND_COUNT,):
            raise ValueError(f"hand_to_bucket must have shape ({HAND_COUNT},)")
        if not np.issubdtype(mapping.dtype, np.integer):
            raise TypeError("hand_to_bucket must have integer dtype")
        legal = possible_hands_mask(board)
        if np.any(mapping[~legal] != -1):
            raise ValueError("board-blocked hands must map to -1")
        if np.any(mapping[legal] < 0) or np.any(mapping[legal] >= self.bucket_count):
            raise ValueError("legal hands must map into [0,bucket_count)")
        object.__setattr__(self, "board", board)
        object.__setattr__(self, "hand_to_bucket", mapping.astype(np.int16, copy=True))

    @property
    def legal_mask(self) -> np.ndarray:
        return self.hand_to_bucket >= 0

    def range_to_buckets(self, card_range: np.ndarray) -> np.ndarray:
        """Sum a card-space probability/reach vector into bucket space.

        No normalization is performed.  That is deliberate: at a chance
        boundary the sum is the probability mass compatible with that board,
        which is needed both for conditioning the NN input and for restoring
        opponent counterfactual reach afterwards.
        """
        x = np.asarray(card_range)
        if x.shape[-1] != HAND_COUNT:
            raise ValueError(f"last dimension must be {HAND_COUNT}")
        out_shape = x.shape[:-1] + (self.bucket_count,)
        out = np.zeros(out_shape, dtype=x.dtype)
        ids = self.hand_to_bucket
        legal = ids >= 0
        flat_in = x.reshape(-1, HAND_COUNT)
        flat_out = out.reshape(-1, self.bucket_count)
        legal_ids = ids[legal].astype(np.int64)
        for r in range(flat_in.shape[0]):
            np.add.at(flat_out[r], legal_ids, flat_in[r, legal])
        return out

    def bucket_values_to_hands(self, bucket_values: np.ndarray) -> np.ndarray:
        """Inverse-bucket a value vector by copying each bucket value to hands."""
        x = np.asarray(bucket_values)
        if x.shape[-1] != self.bucket_count:
            raise ValueError(f"last dimension must be {self.bucket_count}")
        out = np.zeros(x.shape[:-1] + (HAND_COUNT,), dtype=x.dtype)
        ids = self.hand_to_bucket
        legal = ids >= 0
        out[..., legal] = x[..., ids[legal].astype(np.int64)]
        return out

    def possible_bucket_mask(self) -> np.ndarray:
        mask = np.zeros(self.bucket_count, dtype=bool)
        ids = self.hand_to_bucket[self.hand_to_bucket >= 0].astype(np.int64)
        mask[ids] = True
        return mask


class BucketProvider(Protocol):
    bucket_count: int

    def for_board(self, board: tuple[int, ...]) -> BoardBucketMap: ...


class MissingAuthorBucketProvider:
    """Fail-closed placeholder for the unreleased DeepStack cluster artifact."""

    bucket_count = POSTFLOP_BUCKET_COUNT

    def for_board(self, board: tuple[int, ...]) -> BoardBucketMap:
        raise RuntimeError(
            "Original DeepStack HUNL 1000-bucket mapping is not available. "
            "Provide a versioned BucketProvider; no guessed mapping is used."
        )
