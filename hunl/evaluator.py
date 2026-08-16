"""HUNL 7-card evaluator — faithful vectorized port of the ACPC
`rankCardset` (reference_lua/ACPCServer/evalHandTables:4154-4253).

The lookup tables are NOT retyped: they are parsed at import time from an
untouched original `evalHandTables` file and their canonical digest is
pinned. Two author-side copies exist in the certified sources and carry
numerically identical tables (verified):
  - DS repo reference_lua/ACPCServer/evalHandTables
      SHA-256 9b8bb8e1c73503d55073757d0434380a69c40431713448c3d578f1a8dca7c3e4
  - quant-trade third_party/CFR_plus/evalHandTables
      SHA-256 53248e54bafb8fbc67830230baf4ad92abaf1e95425c0326e14e7e7a82ef8425
    (same tables; omits the rankCardset consumer code)

Rank semantics: identical integers to the original (higher = stronger;
class boundaries per the HANDCLASS_* defines).
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

from .cards import CARD_COUNT, HAND_CARDS, HAND_COUNT, MAX_SUITS

_KNOWN_FILE_SHA256 = {
    "9b8bb8e1c73503d55073757d0434380a69c40431713448c3d578f1a8dca7c3e4",
    "53248e54bafb8fbc67830230baf4ad92abaf1e95425c0326e14e7e7a82ef8425",
}
# canonical digest over the parsed numeric tables (both copies agree)
_TABLES_DIGEST = "c1f905c09588e32e3fe756a6f1e608cc"

HANDCLASS_SINGLE_CARD = 0
HANDCLASS_PAIR = 1287
HANDCLASS_TWO_PAIR = 5005
HANDCLASS_TRIPS = 8606
HANDCLASS_STRAIGHT = 9620
HANDCLASS_FLUSH = 9633
HANDCLASS_FULL_HOUSE = 10920
HANDCLASS_QUADS = 11934
HANDCLASS_STRAIGHT_FLUSH = 12103

BLOCKED_SENTINEL = np.uint32(0xFFFFFFFF)

_TABLE_SPECS = (
    ("oneSuitVal", 8192), ("pairOtherVal", 8192), ("anySuitVal", 8192),
    ("topBit", 8192), ("tripsOtherVal", 8192), ("quadsVal", 13),
    ("tripsVal", 13), ("pairsVal", 13), ("twoPairOtherVal", 13),
)


def _default_table_paths() -> list[Path]:
    here = Path(__file__).resolve().parent.parent
    candidates = []
    env = os.environ.get("HUNL_EVAL_TABLES")
    if env:
        candidates.append(Path(env))
    candidates.append(here / "third_party" / "CFR_plus" / "evalHandTables")
    candidates.append(Path(
        "/workspace/deepstack_leduc_v1.1-bitexact-certified/"
        "reference_lua/ACPCServer/evalHandTables"))
    return candidates


def _load_tables() -> dict[str, np.ndarray]:
    for path in _default_table_paths():
        if path.is_file():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() not in _KNOWN_FILE_SHA256:
                raise RuntimeError(f"unrecognized evalHandTables copy: {path}")
            text = data.decode()
            tables: dict[str, np.ndarray] = {}
            digest = hashlib.sha256()
            for name, size in _TABLE_SPECS:
                m = re.search(name + r"\[\s*\d+\s*\]\s*=\s*\{(.*?)\}", text, re.S)
                if not m:
                    raise RuntimeError(f"table {name} not found in {path}")
                nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
                if len(nums) != size:
                    raise RuntimeError(f"table {name}: {len(nums)} != {size}")
                digest.update((name + ":" + ",".join(map(str, nums)) + ";").encode())
                tables[name] = np.asarray(nums, dtype=np.int32)
            if digest.hexdigest()[:32] != _TABLES_DIGEST:
                raise RuntimeError(f"table digest mismatch in {path}")
            tables["__source__"] = str(path)  # type: ignore[assignment]
            return tables
    raise RuntimeError("no evalHandTables copy found")


_T = _load_tables()
_ONE_SUIT = _T["oneSuitVal"]
_PAIR_OTHER = _T["pairOtherVal"]
_ANY_SUIT = _T["anySuitVal"]
_TOP_BIT = _T["topBit"]
_TRIPS_OTHER = _T["tripsOtherVal"]
_QUADS_VAL = _T["quadsVal"]
_TRIPS_VAL = _T["tripsVal"]
_PAIRS_VAL = _T["pairsVal"]
_TWO_PAIR_OTHER = _T["twoPairOtherVal"]
_FULL_HOUSE_OTHER = HANDCLASS_FULL_HOUSE - HANDCLASS_TRIPS
TABLE_SOURCE = _T["__source__"]


def suit_masks(cards: np.ndarray) -> np.ndarray:
    """(N, k) card ids -> (N, 4) int32 per-suit 13-bit rank masks."""
    cards = np.asarray(cards)
    bits = np.left_shift(np.int32(1), (cards >> 2).astype(np.int32))
    suits = (cards & 3).astype(np.int8)
    out = np.zeros(cards.shape[:1] + (MAX_SUITS,), dtype=np.int32)
    for s in range(MAX_SUITS):
        out[:, s] = np.bitwise_or.reduce(
            np.where(suits == s, bits, 0), axis=1)
    return out


def rank_suit_masks(by_suit: np.ndarray) -> np.ndarray:
    """Vectorized rankCardset over (N, 4) suit masks; returns (N,) int32.

    Mirrors evalHandTables:4154-4253 branch-for-branch (branches become
    masked selections)."""
    s = by_suit
    postponed = np.max(_ONE_SUIT[s], axis=1)

    # multiplicity masks (identical algebra to the C code)
    m0 = s[:, 0] | s[:, 1]
    m1 = s[:, 0] & s[:, 1]
    m2 = m1 & s[:, 2]
    m1 = m1 | (m0 & s[:, 2])
    m0 = m0 | s[:, 2]
    m3 = m2 & s[:, 3]
    m2 = m2 | (m1 & s[:, 3])
    m1 = m1 | (m0 & s[:, 3])
    m0 = m0 | s[:, 3]

    res = np.full(s.shape[0], -1, dtype=np.int32)
    done = postponed >= HANDCLASS_STRAIGHT_FLUSH        # straight flush
    res[done] = postponed[done]

    # quads
    quads = (~done) & (m3 != 0)
    if quads.any():
        r = _TOP_BIT[m3[quads]]
        res[quads] = _QUADS_VAL[r] + _TOP_BIT[m0[quads] ^ (1 << r)]
    done |= quads

    # trips branch (trips / full house / flush / straight)
    tb = (~done) & (m2 != 0)
    if tb.any():
        r = _TOP_BIT[m2[tb]]
        m1t = m1[tb] ^ (1 << r)
        out = np.empty(int(tb.sum()), dtype=np.int32)
        fh = m1t != 0
        out[fh] = _TRIPS_VAL[r[fh]] + _FULL_HOUSE_OTHER + _TOP_BIT[m1t[fh]]
        rest = ~fh
        pp = postponed[tb]
        fl = rest & (pp != 0)                            # flush
        out[fl] = pp[fl]
        rest &= ~fl
        anyv = _ANY_SUIT[m0[tb]]
        st = rest & (anyv >= HANDCLASS_STRAIGHT)         # straight
        out[st] = anyv[st]
        rest &= ~st
        m0t = m0[tb] ^ (1 << r)                          # trips
        out[rest] = _TRIPS_VAL[r[rest]] + _TRIPS_OTHER[m0t[rest]]
        res[tb] = out
    done |= tb

    # no set of three: flush / straight, else fall through
    low = (~done)
    if low.any():
        pp = postponed[low]
        m0l = m0[low]
        m1l = m1[low]
        out = np.empty(int(low.sum()), dtype=np.int32)
        fl = pp != 0                                     # flush
        out[fl] = pp[fl]
        rest = ~fl
        anyv = _ANY_SUIT[m0l]
        st = rest & (anyv >= HANDCLASS_STRAIGHT)         # straight
        out[st] = anyv[st]
        rest &= ~st
        pairs = rest & (m1l != 0)                        # pair / two pair
        if pairs.any():
            r = _TOP_BIT[m1l[pairs]]
            a0 = m0l[pairs] ^ (1 << r)
            a1 = m1l[pairs] ^ (1 << r)
            pout = np.empty(int(pairs.sum()), dtype=np.int32)
            tp = a1 != 0
            if tp.any():
                r2 = _TOP_BIT[a1[tp]]
                a0tp = a0[tp] ^ (1 << r2)
                pout[tp] = _PAIRS_VAL[r[tp]] + _TWO_PAIR_OTHER[r2] + _TOP_BIT[a0tp]
            pout[~tp] = _PAIRS_VAL[r[~tp]] + _PAIR_OTHER[a0[~tp]]
            out[pairs] = pout
        high = rest & (m1l == 0)                         # high card
        out[high] = anyv[high]
        res[low] = out
    return res


def rank7(cards7: np.ndarray) -> np.ndarray:
    """(N, 7) ACPC card ids -> (N,) int32 ACPC hand ranks."""
    return rank_suit_masks(suit_masks(cards7))


def hand_class(ranks: np.ndarray) -> np.ndarray:
    """0..8 class index (high card .. straight flush)."""
    bounds = np.array([HANDCLASS_PAIR, HANDCLASS_TWO_PAIR, HANDCLASS_TRIPS,
                       HANDCLASS_STRAIGHT, HANDCLASS_FLUSH,
                       HANDCLASS_FULL_HOUSE, HANDCLASS_QUADS,
                       HANDCLASS_STRAIGHT_FLUSH], dtype=np.int32)
    return np.searchsorted(bounds, np.asarray(ranks), side="right").astype(np.int8)


def rank_board_hands(board5) -> np.ndarray:
    """5-card board -> (1326,) uint32: ACPC rank of board+hand for every
    hole pair; hands colliding with the board get BLOCKED_SENTINEL."""
    board = np.asarray(list(board5), dtype=np.int8)
    if board.shape != (5,) or len(set(board.tolist())) != 5:
        raise ValueError("board must be 5 distinct cards")
    cards7 = np.empty((HAND_COUNT, 7), dtype=np.int8)
    cards7[:, :2] = HAND_CARDS
    cards7[:, 2:] = board
    ranks = rank7(cards7).astype(np.uint32)
    blocked = (
        np.isin(HAND_CARDS[:, 0], board) | np.isin(HAND_CARDS[:, 1], board))
    ranks[blocked] = BLOCKED_SENTINEL
    return ranks
