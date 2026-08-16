"""HUNL 52-card layer, ACPC-exact conventions.

Encoding is the ACPC one (reference_lua/ACPCServer/game.h:249-251):
    card = rank * MAX_SUITS + suit,  rank 0..12 ('2'..'A'), suit 0..3 ('cdhs')
Characters per game.c:103-104: suitChars "cdhs", rankChars "23456789TJQKA".

Private-hand contract (FROZEN — G1 ordering contract):
    the 1326 hole-card pairs are enumerated lexicographically as
    (c0, c1) with c0 < c1 over card ids 0..51; hand_index(c0, c1) is the
    position in that enumeration.
"""
from __future__ import annotations

import numpy as np

MAX_SUITS = 4
MAX_RANKS = 13
CARD_COUNT = MAX_SUITS * MAX_RANKS          # 52
HAND_COUNT = CARD_COUNT * (CARD_COUNT - 1) // 2  # 1326

SUIT_CHARS = "cdhs"
RANK_CHARS = "23456789TJQKA"


def make_card(rank: int, suit: int) -> int:
    return rank * MAX_SUITS + suit


def rank_of_card(card: int) -> int:
    return card // MAX_SUITS


def suit_of_card(card: int) -> int:
    return card % MAX_SUITS


def card_to_string(card: int) -> str:
    return RANK_CHARS[rank_of_card(card)] + SUIT_CHARS[suit_of_card(card)]


def string_to_card(text: str) -> int:
    rank = RANK_CHARS.index(text[0].upper())
    suit = SUIT_CHARS.index(text[1].lower())
    return make_card(rank, suit)


def string_to_cards(text: str) -> tuple[int, ...]:
    text = text.strip()
    if len(text) % 2:
        raise ValueError(f"invalid card string: {text!r}")
    return tuple(string_to_card(text[i:i + 2]) for i in range(0, len(text), 2))


def _build_hand_tables() -> tuple[np.ndarray, np.ndarray]:
    hand_cards = np.empty((HAND_COUNT, 2), dtype=np.int8)
    hand_index = np.full((CARD_COUNT, CARD_COUNT), -1, dtype=np.int32)
    i = 0
    for c0 in range(CARD_COUNT):
        for c1 in range(c0 + 1, CARD_COUNT):
            hand_cards[i, 0] = c0
            hand_cards[i, 1] = c1
            hand_index[c0, c1] = i
            hand_index[c1, c0] = i
            i += 1
    assert i == HAND_COUNT
    return hand_cards, hand_index


HAND_CARDS, HAND_INDEX = _build_hand_tables()


def hand_index(c0: int, c1: int) -> int:
    idx = int(HAND_INDEX[c0, c1])
    if idx < 0:
        raise ValueError(f"not a hand: ({c0},{c1})")
    return idx


def possible_hands_mask(board: tuple[int, ...] | np.ndarray) -> np.ndarray:
    """Boolean (1326,), True where neither hole card is on the board."""
    blocked = np.zeros(CARD_COUNT, dtype=bool)
    for c in board:
        blocked[int(c)] = True
    return ~(blocked[HAND_CARDS[:, 0]] | blocked[HAND_CARDS[:, 1]])
