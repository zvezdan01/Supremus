"""HUNL exact river resolver (Gates G1.7 + G1.8).

NO new CFR code: the solver is the UNTOUCHED certified Golden Baseline
engine (`deepstack_leduc.lookahead.Lookahead` + `cfrd_gadget.CFRDGadget`),
which is dimension-parameterized by `Config.card_count` throughout. This
module only injects the HUNL game layer through documented seams:

  1. Config(card_count=1326, stack=20000, cfr_iters=2000,
     cfr_skip_iters=1000) — the VERIFIED Table-4 river schedule.
  2. A lookahead tree converted from the G1-certified river betting tree
     (`hunl.tree.RiverTreeBuilder`) into golden `deepstack_leduc.tree.Node`
     objects, with the AUTHOR LOOKAHEAD CONVENTION of a fold child as the
     first action at EVERY decision node (Lua tree_builder.lua behavior;
     consistent with Table 4 listing F in every river action set). At
     nodes where no bet is faced this fold is ACPC-dominated but present —
     exactly as in the certified Leduc trees. All non-fold actions are the
     ACPC-certified action sets, unchanged.
  3. `HunlRiverTerminalEquity` (G1.3-derived matrices, Leduc orientation).
  4. The gadget's board mask replaced by the G1.2 1326-hand possibility
     mask (the golden gadget computes a Leduc-semantic mask from the board
     in its constructor; for HUNL the mask is overwritten before any use —
     no gadget math is altered).

The same injection wrapper instantiated with the LEDUC components must be
byte-identical to the golden engine on Leduc inputs — that equivalence is
part of the G1.7 regression gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DS_ROOT = Path("/workspace/deepstack_leduc_v1.1-bitexact-certified")
if str(DS_ROOT) not in sys.path:
    sys.path.insert(0, str(DS_ROOT))

from deepstack_leduc.config import Config as LeducConfig  # noqa: E402
from deepstack_leduc.lookahead import Lookahead, LookaheadResults  # noqa: E402
from deepstack_leduc.cfrd_gadget import CFRDGadget  # noqa: E402
from deepstack_leduc.tree import CALL, FOLD, Node as GoldenNode  # noqa: E402

from .cards import HAND_COUNT, possible_hands_mask  # noqa: E402
from .config import DEFAULT_CONFIG, HunlConfig  # noqa: E402
from .river_terminal import HunlRiverTerminalEquity  # noqa: E402
from .tree import RiverNode, RiverTreeBuilder  # noqa: E402


def hunl_river_config(hunl_cfg: HunlConfig = DEFAULT_CONFIG) -> LeducConfig:
    """Golden-engine Config carrying the VERIFIED river schedule."""
    return LeducConfig(
        ante=hunl_cfg.big_blind,
        stack=hunl_cfg.stack,
        bet_fractions=tuple(),           # tree is pre-built; sizing unused
        cfr_iters=hunl_cfg.river_cfr_iters,
        cfr_skip_iters=hunl_cfg.river_cfr_omit,
        card_count=HAND_COUNT,
    )


def build_lookahead_tree(pot_half: int, board5,
                         hunl_cfg: HunlConfig = DEFAULT_CONFIG,
                         card_count: int = HAND_COUNT) -> GoldenNode:
    """G1-certified river tree -> golden Node tree, author fold convention.

    street=2 marks the terminal street (river) for the golden engine.
    bets are total committed chips per seat (float64, Leduc Node dtype).
    """
    builder = RiverTreeBuilder(hunl_cfg)
    src_root = builder.build(pot_half)
    board = tuple(int(c) for c in board5)

    def convert(src: RiverNode) -> GoldenNode:
        node = GoldenNode(
            street=2,
            current_player=src.player,
            bets=np.asarray(src.spent, dtype=float),
            board=board,
        )
        if src.terminal is not None:
            node.terminal = True
            node.terminal_type = src.terminal
            node.depth = 1
            return node
        children: list[GoldenNode] = []
        actions: list[int] = []
        has_fold = any(a == "fold" for a, _ in src.actions)
        if not has_fold:
            # author lookahead convention: dominated fold present at
            # check-able nodes (Lua tree_builder.lua always emits fold)
            fold_child = GoldenNode(
                street=2, current_player=1 - src.player,
                bets=np.asarray(src.spent, dtype=float), board=board,
                terminal=True, terminal_type="fold")
            fold_child.depth = 1
            children.append(fold_child)
            actions.append(FOLD)
        for (kind, size), sub in zip(src.actions, src.children):
            child = convert(sub)
            children.append(child)
            actions.append(FOLD if kind == "fold"
                           else CALL if kind == "call" else int(size))
        node.children = children
        node.actions = actions
        node.depth = max(c.depth for c in children) + 1
        n = len(children)
        node.strategy = np.full((n, card_count), 1.0 / n, dtype=float)
        return node

    root = convert(src_root)
    assert root.actions[0] == FOLD and root.actions[1] == CALL
    return root


class InjectedLookahead(Lookahead):
    """Golden lookahead with terminal-equity + gadget-mask injection seams.

    No numerical method is overridden; `build_lookahead` replicates the
    golden 5-line body with a pluggable terminal-equity factory, and
    `resolve` replicates the golden body with a pluggable gadget mask.
    Instantiated with the Leduc factories it must be (and is certified)
    byte-identical to the golden class.
    """

    def __init__(self, cfg, terminal_equity_factory, range_mask=None,
                 value_network=None):
        super().__init__(cfg, value_network)
        self._te_factory = terminal_equity_factory
        self._range_mask = range_mask

    def build_lookahead(self, tree) -> None:            # mirror of golden body
        from deepstack_leduc.lookahead_builder import LookaheadBuilder
        self.tree = tree
        self.layout = LookaheadBuilder(self.cfg).build_from_tree(tree)
        self.terminal_equity = self._te_factory(tree.board)
        self._bind_layout_fields()
        self._construct_transition_boxes()

    def resolve(self, player_range, opponent_cfvs) -> None:  # mirror + mask
        self._require_built()
        assert self.tree is not None
        player = np.asarray(player_range, dtype=np.float32)
        opp_cfvs = np.asarray(opponent_cfvs, dtype=np.float32)
        self.reconstruction_gadget = CFRDGadget(
            self.tree.board, player, opp_cfvs, self.cfg)
        if self._range_mask is not None:
            self.reconstruction_gadget.range_mask = np.asarray(
                self._range_mask, dtype=np.float32)
        self.ranges_data[1][..., 0, :] = player
        self.reconstruction_opponent_cfvs = opp_cfvs.copy()
        self._compute()


class RiverResolver:
    """Exact, unabstracted 1326-hand HUNL river resolver.

    Semantics identical to `deepstack_leduc.resolving.Resolving` (root
    player = seat acting first on the river = seat 0/BB), backed by the
    untouched golden engine.
    """

    def __init__(self, hunl_cfg: HunlConfig = DEFAULT_CONFIG,
                 cfr_iters: int | None = None,
                 cfr_skip_iters: int | None = None) -> None:
        self.hunl_cfg = hunl_cfg
        cfg = hunl_river_config(hunl_cfg)
        if cfr_iters is not None:
            cfg = LeducConfig(**{**cfg.__dict__, "cfr_iters": cfr_iters,
                                 "cfr_skip_iters": cfr_skip_iters})
        self.cfg = cfg
        self.lookahead: InjectedLookahead | None = None
        self.results: LookaheadResults | None = None
        self.lookahead_tree: GoldenNode | None = None

    def _build(self, board5, pot_half: int) -> None:
        board = tuple(int(c) for c in board5)
        self.lookahead_tree = build_lookahead_tree(
            pot_half, board, self.hunl_cfg, self.cfg.card_count)
        mask = possible_hands_mask(board).astype(np.float32)
        self.lookahead = InjectedLookahead(
            self.cfg, HunlRiverTerminalEquity, range_mask=mask)
        self.lookahead.build_lookahead(self.lookahead_tree)

    def resolve_first_node(self, board5, pot_half: int,
                           player_range: np.ndarray,
                           opponent_range: np.ndarray) -> LookaheadResults:
        self._build(board5, pot_half)
        assert self.lookahead is not None
        self.lookahead.resolve_first_node(player_range, opponent_range)
        self.results = self.lookahead.get_results()
        return self.results

    def resolve(self, board5, pot_half: int, player_range: np.ndarray,
                opponent_cfvs: np.ndarray) -> LookaheadResults:
        self._build(board5, pot_half)
        assert self.lookahead is not None
        self.lookahead.resolve(player_range, opponent_cfvs)
        self.results = self.lookahead.get_results()
        return self.results

    def get_root_cfv_both_players(self) -> np.ndarray:
        assert self.results is not None
        assert self.results.root_cfvs_both_players is not None
        return self.results.root_cfvs_both_players.copy()

    def get_root_strategy(self) -> np.ndarray:
        assert self.results is not None
        return self.results.strategy.copy()
