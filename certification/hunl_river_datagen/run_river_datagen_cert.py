from __future__ import annotations
import hashlib, json, time
from pathlib import Path

import numpy as np

from hunl.cards import HAND_COUNT, possible_hands_mask
from hunl.river_dcfr_plus import DcfrPlusSpec, RiverDcfrPlusEngine
from hunl.supremus_config import (
    ChipQuantization, SupremusRiverConfig, UnpublishedSupremusDetail,
)
from hunl.tree import RiverTreeBuilder
from hunl_datagen.author_range_v2 import current_public_hand_strength
from hunl_datagen.river_datagen_v1 import (
    RiverDatagenV1Config, RiverDataGeneratorV1, RiverDatagenMode,
)
from hunl_datagen.turn_datagen_v2 import CountingTHRandom
from certification.hunl_river_datagen.lp_supremus_river import solve_lp

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).with_name('HUNL_RIVER_DATAGEN_V1_CERT.json')
rows={}

# A. River current-public hand-strength combinatorics.
board=(0,5,10,15,20)
hs=current_public_hand_strength(board)
assert len(hs.legal_hands)==1081
assert np.all(hs.opponent_counts==990)
rows['river_range_combinatorics']={'legal_hands':1081,'compatible_opponents_per_hand':990}

# B. Strictly expose unpublished chip rounding.
try:
    RiverTreeBuilder(SupremusRiverConfig(rounding=ChipQuantization.AUTHOR_STRICT)).build(201)
    raise AssertionError('strict action quantization unexpectedly succeeded')
except UnpublishedSupremusDetail:
    pass
root100=RiverTreeBuilder(SupremusRiverConfig(rounding=ChipQuantization.AUTHOR_STRICT)).build(100)
rows['action_quantization']={
    'strict_nonintegral_pot201':'FAIL_CLOSED_PASS',
    'strict_integral_pot100_root_actions':root100.actions,
    'paper_rounding_rule_published':False,
}

# C. Deterministic source-constrained input generation.
cfg25=RiverDatagenV1Config(batch_size=1,dcfr_iterations=25,master_seed=123)
g25=RiverDataGeneratorV1(cfg25)
a=g25.make_batch_inputs(CountingTHRandom(123))
b=g25.make_batch_inputs(CountingTHRandom(123))
assert a.board==b.board and np.array_equal(a.ranges,b.ranges) and np.array_equal(a.pot_half,b.pot_half)
assert np.max(np.abs(a.ranges.sum(axis=2)-1.0)) < 2e-6
assert int(a.masks[0].sum())==1081
assert np.all(a.ranges[:,:,a.masks[0]==0]==0)
input_sha=hashlib.sha256(a.ranges.tobytes()+a.pot_half.tobytes()+bytes(a.board)).hexdigest()
rows['input_generation']={
    'seed':123,'board':a.board,'pot_half':a.pot_half.tolist(),
    'rng_draws_after':a.rng_draws_after,'board_rejections':a.board_rejections,
    'boundary_ties':a.boundary_ties,'sha256':input_sha,
}

# D. Deterministic solve / raw-target schema.
t=time.time(); y1=g25.solve_batch(a); dt=time.time()-t
y2=g25.solve_batch(b)
assert np.array_equal(y1.targets_chips,y2.targets_chips)
assert np.max(np.abs(y1.targets_chips[:,:,a.masks[0]==0]))==0
assert np.max(np.abs(y1.targets_per_pot_half/2-y1.targets_per_total_pot))==0
assert np.max(np.abs(y1.expected_utility_residuals)) < 1e-7
raw_sha=hashlib.sha256(y1.targets_chips.tobytes()).hexdigest()
rows['solve25']={
    'seconds':dt,'raw_target_sha256':raw_sha,
    'zero_sum_residual_chip':y1.expected_utility_residuals.tolist(),
    'decision_nodes':y1.decision_nodes.tolist(),'terminal_nodes':y1.terminal_nodes.tolist(),
    'raw_targets_primary':True,
}

# E. Independent sequence-form LP anchor (restricted private-hand support).
pm=possible_hands_mask(board); live=np.flatnonzero(pm); rng=np.random.default_rng(7)
s0=sorted(rng.choice(live,6,replace=False).tolist()); s1=sorted(rng.choice(live,6,replace=False).tolist())
r0=np.zeros(HAND_COUNT); r1=np.zeros(HAND_COUNT)
r0[s0]=rng.random(6)+.1; r1[s1]=rng.random(6)+.1; r0/=r0.sum(); r1/=r1.sum()
game_cfg=SupremusRiverConfig()
lp=solve_lp(board,100,s0,s1,r0,r1,game_cfg)
t=time.time(); sol=RiverDcfrPlusEngine(board,100,game_cfg,DcfrPlusSpec(iterations=500)).solve(r0,r1); lpdt=time.time()-t
v=float(r0@sol.root_cfvs[0]); gap=abs(v-lp); rel=gap/200.0
assert rel < .005, (lp,v,rel)
rows['lp_anchor_500']={
    'lp_value_chips':lp,'dcfr_value_chips':v,'abs_gap_chips':gap,
    'relative_to_total_pot':rel,'seconds':lpdt,
    'support_each_player':6,
}

# F. Paper/source status, deliberately explicit.
rows['evidence_status']={
    'river_samples_paper':50_000_000,
    'dcfr_iterations_per_player_paper':4_000,
    'dcfr_alpha':1.5,'dcfr_beta':0.0,'delayed_average_d':100,
    'simultaneous_update_for_river_private_code':'UNRESOLVED',
    'integer_action_rounding_private_code':'UNRESOLVED',
    'private_rng_seed_schedule':'UNRESOLVED',
    'river_bucket_artifact':'MISSING',
    'raw_fullcard_target_layer':'IMPLEMENTED',
}
OUT.write_text(json.dumps(rows,indent=2,sort_keys=True,default=lambda x:int(x) if isinstance(x,np.integer) else x))
print(json.dumps(rows,indent=2,sort_keys=True,default=lambda x:int(x) if isinstance(x,np.integer) else x))
print('PASS',OUT)
