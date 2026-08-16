#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hunl.cards import HAND_CARDS, hand_index, possible_hands_mask
from hunl.river_bucket_reconstruction import (
    sample_river_strength_features, fit_river_bucket_reconstruction_v1,
    ReconstructedRiverBucketProvider, river_uniform_equity_numerators,
)
from hunl.value_training import prepare_river_training_batch, loss_on_prepared_batch, train_smoke_steps
from hunl.value_network import DeepStackHUNLValueNet, HUNLValueNetworkSpec

HERE=Path(__file__).resolve().parent
SEED=20260816

# 1) Build project reconstruction bucket artifact.
t0=time.perf_counter()
x,meta=sample_river_strength_features(seed=SEED,boards=256,hands_per_board=512)
feature_seconds=time.perf_counter()-t0
if x.shape != (256*512,): raise AssertionError(x.shape)
t0=time.perf_counter()
art=fit_river_bucket_reconstruction_v1(x,k=1000,seed=SEED,iterations=10)
fit_seconds=time.perf_counter()-t0
art.manifest['training_sample']=meta
ap=HERE/'HUNL_RIVER_BUCKET_RECONSTRUCTION_V1.npz'; art.save(ap)
# Deterministic rebuild in memory.
art2=fit_river_bucket_reconstruction_v1(x,k=1000,seed=SEED,iterations=10)
art2.manifest['training_sample']=meta
assert np.array_equal(art.centroids,art2.centroids)
tmp=HERE/'_river_bucket_rebuild_tmp.npz'; art2.save(tmp)
assert ap.read_bytes()==tmp.read_bytes(); tmp.unlink()
provider=ReconstructedRiverBucketProvider(art)

# 2) Exact river feature/card-space invariants on representative board.
board=(6,1,46,50,0)
nums=river_uniform_equity_numerators(board)
legal=nums>=0
assert int(legal.sum())==1081
assert int(nums[legal].min())>=0 and int(nums[legal].max())<=1980
bm=provider.for_board(board); bm2=provider.for_board(board)
assert np.array_equal(bm.hand_to_bucket,bm2.hand_to_bucket)
assert np.array_equal(bm.legal_mask, possible_hands_mask(board))
# Range mass preservation + value/range duality.
rng=np.random.default_rng(7)
r=rng.random(1326); r[~legal]=0; r/=r.sum()
br=bm.range_to_buckets(r)
assert abs(float(br.sum())-1.0)<1e-12
v=rng.normal(size=1000)
hv=bm.bucket_values_to_hands(v)
dual_err=abs(float(np.dot(br,v))-float(np.dot(r,hv)))
assert dual_err<1e-12

# Global suit permutation invariance of reconstructed assignment.
perm=(1,0,3,2)
def pc(c): return (c//4)*4+perm[c%4]
pb=tuple(pc(c) for c in board); pbm=provider.for_board(pb)
suit_checks=0
for h in np.flatnonzero(legal)[::17]:
    c0,c1=map(int,HAND_CARDS[h]); ph=hand_index(pc(c0),pc(c1))
    if int(bm.hand_to_bucket[h]) != int(pbm.hand_to_bucket[ph]):
        raise AssertionError(('suit bucket mismatch',h,ph))
    suit_checks+=1

# 3) Real 4000-iteration full-card anchor -> bucketed river training row.
anchor=ROOT/'certification/hunl_river_datagen/HUNL_RIVER_RANDOM_4000_ANCHOR.npz'
z=np.load(anchor,allow_pickle=False)
boards=[tuple(int(c) for c in z['board'])]
ranges=np.transpose(z['ranges'],(1,0,2)) # [1,2,1326]
pots=z['pot_half']
targets=z['targets_chips']
batch=prepare_river_training_batch(boards=boards,ranges=ranges,pot_halves=pots,
    targets_chips=targets,bucket_provider=provider)
assert batch.inputs.shape==(1,2001)
assert batch.card_targets.shape==(1,2,1326)
assert abs(float(batch.inputs[0,:1000].sum())-1.0)<2e-6
assert abs(float(batch.inputs[0,1000:2000].sum())-1.0)<2e-6
# Supremus-paper-literal total-pot feature: 602 / 20000.
expected_pot=float(2*int(pots[0]))/20000.0
pot_feature=float(batch.inputs[0,-1])
assert abs(pot_feature-expected_pot)<1e-7

# 4) Full architecture + differentiable smoke over real solved target.
torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
model=DeepStackHUNLValueNet(HUNLValueNetworkSpec())
with torch.no_grad():
    loss0=float(loss_on_prepared_batch(model,batch))
t0=time.perf_counter()
losses=train_smoke_steps(model,batch,steps=40,learning_rate=1e-3)
train_seconds=time.perf_counter()-t0
lossf=losses[-1]
if not (np.isfinite(lossf) and lossf < loss0):
    raise AssertionError((loss0,lossf))
# Weighted zero-sum is enforced in bucket-space by model outer layer.
with torch.no_grad():
    bout=model(batch.inputs)
    rr=batch.inputs[:,:2000]
    weighted=float(torch.sum(bout*rr).cpu())
if abs(weighted)>5e-5:
    raise AssertionError(weighted)

cert={
 'schema':'HUNL_RIVER_VALUE_V1_CERT', 'status':'PASS',
 'bucket_claim':'PROJECT_RECONSTRUCTION_NOT_ORIGINAL',
 'bucket_artifact':ap.name,
 'bucket_artifact_sha256':hashlib.sha256(ap.read_bytes()).hexdigest(),
 'centroid_sha256':art.manifest['centroid_sha256'],
 'bucket_training_samples':int(x.size),
 'bucket_feature':'exact river equity vs uniform legal opponent',
 'feature_generation_seconds':feature_seconds,'bucket_fit_seconds':fit_seconds,
 'centroid_min':float(art.centroids.min()),'centroid_max':float(art.centroids.max()),
 'unique_centroids':int(np.unique(art.centroids).size),
 'provider_deterministic':True,'artifact_byte_reproducible':True,'suit_assignment_checks':suit_checks,
 'range_mass_error':abs(float(br.sum())-1.0),'range_value_duality_error':dual_err,
 'real_anchor_board':list(boards[0]),'real_anchor_pot_half':int(pots[0]),
 'pot_feature_convention':'SUPREMUS_PAPER_LITERAL_TOTAL_POT_DIV_STARTING_STACK',
 'pot_feature':pot_feature,'card_target_normalization':'raw chips / total current pot',
 'network_architecture':'2001 -> 7x500 PReLU -> 2000 + zero-sum correction',
 'network_parameters':model.architecture_parameter_count,
 'training_smoke_steps':len(losses),'training_smoke_seconds':train_seconds,
 'loss_initial':loss0,'loss_final':lossf,'loss_min':min(losses),
 'bucket_weighted_zero_sum_residual_after_training':weighted,
 'source_limits':[
   'private Supremus river bucket artifact unpublished',
   'river scalar-strength k-means is a project reconstruction',
   'private Supremus training weights/initialization and dataset ordering unpublished',
 ],
}
(HERE/'HUNL_RIVER_VALUE_V1_CERT.json').write_text(json.dumps(cert,indent=2,sort_keys=True)+'\n')
# Tiny loss trace for deterministic engineering inspection.
np.savez_compressed(HERE/'HUNL_RIVER_VALUE_V1_SMOKE.npz',loss=np.asarray([loss0]+losses,dtype=np.float32),
                    input=batch.inputs.detach().cpu().numpy(),
                    target=batch.card_targets.detach().cpu().numpy())
print(json.dumps(cert,indent=2,sort_keys=True))
