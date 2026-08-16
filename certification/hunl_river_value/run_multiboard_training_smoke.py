#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from hunl.river_bucket_reconstruction import ReconstructedRiverBucketProvider
from hunl.value_training import prepare_river_training_batch,loss_on_prepared_batch,train_smoke_steps,forward_card_values
from hunl.value_network import DeepStackHUNLValueNet,HUNLValueNetworkSpec
HERE=Path(__file__).resolve().parent
provider=ReconstructedRiverBucketProvider.from_file(HERE/'HUNL_RIVER_BUCKET_RECONSTRUCTION_V1.npz')
paths=[
 ROOT/'certification/hunl_river_datagen/HUNL_RIVER_RANDOM_4000_ANCHOR.npz',
 HERE/'river_4000_seed124.npz',
 HERE/'river_4000_seed126.npz',
]
boards=[];pots=[];ranges=[];targets=[];source_hashes=[]
for p in paths:
 z=np.load(p,allow_pickle=False); b=tuple(int(c) for c in z['board']);boards.append(b);pots.append(int(z['pot_half'][0]));
 rr=z['ranges'];
 if rr.shape==(2,1,1326): rr=np.transpose(rr,(1,0,2))
 if rr.shape!=(1,2,1326): raise AssertionError((p,rr.shape))
 ranges.append(rr[0]);targets.append(z['targets_chips'][0]);source_hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
ranges=np.stack(ranges);targets=np.stack(targets);pots=np.asarray(pots,dtype=np.int32)
batch=prepare_river_training_batch(boards=boards,ranges=ranges,pot_halves=pots,targets_chips=targets,bucket_provider=provider)
# Per-board bucket occupancy and feature checks.
occupancy=[]
for bm in batch.maps:
 ids=bm.hand_to_bucket[bm.hand_to_bucket>=0]
 occupancy.append({'used_buckets':int(np.unique(ids).size),'legal_hands':int(ids.size)})
assert all(x['legal_hands']==1081 for x in occupancy)
assert torch.allclose(batch.inputs[:,:1000].sum(1),torch.ones(3),rtol=0,atol=2e-6)
assert torch.allclose(batch.inputs[:,1000:2000].sum(1),torch.ones(3),rtol=0,atol=2e-6)
# deterministic smoke
SEED=20260816
torch.manual_seed(SEED); torch.set_num_threads(1)
model=DeepStackHUNLValueNet(HUNLValueNetworkSpec())
with torch.no_grad():
 l0=float(loss_on_prepared_batch(model,batch)); out0=forward_card_values(model,batch)
t=time.perf_counter();losses=train_smoke_steps(model,batch,steps=120,learning_rate=1e-3);sec=time.perf_counter()-t
with torch.no_grad():
 lf=float(loss_on_prepared_batch(model,batch)); card=forward_card_values(model,batch); bout=model(batch.inputs); residual=(bout*batch.inputs[:,:2000]).sum(1).abs().cpu().numpy()
# per-sample masked Huber after training
per=[]
for i in range(3):
 m=batch.legal_mask[i]
 a=card[i,:,m]; y=batch.card_targets[i,:,m]
 li=float(torch.nn.functional.smooth_l1_loss(a,y,beta=1.0,reduction='mean'))
 per.append(li)
if not lf<l0: raise AssertionError((l0,lf))
# Save clearly-labelled non-production checkpoint.
ckpt=HERE/'HUNL_RIVER_CFVNET_3SAMPLE_SMOKE.pt'
torch.save({'claim':'ENGINEERING_SMOKE_NOT_PRODUCTION_MODEL','seed':SEED,'state_dict':model.state_dict(),
            'bucket_artifact_sha256':hashlib.sha256((HERE/'HUNL_RIVER_BUCKET_RECONSTRUCTION_V1.npz').read_bytes()).hexdigest(),
            'source_npz_sha256':source_hashes},ckpt)
cert={'schema':'HUNL_RIVER_CFVNET_MULTIBOARD_SMOKE_V1','status':'PASS','claim':'ENGINEERING_SMOKE_NOT_PRODUCTION_MODEL',
 'samples':3,'boards':[list(b) for b in boards],'pot_halves':pots.tolist(),'source_npz_sha256':source_hashes,
 'bucket_occupancy':occupancy,'training_steps':120,'training_seconds':sec,'loss_initial':l0,'loss_final':lf,
 'loss_min':min(losses),'per_sample_final_loss':per,'weighted_zero_sum_residual_abs':residual.tolist(),
 'checkpoint':ckpt.name,'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()}
(HERE/'HUNL_RIVER_CFVNET_MULTIBOARD_SMOKE_V1.json').write_text(json.dumps(cert,indent=2,sort_keys=True)+'\n')
# combined raw mini dataset for reuse, still full-card and independent of bucket artifact
mini=HERE/'HUNL_RIVER_FULLCARD_MINISET_3x4000.npz'
np.savez_compressed(mini,boards=np.asarray(boards,dtype=np.int16),pot_half=pots,ranges=ranges.astype(np.float32),targets_chips=targets.astype(np.float32))
cert['miniset']=mini.name;cert['miniset_sha256']=hashlib.sha256(mini.read_bytes()).hexdigest()
(HERE/'HUNL_RIVER_CFVNET_MULTIBOARD_SMOKE_V1.json').write_text(json.dumps(cert,indent=2,sort_keys=True)+'\n')
print(json.dumps(cert,indent=2,sort_keys=True))
