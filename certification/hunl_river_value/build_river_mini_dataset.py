#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from datagen.th_random import THRandom
from hunl_datagen.turn_datagen_v2 import CountingTHRandom
from hunl_datagen.river_datagen_v1 import RiverDataGeneratorV1,RiverDatagenV1Config,RiverDatagenMode

HERE=Path(__file__).resolve().parent
seeds=[124,125,126]
gen=RiverDataGeneratorV1(RiverDatagenV1Config(mode=RiverDatagenMode.PAPER_RECONSTRUCTION,batch_size=1,dcfr_iterations=4000,solver_backend='numba_flat'))
rows=[]
for seed in seeds:
    rng=CountingTHRandom(seed)
    inp=gen.make_batch_inputs(rng)
    t=time.perf_counter();sol=gen.solve_batch(inp);sec=time.perf_counter()-t
    row={
      'seed':seed,'board':list(inp.board),'pot_half':int(inp.pot_half[0]),
      'seconds':sec,'zero_sum':float(sol.expected_utility_residuals[0]),
      'decision_nodes':int(sol.decision_nodes[0]),'terminal_nodes':int(sol.terminal_nodes[0]),
      'target_sha256':hashlib.sha256(sol.targets_chips.astype('<f4').tobytes()).hexdigest(),
    }
    rows.append(row);print(json.dumps(row),flush=True)
    np.savez_compressed(HERE/f'river_4000_seed{seed}.npz',board=np.asarray(inp.board,dtype=np.int16),pot_half=inp.pot_half,
        ranges=np.transpose(inp.ranges,(1,0,2)),targets_chips=sol.targets_chips,masks=inp.masks)
(HERE/'HUNL_RIVER_MINI_DATASET_4000.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
