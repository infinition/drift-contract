"""Stability sweep, no trunk normalization, auxiliary lr fixed at 3e-3 in every arm, 2 seeds."""
import json, time, os
from drift_contract import make_data, train

grids = {
 'dll_adam':    [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
 'dll_gluon':   [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
 'global_adam': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
 'global_muon': [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
}
OUT = 'fair_sweep.json'
state = json.load(open(OUT)) if os.path.exists(OUT) else {}
data = make_data(seed=0)
t0 = time.time()
for arm, lrs in grids.items():
    for lr in lrs:
        k = f"{arm}@{lr}"
        if k in state: continue
        accs, div = [], []
        for s in [0, 1]:
            r = train(arm, lr=lr, seed=s, data=data, epochs=10, aux_lr=3e-3)
            accs.append(round(r['acc'], 4)); div.append(r['diverged'])
        state[k] = dict(acc=accs, diverged=div)
        json.dump(state, open(OUT, 'w'))
        print(f"[{time.time()-t0:6.0f}s] {k}: acc={accs} div={div}", flush=True)
print("DONE runner4_fair")
