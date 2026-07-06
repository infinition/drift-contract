"""Bias ablation: is the drift floor at small eps caused by the Adam bias updates?
Synthetic task, width 128, seed 0, 8 epochs."""
import json, time, os
import numpy as np
from drift_contract import make_data, train

QUEUE = []
for eps in [0.003, 0.01, 0.03]:
    for bc, tag in [(False, 'OFF'), (True, 'ON'), ('clip', 'CLIP')]:
        QUEUE.append((f"eps{eps}_bias{tag}", eps, bc))

OUT = 'bias_ab.json'
state = json.load(open(OUT)) if os.path.exists(OUT) else {}
data = make_data(seed=0)
t0 = time.time()
for name, eps, bc in QUEUE:
    if name in state: continue
    r = train('dll_gluon', lr=1e-2, seed=0, data=data, epochs=8,
              aux_lr=3e-3, drift_budget=eps, bias_contract=bc, probe_every=10)
    dr = np.array(r['drifts'])
    state[name] = dict(acc=round(r['acc'], 4),
                       drift_mean=round(float(dr.mean()), 4),
                       drift_max=round(float(dr.max()), 4))
    json.dump(state, open(OUT, 'w'))
    print(f"[{time.time()-t0:5.0f}s] {name}: {state[name]}", flush=True)
print("DONE runner7_bias")
