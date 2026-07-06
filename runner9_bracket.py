"""Brackets the CIFAR optima with eps=0.003 and lr=1e-4 runs.
Is the found optimum interior or a grid edge?"""
import json, time, os
import numpy as np
from drift_contract import load_cifar, train

QUEUE = []
for seed in [0, 1]:
    for width in [128, 512]:
        QUEUE.append((f"gluon_w{width}_eps0.003_s{seed}",
                      dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003,
                           bias_contract='clip', width=width, seed=seed)))
        QUEUE.append((f"adam_w{width}_lr0.0001_s{seed}",
                      dict(arm='dll_adam', lr=1e-4, width=width, seed=seed)))

OUT = 'cifar_transfer.json'
state = json.load(open(OUT)) if os.path.exists(OUT) else {}
data = load_cifar('cifar_data.npz', n_train=20000, n_test=5000)
t0 = time.time()
for name, kw in QUEUE:
    if name in state: continue
    arm = kw.pop('arm'); seed = kw.pop('seed')
    r = train(arm, seed=seed, data=data, epochs=12, aux_lr=3e-3, probe_every=20, **kw)
    dr = np.array(r['drifts']) if r['drifts'] else np.array([np.nan])
    state[name] = dict(acc=round(r['acc'], 4), diverged=r['diverged'],
                       drift_mean=round(float(np.nanmean(dr)), 4),
                       erank=round(float(np.mean(r['eranks'])), 2) if r['eranks'] else None)
    json.dump(state, open(OUT, 'w'))
    print(f"[{time.time()-t0:6.0f}s] {name}: {state[name]}", flush=True)
print("DONE runner9_bracket")
