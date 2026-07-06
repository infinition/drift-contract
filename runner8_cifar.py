"""Contract transfer on a non-saturated task: CIFAR-10 (20k/5k, flattened, MLP).
Contract (biases included) vs local Adam, widths 128 and 512, 12 epochs.
Question: is eps* identical at both widths while Adam's lr* shifts?"""
import json, time, os
import numpy as np
from drift_contract import load_cifar, train

QUEUE = []
for seed in [0, 1]:
    for width in [128, 512]:
        for eps in [0.01, 0.03, 0.1]:
            QUEUE.append((f"gluon_w{width}_eps{eps}_s{seed}",
                          dict(arm='dll_gluon', lr=1e-2, drift_budget=eps,
                               bias_contract='clip', width=width, seed=seed)))
        for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
            QUEUE.append((f"adam_w{width}_lr{lr}_s{seed}",
                          dict(arm='dll_adam', lr=lr, width=width, seed=seed)))

OUT = 'cifar_transfer.json'
state = json.load(open(OUT)) if os.path.exists(OUT) else {}
data = load_cifar('cifar_data.npz', n_train=20000, n_test=5000)
print(f"CIFAR loaded: train={data[0].shape} test={data[2].shape}", flush=True)
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
print("DONE runner8_cifar")
