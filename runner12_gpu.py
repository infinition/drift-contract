"""Depth control and remaining seeds. (1) Fixed-lr spectral at depth 24/48: is the depth
robustness due to the contract or to the geometry alone? (2) 5 seeds at widths 1024/2048.
Writes cifar_big2.json."""
import os, json, time
os.environ['DRIFT_GPU'] = '1'
import numpy
from drift_contract import load_cifar, train

Q = []
for d in [24, 48]:
    for lr in [1e-3, 3e-3, 1e-2]:
        for s in [0, 1]:
            Q.append((f"Dg_gfix_d{d}_lr{lr}_s{s}",
                      dict(arm='dll_gluon', lr=lr, depth=d, seed=s)))
for s in [2, 3, 4]:
    for w in [1024, 2048]:
        Q.append((f"E2_contrat_w{w}_eps0.003_s{s}",
                  dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003,
                       bias_contract='clip', width=w, seed=s)))
    Q.append((f"E2_adam_w2048_lr0.0003_s{s}",
              dict(arm='dll_adam', lr=3e-4, width=2048, seed=s)))

OUT = 'cifar_big2.json'
state = json.load(open(OUT)) if os.path.exists(OUT) else {}
data = load_cifar('cifar_data.npz', n_train=20000, n_test=5000)
print(f"{len(Q)} runs total, {len(state)} already done", flush=True)
t0 = time.time()
for name, kw in Q:
    if name in state: continue
    arm = kw.pop('arm'); seed = kw.pop('seed')
    r = train(arm, seed=seed, data=data, epochs=12, aux_lr=3e-3, probe_every=20, **kw)
    dr = numpy.array(r['drifts']) if r['drifts'] else numpy.array([float('nan')])
    state[name] = dict(acc=round(r['acc'], 4), diverged=r['diverged'],
                       drift_mean=round(float(numpy.nanmean(dr)), 4),
                       erank=round(float(numpy.mean(r['eranks'])), 2) if r['eranks'] else None)
    json.dump(state, open(OUT, 'w'))
    print(f"[{time.time()-t0:6.0f}s] {name}: {state[name]}", flush=True)
print("DONE runner12", flush=True)
