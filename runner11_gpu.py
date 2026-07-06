"""Main campaign, four phases, CIFAR-10 20k/5k, 12 epochs.
B: fixed-lr spectral control across widths (does the fixed lr* already transfer?)
C: width frontier 1024/2048 (contract vs Adam vs fixed-lr spectral)
D: depth transfer 24/48 at width 128
E: seeds 2-4 on the headline configurations (5 seeds total)
Writes cifar_big.json, checkpointed after every run.
"""
import os, json, time
os.environ['DRIFT_GPU'] = '1'
import numpy
from drift_contract import load_cifar, train

Q = []
# --- phase B: fixed-lr spectral control ---
for w in [128, 512, 1024]:
    for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
        for s in [0, 1]:
            Q.append((f"B_gfix_w{w}_lr{lr}_s{s}",
                      dict(arm='dll_gluon', lr=lr, width=w, seed=s)))
# --- phase C: width frontier ---
for w in [1024, 2048]:
    for eps in [0.0003, 0.001, 0.003, 0.01]:
        for s in [0, 1]:
            Q.append((f"C_contrat_w{w}_eps{eps}_s{s}",
                      dict(arm='dll_gluon', lr=1e-2, drift_budget=eps,
                           bias_contract='clip', width=w, seed=s)))
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        for s in [0, 1]:
            Q.append((f"C_adam_w{w}_lr{lr}_s{s}",
                      dict(arm='dll_adam', lr=lr, width=w, seed=s)))
for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
    for s in [0, 1]:
        Q.append((f"C_gfix_w2048_lr{lr}_s{s}",
                  dict(arm='dll_gluon', lr=lr, width=2048, seed=s)))
# --- phase D: depth transfer, width 128 ---
for d in [24, 48]:
    for eps in [0.0003, 0.001, 0.003]:
        for s in [0, 1]:
            Q.append((f"D_contrat_d{d}_eps{eps}_s{s}",
                      dict(arm='dll_gluon', lr=1e-2, drift_budget=eps,
                           bias_contract='clip', depth=d, seed=s)))
    for lr in [3e-4, 1e-3, 3e-3]:
        for s in [0, 1]:
            Q.append((f"D_adam_d{d}_lr{lr}_s{s}",
                      dict(arm='dll_adam', lr=lr, depth=d, seed=s)))
# --- phase E: extra seeds on headline configurations ---
for s in [2, 3, 4]:
    for w in [128, 512]:
        for eps in [0.001, 0.003]:
            Q.append((f"E_contrat_w{w}_eps{eps}_s{s}",
                      dict(arm='dll_gluon', lr=1e-2, drift_budget=eps,
                           bias_contract='clip', width=w, seed=s)))
    Q.append((f"E_contrat_w1024_eps0.001_s{s}",
              dict(arm='dll_gluon', lr=1e-2, drift_budget=0.001,
                   bias_contract='clip', width=1024, seed=s)))
    for w, lr in [(128, 1e-3), (512, 3e-4), (1024, 3e-4)]:
        Q.append((f"E_adam_w{w}_lr{lr}_s{s}",
                  dict(arm='dll_adam', lr=lr, width=w, seed=s)))

OUT = 'cifar_big.json'
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
    print(f"[{time.time()-t0:7.0f}s] {name}: {state[name]}", flush=True)
print("DONE runner11", flush=True)
