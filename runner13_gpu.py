"""Phase B campaign: validation-based selection and review-response experiments.

Every run trains on 20k images, reports val accuracy (5k held-out train images,
used for selection) and test accuracy (official 5k, reported once per selected
configuration). Phases:
  V  selection grids on val, widths 128/512, contract vs Adam, 2 seeds
  W  width confirmation 1024/2048 (contract, Adam, fixed-lr spectral), 2 seeds
  D  depth 24/48 (contract, Adam, fixed-lr spectral), 2 seeds
  S  seeds 2-4 on the val-selected headline configurations
  X  variants: exact scaling, no scaling, specnorm only, local SGD momentum
  L  long training, 50 epochs, width 128
  G  global-backprop ceiling on the same benchmark, depths 12 and 48
  C  contract violation percentiles (instrumented runs)
Writes cifar_val.json, checkpointed after every run.
"""
import os, json, time
os.environ['DRIFT_GPU'] = '1'
import numpy
from drift_contract import load_cifar_val, train, accuracy

Xtr, ytr, Xv, yv, Xte, yte = load_cifar_val('cifar_data.npz')
data = (Xtr, ytr, Xv, yv)

Q = []
for w in [128, 512]:
    for eps in [0.0003, 0.001, 0.003, 0.01, 0.03]:
        for s in [0, 1]:
            Q.append((f"V_contract_w{w}_eps{eps}_s{s}",
                      dict(arm='dll_gluon', lr=1e-2, drift_budget=eps,
                           bias_contract='clip', width=w, seed=s)))
    for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        for s in [0, 1]:
            Q.append((f"V_adam_w{w}_lr{lr}_s{s}",
                      dict(arm='dll_adam', lr=lr, width=w, seed=s)))
for w in [1024, 2048]:
    for eps in [0.001, 0.003, 0.01]:
        for s in [0, 1]:
            Q.append((f"W_contract_w{w}_eps{eps}_s{s}",
                      dict(arm='dll_gluon', lr=1e-2, drift_budget=eps,
                           bias_contract='clip', width=w, seed=s)))
    for lr in [1e-4, 3e-4, 1e-3]:
        for s in [0, 1]:
            Q.append((f"W_adam_w{w}_lr{lr}_s{s}",
                      dict(arm='dll_adam', lr=lr, width=w, seed=s)))
    for lr in [1e-3, 3e-3, 1e-2]:
        for s in [0, 1]:
            Q.append((f"W_gfix_w{w}_lr{lr}_s{s}",
                      dict(arm='dll_gluon', lr=lr, width=w, seed=s)))
for d in [24, 48]:
    for eps in [0.001, 0.003, 0.01]:
        for s in [0, 1]:
            Q.append((f"D_contract_d{d}_eps{eps}_s{s}",
                      dict(arm='dll_gluon', lr=1e-2, drift_budget=eps,
                           bias_contract='clip', depth=d, seed=s)))
    for lr in [3e-4, 1e-3, 3e-3]:
        for s in [0, 1]:
            Q.append((f"D_adam_d{d}_lr{lr}_s{s}",
                      dict(arm='dll_adam', lr=lr, depth=d, seed=s)))
    for lr in [1e-3, 3e-3]:
        for s in [0, 1]:
            Q.append((f"D_gfix_d{d}_lr{lr}_s{s}",
                      dict(arm='dll_gluon', lr=lr, depth=d, seed=s)))
for s in [2, 3, 4]:
    for w in [128, 512, 1024]:
        Q.append((f"S_contract_w{w}_eps0.003_s{s}",
                  dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003,
                       bias_contract='clip', width=w, seed=s)))
    for w, lr in [(128, 1e-3), (512, 3e-4), (1024, 3e-4)]:
        Q.append((f"S_adam_w{w}_lr{lr}_s{s}",
                  dict(arm='dll_adam', lr=lr, width=w, seed=s)))
    Q.append((f"S_contract_d48_eps0.003_s{s}",
              dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003,
                   bias_contract='clip', depth=48, seed=s)))
    Q.append((f"S_adam_d48_lr0.0003_s{s}",
              dict(arm='dll_adam', lr=3e-4, depth=48, seed=s)))
for w in [128, 512]:
    for s in [0, 1]:
        Q.append((f"X_exact_w{w}_eps0.003_s{s}",
                  dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003,
                       bias_contract='clip', scale_mode='exact', width=w, seed=s)))
        Q.append((f"X_noscale_w{w}_lr0.003_s{s}",
                  dict(arm='dll_gluon', lr=3e-3, scale_mode='none', width=w, seed=s)))
        Q.append((f"X_specnorm_w{w}_lr0.003_s{s}",
                  dict(arm='dll_gluon', lr=3e-3, gluon_mode='specnorm', width=w, seed=s)))
    for lr in [1e-2, 3e-2, 1e-1]:
        for s in [0, 1]:
            Q.append((f"X_sgdm_w{w}_lr{lr}_s{s}",
                      dict(arm='dll_sgdm', lr=lr, width=w, seed=s)))
for s in [0, 1]:
    Q.append((f"L_contract_w128_eps0.003_s{s}",
              dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003,
                   bias_contract='clip', epochs=50, seed=s)))
    Q.append((f"L_adam_w128_lr0.001_s{s}", dict(arm='dll_adam', lr=1e-3, epochs=50, seed=s)))
    Q.append((f"L_adam_w128_lr0.0003_s{s}", dict(arm='dll_adam', lr=3e-4, epochs=50, seed=s)))
for d in [12, 48]:
    for arm, lrs in [('global_adam', [3e-4, 1e-3]), ('global_muon', [3e-3, 1e-2])]:
        for lr in lrs:
            for s in [0, 1]:
                Q.append((f"G_{arm}_d{d}_lr{lr}_s{s}",
                          dict(arm=arm, lr=lr, depth=d, seed=s)))
for w in [128, 512]:
    Q.append((f"C_check_w{w}_eps0.003_s0",
              dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003,
                   bias_contract='clip', width=w, seed=0, measure_contract=True)))
    Q.append((f"C_checkexact_w{w}_eps0.003_s0",
              dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003, bias_contract='clip',
                   scale_mode='exact', width=w, seed=0, measure_contract=True)))

OUT = 'cifar_val.json'
state = json.load(open(OUT)) if os.path.exists(OUT) else {}
print(f"{len(Q)} runs total, {len(state)} already done", flush=True)
t0 = time.time()
for name, kw in Q:
    if name in state: continue
    arm = kw.pop('arm'); seed = kw.pop('seed')
    epochs = kw.pop('epochs', 12)
    r = train(arm, seed=seed, data=data, epochs=epochs, aux_lr=3e-3,
              return_net=True, **kw)
    test_acc = accuracy(r['net'], Xte, yte) if not r['diverged'] else 0.0
    rec = dict(val=round(r['acc'], 4), test=round(test_acc, 4), diverged=r['diverged'])
    if 'contract' in r:
        rec['contract'] = {k: round(v, 3) for k, v in r['contract'].items()}
    state[name] = rec
    json.dump(state, open(OUT, 'w'))
    print(f"[{time.time()-t0:7.0f}s] {name}: {rec}", flush=True)
print("DONE runner13", flush=True)
