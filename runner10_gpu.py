"""Bracket completion and epsilon valley floor.
Writes cifar_gpu.json, to merge with cifar_transfer.json."""
import os, json, time
os.environ['DRIFT_GPU'] = '1'
import numpy
from drift_contract import load_cifar, train

QUEUE = [
    ("gluon_w512_eps0.003_s1", dict(arm='dll_gluon', drift_budget=0.003, width=512, seed=1)),
    ("adam_w512_lr0.0001_s1",  dict(arm='dll_adam', lr=1e-4, width=512, seed=1)),
]
for eps in [0.001, 0.0003]:
    for width in [128, 512]:
        for seed in [0, 1]:
            QUEUE.append((f"gluon_w{width}_eps{eps}_s{seed}",
                          dict(arm='dll_gluon', drift_budget=eps, width=width, seed=seed)))

OUT = 'cifar_gpu.json'
state = json.load(open(OUT)) if os.path.exists(OUT) else {}
data = load_cifar('cifar_data.npz', n_train=20000, n_test=5000)
t0 = time.time()
for name, kw in QUEUE:
    if name in state: continue
    arm = kw.pop('arm'); seed = kw.pop('seed')
    kw.setdefault('lr', 1e-2)
    if arm == 'dll_gluon':
        kw['bias_contract'] = 'clip'
    r = train(arm, seed=seed, data=data, epochs=12, aux_lr=3e-3, probe_every=20, **kw)
    dr = numpy.array(r['drifts']) if r['drifts'] else numpy.array([float('nan')])
    state[name] = dict(acc=round(r['acc'], 4), diverged=r['diverged'],
                       drift_mean=round(float(numpy.nanmean(dr)), 4),
                       erank=round(float(numpy.mean(r['eranks'])), 2) if r['eranks'] else None)
    json.dump(state, open(OUT, 'w'))
    print(f"[{time.time()-t0:6.0f}s] {name}: {state[name]}", flush=True)
print("DONE runner10_gpu")
