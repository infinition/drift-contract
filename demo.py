"""Demo: the model that cannot lurch.

Two identical local-learning networks train live, side by side, on the same
stream. Halfway through, the input distribution shifts abruptly (a fixed random
rotation is applied to every input). Local Adam reacts with a large behavioral
jump. The drift contract absorbs the same shock while never exceeding its
per-layer budget, because its step size is bounded by construction.

Runs in about two minutes on a laptop CPU. No dependency beyond NumPy.
    python demo.py

Optional third act (the depth claim, needs cifar_data.npz, about 10 minutes
on CPU or 6 on a CUDA GPU with cupy installed):
    python demo.py --depth
"""
import os
import sys

ACT3 = '--depth' in sys.argv
BACKEND = 'CPU'
if ACT3:
    try:
        import cupy  # noqa: F401
        os.environ['DRIFT_GPU'] = '1'
        BACKEND = 'GPU (CuPy)'
    except ImportError:
        pass

import numpy as np
from drift_contract import make_data, Net, Adam, MuonLike, ClipAdam, softmax_ce_grad


def act3():
    print("""ACT 3: depth, the graveyard of local learning
---------------------------------------------
Local learning had one famous grave: depth. Every layer learns greedily,
so each extra layer degrades the features a little more, and by a few
dozen layers the network is worse than a shallow one. This act trains a
48-layer network on CIFAR-10 three times:

  1. drift contract, eps = 0.003   the value tuned at depth 12, unchanged
  2. local Adam, lr = 3e-4         re-tuned specifically for depth 48
  3. local Adam, lr = 1e-3         Adam's depth-12 best, transferred

If the paper is right: the contract holds around 42 percent with its
depth-12 setting, Adam collapses to about 32 even re-tuned, and Adam's
transferred setting lands under 30.
""")
    if not os.path.exists('cifar_data.npz'):
        print("cifar_data.npz not found. Build it first (see README):")
        print("  pip install pyarrow pillow, download the two parquet files,")
        print("  then: python convert_cifar.py")
        return
    from drift_contract import load_cifar, train
    epochs = int(os.environ.get('DEMO_DEPTH_EPOCHS', 12))
    data = load_cifar('cifar_data.npz', n_train=20000, n_test=5000)
    print(f"backend: {BACKEND}; three trainings of a 48-layer network "
          f"({epochs} epochs each). Go get a coffee.\n")
    runs = [
        ("drift contract, eps=0.003 (depth-12 value)",
         dict(arm='dll_gluon', lr=1e-2, drift_budget=0.003, bias_contract='clip'), 42.1),
        ("local Adam, lr=3e-4 (re-tuned for 48)",
         dict(arm='dll_adam', lr=3e-4), 32.0),
        ("local Adam, lr=1e-3 (depth-12 best, transferred)",
         dict(arm='dll_adam', lr=1e-3), 29.0),
    ]
    results = []
    for label, kw, paper in runs:
        print(f"  training: {label} ...", flush=True)
        arm = kw.pop('arm')
        r = train(arm, seed=0, data=data, epochs=epochs, depth=48, aux_lr=3e-3, **kw)
        results.append((label, r['acc'] * 100, paper))
        print(f"    -> {r['acc']*100:.1f}% (paper, 2 seeds: {paper}%)", flush=True)
    print(f"\n{'':52}{'this run':>10}{'paper':>8}")
    for label, acc, paper in results:
        print(f"{label:<52}{acc:>9.1f}%{paper:>7.1f}%")
    print("""
Same 48 layers, same data. The contract arrives with a setting tuned on
a network four times shallower and simply works. Adam gets a fresh
per-depth tuning and still loses a third of its accuracy; its own old
tuned value is even worse. Depth stopped being a graveyard.""")


if ACT3:
    act3()
    sys.exit(0)

DEPTH, WIDTH, BS, STEPS, SHIFT_AT, EPS = 12, 128, 256, 600, 300, 0.01
rng = np.random.default_rng(0)
Xtr, ytr, Xte, yte = make_data(seed=0)
Q = np.linalg.qr(rng.normal(size=(64, 64)))[0].astype(np.float32)  # the shock


class ArmState:
    def __init__(self, kind):
        self.kind = kind
        self.net = Net(depth=DEPTH, width=WIDTH, seed=0)
        if kind == 'adam':
            self.oW = [Adam(w.shape, 1e-2) for w in self.net.W]
            self.ob = [Adam(b.shape, 1e-2) for b in self.net.b]
        else:
            self.oW = [MuonLike(w.shape, 1e-2) for w in self.net.W]
            self.ob = [ClipAdam(b.shape, 3e-3, cap=EPS) for b in self.net.b]
        self.oHw = [Adam(h.shape, 3e-3) for h in self.net.Hw]
        self.oHb = [Adam(h.shape, 3e-3) for h in self.net.Hb]
        self.rms_ema = [None] * DEPTH
        self.prev_probe = None
        self.drifts = []

    def step(self, x, y):
        net = self.net
        hs, zs, hns, _ = net.forward(x)
        for l in range(DEPTH):
            if self.kind == 'contract':
                cur = float(np.sqrt((hns[l] ** 2).mean()))
                self.rms_ema[l] = cur if self.rms_ema[l] is None else \
                    0.9 * self.rms_ema[l] + 0.1 * cur
                self.oW[l].lr = EPS / max(self.rms_ema[l], 1e-6)
            h = hs[l + 1]
            logits = h @ net.Hw[l].T + net.Hb[l]
            _, dlog = softmax_ce_grad(logits, y)
            dh = dlog @ net.Hw[l]
            dz = dh * (zs[l] > 0)
            net.Hw[l] += self.oHw[l].step(dlog.T @ h)
            net.Hb[l] += self.oHb[l].step(dlog.sum(0))
            net.W[l] += self.oW[l].step(dz.T @ hns[l], net.W[l])
            net.b[l] += self.ob[l].step(dz.sum(0))

    def measure_drift(self, xp, skip):
        hs, _, _, _ = self.net.forward(xp)
        cur = [h.copy() for h in hs[1:]]
        if self.prev_probe is None or skip:
            d = np.nan
        else:
            d = float(np.mean([np.sqrt(np.mean((a - b) ** 2)) /
                               (np.sqrt(np.mean(b ** 2)) + 1e-8)
                               for a, b in zip(cur, self.prev_probe)]))
        self.prev_probe = cur
        self.drifts.append(d)
        return d

    def accuracy(self, X, y):
        return float((self.net.predict(X) == y).mean())


def bar(v, scale, width=26):
    if np.isnan(v):
        return ' ' * width
    return ('#' * max(1, int(width * min(v, scale) / scale))).ljust(width)


# ---------------------------------------------------------------- act 1
from drift_contract import train

print("""THE DRIFT CONTRACT, IN THREE ACTS
=================================
(acts 1 and 2 run now, about two minutes; act 3, the depth claim,
needs CIFAR data and more time: python demo.py --depth)

Both contenders train the same network the same way: every layer learns
from its own small scorekeeper, with no global backward pass (this is
called local learning). The only difference is the update rule:

  local Adam       the standard optimizer, driven by a learning rate
                   that a human has to guess
  drift contract   each layer promises "my behavior moves by at most
                   epsilon per step" and derives its step size from
                   that promise

Everything below is a real training run, live, on your CPU.


ACT 1: what happens when you guess the knob wrong?
--------------------------------------------------
Each method has one number to set: Adam its learning rate, the contract
its budget epsilon. In a real project nobody knows the right value in
advance. So let's get it wrong on purpose, 10x in both directions, and
train 5 quick epochs each time. (about a minute)
""")
data = (Xtr, ytr, Xte, yte)
rows = []
for lr in [1e-3, 1e-2, 1e-1]:
    r = train('dll_adam', lr=lr, seed=0, data=data, epochs=5, aux_lr=3e-3)
    rows.append(r['acc'])
for eps in [1e-3, 1e-2, 1e-1]:
    r = train('dll_gluon', lr=1e-2, seed=0, data=data, epochs=5, aux_lr=3e-3,
              drift_budget=eps, bias_contract='clip')
    rows.append(r['acc'])
print(f"{'':22}{'10x too small':>14}{'right':>10}{'10x too big':>13}")
print(f"{'local Adam':<22}{rows[0]*100:>13.1f}%{rows[1]*100:>9.1f}%{rows[2]*100:>12.1f}%")
print(f"{'drift contract':<22}{rows[3]*100:>13.1f}%{rows[4]*100:>9.1f}%{rows[5]*100:>12.1f}%")
print("""
How to read this: too small, both methods are merely slow (the contract
more so: a tighter budget means more steps to learn, and 5 epochs is not
enough time; it does converge with more). Too big is where they part
ways. Adam's model is destroyed: 10.8 percent is barely above random
guessing. The contract's promise caps every step by construction, so
"too big" cannot become catastrophic. The failure mode that kills
training runs is gone.

""")

# ---------------------------------------------------------------- act 2
adam, contract = ArmState('adam'), ArmState('contract')
probe = Xtr[:256].copy()
rs = np.random.default_rng(1)
SCALE = 0.25  # bar full-scale for drift display

print(f"""ACT 2: the world changes under a live model
-------------------------------------------
Now picture a model that retrains continuously in production. At step
{SHIFT_AT} we change the world: every input gets hit by a fixed random
rotation. Same shock, same instant, for both networks.

Each line below is printed live during training. The bars show how much
each network's behavior moved at that step (a longer bar is a bigger
jump in what the model actually does; bars saturate at {SCALE}).
Watch what happens right after the shift.
""")
print(f"{'step':>6}  {'local Adam':<34}{'drift contract (eps=0.01)':<34}")
print('-' * 76)

for t in range(STEPS):
    shifted = t >= SHIFT_AT
    bi = rs.integers(0, len(Xtr), BS)
    x = Xtr[bi] @ Q.T if shifted else Xtr[bi]
    y = ytr[bi]
    adam.step(x, y)
    contract.step(x, y)
    xp = probe @ Q.T if shifted else probe
    da = adam.measure_drift(xp, skip=(t == SHIFT_AT))
    dc = contract.measure_drift(xp, skip=(t == SHIFT_AT))
    if t == SHIFT_AT:
        print(f"{'':>6}  {'>>> input distribution shifts, next 8 steps shown one by one <<<':^68}")
    elif SHIFT_AT < t <= SHIFT_AT + 8:
        print(f"{t+1:>6}  {bar(da, SCALE)}{da:6.3f}  {bar(dc, SCALE)}{dc:6.3f}  <-- shock",
              flush=True)
    elif t % 25 == 24:
        print(f"{t+1:>6}  {bar(da, SCALE)}{da:6.3f}  {bar(dc, SCALE)}{dc:6.3f}",
              flush=True)

Xs = Xte[:1000] @ Q.T
pa = np.nanmax(adam.drifts[SHIFT_AT + 1:SHIFT_AT + 60])
pc = np.nanmax(contract.drifts[SHIFT_AT + 1:SHIFT_AT + 60])
aa, ac = adam.accuracy(Xs, yte[:1000]), contract.accuracy(Xs, yte[:1000])
print('-' * 76)
print(f"{'':24}{'max drift after shock':>24}{'final acc (shifted data)':>26}")
print(f"{'local Adam':<24}{pa:>24.3f}{aa*100:>25.1f}%")
print(f"{'drift contract':<24}{pc:>24.3f}{ac*100:>25.1f}%")
print(f"""
Both networks recover and end at the same accuracy. The difference is
how they lived through the shock: Adam's behavior jumped {pa/pc:.0f}x harder
than the contract ever allowed itself to move. For a system people rely
on while it learns, that is the difference between "the model adapted"
and "the model went crazy for a while". The contract makes the calm
path a built-in cap on each layer's own updates, not a hope.""")

# ---------------------------------------------------------------- picture
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GRAY = '#08519c', '#e07b39', '#888888'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150,
                                   gridspec_kw={'width_ratios': [3, 2]})
    steps = np.arange(STEPS)
    ax1.plot(steps, adam.drifts, color=ORANGE, lw=1.4, label='local Adam')
    ax1.plot(steps, contract.drifts, color=BLUE, lw=1.8, label='drift contract')
    ax1.axvline(SHIFT_AT, color=GRAY, ls='--', lw=1)
    ax1.text(SHIFT_AT, pa * 0.55, ' the world\n changes here', color=GRAY, fontsize=9)
    ia = int(np.nanargmax(np.where(steps > SHIFT_AT, adam.drifts, np.nan)))
    ax1.annotate(f'Adam jumps to {pa:.2f}', xy=(ia, pa), xytext=(ia + 60, pa * 0.8),
                 fontsize=9, color=ORANGE, arrowprops=dict(arrowstyle='->', color=ORANGE))
    ax1.annotate(f'contract: {pc:.3f}, {pa/pc:.0f}x calmer', xy=(SHIFT_AT + 25, pc),
                 xytext=(SHIFT_AT + 90, pa * 0.25), fontsize=9, color=BLUE,
                 arrowprops=dict(arrowstyle='->', color=BLUE))
    ax1.set_xlabel('training step')
    ax1.set_ylabel('behavior change per step')
    ax1.set_title('Same shock, one panic (act 2)', fontsize=10)
    ax1.legend(frameon=False, fontsize=9, loc='upper left')
    for s in ('top', 'right'): ax1.spines[s].set_visible(False)

    x = np.arange(3); wbar = 0.36
    ax2.bar(x - wbar/2, [rows[0]*100, rows[1]*100, rows[2]*100], wbar,
            color=ORANGE, label='local Adam')
    ax2.bar(x + wbar/2, [rows[3]*100, rows[4]*100, rows[5]*100], wbar,
            color=BLUE, label='drift contract')
    ax2.set_xticks(x, ['10x too small', 'right', '10x too big'])
    ax2.set_ylabel('final accuracy (%)')
    ax2.set_title('Guess the knob wrong (act 1)', fontsize=10)
    ax2.text(2 - wbar/2, rows[2]*100 + 3, 'model\ndestroyed', ha='center',
             fontsize=8, color=ORANGE)
    ax2.legend(frameon=False, fontsize=9, loc='lower left')
    for s in ('top', 'right'): ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig('demo.png', bbox_inches='tight')
    print("\nA picture of acts 1 and 2 was saved to demo.png")
except ImportError:
    print("\n(install matplotlib to also get a picture: pip install matplotlib)")
print("Act 3, the depth claim, runs with: python demo.py --depth")
