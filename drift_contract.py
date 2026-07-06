"""The drift contract: spectral updates for local learning.

Single-file library: data, Newton-Schulz, optimizers, model, training loop,
metrics. Four training arms: dll_adam, dll_gluon (local), global_adam,
global_muon (end-to-end backprop).

Backend: NumPy by default; set DRIFT_GPU=1 to run on CuPy. Data generation
and RNG stay on the CPU (the `numpy` module); training tensors move to the
device through np.asarray.
"""
import os
import numpy
if os.environ.get('DRIFT_GPU'):
    import cupy as np
else:
    import numpy as np

# ---------------- data ----------------
def make_data(seed=0, n_train=10000, n_test=2000, d=64, n_classes=10):
    """Hard nonlinear 10-class problem: gaussian mixture warped through a random 2-layer teacher."""
    rng = numpy.random.default_rng(seed)
    n = n_train + n_test
    centers = rng.normal(0, 1.6, size=(n_classes * 3, d))
    y = rng.integers(0, n_classes, size=n)
    cl = y * 3 + rng.integers(0, 3, size=n)
    X = centers[cl] + rng.normal(0, 1.0, size=(n, d))
    W1 = rng.normal(0, 1/numpy.sqrt(d), size=(d, d)); W2 = rng.normal(0, 1/numpy.sqrt(d), size=(d, d))
    X = numpy.tanh(X @ W1) + 0.5 * numpy.sin(X @ W2)
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0)  # train statistics only
    X = (X - mu) / (sd + 1e-8)
    return (X[:n_train].astype(numpy.float32), y[:n_train],
            X[n_train:].astype(numpy.float32), y[n_train:])

def load_cifar(npz_path, n_train=20000, n_test=5000):
    """Flattened CIFAR-10 (3072 features) from cifar_data.npz, normalized with train statistics."""
    d = numpy.load(npz_path)
    X = d['Xtr'][:n_train].astype(numpy.float32) / 255.0
    y = d['ytr'][:n_train]
    Xte = d['Xte'][:n_test].astype(numpy.float32) / 255.0
    yte = d['yte'][:n_test]
    mu, sd = X.mean(0), X.std(0)
    return ((X - mu) / (sd + 1e-8), y,
            ((Xte - mu) / (sd + 1e-8)).astype(numpy.float32), yte)

def load_cifar_val(npz_path, n_train=20000, n_val=5000, n_test=5000):
    """Train / validation / test splits. Validation comes from unused train images
    (the npz is pre-shuffled); everything normalized with train statistics."""
    d = numpy.load(npz_path)
    X = d['Xtr'][:n_train].astype(numpy.float32) / 255.0
    Xv = d['Xtr'][n_train:n_train + n_val].astype(numpy.float32) / 255.0
    Xte = d['Xte'][:n_test].astype(numpy.float32) / 255.0
    mu, sd = X.mean(0), X.std(0)
    nz = lambda A: ((A - mu) / (sd + 1e-8)).astype(numpy.float32)
    return (nz(X), d['ytr'][:n_train], nz(Xv), d['ytr'][n_train:n_train + n_val],
            nz(Xte), d['yte'][:n_test])

# ---------------- newton-schulz ----------------
def newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (np.linalg.norm(G) + eps)
    tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tr else X

# ---------------- optimizers (per-matrix) ----------------
class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.m = np.zeros(shape, np.float32); self.v = np.zeros(shape, np.float32)
        self.lr, self.b1, self.b2, self.eps, self.wd, self.t = lr, b1, b2, eps, wd, 0
    def step(self, g, w=None):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t); vh = self.v / (1 - self.b2 ** self.t)
        upd = -self.lr * mh / (np.sqrt(vh) + self.eps)
        if self.wd and w is not None:
            upd -= self.lr * self.wd * w
        return upd

class MuonLike:
    """Muon-style update, global or local arms: momentum -> NS5 -> spectral-scaled step."""
    def __init__(self, shape, lr, mu=0.95, mode='orth', wd=0.0, scale_mode='max'):
        self.m = np.zeros(shape, np.float32); self.lr, self.mu, self.wd = lr, mu, wd
        if scale_mode == 'exact':      # sqrt(d_out/d_in): tight drift bound for every shape
            self.scale = float(np.sqrt(shape[0] / shape[1]))
        elif scale_mode == 'none':
            self.scale = 1.0
        else:                          # Muon default
            self.scale = float(np.sqrt(max(1.0, shape[0] / shape[1])))
        self.mode = mode  # 'orth' | 'specnorm' (ablation)
    def step(self, g, w=None):
        self.m = self.mu * self.m + g
        if self.mode == 'orth':
            o = newton_schulz5(self.m)
        else:
            s = float(np.linalg.svd(self.m, compute_uv=False)[0])  # spectral norm, CuPy-safe
            o = self.m / (s + 1e-8)
        upd = -self.lr * self.scale * o
        if self.wd and w is not None:
            upd -= self.lr * self.wd * w
        return upd

class SGDm:
    """SGD with heavy-ball momentum, local baseline."""
    def __init__(self, shape, lr, mu=0.9):
        self.m = np.zeros(shape, np.float32); self.lr, self.mu = lr, mu
    def step(self, g, w=None):
        self.m = self.mu * self.m + g
        return -self.lr * self.m

class NormMom:
    """RMS-normalized momentum: every step has RMS exactly lr. Kept for the bias
    ablation; constant-size steps never anneal, ClipAdam works better."""
    def __init__(self, shape, lr, mu=0.95):
        self.m = np.zeros(shape, np.float32); self.lr, self.mu = lr, mu
    def step(self, g, w=None):
        self.m = self.mu * self.m + g
        r = np.sqrt((self.m ** 2).mean()) + 1e-12
        return -self.lr * self.m / r

class ClipAdam(Adam):
    """Adam with update RMS clipped to cap: keeps Adam's annealing, bounds per-step drift."""
    def __init__(self, shape, lr, cap, **kw):
        super().__init__(shape, lr, **kw)
        self.cap = cap
    def step(self, g, w=None):
        upd = super().step(g, w)
        r = float(np.sqrt((upd ** 2).mean()))
        if r > self.cap:
            upd *= self.cap / r
        return upd

# ---------------- model ----------------
def he(rng, dout, din): return rng.normal(0, numpy.sqrt(2.0 / din), size=(dout, din)).astype(numpy.float32)

def softmax_ce_grad(logits, y):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z); p = e / e.sum(1, keepdims=True)
    B = len(y)
    loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
    g = p.copy(); g[np.arange(B), y] -= 1.0
    return loss, g / B

class Net:
    def __init__(self, depth=12, width=128, d_in=64, n_cls=10, seed=0, use_norm=False):
        rng = numpy.random.default_rng(seed)
        dims = [d_in] + [width] * depth
        self.W = [np.asarray(he(rng, dims[i+1], dims[i])) for i in range(depth)]
        self.b = [np.zeros(dims[i+1], np.float32) for i in range(depth)]
        self.Hw = [np.asarray(he(rng, n_cls, width)) for _ in range(depth)]
        self.Hb = [np.zeros(n_cls, np.float32) for _ in range(depth)]
        self.depth, self.width, self.use_norm = depth, width, use_norm

    def forward(self, x):
        """Returns hs (post-ReLU activations, hs[0]=x), zs (pre-activations),
        hns (effective input of each layer, RMS-normalized if use_norm),
        rs (per-row RMS, None without normalization)."""
        hs, zs, hns, rs = [np.asarray(x)], [], [], []
        for W, b in zip(self.W, self.b):
            h = hs[-1]
            if self.use_norm:
                r = np.sqrt((h.astype(np.float64) ** 2).mean(1, keepdims=True) + 1e-6).astype(h.dtype)
                hn = h / r
            else:
                hn, r = h, None
            hns.append(hn); rs.append(r)
            z = hn @ W.T + b; zs.append(z); hs.append(np.maximum(z, 0))
        return hs, zs, hns, rs

    def predict(self, x):
        hs, _, _, _ = self.forward(x)
        return (hs[-1] @ self.Hw[-1].T + self.Hb[-1]).argmax(1)

def accuracy(net, X, y, bs=1000):
    correct = 0
    for i in range(0, len(X), bs):
        correct += float((net.predict(X[i:i+bs]) == np.asarray(y[i:i+bs])).sum())
    return correct / len(X)

# ---------------- metrics ----------------
def effective_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    p = (s ** 2) / (s ** 2).sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))

# ---------------- training ----------------
def train(arm, lr, seed=0, depth=12, epochs=10, bs=256, data=None,
          probe_every=0, spectrum_layer=None, gluon_mode='orth',
          aux_lr=None, use_norm=False, wd=0.0, width=128, drift_budget=None,
          bias_contract=False, scale_mode='max', return_net=False,
          measure_contract=False):
    """arm in {'dll_adam','dll_gluon','global_adam','global_muon'}.
    aux_lr: if set, biases and heads run Adam(aux_lr) in every arm.
    drift_budget: drift contract (spectral arms), lr_l = eps / RMS_ema(layer input).
    bias_contract: 'clip' (Adam clipped to eps, recommended), 'normmom',
    'freeze', or False. True is an alias of 'normmom'."""
    Xtr, ytr, Xte, yte = data
    Xtr = np.asarray(Xtr); ytr_d = np.asarray(ytr)  # dataset stays on the device
    net = Net(depth=depth, width=width, d_in=Xtr.shape[1], seed=seed, use_norm=use_norm)
    local = arm.startswith('dll')
    geom = ('sgdm' if 'sgdm' in arm else
            'gluon' if 'gluon' in arm or 'muon' in arm else 'adam')

    if geom == 'adam':
        optW = [Adam(w.shape, lr, wd=wd) for w in net.W]
    elif geom == 'sgdm':
        optW = [SGDm(w.shape, lr) for w in net.W]
    else:
        optW = [MuonLike(w.shape, lr, mode=gluon_mode, wd=wd, scale_mode=scale_mode)
                for w in net.W]
    a_lr = aux_lr if aux_lr is not None else (lr if geom in ('adam', 'sgdm') else 3e-3)
    if bias_contract and drift_budget is not None and geom == 'gluon':
        if bias_contract == 'clip':
            optb = [ClipAdam(b.shape, a_lr, cap=drift_budget) for b in net.b]
        elif bias_contract == 'freeze':
            optb = [Adam(b.shape, 0.0) for b in net.b]
        else:
            optb = [NormMom(b.shape, drift_budget) for b in net.b]
    else:
        optb = [Adam(b.shape, a_lr) for b in net.b]
    optHw = [Adam(h.shape, a_lr) for h in net.Hw]
    optHb = [Adam(h.shape, a_lr) for h in net.Hb]

    rng = numpy.random.default_rng(seed + 1)
    n = len(Xtr); steps_per = n // bs
    probe = np.asarray(Xtr[:256]); prev_probe_h = None; drifts = []
    spectra = []; step_count = 0
    total_steps = epochs * steps_per
    rms_ema = [None] * depth  # drift contract state
    contract_ratios = []
    lr_hist = []

    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(steps_per):
            bi = np.asarray(idx[s*bs:(s+1)*bs])
            x, y = Xtr[bi], ytr_d[bi]
            hs, zs, hns, rs = net.forward(x)

            if drift_budget is not None and geom == 'gluon':
                for l in range(depth):
                    cur = float(np.sqrt((hns[l] ** 2).mean()))
                    rms_ema[l] = cur if rms_ema[l] is None else 0.9 * rms_ema[l] + 0.1 * cur
                    optW[l].lr = drift_budget / max(rms_ema[l], 1e-6)
                lr_hist.append(optW[depth//2].lr)

            if local:
                for l in range(depth):
                    h = hs[l+1]
                    logits = h @ net.Hw[l].T + net.Hb[l]
                    _, dlog = softmax_ce_grad(logits, y)
                    dHw = dlog.T @ h; dHb = dlog.sum(0)
                    dh = dlog @ net.Hw[l]
                    dz = dh * (zs[l] > 0)
                    dW = dz.T @ hns[l]; db = dz.sum(0)
                    if spectrum_layer == l and step_count in (total_steps//4, total_steps//2, 3*total_steps//4):
                        spectra.append(np.linalg.svd(dW, compute_uv=False))
                    net.Hw[l] += optHw[l].step(dHw); net.Hb[l] += optHb[l].step(dHb)
                    updW = optW[l].step(dW, net.W[l])
                    if measure_contract and drift_budget is not None:
                        dz_r = float(np.sqrt(((hns[l] @ updW.T) ** 2).mean())) / drift_budget
                        contract_ratios.append(dz_r)
                    net.W[l] += updW; net.b[l] += optb[l].step(db)
            else:
                logits = hs[-1] @ net.Hw[-1].T + net.Hb[-1]
                _, dlog = softmax_ce_grad(logits, y)
                dHw = dlog.T @ hs[-1]; dHb = dlog.sum(0)
                dh = dlog @ net.Hw[-1]
                gW, gb = [None]*depth, [None]*depth
                for l in range(depth-1, -1, -1):
                    dz = dh * (zs[l] > 0)
                    gW[l] = dz.T @ hns[l]; gb[l] = dz.sum(0)
                    if l > 0:
                        dhn = dz @ net.W[l]
                        if net.use_norm:
                            hn = hns[l]
                            dh = (dhn - hn * (dhn * hn).mean(1, keepdims=True)) / rs[l]
                        else:
                            dh = dhn
                if spectrum_layer is not None and step_count in (total_steps//4, total_steps//2, 3*total_steps//4):
                    spectra.append(np.linalg.svd(gW[spectrum_layer], compute_uv=False))
                net.Hw[-1] += optHw[-1].step(dHw); net.Hb[-1] += optHb[-1].step(dHb)
                for l in range(depth):
                    net.W[l] += optW[l].step(gW[l], net.W[l]); net.b[l] += optb[l].step(gb[l])

            if probe_every and step_count % probe_every == 0:
                ph, _, _, _ = net.forward(probe)
                cur = [h.copy() for h in ph[1:]]
                if prev_probe_h is not None:
                    vals = [float(np.sqrt(np.mean((a.astype(np.float64)-b)**2)) /
                                  (np.sqrt(np.mean(b.astype(np.float64)**2)) + 1e-8))
                            for a, b in zip(cur, prev_probe_h)]
                    drifts.append(sum(vals) / len(vals))
                prev_probe_h = cur
            step_count += 1
            if not (np.isfinite(net.W[0]).all() and np.isfinite(net.W[-1]).all()
                    and np.isfinite(hs[-1]).all()):
                return dict(acc=0.0, diverged=True, drifts=drifts, spectra=spectra,
                            eranks=[], lr_hist=lr_hist)

    acc = accuracy(net, Xte, yte)
    ph, _, _, _ = net.forward(probe)
    if not np.isfinite(ph[-1]).all():
        return dict(acc=float(acc), diverged=True, drifts=drifts, spectra=spectra,
                    eranks=[], lr_hist=lr_hist)
    eranks = [effective_rank(h) for h in ph[1:]]
    out = dict(acc=float(acc), diverged=False, drifts=drifts, spectra=spectra,
               eranks=eranks, lr_hist=lr_hist)
    if measure_contract and contract_ratios:
        cr = numpy.array(contract_ratios)
        out['contract'] = dict(max=float(cr.max()), p99=float(numpy.percentile(cr, 99)),
                               p95=float(numpy.percentile(cr, 95)), mean=float(cr.mean()))
    if return_net:
        out['net'] = net
    return out
