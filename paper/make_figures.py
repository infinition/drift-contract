"""Regenerates the paper figures from results/*.json. Run from the repository root:
    python paper/make_figures.py
Outputs paper/figs/fig1_eps_transfer.{pdf,png} and paper/figs/fig2_depth.{pdf,png}.
"""
import json, re, os, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('paper/figs', exist_ok=True)

allr = {}
for f in ['cifar_transfer.json', 'cifar_gpu.json', 'cifar_big.json', 'cifar_big2.json']:
    for k, v in json.load(open(f'results/{f}')).items():
        allr[re.sub(r'^(B|C|D|E2?|Dg)_', '', k).replace('contrat_', 'gluon_')] = v

def series(prefix):
    """-> {value: (mean, std, n)} for keys like prefix + value + _s<seed>"""
    groups = {}
    for k, v in allr.items():
        m = re.match(re.escape(prefix) + r'([0-9.e-]+)_s\d$', k)
        if m: groups.setdefault(float(m.group(1)), []).append(v['acc'])
    return {x: (st.mean(a), st.stdev(a) if len(a) > 1 else 0.0, len(a))
            for x, a in sorted(groups.items())}

plt.rcParams.update({
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False,
})

# ---- Figure 1 : epsilon transfer across widths (sequential ramp, ordered series)
widths = [128, 512, 1024, 2048]
ramp = ['#9ecae1', '#6baed6', '#3182bd', '#08519c']
markers = ['o', 's', '^', 'D']
fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
for w, c, mk in zip(widths, ramp, markers):
    s = series(f'gluon_w{w}_eps')
    xs = list(s); ys = [s[x][0]*100 for x in xs]; es = [s[x][1]*100 for x in xs]
    ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, ms=4, lw=1.6,
                capsize=2, label=f'width {w}')
ax.axvline(0.003, color='#888888', ls=':', lw=1)
ax.text(0.003, 29, r'$\epsilon^{*}=0.003$', ha='center', va='bottom',
        fontsize=8, color='#555555')
ax.set_xscale('log')
ax.set_xlabel(r'drift budget $\epsilon$')
ax.set_ylabel('test accuracy (%)')
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'paper/figs/fig1_eps_transfer.{ext}', bbox_inches='tight')
plt.close(fig)

# ---- Figure 2 : depth (categorical, 3 methods, line styles for grayscale)
depths = [12, 24, 48]
def depth_series(prefix, best_of):
    out = []
    for d in depths:
        cands = []
        for val in best_of:
            accs = [v['acc'] for k, v in allr.items()
                    if re.match(re.escape(prefix.format(d=d, v=val)) + r'_s\d$', k)]
            if accs: cands.append(st.mean(accs))
        out.append(max(cands)*100)
    return out

# depth 12 values come from the width-128 sweeps
d12 = {'contract': 'gluon_w128_eps{v}', 'gfix': 'gfix_w128_lr{v}', 'adam': 'adam_w128_lr{v}'}
def get(prefix12, prefixD, vals):
    row = []
    for d in depths:
        pref = prefix12 if d == 12 else prefixD
        cands = []
        for val in vals:
            key = pref.format(d=d, v=val)
            accs = [v['acc'] for k, v in allr.items() if re.match(re.escape(key) + r'_s\d$', k)]
            if accs: cands.append(st.mean(accs))
        row.append(max(cands)*100)
    return row

contract = get('gluon_w128_eps{v}', 'gluon_d{d}_eps{v}', ['0.003'])
gfix     = get('gfix_w128_lr{v}',   'gfix_d{d}_lr{v}',   ['0.001', '0.003', '0.01'])
adam     = get('adam_w128_lr{v}',   'adam_d{d}_lr{v}',   ['0.0003', '0.001', '0.003'])

fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
ax.plot(depths, contract, color='#08519c', marker='o', ms=5, lw=1.8,
        label=r'drift contract, $\epsilon=0.003$ everywhere')
ax.plot(depths, gfix, color='#3182bd', marker='s', ms=5, lw=1.6, ls='--',
        label='fixed-lr spectral, 3e-3 everywhere')
ax.plot(depths, adam, color='#e07b39', marker='^', ms=5, lw=1.6, ls='-.',
        label='local Adam, re-tuned per depth')
ax.set_xticks(depths)
ax.set_xlabel('network depth (layers)')
ax.set_ylabel('test accuracy (%)')
ax.legend(loc='lower left', fontsize=7.5)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'paper/figs/fig2_depth.{ext}', bbox_inches='tight')
plt.close(fig)
print('figures written:', os.listdir('paper/figs'))
