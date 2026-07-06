"""Turn results/cifar_val.json into the paper tables by validation-based selection.

Selection rule: for a given arm at a given width or depth, pick the hyperparameter
(drift budget or learning rate) that maximizes mean validation accuracy over the
available seeds, then report the test accuracy at that selected setting. Test is
never used for selection. Prints the depth table and the 5-seed width table in
both Markdown and LaTeX, and a completeness report saying which cells are still
waiting on runs.

Run: python analyze_val.py  (reads results/cifar_val.json)
When the campaign is complete, paste the two printed table bodies into
paper/paper.tex and README.md, and remove the "validation rerun" caveat.
"""
import json, os, re, math

PATH = os.path.join("results", "cifar_val.json")
D = json.load(open(PATH)) if os.path.exists(PATH) else {}

def parse(name):
    m = re.match(r"([A-Z])_([a-z]+)_(w|d)(\d+)_(eps|lr)([0-9.]+)_s(\d+)$", name)
    if not m:
        return None
    phase, method, dim, dimv, hp, hpv, seed = m.groups()
    return dict(phase=phase, method=method, dim=dim, dimv=int(dimv),
                hp=hp, hpv=float(hpv), seed=int(seed))

# Headline tables use only the 12-epoch selection/confirmation phases.
# L (50 epochs), X (variants), G (global), C (contract probes) are different
# regimes and must not be mixed into the width/depth headline selection.
HEADLINE_PHASES = {"V", "W", "D", "S"}
rows = []
for name, rec in D.items():
    p = parse(name)
    if p and p["phase"] in HEADLINE_PHASES and not rec.get("diverged", False):
        p.update(val=rec["val"], test=rec["test"])
        rows.append(p)

# The spectral arms (contract, gfix) keep one setting selected once on
# validation and transferred everywhere; only Adam is re-tuned per configuration.
FIXED_HP = {"contract": 0.003, "gfix": 0.003}

def select(method, dim, dimv):
    """Return (best_hp, test_mean, test_std, n_seeds, mean_val) or None.
    Spectral arms use the fixed transferred setting; Adam is validation-selected."""
    cand = [r for r in rows if r["method"] == method and r["dim"] == dim and r["dimv"] == dimv]
    if not cand:
        return None
    by_hp = {}
    for r in cand:
        by_hp.setdefault(r["hpv"], []).append(r)
    if method in FIXED_HP and FIXED_HP[method] in by_hp:
        best_hp = FIXED_HP[method]
        best = sum(x["val"] for x in by_hp[best_hp]) / len(by_hp[best_hp])
    else:
        best_hp, best = None, -1.0
        for hpv, rs in by_hp.items():
            mv = sum(x["val"] for x in rs) / len(rs)
            if mv > best:
                best, best_hp = mv, hpv
    sel = by_hp[best_hp]
    tests = [x["test"] * 100 for x in sel]
    mean = sum(tests) / len(tests)
    std = math.sqrt(sum((t - mean) ** 2 for t in tests) / len(tests)) if len(tests) > 1 else 0.0
    return dict(hp=best_hp, mean=mean, std=std, n=len(sel), val=best * 100)

def cell(sel, with_std=False):
    if sel is None:
        return "PENDING"
    if with_std:
        return f"{sel['mean']:.2f} +/- {sel['std']:.2f}"
    return f"{sel['mean']:.1f}"

# ---------- Width table (5 seeds, headline) ----------
print("=" * 70)
print("WIDTH TABLE (contract vs Adam, val-selected, test reported)")
print("=" * 70)
print(f"{'Width':>6} | {'Contract':>16} | {'Local Adam':>16} | {'Gap':>5} | seeds c/a")
widths = [128, 512, 1024, 2048]
width_rows = []
for w in widths:
    c = select("contract", "w", w)
    a = select("adam", "w", w)
    gap = f"{c['mean']-a['mean']:+.2f}" if c and a else "-"
    nc = c["n"] if c else 0
    na = a["n"] if a else 0
    width_rows.append((w, c, a, gap))
    print(f"{w:>6} | {cell(c, True):>16} | {cell(a, True):>16} | {gap:>5} | {nc}/{na}")

# ---------- Depth table (width 128) ----------
print("\n" + "=" * 70)
print("DEPTH TABLE (width 128, contract / fixed-lr spectral / Adam)")
print("=" * 70)
print(f"{'Depth':>6} | {'Contract':>10} | {'Fixed-lr spec':>13} | {'Local Adam':>12} | seeds")
# depth 12 is the base width-128 model
depth_rows = []
for depth, dim, dimv in [(12, "w", 128), (24, "d", 24), (48, "d", 48)]:
    c = select("contract", dim, dimv)
    g = select("gfix", dim, dimv)
    a = select("adam", dim, dimv)
    depth_rows.append((depth, c, g, a))
    n = "/".join(str(x["n"]) if x else "0" for x in (c, g, a))
    print(f"{depth:>6} | {cell(c):>10} | {cell(g):>13} | {cell(a):>12} | {n}")

# ---------- completeness ----------
print("\n" + "=" * 70)
print("COMPLETENESS")
print("=" * 70)
phases_seen = {}
for r in rows:
    phases_seen[r["phase"]] = phases_seen.get(r["phase"], 0) + 1
print("runs by phase:", dict(sorted(phases_seen.items())))
need = []
if any(w[1] and w[1]["n"] < 5 for w in width_rows):
    need.append("width table wants 5 seeds (phase S) at eps 0.003")
if any(d[0] == 48 and (d[1] is None or d[3] is None) for d in depth_rows):
    need.append("depth 48 not complete (phase D)")
print("pending:", need if need else "none, tables are final")

# ---------- LaTeX bodies (paste into paper/paper.tex) ----------
print("\n" + "=" * 70)
print("LATEX (tab:seeds width, and tab:depth)")
print("=" * 70)
for w, c, a, gap in width_rows:
    cc = f"${c['mean']:.2f} \\pm {c['std']:.2f}$" if c else "PENDING"
    aa = f"${a['mean']:.2f} \\pm {a['std']:.2f}$" if a else "PENDING"
    print(f"{w} & {cc} & {aa} & {gap} \\\\")
print("-" * 30)
for depth, c, g, a in depth_rows:
    print(f"{depth} & {cell(c)} & {cell(g)} & {cell(a)} \\\\")
