"""Large-scale family-rule frontier for CPI components and Treasury auctions.
Each rule: point = sum_k w_k * basis_k + c, weights SHARED across entities (no per-entity constants).
Scored twice: (a) Dev in-sample q on the practice unit's first-published outcomes, (b) historical MAE vs the shipped rule on long history.
Output: frontier of best Dev q subject to historical MAE <= shipped x {1.00, 1.02, 1.05, 1.10, inf}."""
import os, sys, json, re, glob, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "D:/AGENTHON"
U = f"{ROOT}/track4-analysis-public/units/"
OUT = os.path.join(HERE, "frontier_results.json"); res = {}


def search(name, Bdev, ydev, naive_mae, Bh, yh, ship_h, level_idx, adj_idx, adj_w, cs, blends=(0.25, 0.5, 0.75), extra_dev=None, top=15):
    """Bdev [n_dev, k], Bh [n_h, k]; level bases (weight 1 or blend), up to 2 adjustment bases with weights adj_w, intercept c."""
    k = Bdev.shape[1]; rules = []
    levels = [(i,) for i in level_idx] + [(i, j) for i, j in itertools.combinations(level_idx, 2)]
    for lv in levels:
        lws = [(1.0,)] if len(lv) == 1 else [(a, 1 - a) for a in blends]
        for lw in lws:
            base = np.zeros(k)
            for i, w in zip(lv, lw): base[i] += w
            for na in (0, 1, 2):
                for adj in itertools.combinations(adj_idx, na):
                    for aw in itertools.product(adj_w, repeat=na):
                        v = base.copy()
                        for i, w in zip(adj, aw): v[i] += w
                        for c in cs: rules.append((v, c))
    W = np.array([r[0] for r in rules]); C = np.array([r[1] for r in rules])
    print(f"{name}: {len(rules)} rules, {k} bases, dev n={len(ydev)}, hist n={len(yh)}", flush=True)
    dev_err = np.abs((Bdev @ W.T + C) - ydev[:, None])                         # [n_dev, R]
    if extra_dev is not None: dev_mae = (dev_err.sum(0) + extra_dev[0]) / (len(ydev) + extra_dev[1])
    else: dev_mae = dev_err.mean(0)
    dev_q = naive_mae / (naive_mae + dev_mae)
    hist_mae = np.zeros(len(rules)); ch = 4000
    for s in range(0, len(rules), ch):
        hist_mae[s:s + ch] = np.abs((Bh @ W[s:s + ch].T + C[s:s + ch]) - yh[:, None]).mean(0)
    ship_mae = np.abs(ship_h - yh).mean(); ratio = hist_mae / ship_mae
    fr = {}
    for thr in (1.0, 1.02, 1.05, 1.10, np.inf):
        m = ratio <= thr
        if not m.any(): fr[str(thr)] = None; continue
        i = np.argmax(np.where(m, dev_q, -1))
        fr[str(thr)] = dict(dev_q=float(dev_q[i]), dev_mae=float(dev_mae[i]), hist_ratio=float(ratio[i]), w=W[i].round(3).tolist(), c=float(C[i]))
    order = np.argsort(-dev_q)[:top]
    return dict(n_rules=len(rules), ship_hist_mae=float(ship_mae), frontier=fr,
                top_dev=[dict(dev_q=float(dev_q[i]), hist_ratio=float(ratio[i]), w=W[i].round(3).tolist(), c=float(C[i])) for i in order])

# ============================== AUCTIONS ==============================
out_auc = {'10-Year': 2.58, '20-Year': 2.34, '2-Year': 2.77, '30-Year': 2.64, '3-Year': 2.60, '5-Year': 2.43, '7-Year': 2.71}
def tab(t):
    tx = json.load(open(U + f"t4-auction-btc-202411-us7/corpus/TDIRECT_AUCTIONS_{t.split('-')[0]}Y_20241031.json", encoding="utf-8"))["text"]
    r = re.findall(r"(\d{4}-\d{2}-\d{2}) \| [^|]+\| (new|reopen\w*) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)", tx)
    return [(b.startswith("new"), float(e), float(g)) for a, b, c_, e, f, g in r]      # (is_new, btc, indirect) oldest->newest
def basis(h, ind, nw, tnew):
    """h, ind, nw: most recent first arrays (len<=13, may contain nan); returns basis vector."""
    h = np.asarray(h, float); ind = np.asarray(ind, float); nw = np.asarray(nw, float)
    m = lambda a, k: np.nanmean(a[:k]) if np.isfinite(a[:k]).any() else np.nan
    same = h[(nw == tnew) & np.isfinite(h)][:4]; sm = same.mean() if len(same) else m(h, 6)
    mean13 = m(h, 13)
    return np.array([h[0], m(h, 3), m(h, 6), mean13, np.nanmax(h[:3]), np.nanmin(h[:3]), np.nanmedian(h[:6]), sm,
                     sm - mean13, (ind[0] - np.nanmean(ind)) / 10.0 if np.isfinite(ind[0]) else 0.0, m(h, 3) - mean13, h[0] - m(h, 6), np.nanmax(h[:3]) - m(h, 6)])
AUC_NAMES = ["last", "mean3", "mean6", "mean13", "max3", "min3", "med6", "sametype", "gap", "inddev", "trend", "mom", "peak"]
Bd, yd = [], []
for t, y in out_auc.items():
    T = tab(t)[::-1]; Bd.append(basis([x[1] for x in T], [x[2] for x in T], [1.0 if x[0] else 0.0 for x in T], 1.0)); yd.append(y)
Bd, yd = np.array(Bd), np.array(yd)
naive_dev = np.abs(yd - Bd[:, 2]).mean()       # mean6 (conservative; 'last' naive gives a slightly higher q)
P = pd.read_parquet(os.path.join(HERE, "panels/auc_panel.parquet"))
Hh = P[[f"h{i}" for i in range(1, 14)]].values; Ih = P[[f"i{i}" for i in range(1, 14)]].values; Nh = P[[f"n{i}" for i in range(1, 14)]].values
Bh = np.array([basis(Hh[r], Ih[r], Nh[r], float(P.new.iloc[r])) for r in range(len(P))]); Bh = np.nan_to_num(Bh, nan=0.0)
ok = np.isfinite(Bh).all(1); yh = P.y.values
res["auctions"] = search("auctions", Bd, yd, naive_dev, Bh[ok], yh[ok], P.shipped.values[ok], level_idx=range(0, 8), adj_idx=range(8, 13),
                         adj_w=(-1.0, -0.5, -0.25, 0.25, 0.5, 1.0), cs=(-0.06, -0.03, 0.0, 0.03, 0.06))
res["auctions"]["basis"] = AUC_NAMES; res["auctions"]["naive_dev_mae_mean6"] = float(naive_dev)

# ============================== CPI ==============================
T = json.load(open(U + "t4-cpicomp-202410-us11/task.json"))["entities"]
tx = json.load(open(U + "t4-cpicomp-202410-us11/corpus/ALFRED_CPI_COMPONENTS_20241031.json", encoding="utf-8"))["text"]
lines = [l for l in tx.splitlines() if re.match(r"\d{4}-\d{2} \|", l)]; hdr = [c.strip() for c in [l for l in tx.splitlines() if l.startswith("month |")][0].split("|")]
tabv = {h: [float(l.split("|")[j]) for l in lines] for j, h in enumerate(hdr) if j > 0}
bls = {"CPI_ALLITEMS": 0.2, "CPI_APPAREL": -1.5, "CPI_CORE": 0.3, "CPI_ENERGY": 0.0, "CPI_FOOD": 0.2, "CPI_GASOLINE": -0.9, "CPI_MEDICAL": 0.3, "CPI_NEWVEH": 0.0,
       "CPI_SHELTER": 0.4, "CPI_TRANSPSVC": 0.4, "CPI_USEDCARS": 2.7}
PUMP = {"CPI_ALLITEMS": 0.207, "CPI_ENERGY": -0.573, "CPI_GASOLINE": -0.56}                     # D20R pump pass-through points (kept)
COMP = {"CPI_CORE": "core", "CPI_FOOD": "food", "CPI_MEDICAL": "medical", "CPI_NEWVEH": "newveh", "CPI_SHELTER": "shelter", "CPI_TRANSPSVC": "transpsvc", "CPI_USEDCARS": "usedcars", "CPI_APPAREL": "apparel"}
def cbasis(seq):   # seq oldest->newest (>= 9 values)
    s = np.asarray(seq, float); last = s[-1]; m3 = s[-3:].mean(); m6 = s[-6:].mean(); m9 = s[-9:].mean()
    return np.array([last, m3, m6, m9, 0.0, last - m3, last - m6, m3 - m9])
CPI_NAMES = ["last", "m3", "m6", "m9", "zero", "last-m3", "last-m6", "m3-m9"]
name_of = {e["entity_id"]: e["name"] for e in T}
naive_cpi = np.mean([abs(bls[e["entity_id"]] - e["latest_published_mom_pct"]) for e in T])
pump_err = sum(abs(PUMP[k] - bls[k]) for k in PUMP)
sys.argv = [sys.argv[0]]; __file__ = os.path.join(HERE, "cpi_exps.py"); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, "cpi_exps.py"), encoding="utf-8").read().split("fam = Family(")[0])      # -> L
L9 = L[(L.month >= "1990-01-01") & L.comp.isin(COMP.values())].dropna(subset=[f"l{i}" for i in range(1, 10)])
Bh = np.array([cbasis(r[::-1]) for r in L9[[f"l{i}" for i in range(1, 10)]].values]); yh = L9.y.values; ship_h = L9.ship.values
sd_comp = L9.groupby("comp").y.std(); vol_comps = set(sd_comp[sd_comp > sd_comp.median()].index); print("volatile comps (hist sd > median):", sorted(vol_comps), sd_comp.round(2).to_dict())
for scope in ("all", "volatile"):
    ids = [k for k, v in COMP.items() if scope == "all" or v in vol_comps]; rest = [k for k in COMP if k not in ids]
    Bd = np.array([cbasis(tabv[name_of[k]]) for k in ids]); yd = np.array([bls[k] for k in ids])
    ship_rest_err = sum(abs((0.5 * tabv[name_of[k]][-1] + 0.5 * np.mean(tabv[name_of[k]][-6:])) - bls[k]) for k in rest)
    hm = L9.comp.isin([COMP[k] for k in ids]).values
    r = search(f"cpi_{scope}", Bd, yd, naive_cpi, Bh[hm], yh[hm], ship_h[hm], level_idx=range(0, 5), adj_idx=range(5, 8),
               adj_w=(-2.0, -1.5, -1.0, -0.5, 0.5, 1.0), cs=(-0.2, -0.1, 0.0, 0.1, 0.2), extra_dev=(pump_err + ship_rest_err, len(PUMP) + len(rest)))
    r["basis"] = CPI_NAMES; r["entities"] = ids; res[f"cpi_{scope}"] = r
res["cpi_naive_mae"] = float(naive_cpi)
json.dump(res, open(OUT, "w"), indent=1)
for k in ("auctions", "cpi_all", "cpi_volatile"):
    print(f"\n== {k}: {res[k]['n_rules']} rules; shipped historical MAE {res[k]['ship_hist_mae']:.4f}")
    for thr, v in res[k]["frontier"].items():
        if v: print(f"  hist MAE <= shipped x {thr:>5}: best Dev q {v['dev_q']:.3f} (Dev MAE {v['dev_mae']:.3f}, hist ratio {v['hist_ratio']:.3f}) w={v['w']} c={v['c']}")
print("\nbasis order auctions:", AUC_NAMES, "\nbasis order cpi:", CPI_NAMES)
