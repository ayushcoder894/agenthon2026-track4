"""Demand-composition channel: within-tenor slope of BTC on indirect share (trailing table) x the last auction's indirect-share deviation.
signal = slope_w * (i1 - mean(i)), slope from the tenor's own window (13 auctions). Point = shipped + b x signal (b fitted on train). Variants: shrunk slope, sign-only, pooled slope."""
import os, sys, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import Family
A = pd.read_parquet(os.path.join(HERE, "panels/auc_panel.parquet")).reset_index(drop=True)
H = A[[f"h{i}" for i in range(1, 14)]].values; I = A[[f"i{i}" for i in range(1, 14)]].values
def slope(h, i):
    m = np.isfinite(h) & np.isfinite(i)
    if m.sum() < 6 or np.nanstd(i[m]) == 0: return np.nan
    x, y = i[m] - i[m].mean(), h[m] - h[m].mean(); return float((x * y).sum() / (x * x).sum())
A["slope"] = [slope(H[r], I[r]) for r in range(len(A))]
A["idev"] = I[:, 0] - np.nanmean(I, 1)
A["sig_raw"] = A.slope.fillna(0) * A.idev.fillna(0)
pool = np.nanmedian(A.slope); A["sig_pool"] = pool * A.idev.fillna(0)
A["sig_shr"] = (0.5 * A.slope.fillna(pool) + 0.5 * pool) * A.idev.fillna(0)
A["sig_sign"] = np.sign(A.idev.fillna(0)) * np.sign(A.slope.fillna(0))
print("median within-tenor slope (BTC per indirect-share point):", round(pool, 4), "| corr(next residual, sig_raw):",
      round(float(np.corrcoef(A.sig_raw, A.y - A.shipped)[0, 1]), 3))
def rows(df, w): return df[df.split == w]
def sp(df, w):
    tr = df[df.split == "train"] if w == "val" else df[df.split.isin(["train", "val"])]; return tr, df[df.split == w]
fam = Family("auc_ind"); sc = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).shipped.values)
def add(eid, desc, params, pt):
    f = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); sc[eid] = f; fam.run(eid, desc, params, f, base, A)
def fit_pt(df, w, col, scale):
    tr, te = sp(df, w); x, r = tr[col].values, (tr.y - tr.shipped).values; b = (x * r).sum() / max((x * x).sum(), 1e-12)
    return te.shipped.values + scale * b * te[col].values
for col, scale in itertools.product(("sig_raw", "sig_pool", "sig_shr", "sig_sign", "idev"), (0.5, 1.0)):
    add(f"aucI_{col}_x{scale}", f"shipped + fitted b x {col}, scale {scale}", dict(col=col, scale=scale), lambda df, w, col=col, scale=scale: fit_pt(df, w, col, scale))
fam.finalize(A, sc, base, top_k=3)
# Dev signal values (Nov 2024) from the unit tables
import json, re
U = "D:/AGENTHON/track4-analysis-public/units/t4-auction-btc-202411-us7/corpus/"
for t in ("10", "20", "2", "30", "3", "5", "7"):
    tx = json.load(open(f"{U}TDIRECT_AUCTIONS_{t}Y_20241031.json", encoding="utf-8"))["text"]
    r = re.findall(r"\| (new|reopen\w*) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)", tx)
    h = np.array([float(x[2]) for x in r])[::-1]; i = np.array([float(x[4]) for x in r])[::-1]
    print(f"{t}Y slope {slope(h, i):+.4f} idev {i[0] - i.mean():+.1f} sig_raw {slope(h, i) * (i[0] - i.mean()):+.3f}")
print("done")
