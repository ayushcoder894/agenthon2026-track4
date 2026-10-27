"""Batch 1 (E18-E21): CPI non-pump components. Data FRED <= 2022-06-28 (cpi_study loaders). Train 2000-2015, test 2016-2022H1.
E18 fitted per-component point a + b*last + c*mean6 (vs current 0.5/0.5, no intercept)
E19 core CPI with cross-component features (shelter mean6, used-car last)
E20 exponential smoothing (halflife 1..6 months) per component
E21 interval multiplier on the 9-month sd per component (IS-optimal k), current = t-PI factor 2.08"""
import numpy as np
import pandas as pd

exec(open("t4-local/dev/cpi_study.py").read().split('print("=== per-component')[0])
COMPS = ["CPI_CORE", "CPI_SHELTER", "CPI_FOOD", "CPI_NEWVEH", "CPI_USEDCARS", "CPI_APPAREL", "CPI_MEDICAL", "CPI_TRANSPSVC"]
tr_m = (M.index.year >= 2000) & (M.index.year <= 2015)
te_m = M.index.year >= 2016


def sk(y, p, naive):
    m = y.notna() & p.notna() & naive.notna()
    a, b = np.abs(y[m] - naive[m]).mean(), np.abs(y[m] - p[m]).mean()
    return a / (a + b)

print("=== E18 fitted a + b*last + c*mean6 | E20 EWMA | current")
for c in COMPS:
    s = M[c]; last = s.shift(1); m6 = s.shift(1).rolling(6).mean()
    cur = 0.5 * last + 0.5 * m6
    d = pd.DataFrame({"y": s, "last": last, "m6": m6}).dropna()
    dtr = d[(d.index.year >= 2000) & (d.index.year <= 2015)]
    X = np.column_stack([np.ones(len(dtr)), dtr["last"], dtr["m6"]])
    b = np.linalg.lstsq(X, dtr.y, rcond=None)[0]
    fit = b[0] + b[1] * last + b[2] * m6
    fit0 = pd.Series(np.nan, index=s.index)
    X0 = np.column_stack([dtr["last"], dtr["m6"]]); b0 = np.linalg.lstsq(X0, dtr.y, rcond=None)[0]
    fit0 = b0[0] * last + b0[1] * m6
    ew = {h: s.shift(1).ewm(halflife=h).mean() for h in (1, 2, 3, 6)}
    line = f"{c:14s} current {sk(s[te_m], cur[te_m], last[te_m]):.3f} | fit(a,b,c)={np.round(b, 2)} {sk(s[te_m], fit[te_m], last[te_m]):.3f} | fit-no-int {np.round(b0, 2)} {sk(s[te_m], fit0[te_m], last[te_m]):.3f} |"
    line += " ".join(f" ewm{h}:{sk(s[te_m], v[te_m], last[te_m]):.3f}" for h, v in ew.items())
    print(line)

print("=== E19 core CPI with cross-component features")
d = pd.DataFrame({"y": M.CPI_CORE, "last": M.CPI_CORE.shift(1), "m6": M.CPI_CORE.shift(1).rolling(6).mean(),
                  "sh6": M.CPI_SHELTER.shift(1).rolling(6).mean(), "uc": M.CPI_USEDCARS.shift(1), "uc3": M.CPI_USEDCARS.shift(1).rolling(3).mean()}).dropna()
dtr, dte = d[(d.index.year >= 2000) & (d.index.year <= 2015)], d[d.index.year >= 2016]
for feats in (["last", "m6"], ["last", "m6", "sh6"], ["last", "m6", "uc"], ["last", "m6", "sh6", "uc3"]):
    X = np.column_stack([np.ones(len(dtr))] + [dtr[f] for f in feats]); b = np.linalg.lstsq(X, dtr.y, rcond=None)[0]
    p = np.column_stack([np.ones(len(dte))] + [dte[f] for f in feats]) @ b
    a = np.abs(dte.y - dte["last"]).mean()
    print(f"  {'+'.join(feats):20s} test skill {a / (a + np.abs(dte.y - p).mean()):.3f}  beta {np.round(b, 3)}")

print("=== E21 interval multiplier on the trailing 9-month sd around the current point (IS, coverage) test 2016+")
for c in COMPS + ["CPI_ALLITEMS"]:
    s = M[c]; last = s.shift(1); m6 = s.shift(1).rolling(6).mean(); cur = 0.5 * last + 0.5 * m6
    sd9 = s.shift(1).rolling(9).std(ddof=0)
    d = pd.DataFrame({"y": s, "p": cur, "sd": sd9}).dropna(); d = d[d.index.year >= 2016]
    res = []
    for k in (1.5, 1.8, 2.08, 2.4, 2.8):
        half = np.maximum(k * d.sd, 0.25)
        IS = np.mean(2 * half + 20 * np.maximum(np.abs(d.y - d.p) - half, 0))
        res.append(f"k={k}:{IS:.3f}/{np.mean(np.abs(d.y - d.p) <= half):.2f}")
    print(f"  {c:14s} " + " ".join(res))
