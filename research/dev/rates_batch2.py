"""Experiments 35-40 (rates bands). FRED DGS2..DGS30 daily, origins every 5 trading days 2000..2022-05, horizons 35 and 38 trading days.
Shipped: half = 1.645 x (0.5 x sd10d + 0.5 x 6.5bp) x sqrt(h)  (sd10d = std of the last 10 daily changes in bp).
35 window length for the own sd (5/10/20/40/60)      36 blend weight on the 6.5bp prior (0..1)       37 EWMA vol (halflife 5/10/20)
38 common (cross-tenor mean) sd instead of own        39 yield-level dependence (sd scaled by level/3 for level < 1.5%)
40 band asymmetry after easing vs tightening (policy-rate change over the previous 63 days)"""
import json
import numpy as np
import pandas as pd

def load(s):
    d = json.load(open(f"data_fred/{s}.json"))["observations"]
    return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})
MATS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MATS}).dropna(); Y = Y[Y.index >= "1999-06-01"]
tgt = load("DFEDTAR").dropna(); mid = ((load("DFEDTARU") + load("DFEDTARL")) / 2).dropna()
pol = pd.concat([tgt[tgt.index < mid.index.min()], mid]).sort_index().reindex(Y.index, method="ffill")
dY = Y.diff() * 100
rows = []
for H in (35, 38):
    for i in range(80, len(Y) - H, 5):
        t = Y.index[i]
        cross = np.mean([dY[m].iloc[i - 9:i + 1].std() for m in MATS])
        for m in MATS:
            d = dY[m]
            rows.append(dict(H=H, t=t, year=t.year, m=m, y=(Y[m].iloc[i + H] - Y[m].iloc[i]) * 100, lvl=Y[m].iloc[i],
                             sd5=d.iloc[i - 4:i + 1].std(), sd10=d.iloc[i - 9:i + 1].std(), sd20=d.iloc[i - 19:i + 1].std(), sd40=d.iloc[i - 39:i + 1].std(), sd60=d.iloc[i - 59:i + 1].std(),
                             ew5=np.sqrt((d.iloc[:i + 1].tail(80) ** 2).ewm(halflife=5).mean().iloc[-1]), ew10=np.sqrt((d.iloc[:i + 1].tail(120) ** 2).ewm(halflife=10).mean().iloc[-1]),
                             ew20=np.sqrt((d.iloc[:i + 1].tail(160) ** 2).ewm(halflife=20).mean().iloc[-1]), cross=cross,
                             dpol=pol.iloc[i] - pol.iloc[i - 63]))
R = pd.DataFrame(rows)
IS = lambda lo, hi, e: np.mean((hi - lo) + 20 * np.maximum(lo - e, 0) + 20 * np.maximum(e - hi, 0))
def run(sd, name, z=1.645, df=R):
    half = z * sd * np.sqrt(df.H)
    a = IS(-half[df.year <= 2012], half[df.year <= 2012], df.y[df.year <= 2012]); b = IS(-half[df.year >= 2013], half[df.year >= 2013], df.y[df.year >= 2013])
    ca = np.mean(np.abs(df.y[df.year <= 2012]) <= half[df.year <= 2012]); cb = np.mean(np.abs(df.y[df.year >= 2013]) <= half[df.year >= 2013])
    print(f"   {name:34s} IS 2000-12 {a:7.1f} (cov {ca:.2f}) | 2013-22 {b:7.1f} (cov {cb:.2f})")
print("--- shipped and 35/36 window and prior weights")
run(0.5 * R.sd10 + 0.5 * 6.5, "shipped 0.5*sd10+0.5*6.5")
for w in ("sd5", "sd10", "sd20", "sd40", "sd60"):
    run(0.5 * R[w] + 0.5 * 6.5, f"35 window {w} blend .5")
for a in (0.0, 0.25, 0.75, 1.0):
    run(a * R.sd10 + (1 - a) * 6.5, f"36 own weight {a}")
print("--- 37 EWMA")
for w in ("ew5", "ew10", "ew20"):
    run(0.5 * R[w] + 0.5 * 6.5, f"37 {w} blend .5")
print("--- 38 cross-tenor sd")
run(0.5 * R.cross + 0.5 * 6.5, "38 cross-tenor sd10 blend .5")
run(0.25 * R.sd10 + 0.25 * R.cross + 0.5 * 6.5, "38 .25 own + .25 cross + .5 prior")
print("--- 39 low-level scaling")
run((0.5 * R.sd10 + 0.5 * 6.5) * np.where(R.lvl < 1.5, R.lvl / 3 + 0.5, 1.0), "39 level<1.5% shrink")
print("--- 40 asymmetry after easing (dpol < -0.1) vs tightening (> 0.1): empirical y quantiles / sd-scaled")
base = (0.5 * R.sd10 + 0.5 * 6.5) * np.sqrt(R.H)
R["u"] = R.y / base
for nm, mask in (("easing", R.dpol < -0.1), ("tightening", R.dpol > 0.1), ("neutral", R.dpol.abs() <= 0.1)):
    g = R[mask]; print(f"   {nm:10s} n={len(g)} u quantiles 5/50/95: {np.percentile(g.u, [5, 50, 95]).round(2)}  (pre-2013 {np.percentile(g.u[g.year <= 2012], [5, 50, 95]).round(2)} | 2013+ {np.percentile(g.u[g.year >= 2013], [5, 50, 95]).round(2)})")
