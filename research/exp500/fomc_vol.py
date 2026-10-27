"""FOMC inter-meeting yield change: interval width from RECENT realized volatility (volatility clustering) vs the shipped band.
Point = 0 (no change) for every rule. Score = mean interval score (alpha 0.1, the scorer's own) over 6 maturities, per meeting.
Shipped: half = 1.645 x sd_day x sqrt(n), sd_day = 0.5 x sd(last 10 daily changes of that maturity) + 0.5 x 6.5bp (no table -> 6.5).
Candidates: sd_day from the last W trading days (W = 10, 20, 60), EWMA(lambda 0.94), and a blend with the long-run prior; z fixed at 1.645
(no fitted multiplier), plus a variant with the z that minimised IS on 1995-2012 meetings only."""
import json, numpy as np, pandas as pd
ROOT = "D:/AGENTHON"
def load(s):
    d = json.load(open(f"{ROOT}/data_fred/{s}.json"))["observations"]
    return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})
MATS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MATS}).dropna(); Y = Y[Y.index >= "1990-01-01"]
print("yields", Y.index.min().date(), Y.index.max().date())
dates = sorted(pd.Timestamp(x["date"]) for x in map(json.loads, open(f"{ROOT}/data_fomc/statements.jsonl", encoding="utf-8")))
D = Y.diff() * 100
def IS(lo, hi, y): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
rows = []
for i in range(len(dates) - 1):
    d0, d1 = dates[i], dates[i + 1]; p0 = Y.index.searchsorted(d0, side="right"); p1 = Y.index.searchsorted(d1, side="left") - 1
    if p1 >= len(Y) or p1 <= p0 + 3 or p0 < 130: continue
    n = p1 - p0
    for m in MATS:
        y = (Y[m].iloc[p1] - Y[m].iloc[p0]) * 100; hist = D[m].iloc[p0 - 120:p0 + 1].dropna().values
        r = dict(d0=d0, mat=m, n=n, y=y)
        r["sd10"], r["sd20"], r["sd60"] = hist[-10:].std(), hist[-20:].std(), hist[-60:].std()
        w = 0.94 ** np.arange(len(hist))[::-1]; r["ewma"] = np.sqrt((w * hist ** 2).sum() / w.sum())
        r["m30"] = abs((Y[m].iloc[p0] - Y[m].iloc[p0 - 30]) * 100) / np.sqrt(30)
        rows.append(r)
R = pd.DataFrame(rows); R["year"] = R.d0.dt.year
print("meeting x maturity rows", len(R), "| meetings", R.d0.nunique(), R.d0.min().date(), "->", R.d0.max().date())
rules = {
    "shipped(0.5 sd10 + 0.5 x 6.5)": 0.5 * R.sd10 + 0.5 * 6.5,
    "prior 6.5 only (no table)": pd.Series(6.5, index=R.index),
    "sd10": R.sd10, "sd20": R.sd20, "sd60": R.sd60, "ewma0.94": R.ewma,
    "0.5 sd60 + 0.5 x 6.5": 0.5 * R.sd60 + 0.5 * 6.5,
    "max(6.5, sd60)": np.maximum(6.5, R.sd60),
    "|30d move|/sqrt30 blend": 0.5 * R.m30 + 0.5 * 6.5,
}
out = []
for nm, sd in rules.items():
    half = 1.645 * sd * np.sqrt(R.n); s = IS(-half, half, R.y)
    out.append((nm, s.mean(), s[R.year <= 2012].mean(), s[(R.year > 2012) & (R.year <= 2018)].mean(), s[R.year >= 2019].mean(), (np.abs(R.y) <= half).mean()))
O = pd.DataFrame(out, columns=["rule", "IS_all", "IS_1995_2012", "IS_2013_18", "IS_2019_24", "coverage"]).sort_values("IS_all")
print(O.round(2).to_string(index=False))
# z chosen on 1995-2012 only, for the best estimator, then judged on later years
for est in ("ewma0.94", "sd60", "sd20"):
    sd = rules[est]; tr = R.year <= 2012; best = None
    for z in np.arange(1.0, 3.01, 0.05):
        h = z * sd * np.sqrt(R.n); v = IS(-h[tr], h[tr], R.y[tr]).mean()
        if best is None or v < best[0]: best = (v, z)
    z = best[1]; h = z * sd * np.sqrt(R.n); s = IS(-h, h, R.y)
    print(f"{est}: z fitted on 1995-2012 = {z:.2f} -> IS 2013-18 {s[(R.year > 2012) & (R.year <= 2018)].mean():.2f}, 2019-24 {s[R.year >= 2019].mean():.2f}, coverage {(np.abs(R.y) <= h).mean():.2f}")
# the two Development meetings, for reference only (not used for any choice above)
for d in ("2022-07-27", "2024-09-18"):
    x = R[R.d0 == pd.Timestamp(d)]
    if len(x): print(d, "sd10 %.1f sd60 %.1f ewma %.1f bp/day | realized" % (x.sd10.mean(), x.sd60.mean(), x.ewma.mean()), x.y.round(0).tolist())
