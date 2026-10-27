"""Experiments 1-9 (CPI). Loaders from cpi_study.py. Train 2000-2015, test 2016-2022H1. Skill = naive_MAE/(naive_MAE+MAE), naive = last print.
 1  month dummies for the non-pump components added to the current rule (0.5 last + 0.5 mean6 + month effect)
 2  month dummies only for shelter/apparel/medical/transpsvc/newveh (the seasonal ones)
 3  headline predicted as weighted sum of component forecasts (food, energy-pump model, core) vs the pump headline model
 4  headline: pump model + core-from-rule combination
 5  gasoline: add the last-week-of-month pump change (momentum inside the month) as a feature
 6  energy and gasoline: asymmetric (quantile) intervals from train residuals vs symmetric z*sd
 7  core CPI: month dummies + shelter mean6
 8  shrink of month dummies (0.5x) to limit overfit
 9  interval: empirical residual quantiles by component (5/95) vs 2.08*sd9 for the 8 non-pump components
"""
import numpy as np
import pandas as pd

exec(open("t4-local/dev/cpi_study.py").read().split('print("=== per-component')[0])
NONPUMP = ["CPI_CORE", "CPI_SHELTER", "CPI_FOOD", "CPI_NEWVEH", "CPI_USEDCARS", "CPI_APPAREL", "CPI_MEDICAL", "CPI_TRANSPSVC"]
tr_i = (M.index.year >= 2000) & (M.index.year <= 2015)
te_i = M.index.year >= 2016
def skill(y, p, naive):
    m = y.notna() & p.notna() & naive.notna()
    a, b = np.abs(y[m] - naive[m]).mean(), np.abs(y[m] - p[m]).mean()
    return a / (a + b)
def base(c):
    s = M[c]; return 0.5 * s.shift(1) + 0.5 * s.shift(1).rolling(6).mean()

print("--- 1/2/8: month effects on the residual of the current rule")
for c in NONPUMP:
    s = M[c]; cur = base(c); res = (s - cur)
    eff = res[tr_i].groupby(M.month[tr_i]).mean()
    p1 = cur + M.month.map(eff)
    p8 = cur + 0.5 * M.month.map(eff)
    print(f"{c:14s} current {skill(s[te_i], cur[te_i], s.shift(1)[te_i]):.3f} | +month {skill(s[te_i], p1[te_i], s.shift(1)[te_i]):.3f} | +0.5 month {skill(s[te_i], p8[te_i], s.shift(1)[te_i]):.3f}")

print("--- 3/4: headline from components")
W = {"food": 0.135, "energy": 0.07, "core": 0.795}
fit = json.load(open("t4-local/dev/cpi_pump_fit.json"))
def pump_pred(c):
    f = fit[c]; d = pd.DataFrame({"pump": M.pump, "m6": M[c].shift(1).rolling(6).mean(), "core6": M.CPI_CORE.shift(1).rolling(6).mean(), "food6": M.CPI_FOOD.shift(1).rolling(6).mean()})
    v = f["intercept"] + sum(cf * d[ft] for ft, cf in zip(f["feats"], f["coef"])) + M.month.map(lambda m: f["month"][m - 1])
    return v
head_pump = pump_pred("CPI_ALLITEMS"); energy_p = pump_pred("CPI_ENERGY")
comp = W["food"] * base("CPI_FOOD") + W["energy"] * energy_p + W["core"] * base("CPI_CORE")
y, naive = M.CPI_ALLITEMS, M.CPI_ALLITEMS.shift(1)
for nm, p in (("pump headline model (current)", head_pump), ("weighted components", comp), ("avg of both", 0.5 * head_pump + 0.5 * comp)):
    print(f"  {nm:32s} test skill {skill(y[te_i], p[te_i], naive[te_i]):.3f}  MAE {np.abs(y - p)[te_i].mean():.4f}")

print("--- 5: gasoline with within-month pump momentum (last 2 weeks vs first 2 weeks) -- needs weekly data")
g = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in json.load(open("data_fred/GASREGW.json"))["observations"] if o["value"] not in (".", "")})
wk = pd.DataFrame({"p": g}); wk["m"] = wk.index.to_period("M")
agg = wk.groupby("m").p.agg(["mean", "first", "last", "count"]); agg.index = agg.index.to_timestamp()
mom = (agg["last"] / agg["first"] - 1) * 100
d = pd.DataFrame({"y": M.CPI_GASOLINE, "pump": M.pump, "mom": mom, "month": M.month}).dropna()
d = d[d.index.year >= 2000]
X = lambda df, extra: np.column_stack([df[f] for f in (["pump"] + extra)] + [np.ones(len(df))] + [(df.month == k).astype(float) for k in range(2, 13)])
tr, te = d[d.index.year <= 2015], d[d.index.year >= 2016]
for extra in ([], ["mom"]):
    b = np.linalg.lstsq(X(tr, extra), tr.y, rcond=None)[0]; p = X(te, extra) @ b
    nv = M.CPI_GASOLINE.shift(1).reindex(te.index)
    print(f"  gasoline features pump{'+mom' if extra else ''}: test skill {skill(te.y, pd.Series(p, te.index), nv):.3f} resid sd {np.std(te.y - p):.3f}")

print("--- 6: asymmetric intervals for gasoline/energy (train residual quantiles vs z*sd) on the pump models, test IS")
for c in ("CPI_GASOLINE", "CPI_ENERGY", "CPI_ALLITEMS"):
    p = pump_pred(c); r = (M[c] - p).dropna()
    rtr, rte = r[(r.index.year >= 2000) & (r.index.year <= 2015)], r[r.index.year >= 2016]
    def IS(lo, hi, e): return np.mean((hi - lo) + 20 * np.maximum(lo - e, 0) + 20 * np.maximum(e - hi, 0))
    sym = 1.645 * rtr.std(); q = np.percentile(rtr, [5, 95])
    print(f"  {c:13s} symmetric IS {IS(-sym, sym, rte):.3f} cov {np.mean(np.abs(rte) <= sym):.2f} | quantile IS {IS(q[0], q[1], rte):.3f} cov {np.mean((rte >= q[0]) & (rte <= q[1])):.2f}")

print("--- 7: core with month dummies + shelter mean6")
d = pd.DataFrame({"y": M.CPI_CORE, "last": M.CPI_CORE.shift(1), "m6": M.CPI_CORE.shift(1).rolling(6).mean(), "sh6": M.CPI_SHELTER.shift(1).rolling(6).mean(), "month": M.month}).dropna()
tr, te = d[(d.index.year >= 2000) & (d.index.year <= 2015)], d[d.index.year >= 2016]
for feats, mo in ((["last", "m6"], False), (["last", "m6"], True), (["last", "m6", "sh6"], True)):
    Xf = lambda df: np.column_stack([np.ones(len(df))] + [df[f] for f in feats] + ([(df.month == k).astype(float) for k in range(2, 13)] if mo else []))
    b = np.linalg.lstsq(Xf(tr), tr.y, rcond=None)[0]; p = Xf(te) @ b
    print(f"  {'+'.join(feats)} months={mo}: skill {skill(te.y, pd.Series(p, te.index), te['last']):.3f}")

print("--- 9: empirical quantile bands (per component, train 2000-15) vs 2.08*sd9, test IS")
for c in NONPUMP:
    s = M[c]; cur = base(c); sd9 = s.shift(1).rolling(9).std(ddof=0)
    r = (s - cur); rtr = r[tr_i].dropna(); d = pd.DataFrame({"r": r, "sd": sd9}).dropna(); d = d[d.index.year >= 2016]
    q = np.percentile(rtr, [5, 95])
    IS = lambda lo, hi, e: np.mean((hi - lo) + 20 * np.maximum(lo - e, 0) + 20 * np.maximum(e - hi, 0))
    h = np.maximum(2.08 * d.sd, 0.25)
    print(f"  {c:14s} 2.08*sd9 IS {IS(-h, h, d.r):.3f} cov {np.mean(np.abs(d.r) <= h):.2f} | fixed quantile band {q.round(2)} IS {IS(q[0], q[1], d.r):.3f} cov {np.mean((d.r >= q[0]) & (d.r <= q[1])):.2f}")
