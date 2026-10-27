"""Experiment A: FOMC joint yield-curve model (Dynamic Nelson-Siegel factors). Chronological, as-of tests only.
Target: change in DGS{2,3,5,7,10,30} (bp) over H=33 trading days from origin t (the unit: cutoff close -> resolution close, ~33 trading days).
Origin features use only data <= t (yields, policy target mid). Models are refit on an expanding window of origins whose outcome was known at t (origin <= t - H).
 naive      : 0 change (what the shipped _rates emits)
 direct     : per-maturity ridge on the same feature set (no factor structure)
 dns        : yields -> level/slope/curvature (lambda=0.0609/month); ridge for factor changes; reconstruct yield changes with the DNS loadings
 dns_nogap  : dns without the policy-gap feature
Skill = MAE0/(MAE0 + MAE) pooled over maturities. Shrinkage alpha scales the model forecast toward 0. Block bootstrap by calendar year."""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge


def load(s):
    d = json.load(open(f"data_fred/{s}.json"))["observations"]
    return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})


MAT_M = {"DGS1": 12, "DGS2": 24, "DGS3": 36, "DGS5": 60, "DGS7": 84, "DGS10": 120, "DGS30": 360}
TGT = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MAT_M}).dropna(); Y = Y[Y.index >= "1985-01-01"]
tgt = load("DFEDTAR").dropna(); mid = ((load("DFEDTARU") + load("DFEDTARL")) / 2).dropna()
pol = pd.concat([tgt[tgt.index < mid.index.min()], mid]).sort_index().reindex(Y.index, method="ffill")
print("yields", Y.index.min().date(), "->", Y.index.max().date(), "days", len(Y))
H = 33
LAM = 0.0609
mm = np.array([MAT_M[m] for m in MAT_M], float)
LD = np.column_stack([np.ones_like(mm), (1 - np.exp(-LAM * mm)) / (LAM * mm), (1 - np.exp(-LAM * mm)) / (LAM * mm) - np.exp(-LAM * mm)])   # [7,3]
LD_T = LD[[list(MAT_M).index(m) for m in TGT]]
pinv = np.linalg.pinv(LD)
BET = pd.DataFrame(Y.values @ pinv.T, index=Y.index, columns=["L", "S", "C"])   # factors in percent
resid_fit = np.abs(Y.values - BET.values @ LD.T).mean() * 100
print(f"DNS in-sample mean abs fit error across 7 maturities: {resid_fit:.1f} bp")

# ---------- origins and outcomes ----------
idx = np.arange(40, len(Y) - H, 3)       # every 3rd trading day
O = pd.DataFrame(index=Y.index[idx])
for m in TGT: O[f"d_{m}"] = (Y[m].values[idx + H] - Y[m].values[idx]) * 100
O["end"] = Y.index[idx + H]
# features at origin
b = BET
f = pd.DataFrame(index=Y.index)
for k in "LSC":
    f[f"{k}"] = b[k] * 100
    f[f"d10_{k}"] = (b[k] - b[k].shift(10)) * 100
    f[f"dev_{k}"] = (b[k] - b[k].ewm(halflife=250).mean()) * 100     # distance to its slow mean (mean reversion)
f["gap"] = (Y.DGS1 - pol) * 100                  # front end vs policy midpoint (bp)
f["gap2"] = (Y.DGS2 - pol) * 100
f["polchg"] = (pol - pol.shift(63)) * 100       # policy move over the last quarter
FE = f.loc[O.index]
FE = FE.dropna(); O = O.loc[FE.index]
FEAT = {"dns": ["dev_L", "dev_S", "dev_C", "d10_L", "d10_S", "d10_C", "gap", "gap2", "polchg"],
        "dns_nogap": ["dev_L", "dev_S", "dev_C", "d10_L", "d10_S", "d10_C"]}
# factor-change targets (bp) over H
dB = (b.shift(-H) - b).loc[O.index] * 100

ALPHA_R = 3000.0


def fit_pred(tr_idx, te_idx, kind, cols):
    Xtr, Xte = FE.loc[tr_idx, cols].values, FE.loc[te_idx, cols].values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if kind == "direct":
        P = np.column_stack([Ridge(alpha=ALPHA_R).fit(Xtr, O.loc[tr_idx, f"d_{m}"].clip(-150, 150).values).predict(Xte) for m in TGT])
        return P
    Bp = np.column_stack([Ridge(alpha=ALPHA_R).fit(Xtr, dB.loc[tr_idx, k].clip(-120, 120).values).predict(Xte) for k in "LSC"])   # factor changes (bp)
    return Bp @ LD_T.T


rows = []
years = sorted(set(O.index.year))
preds = {k: pd.DataFrame(np.nan, index=O.index, columns=TGT) for k in ("direct", "dns", "dns_nogap")}
for yr in range(1998, years[-1] + 1):
    te_idx = O.index[O.index.year == yr]
    if len(te_idx) == 0: continue
    cut = te_idx.min()
    tr_idx = O.index[(O.end < cut)]            # outcome fully known before the first origin of the test year
    tr_idx = tr_idx[tr_idx >= "1986-01-01"]
    if len(tr_idx) < 500: continue
    preds["direct"].loc[te_idx] = fit_pred(tr_idx, te_idx, "direct", FEAT["dns"])
    preds["dns"].loc[te_idx] = fit_pred(tr_idx, te_idx, "dns", FEAT["dns"])
    preds["dns_nogap"].loc[te_idx] = fit_pred(tr_idx, te_idx, "dns", FEAT["dns_nogap"])

act = O[[f"d_{m}" for m in TGT]].copy(); act.columns = TGT
ok = preds["dns"].notna().all(axis=1)
act, ok_idx = act[ok], ok[ok].index
print("test origins", len(ok_idx), ok_idx.min().date(), ok_idx.max().date())


def skill(a, p):
    m0 = np.abs(a.values).mean(); return m0 / (m0 + np.abs(a.values - p.values).mean())


def boot(a, p, n=400, seed=0):
    d = (np.abs(a) - np.abs(a - p)).mean(axis=1)      # >0 model better
    g = d.groupby(d.index.year).agg(["sum", "count"]); rng = np.random.default_rng(seed); est = []
    for _ in range(n):
        s = rng.integers(0, len(g), len(g)); est.append(g["sum"].values[s].sum() / g["count"].values[s].sum())
    return float(d.mean()), float(np.std(est))


periods = {"1998-2007": ("1998", "2007"), "2008-2015": ("2008", "2015"), "2016-2024": ("2016", "2024"), "ALL": ("1998", "2024"),
           "ZLB-free 1998-2007+2016-24": None}
print("\nSkill vs 0-change (higher is better; 0.500 = equal to naive)")
out = []
for per, rng_ in periods.items():
    if rng_ is None:
        sel = act.index[(act.index.year <= 2007) | (act.index.year >= 2016)]
    else:
        sel = act.index[(act.index.year >= int(rng_[0])) & (act.index.year <= int(rng_[1]))]
    a = act.loc[sel]
    for kind in ("direct", "dns", "dns_nogap"):
        for al in (0.25, 0.5, 1.0):
            p = preds[kind].loc[sel] * al
            dmean, se = boot(a, p)
            out.append(dict(period=per, model=kind, alpha=al, n=len(a), skill=skill(a, p), skill_naive=skill(a, a * 0), d_mae=dmean, se=se, z=dmean / se if se > 0 else 0))
R = pd.DataFrame(out); pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
R.to_csv("t4-local/exp_out/fomc_dns_results.csv", index=False)
# year-by-year consistency for dns alpha=0.5
yr = []
for y_ in sorted(set(act.index.year)):
    s = act.index[act.index.year == y_]
    yr.append((y_, len(s), float((np.abs(act.loc[s].values) - np.abs(act.loc[s].values - 0.5 * preds["dns"].loc[s].values)).mean()),
               float((np.abs(act.loc[s].values) - np.abs(act.loc[s].values - 0.5 * preds["dns_nogap"].loc[s].values)).mean())))
Yr = pd.DataFrame(yr, columns=["year", "n", "dns_a0.5_gain_bp", "dns_nogap_a0.5_gain_bp"])
print("\nPer-year MAE gain over naive (bp, >0 = better):"); print(Yr.round(2).to_string(index=False))
print("years dns better:", int((Yr["dns_a0.5_gain_bp"] > 0).sum()), "of", len(Yr))
# coherence: do the reconstructed changes keep a sensible curve shape? (share with monotone-ish/sign agreement across 2s..10s)
p = preds["dns"].loc[act.index]
print("share of origins where predicted 2Y..30Y changes all have the same sign:", round(float((np.sign(p).nunique(axis=1) == 1).mean()), 3))
print("mean |pred| bp by maturity:", (p.abs().mean()).round(2).to_dict())
