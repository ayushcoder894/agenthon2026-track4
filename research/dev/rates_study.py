"""E1: can corpus-visible features beat 'no change' for the inter-meeting change in Treasury CMT yields?
Data: FRED DGS1..DGS30, DFEDTAR / DFEDTARU / DFEDTARL (public domain), observations up to 2022-06-28.
Target: y_m(t+h) - y_m(t) in bp, h = 35 trading days. Origins every trading day for fitting; evaluation on
non-overlapping origins of the test period. Score = MAE(model) vs MAE(zero), as the unit's regression leg."""
import json
import numpy as np
import pandas as pd

def load(s):
    d = json.load(open(f"data_fred/{s}.json"))["observations"]
    x = pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})
    return x
MATS = ["DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MATS}).dropna(subset=["DGS2", "DGS10"])
tar = load("DFEDTAR").dropna()
mid = ((load("DFEDTARU") + load("DFEDTARL")) / 2).dropna()
pol = pd.concat([tar[tar.index < mid.index.min()], mid]).sort_index()
Y["mid"] = pol.reindex(Y.index, method="ffill")
Y = Y[Y.index >= "1990-01-01"].ffill()
H = 35
rows = []
idx = Y.index
for i in range(70, len(idx) - H):
    t = idx[i]
    r = Y.iloc[i]
    f = dict(t=t, mid=r.mid, pol3m=(r.mid - Y.mid.iloc[i - 63]), pol6m=(r.mid - Y.mid.iloc[max(0, i - 126)]),
             slope=r.DGS10 - r.DGS2, gap1=r.DGS1 - r.mid, gap2=r.DGS2 - r.mid)
    for m in MATS:
        if np.isnan(r[m]) or np.isnan(Y[m].iloc[i + H]):
            continue
        rows.append(dict(f, mat=m, mom10=(r[m] - Y[m].iloc[i - 10]) * 100, mom60=(r[m] - Y[m].iloc[i - 60]) * 100,
                         gapm=r[m] - r.mid, y=(Y[m].iloc[i + H] - r[m]) * 100))
D = pd.DataFrame(rows)
print("rows", len(D), D.t.min().date(), D.t.max().date())
FEATS = [["pol3m"], ["pol6m"], ["gap1"], ["gap2"], ["gapm"], ["mom10"], ["mom60"], ["slope"], ["pol3m", "gap1"], ["pol3m", "gapm", "mom10"],
         ["pol3m", "pol6m", "gap1", "gap2", "slope", "mom10", "mom60"]]


def fit_eval(train, test, feats, lam=1.0):
    X = train[feats].values; y = train.y.values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    beta = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * 0.01 * np.eye(len(feats)), Xs.T @ (y - 0))  # no intercept: drift is not stable
    te = test[test.t.isin(test.t.drop_duplicates().iloc[::H])]  # non-overlapping origins
    p = ((te[feats].values - mu) / sd) @ beta
    mae0, mae = np.abs(te.y).mean(), np.abs(te.y - p).mean()
    # best shrink of the prediction (as an agent would apply a fixed fraction)
    res = {s: np.abs(te.y - s * p).mean() for s in (0.25, 0.5, 1.0)}
    return mae0, mae, res, len(te.t.unique()), beta


for tr, te_ in (((1990, 2007), (2008, 2022)), ((1990, 2012), (2013, 2022)), ((2000, 2015), (2016, 2022)), ((2008, 2022), (1990, 2007))):
    train = D[D.t.dt.year.between(*tr)]; test = D[D.t.dt.year.between(*te_)]
    print(f"\n=== train {tr} test {te_}")
    for feats in FEATS:
        mae0, mae, res, n, beta = fit_eval(train, test, feats)
        sk = mae0 / (mae0 + mae)
        print(f"  {'+'.join(feats):45s} n_origins={n:3d} MAE0={mae0:6.2f} MAE={mae:6.2f} skill={sk:.3f} | shrink .25:{mae0/(mae0+res[0.25]):.3f} .5:{mae0/(mae0+res[0.5]):.3f} beta={np.round(beta,2)}")
