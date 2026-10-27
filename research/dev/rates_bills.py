"""Experiment 79-84: does the T-bill curve (market-implied policy path) predict the 35-trading-day change of 2Y..30Y yields?
Features known at the origin: b3 = DGS3MO - fed funds target midpoint, b6 = DGS6MO - midpoint, b63 = DGS6MO - DGS3MO (expected path slope),
s26 = DGS2 - DGS6MO. Target: y_m(t+35) - y_m(t) in bp. Pooled regression through the origin (no intercept) per split; skill = MAE0/(MAE0+MAE) on non-overlapping origins."""
import json
import numpy as np
import pandas as pd

def load(s):
    d = json.load(open(f"data_fred/{s}.json"))["observations"]
    return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})
MATS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MATS + ["DGS3MO", "DGS6MO"]}).dropna()
tgt = load("DFEDTAR").dropna(); mid = ((load("DFEDTARU") + load("DFEDTARL")) / 2).dropna()
pol = pd.concat([tgt[tgt.index < mid.index.min()], mid]).sort_index().reindex(Y.index, method="ffill")
Y = Y[Y.index >= "1990-01-01"]; pol = pol.reindex(Y.index)
H = 35
rows = []
for i in range(5, len(Y) - H, 1):
    r = Y.iloc[i]
    f = dict(t=Y.index[i], b3=r.DGS3MO - pol.iloc[i], b6=r.DGS6MO - pol.iloc[i], b63=r.DGS6MO - r.DGS3MO, s26=r.DGS2 - r.DGS6MO)
    for m in MATS:
        rows.append(dict(f, m=m, y=(Y[m].iloc[i + H] - r[m]) * 100, s_m=r[m] - r.DGS6MO))
D = pd.DataFrame(rows)
FEATS = [["b3"], ["b6"], ["b63"], ["s26"], ["b6", "b63"], ["b3", "b6", "b63", "s26"]]
print("origins", D.t.nunique(), D.t.min().date(), D.t.max().date())
for tr, te in (((1990, 2007), (2008, 2022)), ((1990, 2012), (2013, 2022)), ((2000, 2015), (2016, 2022)), ((2008, 2022), (1990, 2007))):
    train = D[D.t.dt.year.between(*tr)]; test = D[D.t.dt.year.between(*te)]
    origins = sorted(test.t.unique())[::H]; te_ = test[test.t.isin(origins)]
    print(f"=== train {tr} test {te} (n origins {len(origins)})")
    for feats in FEATS:
        X = train[feats].values; y = train.y.values
        sd = X.std(0) + 1e-9; beta = np.linalg.solve(((X / sd).T @ (X / sd)) + 0.01 * len(X) * np.eye(len(feats)), (X / sd).T @ y)
        p = (te_[feats].values / sd) @ beta
        mae0, mae = np.abs(te_.y).mean(), np.abs(te_.y - p).mean()
        half = np.abs(te_.y - 0.5 * p).mean()
        print(f"  {'+'.join(feats):22s} MAE0 {mae0:6.2f} MAE {mae:6.2f} skill {mae0 / (mae0 + mae):.3f} | half-weight {mae0 / (mae0 + half):.3f} | beta {np.round(beta, 2)}")
