"""FOMC-event study: origins = days the policy target changed (hikes/cuts), 1990-2022. Target = 35-trading-day change of 2Y..30Y yields (bp).
Features: size of the move (bp), sign, surprise proxy = actual change - bill-implied expectation (DGS6MO(t-1) - DFF(t-1)), continuation (same sign as the prior move within 120 days), turning point.
Pooled through-origin regression, chronological split; skill = MAE0/(MAE0+MAE) with a 'no change' naive."""
import json
import numpy as np, pandas as pd
def load(s):
    d = json.load(open(f"data_fred/{s}.json"))["observations"]
    return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})
MATS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MATS + ["DGS6MO"]}).dropna(); Y = Y[Y.index >= "1990-01-01"]
tgt = load("DFEDTAR").dropna(); mid = ((load("DFEDTARU") + load("DFEDTARL")) / 2).dropna()
pol = pd.concat([tgt[tgt.index < mid.index.min()], mid]).sort_index().reindex(Y.index, method="ffill")
dff = load("DFF").reindex(Y.index, method="ffill")
chg = (pol.diff() * 100).fillna(0); ev = chg[chg.abs() >= 12].index
rows = []; last_sign = 0; last_t = None
for t in ev:
    i = Y.index.get_loc(t)
    if i + 35 >= len(Y) or i < 3: continue
    c = chg.loc[t]; sgn = np.sign(c)
    cont = int(last_t is not None and (t - last_t).days <= 120 and np.sign(chg.loc[last_t]) == sgn)
    exp = (Y.DGS6MO.iloc[i - 1] - dff.iloc[i - 1]) * 100 * 0.5
    for m in MATS: rows.append(dict(t=t, m=m, y=(Y[m].iloc[i + 35] - Y[m].iloc[i]) * 100, size=c, sgn=sgn, surprise=c - exp, cont=cont * sgn, turn=(1 - cont) * sgn))
    last_t = t
D = pd.DataFrame(rows); print("policy-change origins:", D.t.nunique(), D.t.min().date(), D.t.max().date())
for tr_y, te_y in (((1990, 2007), (2008, 2022)), ((1990, 2012), (2013, 2022)), ((2008, 2022), (1990, 2007))):
    tr, te = D[D.t.dt.year.between(*tr_y)], D[D.t.dt.year.between(*te_y)]
    out = []
    for feats in (["sgn"], ["size"], ["surprise"], ["cont"], ["turn"], ["sgn", "surprise", "cont"]):
        X = tr[feats].values; b = np.linalg.lstsq(X, tr.y.values, rcond=None)[0]; p = te[feats].values @ b
        m0, m1 = np.abs(te.y).mean(), np.abs(te.y - p).mean(); out.append(f"{'+'.join(feats)} {m0 / (m0 + m1):.3f}")
    print(f"train {tr_y} test {te_y} (n origins {te.t.nunique()}): " + " | ".join(out))
