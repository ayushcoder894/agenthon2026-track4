"""E2-E4: CPI component month-over-month (SA) point rules, and gasoline / energy / headline pass-through from weekly
retail pump prices (EIA GASREGW via FRED). Data up to 2022-06-28 (current FRED vintage; first prints differ slightly).
Score per component = naive_MAE / (naive_MAE + MAE), naive = last month's change carried forward (the unit's naive)."""
import json
import numpy as np
import pandas as pd

def load(s):
    d = json.load(open(f"data_fred/{s}.json"))["observations"]
    return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d}).dropna()

COMP = {"CPI_ALLITEMS": "CPIAUCSL", "CPI_CORE": "CPILFESL", "CPI_SHELTER": "CUSR0000SAH1", "CPI_ENERGY": "CPIENGSL",
        "CPI_GASOLINE": "CUSR0000SETB01", "CPI_FOOD": "CPIUFDSL", "CPI_NEWVEH": "CUSR0000SETA01", "CPI_USEDCARS": "CUSR0000SETA02",
        "CPI_APPAREL": "CPIAPPSL", "CPI_MEDICAL": "CPIMEDSL", "CPI_TRANSPSVC": "CUSR0000SAS4"}
M = pd.DataFrame({k: load(v).pct_change() * 100 for k, v in COMP.items()})
gas = load("GASREGW")
gm = gas.groupby(gas.index.to_period("M")).mean()
pump = (gm.pct_change() * 100)
pump.index = pump.index.to_timestamp()
M["pump"] = pump
M = M[M.index >= "1995-01-01"]
M["month"] = M.index.month


def rules(s):
    """Return dict of rule -> prediction series for component series s (prediction for month t uses data <= t-1)."""
    lag = s.shift(1)
    out = {"naive_last": lag, "mean3": s.shift(1).rolling(3).mean(), "mean6": s.shift(1).rolling(6).mean(),
           "mean12": s.shift(1).rolling(12).mean(), "current_0.5last+0.5mean6": 0.5 * lag + 0.5 * s.shift(1).rolling(6).mean()}
    for w in (0.25, 0.35):
        out[f"{w}last+{1-w:.2f}mean6"] = w * lag + (1 - w) * s.shift(1).rolling(6).mean()
        out[f"{w}last+{1-w:.2f}mean12"] = w * lag + (1 - w) * s.shift(1).rolling(12).mean()
    out["0.5mean3+0.5mean12"] = 0.5 * s.shift(1).rolling(3).mean() + 0.5 * s.shift(1).rolling(12).mean()
    return out


def skill(pred, y, naive):
    m = pred.notna() & y.notna() & naive.notna()
    a, b = np.abs(y[m] - naive[m]).mean(), np.abs(y[m] - pred[m]).mean()
    return a / (a + b), m.sum()

print("=== per-component point rules: skill vs naive (train 2000-2015 | test 2016-2022H1)")
best = {}
for c in COMP:
    s = M[c]
    R = rules(s)
    tr = (M.index.year >= 2000) & (M.index.year <= 2015)
    te = M.index.year >= 2016
    res = {k: (skill(v[tr], s[tr], R["naive_last"][tr])[0], skill(v[te], s[te], R["naive_last"][te])[0]) for k, v in R.items()}
    b = max((k for k in res if k != "naive_last"), key=lambda k: res[k][0])
    best[c] = b
    print(f"{c:14s} current {res['current_0.5last+0.5mean6'][0]:.3f}/{res['current_0.5last+0.5mean6'][1]:.3f} | best-on-train {b:22s} {res[b][0]:.3f}/{res[b][1]:.3f} | mean12 {res['mean12'][0]:.3f}/{res['mean12'][1]:.3f}")

print("\n=== gasoline / energy / headline from the pump-price change of the target month")
for c in ("CPI_GASOLINE", "CPI_ENERGY", "CPI_ALLITEMS"):
    d = M[[c, "pump", "month"]].copy()
    d["lag"] = d[c].shift(1)
    d["m6"] = d[c].shift(1).rolling(6).mean()
    d["core6"] = M["CPI_CORE"].shift(1).rolling(6).mean()
    d["food6"] = M["CPI_FOOD"].shift(1).rolling(6).mean()
    d = d.dropna()
    tr, te = d[(d.index.year >= 2000) & (d.index.year <= 2015)], d[d.index.year >= 2016]
    for feats, dummies in ((["pump"], False), (["pump"], True), (["pump", "m6"], True), (["pump", "core6", "food6"], True)):
        def X(df):
            cols = [df[f].values for f in feats] + [np.ones(len(df))]
            if dummies:
                cols += [(df.month == k).astype(float).values for k in range(2, 13)]
            return np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X(tr), tr[c].values, rcond=None)
        p_tr, p_te = X(tr) @ beta, X(te) @ beta
        def sk(df, p):
            a, b = np.abs(df[c] - df.lag).mean(), np.abs(df[c] - p).mean()
            return a / (a + b)
        cur = 0.5 * te.lag + 0.5 * te.m6
        a = np.abs(te[c] - te.lag).mean()
        print(f"{c:13s} {'+'.join(feats):18s} month-dummies={dummies!s:5s} skill train {sk(tr, p_tr):.3f} test {sk(te, p_te):.3f} "
              f"(current rule test {a / (a + np.abs(te[c] - cur).mean()):.3f}) pump coef {beta[0]:.3f} resid sd test {np.std(te[c] - p_te):.3f}")
    if c == "CPI_GASOLINE":
        oct_ = d[d.month == 10]
        print("   October rows: mean CPI gas SA", round(oct_[c].mean(), 2), "mean pump NSA", round(oct_.pump.mean(), 2))
