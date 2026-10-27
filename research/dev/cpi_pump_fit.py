"""Fit the CPI pump-price pass-through artifact (gasoline, energy, headline) and check interval coverage out of sample.
Model per component c: mom_c(t) = a + b*pump(t) [+ g*mean6_c(t-1)] [+ core6, food6] + month dummy(t); pump(t) = % change of the
average weekly retail price in month t vs month t-1. Fit 2000-01..2022-05 for the artifact; coverage checked on a 2000-2015 fit."""
import json
import numpy as np
import pandas as pd
exec(open("t4-local/dev/cpi_study.py").read().split('print("=== per-component')[0])  # reuse loaders and M

SPEC = {"CPI_GASOLINE": ["pump"], "CPI_ENERGY": ["pump", "m6"], "CPI_ALLITEMS": ["pump", "core6", "food6"]}
art = {}
for c, feats in SPEC.items():
    d = M[[c, "pump", "month"]].copy()
    d["m6"] = d[c].shift(1).rolling(6).mean()
    d["core6"] = M["CPI_CORE"].shift(1).rolling(6).mean()
    d["food6"] = M["CPI_FOOD"].shift(1).rolling(6).mean()
    d = d.dropna()
    d = d[d.index.year >= 2000]
    def X(df):
        return np.column_stack([df[f].values for f in feats] + [np.ones(len(df))] + [(df.month == k).astype(float).values for k in range(2, 13)])
    tr, te = d[d.index.year <= 2015], d[d.index.year >= 2016]
    b_tr, *_ = np.linalg.lstsq(X(tr), tr[c].values, rcond=None)
    r_tr = tr[c].values - X(tr) @ b_tr
    r_te = te[c].values - X(te) @ b_tr
    for z in (1.645, 1.8, 2.0):
        half = z * r_tr.std()
        cov = np.mean(np.abs(r_te) <= half)
        IS = np.mean(2 * half + 20 * np.maximum(np.abs(r_te) - half, 0))
        print(f"{c:13s} z={z}: half {half:.3f} test coverage {cov:.2f} mean IS {IS:.3f}")
    q = np.quantile(r_tr, [0.05, 0.95])
    cov_q = np.mean((r_te >= q[0]) & (r_te <= q[1]))
    print(f"{c:13s} empirical 5/95 residual quantiles {q.round(3)} test coverage {cov_q:.2f}")
    b, *_ = np.linalg.lstsq(X(d), d[c].values, rcond=None)
    r = d[c].values - X(d) @ b
    art[c] = {"feats": feats, "coef": [round(float(x), 5) for x in b[:len(feats)]], "intercept": round(float(b[len(feats)]), 5),
              "month": [0.0] + [round(float(x), 5) for x in b[len(feats) + 1:]], "resid_sd": round(float(r.std()), 4),
              "n": int(len(d)), "fit_window": "2000-01..2022-05"}
    print(c, art[c])
json.dump(art, open("t4-local/dev/cpi_pump_fit.json", "w"), indent=1)
