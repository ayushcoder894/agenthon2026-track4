"""Fit the final EPS quantile-interval models on ALL outcomes <= 2022-06-28 (revenue >= $1B, |prior| >= 0.5) and export them; verify the pure-Python evaluator."""
import json, sys
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, "t4-local/agent")
from t4_agent import gbm
src = open("t4-local/dev/ml_intervals.py").read().split("print(f\"train")[0]
exec(src)
S2 = S.copy()
models = {}
for a, name in ((0.03, "q03"), (0.97, "q97")):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=7)
    m.fit(S2[FEATS], S2.res)
    models[name] = gbm.convert(m.booster_.dump_model())
    X = S2[FEATS].values[:3000]
    ref = m.predict(X); mine = np.array([gbm.predict(models[name], list(r)) for r in X])
    print(name, "trees", len(models[name]["trees"]), "max |lightgbm - python| on 3000 rows:", float(np.abs(ref - mine).max()))
p = "t4-local/agent/t4_agent/artifacts.json"
a = json.load(open(p, encoding="utf-8")); e = a["families"]["eps_runrate"]
e["interval_gbm"] = {"features": FEATS, "q03": models["q03"], "q97": models["q97"], "n_train": int(len(S2)), "min_abs_prior": 0.5,
    "note": ("LightGBM quantile regression (alpha 0.03 / 0.97, 200 trees x 7 leaves, min_child_samples 200) of the percent-growth residual around the bucket-weight point, fitted on revenue >= $1B "
             "company-quarters with |prior-year EPS| >= 0.5 (all outcomes <= 2022-06-28, first-reported values). Train <= 2018, test 2019-22: interval score 599 -> 527 (-12%), coverage 0.904 "
             "(bucket bands 0.893); better in every test year (-10..-15%) and for financials (-18%) and non-financials (-11%). Features: rr, gl, ratio, ltp, eps_last.")}
open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(a, indent=1))
print("stored; artifacts.json bytes:", len(json.dumps(a)))
