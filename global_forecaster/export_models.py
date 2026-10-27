"""Export trained global models (global_lim<YEAR>.pkl) to the agent's JSON tree format, one copy per data limit (cutoff guard).
Each copy is retrained on ALL rows published up to its data limit (train + validation years), with the tree count and shrinkage
chosen in train_eval.py. Usage: python export_models.py 2005 2010 2015 2020"""
import json, os, pickle, sys
import pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "D:/AGENTHON/t4-local/agent")
from gfeat import NAMES
from t4_agent import gbm

D = "D:/AGENTHON/data_global"
ref = pickle.load(open(f"{D}/global_lim2012.pkl", "rb"))
n_trees, lam = int(ref["pt"].best_iteration_ or ref["pt"].n_estimators), ref["lam"]
P = pd.read_parquet(f"{D}/panel.parquet")
prm = dict(learning_rate=0.03, num_leaves=31, min_child_samples=400, subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=8)
out = {"names": NAMES, "note": "global LightGBM forecaster; target (v[t+h]-v[t])/s; trained on FRED/ALFRED first-release macro and never-revised market series published on or before data_limit", "models": []}
for y in map(int, sys.argv[1:]):
    tr = P[P.origin <= f"{y}-12-31"]
    pt = lgb.LGBMRegressor(objective="l1", n_estimators=n_trees, **prm).fit(tr[NAMES], tr.y)
    q05 = lgb.LGBMRegressor(objective="quantile", alpha=0.05, n_estimators=400, **prm).fit(tr[NAMES], tr.y)
    q95 = lgb.LGBMRegressor(objective="quantile", alpha=0.95, n_estimators=400, **prm).fit(tr[NAMES], tr.y)
    out["models"].append(dict(data_limit=f"{y}-12-31", rows=int(len(tr)), lam=lam, pt=gbm.convert(pt.booster_.dump_model()),
                              q05=gbm.convert(q05.booster_.dump_model()), q95=gbm.convert(q95.booster_.dump_model())))
    print("exported data limit", y, "rows", len(tr))
json.dump(out, open("D:/AGENTHON/t4-local/g1/t4_agent/global_models.json", "w"))
print("size MB", round(os.path.getsize("D:/AGENTHON/t4-local/g1/t4_agent/global_models.json") / 1e6, 1))
