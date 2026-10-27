"""Freeze the validated cutoff-safe predictors as JSON artifacts, fitted ONLY on data published by 2022-06-28 (same limit as D20R's artifacts)."""
import os, sys, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); __file__ = os.path.join(HERE, "cpi_exps.py"); sys.argv = [sys.argv[0]]
exec(open(os.path.join(HERE, "cpi_exps.py"), encoding="utf-8").read().split("fam = Family(")[0])      # -> L
LIM = pd.Timestamp("2022-06-01")                                          # target months <= 2022-05 (published by mid-June 2022)
COMPS = ["headline", "core", "food", "energy", "gasoline", "apparel", "medical", "newveh", "shelter", "transpsvc", "usedcars"]
NONPUMP = ["core", "food", "medical", "newveh", "shelter", "transpsvc", "usedcars", "apparel"]
W = L.pivot_table(index="month", columns="comp", values="y")
FEATS = [f"x_{c}_l{k}" for c in COMPS for k in (1, 2, 3)]
X = pd.concat({f"x_{c}_l{k}": W[c].shift(k) for c in COMPS for k in (1, 2, 3)}, axis=1)
C = L[(L.month >= "1990-01-01") & (L.month < LIM)].merge(X, left_on="month", right_index=True, how="left").dropna(subset=FEATS)
art = {"limit": "2022-06-28", "features": FEATS, "alpha": 1000.0, "scale": 0.5, "xcomp": {}}
for comp in NONPUMP:
    g = C[C.comp == comp]; M = g[FEATS].values; mu, sd = M.mean(0), M.std(0) + 1e-9
    r = Ridge(alpha=1000.0).fit((M - mu) / sd, (g.y - g.ship).values)
    art["xcomp"][comp] = dict(mu=mu.round(6).tolist(), sd=sd.round(6).tolist(), coef=r.coef_.round(8).tolist(), intercept=float(r.intercept_), n=int(len(g)))
# turning-point coefficient on all components, events with target month < LIM
lg = C[[f"l{i}" for i in range(1, 13)]].values
def runlen(rw):
    s = np.sign(rw[1]); k = 0
    for v in rw[1:]:
        if v != 0 and np.sign(v) == s: k += 1
        else: break
    return k
run = np.array([runlen(rw) for rw in lg]); prev3 = lg[:, 1:4].mean(1)
turn = (np.sign(C["last"].values) != np.sign(C.l2.values)) & (run >= 3) & (C["last"].values != 0)
sig = np.sign(C["last"].values) * np.abs(prev3); res = (C.y - C.ship).values
art["turning_b"] = float((sig[turn] * res[turn]).sum() / (sig[turn] ** 2).sum()); art["turning_events"] = int(turn.sum())
# macro: series x revision-number up-shares (ALFRED <= 2022-06-28)
import json as _j
ev = []
for s in ("DGORDER", "HOUST", "INDPRO", "PAYEMS", "PI", "RSAFS"):
    d = pd.DataFrame(_j.load(open(f"D:/AGENTHON/data_alfred/{s}.json"))); d["v"] = pd.to_numeric(d.value, errors="coerce"); d = d.dropna(subset=["v"]).sort_values(["date", "realtime_start"])
    for dt_, gg in d.groupby("date"):
        v = gg.v.values
        for i in range(1, len(v)):
            if v[i] != v[i - 1]: ev.append(dict(series=s, k=min(i, 3), up=int(v[i] > v[i - 1])))
E = pd.DataFrame(ev)
art["macro_series_up"] = {s: round(float(g.up.mean()), 4) for s, g in E.groupby("series")}
art["macro_series_k_up"] = {f"{s}|{k}": round(float(g.up.mean()), 4) for (s, k), g in E.groupby(["series", "k"]) if len(g) >= 20}
json.dump(art, open(os.path.join(HERE, "cutoffsafe_artifacts.json"), "w"))
print("turning b", round(art["turning_b"], 4), "events", art["turning_events"]); print("macro series x k:", art["macro_series_k_up"])
