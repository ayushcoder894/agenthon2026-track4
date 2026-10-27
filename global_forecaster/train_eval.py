"""Train the global LightGBM forecaster on the panel and judge it out of time.
Splits by ORIGIN (publication) date: train <= 2012, validation 2013-2017 (shrinkage and early stopping only), test 2018-2022, test2 2023-2026.
Pseudo-units: rows sharing (origin month, horizon, frequency, transform) are chunked into groups of 10 series, like a Track 4 unit;
per group q = MAE_naive / (MAE_naive + MAE_model) -- the scorer's point-quality formula; 0.5 = carry-forward.
Usage: python train_eval.py [data_limit_year]   (default 2012)"""
import json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfeat import NAMES

D = "D:/AGENTHON/data_global"
LIM = int(sys.argv[1]) if len(sys.argv) > 1 else 2012
P = pd.read_parquet(f"{D}/panel.parquet")
P["yr"] = P.origin.dt.year
tr = P[P.yr <= LIM]; va = P[(P.yr > LIM) & (P.yr <= LIM + 5)]
te = P[(P.yr >= 2018) & (P.yr <= 2022)] if LIM <= 2012 else P.iloc[0:0]; te2 = P[P.yr >= 2023]
print(f"rows train {len(tr):,} val {len(va):,} test {len(te):,} test2 {len(te2):,} | series versions {P.groupby(['sid', 'transform', 'ppy']).ngroups}")
X = NAMES
prm = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=400, subsample=0.7, subsample_freq=1, colsample_bytree=0.8,
           reg_lambda=5.0, verbose=-1, n_jobs=8)
pt = lgb.LGBMRegressor(objective="l1", **prm).fit(tr[X], tr.y, eval_set=[(va[X], va.y)], callbacks=[lgb.early_stopping(100, verbose=False)])
print("point model trees:", pt.best_iteration_)
q05 = lgb.LGBMRegressor(objective="quantile", alpha=0.05, **{**prm, "n_estimators": 400}).fit(tr[X], tr.y)
q95 = lgb.LGBMRegressor(objective="quantile", alpha=0.95, **{**prm, "n_estimators": 400}).fit(tr[X], tr.y)


def evaluate(df, lam, tag):
    if not len(df):
        return
    df = df.copy()
    df["m"] = lam * pt.predict(df[X], num_iteration=pt.best_iteration_)
    df["lo"], df["hi"] = q05.predict(df[X]), q95.predict(df[X])
    df["lo"], df["hi"] = np.minimum(df.lo, df.m), np.maximum(df.hi, df.m)
    e = {k: (df.y - df[c]).abs() if c else df.y.abs() for k, c in (("naive", None), ("half", "b_half"), ("d18chg", "b_d18"), ("model", "m"))}
    print(f"--- {tag} (n={len(df):,})  scaled MAE: " + " | ".join(f"{k} {v.mean():.4f}" for k, v in e.items()))
    df["mon"] = df.origin.dt.to_period("M")
    df["grp"] = df.groupby(["mon", "h", "ppy", "transform"]).cumcount() // 10
    g = df.assign(e0=e["naive"], eh=e["half"], ed=e["d18chg"], em=e["model"]).groupby(["mon", "h", "ppy", "transform", "grp"])
    G = g.agg(e0=("e0", "mean"), eh=("eh", "mean"), ed=("ed", "mean"), em=("em", "mean"), n=("e0", "size"))
    G = G[G.n >= 5]
    q = lambda c: (G.e0 / (G.e0 + G[c])).where((G.e0 + G[c]) > 0, 0.5)
    print(f"    pseudo-units {len(G):,}: mean point quality q  half {q('eh').mean():.4f} | d18chg {q('ed').mean():.4f} | MODEL {q('em').mean():.4f}   "
          f"(0.5 = carry-forward) | model > half in {(q('em') > q('eh')).mean():.0%} of units")
    # interval: model quantiles vs a random-walk band +-1.645 sqrt(h) x 1.25 s (shipped delta-mode style)
    def IS(lo, hi, y): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
    rw = 1.645 * np.sqrt(df.h) * 1.25
    print(f"    interval score (scaled): random-walk band {IS(-rw, rw, df.y).mean():.3f} cov {((df.y.abs() <= rw)).mean():.2f} | "
          f"model q05-q95 {IS(df.lo, df.hi, df.y).mean():.3f} cov {((df.y >= df.lo) & (df.y <= df.hi)).mean():.2f}")
    for t in ("raw", "pct1", "yoy"):
        x = df[df["transform"] == t]
        if len(x): print(f"    {t:5s} n={len(x):,} MAE naive {x.y.abs().mean():.3f} half {(x.y - x.b_half).abs().mean():.3f} d18 {(x.y - x.b_d18).abs().mean():.3f} model {(x.y - x.m).abs().mean():.3f}")
    for a, b in ((8, 10), (11, 13), (14, 26), (27, 120)):
        x = df[(df.n_hist >= a) & (df.n_hist <= b)]
        if len(x):
            e0, eh, ed, em = x.y.abs().mean(), (x.y - x.b_half).abs().mean(), (x.y - x.b_d18).abs().mean(), (x.y - x.m).abs().mean()
            print(f"    history {a:3d}-{b:3d} values n={len(x):7,}: q-like naive/(naive+err)  half {e0 / (e0 + eh):.3f} | d18 {e0 / (e0 + ed):.3f} | model {e0 / (e0 + em):.3f}")
    return df


best = None
for lam in (0.5, 0.75, 1.0):
    m = lam * pt.predict(va[X], num_iteration=pt.best_iteration_); v = (va.y - m).abs().mean()
    if best is None or v < best[0]: best = (v, lam)
lam = best[1]; print(f"shrinkage chosen on validation: {lam}")
evaluate(va, lam, f"VALIDATION {LIM + 1}-{LIM + 5}")
evaluate(te, lam, "TEST 2018-2022")
evaluate(te2, lam, "TEST 2023-2026")
imp = sorted(zip(pt.booster_.feature_importance("gain"), X), reverse=True)
print("top features:", [(n, round(float(g) / sum(i for i, _ in imp), 3)) for g, n in imp[:10]])
import pickle
pickle.dump(dict(pt=pt, q05=q05, q95=q95, lam=lam, lim=LIM), open(f"{D}/global_lim{LIM}.pkl", "wb"))
