"""ML-A: LightGBM vs the shipped run-rate bucket rules for EPS direction and growth. Features limited to what the agent can read at run time:
cur = latest reported quarter EPS (10-Q), tp = target quarter's prior-year EPS (task row), g_last = latest quarter's own YoY growth (parsed pair).
Train outcomes <= 2018-12-31, test 2019-01..2022-03 (same split as the rule fits). Direction: accuracy. Growth: MAE skill vs 0% (the naive point)."""
import numpy as np
import pandas as pd
import lightgbm as lgb

exec(open("t4-local/dev/eps_runrate_fit.py").read().split("# direction")[0])
WB = {'-10..0': 0.0, '-50..-10': 0.5, '0..10': 1.25, '10..30': 1.0, '<=-50': 0.5, '>30': 1.0}
P = P.dropna(subset=["eps_last", "eps_prior_next", "g_last", "growth_next"]).copy()
P["cur"], P["tp"] = P.eps_last, P.eps_prior_next
P["rr"] = P.rr.clip(-300, 300); P["gl"] = P.g_last.clip(-300, 300)
P["ratio"] = (P.cur / P.tp.abs().clip(lower=0.05)).clip(-20, 20)
P["sc"], P["st"] = np.sign(P.cur), np.sign(P.tp)
P["ltp"] = np.log(P.tp.abs().clip(lower=0.01))
P["y"] = P.growth_next.clip(-300, 300); P["up"] = (P.up_next == 1).astype(int)
FEATS = ["rr", "gl", "ratio", "sc", "st", "ltp", "cur"]
tr, te = P[P.pe_next <= "2018-12-31"], P[P.pe_next >= "2019-01-01"]
print(f"train {len(tr)} test {len(te)}")
# ---- direction
pu = tr.groupby("b").up_next.mean()
rule_lab = te.b.map(lambda b: pu.get(b, 0.55) >= 0.5).values
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
clf.fit(tr[FEATS], tr.up)
p = clf.predict_proba(te[FEATS])[:, 1]
res = {}
for nm, sel in (("ALL", np.ones(len(te), bool)), ("revenue >= $1B", te.big.values), ("financials", te.fin.values)):
    u = te.up.values[sel]
    rule = np.mean(rule_lab[sel] == (u == 1)); ml = np.mean((p[sel] > 0.5) == (u == 1))
    blend = np.mean(((0.5 * p[sel] + 0.5 * te.b.map(lambda b: pu.get(b, 0.55)).values[sel]) > 0.5) == (u == 1))
    print(f"direction {nm:15s} n={int(sel.sum()):5d}: rule {rule:.3f} | LightGBM {ml:.3f} | 50/50 blend {blend:.3f}")
# ---- growth (L1 loss = MAE, the scorer's regression leg)
reg = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
reg.fit(tr[FEATS], tr.y)
pm = reg.predict(te[FEATS]); pr = te.b.map(WB).values * te.rr.values
for nm, sel in (("ALL", np.ones(len(te), bool)), ("revenue >= $1B", te.big.values), ("financials", te.fin.values)):
    y = te.y.values[sel]; mae0 = np.abs(y).mean()
    sk = lambda q: mae0 / (mae0 + np.abs(y - q[sel]).mean())
    print(f"growth {nm:15s}: skill rule {sk(pr):.3f} | LightGBM(L1) {sk(pm):.3f} | blend .5 {sk(0.5 * pr + 0.5 * pm):.3f} | blend .25 ML {sk(0.75 * pr + 0.25 * pm):.3f}")
# by-year stability of direction
for yr, g in te.groupby(te.pe_next.dt.year):
    idx = te.index.get_indexer(g.index); u = g.up.values
    print(f"  year {yr}: n={len(g)} rule {np.mean(rule_lab[idx] == (u == 1)):.3f} LightGBM {np.mean((p[idx] > 0.5) == (u == 1)):.3f}")
imp = dict(zip(FEATS, clf.feature_importances_)); print("feature importance (classifier splits):", imp)
