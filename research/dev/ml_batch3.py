"""ML-I: (1) rates intervals by quantile GBM, (2) macro-revision intervals by quantile GBM, (3) EPS direction ensemble, (4) credit ensemble (AUC)."""
import warnings
warnings.filterwarnings("ignore")
import json, glob
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

IS = lambda lo, hi, y: np.mean((hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0))
def lq(X, r, Xt, a, leaves=7, mcs=100, n=200):
    return lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=n, learning_rate=0.03, num_leaves=leaves, min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1).fit(X, r).predict(Xt)
def cq(X, r, Xt, a): return CatBoostRegressor(loss_function=f"Quantile:alpha={a}", iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=5, verbose=0, random_seed=3).fit(X, r).predict(Xt)
def report(name, y, lo0, hi0, lo1, hi1):
    print(f"{name:40s} shipped IS {IS(lo0, hi0, y):10.4f} cov {np.mean((y >= lo0) & (y <= hi0)):.3f} | ML IS {IS(lo1, hi1, y):10.4f} cov {np.mean((y >= lo1) & (y <= hi1)):.3f} | {IS(lo1, hi1, y) / IS(lo0, hi0, y) - 1:+.1%}")

# ------------- (1) rates
exec(open("t4-local/dev/rates_batch2.py").read().split('print("--- shipped')[0])
R["unit"] = (0.5 * R.sd10 + 0.5 * 6.5) * np.sqrt(R.H); R["u"] = R.y / R.unit
R["mat"] = R.m.map({m: i for i, m in enumerate(MATS)})
tr, te = R[R.year <= 2012], R[R.year >= 2013]
F = ["sd5", "sd10", "sd20", "sd60", "ew10", "lvl", "dpol", "H", "mat", "cross"]
lo = np.minimum(lq(tr[F], tr.u, te[F], 0.05, mcs=300), lq(tr[F], tr.u, te[F], 0.95, mcs=300)) * te.unit.values
hi = np.maximum(lq(tr[F], tr.u, te[F], 0.05, mcs=300), lq(tr[F], tr.u, te[F], 0.95, mcs=300)) * te.unit.values
report("rates (test 2013-22, bp)", te.y.values, -1.645 * te.unit.values, 1.645 * te.unit.values, lo, hi)
for g, mask in (("easing", te.dpol < -0.1), ("tightening", te.dpol > 0.1), ("neutral", te.dpol.abs() <= 0.1)):
    m = mask.values
    report(f"   rates {g}", te.y.values[m], -1.645 * te.unit.values[m], 1.645 * te.unit.values[m], lo[m], hi[m])

# ------------- (2) macro revisions (relative revisions of young estimates, non-zero)
exec(open("t4-local/dev/macrorev_batch2.py").read().split('NZ = R[R.rel != 0]')[0])
NZ = R[R.rel != 0].copy(); NZ["ser"] = NZ.s.map({s: i for i, s in enumerate(sorted(NZ.s.unique()))}); NZ["month"] = NZ.v.dt.month
tr, te = NZ[NZ.v.dt.year <= 2016], NZ[NZ.v.dt.year >= 2017]
F = ["sd", "share", "med", "age", "ser", "month"]
KS = {"DGORDER": 0.8, "HOUST": 2.0, "RSAFS": 1.645, "INDPRO": 1.3, "PAYEMS": 1.3, "PI": 1.3}
k = te.s.map(KS).values; sh = 0.25 * te.sd.values
lo0, hi0 = sh - k * te.sd.values, sh + k * te.sd.values
rel_tr = tr.rel / tr.sd
lo1 = np.minimum(lq(tr[F], rel_tr, te[F], 0.05, mcs=60), lq(tr[F], rel_tr, te[F], 0.95, mcs=60)) * te.sd.values
hi1 = np.maximum(lq(tr[F], rel_tr, te[F], 0.05, mcs=60), lq(tr[F], rel_tr, te[F], 0.95, mcs=60)) * te.sd.values
report("macro revisions (test 2017-22, relative)", te.rel.values, lo0, hi0, lo1, hi1)

# ------------- (3) EPS direction ensemble
exec(open("t4-local/dev/ml_eps.py").read().split("# ---- direction")[0])
tr, te = P[P.pe_next <= "2018-12-31"], P[P.pe_next >= "2019-01-01"]
pu = tr.groupby("b").up_next.mean(); rule_p = te.b.map(lambda b: pu.get(b, 0.55)).values
lg = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1).fit(tr[FEATS], tr.up)
cb = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, l2_leaf_reg=5, verbose=0, random_seed=3).fit(tr[FEATS], tr.up)
Xl = lambda d: np.column_stack([np.sign(d.rr) * np.log1p(d.rr.abs()), np.sign(d.gl) * np.log1p(d.gl.abs()), d.sc, d.st, np.sign(d.ratio) * np.log1p(d.ratio.abs())])
lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xl(tr), tr.up)
probs = {"rule": rule_p, "LightGBM": lg.predict_proba(te[FEATS])[:, 1], "CatBoost": cb.predict_proba(te[FEATS])[:, 1], "logit": lr.predict_proba(Xl(te))[:, 1]}
probs["ens(LGBM+Cat)"] = (probs["LightGBM"] + probs["CatBoost"]) / 2
probs["ens(all 4)"] = np.mean([probs[k] for k in ("rule", "LightGBM", "CatBoost", "logit")], axis=0)
for nm, sel in (("ALL", np.ones(len(te), bool)), ("revenue >= $1B", te.big.values)):
    u = te.up.values[sel]
    print("direction " + nm + ": " + " | ".join(f"{k} {np.mean((v[sel] > 0.5) == (u == 1)):.3f}" for k, v in probs.items()))

# ------------- (4) credit ensemble (AUC, grouped CV)
D = pd.read_csv("data_sec_credit/phrases.csv"); idx = json.load(open("data_sec_credit/index.json")); D["cik"] = D.acc.map(lambda a: idx[a]["cik"])
pit = pd.concat([pd.read_parquet(f, columns=["accession_number", "total_equity", "net_income"]) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))]).drop_duplicates("accession_number").set_index("accession_number")
D["neg_eq"] = D.acc.map(lambda a: pit.total_equity.get(a, np.nan) < 0).astype(int); D["loss"] = D.acc.map(lambda a: pit.net_income.get(a, np.nan) < 0).astype(int)
cols = [c for c in D.columns if c[:2].isdigit()]
X = D[cols].astype(int).assign(neg_eq=D.neg_eq, loss=D.loss); X.columns = [("p" + c.split(" ")[0]) if c[:2].isdigit() else c for c in X.columns]
yv = D.label.values; oof = {k: np.zeros(len(D)) for k in ("logit", "lgbm", "cat", "rf")}
for a, b in GroupKFold(5).split(X, yv, D.cik):
    oof["logit"][b] = LogisticRegression(C=0.3, max_iter=3000).fit(X.iloc[a], yv[a]).predict_proba(X.iloc[b])[:, 1]
    oof["lgbm"][b] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1).fit(X.iloc[a], yv[a]).predict_proba(X.iloc[b])[:, 1]
    oof["cat"][b] = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=5, verbose=0, random_seed=3).fit(X.iloc[a], yv[a]).predict_proba(X.iloc[b])[:, 1]
    oof["rf"][b] = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt", random_state=3, n_jobs=-1).fit(X.iloc[a], yv[a]).predict_proba(X.iloc[b])[:, 1]
rank = lambda v: pd.Series(v).rank().values / len(v)
oof["ens(logit+cat+rf+lgbm)"] = np.mean([rank(oof[k]) for k in ("logit", "lgbm", "cat", "rf")], axis=0)
oof["ens(logit+cat)"] = np.mean([rank(oof[k]) for k in ("logit", "cat")], axis=0)
print("credit AUC (5-fold grouped CV): " + " | ".join(f"{k} {roc_auc_score(yv, v):.3f}" for k, v in oof.items()))
