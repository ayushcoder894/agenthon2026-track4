"""ML-B: credit. Runtime-readable features only: 16 phrase flags (+ counts) and the two parsed statement facts (negative equity, net loss).
207 positives (last statement before a bankruptcy) vs 698 controls. 5-fold CV grouped by company. Compare the shipped rule vs logistic regression vs LightGBM (AUC and the
operating point of the rule)."""
import glob, json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

D = pd.read_csv("data_sec_credit/phrases.csv")
idx = json.load(open("data_sec_credit/index.json"))
D["cik"] = D.acc.map(lambda a: idx[a]["cik"])
pit = pd.concat([pd.read_parquet(f, columns=["accession_number", "total_equity", "net_income"]) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
pit = pit.drop_duplicates("accession_number").set_index("accession_number")
D["neg_eq"] = D.acc.map(lambda a: pit.total_equity.get(a, np.nan) < 0).astype(int)
D["loss"] = D.acc.map(lambda a: pit.net_income.get(a, np.nan) < 0).astype(int)
cols = [c for c in D.columns if c[:2].isdigit()]
strong = [c for c in cols if c.split(" ")[0] in ("65", "68", "69", "70", "71", "73", "76", "77")]
D["strong_n"] = D[strong].sum(axis=1)
D["gc"] = D[[c for c in cols if c.startswith("63")][0]].astype(int)
D["rule"] = ((D.gc == 1) | (D.strong_n >= 3) | ((D.neg_eq == 1) & (D.loss == 1))).astype(int)
X = D[cols].astype(int).assign(neg_eq=D.neg_eq, loss=D.loss, strong_n=D.strong_n)
X.columns = [("p" + c.split(" ")[0]) if c[:2].isdigit() else c for c in X.columns]
y = D.label.values
print(f"n={len(D)} positives {int(y.sum())}")
print(f"shipped rule: sens {D.rule[y == 1].mean():.3f} spec {1 - D.rule[y == 0].mean():.3f}")
oof = {"logit": np.zeros(len(D)), "lgbm": np.zeros(len(D)), "rule": D.rule.values.astype(float)}
for trn, tst in GroupKFold(5).split(X, y, D.cik):
    lr = LogisticRegression(C=0.3, max_iter=2000).fit(X.iloc[trn], y[trn]); oof["logit"][tst] = lr.predict_proba(X.iloc[tst])[:, 1]
    gb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1).fit(X.iloc[trn], y[trn])
    oof["lgbm"][tst] = gb.predict_proba(X.iloc[tst])[:, 1]
for k, v in oof.items(): print(f"AUC {k:6s}: {roc_auc_score(y, v):.3f}")
# operating points: match the rule's specificity and compare sensitivity
thr_spec = 1 - D.rule[y == 0].mean()
for k in ("logit", "lgbm"):
    v = oof[k]; t = np.quantile(v[y == 0], thr_spec)
    print(f"{k}: at the rule's specificity ({thr_spec:.3f}) sensitivity {np.mean(v[y == 1] > t):.3f} (rule {D.rule[y == 1].mean():.3f})")
