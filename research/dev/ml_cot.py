"""ML-E: COT ranking. Features readable at run time from a unit's own 26-week table: npo (net %OI), mean26, sd26, d1, d4, d13. Target: 5-week change in net %OI.
Train origins <= 2015, test 2016..2022-06; mean cross-sectional Spearman over the 40 largest-OI markets per week (as scored), vs the shipped signal -npo."""
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
exec(open("t4-local/dev/cot_study.py").read().split("SIG = ")[0])
g = D.groupby("mkt")
D["d13"] = g.npo.diff(13)
D["rng26"] = g.npo.transform(lambda x: x.rolling(26).max() - x.rolling(26).min())
D["pos26"] = (D.npo - g.npo.transform(lambda x: x.rolling(26).min())) / (D.rng26 + 1e-9)
D = D.dropna(subset=["y", "mean26", "sd26", "d4", "d13", "rng26", "pos26", "sddiff"])
F = ["npo", "mean26", "sd26", "d1", "d4", "d13", "pos26", "sddiff"]
tr, te = D[D.date.dt.year <= 2015], D[D.date.dt.year >= 2016]
def xs_spearman(df, score):
    df = df.assign(s=score); r = []
    for t, x in df.groupby("date"):
        x = x.nlargest(40, "oi")
        if len(x) >= 15: r.append(spearmanr(x.s, x.y)[0])
    return np.nanmean(r), len(r)
base, n = xs_spearman(te, -te.npo)
print(f"shipped signal (-net %OI): mean Spearman {base:+.3f} over {n} test weeks")
for obj in ("l2", "l1"):
    gb = lgb.LGBMRegressor(objective=obj, n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1)
    gb.fit(tr[F], tr.y); p = gb.predict(te[F])
    m, _ = xs_spearman(te, p); mb, _ = xs_spearman(te, 0.5 * (-te.npo) / te.npo.std() + 0.5 * p / np.std(p))
    print(f"LightGBM {obj}: mean Spearman {m:+.3f} | 50/50 blend with -npo {mb:+.3f}")
print("importance:", dict(zip(F, gb.feature_importances_)))
