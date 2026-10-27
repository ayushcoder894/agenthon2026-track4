"""Counterfactual scorer audit for EPS growth (regression): composite = 0.7 * Q_pred + 0.3 * min(Q_int, max(0.5, Q_pred)), Q_pred = naiveMAE/(naiveMAE+MAE), Q_int = naiveIS/(naiveIS+IS).
Naive = 0% growth point with a symmetric band of width m x (90% quantile of |growth|) for m in {0.5, 1, 2} (the organizers' naive interval is hidden, so this brackets it).
Evaluated per bootstrap 'unit' of 8 random test companies (like a real unit), averaged over 2000 draws, test 2019-22."""
import numpy as np, pandas as pd
import lightgbm as lgb
src = open("t4-local/dev/ml_intervals.py").read().split("print(f\"train")[0]
exec(src)
IS = lambda lo, hi, y: (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
q = {}
for a in (0.03, 0.97):
    q[a] = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1).fit(tr[FEATS], tr.res).predict(te[FEATS])
y = te.y.values; pt = te.pt.values
qq = {b: np.percentile(tr.res[tr.b == b], [3, 97]) for b in WB}
bk_lo = pt + np.array([qq[b][0] for b in te.b]); bk_hi = pt + np.array([qq[b][1] for b in te.b])
gb_lo = pt + np.minimum(q[0.03], q[0.97]); gb_hi = pt + np.maximum(q[0.03], q[0.97])
old_lo = pt + np.array([-1012 / 2] * len(te)); old_hi = pt + 1012 / 2    # the D10R-style over-wide band (width ~1012)
rng = np.random.default_rng(1)
base_w = np.percentile(np.abs(y), 90)
print(f"test rows {len(te)}; naive 90% quantile of |growth| = {base_w:.0f}%  | our MAE {np.abs(y - pt).mean():.1f} vs naive MAE {np.abs(y).mean():.1f} -> Q_pred(all) = {np.abs(y).mean() / (np.abs(y).mean() + np.abs(y - pt).mean()):.3f}")
for m in (0.5, 1.0, 2.0):
    nlo, nhi = -m * base_w * np.ones(len(te)), m * base_w * np.ones(len(te))
    res = {k: [] for k in ("D10R-style wide", "bucket bands", "LightGBM bands")}
    capped = {k: 0 for k in res}; n = 0
    for _ in range(2000):
        i = rng.choice(len(te), 8, replace=False)
        nm = np.abs(y[i]).mean(); mae = np.abs(y[i] - pt[i]).mean(); Qp = nm / (nm + mae)
        nis = IS(nlo[i], nhi[i], y[i]).mean()
        for k, (lo, hi) in (("D10R-style wide", (old_lo, old_hi)), ("bucket bands", (bk_lo, bk_hi)), ("LightGBM bands", (gb_lo, gb_hi))):
            own = IS(lo[i], hi[i], y[i]).mean(); raw = nis / (nis + own); cap = max(0.5, Qp)
            res[k].append(0.7 * Qp + 0.3 * min(raw, cap)); capped[k] += raw > cap
        n += 1
    print(f"naive band x{m}: " + " | ".join(f"{k}: composite {np.mean(v):.3f} (cap binds {capped[k] / n:.0%})" for k, v in res.items()))
