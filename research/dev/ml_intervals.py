"""ML-F: feature-dependent intervals by quantile regression (LightGBM) vs the shipped bucket bands, interval score and coverage on the held-out period.
EPS: normal-EPS large caps (revenue >= $1B, |prior| >= 0.5), target growth_next in percent, band = point + residual quantiles. Train <= 2018, test 2019+."""
import numpy as np, pandas as pd
import lightgbm as lgb

exec(open("t4-local/dev/eps_runrate_fit.py").read().split("# direction")[0])
WB = {'-10..0': 0.0, '-50..-10': 0.5, '0..10': 1.25, '10..30': 1.0, '<=-50': 0.5, '>30': 1.0}
P = P.dropna(subset=["eps_last", "eps_prior_next", "g_last", "growth_next"]).copy()
P["tp"] = P.eps_prior_next; S = P[P.big & (P.tp.abs() >= 0.5)].copy()
S["rr"] = S.rr.clip(-300, 300); S["gl"] = S.g_last.clip(-300, 300); S["ratio"] = (S.eps_last / S.tp.abs().clip(lower=0.05)).clip(-20, 20)
S["ltp"] = np.log(S.tp.abs()); S["y"] = S.growth_next.clip(-1000, 1000); S["pt"] = S.b.map(WB) * S.rr; S["res"] = S.y - S.pt
FEATS = ["rr", "gl", "ratio", "ltp", "eps_last"]
tr, te = S[S.pe_next <= "2018-12-31"], S[S.pe_next >= "2019-01-01"]
def IS(y, lo, hi): return np.mean((hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0))
print(f"train {len(tr)} test {len(te)}")
qq = {b: np.percentile(tr.res[tr.b == b], [3, 97]) for b in WB}
lo = te.pt + te.b.map(lambda b: qq[b][0]); hi = te.pt + te.b.map(lambda b: qq[b][1])
print(f"shipped bucket bands (3/97): IS {IS(te.y.values, lo.values, hi.values):7.1f} cov {np.mean((te.y >= lo) & (te.y <= hi)):.3f} width {np.mean(hi - lo):6.1f}")
for leaves, mcs, n in ((7, 200, 200), (15, 100, 300), (31, 50, 400)):
    q = {}
    for a in (0.03, 0.97):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=n, learning_rate=0.03, num_leaves=leaves, min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
        m.fit(tr[FEATS], tr.res); q[a] = m.predict(te[FEATS])
    lo2, hi2 = te.pt.values + np.minimum(q[0.03], q[0.97]), te.pt.values + np.maximum(q[0.03], q[0.97])
    print(f"LightGBM quantile leaves={leaves:2d} mcs={mcs:3d} n={n}: IS {IS(te.y.values, lo2, hi2):7.1f} cov {np.mean((te.y.values >= lo2) & (te.y.values <= hi2)):.3f} width {np.mean(hi2 - lo2):6.1f}")
    # blend of the two interval sets (average the endpoints)
    lob, hib = 0.5 * lo2 + 0.5 * lo.values, 0.5 * hi2 + 0.5 * hi.values
    print(f"      50/50 blend with bucket bands: IS {IS(te.y.values, lob, hib):7.1f} cov {np.mean((te.y.values >= lob) & (te.y.values <= hib)):.3f}")
# by year (best config) coverage

# ---- by-year stability for the chosen config (7 leaves) and the final export
q = {}
for a in (0.03, 0.97):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
    m.fit(tr[FEATS], tr.res); q[a] = m.predict(te[FEATS])
lo2, hi2 = te.pt.values + np.minimum(q[0.03], q[0.97]), te.pt.values + np.maximum(q[0.03], q[0.97])
lo1 = (te.pt + te.b.map(lambda b: qq[b][0])).values; hi1 = (te.pt + te.b.map(lambda b: qq[b][1])).values
y = te.y.values
for yr in sorted(te.pe_next.dt.year.unique()):
    mk = (te.pe_next.dt.year == yr).values
    print(f"year {yr}: n={int(mk.sum()):4d} LGBM IS {IS(y[mk], lo2[mk], hi2[mk]):7.1f} cov {np.mean((y[mk] >= lo2[mk]) & (y[mk] <= hi2[mk])):.2f} | bucket IS {IS(y[mk], lo1[mk], hi1[mk]):7.1f} cov {np.mean((y[mk] >= lo1[mk]) & (y[mk] <= hi1[mk])):.2f}")
for nm, mk in (("financials", te.fin.values), ("non-financials", ~te.fin.values)):
    print(f"{nm}: n={int(mk.sum())} LGBM IS {IS(y[mk], lo2[mk], hi2[mk]):7.1f} cov {np.mean((y[mk] >= lo2[mk]) & (y[mk] <= hi2[mk])):.2f} | bucket IS {IS(y[mk], lo1[mk], hi1[mk]):7.1f} cov {np.mean((y[mk] >= lo1[mk]) & (y[mk] <= hi1[mk])):.2f}")
