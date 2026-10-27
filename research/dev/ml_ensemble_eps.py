"""ML-H: ensembles of quantile models for the EPS growth interval (same data/split as ml_intervals.py: revenue >= $1B, |prior| >= 0.5, train <= 2018, test 2019+).
Models: LightGBM, CatBoost (Quantile), XGBoost (quantile error), sklearn HistGradientBoosting (quantile), linear QuantileRegressor. Ensembles average the endpoints.
Also: (a) validation-slice conformal widening to hit 90% coverage, (b) richer features (net income / revenue / margins / shares) as an upper bound of what extra parsing could add."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

src = open("t4-local/dev/ml_intervals.py").read().split("print(f\"train")[0]
exec(src)
IS = lambda lo, hi, y: np.mean((hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0))
y = te.y.values; pt = te.pt.values
def run(name, fit_pred, feats, trn=tr, tst=te):
    out = {}
    for a in (0.03, 0.97): out[a] = fit_pred(trn[feats].values, trn.res.values, tst[feats].values, a)
    lo, hi = np.minimum(out[0.03], out[0.97]), np.maximum(out[0.03], out[0.97])
    return lo, hi
def f_lgb(X, r, Xt, a): return lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1).fit(X, r).predict(Xt)
def f_cat(X, r, Xt, a): return CatBoostRegressor(loss_function=f"Quantile:alpha={a}", iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=5, verbose=0, random_seed=3).fit(X, r).predict(Xt)
def f_xgb(X, r, Xt, a): return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=a, n_estimators=200, learning_rate=0.05, max_depth=3, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, verbosity=0).fit(X, r).predict(Xt)
def f_hgb(X, r, Xt, a): return HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=200, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=5, random_state=3).fit(X, r).predict(Xt)
def f_lin(X, r, Xt, a):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return QuantileRegressor(quantile=a, alpha=0.0005, solver="highs").fit((X - mu) / sd, r).predict((Xt - mu) / sd)
qq = {b: np.percentile(tr.res[tr.b == b], [3, 97]) for b in WB}
bucket_lo = te.pt.values + np.array([qq[b][0] for b in te.b]); bucket_hi = te.pt.values + np.array([qq[b][1] for b in te.b])
print(f"train {len(tr)} test {len(te)} | bucket bands: IS {IS(bucket_lo, bucket_hi, y):.1f} cov {np.mean((y >= bucket_lo) & (y <= bucket_hi)):.3f}")
B = {}
for nm, f in (("LightGBM", f_lgb), ("CatBoost", f_cat), ("XGBoost", f_xgb), ("HistGB", f_hgb), ("Linear quantile", f_lin)):
    lo, hi = run(nm, f, FEATS); B[nm] = (pt + lo, pt + hi)
    print(f"{nm:18s}: IS {IS(pt + lo, pt + hi, y):7.1f} cov {np.mean((y >= pt + lo) & (y <= pt + hi)):.3f} width {np.mean(hi - lo):6.1f}")
def ens(names):
    lo = np.mean([B[n][0] for n in names], axis=0); hi = np.mean([B[n][1] for n in names], axis=0)
    return lo, hi
for names in (["LightGBM", "CatBoost"], ["LightGBM", "CatBoost", "XGBoost"], ["LightGBM", "CatBoost", "XGBoost", "HistGB"], ["LightGBM", "CatBoost", "XGBoost", "HistGB", "Linear quantile"], ["CatBoost", "XGBoost", "HistGB"]):
    lo, hi = ens(names)
    print(f"ENSEMBLE {'+'.join(n[:4] for n in names):26s}: IS {IS(lo, hi, y):7.1f} cov {np.mean((y >= lo) & (y <= hi)):.3f} width {np.mean(hi - lo):6.1f}")
# median-of-endpoints and trimmed
lo = np.median([B[n][0] for n in B], axis=0); hi = np.median([B[n][1] for n in B], axis=0)
print(f"ENSEMBLE median of 5          : IS {IS(lo, hi, y):7.1f} cov {np.mean((y >= lo) & (y <= hi)):.3f}")
# ---- (b) richer features (upper bound): net income / revenue / margin / shares from the pit data
import glob
cols = ["cik", "period_end", "quarters", "form", "knowledge_date", "revision", "net_income", "revenue", "operating_income", "shares_diluted_weighted"]
fx = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
fx = fx[(fx.knowledge_date <= LIMIT) & (fx.quarters == 1) & fx.form.astype(str).str.startswith("10-Q")]
fx = fx.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
fx["period_end"] = pd.to_datetime(fx.period_end).astype("datetime64[ns]")
a_ = fx.rename(columns={"net_income": "ni_L", "revenue": "rev_L", "operating_income": "oi_L", "shares_diluted_weighted": "sh_L"}).drop(columns=["knowledge_date", "revision", "form", "quarters"])
S3 = S.copy(); S3["key"] = (S3.pe_next - pd.Timedelta(days=91)).astype("datetime64[ns]")
S3 = pd.merge_asof(S3.sort_values("key"), a_.sort_values("period_end").rename(columns={"period_end": "pe_last"}), left_on="key", right_on="pe_last", by="cik", tolerance=pd.Timedelta(days=12), direction="nearest")
b_ = fx.rename(columns={"net_income": "ni_P", "revenue": "rev_P", "operating_income": "oi_P", "shares_diluted_weighted": "sh_P"}).drop(columns=["knowledge_date", "revision", "form", "quarters"])
S3["key2"] = (S3.pe_next - pd.DateOffset(years=1)).astype("datetime64[ns]")
S3 = pd.merge_asof(S3.sort_values("key2"), b_.sort_values("period_end").rename(columns={"period_end": "pe_py"}), left_on="key2", right_on="pe_py", by="cik", tolerance=pd.Timedelta(days=20), direction="nearest")
S3["margin_L"] = S3.oi_L / S3.rev_L; S3["dmargin"] = S3.margin_L - S3.oi_P / S3.rev_P
S3["rev_g"] = (S3.rev_L / S3.rev_P - 1).clip(-1, 3); S3["sh_g"] = (S3.sh_L / S3.sh_P - 1).clip(-0.5, 0.5); S3["ni_rr"] = ((S3.ni_L - S3.ni_P) / S3.ni_P.abs().clip(lower=1e6)).clip(-5, 5)
S3["lrev"] = np.log(S3.rev_L.clip(lower=1e6))
EXTRA = FEATS + ["margin_L", "dmargin", "rev_g", "sh_g", "ni_rr", "lrev"]
trn, tst = S3[S3.pe_next <= "2018-12-31"], S3[S3.pe_next >= "2019-01-01"]
yy = tst.y.values; ptt = tst.pt.values
for nm, feats in (("current features (5)", FEATS), ("+ margin / revenue / shares / NI (11)", EXTRA)):
    lo, hi = run(nm, f_lgb, feats, trn, tst)
    print(f"LightGBM {nm:40s}: IS {IS(ptt + lo, ptt + hi, yy):7.1f} cov {np.mean((yy >= ptt + lo) & (yy <= ptt + hi)):.3f} (n test {len(tst)})")
