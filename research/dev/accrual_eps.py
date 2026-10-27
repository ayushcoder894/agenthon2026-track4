"""Sloan-style test: do accruals / cash conversion add to the shipped EPS run-rate rule? Per-filing YTD net income & operating cash flow (same row), assets from the balance-sheet row.
accr = (NI_ytd - OCF_ytd) / assets ; cconv = OCF_ytd / max(|NI_ytd|, 1e6) clipped. Train outcomes <= 2018, test 2019+. Direction: logistic on rule signals vs + accrual features;
growth: LightGBM L1 on the rule's residual vs + features. Company-grouped bootstrap and a shuffled-feature control."""
import glob, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

cols = ["accession_number", "cik", "period_end", "quarters", "form", "knowledge_date", "revision", "net_income", "operating_cash_flow", "total_assets"]
pit = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
pit = pit[(pit.knowledge_date <= "2022-06-28") & pit.form.astype(str).str.startswith("10-Q")].sort_values(["accession_number", "revision"])
rows = []
for acc, g in pit.groupby("accession_number", sort=False):
    ytd = g[g.quarters >= 1].sort_values("quarters").iloc[-1] if (g.quarters >= 1).any() else None
    if ytd is None: continue
    a = g.total_assets.dropna()
    rows.append((acc, g.cik.iloc[0], g.period_end.iloc[0], ytd.net_income, ytd.operating_cash_flow, a.iloc[0] if len(a) else np.nan, int(ytd.quarters)))
F = pd.DataFrame(rows, columns=["acc", "cik", "period_end", "ni", "ocf", "assets", "qmax"])
F["period_end"] = pd.to_datetime(F.period_end).astype("datetime64[ns]")
F["accr"] = ((F.ni - F.ocf) / F.assets).clip(-0.25, 0.25); F["cconv"] = (F.ocf / F.ni.abs().clip(lower=1e6)).clip(-5, 5)
print("filing rows", len(F), "| accr available", round(F.accr.notna().mean(), 3), "| cconv available", round(F.cconv.notna().mean(), 3))
exec(open("t4-local/dev/ml_eps.py").read().split("# ---- direction")[0])             # P with rr, gl, ratio, ltp, sc, st, up, y, b, big, fin ...
Fk = F.drop_duplicates(["cik", "period_end"]).sort_values("period_end")[["cik", "period_end", "accr", "cconv"]]
P["period_end"] = pd.to_datetime(P.period_end).astype("datetime64[ns]"); P["cik"] = P.cik.astype(str).str.zfill(10); Fk["cik"] = Fk.cik.astype(str).str.zfill(10)
P2 = pd.merge_asof(P.sort_values("period_end"), Fk, on="period_end", by="cik", tolerance=pd.Timedelta(days=5), direction="nearest")
P2["d_accr"] = np.nan
print("test-set coverage of accruals:", round(P2[P2.pe_next >= '2019-01-01'].accr.notna().mean(), 3), "| train", round(P2[P2.pe_next <= '2018-12-31'].accr.notna().mean(), 3))
D = P2.dropna(subset=["accr", "cconv"]).copy(); D["srr"] = np.sign(D.rr) * np.log1p(D.rr.abs()); D["sgl"] = np.sign(D.gl) * np.log1p(D.gl.abs())
tr, te = D[D.pe_next <= "2018-12-31"], D[D.pe_next >= "2019-01-01"]
print(f"usable rows: train {len(tr)} test {len(te)} (all companies) | big revenue test {int(te.big.sum())}")
B = ["srr", "sgl", "st", "sc"]; X = B + ["accr", "cconv"]
def fit_eval(cols, sub=None):
    t = tr if sub is None else tr[sub[0]]; s = te if sub is None else te[sub[1]]
    mu, sd = t[cols].mean(), t[cols].std() + 1e-9
    m = LogisticRegression(C=1.0, max_iter=2000).fit(((t[cols] - mu) / sd), t.up); p = m.predict_proba((s[cols] - mu) / sd)[:, 1]
    return p, s
for nm, sub in (("ALL", None), ("revenue >= $1B", (tr.big.values, te.big.values))):
    p0, s = fit_eval(B, sub); p1, _ = fit_eval(X, sub); y = s.up.values
    rng = np.random.default_rng(1); g = s.cik.values; ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}; dd = []
    for _ in range(400):
        ii = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]); dd.append(np.mean((p1[ii] > .5) == y[ii]) - np.mean((p0[ii] > .5) == y[ii]))
    t_ = tr if sub is None else tr[sub[0]]; rs = np.random.default_rng(2); sh = []
    for _ in range(10):
        tt = t_.copy(); tt[["accr", "cconv"]] = tt[["accr", "cconv"]].values[rs.permutation(len(tt))]
        mu, sd = tt[X].mean(), tt[X].std() + 1e-9; m = LogisticRegression(C=1.0, max_iter=2000).fit((tt[X] - mu) / sd, tt.up); ps = m.predict_proba((s[X] - mu) / sd)[:, 1]
        sh.append(np.mean((ps > .5) == y) - np.mean((p0 > .5) == y))
    print(f"direction {nm:15s} n={len(s):5d}: rule signals acc {np.mean((p0 > .5) == y):.4f} AUC {roc_auc_score(y, p0):.4f} | + accruals acc {np.mean((p1 > .5) == y):.4f} AUC {roc_auc_score(y, p1):.4f} | delta acc {np.mean((p1 > .5) == y) - np.mean((p0 > .5) == y):+.4f} (90% CI {np.percentile(dd, 5):+.4f}, {np.percentile(dd, 95):+.4f}) | shuffled control {np.mean(sh):+.4f}")
# growth: LightGBM L1 residual model
def gr(cols):
    m = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1).fit(tr[cols], tr.y); return m.predict(te[cols])
y = te.y.values; mae0 = np.abs(y).mean(); sk = lambda p: mae0 / (mae0 + np.abs(y - p).mean())
print(f"growth (all companies): skill rule-features LightGBM {sk(gr(['rr', 'gl', 'ratio', 'ltp', 'sc', 'st'])):.4f} | + accruals {sk(gr(['rr', 'gl', 'ratio', 'ltp', 'sc', 'st', 'accr', 'cconv'])):.4f}")
