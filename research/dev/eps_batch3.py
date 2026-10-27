"""Experiments 41-43 (EPS direction): extra pre-release features on top of the run-rate sign, test 2019-2022Q1.
41 net-income run-rate (latest-quarter NI vs prior-year NI of the target quarter) vs EPS run-rate (share-count effect)
42 operating margin change YoY of the latest quarter as a tie-breaker when |rr| is small
43 diluted share-count change YoY (buybacks) as a tie-breaker when |rr| is small"""
import glob
import numpy as np
import pandas as pd

exec(open("t4-local/dev/eps_runrate_fit.py").read().split("# direction")[0])
cols = ["cik", "period_end", "period_kind", "quarters", "form", "knowledge_date", "revision", "net_income", "operating_income", "revenue", "shares_diluted_weighted"]
fx = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
fx = fx[(fx.knowledge_date <= LIMIT) & (fx.quarters == 1) & fx.form.astype(str).str.startswith("10-Q")]
fx = fx.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
fx["period_end"] = pd.to_datetime(fx.period_end).astype("datetime64[ns]")
# last quarter's features: join on (cik, period_end of the 'last' quarter = pe_next - ~91d); prior-year same-quarter values
a = fx.rename(columns={c: c + "_L" for c in ("net_income", "operating_income", "revenue", "shares_diluted_weighted")}).drop(columns=["knowledge_date", "revision", "form", "quarters", "period_kind"])
P["key_last"] = (P.pe_next - pd.Timedelta(days=91)).astype("datetime64[ns]")
P3 = pd.merge_asof(P.sort_values("key_last"), a.sort_values("period_end").rename(columns={"period_end": "pe_last"}), left_on="key_last", right_on="pe_last", by="cik", tolerance=pd.Timedelta(days=12), direction="nearest")
b = fx.rename(columns={c: c + "_P" for c in ("net_income", "operating_income", "revenue", "shares_diluted_weighted")}).drop(columns=["knowledge_date", "revision", "form", "quarters", "period_kind"])
P3["key_py"] = (P3.pe_next - pd.DateOffset(years=1)).astype("datetime64[ns]")
P3 = pd.merge_asof(P3.sort_values("key_py"), b.sort_values("period_end").rename(columns={"period_end": "pe_py"}), left_on="key_py", right_on="pe_py", by="cik", tolerance=pd.Timedelta(days=20), direction="nearest")
P3["ni_rr"] = ((P3.net_income_L - P3.net_income_P) / P3.net_income_P.abs().clip(lower=1e6) * 100).clip(-300, 300)
P3["om_L"] = P3.operating_income_L / P3.revenue_L; P3["om_P"] = P3.operating_income_P / P3.revenue_P
P3["dmargin"] = (P3.om_L - P3.om_P) * 100
P3["dshares"] = (P3.shares_diluted_weighted_L / P3.shares_diluted_weighted_P - 1) * 100
tr, te = P3[P3.pe_next <= "2018-12-31"], P3[P3.pe_next >= "2019-01-01"]
for nm, S in (("ALL", te), ("BIG", te[te.big]), ("FIN", te[te.fin])):
    up = (S.up_next == 1).values
    base_acc = np.mean((S.rr > 0).values == up)
    r = {"rr>0": base_acc}
    h = S.dropna(subset=["ni_rr"]); u = (h.up_next == 1).values
    r["NI rr>0 (n=%d)" % len(h)] = np.mean((h.ni_rr > 0).values == u); r["EPS rr>0 same rows"] = np.mean((h.rr > 0).values == u)
    for col, sgn in (("dmargin", 1), ("dshares", -1)):
        h = S.dropna(subset=[col]); u = (h.up_next == 1).values
        small = h.rr.abs() < 15
        lab = (h.rr > 0).values.copy(); lab[small.values] = (sgn * h[col][small] > 0).values
        r[f"{col} tie-break |rr|<15 (n_small={int(small.sum())})"] = np.mean(lab == u); r[f"rr only, same rows"] = np.mean((h.rr > 0).values == u)
    print(f"{nm}: " + " | ".join(f"{k}: {v:.3f}" for k, v in r.items()))
# marginal value of margin/shares as a logistic add-on over the whole test set
from numpy.linalg import lstsq
for col in ("dmargin", "dshares"):
    h = tr.dropna(subset=[col]); X = np.column_stack([np.ones(len(h)), np.sign(h.rr) * np.log1p(h.rr.abs()), h[col].clip(-30, 30) / 10]); y = (h.up_next == 1).values.astype(float)
    b_ = lstsq(X, y, rcond=None)[0]; ht = te.dropna(subset=[col]); Xt = np.column_stack([np.ones(len(ht)), np.sign(ht.rr) * np.log1p(ht.rr.abs()), ht[col].clip(-30, 30) / 10])
    acc = np.mean((Xt @ b_ > 0.5) == (ht.up_next == 1).values); print(f"linear probability with rr + {col}: coef {b_.round(3)} test acc {acc:.3f} (rr alone {np.mean((ht.rr > 0).values == (ht.up_next == 1).values):.3f})")
