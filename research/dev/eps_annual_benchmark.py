"""Does ANNUAL diluted-EPS growth (latest 10-K) predict the next quarter's YoY direction? (cutoff mid-March..mid-April case)
Same data/limits as eps_benchmark.py. Origin = end of the target quarter + 14 days; the 10-K must be public by then."""
import glob

import numpy as np
import pandas as pd

LIMIT = pd.Timestamp("2022-06-28", tz="UTC")
COLS = ["cik", "period_end", "quarters", "period_kind", "form", "knowledge_date", "eps_diluted", "revenue", "revision"]
df = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))], ignore_index=True)
df = df[(df.knowledge_date <= LIMIT) & df.eps_diluted.notna()]
df["period_end"] = pd.to_datetime(df["period_end"]).astype("datetime64[ns]")


def first_reported(d):
    return d.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()


Q = first_reported(df[(df.period_kind == "quarter") & (df.quarters == 1) & df.form.astype(str).str.startswith("10-Q")])
Y = first_reported(df[(df.period_kind == "year") & df.form.astype(str).str.startswith("10-K")])
print("quarterly", len(Q), "annual", len(Y))

# annual YoY growth per company-year
a = Y[["cik", "period_end", "knowledge_date", "eps_diluted", "revenue"]].copy(); a["key"] = (a.period_end - pd.Timedelta(days=365)).astype("datetime64[ns]")
b = a.rename(columns={"period_end": "pe_p", "knowledge_date": "kd_p", "eps_diluted": "eps_p", "revenue": "rev_p"})[["cik", "pe_p", "eps_p"]].sort_values("pe_p")
a = pd.merge_asof(a.sort_values("key"), b, left_on="key", right_on="pe_p", by="cik", tolerance=pd.Timedelta(days=25), direction="nearest").dropna(subset=["eps_p"])
a["g_year"] = (a.eps_diluted - a.eps_p) / a.eps_p.abs().clip(lower=0.01) * 100.0

# quarterly YoY outcome for the first quarter after each fiscal year end
q = Q[["cik", "period_end", "knowledge_date", "eps_diluted"]].copy(); q["key"] = (q.period_end - pd.Timedelta(days=365)).astype("datetime64[ns]")
qp = q.rename(columns={"period_end": "pe_p", "knowledge_date": "kd_p", "eps_diluted": "eps_p"})[["cik", "pe_p", "eps_p"]].sort_values("pe_p")
q = pd.merge_asof(q.sort_values("key"), qp, left_on="key", right_on="pe_p", by="cik", tolerance=pd.Timedelta(days=20), direction="nearest").dropna(subset=["eps_p"])
q["up_next"] = (q.eps_diluted > q.eps_p).astype(int)
a["key2"] = (a.period_end + pd.Timedelta(days=91)).astype("datetime64[ns]")
P = pd.merge_asof(a.sort_values("key2"), q.sort_values("period_end")[["cik", "period_end", "knowledge_date", "up_next"]].rename(
    columns={"period_end": "pe_next", "knowledge_date": "kd_next"}), left_on="key2", right_on="pe_next", by="cik", tolerance=pd.Timedelta(days=12), direction="nearest").dropna(subset=["pe_next"])
P["origin"] = (P.pe_next + pd.Timedelta(days=14)).dt.tz_localize("UTC")
P = P[(P.knowledge_date <= P.origin) & (P.origin <= LIMIT) & (P.kd_next <= LIMIT)]
train, test = P[P.pe_next <= "2018-12-31"], P[P.pe_next >= "2019-01-01"]
print("examples train", len(train), "test", len(test), "| test base up", round(test.up_next.mean(), 3))
bins = [-np.inf, -50, -10, 0, 10, 30, np.inf]; labels = ["<=-50", "-50..-10", "-10..0", "0..10", "10..30", ">30"]
for d in (train, test, P): d["bucket"] = pd.cut(d.g_year, bins=bins, labels=labels)
tab = train.groupby("bucket", observed=True).up_next.agg(["mean", "count"]); print(tab.round(3))
pr = test.bucket.map(tab["mean"].to_dict()).astype(float)
print("TEST all: persist(sign)", round(float(((test.g_year > 0).astype(int) == test.up_next).mean()), 3), "| bucket argmax", round(float(((pr >= 0.5).astype(int) == test.up_next).mean()), 3), "| always_up", round(float(test.up_next.mean()), 3))
big = test[test.revenue >= 1e9]
print("TEST large (annual revenue >= 1e9):", len(big), "persist", round(float(((big.g_year > 0).astype(int) == big.up_next).mean()), 3), "always_up", round(float(big.up_next.mean()), 3))
for y, d in test.groupby(test.pe_next.dt.year):
    print(" ", y, len(d), "persist", round(float(((d.g_year > 0).astype(int) == d.up_next).mean()), 3), "always_up", round(float(d.up_next.mean()), 3))
fit = P.groupby("bucket", observed=True).up_next.mean().round(4).to_dict()
print("final p_up (all years):", fit)
