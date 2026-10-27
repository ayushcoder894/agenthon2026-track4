"""Is the EPS-growth band too wide for calm, very large companies (the bank-unit case)? Interval score by band scale,
for subsets by size and by |last growth|, test years 2019-2022 and calm years 2013-2018. Data <= 2022-06-28."""
import glob
import numpy as np
import pandas as pd

LIMIT = pd.Timestamp("2022-06-28", tz="UTC")
COLS = ["cik", "period_end", "quarters", "period_kind", "form", "knowledge_date", "eps_diluted", "revenue", "revision"]
df = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))], ignore_index=True)
df = df[(df.knowledge_date <= LIMIT) & (df.period_kind == "quarter") & (df.quarters == 1) & df.eps_diluted.notna() & df.form.astype(str).str.startswith("10-Q")]
df = df.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
df["period_end"] = pd.to_datetime(df.period_end).astype("datetime64[ns]")
a = df[["cik", "period_end", "eps_diluted", "revenue", "knowledge_date"]].copy(); a["key"] = (a.period_end - pd.Timedelta(days=365)).astype("datetime64[ns]")
b = a.rename(columns={"period_end": "pe_p", "eps_diluted": "eps_p"})[["cik", "pe_p", "eps_p"]].sort_values("pe_p")
m = pd.merge_asof(a.sort_values("key"), b, left_on="key", right_on="pe_p", by="cik", tolerance=pd.Timedelta(days=20), direction="nearest").dropna(subset=["eps_p"])
m["g"] = ((m.eps_diluted - m.eps_p) / m.eps_p.abs().clip(lower=0.01) * 100).clip(-400, 400)
last = m[["cik", "period_end", "g", "revenue", "knowledge_date"]].rename(columns={"g": "g_last", "revenue": "rev_last", "knowledge_date": "kd_last"})
last["key"] = (last.period_end + pd.Timedelta(days=91)).astype("datetime64[ns]")
nxt = m[["cik", "period_end", "g"]].rename(columns={"period_end": "pe_next", "g": "g_next"}).sort_values("pe_next")
P = pd.merge_asof(last.sort_values("key"), nxt, left_on="key", right_on="pe_next", by="cik", tolerance=pd.Timedelta(days=12), direction="nearest").dropna(subset=["pe_next"])
P["year"] = P.pe_next.dt.year
bands = {"<=-50": (-325.0, 370.4), "-50..-10": (-102.9, 220.5), "-10..0": (-83.2, 141.3), "0..10": (-61.7, 124.5), "10..30": (-66.1, 162.1), ">30": (-155.2, 340.1)}
P["bucket"] = pd.cut(P.g_last.clip(-300, 300), [-np.inf, -50, -10, 0, 10, 30, np.inf], labels=list(bands))
P["e"] = P.g_next - 0.25 * P.g_last.clip(-300, 300)
P["lo0"] = P.bucket.map({k: v[0] for k, v in bands.items()}).astype(float); P["hi0"] = P.bucket.map({k: v[1] for k, v in bands.items()}).astype(float)


def IS(e, lo, hi): return float(np.mean((hi - lo) + 20 * np.maximum(lo - e, 0) + 20 * np.maximum(e - hi, 0)))


def naive45(e_g): return IS(e_g, -45 * np.ones(len(e_g)), 45 * np.ones(len(e_g)))


subsets = {"all rev>=1e9": P.rev_last >= 1e9, "rev>=2e10": P.rev_last >= 2e10,
           "rev>=2e10 & |g_last|<=25": (P.rev_last >= 2e10) & (P.g_last.abs() <= 25),
           "rev>=1e10 & |g_last|<=25": (P.rev_last >= 1e10) & (P.g_last.abs() <= 25)}
for name, mask in subsets.items():
    for yrs, lab in (((2013, 2018), "calm 2013-18"), ((2019, 2022), "test 2019-22")):
        S = P[mask & P.year.between(*yrs)]
        if len(S) < 50:
            continue
        row = {s: IS(S.e, S.lo0 * s, S.hi0 * s) for s in (0.25, 0.4, 0.6, 0.8, 1.0)}
        best = min(row, key=row.get)
        print(f"{name:28s} {lab}: n={len(S):5d} | IS by scale " + " ".join(f"{s}:{v:.0f}" for s, v in row.items())
              + f" | best scale {best} | naive +-45 around 0: {naive45(S.g_next):.0f} | median |g_next| {S.g_next.abs().median():.0f}")
