"""E6-E7: YoY diluted-EPS growth for large financials (banks / broker-dealers) vs all companies: best point weight on last
quarter's growth and the band. Financial proxy (no SIC in the data): no current-assets line, liabilities > 85% of assets,
total assets >= $10B, in the latest 10-Q known at the origin. Same construction as eps_benchmark.py (data <= 2022-06-28)."""
import glob
import numpy as np
import pandas as pd

src = open("t4-local/dev/eps_benchmark.py").read().split("def boot_ci")[0]
exec(src)
bs = pd.concat([pd.read_parquet(f, columns=["cik", "period_end", "period_kind", "quarters", "form", "knowledge_date", "revision",
                                            "current_assets", "total_assets", "total_liabilities"])
                for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))], ignore_index=True)
bs = bs[(bs.knowledge_date <= LIMIT) & bs.form.astype(str).str.startswith("10-Q") & bs.total_assets.notna()]
bs = bs.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
bs["fin"] = bs.current_assets.isna() & (bs.total_liabilities / bs.total_assets > 0.85) & (bs.total_assets >= 1e10)
bs["period_end"] = pd.to_datetime(bs.period_end).astype("datetime64[ns]")
P2 = P.merge(bs[["cik", "period_end", "fin", "total_assets"]], on=["cik", "period_end"], how="left")
P2["fin"] = P2.fin.fillna(False).astype(bool)
print("financial examples:", int(P2.fin.sum()), "companies:", P2[P2.fin].cik.nunique())
P2["gl"] = P2.g_last.clip(-300, 300)
for name, S0 in (("ALL", P2), ("FIN", P2[P2.fin])):
    for per, S in (("train<=2018", S0[S0.pe_next <= "2018-12-31"]), ("test 2019-22", S0[S0.pe_next >= "2019-01-01"])):
        y = S.growth_next.clip(-1000, 1000)
        mae0 = np.abs(y).mean()
        line = f"{name} {per:12s} n={len(S):6d} MAE(0)={mae0:7.1f} |"
        for w in (0.1, 0.25, 0.4, 0.6):
            line += f" w={w}: {mae0 / (mae0 + np.abs(y - w * S.gl).mean()):.3f}"
        for c in (3, 6, 9):
            line += f" c={c}+.25g: {mae0 / (mae0 + np.abs(y - c - 0.25 * S.gl).mean()):.3f}"
        med = np.median(y)
        line += f" | median growth {med:+.1f} | q05/q95 resid(w=.25) {np.percentile(y - 0.25 * S.gl, 5):+.1f}/{np.percentile(y - 0.25 * S.gl, 95):+.1f}"
        print(line)
# band check for financials: interval score of the all-company bucket band vs a financial-specific band (fit train, test 2019+)
import json
art = json.load(open("t4-local/agent/t4_agent/artifacts.json"))["families"]["eps_persistence"]
print("artifact growth bands:", art.get("growth_band"), art.get("growth_band_w0"))
bins = [-np.inf, -50, -10, 0, 10, 30, np.inf]
labels = ["<=-50", "-50..-10", "-10..0", "0..10", "10..30", ">30"]
F = P2[P2.fin].copy()
F["bucket"] = pd.cut(F.g_last, bins=bins, labels=labels)
tr, te = F[F.pe_next <= "2018-12-31"], F[F.pe_next >= "2019-01-01"]
def IS(y, lo, hi):
    return np.mean((hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0))
y = te.growth_next.clip(-1000, 1000).values
pt = 0.25 * te.gl.values
lo_a = np.array([pt[i] + 0 for i in range(len(te))])
gb = art["growth_band"]
lo_all = np.array([gb[str(b)][0] for b in te.bucket]); hi_all = np.array([gb[str(b)][1] for b in te.bucket])
print(f"FIN test: all-company bucket band IS {IS(y, lo_all, hi_all):.1f} coverage {np.mean((y >= lo_all) & (y <= hi_all)):.2f}")
for qlo, qhi in ((5, 95), (4, 96), (3, 97)):
    rl, rh = {}, {}
    for b, g in tr.groupby("bucket", observed=True):
        r = g.growth_next.clip(-1000, 1000) - 0.25 * g.gl
        rl[str(b)], rh[str(b)] = np.percentile(r, qlo), np.percentile(r, qhi)
    r_all = tr.growth_next.clip(-1000, 1000) - 0.25 * tr.gl
    lo_f = np.array([pt[i] + rl.get(str(b), np.percentile(r_all, qlo)) for i, b in enumerate(te.bucket)])
    hi_f = np.array([pt[i] + rh.get(str(b), np.percentile(r_all, qhi)) for i, b in enumerate(te.bucket)])
    lo_p, hi_p = pt + np.percentile(r_all, qlo), pt + np.percentile(r_all, qhi)
    print(f"FIN test: fin bucket band q{qlo}/{qhi} IS {IS(y, lo_f, hi_f):.1f} cov {np.mean((y >= lo_f) & (y <= hi_f)):.2f} | "
          f"fin pooled band IS {IS(y, lo_p, hi_p):.1f} cov {np.mean((y >= lo_p) & (y <= hi_p)):.2f} pooled {np.percentile(r_all, qlo):+.1f}/{np.percentile(r_all, qhi):+.1f}")
    if qlo == 5:
        print("   fin bucket residual quantiles:", {k: (round(rl[k], 1), round(rh[k], 1)) for k in rl},
              "counts", tr.bucket.value_counts().to_dict())
