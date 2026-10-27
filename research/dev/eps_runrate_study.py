"""E13: run-rate vs growth persistence for next-quarter YoY diluted EPS (direction and growth).
rr = (EPS_last_quarter - EPS_prior_year_of_target) / |EPS_prior_year_of_target| x 100  (carry the latest quarter's level forward)
g  = last quarter's own YoY growth (current signal). Same construction / split as eps_benchmark.py (data <= 2022-06-28)."""
import glob
import numpy as np
import pandas as pd

src = open("t4-local/dev/eps_benchmark.py").read().split("def boot_ci")[0]
src = src.replace('last = m[["cik", "period_end", "knowledge_date", "growth", "yoy_up", "revenue"]]',
                  'last = m[["cik", "period_end", "knowledge_date", "growth", "yoy_up", "revenue", "eps_diluted"]]')
src = src.replace('"growth": "g_last", "yoy_up": "up_last", "revenue": "rev_last"}',
                  '"growth": "g_last", "yoy_up": "up_last", "revenue": "rev_last", "eps_diluted": "eps_last"}')
exec(src)
bs = pd.concat([pd.read_parquet(f, columns=["cik", "period_end", "form", "knowledge_date", "revision", "current_assets", "total_assets",
                                            "total_liabilities"]) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
bs = bs[(bs.knowledge_date <= LIMIT) & bs.form.astype(str).str.startswith("10-Q") & bs.total_assets.notna()]
bs = bs.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
bs["fin"] = bs.current_assets.isna() & (bs.total_liabilities / bs.total_assets > 0.85) & (bs.total_assets >= 1e10)
bs["period_end"] = pd.to_datetime(bs.period_end).astype("datetime64[ns]")
P = P.merge(bs[["cik", "period_end", "fin"]], on=["cik", "period_end"], how="left")
P["fin"] = P.fin.fillna(False).astype(bool)
den = P.eps_prior_next.abs().clip(lower=0.01)
P["rr"] = ((P.eps_last - P.eps_prior_next) / den * 100).clip(-300, 300)
P["gl"] = P.g_last.clip(-300, 300)
P["y"] = P.growth_next.clip(-1000, 1000)
P["big"] = P.rev_last >= 1e9
for name, S0 in (("ALL", P), ("REV>=1B", P[P.big]), ("FIN", P[P.fin])):
    for per, S in (("train<=2018", S0[S0.pe_next <= "2018-12-31"]), ("test 2019-22", S0[S0.pe_next >= "2019-01-01"])):
        up = S.up_next.values
        acc = {"always_up": np.mean(up == 1), "g>0": np.mean((S.gl > 0).values == up), "rr>0": np.mean((S.rr > 0).values == up),
               "g+rr>0": np.mean(((S.gl + S.rr) > 0).values == up), "0.5g+rr>0": np.mean(((0.5 * S.gl + S.rr) > 0).values == up)}
        mae0 = np.abs(S.y).mean()
        sk = {}
        for lab, pred in (("0.25g", 0.25 * S.gl), ("0.25rr", 0.25 * S.rr), ("0.5rr", 0.5 * S.rr), ("0.75rr", 0.75 * S.rr),
                          ("0.15g+0.25rr", 0.15 * S.gl + 0.25 * S.rr), ("0.1g+0.5rr", 0.1 * S.gl + 0.5 * S.rr)):
            sk[lab] = mae0 / (mae0 + np.abs(S.y - pred).mean())
        print(f"{name:8s} {per:12s} n={len(S):6d} acc " + " ".join(f"{k}:{v:.3f}" for k, v in acc.items()) +
              " | growth skill " + " ".join(f"{k}:{v:.3f}" for k, v in sk.items()))
