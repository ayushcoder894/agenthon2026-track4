"""EPS yoy-growth interval: firm-volatility-scaled band vs the shipped bucket quantile band (heteroskedasticity).
Point = shipped run-rate point ('base'). Firm scale sigma = 100 x sd(last 4 quarterly diluted EPS) / max(|prior-year EPS|, 0.05), floored.
Band = base + sigma x [q05, q95] of the standardized error (y - base)/sigma, quantiles fitted on TRAIN (<= 2016) only.
Shipped = base + bucket [q05, q95] from artifacts eps_runrate.growth_band_pw (pooled over all firms in the bucket).
Judged by mean interval score (alpha 0.1) on VAL (2017-19) and FINAL (2020-22), for ALL firms, large firms and financials."""
import json, numpy as np, pandas as pd
P = pd.read_parquet("D:/AGENTHON/t4-local/dev/exp500/panels/eps_panel.parquet")
art = json.load(open("D:/AGENTHON/t4-local/agent/t4_agent/artifacts.json"))["families"]["eps_runrate"]
gb = art.get("growth_band_pw") or art["growth_band"]
P = P.dropna(subset=["y", "base"]).copy()
P["q05"] = P.b.map(lambda b: gb.get(b, [np.nan, np.nan])[0]); P["q95"] = P.b.map(lambda b: gb.get(b, [np.nan, np.nan])[1])
lags = P[["eps_last", "eps_diluted_l4", "eps_prior_next"]]   # only what the agent sees: current quarter, its yoy comparator, the target prior-year quarter
P["sig"] = 100 * lags.std(axis=1, ddof=0) / np.maximum(P.abs_prior, 0.05)
P = P.dropna(subset=["q05", "q95", "sig"])
P["sig"] = np.maximum(P.sig, 2.0)
P["large"] = P.rev_last >= 1e9
def IS(lo, hi, y): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
tr = P[P.split == "train"]; z = (tr.y - tr.base) / tr.sig
qlo, qhi = np.quantile(z, 0.05), np.quantile(z, 0.95)
print(f"standardized error quantiles fitted on train (n={len(tr)}): q05 {qlo:.2f}, q95 {qhi:.2f}")
# hybrid: the narrower of the two bands per side (a firm with stable EPS gets its own band, a volatile one keeps the pooled band)
for nm, sub in (("ALL", P.index == P.index), ("LARGE (rev >= 1bn)", P.large), ("FIN", P.fin), ("FIN & LARGE", P.fin & P.large)):
    for sp in ("val", "final"):
        x = P[sub & (P.split == sp)]
        if len(x) < 50: continue
        s0 = IS(x.base + x.q05, x.base + x.q95, x.y).mean()
        lo1, hi1 = x.base + qlo * x.sig, x.base + qhi * x.sig; s1 = IS(lo1, hi1, x.y).mean()
        lo2, hi2 = np.maximum(lo1, x.base + x.q05), np.minimum(hi1, x.base + x.q95); s2 = IS(lo2, hi2, x.y).mean()
        lo3, hi3 = 0.5 * (lo1 + x.base + x.q05), 0.5 * (hi1 + x.base + x.q95); s3 = IS(lo3, hi3, x.y).mean()
        cov = lambda lo, hi: ((x.y >= lo) & (x.y <= hi)).mean()
        print(f"{nm:20s} {sp:5s} n={len(x):6d} | IS shipped {s0:8.2f} (cov {cov(x.base + x.q05, x.base + x.q95):.2f}) | vol-scaled {s1:8.2f} (cov {cov(lo1, hi1):.2f}) | "
              f"narrower-of-two {s2:8.2f} (cov {cov(lo2, hi2):.2f}) | average-of-two {s3:8.2f} (cov {cov(lo3, hi3):.2f})")
