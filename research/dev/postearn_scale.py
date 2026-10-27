"""E8: company-specific width for the post-earnings abnormal-return interval. For each event, the interval is the pooled
5/95 percentiles (fitted on events before the test period) times a shrunk scale = the company's own median |abnormal
return| over its PRIOR events relative to the pooled median. Test events 2019-01..2022-06; interval score at 90%."""
import numpy as np
import pandas as pd

d = pd.read_parquet("t4-local/dev/postearn_events.parquet").sort_values("day")
tr = d[d.year <= 2018]
q05, q95 = np.percentile(tr.abn, [5, 95])
pooled_med = np.median(np.abs(tr.abn))
print(f"pooled q05/q95 {q05:.2f}/{q95:.2f}, pooled median |abn| {pooled_med:.2f}, events {len(d)}")


def IS(y, lo, hi):
    return np.mean((hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0))

te = d[d.year >= 2019].copy()
for k in (0, 2, 4, 8):
    for stat in ("median", "mean"):
        los, his = [], []
        for _, r in te.iterrows():
            prior = d[(d.ticker == r.ticker) & (d.day < r.day)].abn.abs()
            n = len(prior)
            if n == 0:
                s = 1.0
            else:
                own = (prior.median() if stat == "median" else prior.mean() / np.mean(np.abs(tr.abn)) * pooled_med) / pooled_med
                s = (n * own + k * 1.0) / (n + k) if k else own
            los.append(q05 * s); his.append(q95 * s)
        los, his = np.array(los), np.array(his)
        print(f"shrink k={k} stat={stat:6s}: IS {IS(te.abn.values, los, his):.3f} coverage {np.mean((te.abn >= los) & (te.abn <= his)):.2f}")
print(f"pooled band: IS {IS(te.abn.values, np.full(len(te), q05), np.full(len(te), q95)):.3f} coverage {np.mean((te.abn >= q05) & (te.abn <= q95)):.2f}")
# mega-caps only (revenue >= 50B)
mc = te[te.rev >= 5e10]
print("mega-cap test events:", len(mc), "pooled IS", round(IS(mc.abn.values, np.full(len(mc), q05), np.full(len(mc), q95)), 3))
