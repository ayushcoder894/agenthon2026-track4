"""ML-G: LightGBM quantile intervals (alpha 0.05/0.95) vs the shipped bands for auctions, CPI (non-pump components), COT, post-earnings and rates.
Each: train on the early period, test on the later one; report interval score (lower is better) and coverage."""
import json, os
import numpy as np, pandas as pd
import lightgbm as lgb

IS = lambda lo, hi, y: np.mean((hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0))
def qfit(Xtr, ytr, Xte, a, leaves=7, mcs=60, n=200):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=n, learning_rate=0.03, num_leaves=leaves, min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
    return m.fit(Xtr, ytr).predict(Xte)
def report(name, y, lo0, hi0, lo1, hi1):
    print(f"{name:34s} shipped IS {IS(lo0, hi0, y):9.4f} cov {np.mean((y >= lo0) & (y <= hi0)):.3f} | LightGBM IS {IS(lo1, hi1, y):9.4f} cov {np.mean((y >= lo1) & (y <= hi1)):.3f} | change {IS(lo1, hi1, y) / IS(lo0, hi0, y) - 1:+.1%}")

# ---------------- auctions
rows = [r for p in (1, 2, 3) for r in json.load(open(f"data_auct/p{p}.json"))["data"]]
A = pd.DataFrame(rows); T = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]
A = A[A.original_security_term.isin(T)].copy(); A["date"] = pd.to_datetime(A.auction_date); A["btc"] = pd.to_numeric(A.bid_to_cover_ratio, errors="coerce"); A["new"] = A.reopening.eq("No")
A = A.dropna(subset=["btc"]).sort_values("date"); A = A[A.date >= "1998-01-01"]
out = []
for term, g in A.groupby("original_security_term"):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        t = g.date[i]; h = g[(g.date < t) & (g.date >= t - pd.Timedelta(days=380))]
        if len(h) < 6: continue
        base = h.btc.iloc[-6:].mean()
        if h.new.nunique() == 2 and (h.new == g.new[i]).sum() >= 2: base += 0.5 * (h[h.new == g.new[i]].btc.mean() - h.btc.mean())
        out.append(dict(date=t, y=g.btc[i], rule=base, sd=h.btc.std(ddof=0), sd6=h.btc.iloc[-6:].std(ddof=0), rng=h.btc.max() - h.btc.min(), new=int(g.new[i]), term=T.index(term), n=len(h), last_dev=h.btc.iloc[-1] - base))
Q = pd.DataFrame(out); tr, te = Q[Q.date.dt.year <= 2012], Q[Q.date.dt.year >= 2013]
F = ["sd", "sd6", "rng", "new", "term", "n", "last_dev"]
res = tr.y - tr.rule
lo1 = te.rule.values + qfit(tr[F], res, te[F], 0.05); hi1 = te.rule.values + qfit(tr[F], res, te[F], 0.95)
h0 = np.maximum(1.645 * te.sd.values, 0.08)
report("auctions (test 2013-22)", te.y.values, te.rule.values - h0, te.rule.values + h0, np.minimum(lo1, hi1), np.maximum(lo1, hi1))

# ---------------- CPI non-pump components
exec(open("t4-local/dev/cpi_study.py").read().split('print("=== per-component')[0])
rows = []
for c in ["CPI_CORE", "CPI_SHELTER", "CPI_FOOD", "CPI_NEWVEH", "CPI_USEDCARS", "CPI_APPAREL", "CPI_MEDICAL", "CPI_TRANSPSVC"]:
    s = M[c]; d = pd.DataFrame({"y": s, "rule": 0.5 * s.shift(1) + 0.5 * s.shift(1).rolling(6).mean(), "sd9": s.shift(1).rolling(9).std(ddof=0), "sd24": s.shift(1).rolling(24).std(ddof=0),
                                 "last": s.shift(1), "month": M.month, "comp": list(COMP).index(c)})
    rows.append(d.assign(c=c))
C = pd.concat(rows).dropna(); C = C[C.index.year >= 2002]
tr, te = C[C.index.year <= 2015], C[C.index.year >= 2016]
F = ["sd9", "sd24", "last", "month", "comp"]
res = tr.y - tr.rule
lo1 = te.rule.values + qfit(tr[F], res, te[F], 0.05, mcs=80); hi1 = te.rule.values + qfit(tr[F], res, te[F], 0.95, mcs=80)
h0 = np.maximum(2.08 * te.sd9.values, 0.25)
report("CPI non-pump (test 2016-22)", te.y.values, te.rule.values - h0, te.rule.values + h0, np.minimum(lo1, hi1), np.maximum(lo1, hi1))

# ---------------- COT 5-week change interval (point = -0.1086 x net%OI)
exec(open("t4-local/dev/cot_study.py").read().split("SIG = ")[0])
D = D.dropna(subset=["y", "mean26", "sd26", "d4", "sddiff"]); D = D[D.date.dt.year >= 2008]
D["p"] = -0.1086 * D.npo; D["res"] = D.y - D.p
tr, te = D[D.date.dt.year <= 2015], D[D.date.dt.year >= 2016]
F = ["npo", "mean26", "sd26", "d4", "d1", "sddiff"]
lo1 = te.p.values + qfit(tr[F], tr.res, te[F], 0.05, mcs=300); hi1 = te.p.values + qfit(tr[F], tr.res, te[F], 0.95, mcs=300)
h0 = np.maximum(1.96 * te.sddiff.values * np.sqrt(5), 2.0)
report("COT (test 2016-22)", te.y.values, te.p.values - h0, te.p.values + h0, np.minimum(lo1, hi1), np.maximum(lo1, hi1))

# ---------------- post-earnings abnormal return interval (point 0)
ev = pd.read_parquet("t4-local/dev/postearn_events.parquet").sort_values("day"); ev["day"] = pd.to_datetime(ev.day)
mabs = ev[ev.year <= 2018].abn.abs().mean()
feat = []
for r in ev.itertuples():
    prior = ev[(ev.ticker == r.ticker) & (ev.day < r.day)].abn.abs()
    n = len(prior)
    feat.append(((n * (prior.mean() / mabs if n else 1.0)) + 4) / (n + 4) if n else 1.0)
ev["tscale"] = feat
prior_n = ev.groupby("ticker").cumcount(); ev["nprior"] = prior_n
ev["after"] = (ev.session == "after_close").astype(int); ev["dow"] = ev.day.dt.dayofweek; ev["month"] = ev.day.dt.month
tr, te = ev[ev.year <= 2018], ev[ev.year >= 2019]
F = ["tscale", "nprior", "after", "dow", "month"]
q05, q95 = np.percentile(tr.abn, [5, 95])
lo1 = qfit(tr[F], tr.abn, te[F], 0.05, mcs=60); hi1 = qfit(tr[F], tr.abn, te[F], 0.95, mcs=60)
s = te.tscale.values * 1.1
report("post-earnings (test 2019-22)", te.abn.values, q05 * s, q95 * s, np.minimum(lo1, hi1), np.maximum(lo1, hi1))
