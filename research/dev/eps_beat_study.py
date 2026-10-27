"""Experiments 85-90: calibrate the beat / miss / inline-vs-consensus family. Data: Alpha Vantage EARNINGS (estimated vs reported EPS per quarter), reports dated
2010-01-01..2022-06-28 only. Label (threshold t): beat if (actual - est)/max(|est|,0.10) > t, miss if < -t, else inline.
85 overall shares by threshold   86 by year (stability)   87 by quarter   88 by consensus-vs-prior-year-reported growth bucket
89 by pre-market vs after-close   90 base rates of 'beat' by company (is company history informative? previous-quarter beat -> next beat)"""
import glob, json
import numpy as np, pandas as pd

rows = []
for f in glob.glob("data_alpha/*.json"):
    j = json.load(open(f)); t = j["symbol"]
    for q in j["quarterlyEarnings"]:
        try:
            rows.append(dict(ticker=t, fiscal=pd.Timestamp(q["fiscalDateEnding"]), reported=pd.Timestamp(q["reportedDate"]), actual=float(q["reportedEPS"]),
                             est=float(q["estimatedEPS"]), time=q.get("reportTime", "")))
        except (TypeError, ValueError, KeyError):
            continue
D = pd.DataFrame(rows).sort_values(["ticker", "fiscal"])
D = D[(D.reported >= "2010-01-01") & (D.reported <= "2022-06-28")].copy()
D["surprise"] = (D.actual - D.est) / D.est.abs().clip(lower=0.10)
# prior-year reported EPS of the same fiscal quarter and previous-quarter surprise (same company)
D["prior_actual"] = D.groupby("ticker").actual.shift(4)
D["cons_growth"] = (D.est - D.prior_actual) / D.prior_actual.abs().clip(lower=0.10) * 100
D["prev_surprise"] = D.groupby("ticker").surprise.shift(1)
D["year"] = D.reported.dt.year; D["quarter"] = D.fiscal.dt.quarter
D["gbucket"] = pd.cut(D.cons_growth, [-np.inf, -10, 0, 10, 25, np.inf], labels=["<-10", "-10..0", "0..10", "10..25", ">25"])
print(f"companies {D.ticker.nunique()}  quarters {len(D)}  ({D.reported.min().date()}..{D.reported.max().date()})")
def shares(S, t):
    b, m = (S.surprise > t).mean(), (S.surprise < -t).mean()
    return f"beat {b:.3f} inline {1 - b - m:.3f} miss {m:.3f}"
print("--- 85 overall"); [print(f"   threshold {t:.2f}: {shares(D, t)}") for t in (0.0, 0.03, 0.05, 0.10)]
print("--- 86 by year (threshold 5%)")
for y, g in D.groupby("year"): print(f"   {y}: n={len(g):4d} {shares(g, 0.05)}")
print("--- 87 by fiscal quarter"); [print(f"   Q{int(q)}: n={len(g)} {shares(g, 0.05)}") for q, g in D.groupby("quarter")]
print("--- 88 by consensus growth vs prior-year reported EPS"); [print(f"   {b:7s}: n={len(g):4d} {shares(g, 0.05)}") for b, g in D.groupby("gbucket", observed=True)]
print("--- 89 by report time"); [print(f"   {k or '?':12s}: n={len(g):4d} {shares(g, 0.05)}") for k, g in D.groupby("time")]
print("--- 90 persistence: after a beat (>5%) / inline / miss in the previous quarter")
h = D.dropna(subset=["prev_surprise"])
for nm, mask in (("prev beat", h.prev_surprise > 0.05), ("prev inline", h.prev_surprise.abs() <= 0.05), ("prev miss", h.prev_surprise < -0.05)):
    print(f"   {nm:12s}: n={int(mask.sum()):4d} {shares(h[mask], 0.05)}")
# out-of-sample check of the simplest rule: always 'beat' vs always 'inline' vs the base-rate argmax, split by time
tr, te = D[D.year <= 2017], D[D.year >= 2018]
for t in (0.05,):
    for nm, S in (("train", tr), ("test", te)):
        print(f"   {nm} (t={t}): P(beat)={np.mean(S.surprise > t):.3f} P(inline)={np.mean(S.surprise.abs() <= t):.3f} P(miss)={np.mean(S.surprise < -t):.3f}  n={len(S)}")
