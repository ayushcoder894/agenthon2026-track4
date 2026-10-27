"""Experiments 25-34 (post-earnings one-day abnormal return, large caps 2010-03..2022-06-28).
Events: t4-local/dev/postearn_events.parquet (ticker, day = first tradeable session, abn %, rev, session, eps_yoy). Prices: data_px/<T>.csv (daily closes).
25 pre-event 20d abnormal drift -> reaction sign       26 pre-event 5d move                27 trailing 60d realised vol scaling of the band
28 day of week                                           29 previous reaction sign           30 run-rate EPS growth bucket vs reaction
31 revenue size vs |abn|                                 32 asymmetric band by label        33 2016+ distribution vs all years
34 vol scale x ticker scale blend"""
import os
import numpy as np
import pandas as pd

ev = pd.read_parquet("t4-local/dev/postearn_events.parquet").sort_values("day").reset_index(drop=True)
ev["day"] = pd.to_datetime(ev.day)
px = {}
for t in ev.ticker.unique():
    f = f"data_px/{t}.csv"
    if os.path.exists(f):
        d = pd.read_csv(f); d["day"] = pd.to_datetime(d.timestamp, unit="s").dt.normalize()
        px[t] = d.drop_duplicates("day").set_index("day").sort_index().close
P = pd.DataFrame(px)
R = P.pct_change(fill_method=None) * 100
MKT = R.mean(axis=1)   # equal-weighted large-cap market proxy (as in the study)
rows = []
for r in ev.itertuples():
    if r.ticker not in R.columns: continue
    s = R[r.ticker]
    idx = s.index.searchsorted(r.day)
    if idx < 70: continue
    w20 = (s.iloc[idx - 20:idx] - MKT.reindex(s.index).iloc[idx - 20:idx]).sum()
    w5 = (s.iloc[idx - 5:idx] - MKT.reindex(s.index).iloc[idx - 5:idx]).sum()
    vol60 = (s.iloc[idx - 60:idx] - MKT.reindex(s.index).iloc[idx - 60:idx]).std()
    rows.append(dict(ticker=r.ticker, day=r.day, abn=r.abn, rev=r.rev, year=r.year, session=r.session, d20=w20, d5=w5, vol60=vol60, dow=r.day.dayofweek, eps_yoy=r.eps_yoy))
D = pd.DataFrame(rows).dropna(subset=["d20", "vol60", "abn"]).sort_values("day")
D["prev_abn"] = D.groupby("ticker").abn.shift(1)
tr, te = D[D.year <= 2018], D[D.year >= 2019]
lab = lambda x: np.where(x > 1, 1, np.where(x < -1, -1, 0))
print(f"events {len(D)} train {len(tr)} test {len(te)} | test P(pos) {np.mean(te.abn > 1):.3f} P(neg) {np.mean(te.abn < -1):.3f} P(flat) {np.mean(te.abn.abs() <= 1):.3f}")
print("--- 25/26 pre-event drift: P(pos)-P(neg) by drift tercile (train | test)")
for col in ("d20", "d5"):
    qs = tr[col].quantile([1 / 3, 2 / 3]).values
    for nm, df in (("train", tr), ("test", te)):
        b = np.digitize(df[col], qs)
        print(f"   {col} {nm}: " + " ".join(f"T{k}: pos {np.mean(df.abn[b == k] > 1):.3f} neg {np.mean(df.abn[b == k] < -1):.3f} n={int((b == k).sum())} |" for k in range(3)))
print("--- 28 day of week P(pos) (train | test)")
print("   ", {int(k): (round(float(np.mean(g.abn > 1)), 2), round(float(np.mean(g.abn < -1)), 2)) for k, g in tr.groupby("dow")}, "|", {int(k): (round(float(np.mean(g.abn > 1)), 2), round(float(np.mean(g.abn < -1)), 2)) for k, g in te.groupby("dow")})
print("--- 29 previous reaction sign -> next reaction")
for nm, df in (("train", tr), ("test", te)):
    h = df.dropna(subset=["prev_abn"])
    print(f"   {nm}: after pos: P(pos) {np.mean(h.abn[h.prev_abn > 1] > 1):.3f} P(neg) {np.mean(h.abn[h.prev_abn > 1] < -1):.3f} | after neg: P(pos) {np.mean(h.abn[h.prev_abn < -1] > 1):.3f} P(neg) {np.mean(h.abn[h.prev_abn < -1] < -1):.3f}")
print("--- 30 EPS yoy growth of the release (post-release, only to see the size of an information effect; not usable pre-release) vs reaction sign")
h = te.dropna(subset=["eps_yoy"]); print(f"   corr(eps_yoy_clipped, abn) = {np.corrcoef(h.eps_yoy.clip(-1, 1), h.abn)[0, 1]:.3f} (n={len(h)})")
print("--- 27/34 interval scaling by trailing vol, IS on test (band = pooled 5/95 x scale)")
q05, q95 = np.percentile(tr.abn, [5, 95]); mvol = tr.vol60.mean(); mabs = tr.abn.abs().mean()
IS = lambda lo, hi, e: np.mean((hi - lo) + 20 * np.maximum(lo - e, 0) + 20 * np.maximum(e - hi, 0))
ts = {}
for tk, g in D.groupby("ticker"): ts[tk] = g
def tick_scale(r):
    prior = D[(D.ticker == r.ticker) & (D.day < r.day)].abn.abs(); n = len(prior)
    return ((n * (prior.mean() / mabs if n else 1.0)) + 4) / (n + 4)
te = te.copy(); te["tscale"] = [tick_scale(r) for r in te.itertuples()]
for nm, s in (("pooled", np.ones(len(te))), ("ticker scale", te.tscale.values), ("vol60 scale", (te.vol60 / mvol).values), ("vol60^0.5", np.sqrt(te.vol60 / mvol).values),
              ("ticker x vol^0.5 / mean", (te.tscale * np.sqrt(te.vol60 / mvol)).values), ("0.5 ticker + 0.5 vol", (0.5 * te.tscale + 0.5 * te.vol60 / mvol).values)):
    for m in (1.0, 1.15, 1.3):
        lo, hi = q05 * s * m, q95 * s * m
        print(f"   {nm:26s} m={m}: IS {IS(lo, hi, te.abn.values):.3f} cov {np.mean((te.abn >= lo) & (te.abn <= hi)):.2f}")
print("--- 31 revenue vs |abn| corr (train):", round(np.corrcoef(np.log(tr.rev), tr.abn.abs())[0, 1], 3))
print("--- 33 2016+ distribution vs all years (band IS on 2019-22): all", round(IS(*np.percentile(tr.abn, [5, 95]), te.abn.values), 3), "| 2016-18 only", round(IS(*np.percentile(tr[tr.year >= 2016].abn, [5, 95]), te.abn.values), 3))
