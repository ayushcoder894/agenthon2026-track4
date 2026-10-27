"""Experiments 17-24 (macro revisions), ALFRED vintages of PAYEMS, RSAFS, INDPRO, HOUST, PI, DGORDER, realtime 2011..2022-06-28.
Row = (series, vintage v, young reference month m); features are known at v. Non-zero revisions only for direction accuracy.
17 sign of the SAME month's previous revision (own-row momentum)      18 sign of previous revision if it was large vs small
19 month-of-year of the release (seasonal / annual-update months)     20 age of the estimate (vintages since first print)
21 series x age table (fit train)                                      22 point shift by f x median |rev| toward the label (relative MAE vs latest)
23 per-series band multiplier (IS-optimal k)                           24 direction = opposite of previous revision of same month"""
import json
import numpy as np
import pandas as pd

LIMIT = pd.Timestamp("2022-06-28")
rows = []
for s in ("PAYEMS", "RSAFS", "INDPRO", "HOUST", "PI", "DGORDER"):
    d = pd.DataFrame(json.load(open(f"data_alfred/{s}.json")))
    d = d[d.value != "."]; d["v"] = pd.to_datetime(d.realtime_start); d["m"] = pd.to_datetime(d.date); d["x"] = d.value.astype(float)
    d = d[d.v <= LIMIT]; vint = sorted(d.v.unique())
    last_rev, nrev, first_v = {}, {}, {}
    hist = []
    for i in range(len(vint) - 1):
        v, vn = vint[i], vint[i + 1]
        cur = d[d.v <= v].sort_values("v").groupby("m").x.last(); nxt = d[d.v <= vn].sort_values("v").groupby("m").x.last()
        for m in cur.index:
            first_v.setdefault(m, v)
        h12 = [r for (t, r) in hist if (v - t).days <= 365 and r != 0]
        for m in [m for m in cur.index if 0 <= (v - m).days <= 100]:
            if m in nxt.index and cur[m]:
                rel = (nxt[m] - cur[m]) / abs(cur[m])
                rows.append(dict(s=s, v=v, m=m, e=cur[m], rel=rel, prev=last_rev.get(m, np.nan), age=(v - first_v[m]).days // 25,
                                 sd=np.std(h12) if len(h12) >= 3 else np.nan, share=np.mean([r > 0 for r in h12[-12:]]) if len(h12) >= 3 else np.nan,
                                 med=np.median(np.abs(h12)) if len(h12) >= 3 else np.nan))
        for m in cur.index:
            if m in nxt.index and cur[m]:
                r = (nxt[m] - cur[m]) / abs(cur[m]); hist.append((vn, r))
                if r != 0: last_rev[m] = r
R = pd.DataFrame(rows).dropna(subset=["sd"]); R = R[R.v >= "2011-01-01"]
R["up"] = R.rel > 0; R["month"] = R.v.dt.month
NZ = R[R.rel != 0].copy(); tr, te = NZ[NZ.v.dt.year <= 2016], NZ[NZ.v.dt.year >= 2017]
print("rows", len(R), "non-zero", len(NZ), "| test base up-share", round(te.up.mean(), 3))
acc = lambda lab, df: np.mean(lab == df.up.values)
print("--- 17/24 previous revision of the same month: share of rows with a previous revision", round(NZ.prev.notna().mean(), 3))
for nm, df in (("train", tr), ("test", te)):
    h = df.dropna(subset=["prev"])
    print(f"   {nm}: n={len(h)} P(up | prev up) {h[h.prev > 0].up.mean():.3f} (n={int((h.prev > 0).sum())}) P(up | prev down) {h[h.prev < 0].up.mean():.3f} (n={int((h.prev < 0).sum())}) | momentum acc {acc((h.prev > 0).values, h):.3f} vs always-up {h.up.mean():.3f}")
print("--- 18 large vs small previous revision (relative to the series sd)")
h = NZ.dropna(subset=["prev"]); big = np.abs(h.prev) > h.sd
for nm, mask in (("large prev", big), ("small prev", ~big)):
    g = h[mask]; print(f"   {nm}: n={len(g)} P(up|prev up) {g[g.prev > 0].up.mean():.3f} P(up|prev down) {g[g.prev < 0].up.mean():.3f}")
print("--- 19 release-month up-share (train 2011-16 vs test 2017-22):")
tm = tr.groupby("month").up.mean().round(2).to_dict(); sm = te.groupby("month").up.mean().round(2).to_dict()
print("   train", tm); print("   test ", sm)
lab = te.month.map(lambda m: tm.get(m, 0.56) >= 0.5).values
print(f"   month rule acc {acc(lab, te):.3f} vs always-up {te.up.mean():.3f}")
print("--- 20/21 age of the estimate:")
print("   up-share by age (train):", tr.groupby("age").up.agg(['mean', 'count']).round(2).to_dict('index'))
t21 = tr.groupby(["s", "age"]).up.mean()
lab = np.array([t21.get((s, a), 0.56) >= 0.5 for s, a in zip(te.s, te.age)]); print(f"   series x age rule acc {acc(lab, te):.3f}")
print("--- 22 point shift (relative MAE of point vs 'latest estimate'), test non-zero+zero rows")
T = R[R.v.dt.year >= 2017]; base_mae = np.abs(T.rel).mean()
for f in (0.0, 0.25, 0.5, 1.0):
    lab = np.where(T.share < 0.3, False, True) if False else np.ones(len(T), bool)  # label = always-up (the shipped long-run prior)
    pt = np.where(lab, 1, -1) * f * T.med
    print(f"   f={f}: MAE {np.abs(T.rel - pt).mean() * 1e4:.2f}bp (latest estimate {base_mae * 1e4:.2f}bp)")
print("--- 23 per-series band multiplier (IS-optimal k on test, shift 0.25 sd toward 'up')")
for s, g in T.groupby("s"):
    best = []
    for k in (0.8, 1.0, 1.3, 1.645, 2.0):
        lo, hi = 0.25 * g.sd - k * g.sd, 0.25 * g.sd + k * g.sd
        IS = np.mean((hi - lo) + 20 * np.maximum(lo - g.rel, 0) + 20 * np.maximum(g.rel - hi, 0)); best.append((round(IS * 1e4, 1), k, round(float(np.mean((g.rel >= lo) & (g.rel <= hi))), 2)))
    print(f"   {s:8s} n={len(g):4d}", best)
