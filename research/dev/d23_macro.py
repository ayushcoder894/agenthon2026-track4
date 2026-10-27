"""D23-3: macro-revision direction. Build non-zero revision events from local ALFRED vintages (<= 2022-06-28). Event = a reference month whose published value changes between consecutive vintages.
Chronological test of direction rules (fit on events before each test year): pooled up-share, series up-share (trailing 36 / all), last-revision-sign persistence, and a Bayes blend (D20R-style prior weight w).
Metric: direction accuracy by series and pooled, 2012-2022."""
import json
import numpy as np, pandas as pd
ev = []
for s in ("DGORDER", "HOUST", "INDPRO", "PAYEMS", "PI", "RSAFS"):
    d = pd.DataFrame(json.load(open(f"data_alfred/{s}.json")))
    d["v"] = pd.to_numeric(d.value, errors="coerce"); d = d.dropna(subset=["v"]).sort_values(["date", "realtime_start"])
    for dt, g in d.groupby("date"):
        v, rs = g.v.values, g.realtime_start.values
        for i in range(1, len(v)):
            if v[i] != v[i - 1]:
                ev.append(dict(series=s, ref=dt, when=rs[i], dir=1 if v[i] > v[i - 1] else -1, rel=(v[i] - v[i - 1]) / abs(v[i - 1]), k=i))   # k = revision number
E = pd.DataFrame(ev).sort_values("when").reset_index(drop=True); E["year"] = pd.to_datetime(E.when).dt.year
print("events", len(E), "up share", round((E.dir > 0).mean(), 3)); print(E.groupby("series").dir.agg(lambda x: f"n={len(x)} up={np.mean(x > 0):.3f}").to_string())
print("by revision number k=1,2,3 up share:", E.groupby(E.k.clip(upper=3)).dir.apply(lambda x: round(float(np.mean(x > 0)), 3)).to_dict())
rows = []
for yr in range(2012, 2023):
    te = E[E.year == yr]; tr = E[E.year < yr]
    if te.empty: continue
    pool = (tr.dir > 0).mean()
    for r in te.itertuples():
        trs = tr[tr.series == r.series]
        pred = {}
        pred["pooled"] = 1 if pool >= 0.5 else -1
        pred["series_all"] = 1 if (trs.dir > 0).mean() >= 0.5 else -1
        t36 = trs.tail(36); pred["series_36"] = 1 if (t36.dir > 0).mean() >= 0.5 else -1
        for w in (20, 100):
            p = ((trs.dir > 0).sum() + w * pool) / (len(trs) + w); pred[f"blend_w{w}"] = 1 if p >= 0.5 else -1
        last = trs.dir.iloc[-1] if len(trs) else 1; pred["persist_last"] = last
        pred["always_down"] = -1; pred["always_up"] = 1
        # revision-number-aware: share of up among events with the same k within the series (min 20), else pooled
        sk = trs[trs.k.clip(upper=3) == min(r.k, 3)]; pred["series_k"] = (1 if (sk.dir > 0).mean() >= 0.5 else -1) if len(sk) >= 20 else pred["pooled"]
        for m, p in pred.items(): rows.append(dict(year=r.year, series=r.series, model=m, ok=int(p == r.dir)))
R = pd.DataFrame(rows)
print("\nDirection accuracy 2012-2022 (pooled over events):"); print(R.groupby("model").ok.mean().round(3).sort_values(ascending=False).to_string())
print("\nBy series (series_all / blend_w20 / pooled / always_down / always_up):")
print(R[R.model.isin(["series_all", "blend_w20", "pooled", "always_down", "always_up", "series_k"])].pivot_table(index="series", columns="model", values="ok", aggfunc="mean").round(3).to_string())
print("\nrecent 2018-2022:"); print(R[R.year >= 2018].groupby("model").ok.mean().round(3).sort_values(ascending=False).to_string())
