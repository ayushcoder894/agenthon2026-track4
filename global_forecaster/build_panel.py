"""Training/evaluation panel for the global forecaster: every (series version, origin, horizon) with scale-free features and targets.
Origin date = publication date of the latest value (first-release series) or its observation date (market series), so splits respect
when information was public. Baselines stored per row (in scaled units, relative to the last value):
  naive  : 0 (carry-forward; the documented regression naive rule)
  half   : 0.5 x last + 0.5 x mean of last 6  (the current generic rule for level tables)
  d18chg : 0.5 x AR(1)-to-mean36 + 0.5 x half (the current generic rule for change tables)"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfeat import features, NAMES

D = "D:/AGENTHON/data_global"
cat = json.load(open(f"{D}/catalog.json"))
HZ = {12: (1, 2, 3, 6), 52: (1, 2, 4, 8, 13)}
RNG = np.random.default_rng(7)
LENS = (8, 9, 10, 12, 13, 16, 20, 26, 36, 52, 80, 120)


def versions(sid, meta):
    pts = json.load(open(f"{D}/{sid}.json"))
    s = pd.Series({pd.Timestamp(p[0]): p[1] for p in pts}).sort_index()
    pub = pd.Series({pd.Timestamp(p[0]): pd.Timestamp(p[2]) if len(p) > 2 and p[2] else pd.Timestamp(p[0]) for p in pts}).sort_index()
    out = []
    if meta["freq"] == "D":
        w = s.resample("W-FRI").last().dropna(); m = s.resample("MS").mean().dropna()
        out += [("W", w, w.index.to_series()), ("M", m, (m.index + pd.offsets.MonthEnd(0)).to_series())]
    elif meta["freq"] == "W":
        out += [("W", s, pub.reindex(s.index))]
    else:
        out += [("M", s, pub.reindex(s.index))]
    res = []
    for fq, x, pb in out:
        ppy = 12 if fq == "M" else 52
        res.append(("raw", ppy, x, pb))
        if (x > 0).all() and not meta["units"].lower().startswith(("%", "percent")):
            r1 = 100 * (x / x.shift(1) - 1); res.append(("pct1", ppy, r1.dropna(), pb.reindex(r1.dropna().index)))
            ry = 100 * (x / x.shift(ppy) - 1); res.append(("yoy", ppy, ry.dropna(), pb.reindex(ry.dropna().index)))
    return res


def baselines(v):
    last = v[-1]; m6 = sum(v[-6:]) / 6; half = 0.5 * last + 0.5 * m6
    w = v[-36:]; mu = sum(w) / len(w); dev = [a - mu for a in w]; den = sum(a * a for a in dev[:-1])
    phi = min(max(sum(a * b for a, b in zip(dev[:-1], dev[1:])) / den, -0.2), 0.98) if den > 1e-12 else 0.0
    return half, 0.5 * (mu + phi * (last - mu)) + 0.5 * half


rows = []
for sid, meta in cat.items():
    try:
        vs = versions(sid, meta)
    except Exception as e:
        print("skip", sid, e); continue
    for tr, ppy, x, pb in vs:
        vals = x.values.astype(float); idx = x.index; pubs = pb.values
        if len(vals) < 60:
            continue
        step = 1 if ppy == 12 else 2
        for i in range(12, len(vals) - 1, step):
            L = int(RNG.choice(LENS))                       # G3: train on short tables too (Track 4 corpora show 9-26 values)
            hist = [float(a) for a in vals[max(0, i - L + 1):i + 1]]
            for h in HZ[ppy]:
                if i + h >= len(vals):
                    continue
                fs = features(hist, h, ppy)
                if fs is None:
                    continue
                f, s = fs
                half, d18 = baselines(hist)
                y = (vals[i + h] - vals[i]) / s
                rows.append([sid, meta["release"], tr, idx[i], pd.Timestamp(pubs[i]) if pd.notna(pubs[i]) else idx[i], s, vals[i], vals[i + h],
                             max(-15, min(15, y)), (half - vals[i]) / s, (d18 - vals[i]) / s] + f)
    if len(rows) and len(rows) % 200000 < 2000:
        print(len(rows), flush=True)
cols = ["sid", "release", "transform", "obs_date", "origin", "s", "last", "future", "y", "b_half", "b_d18"] + NAMES   # h and ppy are feature columns
P = pd.DataFrame(rows, columns=cols)
P["origin"] = pd.to_datetime(P.origin, errors="coerce").fillna(pd.to_datetime(P.obs_date))
P.to_parquet(f"{D}/panel.parquet")
print("panel", P.shape, P.groupby("transform").size().to_dict(), P.origin.min(), P.origin.max())
