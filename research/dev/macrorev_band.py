"""E12: interval for the next published estimate of a young macro estimate (ALFRED vintages of PAYEMS, RSAFS, INDPRO, HOUST, PI,
DGORDER; realtime 2010..2022-06-28). Row = (series, vintage v, reference month m with m within 3 months of v); estimate e = value
as of v; truth = value in the next vintage that changes the release (zero revisions kept and reported separately).
History for the band = relative non-zero revisions of that series seen in the 12 months before v (as a unit's vintage tables).
Band options: symmetric k x sd x |e| around e; shifted toward the predicted direction (regime rule) by f x sd x |e|."""
import json
import numpy as np
import pandas as pd

LIMIT = pd.Timestamp("2022-06-28")
rows = []
for s in ("PAYEMS", "RSAFS", "INDPRO", "HOUST", "PI", "DGORDER"):
    d = pd.DataFrame(json.load(open(f"data_alfred/{s}.json")))
    d = d[d.value != "."]
    d["v"] = pd.to_datetime(d.realtime_start); d["m"] = pd.to_datetime(d.date); d["x"] = d.value.astype(float)
    d = d[d.v <= LIMIT]
    vint = sorted(d.v.unique())
    hist = []  # (vintage, relative revision)
    for i in range(len(vint) - 1):
        v, vn = vint[i], vint[i + 1]
        cur = d[(d.v <= v)].sort_values("v").groupby("m").x.last()
        nxt = d[(d.v <= vn)].sort_values("v").groupby("m").x.last()
        young = [m for m in cur.index if 0 <= (v - m).days <= 100]
        h12 = [r for (t, r) in hist if (v - t).days <= 365]
        for m in young:
            if m in nxt.index and cur[m]:
                rel = (nxt[m] - cur[m]) / abs(cur[m])
                own = [r for r in h12 if r != 0]
                rows.append(dict(s=s, v=v, m=m, e=cur[m], y=nxt[m], rel=rel, n=len(own),
                                 sd=np.std(own) if len(own) >= 3 else np.nan,
                                 share=np.mean([r > 0 for r in own[-12:]]) if len(own) >= 3 else np.nan))
        for m in cur.index:
            if m in nxt.index and cur[m] and nxt[m] != cur[m]:
                hist.append((vn, (nxt[m] - cur[m]) / abs(cur[m])))
R = pd.DataFrame(rows).dropna(subset=["sd"])
R = R[R.v >= "2011-01-01"]
print("rows", len(R), "non-zero", int((R.rel != 0).sum()))
R["lab_up"] = np.where(R.share < 0.3, False, np.where(R.share > 0.7, True, True))  # regime rule, else long-run 'up'


def IS(y, lo, hi):
    return np.mean((hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0))

for name, S in (("non-zero revisions", R[R.rel != 0]), ("all rows", R)):
    print(f"=== {name}: n={len(S)}")
    y = S.rel.values
    for k in (0.6, 0.8, 1.0, 1.3, 1.645):
        for f in (0.0, 0.25, 0.5):
            sh = np.where(S.lab_up, 1, -1) * f * S.sd.values
            lo, hi = sh - k * S.sd.values, sh + k * S.sd.values
            print(f"  k={k:5.3f} shift f={f:.2f}: coverage {np.mean((y >= lo) & (y <= hi)):.2f} relative IS {IS(y, lo, hi) * 1e4:.2f}bp")
