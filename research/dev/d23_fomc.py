"""D23-1: shared feature-based FOMC adjustment. Fit on ALL historical meetings (2002-2022-05, 168) with the runtime-available curve features, predict the two Dev meetings,
and compare with what a constant offset would be worth. Features at the cutoff: 10-day change of DGS2 and DGS10 (from the snapshot window), slope DGS10-DGS2, gap DGS1-policy mid, policy move at the meeting."""
import re, json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
src = open("t4-local/dev/fomc_text.py").read().split("TEXT = [")[0]
exec(src)
CURVE = ["slope", "gap", "lvl10", "mom10", "mom2", "pmove"]
tr = D.copy()
Xtr = tr[CURVE].fillna(0).values; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Ytr = tr[["d_" + m for m in MATS]].clip(-120, 120).values
models = [Ridge(alpha=60.0).fit((Xtr - mu) / sd, Ytr[:, k]) for k in range(len(MATS))]
U = "track4-analysis-public/units/"
units = {"t4-fomc-curve-20220728": dict(pol=2.375, pmove=75.0, snap="RATES_SNAPSHOT_20220728.json"), "t4-fomc-curve-20240918": dict(pol=4.875, pmove=-50.0, snap="RATES_SNAPSHOT_20240919.json")}
act = {"t4-fomc-curve-20220728": [111, 113, 106, 100, 89, 57], "t4-fomc-curve-20240918": [68, 73, 78, 77, 69, 54]}   # DGS2,3,5,7,10,30 first-published outcomes (audit table)
for u, c in units.items():
    t = json.load(open(U + u + "/corpus/" + c["snap"]))["text"]
    rows = re.findall(r"(\d{4}-\d{2}-\d{2})\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", t)
    cols = ["date", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
    S = pd.DataFrame(rows, columns=cols).set_index("date").astype(float)
    print(u, "snapshot days", len(S), S.index[0], "->", S.index[-1])
    last = S.iloc[-1]; first = S.iloc[0]
    f = dict(slope=(last.DGS10 - last.DGS2) * 100, gap=(last.DGS1 - c["pol"]) * 100, lvl10=last.DGS10, mom10=(last.DGS10 - first.DGS10) * 100, mom2=(last.DGS2 - first.DGS2) * 100, pmove=c["pmove"])
    x = (np.array([f[k] for k in CURVE]) - mu) / sd
    p = np.array([m.predict(x[None, :])[0] for m in models])
    a = np.array(act[u], float)
    print("  features", {k: round(v, 2) for k, v in f.items()})
    print("  shared-model predicted change (bp) DGS2..30:", np.round(p, 1), "| outcome:", a)
    mae0 = np.abs(a).mean()
    for al in (1.0, 3.0, 10.0):
        mae = np.abs(a - al * p).mean(); print(f"  scale x{al:>4}: MAE {mae:.1f} vs zero-change {mae0:.1f}  q={mae0 / (mae0 + mae):.3f}")
    for c0 in (30, 50, 70):
        mae = np.abs(a - c0).mean(); print(f"  constant +{c0}bp (outcome-derived offset, NOT a forecast): MAE {mae:.1f}  q={mae0 / (mae0 + mae):.3f}")
