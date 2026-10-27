"""Targeted CPI component tests on FIRST-PRINT targets, time-ordered (expanding window, refit yearly), each component judged alone:
 (1) Gasoline / energy: D20R pump pass-through artifact (fitted on revised CPI 2000-2015) vs the same model refit on first prints, using EIA weekly
     regular-gasoline prices (target month mean vs previous month mean, both months fully observed before the CPI release; the corpus carries the same table).
 (2) Apparel: does it reverse after spikes or in October? conditional means of the next first print; D20R rule vs reversal rules, out of time."""
import json, numpy as np, pandas as pd
FP = pd.read_csv("D:/AGENTHON/data_alfred_cpi/first_print_mom.csv", parse_dates=["month"])
W = FP.pivot_table(index="month", columns="series", values="first_mom").sort_index()
g = pd.DataFrame(json.load(open("D:/AGENTHON/data_alfred_cpi/GASREGW.json"))["observations"]); g = g[g.value != "."]
g["date"] = pd.to_datetime(g.date); g["v"] = g.value.astype(float); g["month"] = g.date.dt.to_period("M").dt.to_timestamp()
pm = g.groupby("month").v.mean(); pump = (pm / pm.shift(1) - 1) * 100
art = json.load(open("D:/AGENTHON/t4-local/agent/t4_agent/artifacts.json"))["families"]["cpi_pump"]["series"]
def d20r_pump(sid, t, m6, core6, food6):
    s = art[sid]; feats = {"pump": pump.get(t, np.nan), "m6": m6, "core6": core6, "food6": food6}
    return s["intercept"] + sum(c * feats[f] for f, c in zip(s["feats"], s["coef"])) + s["month"][t.month - 1]
rows = []
for t in W.index:
    if t < pd.Timestamp("2001-01-01") or t > pd.Timestamp("2024-10-01") or t not in pump.index or np.isnan(pump.get(t, np.nan)): continue
    i = W.index.get_loc(t); prev = W.iloc[i - 6:i]
    for sid in ("CUSR0000SETB01", "CPIENGSL", "CPIAUCSL"):
        if np.isnan(W.at[t, sid]): continue
        rows.append(dict(month=t, sid=sid, y=W.at[t, sid], last=W.iloc[i - 1][sid], pump=pump[t], mon=t.month,
                         d20r=d20r_pump(sid, t, prev[sid].mean(), prev["CPILFESL"].mean(), prev["CPIUFDSL"].mean())))
P = pd.DataFrame(rows); P["year"] = P.month.dt.year
res = []
for yr in range(2010, 2025):
    te = P[P.year == yr]; tr = P[P.year < yr]
    for sid in ("CUSR0000SETB01", "CPIENGSL", "CPIAUCSL"):
        a, b = tr[tr.sid == sid], te[te.sid == sid]
        if len(b) == 0: continue
        X = np.c_[np.ones(len(a)), a.pump, pd.get_dummies(a.mon).reindex(columns=range(1, 13), fill_value=0).values[:, 1:]]
        coef = np.linalg.lstsq(X, a.y.values, rcond=None)[0]
        Xb = np.c_[np.ones(len(b)), b.pump, pd.get_dummies(b.mon).reindex(columns=range(1, 13), fill_value=0).values[:, 1:]]
        for r_, p in zip(b.itertuples(), Xb @ coef): res.append(dict(year=yr, sid=sid, y=r_.y, naive=r_.last, d20r=r_.d20r, fp_refit=p, month=r_.month))
R = pd.DataFrame(res)
print("Pump pass-through on FIRST-PRINT targets, out of time (refit yearly), MAE:")
for sid, nm in (("CUSR0000SETB01", "gasoline"), ("CPIENGSL", "energy"), ("CPIAUCSL", "headline")):
    s = R[R.sid == sid]
    for per, m in (("2010-24", s.year >= 2010), ("2019-24", s.year >= 2019)):
        x = s[m]; print(f"  {nm:8s} {per}: naive {np.abs(x.y - x.naive).mean():.3f} | D20R artifact {np.abs(x.y - x.d20r).mean():.3f} | first-print refit {np.abs(x.y - x.fp_refit).mean():.3f}  (n={len(x)})")
oc = R[R.month == pd.Timestamp("2024-10-01")]
print("Oct-2024:", oc[["sid", "y", "d20r", "fp_refit"]].round(3).to_string(index=False))
# ---------------- apparel
ap = W["CPIAPPSL"].dropna(); A = pd.DataFrame({"y": ap, "last": ap.shift(1), "m6": ap.shift(1).rolling(6).mean(), "sd": ap.shift(1).rolling(24).std()}).dropna()
A["z"] = (A["last"] - A.m6) / A.sd; A["mon"] = A.index.month; A["ship"] = 0.5 * A["last"] + 0.5 * A.m6
print("\nApparel first prints (1990-2024):")
for nm, m in (("all months", A.index == A.index), ("after z>1.5 spikes", A.z > 1.5), ("October", A.mon == 10), ("October after Sep spike z>1.5", (A.mon == 10) & (A.z > 1.5))):
    s = A[m]; print(f"  {nm:30s} n={len(s):3d} mean next {s.y.mean():+.3f} | mean last {s['last'].mean():+.3f} | MAE D20R {np.abs(s.y - s.ship).mean():.3f} vs reversal(-0.5*last) {np.abs(s.y + 0.5 * s['last']).mean():.3f} vs zero {np.abs(s.y).mean():.3f}")
print("  Octobers after big Septembers:", A[(A.mon == 10) & (A.z > 1.0)][["last", "y"]].round(2).to_string())
