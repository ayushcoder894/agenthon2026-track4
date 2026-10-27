"""CPI rules re-tested on FIRST-PRINT targets (ALFRED vintages), month pseudo-units of the 8 non-pump components, composite formula vs D20R.
Features are first prints of earlier months (what a cutoff-time table shows, except annual SA revisions). Expanding window, refit yearly."""
import os, sys, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
def iscore(y, lo, hi): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
def composite(y, p, lo, hi, npnt, nlo, nhi):
    m0, m1 = np.abs(y - npnt).mean(), np.abs(y - p).mean(); q = m0 / (m0 + m1) if (m0 + m1) > 0 else 0.5
    i0, i1 = iscore(y, nlo, nhi).mean(), iscore(y, lo, hi).mean(); iq = i0 / (i0 + i1)
    return 0.7 * q + 0.3 * min(iq, max(0.5, q)), q, iq
def boot(diff, years, n=2000, seed=0):
    diff, years = np.asarray(diff), np.asarray(years); u = np.unique(years); idx = {y: np.where(years == y)[0] for y in u}; rng = np.random.default_rng(seed); est = []
    for _ in range(n):
        s = rng.choice(u, len(u)); est.append(diff[np.concatenate([idx[x] for x in s])].mean())
    return float(diff.mean()), float(np.std(est))
FP = pd.read_csv("D:/AGENTHON/data_alfred_cpi/first_print_mom.csv", parse_dates=["month", "vintage"])
MAP = {"CPIAUCSL": "headline", "CPILFESL": "core", "CUSR0000SAH1": "shelter", "CPIENGSL": "energy", "CUSR0000SETB01": "gasoline", "CPIUFDSL": "food",
       "CUSR0000SETA01": "newveh", "CUSR0000SETA02": "usedcars", "CPIAPPSL": "apparel", "CPIMEDSL": "medical", "CUSR0000SAS4": "transpsvc"}
FP["comp"] = FP.series.map(MAP); FP = FP[FP.month <= "2024-10-01"]
W = FP.pivot_table(index="month", columns="comp", values="first_mom").sort_index()
NONPUMP = ["core", "food", "medical", "newveh", "shelter", "transpsvc", "usedcars", "apparel"]
rows = []
for c in NONPUMP:
    s = W[c]
    for i in range(12, len(s)):
        t = s.index[i]; h = s.iloc[i - 12:i].values[::-1]            # h[0] = t-1 ... h[11] = t-12
        if np.isnan(h).any() or np.isnan(s.iloc[i]): continue
        rows.append(dict(comp=c, month=t, y=s.iloc[i], **{f"l{k + 1}": h[k] for k in range(12)}))
C = pd.DataFrame(rows); lg = C[[f"l{i}" for i in range(1, 13)]].values
C["last"] = C.l1; C["m6"] = lg[:, :6].mean(1); C["m12"] = lg.mean(1); C["ship"] = 0.5 * C["last"] + 0.5 * C.m6
C["sd9"] = lg[:, :9].std(1, ddof=1); C["half"] = np.maximum(2.0 * C.sd9, 0.25)
def runlen(r):
    s = np.sign(r[1]); k = 0
    for v in r[1:]:
        if v != 0 and np.sign(v) == s: k += 1
        else: break
    return k
C["run_prev"] = [runlen(r) for r in lg]; C["prev3"] = lg[:, 1:4].mean(1)
C["tsig"] = ((np.sign(C["last"]) != np.sign(C.l2)) & (C.run_prev >= 3) & (C["last"] != 0)) * np.sign(C["last"]) * np.abs(C.prev3)
X = pd.concat({f"x_{c}_l{k}": W[c].shift(k) for c in W.columns for k in (1, 2, 3)}, axis=1)
C = C.merge(X, left_on="month", right_index=True, how="left").dropna().reset_index(drop=True); XC = list(X.columns)
C["year"] = C.month.dt.year
# --- descriptive: first-print persistence (phi toward m12) by component, and conditional reversal after spikes
print("first-print phi toward 12m mean (1995-2023):")
H = C[(C.month >= "1995-01-01") & (C.month < "2024-01-01")]
for c in NONPUMP:
    g = H[H.comp == c]; x = g["last"] - g.m12; yy = g.y - g.m12; z = (g["last"] - g.m12) / (g.l1.std() + 1e-9)
    big = g[z > 1.5]; print(f"  {c:9s} phi {(x * yy).sum() / (x * x).sum():+.3f} | after z>1.5 spikes (n={len(big)}): mean next {big.y.mean() if len(big) else float('nan'):+.2f} vs last {big['last'].mean() if len(big) else float('nan'):+.2f}")
# --- rules, expanding yearly refit, pseudo-unit = month (8 components)
res = []; cache = {}
for mo in sorted(C.month.unique()):
    mo = pd.Timestamp(mo)
    if mo < pd.Timestamp("2000-01-01"): continue
    te = C[C.month == mo]
    if len(te) < 8: continue
    yr = mo.year
    if yr not in cache:
        tr = C[C.month < pd.Timestamp(f"{yr}-01-01")]
        ts = tr[tr.tsig != 0]; b = float((ts.tsig * (ts.y - ts.ship)).sum() / (ts.tsig ** 2).sum()) if len(ts) > 30 else 0.0
        rid = {}
        for c in NONPUMP:
            g = tr[tr.comp == c]; M = g[XC].values; mu, sd = M.mean(0), M.std(0) + 1e-9
            rid[c] = (Ridge(alpha=1000.0).fit((M - mu) / sd, (g.y - g.ship).values), mu, sd)
        phi = {}
        for c in NONPUMP:
            g = tr[tr.comp == c]; x = g["last"] - g.m12; yy = g.y - g.m12; phi[c] = float((x * yy).sum() / (x * x).sum())
        cache[yr] = (b, rid, phi)
    b, rid, phi = cache[yr]
    y = te.y.values; npnt = te["last"].values; nlo, nhi = npnt - 1.645 * te.sd9.values, npnt + 1.645 * te.sd9.values; h = te.half.values
    p0 = te.ship.values; p1 = p0 + b * te.tsig.values
    adj = np.array([0.5 * rid[c][0].predict(((te[te.comp == c][XC].values - rid[c][1]) / rid[c][2]))[0] for c in te.comp]); p2 = p1 + adj
    p3 = te.m12.values + np.array([phi[c] for c in te.comp]) * (te["last"].values - te.m12.values)
    p4 = 0.5 * p2 + 0.5 * p3
    row = dict(month=mo, year=yr)
    for nm, p in (("D20R", p0), ("turning", p1), ("turn+xcomp", p2), ("AR_firstprint", p3), ("blend_xcomp_AR", p4)):
        row[nm] = composite(y, p, p - h, p + h, npnt, nlo, nhi)[0]
    res.append(row)
R = pd.DataFrame(res)
print(f"\nmonth pseudo-units {len(R)} (2000-2024)")
for m in ("turning", "turn+xcomp", "AR_firstprint", "blend_xcomp_AR"):
    d, se = boot(R[m] - R.D20R, R.year); r2 = R[R.year >= 2019]; d2, se2 = boot(r2[m] - r2.D20R, r2.year)
    print(f"  {m:16s} vs D20R {d:+.4f} (z {d / se:+.2f}) | 2019-24 {d2:+.4f} (z {d2 / se2:+.2f})")
# --- Dev October-2024 forecast with first-print-trained rules (fits use months < 2024-10, all published before the cutoff)
te = C[C.month == pd.Timestamp("2024-10-01")]; tr = C[C.month < pd.Timestamp("2024-10-01")]
ts = tr[tr.tsig != 0]; b = float((ts.tsig * (ts.y - ts.ship)).sum() / (ts.tsig ** 2).sum())
out = {}
for c in NONPUMP:
    g = tr[tr.comp == c]; M = g[XC].values; mu, sd = M.mean(0), M.std(0) + 1e-9; r = Ridge(alpha=1000.0).fit((M - mu) / sd, (g.y - g.ship).values)
    t1 = te[te.comp == c].iloc[0]; adj = 0.5 * r.predict(((te[te.comp == c][XC].values - mu) / sd))[0]
    x = g["last"] - g.m12; yy = g.y - g.m12; ph = float((x * yy).sum() / (x * x).sum())
    out[c] = dict(actual=t1.y, D20R=t1.ship, turn_xcomp=t1.ship + b * t1.tsig + adj, AR=t1.m12 + ph * (t1["last"] - t1.m12))
O = pd.DataFrame(out).T; print("\nOct-2024 (first-print targets):"); print(O.round(3).to_string())
for col in ("D20R", "turn_xcomp", "AR"): print(f"  {col}: MAE over 8 non-pump comps {np.abs(O[col] - O.actual).mean():.3f}")
