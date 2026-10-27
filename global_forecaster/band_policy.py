"""Score-aware interval width (Idea 2B). The interval leg is 0.3 x min(iq, max(0.5, q)): interval credit above 0.5 needs a point edge.
Policies (half-widths around the point scaled by k, asymmetry kept):
  uniform k                      : one multiplier everywhere
  edge-dependent (k_lo, k_hi)    : k_hi on pseudo-units whose predicted edge is small, k_lo where it is large;
                                   predicted edge = mean |model delta| / mean band half-width of the unit (ex ante, no outcome used)
Chosen on VALIDATION 2013-2017 by mean composite; judged on 2018-2022 and 2023-2026.
Naive answer for a pseudo-unit: carry-forward point, random-walk band +-1.645 sqrt(h) x 1.25 s (stand-in: real naive bands are hidden)."""
import pickle, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "D:/AGENTHON/t4-local/global")
from gfeat import NAMES
D = "D:/AGENTHON/data_global"
M = pickle.load(open(f"{D}/global_lim2012.pkl", "rb"))
P = pd.read_parquet(f"{D}/panel.parquet"); P = P[P.origin >= "2013-01-01"].copy()
P["yr"] = P.origin.dt.year
P["m"] = M["lam"] * M["pt"].predict(P[NAMES]); P["lo"] = np.minimum(M["q05"].predict(P[NAMES]), P.m); P["hi"] = np.maximum(M["q95"].predict(P[NAMES]), P.m)
P["mon"] = P.origin.dt.to_period("M"); P["grp"] = P.groupby(["mon", "h", "ppy", "transform"]).cumcount() // 10
P["rw"] = 1.645 * np.sqrt(P.h) * 1.25
P["ratio"] = P.m.abs() / ((P.hi - P.lo) / 2)
U = P.groupby(["mon", "h", "ppy", "transform", "grp"])
P["edge"] = U.ratio.transform("mean"); P["n"] = U.y.transform("size"); P = P[P.n >= 5]
def IS(lo, hi, y): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
P["e0"], P["em"] = P.y.abs(), (P.y - P.m).abs(); P["is0"] = IS(-P.rw, P.rw, P.y)
keys = ["mon", "h", "ppy", "transform", "grp"]
G = P.groupby(keys).agg(e0=("e0", "mean"), em=("em", "mean"), is0=("is0", "mean"), edge=("edge", "first"), yr=("yr", "first"))
G["q"] = (G.e0 / (G.e0 + G.em)).where(G.e0 + G.em > 0, 0.5)
def unit_is(k):
    lo, hi = P.m - k * (P.m - P.lo), P.m + k * (P.hi - P.m)
    return P.assign(isk=IS(lo, hi, P.y)).groupby(keys).isk.mean()
KS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
ISK = {k: unit_is(k) for k in KS}
def composite(kser):
    isk = pd.concat([ISK[k].where(kser == k) for k in KS], axis=1).bfill(axis=1).iloc[:, 0]
    iq = G.is0 / (G.is0 + isk); return 0.7 * G.q + 0.3 * np.minimum(iq, np.maximum(0.5, G.q))
med = G[G.yr <= 2017].edge.median()
print(f"pseudo-units {len(G):,} | median predicted edge (validation) {med:.3f}")
res = []
for k in KS:
    c = composite(pd.Series(k, index=G.index)); res.append(("uniform", k, k, c))
for klo, khi in itertools.product(KS, KS):
    if khi <= klo: continue
    c = composite(pd.Series(np.where(G.edge >= med, klo, khi), index=G.index)); res.append(("edge-dependent", klo, khi, c))
val = [(nm, a, b, c[G.yr <= 2017].mean()) for nm, a, b, c in res]
best_u = max((r for r in val if r[0] == "uniform"), key=lambda r: r[3]); best_e = max((r for r in val if r[0] != "uniform"), key=lambda r: r[3])
base = next(c for nm, a, b, c in res if nm == "uniform" and a == 1.0)
for tag, (nm, a, b, _) in (("best uniform", best_u), ("best edge-dependent", best_e)):
    c = next(cc for n2, a2, b2, cc in res if (n2, a2, b2) == (nm, a, b))
    print(f"{tag:20s} k={a}/{b}: composite vs k=1 | val {(c - base)[G.yr <= 2017].mean():+.4f} | 2018-22 {(c - base)[(G.yr >= 2018) & (G.yr <= 2022)].mean():+.4f} | 2023-26 {(c - base)[G.yr >= 2023].mean():+.4f}")
print(f"k=1 composite level: val {base[G.yr <= 2017].mean():.4f} 2018-22 {base[(G.yr >= 2018) & (G.yr <= 2022)].mean():.4f} 2023-26 {base[G.yr >= 2023].mean():.4f}")
