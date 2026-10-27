"""Research-backed CPI rules (general, validated chronologically): component-specific AR(1) persistence toward the 12-month mean (phi_c fitted on train, shrunk to pooled),
optionally capped; plus the shipped blend as baseline. Uses cpi_exps.py's panel construction."""
import sys, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpi_exps.py")).read().split("fam = Family(")[0]
exec(src)
from common import Family
fam = Family("cpi_ar"); scorers = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).ship.values)
def add(eid, desc, params, pt):
    sc = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); scorers[eid] = sc; fam.run(eid, desc, params, sc, base, L)
def ar_pt(df, w, center, shrink, recent_years, cap):
    tr, te = sp(df, w)
    if recent_years: tr = tr[tr.month >= tr.month.max() - pd.DateOffset(years=recent_years)]
    c_tr = tr.m12 if center == "m12" else tr.m6; x = tr["last"] - c_tr; y = tr.y - c_tr
    pooled = (x * y).sum() / (x * x).sum(); phi = {}
    for c, g in tr.assign(x=x, yy=y).groupby("comp"):
        p = (g.x * g.yy).sum() / max((g.x * g.x).sum(), 1e-9); n = len(g); phi[c] = (n * p + shrink * pooled) / (n + shrink)
    c_te = te.m12 if center == "m12" else te.m6
    dev = (te["last"] - c_te).clip(-cap, cap) if cap else (te["last"] - c_te)
    return c_te.values + te.comp.map(phi).values * dev.values
for center, shrink, ry, cap in itertools.product(("m12", "m6"), (0, 50, 200), (0, 10, 20), (0, 2.0)):
    add(f"cpi_ar_{center}_s{shrink}_y{ry}_c{cap}", f"AR(1) toward {center}, shrink {shrink}, recent {ry}y, cap {cap}", dict(center=center, shrink=shrink, years=ry, cap=cap),
        lambda df, w, center=center, shrink=shrink, ry=ry, cap=cap: ar_pt(df, w, center, shrink, ry, cap))
fam.finalize(L, scorers, base, top_k=3)
# report fitted phi on all data <= 2024 for the record
tr = L; x = tr["last"] - tr.m12; y = tr.y - tr.m12
print({c: round(float((g_x := x[tr.comp == c]) .dot(y[tr.comp == c]) / g_x.dot(g_x)), 3) for c in sorted(L.comp.unique())})
print("done")
