"""Out-of-the-box experiments (general rules, chronological + one-shot final + Holm):
AUCTIONS: (1) reach-for-yield: tenor yield change over the 21 trading days before the cutoff; (2) rate volatility (sd of daily changes, 21d); (3) yield level vs 1y mean;
          (4) forecast combinations of simple rules. Point = shipped + beta x feature (beta fitted on train through-origin, shrunk by alpha).
CPI:      (5) cross-component leading signals (all 11 components' last 1-3 prints) ridge per target component; (6) equal-weight forecast combinations."""
import sys, os, json, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Family
ROOT = os.environ.get("EXP500_DATA", "D:/AGENTHON")

if os.environ.get("SKIP_AUC") != "1":
  pass

# CPI only

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpi_exps.py")).read().split("fam = Family(")[0]
exec(src)    # -> L (component-month panel), rows, sp
W = L.pivot_table(index="month", columns="comp", values="y")
LAG = pd.concat({f"x_{c}_l{k}": W[c].shift(k) for c in W.columns for k in (1, 2, 3)}, axis=1)
L2 = L.merge(LAG, left_on="month", right_index=True, how="left").dropna(subset=list(LAG.columns)).reset_index(drop=True)
fam = Family("cpi_ootb"); sc = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).ship.values)
def add(eid, desc, params, pt):
    f = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); sc[eid] = f; fam.run(eid, desc, params, f, base, L2)
def xcomp(df, w, k, a, al):
    tr, te = sp(df, w); cols = [c for c in LAG.columns if int(c[-1]) <= k]; out = np.zeros(len(te))
    for comp in te.comp.unique():
        a_, b_ = tr[tr.comp == comp], te[te.comp == comp]; X = a_[cols].values; mu, sd = X.mean(0), X.std(0) + 1e-9
        out[(te.comp == comp).values] = b_.ship.values + al * Ridge(alpha=a).fit((X - mu) / sd, (a_.y - a_.ship).values).predict((b_[cols].values - mu) / sd)
    return out
for k, a, al in itertools.product((1, 3), (100.0, 1000.0, 5000.0), (0.5, 1.0)):
    add(f"cpiO_xcomp_k{k}_a{a}_s{al}", f"cross-component ridge, {k} lags, a={a}, x{al}", dict(k=k, alpha=a, scale=al), lambda df, w, k=k, a=a, al=al: xcomp(df, w, k, a, al))
Lg = L2[[f"l{i}" for i in range(1, 13)]].values
R = {"ship": L2.ship.values, "last": L2["last"].values, "m12": L2.m12.values, "median6": np.median(Lg[:, :6], 1), "zero": np.zeros(len(L2))}
for combo in (("ship", "m12"), ("ship", "median6"), ("ship", "m12", "median6"), ("last", "m12", "median6"), ("ship", "zero")):
    L2[f"comb_{'_'.join(combo)}"] = np.mean([R[c] for c in combo], axis=0)
    add(f"cpiO_comb_{'_'.join(combo)}", f"equal-weight combination {combo}", dict(combo=combo), lambda df, w, c=f"comb_{'_'.join(combo)}": rows(df, w)[c].values)
fam.finalize(L2, sc, base, top_k=3)
print("done")
