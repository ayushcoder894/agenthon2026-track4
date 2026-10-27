"""Class-specific CPI rules: stable components (historical sd below median: core, food, medical, shelter) use longer means; volatile keep shipped.
Chronological val/final with Holm; Dev in-sample effect printed separately."""
import os, sys, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.argv = [sys.argv[0]]; __file__ = os.path.join(HERE, "cpi_exps.py")
exec(open(os.path.join(HERE, "cpi_exps.py"), encoding="utf-8").read().split("fam = Family(")[0])
from common import Family
L = L[L.month >= "1985-01-01"].reset_index(drop=True)
STABLE = {"core", "food", "medical", "shelter"}
lg = L[[f"l{i}" for i in range(1, 13)]].values
for k in (3, 6, 9, 12): L[f"mk{k}"] = lg[:, :k].mean(1)
fam = Family("cpi_class"); sc = {}
def rows(df, w): return df[df.split == w]
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).ship.values)
def add(eid, desc, params, pt):
    f = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); sc[eid] = f; fam.run(eid, desc, params, f, base, L)
def cls(df, w, k, lam, scope):
    te = rows(df, w); stable = te.comp.isin(STABLE).values; rule = lam * te["last"].values + (1 - lam) * te[f"mk{k}"].values
    use = stable if scope == "stable" else ~stable
    return np.where(use, rule, te.ship.values)
for k, lam, scope in itertools.product((6, 9, 12), (0.0, 0.25), ("stable", "volatile")):
    add(f"cpiC_{scope}_k{k}_lam{lam}", f"{scope} components: {lam} x last + {1 - lam} x mean{k}; others shipped", dict(k=k, lam=lam, scope=scope), lambda df, w, k=k, lam=lam, scope=scope: cls(df, w, k, lam, scope))
fam.finalize(L, sc, base, top_k=3)
print("done")
