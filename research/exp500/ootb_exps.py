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
# ------------------------------------------------ yields
def yld(s):
    a = json.load(open(f"{ROOT}/data_fred/{s}.json"))["observations"]
    x = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in a if o["value"] not in (".", "")})
    h = f"{ROOT}/data_fred_holdout/{s}.csv"
    if os.path.exists(h):
        c = pd.read_csv(h); c = pd.to_numeric(c.set_index(pd.to_datetime(c.iloc[:, 0])).iloc[:, 1], errors="coerce").dropna(); x = pd.concat([x[x.index < c.index.min()], c])
    return x[~x.index.duplicated()].sort_index()
TEN2S = {"2-Year": "DGS2", "3-Year": "DGS3", "5-Year": "DGS5", "7-Year": "DGS7", "10-Year": "DGS10", "20-Year": "DGS20", "30-Year": "DGS30"}
Y = {t: yld(s) for t, s in TEN2S.items()}
A = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "panels/auc_panel.parquet")).reset_index(drop=True)
A["cutoff"] = (A.date.dt.to_period("M") - 1).dt.to_timestamp(how="end").dt.normalize()
def yfeat(r):
    s = Y[r.ten]; s = s[s.index <= r.cutoff]
    if len(s) < 260: return (np.nan, np.nan, np.nan)
    return ((s.iloc[-1] - s.iloc[-22]) * 100, s.diff().iloc[-21:].std() * 100, (s.iloc[-1] - s.iloc[-252:].mean()) * 100)
F = np.array([yfeat(r) for r in A.itertuples()]); A["dy21"], A["vol21"], A["lvl_dev"] = F[:, 0], F[:, 1], F[:, 2]
A = A.dropna(subset=["dy21"]).reset_index(drop=True)
print("auction rows with yield features", len(A), A.split.value_counts().to_dict())
def rows(df, w): return df[df.split == w]
def sp(df, w):
    tr = df[df.split == "train"] if w == "val" else df[df.split.isin(["train", "val"])]; return tr, df[df.split == w]
fam = Family("auc_ootb"); sc = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).shipped.values)
def add(eid, desc, params, pt):
    f = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); sc[eid] = f; fam.run(eid, desc, params, f, base, A)
def beta_pt(df, w, cols, a, alpha):
    tr, te = sp(df, w); X = tr[cols].values; mu, sd = X.mean(0), X.std(0) + 1e-9
    m = Ridge(alpha=a).fit((X - mu) / sd, (tr.y - tr.shipped).values); return te.shipped.values + alpha * m.predict((te[cols].values - mu) / sd)
for cols, a, al in itertools.product((["dy21"], ["vol21"], ["lvl_dev"], ["dy21", "vol21"], ["dy21", "vol21", "lvl_dev"], ["dy21", "vol21", "lvl_dev", "ind3", "szr"]), (10.0, 100.0), (0.5, 1.0)):
    add(f"aucO_{'+'.join(cols)}_a{a}_s{al}", f"shipped + ridge on {cols} a={a} x{al}", dict(cols=cols, alpha=a, scale=al), lambda df, w, cols=cols, a=a, al=al: beta_pt(df, w, cols, a, al))
H = A[[f"h{i}" for i in range(1, 14)]].values
rules = {"ship": A.shipped.values, "last": A.h1.values, "mean3": np.nanmean(H[:, :3], 1), "max3": np.nanmax(H[:, :3], 1), "median6": np.nanmedian(H[:, :6], 1), "mean13": np.nanmean(H, 1)}
for combo in (("ship", "last"), ("ship", "max3"), ("ship", "mean3", "last"), ("ship", "median6", "mean13"), ("ship", "last", "max3", "mean3")):
    v = np.mean([rules[c] for c in combo], axis=0); A[f"comb_{'_'.join(combo)}"] = v
    add(f"aucO_comb_{'_'.join(combo)}", f"equal-weight combination {combo}", dict(combo=combo), lambda df, w, c=f"comb_{'_'.join(combo)}": rows(df, w)[c].values)
fam.finalize(A, sc, base, top_k=3)

# ------------------------------------------------ CPI cross-component
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
