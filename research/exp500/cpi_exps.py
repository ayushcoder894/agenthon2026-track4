"""CPI component m/m experiments (~45). Panel built from local FRED history (SA indexes; revised vintages = proxy for first prints, a stated limitation). One-step-ahead: features use data up to the previous month.
Baseline = shipped point 0.5 x last + 0.5 x mean(last 6). Components: 11 (headline, core, food, energy, gasoline, apparel, medical, new vehicles, shelter, transport services, used cars)."""
import json, sys, itertools, glob, os
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Family

ROOT = os.environ.get("EXP500_DATA", "D:/AGENTHON")
SER = {"CPIAUCSL": "headline", "CPILFESL": "core", "CPIUFDSL": "food", "CPIENGSL": "energy", "CUSR0000SETB01": "gasoline", "CPIAPPSL": "apparel", "CPIMEDSL": "medical",
       "CUSR0000SETA01": "newveh", "CUSR0000SAH1": "shelter", "CUSR0000SAS4": "transpsvc", "CUSR0000SETA02": "usedcars"}
frames = []
for sid, nm in SER.items():
    a = json.load(open(f"{ROOT}/data_fred/{sid}.json"))["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in a if o["value"] not in (".", "")})
    h = f"{ROOT}/data_fred_holdout/{sid}.csv"
    if os.path.exists(h):
        c = pd.read_csv(h); c = c.set_index(c.columns[0])[c.columns[1]]; c.index = pd.to_datetime(c.index); c = pd.to_numeric(c, errors="coerce").dropna()
        s = pd.concat([s[s.index < c.index.min()], c])
    s = s[~s.index.duplicated()].sort_index(); m = (s / s.shift(1) - 1) * 100
    frames.append(pd.DataFrame({"comp": nm, "month": m.index, "y": m.values}))
L = pd.concat(frames).dropna().sort_values(["comp", "month"]).reset_index(drop=True)
g = L.groupby("comp").y
for k in range(1, 25): L[f"l{k}"] = g.shift(k)
L["mon"] = L.month.dt.month
L = L.dropna(subset=[f"l{k}" for k in range(1, 13)]).reset_index(drop=True)
L["split"] = np.where(L.month <= "2012-12-31", "train", np.where(L.month <= "2018-12-31", "val", "final"))
L["block"] = L.month.dt.year * 10 + (L.month.dt.month - 1) // 3 + 1
lags = L[[f"l{k}" for k in range(1, 13)]].values
core_s = L[L.comp == "core"].set_index("month").y; L["core_l1"] = L.month.map(core_s.shift(1)); L["core_m6"] = L.month.map(core_s.rolling(6).mean().shift(1))
L["last"] = L.l1; L["m6"] = lags[:, :6].mean(1); L["m12"] = lags.mean(1)
L["ship"] = 0.5 * L["last"] + 0.5 * L["m6"]
print("rows", len(L), L.split.value_counts().to_dict(), L.month.min().date(), L.month.max().date())
def rows(df, w): return df[df.split == w]
def sp(df, w):
    tr = df[df.split == "train"] if w == "val" else df[df.split.isin(["train", "val"])]; return tr, df[df.split == w]

fam = Family("cpi_point"); scorers = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).ship.values)
def add(eid, desc, params, pt):
    sc = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); scorers[eid] = sc; fam.run(eid, desc, params, sc, base, L)
LAGM = {k: np.array([f"l{i}" for i in range(1, k + 1)]) for k in (1, 3, 6, 12)}
def kmean(df, w, k): return rows(df, w)[[f"l{i}" for i in range(1, k + 1)]].mean(axis=1).values
for lam, k in itertools.product((0.0, 0.25, 0.75, 1.0), (3, 6, 12)):
    add(f"cpi_blend_l{lam}_k{k}", f"{lam} x last + {1 - lam} x mean{k}", dict(lam=lam, k=k), lambda df, w, lam=lam, k=k: lam * rows(df, w)["last"].values + (1 - lam) * kmean(df, w, k))
add("cpi_mean6_only", "mean6", dict(k=6), lambda df, w: kmean(df, w, 6))
for hl in (1, 2, 3, 6, 12):
    wts = 0.5 ** (np.arange(12) / hl)
    add(f"cpi_ewma{hl}", f"EWMA half-life {hl}m", dict(hl=hl), lambda df, w, wts=wts: (rows(df, w)[[f"l{i}" for i in range(1, 13)]].values * wts).sum(1) / wts.sum())
for k in (3, 6, 12):
    add(f"cpi_median{k}", f"median of last {k}", dict(k=k), lambda df, w, k=k: np.median(rows(df, w)[[f"l{i}" for i in range(1, k + 1)]].values, axis=1))
# component-specific seasonal deviation from train
def seasonal(df, w, sh):
    tr, te = sp(df, w); dev = (tr.y - tr.m12).groupby([tr.comp, tr.mon]).mean(); return te.ship.values + sh * np.array([dev.get((c, m), 0.0) for c, m in zip(te.comp, te.mon)])
for sh in (0.25, 0.5, 1.0):
    add(f"cpi_seasonal{sh}", f"shipped + {sh} x component-month seasonal dev", dict(shrink=sh), lambda df, w, sh=sh: seasonal(df, w, sh))
FE = {"lags6": [f"l{i}" for i in range(1, 7)], "lags12": [f"l{i}" for i in range(1, 13)], "lags12+core": [f"l{i}" for i in range(1, 13)] + ["core_l1", "core_m6"]}
def ridge_pt(df, w, fs, a, per_comp):
    tr, te = sp(df, w); cols = FE[fs]; out = np.zeros(len(te))
    if not per_comp:
        Xtr, Xte = tr[cols].fillna(0).values, te[cols].fillna(0).values; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        return Ridge(alpha=a).fit((Xtr - mu) / sd, (tr.y - tr.ship).values).predict((Xte - mu) / sd) + te.ship.values
    for c in te.comp.unique():
        a_, b_ = tr[tr.comp == c], te[te.comp == c]; Xtr, Xte = a_[cols].fillna(0).values, b_[cols].fillna(0).values; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        out[(te.comp == c).values] = Ridge(alpha=a).fit((Xtr - mu) / sd, (a_.y - a_.ship).values).predict((Xte - mu) / sd) + b_.ship.values
    return out
for fs, a, pc in itertools.product(FE, (100.0, 1000.0), (False, True)):
    add(f"cpi_ridge_{fs}_a{a}_{'comp' if pc else 'pool'}", f"ridge residual {fs} a={a} {'per-component' if pc else 'pooled'}", dict(fs=fs, alpha=a, per_comp=pc), lambda df, w, fs=fs, a=a, pc=pc: ridge_pt(df, w, fs, a, pc))
def gbm_pt(df, w, lv, ne):
    tr, te = sp(df, w); cols = FE["lags12+core"] + ["mon"]; tr_ = tr.assign(ci=tr.comp.astype("category").cat.codes); cats = tr.comp.astype("category").cat.categories
    ci_tr = tr.comp.map({c: i for i, c in enumerate(cats)}).values; ci_te = te.comp.map({c: i for i, c in enumerate(cats)}).fillna(-1).values
    Xtr = np.column_stack([tr[cols].values, ci_tr]); Xte = np.column_stack([te[cols].values, ci_te])
    m = lgb.LGBMRegressor(objective="l1", n_estimators=ne, learning_rate=0.05, num_leaves=lv, min_child_samples=40, subsample=0.8, subsample_freq=1, reg_lambda=10.0, verbose=-1, random_state=0)
    return te.ship.values + m.fit(Xtr, (tr.y - tr.ship).values).predict(Xte)
for lv, ne in itertools.product((4, 8), (100, 300)):
    add(f"cpi_gbm_l{lv}_n{ne}", f"lgbm-L1 residual leaves={lv} n={ne}", dict(leaves=lv, n_est=ne), lambda df, w, lv=lv, ne=ne: gbm_pt(df, w, lv, ne))
# per-component choice of best simple rule on train (partial adaptive)
def best_rule(df, w):
    tr, te = sp(df, w); out = np.zeros(len(te))
    for c in te.comp.unique():
        a_, b_ = tr[tr.comp == c], te[te.comp == c]; opts = {"ship": (a_.ship, b_.ship), "last": (a_["last"], b_["last"]), "m6": (a_.m6, b_.m6), "m12": (a_.m12, b_.m12)}
        key = min(opts, key=lambda k: np.abs(a_.y - opts[k][0]).mean()); out[(te.comp == c).values] = opts[key][1].values
    return out
add("cpi_percomp_bestrule", "per-component best of {ship,last,m6,m12} on train", dict(), best_rule)
fam.finalize(L, scorers, base, top_k=3)
print("done")
