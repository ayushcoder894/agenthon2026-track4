"""EPS experiments (direction + growth) x 3 universes (ALL / LARGE / FIN) ~ 237 experiments. Chronological: fit on split train (for val) or train+val (for final)."""
import json, sys, itertools
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
sys.path.insert(0, ".")
from common import Family, log

P = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "panels/eps_panel.parquet")
W_BUCKET = {"-10..0": 0.0, "-50..-10": 0.5, "0..10": 1.25, "10..30": 1.0, "<=-50": 0.5, ">30": 1.0}      # shipped (eps_runrate artifact)
P["block"] = P.pe_next.dt.year * 10 + P.pe_next.dt.quarter      # quarter blocks (year blocks gave only 3 validation blocks -> unusable SEs)
P["rr_c"] = P.rr.clip(-300, 300); P["gl"] = P.g_last.clip(-300, 300)
P["ratio"] = (P.eps_last / P.abs_prior.clip(lower=0.05)).clip(-20, 20); P["labs"] = np.log(P.abs_prior.clip(lower=0.01))
P["yc"] = P.y.clip(-300, 300); P["up"] = P.up_next.astype(int)
e0, e1, e2, e3, e4 = P.eps_last, P.eps_diluted_l1, P.eps_diluted_l2, P.eps_diluted_l3, P.eps_diluted_l4
pr = P.eps_prior_next
mult = (e3 / e4.where(e4.abs() > 0.02)).clip(0.5, 2.0).fillna(1.0)
VAR = {"V1_add_seasonal": e0 + (e3 - e4), "V2_mult_seasonal": e0 * mult, "V3_median4": pd.concat([e0, e1, e2, e3], axis=1).median(axis=1),
       "V4_blend_seas": 0.5 * e0 + 0.5 * e0 * mult, "V5_mean4": pd.concat([e0, e1, e2, e3], axis=1).mean(axis=1), "V6_blend_med": 0.5 * e0 + 0.5 * pd.concat([e0, e1, e2, e3], axis=1).median(axis=1)}
SEQ = ["f_qoq", "f_jump", "f_z", "f_trend"]
BRIDGE = ["f_pm", "f_pm_yoy", "f_etr", "f_etr_dev", "f_sh_yoy", "f_rev_yoy", "f_nonop", "f_ni_rev", "f_int_rev", "f_ni_yoy"]
FS = {"rr": ["rr_c"], "seq": ["rr_c"] + SEQ, "bridge": ["rr_c"] + BRIDGE, "all": ["rr_c", "gl", "ratio", "labs"] + SEQ + BRIDGE + ["f_roa", "f_eq_assets"]}
UNIV = {"ALL": lambda d: d, "LARGE": lambda d: d[(d.rev_last >= 1e9) & (d.abs_prior >= 0.5)], "FIN": lambda d: d[d.fin]}
EDGES = [-50, -10, 0, 10, 30]; LABS = ["<=-50", "-50..-10", "-10..0", "0..10", "10..30", ">30"]
P["bucket"] = pd.cut(P.rr_c, bins=[-np.inf] + EDGES + [np.inf], labels=LABS).astype(str)
P["w_ship"] = P.bucket.map(W_BUCKET).astype(float)


def std_fit(Xtr):
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9; return mu, sd


def cached(fn):
    memo = {}
    def g(key, df, which):
        k = (key, which, id(df))
        if k not in memo: memo[k] = fn(key, df, which)
        return memo[k]
    return g


def split_rows(df, which):
    tr = df[df.split == "train"] if which == "val" else df[df.split.isin(["train", "val"])]
    return tr, df[df.split == which]


def logit_pred(df, which, fs, C):
    tr, te = split_rows(df, which); cols = FS[fs]
    Xtr, Xte = tr[cols].fillna(0).values, te[cols].fillna(0).values; mu, sd = std_fit(Xtr)
    m = LogisticRegression(C=C, max_iter=300).fit((Xtr - mu) / sd, tr.up.values); return m.predict_proba((Xte - mu) / sd)[:, 1]


def gbm_pred(df, which, fs, leaves, n_est):
    tr, te = split_rows(df, which); cols = FS[fs]
    m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.05, num_leaves=leaves, min_child_samples=100, subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=20.0, verbose=-1, random_state=0)
    m.fit(tr[cols].values, tr.up.values); return m.predict_proba(te[cols].values)[:, 1]


def resid_pred(df, which, fs, kind, a, leaves=None, n_est=None):
    tr, te = split_rows(df, which); cols = FS[fs]
    rtr = (tr.yc - tr.w_ship * tr.rr_c).clip(-300, 300).values
    if kind == "ridge":
        Xtr, Xte = tr[cols].fillna(0).values, te[cols].fillna(0).values; mu, sd = std_fit(Xtr)
        return Ridge(alpha=a).fit((Xtr - mu) / sd, rtr).predict((Xte - mu) / sd)
    m = lgb.LGBMRegressor(objective="l1", n_estimators=n_est, learning_rate=0.05, num_leaves=leaves, min_child_samples=150, subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=30.0, verbose=-1, random_state=0)
    return m.fit(tr[cols].values, rtr).predict(te[cols].values)


results_all = []
for uname, uf in UNIV.items():
    u_ = uf(P); d = u_.reset_index(drop=True); d["index_orig"] = u_.index.values
    # ---------------- direction ----------------
    fam = Family(f"eps_dir_{uname}", higher_better=True); scorers = {}
    base = lambda df, w: ((df[df.split == w].rr_c > 0).astype(int) == df[df.split == w].up).astype(float).values
    def add(eid, desc, params, sc): scorers[eid] = sc; fam.run(eid, desc, params, sc, base, d)
    for dl in (-10, -5, -2, 2, 5, 10):
        add(f"dir_{uname}_thr{dl}", f"up iff rr>{dl}", dict(delta=dl), lambda df, w, dl=dl: ((df[df.split == w].rr_c > dl).astype(int) == df[df.split == w].up).astype(float).values)
    for vn in VAR:
        add(f"dir_{uname}_{vn}", f"up iff {vn} > prior", dict(variant=vn),
            lambda df, w, vn=vn: ((VAR[vn].loc[df[df.split == w].index_orig.values].values > pr.loc[df[df.split == w].index_orig.values].values).astype(int) == df[df.split == w].up.values).astype(float))
    for fs, C in itertools.product(FS, (0.003, 0.03, 0.3, 3.0)):
        add(f"dir_{uname}_logit_{fs}_C{C}", f"logit {fs} C={C}", dict(model="logit", fs=fs, C=C),
            lambda df, w, fs=fs, C=C: ((logit_pred(df, w, fs, C) >= 0.5).astype(int) == df[df.split == w].up.values).astype(float))
    for fs, lv, ne in itertools.product(("rr", "seq", "all"), (4, 8), (100, 300)):
        add(f"dir_{uname}_gbm_{fs}_l{lv}_n{ne}", f"lgbm {fs} leaves={lv} n={ne}", dict(model="gbm", fs=fs, leaves=lv, n_est=ne),
            lambda df, w, fs=fs, lv=lv, ne=ne: ((gbm_pred(df, w, fs, lv, ne) >= 0.5).astype(int) == df[df.split == w].up.values).astype(float))
    fam.finalize(d, scorers, base, top_k=3)
    # ---------------- growth ----------------
    fam = Family(f"eps_gro_{uname}", higher_better=True); scorers = {}
    base = lambda df, w: -np.abs(df[df.split == w].yc.values - (df[df.split == w].w_ship * df[df.split == w].rr_c).values)
    def add(eid, desc, params, sc): scorers[eid] = sc; fam.run(eid, desc, params, sc, base, d)
    for wg in (0.25, 0.5, 0.75, 1.0):
        add(f"gro_{uname}_w{wg}", f"point = {wg} x rr", dict(w=wg), lambda df, w, wg=wg: -np.abs(df[df.split == w].yc.values - wg * df[df.split == w].rr_c.values))
    for sh in (0.5, 0.75, 1.25):
        add(f"gro_{uname}_ship_x{sh}", f"shipped bucket weights x{sh}", dict(scale=sh), lambda df, w, sh=sh: -np.abs(df[df.split == w].yc.values - sh * (df[df.split == w].w_ship * df[df.split == w].rr_c).values))
    def refit_w(df, w):
        tr, te = split_rows(df, w); wb = {}
        for b, g in tr.groupby("bucket"):
            x, y = g.rr_c.values, g.yc.values; wb[b] = float(np.clip((x * y).sum() / max((x * x).sum(), 1e-9), -0.5, 1.5))
        return -np.abs(te.yc.values - te.bucket.map(wb).fillna(0.5).values * te.rr_c.values)
    add(f"gro_{uname}_refit_bucket_w", "bucket weights refit on train (through-origin L2)", dict(), refit_w)
    for vn in ("V3_median4", "V5_mean4", "V2_mult_seasonal"):
        def sc(df, w, vn=vn):
            te = df[df.split == w]; v = VAR[vn].loc[te.index_orig.values].values; pri = pr.loc[te.index_orig.values].values
            rrv = np.clip((v - pri) / np.maximum(np.abs(pri), 0.01) * 100, -300, 300); return -np.abs(te.yc.values - 0.75 * rrv)
        add(f"gro_{uname}_{vn}", f"point = .75 x run-rate from {vn}", dict(variant=vn), sc)
    for fs, a, sc_ in itertools.product(("seq", "bridge", "all"), (300.0, 3000.0), (0.5, 1.0)):
        add(f"gro_{uname}_ridge_{fs}_a{a}_s{sc_}", f"ridge residual {fs} a={a} scale={sc_}", dict(model="ridge", fs=fs, alpha=a, scale=sc_),
            lambda df, w, fs=fs, a=a, sc_=sc_: -np.abs(df[df.split == w].yc.values - (df[df.split == w].w_ship.values * df[df.split == w].rr_c.values + sc_ * resid_pred(df, w, fs, "ridge", a))))
    for fs, lv, ne, sc_ in itertools.product(("seq", "all"), (4, 8), (100, 300), (0.5, 1.0)):
        add(f"gro_{uname}_gbmres_{fs}_l{lv}_n{ne}_s{sc_}", f"lgbm-L1 residual {fs} leaves={lv} n={ne} scale={sc_}", dict(model="gbm_res", fs=fs, leaves=lv, n_est=ne, scale=sc_),
            lambda df, w, fs=fs, lv=lv, ne=ne, sc_=sc_: -np.abs(df[df.split == w].yc.values - (df[df.split == w].w_ship.values * df[df.split == w].rr_c.values + sc_ * resid_pred(df, w, fs, "gbm", 0, lv, ne))))
    fam.finalize(d, scorers, base, top_k=3)
print("done")
