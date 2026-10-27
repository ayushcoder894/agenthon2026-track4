"""Auction experiments (~55): point rules and interval estimators for bid-to-cover. Score per row = -|y - point| (point experiments) or -IS (interval experiments, point fixed to shipped)."""
import sys, itertools
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
sys.path.insert(0, ".")
from common import Family, interval_score

P = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "panels/auc_panel.parquet").reset_index(drop=True)
H = [f"h{i}" for i in range(1, 14)]
Hm = P[H].values
def kmean(k): return np.nanmean(Hm[:, :k], axis=1)
def ewma(hl):
    w = 0.5 ** (np.arange(13) / hl); m = ~np.isnan(Hm); ww = np.where(m, w[None, :], 0.0); return (np.nan_to_num(Hm) * ww).sum(1) / np.maximum(ww.sum(1), 1e-9)
P["y_"] = P.y
P["pt_ship"] = P.shipped
def rows(df, w): return df[df.split == w]


def split_rows(df, which):
    tr = df[df.split == "train"] if which == "val" else df[df.split.isin(["train", "val"])]
    return tr, df[df.split == which]


fam = Family("auc_point"); scorers = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).pt_ship.values)
def add(eid, desc, params, pt):
    """pt(df, w) -> point vector for rows(df, w)"""
    sc = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); scorers[eid] = sc; fam.run(eid, desc, params, sc, base, P)
col = lambda name: (lambda df, w: rows(df, w)[name].values)
for k in (1, 2, 3, 4, 6, 8, 13):
    add(f"auc_mean{k}", f"mean of last {k}", dict(k=k), lambda df, w, k=k: pd.Series(kmean(k), index=df.index)[(df.split == w)].values)
for hl in (1, 2, 3, 5):
    add(f"auc_ewma{hl}", f"EWMA half-life {hl} auctions", dict(hl=hl), lambda df, w, hl=hl: pd.Series(ewma(hl), index=df.index)[(df.split == w)].values)
for k in (3, 6):
    add(f"auc_median{k}", f"median of last {k}", dict(k=k), lambda df, w, k=k: pd.Series(np.nanmedian(Hm[:, :k], axis=1), index=df.index)[(df.split == w)].values)
for g in (0.0, 0.25, 0.75, 1.0):
    add(f"auc_gap{g}", f"mean6 + {g} x newreopen gap", dict(g=g), lambda df, w, g=g: rows(df, w).m6.values + g * rows(df, w).gap.values)
for b in (0.25, 0.5, 1.0):
    add(f"auc_trend{b}", f"shipped + {b}*(m3-m12)", dict(beta=b), lambda df, w, b=b: rows(df, w).shipped.values + b * (pd.Series(kmean(3), index=df.index)[(df.split == w)].values - rows(df, w).m12.values))
for ph in (0.25, 0.5, 1.0):
    add(f"auc_factor{ph}", f"shipped + {ph}*cross-tenor factor", dict(phi=ph), lambda df, w, ph=ph: rows(df, w).shipped.values + ph * rows(df, w).F.values)
for ka in (0.005, 0.01, 0.02):
    add(f"auc_indirect{ka}", f"shipped + {ka}*indirect-share dev", dict(kappa=ka), lambda df, w, ka=ka: rows(df, w).shipped.values + ka * rows(df, w).ind3.values)
for la in (-0.1, -0.2, -0.4):
    add(f"auc_size{la}", f"shipped + {la}*log size ratio (known only)", dict(lam=la), lambda df, w, la=la: rows(df, w).shipped.values + la * rows(df, w).szr.values)
FE = ["m6", "last", "m12", "sd", "own", "F", "lm6", "gap", "ind3", "szr", "known", "tix", "new", "n"]
def resid_pt(df, w, kind, a=None, lv=None, ne=None, fe=FE):
    tr, te = split_rows(df, w); rt = (tr.y - tr.shipped).values
    if kind == "ridge":
        Xtr, Xte = tr[fe].fillna(0).values, te[fe].fillna(0).values; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        return te.shipped.values + Ridge(alpha=a).fit((Xtr - mu) / sd, rt).predict((Xte - mu) / sd)
    m = lgb.LGBMRegressor(objective="l1", n_estimators=ne, learning_rate=0.05, num_leaves=lv, min_child_samples=30, subsample=0.8, subsample_freq=1, reg_lambda=10.0, verbose=-1, random_state=0)
    return te.shipped.values + m.fit(tr[fe].values, rt).predict(te[fe].values)
for a in (10.0, 100.0, 1000.0):
    add(f"auc_ridge{a}", f"pooled ridge residual a={a}", dict(model="ridge", alpha=a), lambda df, w, a=a: resid_pt(df, w, "ridge", a=a))
for lv, ne in itertools.product((4, 8), (100, 300)):
    add(f"auc_gbmres_l{lv}_n{ne}", f"lgbm-L1 residual leaves={lv} n={ne}", dict(model="gbm", leaves=lv, n_est=ne), lambda df, w, lv=lv, ne=ne: resid_pt(df, w, "gbm", lv=lv, ne=ne))
for fe_name, fe in (("own_only", ["m6", "last", "m12", "own", "lm6"]), ("no_size", [c for c in FE if c not in ("szr", "known")])):
    add(f"auc_ridge100_{fe_name}", f"ridge a=100 features {fe_name}", dict(model="ridge", alpha=100.0, fe=fe_name), lambda df, w, fe=fe: resid_pt(df, w, "ridge", a=100.0, fe=fe))
fam.finalize(P, scorers, base, top_k=3)

# ------------- interval experiments (point fixed to shipped) -------------
fam2 = Family("auc_interval"); sc2 = {}
base2 = lambda df, w: -interval_score(rows(df, w).y.values, rows(df, w).shipped.values - rows(df, w).shipped_half.values, rows(df, w).shipped.values + rows(df, w).shipped_half.values)
def add2(eid, desc, params, band):
    def sc(df, w):
        lo, hi = band(df, w); return -interval_score(rows(df, w).y.values, lo, hi)
    sc2[eid] = sc; fam2.run(eid, desc, params, sc, base2, P)
for z, sc_ in itertools.product((1.0, 1.3, 1.645, 2.0), ("sd", "mad")):
    def band(df, w, z=z, sc_=sc_):
        te = rows(df, w)
        if sc_ == "sd": h = np.maximum(z * te.sd.values, 0.08)
        else:
            Hh = pd.DataFrame(Hm, index=df.index).loc[te.index].values; mad = 1.4826 * np.nanmedian(np.abs(Hh - np.nanmedian(Hh, axis=1, keepdims=True)), axis=1); h = np.maximum(z * mad, 0.08)
        return te.shipped.values - h, te.shipped.values + h
    add2(f"aucI_z{z}_{sc_}", f"band {z} x {sc_}", dict(z=z, scale=sc_), band)
def qres(df, w, group):
    tr, te = split_rows(df, w); r = tr.y - tr.shipped
    if not group: a, b = r.quantile(0.05), r.quantile(0.95); return te.shipped.values + a, te.shipped.values + b
    q = {k: (g_.y - g_.shipped).quantile([0.05, 0.95]).values for k, g_ in tr.groupby("ten") if len(g_) >= 30}; pool = r.quantile([0.05, 0.95]).values
    lo = np.array([te.shipped.values[i] + q.get(t, pool)[0] for i, t in enumerate(te.ten.values)]); hi = np.array([te.shipped.values[i] + q.get(t, pool)[1] for i, t in enumerate(te.ten.values)]); return lo, hi
add2("aucI_resq_pooled", "pooled residual 5/95 quantiles", dict(), lambda df, w: qres(df, w, False))
add2("aucI_resq_tenor", "tenor-grouped residual quantiles", dict(), lambda df, w: qres(df, w, True))
def lgbq(df, w, conformal):
    tr, te = split_rows(df, w); fe = FE; out = []
    if conformal:
        cut = tr.date.max() - pd.Timedelta(days=365); fit_, cal = tr[tr.date <= cut], tr[tr.date > cut]
    else: fit_, cal = tr, None
    ms = [lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=30, subsample=0.7, subsample_freq=1, reg_lambda=20.0, verbose=-1, random_state=0).fit(fit_[fe].values, (fit_.y - fit_.shipped).values) for a in (0.05, 0.95)]
    lo, hi = ms[0].predict(te[fe].values), ms[1].predict(te[fe].values); lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    e = 0.0
    if cal is not None and len(cal) > 20:
        l2, h2 = ms[0].predict(cal[fe].values), ms[1].predict(cal[fe].values); s = np.maximum(np.minimum(l2, h2) - (cal.y - cal.shipped).values, (cal.y - cal.shipped).values - np.maximum(l2, h2)); e = float(np.sort(s)[min(int(np.ceil((len(s) + 1) * 0.9)) - 1, len(s) - 1)])
    return te.shipped.values + lo - e, te.shipped.values + hi + e
add2("aucI_lgbq", "lgbm quantile 5/95", dict(model="lgbq"), lambda df, w: lgbq(df, w, False))
add2("aucI_lgbq_cqr", "lgbm quantile + conformal offset", dict(model="lgbq_cqr"), lambda df, w: lgbq(df, w, True))
fam2.finalize(P, sc2, base2, top_k=3)
print("done")
