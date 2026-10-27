"""Macro-revision direction (~40) and FOMC rate-curve (~35) experiments. Data: local ALFRED vintages (<=2022-06-28), FRED yields / SEP medians / FOMC statements."""
import json, sys, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Family
ROOT = os.environ.get("EXP500_DATA", "D:/AGENTHON")

# ================= macro revisions =================
ev = []
for s in ("DGORDER", "HOUST", "INDPRO", "PAYEMS", "PI", "RSAFS"):
    d = pd.DataFrame(json.load(open(f"{ROOT}/data_alfred/{s}.json"))); d["v"] = pd.to_numeric(d.value, errors="coerce"); d = d.dropna(subset=["v"]).sort_values(["date", "realtime_start"])
    for dt, g in d.groupby("date"):
        v, rs = g.v.values, g.realtime_start.values
        for i in range(1, len(v)):
            if v[i] != v[i - 1]: ev.append(dict(series=s, ref=dt, when=pd.Timestamp(rs[i]), up=int(v[i] > v[i - 1]), k=i, rel=(v[i] - v[i - 1]) / abs(v[i - 1])))
E = pd.DataFrame(ev).sort_values("when").reset_index(drop=True)
# lag features from STRICTLY EARLIER release dates only (events in the same release are not known beforehand)
agg = E.groupby(["series", "when"]).agg(up_m=("up", "mean"), rel_m=("rel", "mean")).reset_index().sort_values(["series", "when"])
agg["lag_dir"] = agg.groupby("series").up_m.shift(1); agg["prev_rel"] = agg.groupby("series").rel_m.shift(1)
E = E.merge(agg[["series", "when", "lag_dir", "prev_rel"]], on=["series", "when"], how="left")
E["lag_dir"] = (E.lag_dir >= 0.5).astype(float).where(E.lag_dir.notna())
E["same_ref_prev"] = E.groupby(["series", "ref"]).up.shift(1)            # direction of the previous revision of the SAME reference month
E["mon"] = pd.to_datetime(E.ref).dt.month
E["split"] = np.where(E.when <= "2015-12-31", "train", np.where(E.when <= "2019-12-31", "val", "final")); E["block"] = E.when.dt.year * 10 + (E.when.dt.month - 1) // 3 + 1
print("macro events", len(E), E.split.value_counts().to_dict())
def rows(df, w): return df[df.split == w]
def sp(df, w):
    tr = df[df.split == "train"] if w == "val" else df[df.split.isin(["train", "val"])]; return tr, df[df.split == w]
fam = Family("macro_dir"); scorers = {}
def prior_up(tr): return float(tr.up.mean())
base = lambda df, w: ((np.full(len(rows(df, w)), 1 if prior_up(sp(df, w)[0]) >= 0.5 else 0)) == rows(df, w).up.values).astype(float)    # shipped-like: pooled long-run direction
def add(eid, desc, params, pred):
    sc = lambda df, w: (pred(df, w) == rows(df, w).up.values).astype(float); scorers[eid] = sc; fam.run(eid, desc, params, sc, base, E)
add("mac_always_up", "always up", dict(), lambda df, w: np.ones(len(rows(df, w)), int)); add("mac_always_down", "always down", dict(), lambda df, w: np.zeros(len(rows(df, w)), int))
def series_share(df, w, last_n=None, k_aware=False, w_pool=None):
    tr, te = sp(df, w); pool = prior_up(tr); out = np.zeros(len(te), int)
    for i, (s, k) in enumerate(zip(te.series.values, te.k.values)):
        t = tr[tr.series == s]
        if k_aware: t = t[t.k.clip(upper=3) == min(k, 3)]
        if last_n: t = t.tail(last_n)
        p = (t.up.sum() + (w_pool or 0) * pool) / (len(t) + (w_pool or 0)) if len(t) else pool
        out[i] = int(p >= 0.5)
    return out
for ln in (None, 24, 60, 120): add(f"mac_series_last{ln}", f"series up-share, last {ln}", dict(last_n=ln), lambda df, w, ln=ln: series_share(df, w, ln))
for kaw in (False, True):
    for wp in (0, 20, 100): add(f"mac_series_k{int(kaw)}_w{wp}", f"series share k_aware={kaw} prior weight {wp}", dict(k_aware=kaw, w=wp), lambda df, w, kaw=kaw, wp=wp: series_share(df, w, None, kaw, wp))
def persist(df, w, col): te = rows(df, w); return te[col].fillna(1).astype(int).values
add("mac_persist_series_last", "same direction as the series' previous revision event", dict(), lambda df, w: persist(df, w, "lag_dir"))
add("mac_persist_same_ref", "same direction as previous revision of the same reference month", dict(), lambda df, w: persist(df, w, "same_ref_prev"))
add("mac_reverse_same_ref", "OPPOSITE of the previous revision of the same month", dict(), lambda df, w: 1 - persist(df, w, "same_ref_prev"))
add("mac_reverse_series_last", "opposite of the series' previous event", dict(), lambda df, w: 1 - persist(df, w, "lag_dir"))
def month_share(df, w, sh):
    tr, te = sp(df, w); pool = prior_up(tr); m = tr.groupby(["series", "mon"]).up.agg(["sum", "count"]); out = []
    for s, mo in zip(te.series, te.mon):
        r = m.loc[(s, mo)] if (s, mo) in m.index else None; p = (r["sum"] + sh * pool) / (r["count"] + sh) if r is not None else pool; out.append(int(p >= 0.5))
    return np.array(out)
for sh in (5, 20, 60): add(f"mac_seas_series_{sh}", f"series x ref-month up-share, prior weight {sh}", dict(shrink=sh), lambda df, w, sh=sh: month_share(df, w, sh))
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
E["si"] = E.series.astype("category").cat.codes
for C, fs in itertools.product((0.01, 0.1, 1.0), ("series+k", "series+k+prev")):
    def lg(df, w, C=C, fs=fs):
        tr, te = sp(df, w); cols = ["k"] + (["lag_dir", "same_ref_prev", "prev_rel"] if "prev" in fs else [])
        X = lambda d: np.column_stack([pd.get_dummies(d.series).reindex(columns=sorted(E.series.unique()), fill_value=0).values.astype(float), d[cols].fillna(0).values])
        return LogisticRegression(C=C, max_iter=300).fit(X(tr), tr.up.values).predict(X(te))
    add(f"mac_logit_{fs}_C{C}", f"logit {fs} C={C}", dict(C=C, fs=fs), lg)
for lv, ne in itertools.product((4, 8), (100, 300)):
    def gb(df, w, lv=lv, ne=ne):
        tr, te = sp(df, w); cols = ["si", "k", "lag_dir", "same_ref_prev", "prev_rel", "mon"]
        m = lgb.LGBMClassifier(n_estimators=ne, learning_rate=0.05, num_leaves=lv, min_child_samples=100, subsample=0.7, subsample_freq=1, reg_lambda=20.0, verbose=-1, random_state=0)
        return m.fit(tr[cols].values, tr.up.values).predict(te[cols].values)
    add(f"mac_gbm_l{lv}_n{ne}", f"lgbm leaves={lv} n={ne}", dict(leaves=lv, n_est=ne), gb)
fam.finalize(E, scorers, base, top_k=3)

# ================= FOMC rate curve =================
def load(s):
    d = json.load(open(f"{ROOT}/data_fred/{s}.json"))["observations"]; return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})
MATS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MATS + ["DGS1", "DGS3MO"]}).dropna(); Y = Y[Y.index >= "1995-01-01"]
tgt = load("DFEDTAR").dropna(); mid = ((load("DFEDTARU") + load("DFEDTARL")) / 2).dropna(); pol = pd.concat([tgt[tgt.index < mid.index.min()], mid]).sort_index().reindex(Y.index, method="ffill")
dates = sorted(pd.Timestamp(x["date"]) for x in map(json.loads, open(f"{ROOT}/data_fomc/statements.jsonl", encoding="utf-8")))
R = []
for i in range(len(dates) - 1):
    d0, d1 = dates[i], dates[i + 1]; p0 = Y.index.searchsorted(d0, side="right"); p1 = Y.index.searchsorted(d1, side="left") - 1
    if p1 >= len(Y) or p1 <= p0 + 3 or p0 < 70: continue
    r = dict(d0=d0, pos0=p0, hd=p1 - p0, y1=Y.DGS1.iloc[p0], pmove=(pol.iloc[p0] - pol.iloc[p0 - 2]) * 100, slope=(Y.DGS10.iloc[p0] - Y.DGS2.iloc[p0]) * 100, gap=(Y.DGS1.iloc[p0] - pol.iloc[p0]) * 100)
    for h in (5, 10, 30, 60): r[f"m{h}"] = (Y.DGS2.iloc[p0] - Y.DGS2.iloc[p0 - h]) * 100
    for m in MATS: r["d_" + m] = (Y[m].iloc[p1] - Y[m].iloc[p0]) * 100
    R.append(r)
Rt = pd.DataFrame(R); Rt["z"] = Rt[["d_" + m for m in MATS]].mean(axis=1)
Rt["split"] = np.where(Rt.d0 <= "2012-12-31", "train", np.where(Rt.d0 <= "2018-12-31", "val", "final")); Rt["block"] = Rt.d0.dt.year * 10 + (Rt.d0.dt.month - 1) // 3 + 1
print("rate meetings", len(Rt), Rt.split.value_counts().to_dict())
fam = Family("fomc_curve"); scorers = {}
base = lambda df, w: -np.abs(df[df.split == w].z.values)                         # shipped: zero change
def add(eid, desc, params, pred):
    sc = lambda df, w: -np.abs(df[df.split == w].z.values - pred(df, w)); scorers[eid] = sc; fam.run(eid, desc, params, sc, base, Rt)
def spf(df, w):
    tr = df[df.split == "train"] if w == "val" else df[df.split.isin(["train", "val"])]; return tr, df[df.split == w]
def slope_fit(df, w, col, al):
    tr, te = spf(df, w); x, y = tr[col].fillna(0).values, tr.z.values; b = (x * y).sum() / max((x * x).sum(), 1e-9); return al * b * te[col].fillna(0).values
for col, al in itertools.product(("m5", "m10", "m30", "m60", "pmove", "slope", "gap"), (0.5, 1.0)):
    add(f"fomc_{col}_a{al}", f"through-origin on {col}, scale {al}", dict(col=col, alpha=al), lambda df, w, col=col, al=al: slope_fit(df, w, col, al))
from sklearn.linear_model import Ridge
for fs, a in itertools.product((["m10", "m30", "pmove", "slope", "gap"], ["m30", "m60"], ["pmove", "gap"], ["m5", "m10", "m30", "m60", "pmove", "slope", "gap"]), (30.0, 300.0)):
    def rg(df, w, fs=fs, a=a):
        tr, te = spf(df, w); Xtr, Xte = tr[fs].fillna(0).values, te[fs].fillna(0).values; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        return Ridge(alpha=a).fit((Xtr - mu) / sd, tr.z.clip(-100, 100).values).predict((Xte - mu) / sd) * 0.5
    add(f"fomc_ridge_{len(fs)}f_a{a}", f"ridge {fs} a={a} x0.5", dict(fs=fs, alpha=a), rg)
for c in (-10, -5, 5, 10): add(f"fomc_const{c}", f"constant drift {c}bp (diagnostic)", dict(c=c), lambda df, w, c=c: np.full(len(df[df.split == w]), float(c)))
def sign_follow(df, w, k):
    te = df[df.split == w]; return k * np.sign(te.pmove.values)
for k in (2.0, 5.0, 10.0): add(f"fomc_follow_pmove{k}", f"drift {k}bp in the direction of the policy move", dict(k=k), lambda df, w, k=k: sign_follow(df, w, k))
fam.finalize(Rt, scorers, base, top_k=3)
print("done")
