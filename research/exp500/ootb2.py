"""Round 2 of new ideas (general rules; chronological val + one-shot final + Holm), then in-sample effect on the Dev units is reported separately.
AUCTIONS: event proximity (FOMC statement within d days, US general election within 3 days), Thanksgiving / Christmas week, month-of-year seasonality,
          same-week coupon supply, days since the tenor's previous auction; ridge / LightGBM residual on the shipped point.
CPI:      fuzzy (piecewise) persistence by standardised size of the last print, turning-point rule after a run, run-length features, pooled LightGBM."""
import os, sys, re, json, itertools, datetime as dt
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import Family
ROOT = "D:/AGENTHON"

# ------------------------------------------------------------------ calendars
fomc = sorted({pd.Timestamp(json.loads(l)["date"]) for l in open(f"{ROOT}/data_fomc/statements.jsonl", encoding="utf-8")} |
              {pd.Timestamp(f"{m[:4]}-{m[4:6]}-{m[6:]}") for m in re.findall(r"monetary(\d{8})a\.htm", open(f"{ROOT}/data_fomc/cal.htm", encoding="utf-8", errors="ignore").read())})
fomc = pd.DatetimeIndex(fomc)
ELECTIONS = pd.to_datetime(["2008-11-04", "2010-11-02", "2012-11-06", "2014-11-04", "2016-11-08", "2018-11-06", "2020-11-03", "2022-11-08", "2024-11-05"])
def thanksgiving(y):
    d = dt.date(y, 11, 1); first_thu = d + dt.timedelta(days=(3 - d.weekday()) % 7); return pd.Timestamp(first_thu + dt.timedelta(days=21))
def afeat(dates):
    out = []
    for d in pd.to_datetime(dates):
        dd = np.abs((fomc - d).days); fd = dd.min() if len(dd) else 99
        ed = np.abs((ELECTIONS - d).days).min()
        tg = thanksgiving(d.year); xmas = pd.Timestamp(d.year, 12, 25)
        out.append(dict(fomc_near=float(fd <= 2), fomc_days=float(min(fd, 30)), elect_near=float(ed <= 3), tg_week=float(abs((d - tg).days) <= 4), xmas_week=float(abs((d - xmas).days) <= 5),
                        nov=float(d.month == 11), dec=float(d.month == 12), month=float(d.month)))
    return pd.DataFrame(out)

def rows(df, w): return df[df.split == w]
def sp(df, w):
    tr = df[df.split == "train"] if w == "val" else df[df.split.isin(["train", "val"])]; return tr, df[df.split == w]

# ================================================================== AUCTIONS
A = pd.read_parquet(os.path.join(HERE, "panels/auc_panel.parquet")).reset_index(drop=True)
A = pd.concat([A, afeat(A.date)], axis=1)
wk = A.date.dt.to_period("W"); A["week_supply"] = A.groupby(wk).ten.transform("count").astype(float)
A["gap_days"] = A.groupby("ten").date.diff().dt.days.fillna(30).clip(0, 120)
print("auction rows", len(A), "| fomc_near share", round(A.fomc_near.mean(), 3), "| elect_near", int(A.elect_near.sum()), "| tg_week", int(A.tg_week.sum()))
fam = Family("auc_ootb2"); sc = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).shipped.values)
def add(eid, desc, params, pt):
    f = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); sc[eid] = f; fam.run(eid, desc, params, f, base, A)
def ridge_pt(df, w, cols, a, al):
    tr, te = sp(df, w); X = tr[cols].values; mu, sd = X.mean(0), X.std(0) + 1e-9
    return te.shipped.values + al * Ridge(alpha=a).fit((X - mu) / sd, (tr.y - tr.shipped).values).predict((te[cols].values - mu) / sd)
def month_pt(df, w, shrink):
    tr, te = sp(df, w); r = tr.y - tr.shipped; g = r.groupby(tr.month).agg(["sum", "count"]); eff = g["sum"] / (g["count"] + shrink)
    return te.shipped.values + te.month.map(eff).fillna(0).values
EV = {"fomc": ["fomc_near"], "fomc_days": ["fomc_days"], "election": ["elect_near"], "holiday": ["tg_week", "xmas_week"], "supply": ["week_supply"], "gapdays": ["gap_days"],
      "nov_dec": ["nov", "dec"], "events_all": ["fomc_near", "fomc_days", "elect_near", "tg_week", "xmas_week", "week_supply", "gap_days", "nov", "dec"]}
for nm, a, al in itertools.product(EV, (10.0, 100.0), (0.5, 1.0)):
    add(f"auc2_{nm}_a{a}_s{al}", f"shipped + ridge on {EV[nm]} a={a} x{al}", dict(feats=EV[nm], alpha=a, scale=al), lambda df, w, nm=nm, a=a, al=al: ridge_pt(df, w, EV[nm], a, al))
for sh in (10, 30, 100):
    add(f"auc2_month_s{sh}", f"shipped + month-of-year residual effect (shrink {sh})", dict(shrink=sh), lambda df, w, sh=sh: month_pt(df, w, sh))
FEALL = ["m6", "last", "m12", "sd", "own", "F", "lm6", "gap", "ind3", "szr", "known", "tix", "new", "n"] + EV["events_all"] + ["month"]
def gbm_pt(df, w, lv, ne):
    tr, te = sp(df, w)
    m = lgb.LGBMRegressor(objective="l1", n_estimators=ne, learning_rate=0.03, num_leaves=lv, min_child_samples=30, subsample=0.8, subsample_freq=1, reg_lambda=10.0, verbose=-1, random_state=0)
    return te.shipped.values + m.fit(tr[FEALL].values, (tr.y - tr.shipped).values).predict(te[FEALL].values)
for lv, ne in itertools.product((4, 8), (100, 300)):
    add(f"auc2_gbm_all_l{lv}_n{ne}", f"lgbm-L1 residual, all + event features, leaves={lv} n={ne}", dict(leaves=lv, n_est=ne), lambda df, w, lv=lv, ne=ne: gbm_pt(df, w, lv, ne))
fam.finalize(A, sc, base, top_k=3)
# descriptive: historical mean residual of the shipped rule by event flag (all years)
for c in ("fomc_near", "elect_near", "tg_week", "xmas_week", "nov"):
    r = (A.y - A.shipped); print(f"  mean residual vs shipped | {c}=1: {r[A[c] == 1].mean():+.4f} (n={int((A[c] == 1).sum())})  {c}=0: {r[A[c] == 0].mean():+.4f}")

# ================================================================== CPI fuzzy rules
sys.argv = [sys.argv[0]]; __file__ = os.path.join(HERE, "cpi_exps.py")
exec(open(os.path.join(HERE, "cpi_exps.py"), encoding="utf-8").read().split("fam = Family(")[0])      # -> L, rows, sp
L = L[L.month >= "1985-01-01"].reset_index(drop=True)
lg = L[[f"l{i}" for i in range(1, 13)]].values
sdc = L[L.split == "train"].groupby("comp").y.std(); L["z"] = L["last"] / L.comp.map(sdc)
run = np.zeros(len(L))
for i, r in enumerate(lg):
    s = np.sign(r[1]); k = 0
    for v in r[1:]:
        if np.sign(v) == s and s != 0: k += 1
        else: break
    run[i] = k * s                                               # signed length of the run BEFORE the last print
L["run_prev"] = run; L["turn"] = ((np.sign(L["last"]) != np.sign(L.l2)) & (np.abs(L.run_prev) >= 3)).astype(float); L["prev3"] = lg[:, 1:4].mean(1)
fam = Family("cpi_ootb2"); sc = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).ship.values)
def add(eid, desc, params, pt):
    f = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); sc[eid] = f; fam.run(eid, desc, params, f, base, L)
EDG = {"z5": [-np.inf, -2, -1, 1, 2, np.inf], "z3": [-np.inf, -1.5, 1.5, np.inf], "z7": [-np.inf, -3, -2, -1, 1, 2, 3, np.inf]}
def piece_pt(df, w, edges, center, shrink):
    tr, te = sp(df, w)
    def bins(d): return pd.cut(d.z, bins=edges, labels=False)
    c_tr = tr.m12 if center == "m12" else tr.m6; c_te = te.m12 if center == "m12" else te.m6
    x, y = tr["last"] - c_tr, tr.y - c_tr; pooled = (x * y).sum() / (x * x).sum(); phi = {}
    for b, idx in tr.groupby(bins(tr)).groups.items():
        xb, yb = x.loc[idx], y.loc[idx]; n = len(xb); p = (xb * yb).sum() / max((xb * xb).sum(), 1e-9); phi[b] = (n * p + shrink * pooled) / (n + shrink)
    return c_te.values + bins(te).map(phi).fillna(pooled).values * (te["last"] - c_te).values
for e, center, sh in itertools.product(EDG, ("m12", "m6"), (20, 200)):
    add(f"cpi2_piece_{e}_{center}_s{sh}", f"piecewise persistence by z of last ({e}) toward {center}, shrink {sh}", dict(edges=e, center=center, shrink=sh), lambda df, w, e=e, center=center, sh=sh: piece_pt(df, w, EDG[e], center, sh))
def turn_pt(df, w, a):
    te = rows(df, w); return te.ship.values + a * te.turn.values * np.sign(te["last"].values) * np.abs(te.prev3.values)
for a in (0.25, 0.5, 1.0):
    add(f"cpi2_turn_a{a}", f"shipped + {a} x |mean prev3| in the new direction after a >=3 run breaks", dict(a=a), lambda df, w, a=a: turn_pt(df, w, a))
def turn_fit(df, w):
    tr, te = sp(df, w); x = (tr.turn * np.sign(tr["last"]) * np.abs(tr.prev3)).values; r = (tr.y - tr.ship).values; b = (x * r).sum() / max((x * x).sum(), 1e-9)
    return te.ship.values + b * (te.turn * np.sign(te["last"]) * np.abs(te.prev3)).values
add("cpi2_turn_fitted", "shipped + fitted coefficient on the turning-point signal", dict(), turn_fit)
CF = ["last", "m6", "m12", "z", "run_prev", "turn", "prev3", "mon"] + [f"l{i}" for i in range(1, 7)]
def gbm_c(df, w, lv, ne):
    tr, te = sp(df, w); cats = sorted(L.comp.unique()); ci = lambda d: d.comp.map({c: i for i, c in enumerate(cats)}).values
    m = lgb.LGBMRegressor(objective="l1", n_estimators=ne, learning_rate=0.03, num_leaves=lv, min_child_samples=40, subsample=0.8, subsample_freq=1, reg_lambda=10.0, verbose=-1, random_state=0)
    return te.ship.values + m.fit(np.c_[tr[CF].values, ci(tr)], (tr.y - tr.ship).values).predict(np.c_[te[CF].values, ci(te)])
for lv, ne in itertools.product((4, 8), (100, 300)):
    add(f"cpi2_gbm_fuzzy_l{lv}_n{ne}", f"lgbm-L1 residual with z / run / turn features, leaves={lv} n={ne}", dict(leaves=lv, n_est=ne), lambda df, w, lv=lv, ne=ne: gbm_c(df, w, lv, ne))
fam.finalize(L, sc, base, top_k=3)
# descriptive: next-month change conditional on z bucket of the last print (all comps, 1985-2024)
L["zb"] = pd.cut(L.z, bins=EDG["z7"]); d = (L.y - L.m12); x = (L["last"] - L.m12)
print(L.assign(nx=d, lx=x).groupby("zb", observed=True).apply(lambda g: pd.Series(dict(n=len(g), mean_last_dev=g.lx.mean(), mean_next_dev=g.nx.mean(), ratio=g.nx.mean() / g.lx.mean() if g.lx.mean() else np.nan))).round(3).to_string())
t = L[L.turn == 1]; print("turning points:", len(t), " mean next change in new direction:", round(float((np.sign(t["last"]) * (t.y - t.ship)).mean()), 3), " vs shipped residual overall", round(float((L.y - L.ship).mean()), 3))
print("done")
