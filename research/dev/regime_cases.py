"""Build the case table for D:/AGENTHON/regime_gate_audit.py (columns: family, cutoff_date, label_available_date, regime, loss_d16r, loss_alt).
Every expert forecast is walk-forward: parameters are fitted only on cases whose outcome was released before the cutoff. Regimes use cutoff-available data only.
family string = '<domain>|alt=<expert>|regime=<definition>' so each (alternative, regime definition) is audited on its own."""
import csv, datetime as dt, json, os
import numpy as np, pandas as pd
import lightgbm as lgb

OUT = "t4-local/dev/regime_cases.csv"
rows = []
def add(fam, cutoff, released, regime, la, lb):
    rows.append((fam, pd.Timestamp(cutoff).date().isoformat(), pd.Timestamp(released).date().isoformat(), regime, f"{la:.8f}", f"{lb:.8f}"))
def load(s):
    d = json.load(open(f"data_fred/{s}.json"))["observations"]
    return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})

# =============================================================== CPI
exec(open("t4-local/dev/cpi_study.py").read().split('print("=== per-component')[0])           # M (monthly MoM per component + pump), COMP
fitj = json.load(open("t4-local/dev/cpi_pump_fit.json"))
names = list(COMP)
def rule(c): s = M[c]; return 0.5 * s.shift(1) + 0.5 * s.shift(1).rolling(6).mean()
def feats(c, s):
    return pd.DataFrame({"y": s, "last": s.shift(1), "m3": s.shift(1).rolling(3).mean(), "m6": s.shift(1).rolling(6).mean(), "m12": s.shift(1).rolling(12).mean(),
                         "sd9": s.shift(1).rolling(9).std(ddof=0), "pump": M.pump, "month": M.month, "comp": names.index(c)})
panel = pd.concat([feats(c, M[c]).assign(c=c) for c in names]).dropna(); panel = panel[panel.index.year >= 2000]
months = sorted(panel.index.unique()); test_months = [m for m in months if m >= pd.Timestamp("2004-01-01")]
head = M.CPI_ALLITEMS
def ols_pump(c, upto):                          # expanding fit of the pump pass-through model on months before `upto`
    spec = fitj[c]["feats"]; d = pd.DataFrame({"y": M[c], "pump": M.pump, "m6": M[c].shift(1).rolling(6).mean(), "core6": M.CPI_CORE.shift(1).rolling(6).mean(), "food6": M.CPI_FOOD.shift(1).rolling(6).mean(), "month": M.month})
    d = d[(d.index >= "2000-01-01") & (d.index < upto)].dropna()
    X = np.column_stack([d[f] for f in spec] + [np.ones(len(d))] + [(d.month == k).astype(float) for k in range(2, 13)])
    b = np.linalg.lstsq(X, d.y.values, rcond=None)[0]; return spec, b
preds = {k: {} for k in ("A", "naive", "m12", "lgbm")}
for yr in sorted({m.year for m in test_months}):
    upto = pd.Timestamp(f"{yr}-01-01")
    fits = {c: ols_pump(c, upto) for c in ("CPI_GASOLINE", "CPI_ENERGY")}
    tr = panel[panel.index < upto]; te_idx = [m for m in test_months if m.year == yr]
    F = ["last", "m3", "m6", "sd9", "pump", "month", "comp"]
    gb = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3, verbose=-1).fit(tr[F], tr.y)
    for m in te_idx:
        sl = panel[panel.index == m]
        pa = {}
        for c in names:
            r = sl[sl.c == c]
            if r.empty: break
            pa[c] = float(0.5 * r["last"].iloc[0] + 0.5 * r.m6.iloc[0])
        if len(pa) < len(names): continue
        for c in ("CPI_GASOLINE", "CPI_ENERGY"):
            spec, b = fits[c]; r = sl[sl.c == c].iloc[0]
            f = {"pump": r.pump, "m6": r.m6, "core6": M.CPI_CORE.shift(1).rolling(6).mean().loc[m], "food6": M.CPI_FOOD.shift(1).rolling(6).mean().loc[m]}
            pa[c] = float(sum(b[i] * f[s_] for i, s_ in enumerate(spec)) + b[len(spec)] + ([0.0] + list(b[len(spec) + 1:]))[int(r.month) - 1])
        pa["CPI_ALLITEMS"] = 0.135 * pa["CPI_FOOD"] + 0.07 * pa["CPI_ENERGY"] + 0.795 * pa["CPI_CORE"]
        truth = {c: float(sl[sl.c == c].y.iloc[0]) for c in names}
        eA = np.mean([abs(truth[c] - pa[c]) for c in names])
        eN = np.mean([abs(truth[c] - float(sl[sl.c == c]["last"].iloc[0])) for c in names])
        eM = np.mean([abs(truth[c] - float(sl[sl.c == c].m12.iloc[0])) for c in names])
        pg = gb.predict(sl[F]); eG = np.mean(np.abs(sl.y.values - pg))
        preds["A"][m], preds["naive"][m], preds["m12"][m], preds["lgbm"][m] = eA, eN, eM, eG
# regimes (cutoff = last day of month m; data through month m-1 only)
hd = head.copy(); infl = hd.shift(1).rolling(12).mean(); gas = M.CPI_GASOLINE; gsd = gas.shift(1).rolling(6).std(ddof=0); gsd_med = gsd.expanding().median().shift(1)
shock = (hd.shift(1) - hd.shift(1).rolling(12).mean()).abs(); shock_sd = hd.shift(1).expanding().std().shift(1)
for m in test_months:
    if m not in preds["A"]: continue
    cutoff = (m + pd.offsets.MonthEnd(0)); released = (m + pd.offsets.MonthBegin(1)) + pd.Timedelta(days=12)
    regs = {"infl": "high_infl" if infl.loc[m] > 0.30 else "normal", "energyvol": "vol_high" if gsd.loc[m] > gsd_med.loc[m] else "vol_low",
            "shock": "shock" if shock.loc[m] > shock_sd.loc[m] else "calm"}
    for alt in ("naive", "m12", "lgbm"):
        for rn, rv in regs.items(): add(f"cpi|alt={alt}|regime={rn}", cutoff, released, rv, preds["A"][m], preds[alt][m])

# =============================================================== FOMC / rates (non-overlapping 35-trading-day origins)
MATS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MATS + ["DGS6MO"]}).dropna(); Y = Y[Y.index >= "1990-01-01"]
tgt = load("DFEDTAR").dropna(); mid = ((load("DFEDTARU") + load("DFEDTARL")) / 2).dropna()
pol = pd.concat([tgt[tgt.index < mid.index.min()], mid]).sort_index().reindex(Y.index, method="ffill"); dY = Y.diff() * 100
H = 35; orig = list(range(80, len(Y) - H, H)); rec = []
for i in orig:
    t, r = Y.index[i], Y.iloc[i]
    y = np.array([(Y[m].iloc[i + H] - r[m]) * 100 for m in MATS])
    x = dict(b6=r.DGS6MO - pol.iloc[i], s26=r.DGS2 - r.DGS6MO, pol3=pol.iloc[i] - pol.iloc[i - 63], mom60=np.mean([(r[m] - Y[m].iloc[i - 60]) * 100 for m in MATS]))
    sd10 = np.mean([dY[m].iloc[i - 9:i + 1].std() for m in MATS])
    rec.append(dict(i=i, t=t, rel=Y.index[i + H], y=y, x=x, mid=pol.iloc[i], inv=r.DGS10 - r.DGS2, sd10=sd10, pol3=x["pol3"]))
for ex in ("b6", "s26", "pol3", "mom60"):
    for k, r in enumerate(rec):
        past = [q for q in rec[:k] if q["rel"] < r["t"]]
        if len(past) < 20: r.setdefault("alt", {})[ex] = None; continue
        num = sum(q["x"][ex] * q["y"].sum() for q in past); den = sum(len(MATS) * q["x"][ex] ** 2 for q in past); beta = num / den if den else 0.0
        r.setdefault("alt", {})[ex] = beta * r["x"][ex]
sd_hist = []
for k, r in enumerate(rec):
    med = np.median([q["sd10"] for q in rec[:k]]) if k > 5 else np.inf
    regs = {"policylevel": "high_rate" if r["mid"] > 2.0 else "low_rate", "curve": "inverted" if r["inv"] < 0 else "normal",
            "cycle": "hiking" if r["pol3"] > 0.1 else ("easing" if r["pol3"] < -0.1 else "flat"), "vol": "vol_high" if r["sd10"] > med else "vol_low"}
    la = float(np.mean(np.abs(r["y"])))
    for ex in ("b6", "s26", "pol3", "mom60"):
        p = r["alt"][ex]
        if p is None: continue
        lb = float(np.mean(np.abs(r["y"] - p)))
        for rn, rv in regs.items(): add(f"rates|alt={ex}|regime={rn}", r["t"], r["rel"], rv, la, lb)

# =============================================================== post-earnings label (0/1 loss)
ev = pd.read_parquet("t4-local/dev/postearn_events.parquet"); ev["day"] = pd.to_datetime(ev.day); ev = ev[(ev.day >= "2011-01-01") & (ev.day <= "2022-06-28")]
px = {}
for t in ev.ticker.unique():
    f = f"data_px/{t}.csv"
    if os.path.exists(f):
        d = pd.read_csv(f); d["day"] = pd.to_datetime(d.timestamp, unit="s").dt.normalize(); px[t] = d.drop_duplicates("day").set_index("day").sort_index().close
R = pd.DataFrame(px).pct_change(fill_method=None) * 100; mk = R.mean(axis=1).dropna()
mk60 = mk.rolling(60).sum().shift(1); mkvol = mk.rolling(20).std().shift(1); mkvol_med = mkvol.expanding().median().shift(1)
for r in ev.sort_values("day").itertuples():
    d = r.day
    if d not in mk60.index or np.isnan(mk60.loc[d]) or np.isnan(mkvol_med.loc[d]): continue
    regs = {"trend": "mkt_up" if mk60.loc[d] > 0 else "mkt_down", "vol": "vol_high" if mkvol.loc[d] > mkvol_med.loc[d] else "vol_low"}
    a = 1.0 if not (r.abn > 1) else 0.0
    for alt, lb in (("negative", 0.0 if r.abn < -1 else 1.0), ("flat", 0.0 if abs(r.abn) <= 1 else 1.0)):
        for rn, rv in regs.items(): add(f"postearn|alt={alt}|regime={rn}", d - pd.Timedelta(days=1), d + pd.Timedelta(days=1), rv, a, lb)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["family", "cutoff_date", "label_available_date", "regime", "loss_d16r", "loss_alt"]); w.writerows(rows)
print("rows", len(rows), "families", len({r[0] for r in rows}))
