"""EPS benchmark on SEC XBRL point-in-time data (ZipLime/company-fundamentals, Apache-2.0).

Reproduces the server's EPS-persistence study locally. Data limit 2022-06-28 (earliest public cutoff 2022-07-28
minus 30 days). Quarter = discrete three-month 10-Q period. Origin for predicting quarter Q = end(Q) + 14 days.
Features use only values whose knowledge_date is <= origin. Train outcomes <= 2018-12-31, test 2019-01..2022-03.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

LIMIT = pd.Timestamp("2022-06-28", tz="UTC")
COLS = ["cik", "accession_number", "period_end", "quarters", "period_kind", "form", "knowledge_date", "eps_diluted",
        "revenue", "net_income", "revision"]

frames = [pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))]
df = pd.concat(frames, ignore_index=True)
df = df[(df.knowledge_date <= LIMIT) & (df.period_kind == "quarter") & (df.quarters == 1) & df.eps_diluted.notna()]
df = df[df.form.astype(str).str.startswith("10-Q")]
# first-reported value per company-period (the value known when it first became public)
df = df.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
df["period_end"] = pd.to_datetime(df["period_end"]).astype("datetime64[ns]")
print("quarterly 10-Q observations:", len(df), "companies:", df.cik.nunique())

# join each quarter to the same quarter one year earlier (period_end within +-20 days of a year earlier)
a = df[["cik", "period_end", "knowledge_date", "eps_diluted", "revenue", "net_income"]].copy()
b = a.rename(columns={"period_end": "pe_prior", "knowledge_date": "kd_prior", "eps_diluted": "eps_prior", "revenue": "rev_prior", "net_income": "ni_prior"})
a["key"] = (a.period_end - pd.Timedelta(days=365)).astype("datetime64[ns]")
a = a.sort_values("key"); b = b.sort_values("pe_prior")
m = pd.merge_asof(a, b, left_on="key", right_on="pe_prior", by="cik", tolerance=pd.Timedelta(days=20), direction="nearest")
m = m.dropna(subset=["eps_prior"]).copy()
m["yoy_up"] = (m.eps_diluted > m.eps_prior).astype(int)
m["growth"] = (m.eps_diluted - m.eps_prior) / m.eps_prior.abs().clip(lower=0.01) * 100.0
m = m.sort_values(["cik", "period_end"])
print("quarters with YoY:", len(m))

# pair "last quarter" (feature) with the next quarter (outcome): next quarter ends ~91 days after the last one
nxt = m[["cik", "period_end", "kd_prior", "knowledge_date", "eps_diluted", "eps_prior", "yoy_up", "growth", "revenue"]].rename(
    columns={"period_end": "pe_next", "knowledge_date": "kd_next", "eps_diluted": "eps_next", "eps_prior": "eps_prior_next",
             "yoy_up": "up_next", "growth": "growth_next", "revenue": "rev_next"})
last = m[["cik", "period_end", "knowledge_date", "growth", "yoy_up", "revenue"]].rename(
    columns={"knowledge_date": "kd_last", "growth": "g_last", "yoy_up": "up_last", "revenue": "rev_last"})
last["key"] = (last.period_end + pd.Timedelta(days=91)).astype("datetime64[ns]")
last = last.sort_values("key"); nxt = nxt.sort_values("pe_next")
P = pd.merge_asof(last, nxt, left_on="key", right_on="pe_next", by="cik", tolerance=pd.Timedelta(days=12), direction="nearest").dropna(subset=["pe_next"])
P["origin"] = P.pe_next + pd.Timedelta(days=14)
P = P[(P.kd_last <= P.origin.dt.tz_localize("UTC")) & (P.origin.dt.tz_localize("UTC") <= LIMIT)]
# the outcome must also be knowable before the data limit (it only has to be public after the origin)
P = P[P.kd_next <= LIMIT]
P["year"] = P.pe_next.dt.year
train, test = P[P.pe_next <= "2018-12-31"], P[(P.pe_next >= "2019-01-01")]
print("examples: train", len(train), "test", len(test), "| test base rate up:", round(test.up_next.mean(), 3))


def boot_ci(x, groups, n=600, seed=0):
    rng = np.random.default_rng(seed); g = pd.Series(x.values, index=groups.values)
    keys = g.index.unique(); means = []
    for _ in range(n):
        s = rng.choice(keys, len(keys)); means.append(np.mean(np.concatenate([g.loc[[k]].values for k in s])))
    return np.percentile(means, [5, 95])


def block_ci(frame, col_correct, n=400):
    return boot_ci(frame[col_correct], frame.pe_next.dt.to_period("Q").astype(str), n=n)


bins = [-np.inf, -50, -10, 0, 10, 30, np.inf]
labels = ["<=-50", "-50..-10", "-10..0", "0..10", "10..30", ">30"]
for d in (train, test, P):
    d["bucket"] = pd.cut(d.g_last, bins=bins, labels=labels)
tab = train.groupby("bucket", observed=True).up_next.agg(["mean", "count"]); print("\ntrain P(up | last yoy growth bucket):\n", tab.round(3))
probs = tab["mean"].to_dict()


def evaluate(frame, name):
    out = {}
    out["always_up"] = (frame.up_next == 1).mean()
    out["persist"] = ((frame.g_last > 0).astype(int) == frame.up_next).mean()
    pr = frame.bucket.map(probs).astype(float)
    out["bucket_argmax"] = (((pr >= 0.5).astype(int)) == frame.up_next).mean()
    conf = (pr >= 0.58) | (pr <= 0.42)
    pred = np.where(conf, (pr >= 0.5).astype(int), 1)  # fall back to the base-rate label "up" when not confident
    out["bucket_conf_or_up"] = (pred == frame.up_next).mean()
    out["share_confident"] = conf.mean()
    print(f"\n{name}: n={len(frame)}", {k: round(float(v), 3) for k, v in out.items()})
    return out


evaluate(test, "TEST all")
big = test[test.rev_last >= 250e6]; evaluate(big, "TEST large (quarterly revenue >= 250M)")
pers = pd.Series(((test.g_last > 0).astype(int) == test.up_next).astype(float).values, index=test.index)
alwup = pd.Series((test.up_next == 1).astype(float).values, index=test.index)
gain = pers - alwup
print("persist - always_up gain:", round(gain.mean(), 3), "CI90 (block bootstrap by calendar quarter):", np.round(block_ci(test.assign(g=gain), "g"), 3))
print("\nby year (test):")
for y, d in test.groupby("year"):
    print(y, len(d), "persist", round(((d.g_last > 0).astype(int) == d.up_next).mean(), 3), "always_up", round(d.up_next.mean(), 3))

# --- accuracy by bucket on test (where is the table confident?)
tb = test.groupby("bucket", observed=True).up_next.agg(["mean", "count"]); print("\ntest P(up | bucket):\n", tb.round(3))

# --- EPS growth regression: point = w * g_last (shrunk), compare with 0% naive; empirical quantile band by bucket
def mae(e): return float(np.mean(np.abs(e)))


G = lambda f: f.growth_next.clip(-400, 400)
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    tr_e = G(train) - w * train.g_last.clip(-400, 400); te_e = G(test) - w * test.g_last.clip(-400, 400)
    print(f"growth point w={w}: train MAE {mae(tr_e):.1f}  test MAE {mae(te_e):.1f}  test median|err| {np.median(np.abs(te_e)):.1f}")
w = 0.5
tr_e = G(train) - w * train.g_last.clip(-400, 400)
q = tr_e.groupby(train.bucket, observed=True).quantile([0.05, 0.95]).unstack(); q.columns = ["q05", "q95"]; print("\nerror quantiles by bucket (train):\n", q.round(1))
te_e = G(test) - w * test.g_last.clip(-400, 400)
lo = test.bucket.map(q.q05).astype(float); hi = test.bucket.map(q.q95).astype(float)
cov = ((te_e >= lo) & (te_e <= hi)).mean()
def interval_score(e, lo, hi, a=0.1): return np.mean((hi - lo) + (2 / a) * np.maximum(lo - e, 0) + (2 / a) * np.maximum(e - hi, 0))
print("band coverage on test:", round(float(cov), 3), "interval score:", round(float(interval_score(te_e, lo, hi)), 1),
      "| naive +-45pp band (point 0):", round(float(interval_score(G(test), -45 * np.ones(len(test)), 45 * np.ones(len(test)))), 1),
      "coverage:", round(float(((G(test) >= -45) & (G(test) <= 45)).mean()), 3))
json.dump({"probs": {k: float(v) for k, v in probs.items()}, "q": {k: [float(q.loc[k, 'q05']), float(q.loc[k, 'q95'])] for k in q.index}},
          open("t4-local/dev/eps_fit.json", "w"), indent=1)

# ---------------------------------------------------------------- large-company subset for growth bands (banks etc.)
print("\n================ LARGE subset (quarterly revenue >= 1e9)")
tl, sl = train[train.rev_last >= 1e9], test[test.rev_last >= 1e9]
print("train", len(tl), "test", len(sl), "| test base rate up", round(sl.up_next.mean(), 3), "persist acc", round(((sl.g_last > 0).astype(int) == sl.up_next).mean(), 3))
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    e_tr = G(tl) - w * tl.g_last.clip(-300, 300); e_te = G(sl) - w * sl.g_last.clip(-300, 300)
    print(f"  w={w}: train MAE {mae(e_tr):.1f} test MAE {mae(e_te):.1f} | test median|err| {np.median(np.abs(e_te)):.1f}")
for w in (0.25, 0.5):
    e_tr = G(tl) - w * tl.g_last.clip(-300, 300); e_te = G(sl) - w * sl.g_last.clip(-300, 300)
    q2 = e_tr.groupby(tl.bucket, observed=True).quantile([0.05, 0.95]).unstack(); q2.columns = ["q05", "q95"]
    lo2 = sl.bucket.map(q2.q05).astype(float); hi2 = sl.bucket.map(q2.q95).astype(float)
    print(f"  w={w} bucket quantile bands: cov {((e_te >= lo2) & (e_te <= hi2)).mean():.3f} IS {interval_score(e_te, lo2, hi2):.1f}", q2.round(0).to_dict("index"))
    lo3, hi3 = e_tr.quantile(0.05), e_tr.quantile(0.95)
    print(f"  w={w} pooled band [{lo3:.0f},{hi3:.0f}]: cov {((e_te >= lo3) & (e_te <= hi3)).mean():.3f} IS {interval_score(e_te, lo3 * np.ones(len(e_te)), hi3 * np.ones(len(e_te))):.1f}")
print("  naive 0% point, +-45pp band: cov", round(float(((G(sl) >= -45) & (G(sl) <= 45)).mean()), 3), "IS", round(float(interval_score(G(sl), -45 * np.ones(len(sl)), 45 * np.ones(len(sl)))), 1))

print("\n================ BAND WIDTH vs YEAR (large subset, w=0.25, bucket quantiles fitted on ALL years <= 2022 then scaled by s)")
L = P[P.rev_last >= 1e9].copy(); w = 0.25
L["e"] = G(L) - w * L.g_last.clip(-300, 300)
qq = L.groupby("bucket", observed=True).e.quantile([0.05, 0.95]).unstack(); qq.columns = ["q05", "q95"]
L["lo0"] = L.bucket.map(qq.q05).astype(float); L["hi0"] = L.bucket.map(qq.q95).astype(float)
rows = []
for y, d in L.groupby("year"):
    r = {"year": y, "n": len(d), "naive45": interval_score(G(d), -45 * np.ones(len(d)), 45 * np.ones(len(d)))}
    for s in (0.3, 0.45, 0.6, 0.8, 1.0, 1.25):
        r[f"s{s}"] = interval_score(d.e, d.lo0 * s, d.hi0 * s)
        r[f"c{s}"] = float(((d.e >= d.lo0 * s) & (d.e <= d.hi0 * s)).mean())
    rows.append(r)
R = pd.DataFrame(rows).set_index("year")
print(R[["n", "naive45"] + [f"s{s}" for s in (0.3, 0.45, 0.6, 0.8, 1.0, 1.25)]].round(0).to_string())
print(R[[f"c{s}" for s in (0.3, 0.45, 0.6, 0.8, 1.0, 1.25)]].round(2).to_string())
print("calm years 2013-2018 mean IS by s:", {s: round(float(R.loc[2013:2018, f"s{s}"].mean()), 0) for s in (0.3, 0.45, 0.6, 0.8, 1.0, 1.25)}, "naive45:", round(float(R.loc[2013:2018, "naive45"].mean()), 0))

# ---------------------------------------------------------------- final artifacts (all years <= data limit)
fit = {"edges": [-50, -10, 0, 10, 30], "labels": labels, "feature_clip": 300, "growth_floor_eps": 0.01, "point_weight": 0.25}
allp = P.copy()
tabA = allp.groupby("bucket", observed=True).up_next.agg(["mean", "count"])
fit["p_up"] = {k: round(float(v), 4) for k, v in tabA["mean"].items()}; fit["n_direction"] = int(len(allp))
LL = P[P.rev_last >= 1e9].copy(); LL["e"] = G(LL) - 0.25 * LL.g_last.clip(-300, 300)
qA = LL.groupby("bucket", observed=True).e.quantile([0.05, 0.95]).unstack(); qA.columns = ["q05", "q95"]
fit["growth_band"] = {k: [round(float(qA.loc[k, "q05"]), 1), round(float(qA.loc[k, "q95"]), 1)] for k in qA.index}
fit["growth_band_pooled"] = [round(float(LL.e.quantile(0.05)), 1), round(float(LL.e.quantile(0.95)), 1)]
fit["n_growth"] = int(len(LL))
fit["test_summary"] = {"direction_acc_test_all": 0.658, "always_up_test_all": 0.544, "gain_ci90": [0.065, 0.164], "n_test": 18133,
                       "growth_w0.25_test_MAE_large": 92.0, "naive_test_MAE_large": 98.0}
json.dump(fit, open("t4-local/dev/eps_fit_final.json", "w"), indent=1)
print("\nFINAL FIT", json.dumps(fit, indent=1))

# ---------------------------------------------------------------- pooled band around a ZERO growth point (unparsed entities)
fit["growth_band_w0"] = [round(float(G(LL).quantile(0.05)), 1), round(float(G(LL).quantile(0.95)), 1)]
e0 = G(LL[LL.year >= 2019]); lo0_, hi0_ = fit["growth_band_w0"]
print("zero-point pooled band", fit["growth_band_w0"], "| test-years coverage", round(float(((e0 >= lo0_) & (e0 <= hi0_)).mean()), 3),
      "IS", round(float(interval_score(e0, lo0_ * np.ones(len(e0)), hi0_ * np.ones(len(e0)))), 1),
      "vs naive +-45pp:", round(float(interval_score(e0, -45 * np.ones(len(e0)), 45 * np.ones(len(e0)))), 1))
json.dump(fit, open("t4-local/dev/eps_fit_final.json", "w"), indent=1)

print("\n================ SIZE CHECK: do mega-caps need narrower bands? (point w=0.25; all years pooled)")
for thr in (1e9, 3e9, 1e10):
    S = P[P.rev_last >= thr].copy(); S["e"] = G(S) - 0.25 * S.g_last.clip(-300, 300)
    own = S.groupby("bucket", observed=True).e.quantile([0.05, 0.95]).unstack(); own.columns = ["q05", "q95"]
    lo_o = S.bucket.map(own.q05).astype(float); hi_o = S.bucket.map(own.q95).astype(float)
    lo_b = S.bucket.map(qA.q05).astype(float); hi_b = S.bucket.map(qA.q95).astype(float)   # bands fitted on the >=1e9 group
    print(f"rev >= {thr:.0e}: n={len(S)} | own-quantile band IS {interval_score(S.e, lo_o, hi_o):.0f} cov {((S.e>=lo_o)&(S.e<=hi_o)).mean():.3f}"
          f" | the >=1e9 band IS {interval_score(S.e, lo_b, hi_b):.0f} cov {((S.e>=lo_b)&(S.e<=hi_b)).mean():.3f}"
          f" | direction persist acc {((S.g_last>0).astype(int)==S.up_next).mean():.3f} always_up {S.up_next.mean():.3f}")
