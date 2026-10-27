"""Cutoff-safe, target-specific predictors for three Development gaps, scored with the Track-4 composite formulas against D20R.
Composite per pseudo-unit (regression): q = naiveMAE/(naiveMAE+MAE); iq = naiveIS/(naiveIS+IS) (90% interval score, 20x miss);
composite = 0.7 q + 0.3 min(iq, max(0.5, q)).  Classification: anchored accuracy (naive accuracy 0.5 stand-in) = accuracy.
Naive rule (stand-in, identical for every model): carry the prior value forward, band +-1.645 sd of the recent history.
Every predictor is fitted only on observations whose outcome was published before the pseudo-unit's cutoff (expanding window).
Pseudo-units mirror the Dev units: one calendar month of coupon auctions; one month of the 8 non-pump CPI components; one quarter of macro revision events."""
import os, sys, json, re, math, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "D:/AGENTHON"; sys.path.insert(0, HERE)
Z = 1.645


def iscore(y, lo, hi): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)


def composite(y, p, lo, hi, npnt, nlo, nhi):
    y, p, lo, hi, npnt, nlo, nhi = map(np.asarray, (y, p, lo, hi, npnt, nlo, nhi))
    m0, m1 = np.abs(y - npnt).mean(), np.abs(y - p).mean(); q = m0 / (m0 + m1) if (m0 + m1) > 0 else 0.5
    i0, i1 = iscore(y, nlo, nhi).mean(), iscore(y, lo, hi).mean(); iq = i0 / (i0 + i1)
    return 0.7 * q + 0.3 * min(iq, max(0.5, q)), q, iq


def boot(diff, years, n=2000, seed=0):
    diff, years = np.asarray(diff), np.asarray(years); u = np.unique(years); idx = {y: np.where(years == y)[0] for y in u}; rng = np.random.default_rng(seed); est = []
    for _ in range(n):
        s = rng.choice(u, len(u)); est.append(diff[np.concatenate([idx[x] for x in s])].mean())
    return float(diff.mean()), float(np.std(est))


def report(name, R, models, base="D20R", periods=((2013, 2018), (2019, 2024))):
    print(f"\n==== {name}: {len(R)} pseudo-units")
    for m in models:
        line = f"  {m:28s} mean composite {R[m].mean():.4f}"
        if m != base:
            d, se = boot(R[m] - R[base], R.year); line += f" | vs {base} {d:+.4f} (se {se:.4f}, z {d / se if se else 0:+.2f})"
            for a, b in periods:
                s = R[(R.year >= a) & (R.year <= b)]
                if len(s): dd, ss = boot(s[m] - s[base], s.year); line += f" | {a}-{b}: {dd:+.4f} (z {dd / ss if ss else 0:+.2f}, n={len(s)})"
        print(line)

out = {}
# ====================================================================================== AUCTIONS
os.chdir(ROOT)
exec(open(f"{ROOT}/t4-local/dev/auction_hier.py", encoding="utf-8").read().split("# ---------- build examples ----------")[0])      # -> A
rows = []
G = {t: g.sort_values("date").reset_index(drop=True) for t, g in A.groupby("ten")}
for t, g in G.items():
    for i in range(len(g)):
        r = g.iloc[i]; mo = pd.Period(r.date, "M"); cutoff = (mo - 1).to_timestamp(how="end").normalize()
        w = g[(g.date <= cutoff) & (g.date > cutoff - pd.Timedelta(days=380))].tail(13).iloc[::-1]
        if len(w) < 6 or w["size"].isna().any() or not np.isfinite(r["size"]): continue
        b, s, nw = w.btc.values, w["size"].values, w.new.values.astype(int)
        same = b[nw == int(r.new)]; gap = (same.mean() - b.mean()) if (len(set(nw)) == 2 and len(same) >= 2) else 0.0
        sd = float(np.std(b, ddof=1)); ship = b[:6].mean() + 0.5 * gap
        same_sizes = s[nw == int(r.new)]
        size_known = r.date.day <= 10                                    # formal announcement before a month-end cutoff (early refunding auctions)
        size_t = r["size"] if size_known else (same_sizes[0] if len(same_sizes) else s[0])   # otherwise: last same-type size (cutoff-safe proxy)
        rows.append(dict(date=r.date, mo=str(mo), year=r.date.year, ten=t, y=r.btc, last=b[0], sd=sd, ship=ship, half=max(Z * sd, 0.08),
                         size_t=size_t, size_m6=s[:6].mean()))
AU = pd.DataFrame(rows).sort_values("date").reset_index(drop=True); AU = AU[AU.date >= "2004-01-01"].reset_index(drop=True)
# expanding fits (refit at each pseudo-unit month): elasticity e and tenor residual quantiles
months = sorted(AU.mo.unique()); res_rows = []
for mo in months:
    te = AU[AU.mo == mo]; tr = AU[AU.date < te.date.min() - pd.Timedelta(days=0)]
    tr = tr[tr.mo < mo]
    if len(tr) < 150 or len(te) < 3: continue
    x = np.log(tr.size_t / tr.size_m6); r_ = np.log(tr.y / tr.ship); e = float(np.clip(-(x * r_).sum() / max((x * x).sum(), 1e-12), 0.0, 0.3))
    pA = te.ship.values * (te.size_t.values / te.size_m6.values) ** (-e)
    rr = np.log(tr.y / (tr.ship * (tr.size_t / tr.size_m6) ** (-e))); pool = (rr.quantile(0.05), rr.quantile(0.95)); qt = {}
    for t, g in tr.assign(rr=rr).groupby("ten"):
        n = len(g); w_ = n / (n + 60); qt[t] = (w_ * g.rr.quantile(0.05) + (1 - w_) * pool[0], w_ * g.rr.quantile(0.95) + (1 - w_) * pool[1])
    qlo = np.array([qt.get(t, pool)[0] for t in te.ten]); qhi = np.array([qt.get(t, pool)[1] for t in te.ten])
    y = te.y.values; npnt = te["last"].values; nlo, nhi = npnt - Z * te.sd.values, npnt + Z * te.sd.values
    c0 = composite(y, te.ship, te.ship - te.half, te.ship + te.half, npnt, nlo, nhi)[0]
    c1 = composite(y, pA, pA - te.half.values, pA + te.half.values, npnt, nlo, nhi)[0]
    c2 = composite(y, pA, pA * np.exp(qlo), pA * np.exp(qhi), npnt, nlo, nhi)[0]
    c3 = composite(y, te.ship, te.ship * np.exp(qlo), te.ship * np.exp(qhi), npnt, nlo, nhi)[0]
    res_rows.append(dict(mo=mo, year=int(mo[:4]), D20R=c0, A1_size_elasticity=c1, A2_elast_plus_resid_band=c2, A3_resid_band_only=c3, e=e))
RA = pd.DataFrame(res_rows)
report("AUCTIONS (month pseudo-units, 2004-2024)", RA, ["D20R", "A1_size_elasticity", "A2_elast_plus_resid_band", "A3_resid_band_only"])
print("  elasticity e (last fit):", round(RA.e.iloc[-1], 4))
out["auction_e_last"] = float(RA.e.iloc[-1])

# ====================================================================================== CPI (8 non-pump components)
__file__ = os.path.join(HERE, "cpi_exps.py"); sys.argv = [sys.argv[0]]
exec(open(os.path.join(HERE, "cpi_exps.py"), encoding="utf-8").read().split("fam = Family(")[0])        # -> L
NONPUMP = ["core", "food", "medical", "newveh", "shelter", "transpsvc", "usedcars", "apparel"]
C = L[L.comp.isin(NONPUMP) & (L.month >= "1990-01-01")].copy().reset_index(drop=True)
lg = C[[f"l{i}" for i in range(1, 13)]].values
C["sd9"] = lg[:, :9].std(1, ddof=1); C["d20_half"] = np.maximum(2.0 * C.sd9, 0.25)        # D20R: small-sample t factor (~1.96-2.0 for n=9) x sd, floor 0.25
def runlen(r):
    s = np.sign(r[1]); k = 0
    for v in r[1:]:
        if v != 0 and np.sign(v) == s: k += 1
        else: break
    return k
C["run_prev"] = [runlen(r) for r in lg]; C["prev3"] = lg[:, 1:4].mean(1)
C["turn"] = ((np.sign(C["last"]) != np.sign(C.l2)) & (C.run_prev >= 3) & (C["last"] != 0)).astype(float)
C["tsig"] = C.turn * np.sign(C["last"]) * np.abs(C.prev3)
W = L.pivot_table(index="month", columns="comp", values="y")
X = pd.concat({f"x_{c}_l{k}": W[c].shift(k) for c in W.columns for k in (1, 2, 3)}, axis=1)
C = C.merge(X, left_on="month", right_index=True, how="left").dropna(subset=list(X.columns)).reset_index(drop=True)
XC = list(X.columns)
res_rows = []; cache = {}
for mo in sorted(C.month.unique()):
    if mo < pd.Timestamp("2000-01-01"): continue
    te = C[C.month == mo]; tr = C[C.month < mo]
    yr = mo.year
    if yr not in cache:                                      # refit yearly on data before the year (cutoff-safe)
        trY = C[C.month < pd.Timestamp(f"{yr}-01-01")]
        tsel = trY[trY.turn == 1]; b = float((tsel.tsig * (tsel.y - tsel.ship)).sum() / max((tsel.tsig ** 2).sum(), 1e-9)) if len(tsel) > 30 else 0.0
        rid = {}
        for comp in NONPUMP:
            g = trY[trY.comp == comp]; Xm = g[XC].values; mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
            rid[comp] = (Ridge(alpha=1000.0).fit((Xm - mu) / sd, (g.y - g.ship).values), mu, sd)
        rq = {}
        for comp in NONPUMP:
            g = trY[trY.comp == comp]; r_ = (g.y - (g.ship + b * g.tsig)).values; rq[comp] = (np.quantile(r_[-240:], 0.05), np.quantile(r_[-240:], 0.95))
        cache[yr] = (b, rid, rq)
    b, rid, rq = cache[yr]
    y = te.y.values; npnt = te["last"].values; nlo, nhi = npnt - Z * te.sd9.values, npnt + Z * te.sd9.values
    p0 = te.ship.values; h0 = te.d20_half.values
    p1 = p0 + b * te.tsig.values
    adj = np.array([0.5 * rid[c][0].predict(((te[te.comp == c][XC].values - rid[c][1]) / rid[c][2]))[0] for c in te.comp])
    p2 = p1 + adj
    qlo = np.array([rq[c][0] for c in te.comp]); qhi = np.array([rq[c][1] for c in te.comp])
    res_rows.append(dict(mo=mo, year=yr, D20R=composite(y, p0, p0 - h0, p0 + h0, npnt, nlo, nhi)[0],
                         C1_turning_point=composite(y, p1, p1 - h0, p1 + h0, npnt, nlo, nhi)[0],
                         C2_turn_plus_crosscomp=composite(y, p2, p2 - h0, p2 + h0, npnt, nlo, nhi)[0],
                         C3_turn_resid_band=composite(y, p1, p1 + qlo, p1 + qhi, npnt, nlo, nhi)[0], b=b))
RC = pd.DataFrame(res_rows)
report("CPI (8 non-pump components, month pseudo-units, 2000-2024; revised SA data)", RC, ["D20R", "C1_turning_point", "C2_turn_plus_crosscomp", "C3_turn_resid_band"])
out["cpi_b_last"] = float(RC.b.iloc[-1])

# ====================================================================================== MACRO REVISIONS
ev = []
for s in ("DGORDER", "HOUST", "INDPRO", "PAYEMS", "PI", "RSAFS"):
    d = pd.DataFrame(json.load(open(f"{ROOT}/data_alfred/{s}.json"))); d["v"] = pd.to_numeric(d.value, errors="coerce"); d = d.dropna(subset=["v"]).sort_values(["date", "realtime_start"])
    for dt_, g in d.groupby("date"):
        v, rs = g.v.values, g.realtime_start.values
        for i in range(1, len(v)):
            if v[i] != v[i - 1]: ev.append(dict(series=s, ref=dt_, when=pd.Timestamp(rs[i]), up=int(v[i] > v[i - 1]), k=min(i, 3)))
E = pd.DataFrame(ev).sort_values("when").reset_index(drop=True); E["q"] = E.when.dt.to_period("Q").astype(str); E["year"] = E.when.dt.year
res_rows = []
for qq in sorted(E.q.unique()):
    te = E[E.q == qq]; tr = E[E.when < te.when.min()]
    if len(tr) < 500 or te.year.iloc[0] < 2013: continue
    pool = tr.up.mean(); last12 = tr[tr.when >= te.when.min() - pd.Timedelta(days=365)]
    # D20R proxy: pooled recent up-share with prior weight 20 toward the long-run 0.559, then series tilt k=100 on the series' recent events
    pu = (last12.up.sum() + 20 * 0.559) / (len(last12) + 20)
    d20, m1, m2 = [], [], []
    for r in te.itertuples():
        own = last12[last12.series == r.series]; p_own = (own.up.sum() + 100 * pu) / (len(own) + 100); d20.append(int(p_own >= 0.5))
        ser = tr[tr.series == r.series]; m1.append(int(ser.up.mean() >= 0.5) if len(ser) else int(pool >= 0.5))
        sk = ser[ser.k == r.k]; m2.append(int(sk.up.mean() >= 0.5) if len(sk) >= 20 else m1[-1])
    y = te.up.values
    res_rows.append(dict(q=qq, year=int(qq[:4]), D20R=float(np.mean(np.array(d20) == y)), M1_series_direction=float(np.mean(np.array(m1) == y)),
                         M2_series_x_revision_number=float(np.mean(np.array(m2) == y)), n=len(te)))
RM = pd.DataFrame(res_rows)
report("MACRO REVISIONS (quarter pseudo-units, 2013-2022; composite = anchored accuracy)", RM, ["D20R", "M1_series_direction", "M2_series_x_revision_number"], periods=((2013, 2017), (2018, 2022)))
# frozen tables for the agent (fitted on all events up to 2022-06, i.e. before any 2024 cutoff)
out["macro_series_up"] = {s: float(g.up.mean()) for s, g in E.groupby("series")}
out["macro_series_k_up"] = {f"{s}|{k}": float(g.up.mean()) for (s, k), g in E.groupby(["series", "k"]) if len(g) >= 20}
json.dump(out, open(os.path.join(HERE, "predictors_fit.json"), "w"), indent=1)
print("\nfrozen tables written to predictors_fit.json:", json.dumps(out)[:600])
