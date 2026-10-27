"""Experiment B: hierarchical (partially pooled) bid-to-cover model for Treasury coupon auctions.
Unit emulation: targets are auctions in month M, cutoff = last day of month M-1, features from the tenor's own auctions in the 380 days before the cutoff (>= 6 obs),
plus the same-cutoff window of the other tenors (market-wide demand factor). Offering size counts as KNOWN only for auctions on day <= 10 of the month (the real pattern:
3Y/10Y/30Y announced, 2Y/5Y/7Y/20Y not). Models refit each test year on auctions whose targets precede that year (expanding window).
 shipped  : mean of last 6 + 0.5 x (same-type mean - all mean); band = max(1.645 sd, 0.08)   (the code in predict._auction)
 last     : last auction carried forward;  mean6 : mean of last 6
 hier     : m6 + pooled ridge on [own recent dev, cross-tenor factor, last-m6, new/reopen gap, indirect-share dev, size ratio (if known)] + tenor-specific slope on own dev (strongly shrunk);
            band = k x partially pooled tenor sd (k fitted on past standardised residuals, 90%)
Metrics: MAE skill vs last (anchor used by the tasks) and vs mean6; interval score (alpha 0.1, 20x penalty); coverage. Block bootstrap by year."""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

rows = [r for p in ("p1", "p2", "p3", "holdout") for r in json.load(open(f"data_auct/{p}.json"))["data"]]
A = pd.DataFrame(rows).drop_duplicates(subset=["auction_date", "original_security_term", "cusip"] if "cusip" in rows[0] else None)
TEN = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]
A = A[A.original_security_term.isin(TEN) & A.security_type.isin(["Note", "Bond"])].copy()
A["date"] = pd.to_datetime(A.auction_date)
A["btc"] = pd.to_numeric(A.bid_to_cover_ratio, errors="coerce")
A["size"] = pd.to_numeric(A.offering_amt, errors="coerce") / 1e9
for c in ("total_accepted", "indirect_bidder_accepted"): A[c] = pd.to_numeric(A.get(c), errors="coerce")
A["ind"] = A.indirect_bidder_accepted / A.total_accepted * 100 if "indirect_bidder_accepted" in A else np.nan
A["new"] = A.reopening.eq("No")
A["ten"] = A.original_security_term
A = A.dropna(subset=["btc"]).sort_values("date")
A = A[A.date >= "1998-01-01"].drop_duplicates(["ten", "date"]).reset_index(drop=True)
print("auctions", len(A), A.date.min().date(), "->", A.date.max().date(), "| by tenor:", A.ten.value_counts().to_dict())

IS_A = 0.1


def iscore(y, lo, hi): return (hi - lo) + (2 / IS_A) * np.maximum(lo - y, 0) + (2 / IS_A) * np.maximum(y - hi, 0)


# ---------- build examples ----------
months = sorted(set(A.date.dt.to_period("M")))
ex = []
for mo in months:
    cutoff = (mo - 1).to_timestamp(how="end").normalize()
    tg = A[A.date.dt.to_period("M") == mo]
    if tg.empty: continue
    win = A[(A.date <= cutoff) & (A.date > cutoff - pd.Timedelta(days=380))]
    devs = {}
    stats = {}
    for t, g in win.groupby("ten"):
        if len(g) < 6: continue
        g = g.sort_values("date"); b = g.btc.values
        stats[t] = dict(m6=b[-6:].mean(), m3=b[-3:].mean(), m12=b.mean(), last=b[-1], sd=b.std(ddof=1), n=len(b), g=g,
                        ind3=g.ind.tail(3).mean() - g.ind.mean() if g.ind.notna().sum() > 4 else 0.0, size6=g["size"].tail(6).mean())
        devs[t] = stats[t]["m3"] - stats[t]["m12"]
    if len(devs) < 4: continue
    F = float(np.mean(list(devs.values())))
    for r in tg.itertuples():
        s = stats.get(r.ten)
        if s is None: continue
        g = s["g"]; same = g[g.new == r.new].btc
        gap = (same.mean() - g.btc.mean()) if (g.new.nunique() == 2 and len(same) >= 2) else 0.0
        known = r.date.day <= 10 and pd.notna(r.size)
        ex.append(dict(date=r.date, mo=str(mo), year=r.date.year, ten=r.ten, y=r.btc, m6=s["m6"], last=s["last"], m12=s["m12"], sd=s["sd"], n=s["n"],
                       own=s["m6"] - s["m12"], F=F, lm6=s["last"] - s["m6"], gap=gap, ind3=s["ind3"] if np.isfinite(s["ind3"]) else 0.0,
                       szr=np.log(r.size / s["size6"]) if known and s["size6"] > 0 else 0.0, known=int(known),
                       shipped=s["m6"] + 0.5 * gap, shipped_half=max(1.645 * s["sd"], 0.08), new=int(r.new)))
D = pd.DataFrame(ex)
D["end"] = D.date
print("examples", len(D), "test-eligible", D.year.min(), "->", D.year.max(), "| size known share", round(D.known.mean(), 2))

TEN_IDX = {t: i for i, t in enumerate(TEN)}
BASE = ["own", "F", "lm6", "gap", "ind3", "szr", "known"]


def design(d, use_size=True):
    X = d[BASE].copy()
    if not use_size: X["szr"] = 0.0; X["known"] = 0
    inter = np.zeros((len(d), len(TEN)))
    for k, (t, o) in enumerate(zip(d.ten.values, d.own.values)): inter[k, TEN_IDX[t]] = o
    return np.hstack([X.values, inter])


def fit_hier(tr, use_size=True, a_main=5.0):
    Xtr = design(tr, use_size); ytr = (tr.y - tr.m6).values
    # separate penalty: pooled coefficients lightly, tenor-specific slopes strongly (partial pooling)
    sc = np.ones(Xtr.shape[1]); sc[len(BASE):] = 0.15        # scaling a column by 0.15 == penalty (1/0.15^2)=44x heavier
    m = Ridge(alpha=a_main).fit(Xtr * sc, ytr); m.sc = sc
    res = ytr - m.predict(Xtr * sc)
    sd_pool = np.sqrt(np.mean(tr.sd ** 2))
    # partial pooling of tenor sd: w = n / (n + k0), shrink towards pooled sd
    z = np.abs(res) / np.sqrt(0.5 * tr.sd ** 2 + 0.5 * sd_pool ** 2)
    m.k90 = float(np.quantile(z, 0.90)); m.sd_pool = sd_pool
    return m


def pred_hier(m, te, use_size=True):
    p = te.m6.values + m.predict(design(te, use_size) * m.sc)
    sdp = np.sqrt(0.5 * te.sd ** 2 + 0.5 * m.sd_pool ** 2).values
    return p, m.k90 * sdp


out = {k: [] for k in ("shipped", "last", "mean6", "hier", "hier_nosize")}
recs = []
for yr in range(2008, D.year.max() + 1):
    te = D[D.year == yr]
    tr = D[(D.date < f"{yr}-01-01") & (D.date >= "1998-06-01")]
    if te.empty or len(tr) < 300: continue
    mh, mn = fit_hier(tr, True), fit_hier(tr, False)
    ph, hh = pred_hier(mh, te, True); pn, hn = pred_hier(mn, te, False)
    for name, p, h in (("shipped", te.shipped.values, te.shipped_half.values), ("last", te["last"].values, te.shipped_half.values),
                       ("mean6", te.m6.values, te.shipped_half.values), ("hier", ph, hh), ("hier_nosize", pn, hn)):
        r = te[["date", "year", "ten", "y", "known", "last"]].copy(); r["model"] = name; r["p"] = p; r["h"] = h
        recs.append(r)
R = pd.concat(recs)
R["err"] = (R.y - R.p).abs(); R["is"] = iscore(R.y.values, (R.p - R.h).values, (R.p + R.h).values); R["cov"] = ((R.y >= R.p - R.h) & (R.y <= R.p + R.h)).astype(float)
R["lasterr"] = (R.y - R["last"]).abs()


def block_ci(sub, base, col, n=500, seed=0):
    a = sub[col].values - base[col].values
    g = pd.DataFrame({"y": sub.year.values, "d": a}).groupby("y").d.agg(["sum", "count"]); rng = np.random.default_rng(seed); est = []
    for _ in range(n):
        s = rng.integers(0, len(g), len(g)); est.append(g["sum"].values[s].sum() / g["count"].values[s].sum())
    return float(np.mean(a)), float(np.std(est))


print("\n=== pooled over test years 2008-latest, all auctions; skill vs LAST = MAE_last/(MAE_last+MAE) ===")
tab = []
for per, (lo, hi) in {"2008-2015": (2008, 2015), "2016-2022": (2016, 2022), "2008-latest": (2008, 2100)}.items():
    for sub, f in (("all", lambda d: d), ("size_unknown", lambda d: d[d.known == 0]), ("size_known", lambda d: d[d.known == 1])):
        base = sub_s = None
        Rs = f(R[(R.year >= lo) & (R.year <= hi)])
        sh = Rs[Rs.model == "shipped"].reset_index(drop=True)
        for md in ("shipped", "mean6", "hier", "hier_nosize"):
            x = Rs[Rs.model == md].reset_index(drop=True)
            if len(x) == 0: continue
            lm = x.lasterr.mean(); dm, se = block_ci(x, sh, "err"); di, sei = block_ci(x, sh, "is")
            tab.append(dict(period=per, sub=sub, model=md, n=len(x), mae=x.err.mean(), skill_vs_last=lm / (lm + x.err.mean()), d_mae_vs_shipped=dm, se=se, z=-dm / se if se else 0,
                            IS=x["is"].mean(), d_IS=di, z_IS=-di / sei if sei else 0, cover=x["cov"].mean()))
T = pd.DataFrame(tab); pd.set_option("display.width", 220)
print(T.round(4).to_string(index=False))
T.to_csv("t4-local/exp_out/auction_hier_results.csv", index=False)
print("\nper-year MAE (all auctions): shipped vs hier vs hier_nosize")
y = R.groupby(["year", "model"]).err.mean().unstack()[["shipped", "hier", "hier_nosize"]]
y["hier_better"] = y.hier < y.shipped; print(y.round(4).to_string()); print("years hier better:", int(y.hier_better.sum()), "of", len(y))
print("\nper-tenor MAE 2016-latest:"); print(R[R.year >= 2016].groupby(["ten", "model"]).err.mean().unstack()[["shipped", "hier", "hier_nosize"]].round(4).to_string())
