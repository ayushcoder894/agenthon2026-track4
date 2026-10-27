"""Tenor x auction-type residual correction: y_new = y_base + lambda * mean(historical residuals of the base rule for the same tenor and the same
type (new issue / reopening)), using only auctions completed before the cutoff. Base rules: D20R (m6 + 0.5 x 13-window new/reopen gap) and the
D27 in-sample blend (0.7 max3 + 0.3 same-type mean3). Windows: all prior history, or the last 3 / 5 years. Composite formula per month, expanding."""
import json, numpy as np, pandas as pd
R = pd.DataFrame(json.load(open("D:/AGENTHON/data_auct/auctions_with_yield.json")))
TEN = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]
R = R[R.original_security_term.isin(TEN)].copy(); R["date"] = pd.to_datetime(R.auction_date)
R["btc"] = pd.to_numeric(R.bid_to_cover_ratio, errors="coerce"); R = R.dropna(subset=["btc"]).sort_values("date").drop_duplicates(["original_security_term", "date"])
R["new"] = (R.reopening == "No").astype(int); R["ten"] = R.original_security_term
rows = []
for mo in sorted(R.date.dt.to_period("M").unique()):
    cutoff = (mo - 1).to_timestamp(how="end").normalize(); tg = R[R.date.dt.to_period("M") == mo]
    hist = R[(R.date <= cutoff) & (R.date > cutoff - pd.Timedelta(days=380))]
    for r in tg.itertuples():
        g = hist[hist.ten == r.ten].sort_values("date")
        if len(g) < 6: continue
        b = g.btc.values[::-1]; nw = g.new.values[::-1]; same = b[nw == r.new]
        gap = (same.mean() - b.mean()) if (len(set(nw)) == 2 and len(same) >= 2) else 0.0
        d20 = b[:6].mean() + 0.5 * gap; s3 = b[nw == r.new][:3]; blend = 0.7 * b[:3].max() + 0.3 * (s3.mean() if len(s3) else b[:3].mean())
        rows.append(dict(date=r.date, mo=str(mo), year=mo.year, ten=r.ten, new=int(r.new), y=r.btc, last=b[0], sd=float(np.std(b, ddof=1)), d20=d20, blend=blend))
D = pd.DataFrame(rows).sort_values("date").reset_index(drop=True); D["r_d20"] = D.y - D.d20; D["r_blend"] = D.y - D.blend
def iscore(y, lo, hi): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
def comp(y, p, h, n, nh):
    m0, m1 = np.abs(y - n).mean(), np.abs(y - p).mean(); q = m0 / (m0 + m1); i0, i1 = iscore(y, n - nh, n + nh).mean(), iscore(y, p - h, p + h).mean(); iq = i0 / (i0 + i1)
    return 0.7 * q + 0.3 * min(iq, max(0.5, q))
def corr_for(tr, base, ten, new, years):
    s = tr[(tr.ten == ten) & (tr.new == new)]
    if years: s = s[s.date >= s.date.max() - pd.DateOffset(years=years)] if len(s) else s
    return float(s[f"r_{base}"].mean()) if len(s) >= 4 else 0.0
out = []
for mo in sorted(D.mo.unique()):
    te = D[D.mo == mo]; tr = D[D.date < te.date.min()]
    if len(tr) < 300 or len(te) < 3 or te.year.iloc[0] < 2010: continue
    y = te.y.values; n = te["last"].values; nh = 1.645 * te.sd.values; h = np.maximum(1.645 * te.sd.values, 0.08)
    row = dict(mo=mo, year=te.year.iloc[0], D20R=comp(y, te.d20.values, h, n, nh))
    for base in ("d20", "blend"):
        for yrs in (0, 3, 5):
            c = np.array([corr_for(tr, base, t, nw, yrs) for t, nw in zip(te.ten, te.new)])
            for lam in (0.5, 1.0):
                p = te[base].values + lam * c; row[f"{base}+corr{yrs or 'all'}_l{lam}"] = comp(y, p, h, n, nh)
        if base == "blend": row["blend_raw"] = comp(y, te.blend.values, h, n, nh)
    out.append(row)
O = pd.DataFrame(out); cols = [c for c in O.columns if c not in ("mo", "year", "D20R")]
print(f"month pseudo-units {len(O)} (2010-2024); composite vs D20R (year-block bootstrap):")
u = sorted(O.year.unique()); rng = np.random.default_rng(0); draws = [rng.choice(u, len(u)) for _ in range(1000)]
for c in cols:
    d = (O[c] - O.D20R).values; est = [np.concatenate([d[O.year.values == yy] for yy in s]).mean() for s in draws]
    rec = O[O.year >= 2019]; print(f"  {c:22s} {d.mean():+.4f} (z {d.mean() / np.std(est):+.2f}) | 2019-24 {(rec[c] - rec.D20R).mean():+.4f}")
# Nov-2024 rows: corrections from auctions before 2024-11-01
tr = D[D.date < "2024-11-01"]
dev = {"10-Year": (2.58, 2.503, 2.585), "20-Year": (2.34, 2.547, 2.557), "2-Year": (2.77, 2.623, 2.653), "30-Year": (2.64, 2.375, 2.462), "3-Year": (2.60, 2.565, 2.628), "5-Year": (2.43, 2.372, 2.405), "7-Year": (2.71, 2.587, 2.705)}
print("\nNov-2024 (all new issues): tenor x new-issue mean residual of D20R (all history / last 5y) and of the blend:")
for t, (a, p20, pbl) in dev.items():
    c_all, c5 = corr_for(tr, "d20", t, 1, 0), corr_for(tr, "d20", t, 1, 5); b5 = corr_for(tr, "blend", t, 1, 5)
    print(f"  {t:8s} actual {a:.2f} | D20R {p20:.3f} -> +corr5y {p20 + c5:.3f} (corr_all {c_all:+.3f}, corr5y {c5:+.3f}) | blend {pbl:.3f} -> +corr5y {pbl + b5:.3f} ({b5:+.3f})")
