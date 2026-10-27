"""Auction 'concession since last auction' idea, corpus-feasible: at a month-end cutoff, each tenor's table shows its last auction's high yield;
the other tenors that auctioned AFTER it show how far yields have moved since. Feature dY_T = mean over tenors S with an auction after T's last auction
(and before the cutoff) of [S's latest high yield - S's previous high yield], in bp. Hypothesis: a bigger rise since T last auctioned -> cheaper -> stronger demand.
Also: dY_own = T's last high yield - its 6-auction mean (level effect). Point = shipped (m6 + 0.5 new/reopen gap) + beta x feature, beta fitted on prior data.
Scored per month pseudo-unit with the composite formula (naive = last BTC, band 1.645 sd) vs D20R, expanding window."""
import json, numpy as np, pandas as pd
R = pd.DataFrame(json.load(open("D:/AGENTHON/data_auct/auctions_with_yield.json")))
TEN = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]
R = R[R.original_security_term.isin(TEN)].copy(); R["date"] = pd.to_datetime(R.auction_date)
for c in ("bid_to_cover_ratio", "high_yield", "offering_amt"): R[c] = pd.to_numeric(R[c], errors="coerce")
R = R.dropna(subset=["bid_to_cover_ratio", "high_yield"]).sort_values("date").drop_duplicates(["original_security_term", "date"]).reset_index(drop=True)
R["new"] = (R.reopening == "No").astype(int); R["ten"] = R.original_security_term
rows = []
for mo in sorted(R.date.dt.to_period("M").unique()):
    cutoff = (mo - 1).to_timestamp(how="end").normalize(); tg = R[R.date.dt.to_period("M") == mo]
    hist = R[(R.date <= cutoff) & (R.date > cutoff - pd.Timedelta(days=380))]
    if tg.empty or hist.ten.nunique() < 5: continue
    last = {t: g.sort_values("date") for t, g in hist.groupby("ten")}
    for r in tg.itertuples():
        g = last.get(r.ten)
        if g is None or len(g) < 6: continue
        b = g.bid_to_cover_ratio.values[::-1]; nw = g.new.values[::-1]; same = b[nw == r.new]
        gap = (same.mean() - b.mean()) if (len(set(nw)) == 2 and len(same) >= 2) else 0.0
        ship = b[:6].mean() + 0.5 * gap; sd = float(np.std(b, ddof=1))
        t_last = g.date.iloc[-1]; moves = []
        for s, gs in last.items():
            if s == r.ten or len(gs) < 2: continue
            if gs.date.iloc[-1] > t_last: moves.append((gs.high_yield.iloc[-1] - gs.high_yield.iloc[-2]) * 100)
        rows.append(dict(mo=str(mo), year=mo.year, ten=r.ten, y=r.bid_to_cover_ratio, last=b[0], sd=sd, ship=ship, half=max(1.645 * sd, 0.08),
                         dY=float(np.mean(moves)) if moves else 0.0, nmoves=len(moves), dY_own=(g.high_yield.iloc[-1] - g.high_yield.values[-6:].mean()) * 100))
D = pd.DataFrame(rows); D = D[D.year >= 2005].reset_index(drop=True)
print("rows", len(D), "| corr(residual vs shipped, dY) all years:", round(float(np.corrcoef(D.dY, D.y - D.ship)[0, 1]), 3), "| corr with dY_own:", round(float(np.corrcoef(D.dY_own, D.y - D.ship)[0, 1]), 3))
def iscore(y, lo, hi): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
def comp(y, p, lo, hi, n, nlo, nhi):
    m0, m1 = np.abs(y - n).mean(), np.abs(y - p).mean(); q = m0 / (m0 + m1); i0, i1 = iscore(y, nlo, nhi).mean(), iscore(y, lo, hi).mean(); iq = i0 / (i0 + i1)
    return 0.7 * q + 0.3 * min(iq, max(0.5, q))
out = []
for mo in sorted(D.mo.unique()):
    te = D[D.mo == mo]; tr = D[D.mo < mo]
    if len(tr) < 200 or len(te) < 3: continue
    res = []
    for f in ("dY", "dY_own"):
        x, r_ = tr[f].values, (tr.y - tr.ship).values; b = (x * r_).sum() / max((x * x).sum(), 1e-9); res.append(te.ship.values + b * te[f].values)
    y = te.y.values; n = te["last"].values; nlo, nhi = n - 1.645 * te.sd.values, n + 1.645 * te.sd.values; h = te.half.values
    out.append(dict(mo=mo, year=int(mo[:4]), D20R=comp(y, te.ship.values, te.ship.values - h, te.ship.values + h, n, nlo, nhi),
                    concession=comp(y, res[0], res[0] - h, res[0] + h, n, nlo, nhi), own_level=comp(y, res[1], res[1] - h, res[1] + h, n, nlo, nhi)))
O = pd.DataFrame(out)
for m in ("concession", "own_level"):
    d = O[m] - O.D20R; u = sorted(O.year.unique()); rng = np.random.default_rng(0)
    est = [np.concatenate([d[O.year == yy].values for yy in rng.choice(u, len(u))]).mean() for _ in range(1000)]
    rec = O[O.year >= 2019]; print(f"{m:12s} vs D20R {d.mean():+.4f} (z {d.mean() / np.std(est):+.2f}) | 2019-24 {(rec[m] - rec.D20R).mean():+.4f} | n months {len(O)}")
# Nov-2024 feature values (corpus-feasible)
nov = D[D.mo == "2024-11"]
print(nov[["ten", "y", "ship", "dY", "nmoves", "dY_own"]].round(3).to_string(index=False) if len(nov) else "Nov-2024 not in panel (data ends 2024-10)")
