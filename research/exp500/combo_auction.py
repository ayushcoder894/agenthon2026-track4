"""Auction bid-to-cover, history 2013-2024: shipped D20R rule vs the global forecaster (data limit 2012; never trained on auctions) vs a
50/50 combination. Same table the agent sees: the tenor's auctions in the 380 days before the month-end cutoff (about 13 values).
Pseudo-unit = month (all tenors auctioned that month); composite formula with naive = last auction, naive band 1.645 sd.
Bands: shipped 1.645 sd (floor 0.08); global q05-q95 rescaled to the table units; combination keeps the shipped band around the blended point."""
import json, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "D:/AGENTHON/t4-local/global")
from gfeat import features, NAMES
M = pickle.load(open("D:/AGENTHON/data_global/global_lim2012.pkl", "rb"))
R = pd.DataFrame(json.load(open("D:/AGENTHON/data_auct/auctions_with_yield.json")))
TEN = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]
R = R[R.original_security_term.isin(TEN)].copy(); R["date"] = pd.to_datetime(R.auction_date)
R["btc"] = pd.to_numeric(R.bid_to_cover_ratio, errors="coerce"); R = R.dropna(subset=["btc"]).sort_values("date").drop_duplicates(["original_security_term", "date"])
R["new"] = (R.reopening == "No").astype(int); R["ten"] = R.original_security_term
rows = []
for mo in sorted(R.date.dt.to_period("M").unique()):
    if mo.year < 2013: continue
    cutoff = (mo - 1).to_timestamp(how="end").normalize(); tg = R[R.date.dt.to_period("M") == mo]
    hist = R[(R.date <= cutoff) & (R.date > cutoff - pd.Timedelta(days=380))]
    for r in tg.itertuples():
        g = hist[hist.ten == r.ten].sort_values("date")
        if len(g) < 6: continue
        b = g.btc.values[::-1]; nw = g.new.values[::-1]; same = b[nw == r.new]
        gap = (same.mean() - b.mean()) if (len(set(nw)) == 2 and len(same) >= 2) else 0.0
        ship = b[:6].mean() + 0.5 * gap; sd = float(np.std(b, ddof=1))
        v = [float(x) for x in g.btc.values]
        fs = features(v, 1, 12)
        if fs is None: continue
        f, s = fs; X = pd.DataFrame([f], columns=NAMES)
        d = M["lam"] * M["pt"].predict(X)[0]; lo, hi = M["q05"].predict(X)[0], M["q95"].predict(X)[0]
        rows.append(dict(mo=str(mo), year=mo.year, ten=r.ten, y=r.btc, last=v[-1], sd=sd, ship=ship, glob=v[-1] + d * s, glo=v[-1] + min(lo, d) * s, ghi=v[-1] + max(hi, d) * s))
D = pd.DataFrame(rows)
def IS(lo, hi, y): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
def comp(y, p, lo, hi, n, nlo, nhi):
    m0, m1 = np.abs(y - n).mean(), np.abs(y - p).mean(); q = m0 / (m0 + m1) if m0 + m1 > 0 else 0.5
    i0, i1 = IS(nlo, nhi, y).mean(), IS(lo, hi, y).mean(); iq = i0 / (i0 + i1)
    return 0.7 * q + 0.3 * min(iq, max(0.5, q))
out = []
for mo, te in D.groupby("mo"):
    if len(te) < 3: continue
    y, n = te.y.values, te["last"].values; nh = 1.645 * te.sd.values; h = np.maximum(1.645 * te.sd.values, 0.08)
    cb = 0.5 * (te.ship.values + te.glob.values)
    out.append(dict(mo=mo, year=te.year.iloc[0], ship=comp(y, te.ship.values, te.ship.values - h, te.ship.values + h, n, n - nh, n + nh),
                    glob=comp(y, te.glob.values, te.glo.values, te.ghi.values, n, n - nh, n + nh),
                    glob_shipband=comp(y, te.glob.values, te.glob.values - h, te.glob.values + h, n, n - nh, n + nh),
                    combo=comp(y, cb, cb - h, cb + h, n, n - nh, n + nh)))
O = pd.DataFrame(out); u = sorted(O.year.unique()); rng = np.random.default_rng(0); draws = [rng.choice(u, len(u)) for _ in range(2000)]
print(f"auction month pseudo-units 2013-2024: {len(O)} | mean composite: shipped {O.ship.mean():.4f}")
for c in ("glob", "glob_shipband", "combo"):
    d = (O[c] - O.ship).values; se = np.std([np.concatenate([d[O.year.values == x] for x in s]).mean() for s in draws])
    per = {f"{a}-{b}": round(float((O[c] - O.ship)[O.year.between(a, b)].mean()), 4) for a, b in ((2013, 2016), (2017, 2020), (2021, 2024))}
    print(f"  {c:14s} {O[c].mean():.4f}  diff {d.mean():+.4f} (z {d.mean() / se:+.2f})  by period {per}")
