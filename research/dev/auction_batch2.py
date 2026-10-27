"""Experiments 10-16 (auctions), test 2013-2022H1, history = tenor's auctions in the prior 380 days, baseline = mean of last 6 + half the
same-type gap (the shipped rule). Skill = MAE(last)/(MAE(last)+MAE(rule)).
 10 offering-size ratio as a linear adjustment      11 month-of-year effect          12 shrink toward the tenor's 3-year mean
 13 median instead of mean                          14 window length 4 / 8 / 10      15 mean of last 6 vs last 6 trimmed of the extremes
 16 empirical-quantile band from the window (5/95 of the last 12) vs 1.645*sd"""
import json
import numpy as np
import pandas as pd

rows = [r for p in (1, 2, 3) for r in json.load(open(f"data_auct/p{p}.json"))["data"]]
A = pd.DataFrame(rows)
A = A[A.original_security_term.isin(["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"])].copy()
A["date"] = pd.to_datetime(A.auction_date); A["btc"] = pd.to_numeric(A.bid_to_cover_ratio, errors="coerce")
A["size"] = pd.to_numeric(A.offering_amt, errors="coerce") / 1e9; A["new"] = A.reopening.eq("No")
A = A.dropna(subset=["btc"]).sort_values("date"); A = A[A.date >= "1998-01-01"]
out = []
for term, g in A.groupby("original_security_term"):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        t = g.date[i]; h = g[(g.date < t) & (g.date >= t - pd.Timedelta(days=380))]
        if len(h) < 6: continue
        h3 = g[(g.date < t) & (g.date >= t - pd.Timedelta(days=1100))]
        def adj(base):
            return base + 0.5 * (h[h.new == g.new[i]].btc.mean() - h.btc.mean()) if h.new.nunique() == 2 and (h.new == g.new[i]).sum() >= 2 else base
        r = dict(date=t, term=term, y=g.btc[i], last=h.btc.iloc[-1], month=t.month,
                 size_ratio=g["size"][i] / h["size"].iloc[-1] if h["size"].iloc[-1] else 1.0,
                 cur=adj(h.btc.iloc[-6:].mean()), med6=adj(h.btc.iloc[-6:].median()), m4=adj(h.btc.iloc[-4:].mean()),
                 m8=adj(h.btc.iloc[-8:].mean()), m10=adj(h.btc.iloc[-10:].mean()), long3=h3.btc.mean() if len(h3) >= 12 else np.nan,
                 trim6=adj(np.sort(h.btc.iloc[-6:].values)[1:-1].mean()), sd=h.btc.std(ddof=0),
                 q05=np.percentile(h.btc.iloc[-12:], 5), q95=np.percentile(h.btc.iloc[-12:], 95), m12=h.btc.iloc[-12:].mean())
        out.append(r)
D = pd.DataFrame(out)
tr, te = D[D.date.dt.year.between(2003, 2012)], D[D.date.dt.year >= 2013]
sk = lambda df, p: (lambda a, b: a / (a + b))(np.abs(df.y - df["last"]).mean(), np.abs(df.y - p).mean())
print(f"n test {len(te)} | current rule skill {sk(te, te.cur):.3f}")
print("--- 10 size ratio:", end=" ")
b = np.polyfit(np.log(tr.size_ratio.clip(0.5, 2)), tr.y - tr.cur, 1); print("slope", b.round(3), f"skill {sk(te, te.cur + b[0] * np.log(te.size_ratio.clip(0.5, 2)) + b[1]):.3f}")
print("--- 11 month effect:", end=" ")
eff = (tr.y - tr.cur).groupby(tr.month).mean()
for k in (1.0, 0.5): print(f"k={k}: {sk(te, te.cur + k * te.month.map(eff)):.3f}", end="  ")
print()
print("--- 12 shrink to 3y mean:", {k: round(sk(te.dropna(subset=['long3']), (1 - k) * te.dropna(subset=['long3']).cur + k * te.dropna(subset=['long3']).long3), 3) for k in (0, 0.15, 0.3, 0.5)})
print("--- 13/14/15:", {nm: round(sk(te, te[nm]), 3) for nm in ("cur", "med6", "m4", "m8", "m10", "trim6")})
IS = lambda lo, hi, e: np.mean((hi - lo) + 20 * np.maximum(lo - e, 0) + 20 * np.maximum(e - hi, 0))
h1 = np.maximum(1.645 * te.sd, 0.08)
print(f"--- 16 band: sd-based IS {IS(te.cur - h1, te.cur + h1, te.y):.3f} cov {np.mean(np.abs(te.y - te.cur) <= h1):.2f} | last-12 quantile band IS {IS(te.q05, te.q95, te.y):.3f} cov {np.mean((te.y >= te.q05) & (te.y <= te.q95)):.2f}")
# quantile band centred on the rule: [cur + (q05-m12), cur + (q95-m12)]
lo, hi = te.cur + (te.q05 - te.m12), te.cur + (te.q95 - te.m12)
print(f"    centred quantile band IS {IS(lo, hi, te.y):.3f} cov {np.mean((te.y >= lo) & (te.y <= hi)):.2f}")
