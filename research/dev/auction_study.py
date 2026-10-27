"""E5: bid-to-cover point rules for nominal coupon auctions, on the history a unit shows (the tenor's auctions in the prior
~12 months). Data: FiscalData auctions_query (U.S. Treasury, public domain), Notes and Bonds auctioned on or before 2022-06-28.
Skill = MAE(last) / (MAE(last) + MAE(rule)), the regression leg with 'last auction carried forward' as the naive."""
import json
import numpy as np
import pandas as pd

rows = [r for p in (1, 2, 3) for r in json.load(open(f"data_auct/p{p}.json"))["data"]]
A = pd.DataFrame(rows)
A = A[A.original_security_term.isin(["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"])].copy()
A["date"] = pd.to_datetime(A.auction_date)
A["btc"] = pd.to_numeric(A.bid_to_cover_ratio, errors="coerce")
A["size"] = pd.to_numeric(A.offering_amt, errors="coerce") / 1e9
A["new"] = A.reopening.eq("No")
A = A.dropna(subset=["btc"]).sort_values("date")
A = A[A.date >= "1998-01-01"]
out = []
for term, g in A.groupby("original_security_term"):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        t = g.date[i]
        h = g[(g.date < t) & (g.date >= t - pd.Timedelta(days=380))]
        if len(h) < 6:
            continue
        same = h[h.new == g.new[i]]
        r = dict(term=term, date=t, y=g.btc[i], new=g.new[i], last=h.btc.iloc[-1], mean6=h.btc.iloc[-6:].mean(), mean_all=h.btc.mean(),
                 mean3=h.btc.iloc[-3:].mean(), ewma=h.btc.ewm(halflife=3).mean().iloc[-1],
                 same_type=same.btc.mean() if len(same) >= 2 else h.btc.iloc[-6:].mean(),
                 same_type_last3=same.btc.iloc[-3:].mean() if len(same) >= 2 else h.btc.iloc[-6:].mean(),
                 has_mix=h.new.nunique() == 2,
                 size_ratio=g["size"][i] / h["size"].iloc[-1] if h["size"].iloc[-1] else 1.0)
        # mean6 shifted by the window's new-vs-reopen gap
        if r["has_mix"]:
            gap = h[h.new == g.new[i]].btc.mean() - h.btc.mean()
            r["mean6_typeadj"] = r["mean6"] + gap
            r["mean6_typeadj_half"] = r["mean6"] + 0.5 * gap
        else:
            r["mean6_typeadj"] = r["mean6_typeadj_half"] = r["mean6"]
        out.append(r)
D = pd.DataFrame(out)
RULES = ["mean3", "mean6", "mean_all", "ewma", "same_type", "same_type_last3", "mean6_typeadj", "mean6_typeadj_half"]


def report(S, name):
    a = np.abs(S.y - S["last"]).mean()
    line = f"{name:34s} n={len(S):4d} MAE_last={a:.3f} |"
    for k in RULES:
        b = np.abs(S.y - S[k]).mean()
        line += f" {k}:{a / (a + b):.3f}"
    print(line)

for per, m in (("1998-2012", D.date.dt.year <= 2012), ("2013-2022H1", D.date.dt.year >= 2013)):
    report(D[m], f"ALL {per}")
    report(D[m & D.has_mix], f"  tenors with new+reopen {per}")
    report(D[m & D.has_mix & D.new], f"    new issues only {per}")
    for term in ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]:
        report(D[m & (D.term == term)], f"    {term} {per}")
# size effect: does a size increase vs last auction lower the btc relative to mean6?
S = D[D.date.dt.year >= 2005].copy()
S["res"] = S.y - S.mean6_typeadj
for lo, hi in ((0, 0.97), (0.97, 1.03), (1.03, 9)):
    s = S[(S.size_ratio >= lo) & (S.size_ratio < hi)]
    print(f"size ratio [{lo},{hi}) n={len(s)} mean residual {s.res.mean():+.3f}")
# interval: residual sd of the best rule as a multiple of the window sd
D["sd_win"] = np.nan
