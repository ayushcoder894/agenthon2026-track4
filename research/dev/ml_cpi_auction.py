"""ML-C (CPI) and ML-D (auctions): pooled LightGBM / ridge vs the shipped rules, runtime-readable features only, time split."""
import json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

# ---------------- CPI: pooled over 11 components; features: last, mean3, mean6, sd9, month, pump (for pump-able comps), component id
exec(open("t4-local/dev/cpi_study.py").read().split('print("=== per-component')[0])
rows = []
for c in COMP:
    s = M[c]
    d = pd.DataFrame({"y": s, "last": s.shift(1), "m3": s.shift(1).rolling(3).mean(), "m6": s.shift(1).rolling(6).mean(), "sd9": s.shift(1).rolling(9).std(ddof=0),
                      "pump": M.pump, "month": M.month, "comp": list(COMP).index(c)})
    d["c"] = c; rows.append(d)
D = pd.concat(rows).dropna(); D = D[D.index.year >= 2000]
tr, te = D[D.index.year <= 2015], D[D.index.year >= 2016]
F = ["last", "m3", "m6", "sd9", "pump", "month", "comp"]
rule = 0.5 * te["last"] + 0.5 * te["m6"]
def skill(df, p): a = np.abs(df.y - df["last"]); b = np.abs(df.y - p); return a.mean() / (a.mean() + b.mean())
gb = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3, verbose=-1)
gb.fit(tr[F], tr.y); pg = gb.predict(te[F])
print("CPI pooled (11 components, test 2016-22H1):")
print(f"   shipped rule (0.5 last + 0.5 mean6): {skill(te, rule):.3f}")
print(f"   LightGBM (L1): {skill(te, pg):.3f} | 50/50 blend {skill(te, 0.5 * rule + 0.5 * pg):.3f}")
for c in ("CPI_GASOLINE", "CPI_ENERGY", "CPI_ALLITEMS", "CPI_CORE", "CPI_SHELTER"):
    m = (te.c == c).values
    print(f"   {c:13s}: rule {skill(te[m], rule[m]):.3f} | LightGBM {skill(te[m], pg[m]):.3f}")

# ---------------- auctions
rows = [r for p in (1, 2, 3) for r in json.load(open(f"data_auct/p{p}.json"))["data"]]
A = pd.DataFrame(rows)
A = A[A.original_security_term.isin(["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"])].copy()
A["date"] = pd.to_datetime(A.auction_date); A["btc"] = pd.to_numeric(A.bid_to_cover_ratio, errors="coerce"); A["new"] = A.reopening.eq("No")
A["size"] = pd.to_numeric(A.offering_amt, errors="coerce") / 1e9
A = A.dropna(subset=["btc"]).sort_values("date"); A = A[A.date >= "1998-01-01"]
out = []
for term, g in A.groupby("original_security_term"):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        t = g.date[i]; h = g[(g.date < t) & (g.date >= t - pd.Timedelta(days=380))]
        if len(h) < 6: continue
        base = h.btc.iloc[-6:].mean()
        if h.new.nunique() == 2 and (h.new == g.new[i]).sum() >= 2: base += 0.5 * (h[h.new == g.new[i]].btc.mean() - h.btc.mean())
        out.append(dict(date=t, y=g.btc[i], last=h.btc.iloc[-1], rule=base, m3=h.btc.iloc[-3:].mean(), m12=h.btc.mean(), sd=h.btc.std(ddof=0), new=int(g.new[i]),
                        term=["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"].index(term), month=t.month, n=len(h)))
Q = pd.DataFrame(out); tr, te = Q[Q.date.dt.year <= 2012], Q[Q.date.dt.year >= 2013]
G = ["last", "rule", "m3", "m12", "sd", "new", "term", "month"]
gb = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3, verbose=-1)
gb.fit(tr[G], tr.y - tr.rule); pg = te.rule + gb.predict(te[G])
a = np.abs(te.y - te["last"]).mean()
print("Auctions (test 2013-22H1), skill vs last auction:")
print(f"   shipped rule {a / (a + np.abs(te.y - te.rule).mean()):.3f} | LightGBM residual model {a / (a + np.abs(te.y - pg).mean()):.3f} | 50/50 {a / (a + np.abs(te.y - 0.5 * te.rule - 0.5 * pg).mean()):.3f}")
