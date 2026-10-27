"""Deep-research auction ideas, tested on history (2009-2024), chronological val/final + Holm, then the Nov-2024 in-sample effect.
(1) Bid-dollar model: forecast total bids (BTC x size) from recent auctions, divide by the target's anticipated size (QRA sizes ~ actual sizes historically).
(2) Size elasticity: BTC_hat = base x (size_target / size_ref)^(-e).
(3) Same-type bid dollars (new vs reopen), blends with the shipped BTC rule."""
import os, sys, itertools, json, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import Family
src = open("D:/AGENTHON/t4-local/dev/auction_hier.py", encoding="utf-8").read().split("# ---------- build examples ----------")[0]
os.chdir("D:/AGENTHON"); exec(src)            # -> A (all auctions with btc, size, new, ten, date)
H = 13; rows_ = []
Aidx = {t: g.sort_values("date").reset_index(drop=True) for t, g in A.groupby("ten")}
for t, g in Aidx.items():
    for i in range(len(g)):
        r = g.iloc[i]; mo = pd.Period(r.date, "M"); cutoff = (mo - 1).to_timestamp(how="end").normalize()
        w = g[(g.date <= cutoff) & (g.date > cutoff - pd.Timedelta(days=380))].tail(H).iloc[::-1]
        if len(w) < 6 or not np.isfinite(r["size"]) or w["size"].isna().any(): continue
        b, s, nw = w.btc.values, w["size"].values, w.new.values.astype(int)
        m6 = b[:6].mean(); same = b[nw == int(r.new)]; gap = (same.mean() - b.mean()) if (len(set(nw)) == 2 and len(same) >= 2) else 0.0
        bids = b * s
        same_b = bids[nw == int(r.new)][:4]; same_s = s[nw == int(r.new)][:4]
        rows_.append(dict(date=r.date, ten=t, y=r.btc, size=r["size"], new=int(r.new), shipped=m6 + 0.5 * gap,
                          bid_mean3=bids[:3].mean(), bid_mean6=bids[:6].mean(), bid_same=same_b.mean() if len(same_b) else bids[:6].mean(),
                          size_mean6=s[:6].mean(), size_last=s[0], size_same=same_s.mean() if len(same_s) else s[:6].mean(), m6=m6, last=b[0], max3=b[:3].max(),
                          same_mean3=b[nw == int(r.new)][:3].mean() if (nw == int(r.new)).any() else b[:3].mean()))
D = pd.DataFrame(rows_).sort_values("date").reset_index(drop=True)
D = D[D.date >= "2009-01-01"].reset_index(drop=True)
D["split"] = np.where(D.date <= "2014-12-31", "train", np.where(D.date <= "2019-12-31", "val", "final")); D["block"] = D.date.dt.year * 10 + D.date.dt.quarter
D["sz_ratio"] = D["size"] / D.size_mean6; D["sz_step"] = D["size"] / D.size_last
print("rows", len(D), D.split.value_counts().to_dict(), "| size step-ups >10% vs last:", int((D.sz_step > 1.1).sum()))
def rows(df, w): return df[df.split == w]
def sp(df, w):
    tr = df[df.split == "train"] if w == "val" else df[df.split.isin(["train", "val"])]; return tr, df[df.split == w]
fam = Family("auc_size"); sc = {}
base = lambda df, w: -np.abs(rows(df, w).y.values - rows(df, w).shipped.values)
def add(eid, desc, params, pt):
    f = lambda df, w: -np.abs(rows(df, w).y.values - pt(df, w)); sc[eid] = f; fam.run(eid, desc, params, f, base, D)
for col in ("bid_mean3", "bid_mean6", "bid_same"):
    add(f"auS_{col}", f"bid dollars {col} / target size", dict(col=col), lambda df, w, col=col: (rows(df, w)[col] / rows(df, w)["size"]).values)
    for a in (0.25, 0.5):
        add(f"auS_{col}_blend{a}", f"{a} x bids/size + {1 - a} x shipped", dict(col=col, a=a), lambda df, w, col=col, a=a: (a * rows(df, w)[col] / rows(df, w)["size"] + (1 - a) * rows(df, w).shipped).values)
for ref, e in itertools.product(("size_mean6", "size_last", "size_same"), (0.25, 0.5, 1.0)):
    add(f"auS_elast_{ref}_e{e}", f"shipped x (size/{ref})^-{e}", dict(ref=ref, e=e), lambda df, w, ref=ref, e=e: (rows(df, w).shipped * (rows(df, w)["size"] / rows(df, w)[ref]) ** (-e)).values)
def fit_e(df, w, ref):
    tr, te = sp(df, w); x = np.log(tr["size"] / tr[ref]); r = np.log(tr.y / tr.shipped); e = -(x * r).sum() / max((x * x).sum(), 1e-12)
    return (te.shipped * (te["size"] / te[ref]) ** (-e)).values
for ref in ("size_mean6", "size_last", "size_same"):
    add(f"auS_elast_{ref}_fitted", f"shipped x (size/{ref})^-e, e fitted on train", dict(ref=ref), lambda df, w, ref=ref: fit_e(df, w, ref))
fam.finalize(D, sc, base, top_k=3)
for ref in ("size_mean6", "size_last", "size_same"):
    x = np.log(D["size"] / D[ref]); r = np.log(D.y / D.shipped); print(f"  fitted elasticity (all years) vs {ref}: e = {-(x * r).sum() / (x * x).sum():+.3f}; corr {np.corrcoef(x, r)[0, 1]:+.3f}")
# subsets the document asks for: 20Y new issues after a step-up, refunding months
for nm, m in (("20Y new issues", (D.ten == "20-Year") & (D.new == 1)), ("size step-up >10%", D.sz_step > 1.1), ("30Y new issues", (D.ten == "30-Year") & (D.new == 1))):
    S = D[m & (D.split != "train")]
    if len(S): print(f"  {nm}: n={len(S)} MAE shipped {np.abs(S.y - S.shipped).mean():.3f} | bid_same/size {np.abs(S.y - S.bid_same / S['size']).mean():.3f} | bid_mean6/size {np.abs(S.y - S.bid_mean6 / S['size']).mean():.3f}")
# Nov-2024 Dev in-sample effect with QRA sizes (corpus: 20Y new $16B, 10Y $42B, 30Y $25B, 3Y $58B, 2Y $69B, 5Y $70B, 7Y $44B)
U = "D:/AGENTHON/track4-analysis-public/units/t4-auction-btc-202411-us7/corpus/"
qra = {"10": 42.0, "20": 16.0, "2": 69.0, "30": 25.0, "3": 58.0, "5": 70.0, "7": 44.0}; act = {"10": 2.58, "20": 2.34, "2": 2.77, "30": 2.64, "3": 2.60, "5": 2.43, "7": 2.71}
d30 = {"10": 2.585, "20": 2.557, "2": 2.653, "30": 2.462, "3": 2.628, "5": 2.405, "7": 2.705}
out = []
for t in qra:
    tx = json.load(open(f"{U}TDIRECT_AUCTIONS_{t}Y_20241031.json", encoding="utf-8"))["text"]
    r = re.findall(r"\| (new|reopen\w*) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)", tx)
    nw = np.array([x[0] == "new" for x in r])[::-1]; s = np.array([float(x[1]) for x in r])[::-1]; b = np.array([float(x[2]) for x in r])[::-1]
    bids = b * s; same = bids[nw][:4]
    out.append(dict(t=t + "Y", act=act[t], d30=d30[t], bid_mean6=bids[:6].mean() / qra[t], bid_same=same.mean() / qra[t], elast_last=d30[t] * (qra[t] / s[0]) ** -0.5))
O = pd.DataFrame(out); print(O.round(3).to_string(index=False))
for c in ("d30", "bid_mean6", "bid_same", "elast_last"): print(f"  Dev MAE {c}: {np.abs(O.act - O[c]).mean():.4f}")
print("done")
