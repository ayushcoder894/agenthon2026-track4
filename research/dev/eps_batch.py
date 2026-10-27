"""Batch 2 (E22-E25) on the run-rate EPS set (train <= 2018, test 2019-2022Q1).
E22 point = train median of growth per rr bucket (MAE-optimal per bucket) vs 0.75 x rr
E23 direction from a 2-D table P(up | rr bucket, sign of last-quarter YoY growth g)
E24 piecewise point weight per bucket (w_b minimizing train MAE)
E25 direction with the revenue run-rate (latest quarter revenue vs the target quarter's prior-year revenue)"""
import numpy as np
import pandas as pd

exec(open("t4-local/dev/eps_runrate_fit.py").read().split("# direction")[0])
# prior-year revenue of the target quarter
rv = m[["cik", "period_end", "rev_prior"]].rename(columns={"period_end": "pe_next", "rev_prior": "rev_prior_next"})
P2 = P.merge(rv, on=["cik", "pe_next"], how="left")
P2["rrr"] = ((P2.rev_last - P2.rev_prior_next) / P2.rev_prior_next.abs()).clip(-3, 3) * 100
tr, te = P2[P2.pe_next <= "2018-12-31"], P2[P2.pe_next >= "2019-01-01"]
for nm, sel in (("ALL", lambda d: d), ("BIG", lambda d: d[d.big]), ("FIN", lambda d: d[d.fin])):
    a, b = sel(tr), sel(te)
    up = (b.up_next == 1).values; mae0 = np.abs(b.y).mean()
    base_acc = np.mean((b.rr > 0).values == up)
    # E22
    med = a.groupby("b").y.median()
    p22 = b.b.map(med)
    # E24
    wb = {}
    for k, g in a.groupby("b"):
        wb[k] = min((0.0, 0.25, 0.5, 0.75, 1.0, 1.25), key=lambda w: np.abs(g.y - w * g.rr).mean())
    p24 = b.b.map(wb) * b.rr
    # E23
    a2 = a.assign(gs=np.sign(a.g_last)); b2 = b.assign(gs=np.sign(b.g_last))
    t2 = a2.groupby(["b", "gs"]).up_next.mean()
    lab23 = [t2.get((bb, gg), 0.5) >= 0.5 for bb, gg in zip(b2.b, b2.gs)]
    # E25
    ok = b.rrr.notna()
    acc25 = {}
    for wr in (0.1, 0.25, 0.5):
        acc25[wr] = np.mean(((b.rr + wr * b.rrr.fillna(0)) > 0).values == up)
    s = lambda p: mae0 / (mae0 + np.abs(b.y - p).mean())
    print(f"{nm}: test n={len(b)} | E22 skill median-per-bucket {s(p22):.3f} vs 0.75rr {s(0.75 * b.rr):.3f} | E24 piecewise {s(p24):.3f} w={wb} | "
          f"E23 acc 2-D {np.mean(np.array(lab23) == up):.3f} vs rr>0 {base_acc:.3f} | E25 acc rr+w*revrr {acc25} (rev available {ok.mean():.2f})")
