"""D22-A/B: EPS direction candidates beyond the shipped run-rate. Chronological (train<=2018 -> test 2019-22) historical diagnostic on the PIT/ZipLime history, AND an in-sample view on the Dev units.
V0 raw run-rate (shipped): up iff cur > prior_target
V1 additive seasonal: cur + (e3 - e4)  [equivalent to last-quarter yoy persistence]
V2 multiplicative seasonal: cur * clip(e3/e4, .5, 2)
V3 robust: median(cur, e1, e2, e3) vs prior  (one-off robust)
V4 blend: mean(V0, V2)
V5 trailing-4 mean (cur,e1,e2,e3)
Accuracy of 'up' direction vs actual up_next; also growth skill (MAE-based) of the point = prior + w*(V - prior)."""
import sys
import numpy as np, pandas as pd
src = open("t4-local/dev/eps_decomp.py").read().split("SETS = {")[0]
exec(src)
lag = F[["cik", "period_end", "eps_diluted_l1", "eps_diluted_l2", "eps_diluted_l3", "eps_diluted_l4"]]
X = D.merge(lag, on=["cik", "period_end"], how="left")
X = X.dropna(subset=["eps_diluted_l3", "eps_diluted_l4"]).copy()
e0, e1, e2, e3, e4 = X.eps_last, X.eps_diluted_l1, X.eps_diluted_l2, X.eps_diluted_l3, X.eps_diluted_l4
pr = X.eps_prior_next
V = {"V0_raw": e0, "V1_add_seasonal": e0 + (e3 - e4), "V2_mult_seasonal": e0 * (e3 / e4.where(e4.abs() > 0.02)).clip(0.5, 2.0).fillna(1.0),
     "V3_median4": pd.concat([e0, e1, e2, e3], axis=1).median(axis=1), "V5_mean4": pd.concat([e0, e1, e2, e3], axis=1).mean(axis=1)}
V["V4_blend"] = 0.5 * V["V0_raw"] + 0.5 * V["V2_mult_seasonal"]
V["V6_blend_med"] = 0.5 * V["V0_raw"] + 0.5 * V["V3_median4"]
X["up_true"] = X.up_next.astype(int)
for per, S in (("train<=2018", X.pe_next <= "2018-12-31"), ("test 2019-22", X.pe_next >= "2019-01-01")):
    for nm, msk in (("ALL", S), ("LARGE", S & (X.rev_last >= 1e9) & (X.abs_prior >= 0.5)), ("FIN", S & X.fin)):
        m = msk
        line = f"{per:13s} {nm:6s} n={int(m.sum()):6d} | "
        for k, v in V.items():
            acc = ((v[m] > pr[m]).astype(int) == X.up_true[m]).mean()
            line += f"{k}:{acc:.3f} "
        print(line)
