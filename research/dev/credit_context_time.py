import re, sys
import numpy as np, pandas as pd
src = open("t4-local/dev/credit_context.py").read().split('print("flag prevalence')[0]
exec(src)
BASE = flag + ["neg_eq", "loss"]
D["year"] = D.acc.map(lambda a: int(idx[a]["knowledge"][:4]))
rule = ((D.p63 == 1) | (D[["p65", "p68", "p69", "p70", "p71", "p73", "p76", "p77"]].sum(axis=1) >= 3) | ((D.neg_eq == 1) & (D.loss == 1))).astype(int)
risk = [k for k in NEW if "PROTECTIVE" not in k and not k.startswith(("n09", "n14"))]
for cut in (2015, 2016, 2017):
    tr, te = D[D.year <= cut], D[D.year > cut]
    res = {}
    for nm, cols in (("logistic: 16 flags + equity/loss", BASE), ("logistic: + risk context flags", BASE + risk), ("logistic: + all 14 context flags", BASE + list(NEW))):
        mu, sd = tr[cols].mean(), tr[cols].std() + 1e-9
        m = LogisticRegression(C=0.3, max_iter=3000).fit((tr[cols] - mu) / sd, tr.label)
        ptr = m.predict_proba((tr[cols] - mu) / sd)[:, 1]; thr = np.quantile(ptr[tr.label.values == 0], 0.94)      # threshold fixed on TRAINING controls
        pte = m.predict_proba((te[cols] - mu) / sd)[:, 1]
        res[nm] = (np.mean(pte[te.label.values == 1] > thr), 1 - np.mean(pte[te.label.values == 0] > thr), roc_auc_score(te.label, pte))
    r = rule[te.index]; ys = te.label.values
    print(f"train <= {cut} (n={len(tr)}, pos {int(tr.label.sum())}) | test > {cut} (n={len(te)}, pos {int(te.label.sum())}) | shipped rule: sens {r[ys == 1].mean():.3f} spec {1 - r[ys == 0].mean():.3f}")
    for nm, (s, sp, a) in res.items(): print(f"      {nm:36s} sens {s:.3f} spec {sp:.3f} AUC {a:.4f}")
