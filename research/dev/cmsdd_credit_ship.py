import re, sys
import numpy as np, pandas as pd
src = open("t4-local/dev/cmsdd_credit_checks.py").read().split('print("positives by form')[0]
exec(src)
C["tone_z"] = (C.tone - C.tone.mean()) / C.tone.std()
RT = flag + ["neg_eq", "loss", "is10k", "lw"]                    # runtime-feasible: phrase flags, parsed equity/net loss, form type, length
def oof(cols, df):
    X = ((df[cols] - df[cols].mean()) / (df[cols].std() + 1e-9)).values; y = df.label.values.astype(int)
    return cv_scores(X, y, df.cik.values), y
df = C.dropna(subset=RT + ["tone_z"]).copy(); y = df.label.values.astype(int); g = df.cik.values
rule = ((df.p63 == 1) | ((df[[f for f in flag if f in ("p65", "p68", "p69", "p70", "p71", "p73", "p76", "p77")]].sum(axis=1)) >= 3) | ((df.neg_eq == 1) & (df.loss == 1))).astype(int).values
spec = 1 - rule[y == 0].mean(); print(f"n={len(df)} pos={int(y.sum())} | shipped rule: sensitivity {rule[y == 1].mean():.3f} specificity {spec:.3f}")
def at_spec(p): t = np.quantile(p[y == 0], spec); return np.mean(p[y == 1] > t)
res = {}
for nm, cols in (("logistic: flags + equity/loss + form/length", RT), ("  + tone", RT + ["tone_z"]), ("  + tone + topline_share", RT + ["tone_z", "topline_share"]), ("tone only", ["tone_z"])):
    p, _ = oof(cols, df); res[nm] = p
    print(f"{nm:46s} AUC {auc(y, p):.4f} | sensitivity at the rule's specificity {at_spec(p):.3f}")
p0, p1 = res["logistic: flags + equity/loss + form/length"], res["  + tone"]
print("delta AUC (tone) 90% CI:", np.round(boot(auc, y, p0, p1, g, n=1000), 4), "| delta sensitivity at fixed spec:", round(at_spec(p1) - at_spec(p0), 3))
rng = np.random.default_rng(2); sens = []
for _ in range(500):
    s = np.concatenate([np.where(g == k)[0] for k in rng.choice(np.unique(g), len(np.unique(g)))])
    t0 = np.quantile(p0[s][y[s] == 0], spec); t1 = np.quantile(p1[s][y[s] == 0], spec)
    sens.append(np.mean(p1[s][y[s] == 1] > t1) - np.mean(p0[s][y[s] == 1] > t0))
print("bootstrap 90% CI of the sensitivity gain at fixed specificity:", np.round(np.percentile(sens, [5, 95]), 3))
