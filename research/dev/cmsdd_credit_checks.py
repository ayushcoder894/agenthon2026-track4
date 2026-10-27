import glob, json, re, sys
import numpy as np, pandas as pd
src = open("t4-local/dev/cmsdd_eval.py").read().split("# ---------------- credit")[0]
exec(src)
auc = lambda y, p: roc_auc_score(y, p)
ph = pd.read_csv("data_sec_credit/phrases.csv"); flag = [c for c in ph.columns if c[:2].isdigit()]
ph = ph.rename(columns={c: "p" + c.split(" ")[0] for c in flag}); flag = [c for c in ph.columns if c.startswith("p") and c[1:].isdigit()]
for c in flag: ph[c] = ph[c].astype(int)
C = T[T.set == "credit"].merge(ph[["acc", "label"] + flag], on="acc", how="inner"); C = growth(C); C = divergence(C, C)
def form(a):
    t = open(f"data_sec_credit/txt/{a}.txt", encoding="utf-8").read()[:6000].upper()
    return 1 if re.search(r"FORM\s*10-K", t) else 0
C["is10k"] = C.acc.map(form); C["lw"] = np.log(C.n_words)
C["year"] = C.filed.dt.year
print("positives by form: 10-K share pos %.2f vs controls %.2f | median words pos %d vs controls %d" % (C[C.label == 1].is10k.mean(), C[C.label == 0].is10k.mean(), C[C.label == 1].n_words.median(), C[C.label == 0].n_words.median()))
BASE = flag + ["neg_eq", "loss", "rev_g", "dmargin", "ni_g", "F", "fund_n"]
def ablate(name, df, basecols, narr):
    global NARR
    NARR = narr
    df = df.dropna(subset=basecols + narr).copy(); yy = df.label.values.astype(int); g = df.cik.values
    mu, sd = df[basecols].mean(), df[basecols].std() + 1e-9; Xb = ((df[basecols] - mu) / sd).values; Xn = ((df[narr] - df[narr].mean()) / (df[narr].std() + 1e-9)).values
    p0 = cv_scores(Xb, yy, g); p1 = cv_scores(np.hstack([Xb, Xn]), yy, g); ci = boot(auc, yy, p0, p1, g, n=600)
    print(f"{name:52s} n={len(df):4d} baseline {auc(yy, p0):.4f} -> {auc(yy, p1):.4f} delta {auc(yy, p1) - auc(yy, p0):+.4f} (90% CI {ci[0]:+.4f}, {ci[1]:+.4f})")
print("--- controls for format / length")
ablate("baseline + (is10k, log words) controls, + D1,D2,D3,adj,hedge", C, BASE + ["is10k", "lw"], ["D1", "D2", "D3", "adj_z", "hedge_z"])
print("--- one feature at a time (with the format controls in the baseline)")
for f in ["D1", "D2", "D3", "adj_z", "hedge_z"]:
    ablate(f"+ {f} only", C, BASE + ["is10k", "lw"], [f])
print("--- divergence features only (no narrative-only adj/hedge)")
ablate("+ D1, D2, D3", C, BASE + ["is10k", "lw"], ["D1", "D2", "D3"])
print("--- 10-Q only / 10-K only")
ablate("10-Q only: + all five", C[C.is10k == 0], BASE + ["lw"], ["D1", "D2", "D3", "adj_z", "hedge_z"])
ablate("10-K only: + all five", C[C.is10k == 1], BASE + ["lw"], ["D1", "D2", "D3", "adj_z", "hedge_z"])
print("--- chronological: fit on filings <= 2016, test on later filings")
tr, te = C[C.year <= 2016].dropna(subset=BASE + ["is10k", "lw", "D1", "D2", "D3", "adj_z", "hedge_z"]), C[C.year > 2016].dropna(subset=BASE + ["is10k", "lw", "D1", "D2", "D3", "adj_z", "hedge_z"])
B2 = BASE + ["is10k", "lw"]; N2 = ["D1", "D2", "D3", "adj_z", "hedge_z"]
mu, sd = tr[B2 + N2].mean(), tr[B2 + N2].std() + 1e-9
Xtr, Xte = ((tr[B2 + N2] - mu) / sd).values, ((te[B2 + N2] - mu) / sd).values
m0 = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr[:, :len(B2)], tr.label); m1 = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr, tr.label)
a0, a1 = auc(te.label, m0.predict_proba(Xte[:, :len(B2)])[:, 1]), auc(te.label, m1.predict_proba(Xte)[:, 1])
print(f"train n={len(tr)} (pos {int(tr.label.sum())}) test n={len(te)} (pos {int(te.label.sum())}): baseline AUC {a0:.4f} -> with narrative/divergence {a1:.4f} (delta {a1 - a0:+.4f})")
print("coefficients (standardized):", dict(zip(N2, np.round(m1.coef_[0][-5:], 2))))
