"""Fit the credit semantic model with RUNTIME-PARITY features (flags from credit_sem.flags, negative equity / loss from the agent's own statement parser, LLM events through
credit_sem.llm_features), check it out of time against the shipped rule and the flags-only logistic, and export the artifact (artifacts.json -> credit_sem)."""
import json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "t4-local/agent"); sys.path.insert(0, "t4-local/dev/exp")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from t4_agent import predict as P, credit_sem as C
from common import log

idx = json.load(open("data_sec_credit/index.json"))
ph = pd.read_csv("data_sec_credit/phrases.csv", usecols=["acc", "label"])
EV = {}
for l in open("t4-local/exp_out/g50_events.jsonl", encoding="utf-8"):
    r = json.loads(l); EV[r["acc"]] = r
rows = []
mism = 0; fl_ref = pd.read_csv("data_sec_credit/phrases.csv"); fl_cols = [c for c in fl_ref.columns if c[:2].isdigit()]
for i, a in enumerate(ph.acc):
    t = open(f"data_sec_credit/txt/{a}.txt", encoding="utf-8").read()
    gc = P._going_concern_hits(t)
    f = C.flags(t, gc)
    ref = fl_ref.iloc[i]
    mism += sum(int(f[c.split(" ")[0]]) != int(ref[c]) for c in fl_cols)
    eq = P._first_value(P._EQ_ROW, t); ni = P._first_value(P._NI_ROW, t)
    if eq is None or ni is None: neg_eq = loss = 0
    else:
        v, m = ni; loss = int(v < 0 or (m.group(1).lower() == "loss" and v > 0)); neg_eq = int(eq[0] < 0)
    ps = EV[a]["passages"]; lf = C.llm_features(EV[a]["events"], ps) or {k: 0 for k in C.LLM_KEYS}
    strong = P._strong_distress([t])
    rule = int(gc > 0 or strong >= 3 or (neg_eq and loss))
    rows.append(dict(strong=strong, gc=int(gc>0), neg_eq=neg_eq, loss=loss, llm=lf, acc=a, label=int(ph.label[i]), year=int(idx[a]["knowledge"][:4]), cik=idx[a]["cik"], rule=rule, xa=C.design(f, neg_eq, loss), xc=C.design(f, neg_eq, loss, lf)))
print("flag mismatches vs the fitted dataset:", mism, "of", len(rows) * len(fl_cols))
D = pd.DataFrame(rows); y = D.label.values
XA = np.array(D.xa.tolist()); XC = np.array(D.xc.tolist())


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=0.3, max_iter=5000).fit((X - mu) / sd, y)
    s = m.predict_proba((X - mu) / sd)[:, 1]
    return dict(mu=mu.tolist(), sd=sd.tolist(), coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]), threshold=float(np.quantile(s[y == 0], 0.94))), m


def apply(mod, X):
    z = mod["intercept"] + ((X - np.array(mod["mu"])) / np.array(mod["sd"])) @ np.array(mod["coef"])
    return 1 / (1 + np.exp(-z))



D["strong"]=D.strong.astype(float)
XE = np.hstack([XC, D[["strong","rule"]].values.astype(float)])
XE2 = np.hstack([XA, D[["strong","rule"]].values.astype(float)])
QS=(0.94,0.96,0.97,0.98)
for cut in (2015, 2016, 2017):
    tr, te = D.year <= cut, D.year > cut; ys = y[te.values]; r = D.rule.values[te.values]
    print(f"train <= {cut}: shipped rule sens {r[ys==1].mean():.3f} spec {1-r[ys==0].mean():.3f}")
    for nm, X in (("A+rule feats", XE2), ("C+rule feats", XE)):
        mod, _ = fit(X[tr.values], y[tr.values]); ptr = apply(mod, X[tr.values]); p = apply(mod, X[te.values])
        print("   ", nm, " ".join(f"q{q}: sens {(p[ys==1] > np.quantile(ptr[y[tr.values]==0], q)).mean():.3f} spec {1-(p[ys==0] > np.quantile(ptr[y[tr.values]==0], q)).mean():.3f} |" for q in QS), "AUC %.3f" % roc_auc_score(ys, p))
    # hybrids on the rule
    L = D.llm[te.values]; res = np.array([l["llm_resolved"] for l in L]); anyc = np.array([l["llm_any"] for l in L]); poss = np.array([l["llm_possible_only"] for l in L]); gc = D.gc.values[te.values]
    veto = r.copy(); veto[(res == 1) & (anyc == 0) & (gc == 0)] = 0
    orr = r.copy(); orr[anyc == 1] = 1
    for nm, h in (("rule, vetoed when LLM says resolved & nothing confirmed", veto), ("rule OR LLM-confirmed event", orr)):
        print(f"    {nm:60s} sens {h[ys==1].mean():.3f} spec {1-h[ys==0].mean():.3f}")
