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
    rows.append(dict(acc=a, label=int(ph.label[i]), year=int(idx[a]["knowledge"][:4]), cik=idx[a]["cik"], rule=rule, xa=C.design(f, neg_eq, loss), xc=C.design(f, neg_eq, loss, lf)))
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


for cut in (2015, 2016, 2017):
    tr, te = D.year <= cut, D.year > cut
    ys = y[te.values]; out = {}
    r = D.rule.values[te.values]; out["shipped rule"] = (r[ys == 1].mean(), 1 - r[ys == 0].mean(), np.nan)
    for nm, X in (("A logistic flags", XA), ("C + validated LLM events", XC)):
        mod, _ = fit(X[tr.values], y[tr.values]); p = apply(mod, X[te.values])
        out[nm] = ((p[ys == 1] > mod["threshold"]).mean(), 1 - (p[ys == 0] > mod["threshold"]).mean(), roc_auc_score(ys, p))
    print(f"train <= {cut}: test n={int(te.sum())} pos={int(ys.sum())}")
    for k, (s, sp, au) in out.items(): print(f"    {k:28s} sens {s:.3f} spec {sp:.3f} AUC {au:.3f}")
    log("G52", "credit_sem_parity", f"runtime-parity features, train<={cut}: C sens {out['C + validated LLM events'][0]:.3f} spec {out['C + validated LLM events'][1]:.3f} | A sens {out['A logistic flags'][0]:.3f} spec {out['A logistic flags'][1]:.3f} | shipped rule sens {out['shipped rule'][0]:.3f} spec {out['shipped rule'][1]:.3f}",
        float(out['C + validated LLM events'][0]), None, float(out['shipped rule'][0]), None, unit="sensitivity", decision="eval")
ma, _ = fit(XA, y); mc, mm = fit(XC, y)
names = [f"flag{k}" for k in C.FLAG_KEYS] + ["neg_equity", "net_loss"] + C.LLM_KEYS
print("C coefficients (standardized):", {n: round(c, 2) for n, c in zip(names, mc["coef"])})
art = json.load(open("t4-local/agent/t4_agent/artifacts.json", encoding="utf-8"))
art["families"]["credit_sem"] = {"enabled": True, "model_a": ma, "model_c": mc, "feature_names": names,
    "source": "SEC EDGAR 10-K/10-Q filings (public record): 207 last statements before an 8-K item 1.03 bankruptcy filing and 698 random non-bankrupt statements with assets >= $500M, all filed on or before 2022-06-28; LLM events from a local 7B extractor (training stand-in), quotes validated verbatim",
    "fit": "L2 logistic regression (C=0.3) on standardized features; threshold = 94th percentile of control scores (6% false-positive rate on the controls)",
    "test": "chronological (train <= 2015/2016/2017, fixed threshold): see ARTIFACT_PROVENANCE D19", "n": int(len(D)), "n_pos": int(y.sum())}
json.dump(art, open("t4-local/agent/t4_agent/artifacts.json", "w", encoding="utf-8"), indent=1)
print("exported credit_sem; thresholds", ma["threshold"], mc["threshold"])
