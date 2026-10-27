"""Experiments 63-78: which phrases in a filing discriminate companies that file for bankruptcy (8-K item 1.03) within 12 months?
Data: data_sec_credit (208 last statements before a bankruptcy filing = positives; 700 random non-bankrupt statements with assets >= $500M = controls).
Sensitivity = P(phrase | positive), specificity = 1 - P(phrase | control), LR+ = sensitivity / (1 - specificity).
The sample over-represents positives, so only likelihood ratios (not raw precision) transfer to the real base rate (0.6%)."""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "t4-local/agent")
sys.path.insert(0, "t4-local/dev")
src = open("t4-local/dev/sec_eval.py").read().split("idx = json.load")[0]
exec(src)
from t4_agent import predict as P

idx = json.load(open("data_sec_credit/index.json"))
os.makedirs("data_sec_credit/txt", exist_ok=True)
def text_c(a):
    f = f"data_sec_credit/txt/{a}.txt"
    if os.path.exists(f): return open(f, encoding="utf-8").read()
    t = to_text(open(f"data_sec_credit/raw/{a}.htm", "rb").read()); open(f, "w", encoding="utf-8").write(t); return t

SP = r"[\s ​]"
PH = {
 "63 going concern (shipped: substantial doubt near 'going concern', negation-aware)": lambda t: P._going_concern_hits(t) > 0,
 "64 'substantial doubt' anywhere": lambda t: bool(re.search(r"(?i)substantial" + SP + r"+doubt", t)),
 "65 'forbearance'": lambda t: bool(re.search(r"(?i)forbearance", t)),
 "66 'event of default' / 'events of default' occurred": lambda t: bool(re.search(r"(?i)(?:an?|the)" + SP + r"+event" + SP + r"+of" + SP + r"+default" + SP + r"+(?:has|have)" + SP + r"+occurred|we" + SP + r"+(?:are|were)" + SP + r"+in" + SP + r"+default|in" + SP + r"+default" + SP + r"+(?:under|on)", t)),
 "67 covenant breach/violation/non-compliance": lambda t: bool(re.search(r"(?i)(?:not" + SP + r"+in" + SP + r"+compliance|(?:breach|violation|failure)" + SP + r"+(?:of|to" + SP + r"+comply" + SP + r"+with)" + SP + r"+(?:the|certain|our|its)?" + SP + r"*(?:financial" + SP + r"+)?covenants?)", t)),
 "68 waiver obtained/required": lambda t: bool(re.search(r"(?i)waiver" + SP + r"+(?:of|from|to)" + SP + r"+(?:the|certain|our|its)?" + SP + r"*(?:financial" + SP + r"+)?(?:covenant|default|compliance)|obtain(?:ed)?" + SP + r"+a" + SP + r"+waiver", t)),
 "69 restructuring support agreement / chapter 11 plan": lambda t: bool(re.search(r"(?i)restructuring" + SP + r"+support" + SP + r"+agreement|prepackaged|plan" + SP + r"+of" + SP + r"+reorganization", t)),
 "70 'chapter 11' mentioned": lambda t: bool(re.search(r"(?i)chapter" + SP + r"+11", t)),
 "71 'strategic alternatives'": lambda t: bool(re.search(r"(?i)strategic" + SP + r"+alternatives", t)),
 "72 'ability to continue' (as a going concern/operate)": lambda t: bool(re.search(r"(?i)ability" + SP + r"+to" + SP + r"+continue" + SP + r"+(?:as|operations|our|operating)", t)),
 "73 delisting / minimum bid / continued listing notice": lambda t: bool(re.search(r"(?i)minimum" + SP + r"+bid|continued" + SP + r"+listing|delist", t)),
 "74 'insufficient liquidity' / 'liquidity is limited' / 'sufficient liquidity' negated": lambda t: bool(re.search(r"(?i)(?:insufficient|inadequate|lack" + SP + r"+of)" + SP + r"+(?:liquidity|cash|funds|capital)|may" + SP + r"+not" + SP + r"+(?:have|be" + SP + r"+able" + SP + r"+to" + SP + r"+(?:obtain|raise|generate))" + SP + r"+(?:sufficient|enough)", t)),
 "75 'accumulated deficit' or 'stockholders' deficit' wording": lambda t: bool(re.search(r"(?i)accumulated" + SP + r"+deficit|(?:stock|share)holders['’�]?" + SP + r"+(?:deficit|deficiency)|negative" + SP + r"+(?:equity|working" + SP + r"+capital)", t)),
 "76 'missed' / 'failed to make' interest or principal payment": lambda t: bool(re.search(r"(?i)(?:miss(?:ed)?|fail(?:ed|ure)?" + SP + r"+to" + SP + r"+make|did" + SP + r"+not" + SP + r"+(?:make|pay))" + SP + r"+(?:the|an?|a" + SP + r"+scheduled)?" + SP + r"*(?:interest|principal|coupon)" + SP + r"*(?:payment)?", t)),
 "77 'grace period' (debt) ": lambda t: bool(re.search(r"(?i)grace" + SP + r"+period", t)),
 "78 'cross-default' / 'cross default' / 'cross-acceleration'": lambda t: bool(re.search(r"(?i)cross[- ]?(?:default|acceleration)", t)),
}
rows = []
for a, m in idx.items():
    if not os.path.exists(f"data_sec_credit/raw/{a}.htm"): continue
    t = text_c(a)
    r = {"acc": a, "label": m["label"], "len": len(t)}
    for k, fn in PH.items():
        try: r[k] = bool(fn(t))
        except Exception: r[k] = False
    d = P._distress_text(t) if hasattr(P, "_distress_text") else None
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv("data_sec_credit/phrases.csv", index=False)
pos, neg = D[D.label == 1], D[D.label == 0]
print(f"positives {len(pos)}  controls {len(neg)}  (median text length {int(D.len.median())})")
print(f"{'phrase':80s} {'sens':>5s} {'spec':>5s} {'LR+':>6s}  n_pos n_neg")
for k in PH:
    s = pos[k].mean(); fp = neg[k].mean()
    lr = s / fp if fp > 0 else np.inf
    print(f"{k[:80]:80s} {s:5.3f} {1 - fp:5.3f} {lr:6.1f}  {int(pos[k].sum()):5d} {int(neg[k].sum()):5d}")
