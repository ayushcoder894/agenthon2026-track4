"""CMS-DD step 1: MD&A extraction + deterministic narrative features. No model is used.
Narrative features (from the MD&A only): tone = (up - down) / (up + down) over metric sentences; topline_share = share of metric sentences about revenue/sales vs profit/margin/cash;
adjusted_rate = adjusted / non-GAAP / 'excluding' mentions per 1,000 words; hedge_rate = uncertainty words per 1,000 words; n_words."""
import re, json, os, sys
import numpy as np, pandas as pd

SP = r"[\s ​]"
START = re.compile(r"(?i)item" + SP + r"*[27]" + SP + r"*[.:\-—–�]*" + SP + r"*management.{0,3}s" + SP + r"+discussion")
END = re.compile(r"(?i)item" + SP + r"*(3|7a|4)" + SP + r"*[.:\-—–�]*" + SP + r"*(quantitative|controls|mine|legal|financial)")
def mdna(text):
    best = ""
    for m in START.finditer(text):
        e = END.search(text, m.end() + 3000)
        seg = text[m.end(): e.start()] if e else text[m.end(): m.end() + 120000]
        if len(seg) > len(best): best = seg
    return best[:400000]

REV = r"(revenue|revenues|net sales|sales|net revenue)"
PROFIT = r"(operating income|operating profit|net income|net earnings|earnings|margin|profit|ebitda|cash flow|cash provided|income from operations|gross profit)"
UP = r"(increas|grew|growth|higher|improv|record|expan|strong|rose|gain|favorabl|benefit)"
DOWN = r"(decreas|declin|lower|reduc|weak|loss|deteriorat|decline|unfavorabl|adverse|pressure|impair|fell|dropp)"
HEDGE = r"(uncertain|may not|could adversely|no assurance|difficult|volatil|substantial doubt|challeng)"
def features(text):
    seg = mdna(text)
    if len(seg) < 4000: return None
    sents = re.split(r"(?<=[.!?])" + SP + r"+(?=[A-Z])", seg)
    up = down = rev = prof = 0
    for s in sents:
        sl = s.lower(); is_rev = re.search(REV, sl); is_prof = re.search(PROFIT, sl)
        if not (is_rev or is_prof): continue
        u, d = len(re.findall(UP, sl)), len(re.findall(DOWN, sl))
        if u == d: continue
        if u > d: up += 1
        else: down += 1
        rev += bool(is_rev) and not is_prof; prof += bool(is_prof)
    words = max(len(seg.split()), 1)
    adj = len(re.findall(r"(?i)\badjusted\b|non-?gaap|excluding|as adjusted", seg))
    hedge = len(re.findall(HEDGE, seg.lower()))
    n = up + down
    return dict(tone=(up - down) / n if n else 0.0, n_metric_sent=n, topline_share=rev / max(rev + prof, 1), adjusted_rate=adj / words * 1000, hedge_rate=hedge / words * 1000, n_words=words)

if __name__ == "__main__":
    rows = []
    for tag, d in (("eps", "data_sec"), ("credit", "data_sec_credit")):
        idx = json.load(open(f"{d}/index.json"))
        for a, m in idx.items():
            f = f"{d}/txt/{a}.txt"
            if not os.path.exists(f): continue
            t = open(f, encoding="utf-8").read(); r = features(t)
            rows.append(dict(set=tag, acc=a, cik=m["cik"], found=r is not None, **(r or {})))
    D = pd.DataFrame(rows); D.to_csv("data_cmsdd_text.csv", index=False)
    print(D.groupby("set").found.agg(["mean", "sum", "count"]))
    print(D[D.found].groupby("set")[["tone", "n_metric_sent", "topline_share", "adjusted_rate", "hedge_rate", "n_words"]].describe().T.round(2).iloc[:, :8].to_string())
