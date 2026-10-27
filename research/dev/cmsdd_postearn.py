"""CMS-DD on post-earnings: the latest 10-Q/10-K before each large-cap earnings event -> narrative / divergence features -> sign of the one-day abnormal return.
Baseline = fundamentals composite F; variants add the divergence features; shuffled control; company-grouped CV and bootstrap."""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "t4-local/dev"); sys.path.insert(0, "t4-local/agent")
src_text = open("t4-local/dev/sec_eval.py").read().split("idx = json.load")[0]
exec(src_text)
from cmsdd_text import features
idx = json.load(open("data_sec_events/index.json")); os.makedirs("data_sec_events/txt", exist_ok=True)
rows = []
for a, m in idx.items():
    f = f"data_sec_events/raw/{a}.htm"
    if not os.path.exists(f): continue
    tf = f"data_sec_events/txt/{a}.txt"
    if os.path.exists(tf): t = open(tf, encoding="utf-8").read()
    else: t = to_text(open(f, "rb").read()); open(tf, "w", encoding="utf-8").write(t)
    r = features(t)
    if r is None or not m["events"]: continue
    ev0 = sorted(m["events"], key=lambda e: e["day"])[0]
    rows.append(dict(set="events", acc=a, cik=m["cik"], found=True, ticker=m["ticker"], abn=ev0["abn"], **r))
T0 = pd.DataFrame(rows); print("filings with MD&A and an event:", len(T0))
T0.to_csv("data_cmsdd_events_text.csv", index=False)
