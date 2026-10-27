"""Fetch the filings behind a credit study: statements (10-Q/10-K) followed by a bankruptcy (8-K item 1.03) within 12 months vs controls.
Same SEC fair-access rules as sec_fetch.py (4 req/s, descriptive User-Agent with contact email). Output data_sec_credit/raw/<accession>.htm + index.json."""
import glob, json, os, random, re, sys, time
import numpy as np, pandas as pd, requests

UA = {"User-Agent": "Agenthon2026-research ayushgupta19091311@gmail.com", "Accept-Encoding": "gzip, deflate"}
N_POS, N_NEG = int(sys.argv[1]), int(sys.argv[2])
src = open("t4-local/dev/credit_study.py").read().split("def show(")[0]
src = src.replace('COLS = ["cik",', 'COLS = ["accession_number", "cik",')
exec(src)
f = f[f.knowledge_date >= "2010-06-01"]
pos = f[f.bk12 == 1].sort_values("knowledge_date").groupby("cik").tail(1)          # latest statement before the filing, one per company
never = set(bk.keys())
neg = f[(f.bk12 == 0) & ~f.cik.isin(never)]
rng = np.random.default_rng(11)
pos = pos.sample(min(N_POS, len(pos)), random_state=5); neg = neg.iloc[rng.choice(len(neg), N_NEG, replace=False)]
sample = pd.concat([pos.assign(label=1), neg.assign(label=0)])
print("positives", len(pos), "controls", len(neg), flush=True)
sess = requests.Session(); sess.headers.update(UA); last = [0.0]
def get(url):
    wait = 0.26 - (time.time() - last[0])
    if wait > 0: time.sleep(wait)
    last[0] = time.time()
    for k in range(3):
        r = sess.get(url, timeout=60)
        if r.status_code == 200: return r
        if r.status_code in (429, 503): time.sleep(5 * (k + 1)); continue
        return r
    return r
idx_path = "data_sec_credit/index.json"
index = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
for n, r in enumerate(sample.itertuples()):
    a = r.accession_number
    if a in index: continue
    base = f"https://www.sec.gov/Archives/edgar/data/{int(r.cik)}/{a.replace('-', '')}"
    j = get(base + "/index.json")
    if j.status_code != 200: continue
    items = j.json().get("directory", {}).get("item", [])
    docs = [i for i in items if i["name"].lower().endswith((".htm", ".html")) and not re.search(r"(?i)^(r\d+|ex|exhibit|.*-index)", i["name"]) and not re.search(r"(?i)ex[-_]?\d", i["name"])]
    docs = sorted(docs, key=lambda i: -int(i.get("size") or 0))
    if not docs: continue
    d = get(base + "/" + docs[0]["name"])
    if d.status_code != 200: continue
    open(f"data_sec_credit/raw/{a}.htm", "wb").write(d.content)
    index[a] = {"cik": str(r.cik).zfill(10), "label": int(r.label), "knowledge": str(r.knowledge_date.date()), "period": str(pd.Timestamp(r.period_end).date()), "doc": docs[0]["name"]}
    if n % 40 == 0:
        json.dump(index, open(idx_path, "w")); print(n, len(index), flush=True)
json.dump(index, open(idx_path, "w")); print("done", len(index))
