"""Download, for each large-cap earnings event (2016-2022), the latest 10-Q/10-K filed before the event day (what a unit's cutoff corpus would contain).
SEC fair-access rules: 4 requests/s, descriptive User-Agent with a contact email. Resumable. Output data_sec_events/raw/<accession>.htm + index.json."""
import json, os, sys, time
import pandas as pd, requests

UA = {"User-Agent": "Agenthon2026-research ayushgupta19091311@gmail.com", "Accept-Encoding": "gzip, deflate"}
ev = pd.read_parquet("t4-local/dev/postearn_events.parquet"); ev["day"] = pd.to_datetime(ev.day); ev = ev[(ev.day >= "2016-01-01") & (ev.day <= "2022-06-28")]
em = pd.read_parquet("data_earn/earnings.parquet", columns=["cik", "ticker"]).dropna(); em["cik"] = em.cik.astype(str).str.zfill(10)
tk2cik = em.groupby("ticker").cik.agg(lambda x: x.value_counts().index[0]).to_dict()
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
idx_path = "data_sec_events/index.json"; index = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
subs = {}
def filings(cik):
    if cik in subs: return subs[cik]
    r = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if r.status_code != 200: subs[cik] = []; return []
    j = r.json(); rec = j["filings"]["recent"]
    rows = [dict(acc=a, form=f, filed=d, doc=p, period=rd) for a, f, d, p, rd in zip(rec["accessionNumber"], rec["form"], rec["filingDate"], rec["primaryDocument"], rec["reportDate"]) if f in ("10-Q", "10-K")]
    if min([x["filed"] for x in rows] or ["9999"]) > "2016-01-01":
        for fx in j["filings"].get("files", []):
            r2 = get("https://data.sec.gov/submissions/" + fx["name"])
            if r2.status_code == 200:
                rec = r2.json(); rows += [dict(acc=a, form=f, filed=d, doc=p, period=rd) for a, f, d, p, rd in zip(rec["accessionNumber"], rec["form"], rec["filingDate"], rec["primaryDocument"], rec["reportDate"]) if f in ("10-Q", "10-K")]
    subs[cik] = sorted(rows, key=lambda x: x["filed"]); return subs[cik]
n = 0
for t, g in ev.groupby("ticker"):
    cik = tk2cik.get(t)
    if not cik: continue
    fl = filings(cik)
    for r in g.itertuples():
        cand = [x for x in fl if pd.Timestamp(x["filed"]) < r.day - pd.Timedelta(days=1) and pd.Timestamp(x["filed"]) >= r.day - pd.Timedelta(days=150)]
        if not cand: continue
        x = cand[-1]; a = x["acc"]
        if a not in index:
            d = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a.replace('-', '')}/{x['doc']}")
            if d.status_code != 200: continue
            open(f"data_sec_events/raw/{a}.htm", "wb").write(d.content)
            index[a] = dict(cik=cik, ticker=t, filed=x["filed"], period=x["period"], form=x["form"], events=[])
        ev_list = index[a]["events"]
        if not any(e["day"] == str(r.day.date()) for e in ev_list): ev_list.append(dict(day=str(r.day.date()), abn=float(r.abn)))
        n += 1
    json.dump(index, open(idx_path, "w")); print(t, len(index), flush=True)
json.dump(index, open(idx_path, "w")); print("done", len(index), "filings for", n, "events")
