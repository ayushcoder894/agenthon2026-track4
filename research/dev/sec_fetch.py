"""Download a sample of real 10-Q filings from SEC EDGAR (public record) to stress-test the EPS parser.
Rate limit 4 req/s, descriptive User-Agent with a contact email (SEC fair-access policy). Output: data_sec/raw/<accession>.htm + index.json.
Sample: companies in the ZipLime pit data with quarterly revenue >= $1B (plus large financials), 10-Q filings 2015-2022 up to 2022-06-28."""
import glob, json, os, sys, time, random
import pandas as pd, requests

UA = {"User-Agent": "Agenthon2026-research ayushgupta19091311@gmail.com", "Accept-Encoding": "gzip, deflate"}
OUT = "data_sec"; N_COMP = int(sys.argv[1]) if len(sys.argv) > 1 else 260; PER = 5
pit = pd.concat([pd.read_parquet(f, columns=["cik", "accession_number", "period_end", "quarters", "period_kind", "form", "knowledge_date",
                 "eps_diluted", "revenue", "current_assets", "total_assets", "total_liabilities", "revision"]) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
pit = pit[(pit.form.astype(str).str.startswith("10-Q")) & (pit.quarters == 1) & pit.eps_diluted.notna() & (pit.knowledge_date <= "2022-06-28")]
pit = pit[pit.knowledge_date >= "2015-01-01"].sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
pit["fin"] = pit.current_assets.isna() & (pit.total_liabilities / pit.total_assets > 0.85) & (pit.total_assets >= 1e10)
big = pit[pit.revenue >= 1e9]
ciks = big.cik.unique().tolist(); random.Random(7).shuffle(ciks)
fins = [c for c in ciks if big[big.cik == c].fin.any()]
pick = fins[:40] + [c for c in ciks if c not in fins][:N_COMP - 40]
print("companies", len(pick), "financials", len(fins[:40]))
sess = requests.Session(); sess.headers.update(UA)
last = [0.0]
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
index = json.load(open(f"{OUT}/index.json")) if os.path.exists(f"{OUT}/index.json") else {}
for n, cik in enumerate(pick):
    sub = get(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json")
    if sub.status_code != 200: continue
    rec = sub.json()["filings"]["recent"]
    rows = [(a, d, p, rd) for a, f, d, p, rd in zip(rec["accessionNumber"], rec["form"], rec["filingDate"], rec["primaryDocument"], rec["reportDate"]) if f == "10-Q" and "2015-01-01" <= d <= "2022-06-28"]
    random.Random(int(cik)).shuffle(rows)
    for a, d, p, rd in rows[:PER]:
        if a in index: continue
        r = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a.replace('-', '')}/{p}")
        if r.status_code != 200: continue
        open(f"{OUT}/raw/{a}.htm", "wb").write(r.content)
        index[a] = {"cik": str(cik).zfill(10), "filed": d, "period": rd, "doc": p}
    if n % 20 == 0:
        json.dump(index, open(f"{OUT}/index.json", "w")); print(n, len(index), flush=True)
json.dump(index, open(f"{OUT}/index.json", "w")); print("done", len(index))
