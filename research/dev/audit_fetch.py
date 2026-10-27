"""Fetch first-published outcomes for the 10 Track 4 practice units (diagnosis only; see Issue #24 ruling). Saves raw responses + provenance under t4-local/exp_out/d20_audit/outcomes/.
Every value keeps its source URL and retrieval timestamp."""
import datetime as dt, json, os, re, sys, time, urllib.parse, urllib.request
OUT = "t4-local/exp_out/d20_audit/outcomes"; os.makedirs(OUT, exist_ok=True)
PROV = []
UA = {"User-Agent": "Mozilla/5.0 (research; ayushgupta19091311@gmail.com)", "Accept": "*/*"}


def get(url, name, headers=None, tries=3):
    h = dict(UA); h.update(headers or {})
    for k in range(tries):
        try:
            b = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=45).read()
            open(f"{OUT}/{name}", "wb").write(b)
            PROV.append(dict(file=name, url=url, retrieved=dt.datetime.utcnow().isoformat() + "Z", bytes=len(b)))
            return b
        except Exception as e:
            err = repr(e)[:150]; time.sleep(2)
    PROV.append(dict(file=name, url=url, error=err)); print("FAIL", name, err); return None


# 1 auctions
q = urllib.parse.urlencode({"filter": "auction_date:gte:2024-11-01,auction_date:lte:2024-11-30", "fields": "auction_date,security_type,original_security_term,reopening,bid_to_cover_ratio,offering_amt,total_accepted,total_tendered",
                            "page[size]": "100"})
get("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?" + q, "auctions_2024_11.json")
# 2 COT (legacy futures only)
for d in ("2024-10-22", "2024-11-26"):
    qq = urllib.parse.urlencode({"$where": f"report_date_as_yyyy_mm_dd='{d}T00:00:00.000'", "$limit": "2000"})
    get("https://publicreporting.cftc.gov/resource/6dca-aqww.json?" + qq, f"cot_{d}.json")
# 3 CPI first print (BLS archived release)
get("https://www.bls.gov/news.release/archive/cpi_11132024.htm", "cpi_release_20241113.htm")
get("https://www.bls.gov/news.release/archive/cpi_11132024.pdf", "cpi_release_20241113.pdf")
# 4 SEC XBRL EPS facts
CIK = {"BAC": "0000070858", "C": "0000831001", "GS": "0000886982", "JPM": "0000019617", "MS": "0000895421", "PNC": "0000713676", "USB": "0000036104", "WFC": "0000072971",
       "AMD": "0000002488", "AMGN": "0000318154", "DOW": "0001751788", "HON": "0000773840", "IBM": "0000051143", "TMO": "0000097745"}
for tk, c in CIK.items():
    get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{c}/us-gaap/EarningsPerShareDiluted.json", f"eps_{tk}.json")
    time.sleep(0.3)
# 5 yields (FRED graph csv)
for s in ("DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"):
    get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd=2022-07-20&coed=2024-11-10", f"fred_{s}.csv")
# 6 ALFRED vintages for macro revisions
for s in ("DGORDER", "HOUST", "INDPRO", "PAYEMS", "PI", "RSAFS"):
    get(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={s}&vintage_date=2024-09-30,2024-10-04,2024-10-17,2024-10-18,2024-10-25,2024-11-01,2024-11-15,2024-11-27,2024-10-31&cosd=2024-05-01&coed=2024-09-30", f"alfred_{s}.csv")
# 7 credit: SEC submissions for 8-K Item 1.03
for tk, c in {"BBBY": "0000886158", "BBY": "0000764478", "M": "0000794367", "ODFL": "0000878927", "RAD": "0000084129", "WBA": "0001618921", "WE": "0001813756", "YELL": "0000716006"}.items():
    get(f"https://data.sec.gov/submissions/CIK{c}.json", f"sub_{tk}.json"); time.sleep(0.3)
# 8 prices
for tk in ("AAPL", "AMZN", "META", "SPY"):
    p1 = int(dt.datetime(2024, 1, 29).timestamp()); p2 = int(dt.datetime(2024, 2, 6).timestamp())
    get(f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}?period1={p1}&period2={p2}&interval=1d&events=div", f"px_{tk}.json")
json.dump(PROV, open(f"{OUT}/provenance.json", "w"), indent=1)
print("done", sum(1 for p in PROV if "error" not in p), "ok,", sum(1 for p in PROV if "error" in p), "failed")
