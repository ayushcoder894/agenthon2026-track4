"""Fetch full real-time vintage histories (ALFRED via the FRED API) for the 11 CPI series in the CPI unit, then rebuild
FIRST-PRINT month-over-month % changes: m/m(t) = level(t) / level(t-1), both taken from the vintage in which month t was first published."""
import json, os, time, urllib.request, urllib.parse
import pandas as pd
KEY = open("D:/AGENTHON/fred_key.txt").read().strip()
SER = ["CPIAUCSL", "CPILFESL", "CUSR0000SAH1", "CPIENGSL", "CUSR0000SETB01", "CPIUFDSL", "CUSR0000SETA01", "CUSR0000SETA02", "CPIAPPSL", "CPIMEDSL", "CUSR0000SAS4"]
OUT = "D:/AGENTHON/data_alfred_cpi"
for s in SER:
    f = f"{OUT}/{s}.json"
    if os.path.exists(f) and os.path.getsize(f) > 1000: continue
    q = urllib.parse.urlencode(dict(series_id=s, api_key=KEY, file_type="json", realtime_start="1776-07-04", realtime_end="9999-12-31",
                                    observation_start="1988-01-01", output_type=1, limit=100000))
    for attempt in range(4):
        try:
            b = urllib.request.urlopen("https://api.stlouisfed.org/fred/series/observations?" + q, timeout=90).read(); break
        except Exception as e:
            print("retry", s, repr(e)[:80]); time.sleep(3)
    else:
        print("FAILED", s); continue
    open(f, "wb").write(b); d = json.loads(b); print(s, "observations (all vintages):", len(d.get("observations", [])), flush=True); time.sleep(0.6)
rows = []
for s in SER:
    d = pd.DataFrame(json.load(open(f"{OUT}/{s}.json"))["observations"])
    d = d[d.value != "."].copy(); d["v"] = d.value.astype(float); d["date"] = pd.to_datetime(d.date)
    d["rs"] = pd.to_datetime(d.realtime_start); d["re"] = pd.to_datetime(d.realtime_end.replace("9999-12-31", "2262-01-01"))
    first = d.sort_values("rs").groupby("date").first()            # first-published level of each month and its vintage date
    for t, r in first.iterrows():
        prev = d[(d.date == t - pd.DateOffset(months=1)) & (d.rs <= r.rs) & (d.re >= r.rs)]
        if len(prev):
            rows.append(dict(series=s, month=t, vintage=r.rs, first_mom=(r.v / prev.v.iloc[0] - 1) * 100))
FP = pd.DataFrame(rows).sort_values(["series", "month"])
FP.to_csv(f"{OUT}/first_print_mom.csv", index=False)
print(FP.groupby("series").month.agg(["min", "max", "count"]).to_string())
print(FP[(FP.month >= "2024-07-01")].pivot(index="month", columns="series", values="first_mom").round(2).to_string())
