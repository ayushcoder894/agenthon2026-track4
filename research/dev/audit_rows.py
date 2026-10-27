"""Row-level D20R diagnostic on the 10 practice units (DIAGNOSIS ONLY; practice-unit exception per Issue #24; first-published values with source + retrieval date in outcomes/provenance.json).
Outputs: t4-local/exp_out/d20_audit/audit_rows.csv and audit_report.md"""
import csv, glob, io, json, math, os, re
import numpy as np, pandas as pd

R = "t4-local/exp_out/d20_audit"; O = R + "/outcomes"; U = "track4-analysis-public/units/"
rows = pd.read_csv(R + "/d20_rows.csv")
out = {}   # (unit, entity) -> dict(outcome=..., note=...)


def task(u): return json.load(open(U + u + "/task.json"))


# ---- 1 auctions
u = "t4-auction-btc-202411-us7"; A = pd.DataFrame(json.load(open(O + "/auctions_2024_11.json"))["data"])
A["btc"] = pd.to_numeric(A.bid_to_cover_ratio)
for e in task(u)["entities"]:
    m = A[(A.original_security_term == e["tenor"]) & (A.auction_date == e["auction_date"])]
    out[(u, e["entity_id"])] = dict(outcome=float(m.btc.iloc[0]) if len(m) else np.nan)

# ---- 2 COT
u = "t4-cotpos-202411-us10"; CODE = {"CORN_CBT": "002602", "ES_SP500": "13874A", "EURO_FX": "099741", "GOLD_CMX": "088691", "JPY_CME": "097741", "NATGAS_NYMEX": "023651",
                                     "SILVER_CMX": "084691", "UST_10Y": "043602", "UST_2Y": "042601", "WTI_NYMEX": "067651"}
c0 = {r["cftc_contract_market_code"]: r for r in json.load(open(O + "/cot_2024-10-22.json"))}
c1 = {r["cftc_contract_market_code"]: r for r in json.load(open(O + "/cot_2024-11-26.json"))}
for e in task(u)["entities"]:
    k = CODE[e["entity_id"]]; a, b = c0[k], c1[k]
    net = lambda r: float(r["noncomm_positions_long_all"]) - float(r["noncomm_positions_short_all"])
    chk = abs(net(a) - e["net_noncommercial_20241022"]) < 1 and abs(float(a["open_interest_all"]) - e["open_interest_20241022"]) < 1
    out[(u, e["entity_id"])] = dict(outcome=(net(b) - net(a)) / float(a["open_interest_all"]) * 100, note="" if chk else "MAPPING_MISMATCH")

# ---- 3 EPS (SEC XBRL, earliest-filed 3-month value)
def eps(tk, end, start_after):
    d = json.load(open(f"{O}/eps_{tk}.json"))["units"]["USD/shares"]
    c = [x for x in d if abs((pd.Timestamp(x["end"]) - pd.Timestamp(end)).days) <= 6 and x["start"] >= start_after and 80 <= (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days <= 100 and x["form"].startswith("10-Q")]
    c.sort(key=lambda x: x["filed"]); return (c[0]["val"], c[0]["filed"]) if c else (np.nan, None)


u = "t4-eps-growth-2024Q3-banks"
for e in task(u)["entities"]:
    cur, f = eps(e["entity_id"], "2024-09-30", "2024-06-30"); pri = e["prior_year_q_eps"]
    out[(u, e["entity_id"])] = dict(outcome=(cur - pri) / max(abs(pri), 0.01) * 100, note=f"EPS {cur} vs prior {pri} (10-Q filed {f})")
u = "t4-eps-yoy-2023Q2-mixed"
for e in task(u)["entities"]:
    cur, f = eps(e["entity_id"], "2023-06-30", "2023-03-31"); pri = e["prior_year_q_eps"]
    out[(u, e["entity_id"])] = dict(outcome=cur, note=f"EPS {cur} vs prior {pri} -> {'up' if cur > pri else 'down'} (10-Q filed {f})", prior=pri)

# ---- 4 FOMC yields
for u, d0, d1 in (("t4-fomc-curve-20220728", "2022-07-28", "2022-09-20"), ("t4-fomc-curve-20240918", "2024-09-19", "2024-11-06")):
    for e in task(u)["entities"]:
        s = pd.read_csv(f"{O}/fred_{e['series_fred']}.csv").set_index("observation_date")[e["series_fred"]]
        s = pd.to_numeric(s, errors="coerce"); v0, v1 = s.get(d0), s.get(d1)
        out[(u, e["entity_id"])] = dict(outcome=(v1 - v0) * 100 if pd.notna(v0) and pd.notna(v1) else np.nan, note=f"{d0} {v0} -> {d1} {v1}; task start {e['start_yield_pct']}")

# ---- 5 credit events (SEC 8-K item 1.03 within the window)
u = "t4-credit-event-2023"
for e in task(u)["entities"]:
    sub = json.load(open(f"{O}/sub_{e['entity_id']}.json"))["filings"]["recent"]
    hits = [(d, it) for d, it, f in zip(sub["filingDate"], sub["items"], sub["form"]) if f.startswith("8-K") and "1.03" in str(it) and "2023-03-31" < d <= "2024-03-31"]
    out[(u, e["entity_id"])] = dict(outcome=1.0 if hits else 0.0, note=("8-K Item 1.03 filed " + hits[0][0]) if hits else "no 8-K Item 1.03 in window (default/exchange events not checked)")

# ---- 6 post-earnings abnormal return
u = "t4-postearn-20240201-megacap"
def px(tk):
    j = json.load(open(f"{O}/px_{tk}.json"))["chart"]["result"][0]; ts = j["timestamp"]; cl = j["indicators"]["quote"][0]["close"]
    return {pd.Timestamp(t, unit="s").strftime("%Y-%m-%d"): c for t, c in zip(ts, cl)}
spy = px("SPY")
for e in task(u)["entities"]:
    p = px(e["entity_id"]); r = (p["2024-02-02"] / p["2024-02-01"] - 1) * 100; rs = (spy["2024-02-02"] / spy["2024-02-01"] - 1) * 100
    out[(u, e["entity_id"])] = dict(outcome=r - rs, note=f"{e['entity_id']} {r:+.2f}% vs SPY {rs:+.2f}%")

# ---- 7 macro revision + 8 CPI (ALFRED vintages, if fetched)
u = "t4-macrorev-20240930-us6"
def av(s, v, month):
    f = f"{O}/av_{s}_{v}.csv"
    if not os.path.exists(f): return np.nan
    d = pd.read_csv(f); col = [c for c in d.columns if c != "observation_date"][0]
    d = d.set_index("observation_date")[col]; d = pd.to_numeric(d, errors="coerce"); return d.get(month + "-01", np.nan)
for e in task(u)["entities"]:
    new = av(e["series_id"], e["resolving_release_date"], e["ref_month"]); old = av(e["series_id"], e["latest_precutoff_vintage"], e["ref_month"])
    out[(u, e["entity_id"])] = dict(outcome=new, note=f"pre-cutoff given {e['latest_precutoff_estimate']}, alfred old {old}, resolving {new}", prior=e["latest_precutoff_estimate"])
BLS_CPI = {"CPI_ALLITEMS": 0.2, "CPI_APPAREL": -1.5, "CPI_CORE": 0.3, "CPI_ENERGY": 0.0, "CPI_FOOD": 0.2, "CPI_GASOLINE": -0.9, "CPI_MEDICAL": 0.3,
           "CPI_NEWVEH": 0.0, "CPI_SHELTER": 0.4, "CPI_TRANSPSVC": 0.4, "CPI_USEDCARS": 2.7}
u = "t4-cpicomp-202410-us11"
for e in task(u)["entities"]:
    f = f"{O}/cpi_{e['series_fred']}_20241113.csv"
    if os.path.exists(f):
        d = pd.read_csv(f); col = [c for c in d.columns if c != "observation_date"][0]; s = pd.to_numeric(d.set_index("observation_date")[col], errors="coerce")
        oct_, sep = s.get("2024-10-01"), s.get("2024-09-01"); aug = s.get("2024-08-01")
        out[(u, e["entity_id"])] = dict(outcome=(oct_ / sep - 1) * 100, note=f"index Oct {oct_} Sep {sep}; Sep mom from this vintage {(sep / aug - 1) * 100:.2f} vs given {e['latest_published_mom_pct']}")
    else:
        v = BLS_CPI.get(e["entity_id"], np.nan)
        out[(u, e["entity_id"])] = dict(outcome=v, note="BLS CPI release 2024-11-13, SA m/m, rounded to 0.1 (user-supplied transcription of the archived release; not independently retrieved)")

# ---------------- join + error classification ----------------
rows["outcome"] = [out.get((r.unit, r.entity), {}).get("outcome", np.nan) for r in rows.itertuples()]
rows["note"] = [out.get((r.unit, r.entity), {}).get("note", "") for r in rows.itertuples()]
rows["prior"] = [out.get((r.unit, r.entity), {}).get("prior", np.nan) for r in rows.itertuples()]
rows.to_csv(R + "/audit_rows.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 200)
print(rows[["unit", "entity", "label", "point", "lo", "hi", "outcome", "note"]].round(3).to_string(index=False))
