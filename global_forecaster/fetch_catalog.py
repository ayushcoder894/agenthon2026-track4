"""Fetch a broad, compliant series catalogue from the FRED/ALFRED API for the global forecaster.
Revised macro releases: INITIAL RELEASE values only (ALFRED output_type=4), so every training value is the first-published one.
Market releases (rates, FX, spreads, commodities): never revised; current values (output_type=1).
Daily series are reduced to weekly (Friday close) and monthly (month mean) later. Output: D:/AGENTHON/data_global/<id>.json + catalog.json"""
import json, os, time, urllib.parse, urllib.request
KEY = open("D:/AGENTHON/fred_key.txt").read().strip()
OUT = "D:/AGENTHON/data_global"; os.makedirs(OUT, exist_ok=True)
REVISED = {9: 20, 10: 40, 13: 30, 14: 8, 21: 8, 25: 15, 27: 15, 46: 30, 50: 40, 51: 15, 54: 25, 91: 5, 97: 10, 148: 10, 180: 6, 188: 15, 192: 12, 229: 12}
MARKET = {18: 30, 17: 20, 304: 10, 86: 10, 365: 20, 183: 10, 64: 3}
API = "https://api.stlouisfed.org/fred/"


def get(path, **kw):
    q = urllib.parse.urlencode(dict(api_key=KEY, file_type="json", **kw))
    for a in range(5):
        try:
            return json.loads(urllib.request.urlopen(API + path + "?" + q, timeout=90).read())
        except Exception as e:
            time.sleep(2 + 3 * a)
    return None


cat = json.load(open(f"{OUT}/catalog.json")) if os.path.exists(f"{OUT}/catalog.json") else {}
for rel, top in list(REVISED.items()) + list(MARKET.items()):
    d = get("release/series", release_id=rel, order_by="popularity", sort_order="desc", limit=1000)
    if not d:
        print("release failed", rel); continue
    keep = []
    for s in d["seriess"]:
        if s["frequency_short"] not in ("M", "W", "D"): continue
        if s["observation_start"] > "2000-01-01" or s["observation_end"] < "2022-01-01": continue
        if s["frequency_short"] == "M" and rel in REVISED and s.get("seasonal_adjustment_short") not in ("SA", "SAAR"): continue
        keep.append(s)
        if len(keep) >= top: break
    for s in keep:
        sid = s["id"]
        if sid in cat: continue
        mode = 4 if rel in REVISED else 1
        kw = dict(series_id=sid, output_type=mode, observation_start="1960-01-01")
        if mode == 4:
            kw.update(realtime_start="1776-07-04", realtime_end="9999-12-31")
        o = get("series/observations", **kw)
        time.sleep(0.55)
        if not o or not o.get("observations"):
            # some revised series have no ALFRED vintages: skip rather than use revised data
            print("  no data", sid, "mode", mode); continue
        obs = o["observations"]
        pts = [(x["date"], x.get("value"), x.get("realtime_start")) for x in obs]
        pts = [(dt, float(v), rs) for dt, v, rs in pts if v not in (".", "", None)]   # (observation date, first-published value, publication date)
        if len(pts) < 120 and s["frequency_short"] == "M": continue
        json.dump(pts, open(f"{OUT}/{sid}.json", "w"))
        cat[sid] = dict(release=rel, title=s["title"], freq=s["frequency_short"], units=s["units_short"], sa=s.get("seasonal_adjustment_short"),
                        revised=rel in REVISED, n=len(pts), start=pts[0][0], end=pts[-1][0])
        json.dump(cat, open(f"{OUT}/catalog.json", "w"), indent=0)
    print(f"release {rel}: kept {len(keep)} | catalogue {len(cat)}", flush=True)
print("DONE", len(cat))
