import json, urllib.request, datetime as dt, os, time, sys
OUT = "t4-local/exp_out/d20_audit/outcomes"; U = "track4-analysis-public/units/"
UA = {"User-Agent": "Mozilla/5.0 (research)"}
mr = json.load(open(U + "t4-macrorev-20240930-us6/task.json"))["entities"]; cp = json.load(open(U + "t4-cpicomp-202410-us11/task.json"))["entities"]
jobs = {(e["series_id"], v): f"av_{e['series_id']}_{v}.csv" for e in mr for v in (e["resolving_release_date"], e["latest_precutoff_vintage"])}
jobs = [(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={s}&vintage_date={v}&cosd=2024-05-01&coed=2024-10-31", n) for (s, v), n in jobs.items()]
jobs += [(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={e['series_fred']}&vintage_date=2024-11-13&cosd=2024-08-01&coed=2024-10-31", f"cpi_{e['series_fred']}_20241113.csv") for e in cp]
prov = []
for rnd in range(4):
    todo = [(u, n) for u, n in jobs if not os.path.exists(f"{OUT}/{n}")]
    if not todo: break
    for u, n in todo:
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40).read()
            open(f"{OUT}/{n}", "wb").write(b); prov.append(dict(file=n, url=u, retrieved=dt.datetime.utcnow().isoformat() + "Z", bytes=len(b))); print("ok", n, flush=True)
        except Exception as ex:
            print("retry", n, repr(ex)[:60], flush=True)
        time.sleep(1.5)
p = json.load(open(OUT + "/provenance.json")); json.dump(p + prov, open(OUT + "/provenance.json", "w"), indent=1)
print("done; missing", [n for u, n in jobs if not os.path.exists(f"{OUT}/{n}")])
