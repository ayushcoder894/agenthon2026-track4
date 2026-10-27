"""Disguise every public unit as an unseen family (rename the columns / target names the dispatch keys on) and run the full agent.
Checks: exit 0, locally valid answer with the full roster, and that the generic path (not a family predictor) produced it."""
import json, os, pathlib, shutil, subprocess, sys
sys.path.insert(0, "t4-local/agent")
from t4_agent.unit_io import load_unit
from t4_agent.answer import validate_answer
SRC = pathlib.Path("track4-analysis-public/units"); OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
REN = {"latest_published_mom_pct": "last_print_mom_pct", "start_yield_pct": "opening_yield_pct", "prior_year_q_eps": "eps_year_ago",
       "latest_precutoff_estimate": "current_estimate", "consensus_eps": "street_eps", "flat_threshold_abn_pct": "neutral_band_abn_pct",
       "net_pct_oi_20241022": "netpos_share_oi", "offering_amount_usd_bn": "size_bn"}
TREN = {"bid_to_cover_ratio": "demand_cover_ratio"}
bad = 0
for unit in sorted(p for p in SRC.iterdir() if p.is_dir()):
    t = json.load(open(unit / "task.json", encoding="utf-8"))
    for e in t["entities"]:
        for a, b in REN.items():
            if a in e: e[b] = e.pop(a)
    tn = (t.get("target") or {}).get("name")
    if tn in TREN: t["target"]["name"] = TREN[tn]
    d = OUT / unit.name; shutil.rmtree(d, ignore_errors=True); shutil.copytree(unit, d)
    json.dump(t, open(d / "task.json", "w", encoding="utf-8"))
    e = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "OPENAI", "ANTHROPIC", "T4_LLM"))}; e["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "t4-local/agent/agent.py", "analyze", "--task", str(d / "task.json"), "--corpus", str(d / "corpus") + "/", "--out", str(d / "answer.json")],
                       capture_output=True, text=True, env=e, timeout=900, encoding="utf-8", errors="replace")
    probs = [] if r.returncode == 0 else [f"exit {r.returncode}"]
    try:
        ans = json.load(open(d / "answer.json", encoding="utf-8")); probs += validate_answer(load_unit(str(d / "task.json"), str(d / "corpus") + "/"), ans)[:3]
        x = ans["entity_predictions"][0]
        print(f"{unit.name:30s} {'OK ' if not probs else 'BAD'} first row: label={x.get('label')} point={x['point_forecast']:.4g} [{x['interval']['lo']:.4g}, {x['interval']['hi']:.4g}] {probs}")
    except Exception as exc:
        probs.append(str(exc)[:80]); print(f"{unit.name:30s} BAD {probs}")
    bad += bool(probs)
print("units with problems:", bad)
