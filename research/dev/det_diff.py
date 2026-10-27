"""Run the agent deterministically (no model) on every public unit and diff predictions/claims against a saved run.
usage: det_diff.py <agent_dir> <saved_run_dir> <out_dir> [--show-reasons]"""
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, "track4-analysis-public")
UNITS = pathlib.Path("track4-analysis-public/units")
agent, saved, out_root = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
env = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "OPENAI", "ANTHROPIC"))}
env["PYTHONIOENCODING"] = "utf-8"
KEYS = ("label", "point_forecast", "interval", "rank", "claims")
for u in sorted(p for p in UNITS.iterdir() if p.is_dir()):
    out = out_root / u.name
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(pathlib.Path(agent) / "agent.py"), "analyze", "--task", str(u / "task.json"),
                    "--corpus", str(u / "corpus") + "/", "--out", str(out / "answer.json")], capture_output=True, env=env, timeout=600)
    a = json.loads((out / "answer.json").read_text(encoding="utf-8"))
    b = json.loads((saved / u.name / "answer.json").read_text(encoding="utf-8"))
    bb = {e["entity_id"]: e for e in b["entity_predictions"]}
    diffs = [e["entity_id"] for e in a["entity_predictions"] if any(e.get(k) != bb[e["entity_id"]].get(k) for k in KEYS)]
    rs = a.get("submitted_reasons") or []
    print(f"{u.name:32s} changed entities: {len(diffs)} {diffs[:4]} | reasons {len(rs)} cites {[len(r.get('citations', [])) for r in rs]}")
    if "--show-reasons" in sys.argv:
        for r in rs:
            print("   P:", r["premise"][:260].replace("\n", " "))
            print("   M:", r["mechanism"][:300])
            print("   I:", r["answer_implication"][:260])
