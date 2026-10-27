"""Run an agent directory on every public unit (as the platform would) and diff against another agent's outputs."""
import json
import os
import pathlib
import subprocess
import sys

UNITS = pathlib.Path("track4-analysis-public/units")
new_dir, base_dir, out_root = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
env = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "OPENAI", "ANTHROPIC"))}
env["PYTHONIOENCODING"] = "utf-8"


def run(agent_dir, tag):
    results = {}
    for u in sorted(p for p in UNITS.iterdir() if p.is_dir()):
        out = out_root / tag / u.name
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, str(pathlib.Path(agent_dir) / "agent.py"), "analyze", "--task", str(u / "task.json"),
                            "--corpus", str(u / "corpus") + "/", "--out", str(out / "answer.json")],
                           capture_output=True, text=True, env=env, timeout=600)
        results[u.name] = (r.returncode, json.loads((out / "answer.json").read_text(encoding="utf-8")) if (out / "answer.json").exists() else None, r.stderr[-300:])
    return results


new, base = run(new_dir, "new"), run(base_dir, "base")
for name in new:
    rc, a, err = new[name]
    _, b, _ = base[name]
    if a is None:
        print(name, "NO OUTPUT rc", rc, err)
        continue
    diffs = []
    bmap = {e["entity_id"]: e for e in (b or {}).get("entity_predictions", [])}
    for e in a["entity_predictions"]:
        o = bmap.get(e["entity_id"])
        if o is None:
            continue
        ch = []
        for k in ("label", "point_forecast", "rank"):
            if e.get(k) != o.get(k):
                ch.append(f"{k}: {o.get(k)} -> {e.get(k)}")
        if e["interval"] != o["interval"]:
            ch.append(f"interval: [{o['interval']['lo']}, {o['interval']['hi']}] -> [{e['interval']['lo']}, {e['interval']['hi']}]")
        if [c["claim"] for c in e["claims"]] != [c["claim"] for c in o["claims"]]:
            ch.append("claims changed")
        if ch:
            diffs.append(f"    {e['entity_id']}: " + "; ".join(ch))
    print(f"{name}: rc={rc} entities={len(a['entity_predictions'])} changed={len(diffs)}")
    for d in diffs:
        print(d)
