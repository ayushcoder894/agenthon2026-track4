import json, os, pathlib, subprocess, sys, time
UNITS = pathlib.Path("track4-analysis-public/units"); out_root = pathlib.Path(sys.argv[1])
env = dict(os.environ, PYTHONIOENCODING="utf-8", T4_ANALYST="0", T4_LLM_REASONS="1", T4_NO_REFINE="0", T4_MOVE_FRAC="0", T4_ALLOW_LABEL_FLIP="0", T4_LLM_LABELS="0",
           MODEL_ENDPOINT=sys.argv[2], MODEL_NAME=sys.argv[3], MODEL_TOKEN="x")
for u in sorted(p for p in UNITS.iterdir() if p.is_dir()):
    o = out_root / u.name; o.mkdir(parents=True, exist_ok=True); t0 = time.time()
    r = subprocess.run([sys.executable, "t4-local/agent/agent.py", "analyze", "--task", str(u / "task.json"), "--corpus", str(u / "corpus") + "/", "--out", str(o / "answer.json")],
                       capture_output=True, text=True, env=env, timeout=700, encoding="utf-8", errors="replace")
    key = [l for l in r.stderr.splitlines() if "[reasons]" in l or "reasons accepted" in l]
    print(f"{u.name:34s} rc={r.returncode} {time.time()-t0:4.0f}s |", " | ".join(k.strip()[:70] for k in key[-2:]), flush=True)
