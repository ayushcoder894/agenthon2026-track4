"""Robustness fuzzer: perturb every public unit and require that the agent exits 0 and writes a locally valid answer.json with the full roster."""
import copy, json, os, pathlib, shutil, subprocess, sys
sys.path.insert(0, "t4-local/agent")
from t4_agent.unit_io import load_unit
from t4_agent.answer import validate_answer
SRC = pathlib.Path("track4-analysis-public/units"); OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
def variants(unit):
    t = json.load(open(unit / "task.json", encoding="utf-8"))
    ents = t["entities"]
    def mk(name, task=None, corpus=None, env=None):
        return name, task or t, corpus, env or {}
    num_none = copy.deepcopy(t); num_str = copy.deepcopy(t); nolabels = copy.deepcopy(t); noname = copy.deepcopy(t); extra = copy.deepcopy(t)
    for a, b, c, d in zip(num_none["entities"], num_str["entities"], noname["entities"], extra["entities"]):
        for k, v in list(a.items()):
            if isinstance(v, (int, float)) and not isinstance(v, bool): a[k] = None; b[k] = str(v)
        c.pop("name", None); d["weird_column"] = {"nested": [1, 2, 3]}; d["unit"] = None
    if (nolabels.get("target") or {}).get("labels"):
        nolabels["target"]["labels"] = [f"zz_{l}" for l in nolabels["target"]["labels"]]; nolabels["target"].pop("label_assertions", None)
    return [mk("v1_numbers_none", num_none), mk("v2_numbers_as_strings", num_str), mk("v3_unseen_labels", nolabels), mk("v4_empty_corpus", corpus="empty"),
            mk("v5_truncated_docs", corpus="trunc"), mk("v6_garbage_docs", corpus="garbage"), mk("v7_dead_model_endpoint", env={"MODEL_ENDPOINT": "http://127.0.0.1:9", "MODEL_TOKEN": "x", "T4_LLM_REASONS": "1", "MODEL_NAME": "m"}),
            mk("v8_no_names_extra_cols", noname), mk("v9_extra_cols", extra)]
bad = 0; total = 0
for unit in sorted(p for p in SRC.iterdir() if p.is_dir()):
    for name, task, corpus, env in variants(unit):
        d = OUT / unit.name / name; shutil.rmtree(d, ignore_errors=True); (d / "corpus").mkdir(parents=True)
        json.dump(task, open(d / "task.json", "w", encoding="utf-8"))
        mans = {}
        for loc in ((unit / "corpus" / "manifest.json"), (unit / "manifest.json")):
            if loc.is_file(): mans[loc.relative_to(unit).as_posix()] = json.load(open(loc, encoding="utf-8"))
        for f in (unit / "corpus").iterdir():
            if f.name == "manifest.json": continue
            if corpus == "empty": continue
            txt = json.load(open(f, encoding="utf-8"))
            if corpus == "trunc": txt["text"] = txt.get("text", "")[:200]
            if corpus == "garbage": txt["text"] = "\x00\x01\ufffd" * 50 + "|||| | | 9999999999999999 NaN inf -inf " * 20
            json.dump(txt, open(d / "corpus" / f.name, "w", encoding="utf-8"))
        for rel, man in mans.items():
            if corpus == "empty": man = {**man, "files": []}
            json.dump(man, open(d / rel, "w", encoding="utf-8"))
        e = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "OPENAI", "ANTHROPIC", "T4_LLM"))}; e.update(env); e["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([sys.executable, "t4-local/agent/agent.py", "analyze", "--task", str(d / "task.json"), "--corpus", str(d / "corpus") + "/", "--out", str(d / "answer.json")],
                           capture_output=True, text=True, env=e, timeout=900, encoding="utf-8", errors="replace")
        total += 1; problems = []
        if r.returncode != 0: problems.append(f"exit {r.returncode}")
        try:
            ans = json.load(open(d / "answer.json", encoding="utf-8"))
            u = load_unit(str(d / "task.json"), str(d / "corpus") + "/")
            problems += validate_answer(u, ans)[:3]
        except Exception as exc:
            problems.append(f"{type(exc).__name__}: {str(exc)[:90]}")
        if problems: bad += 1; print(f"FAIL {unit.name:34s} {name:24s} {problems}")
print(f"fuzz runs {total}, failures {bad}")
