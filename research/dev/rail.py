import json, pathlib, sys
sys.path.insert(0, "/work/track4-analysis-public"); sys.path.insert(0, "/work/Agenthon2026-public/common")
from baselines.guardrails_example import citation_rail as R
root = pathlib.Path("/work"); tot = 0
for u in sorted((root / "track4-analysis-public/units").iterdir()):
    ans = json.loads((root / f"t4-local/out/{sys.argv[1]}/{u.name}/answer.json").read_text(encoding="utf-8"))
    f1 = R.check_claim_rules(ans, u, token_counter=None)
    corpus = R.load_corpus(u / "corpus"); task = json.loads((u / "task.json").read_text(encoding="utf-8"))
    f2 = R.check_submitted_reasons(ans, corpus, task["cutoff_date"]) if "submitted_reasons" in ans else []
    n_claims = sum(len(e["claims"]) for e in ans["entity_predictions"]); bad = [x for x in f1 if "unchecked" not in str(x)]
    tot += len(bad) + len(f2)
    print(f"{u.name:34s} claims={n_claims:3d} claim_findings={len(bad)} reason_findings={len(f2)} reasons={len(ans.get('submitted_reasons', []))}", [str(x)[:140] for x in (bad + f2)[:2]])
print("TOTAL findings:", tot)
