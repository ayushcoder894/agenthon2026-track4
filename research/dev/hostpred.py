"""Host-side prediction harness (no container, no LLM): runs an agent's predict_all() on the ten Development units and writes
minimal answer.json files (point, interval, label) that localscore.py can score.
  python hostpred.py AGENT_PKG_PARENT OUTDIR [--generic]
--generic forces every unit through the generic solver (leave-family-out test: how the agent does on a family it has no code for)."""
import json, os, sys

pkg_parent, outdir = sys.argv[1], sys.argv[2]
force_generic = "--generic" in sys.argv
sys.path.insert(0, pkg_parent)
os.environ.setdefault("T4_DEVONLY", "0")
from t4_agent import predict as P          # noqa: E402
from t4_agent.unit_io import load_unit     # noqa: E402

U = "D:/AGENTHON/track4-analysis-public/units/"
if force_generic:
    def _all_generic(unit):
        rows, t = unit.entities, unit.target_type
        if t == "ranking":
            return P._generic_rank(unit, rows)
        fn = P._generic_class if t == "classification" else P._generic_regress
        return {r["entity_id"]: fn(unit, r) for r in rows}
    predict_all = _all_generic
else:
    predict_all = P.predict_all

for u in sorted(os.listdir(U)):
    if "EXAMPLE" in u or not os.path.isdir(U + u):
        continue
    unit = load_unit(U + u + "/task.json", U + u + "/corpus/")
    preds = predict_all(unit)
    rows = []
    order = sorted(preds, key=lambda e: -preds[e].point)
    for e, p in preds.items():
        r = {"entity_id": e, "point_forecast": p.point, "interval": {"level": 0.9, "lo": p.lo, "hi": p.hi}}
        if p.label is not None:
            r["label"] = p.label
        if unit.target_type == "ranking":
            r["rank"] = order.index(e) + 1
        rows.append(r)
    os.makedirs(f"{outdir}/{u}", exist_ok=True)
    json.dump({"task_id": u, "entity_predictions": rows}, open(f"{outdir}/{u}/answer.json", "w"), indent=1)
print("wrote", outdir)
