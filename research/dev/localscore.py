"""Local per-unit scorer for the ten Development units (diagnosis only; outcomes = first-published values in exp_out/d20_audit).
The organizers' naive answers are hidden, so each unit's unknown naive parameters (naive MAE / interval score / anchor) are
back-solved from two official per-unit results (D20R and D30) using the scorer 5.3.0 composite:
  regression : 0.7 q + 0.3 min(iq, max(.5, q)),  q = nMAE/(nMAE+MAE),  iq = nIS/(nIS+IS)
  class/rank : prediction leg anchored (anchor -> 0.5, perfect -> 1); interval leg as above where the unit scores one.
Usage: python localscore.py OUTDIR [OUTDIR ...]   (each OUTDIR holds <unit>/answer.json)"""
import json, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import brentq

U = "D:/AGENTHON/track4-analysis-public/units/"
A = pd.read_csv("D:/AGENTHON/t4-local/exp_out/d20_audit/audit_rows.csv")
MACRO = {'DGORDER_2024-07_20241025': 'down', 'DGORDER_2024-08_20241025': 'down', 'HOUST_2024-07_20241018': 'up', 'INDPRO_2024-07_20241017': 'up',
         'INDPRO_2024-08_20241017': 'down', 'PAYEMS_2024-07_20241004': 'up', 'PAYEMS_2024-08_20241004': 'up', 'PAYEMS_2024-08_20241101': 'down',
         'PI_2024-07_20241031': 'up', 'PI_2024-08_20241127': 'down', 'RSAFS_2024-07_20241017': 'up', 'RSAFS_2024-08_20241115': 'down'}
OFFICIAL = {  # unit: (D20R, D30)
    "t4-auction-btc-202411-us7": (0.4913, 0.6088), "t4-cotpos-202411-us10": (0.726, 0.8108), "t4-cpicomp-202410-us11": (0.633, 0.6592),
    "t4-credit-event-2023": (0.75, 1.0), "t4-eps-growth-2024Q3-banks": (0.5568, 0.8225), "t4-eps-yoy-2023Q2-mixed": (0.734, 0.8529),
    "t4-fomc-curve-20220728": (0.500, 0.8801), "t4-fomc-curve-20240918": (0.481, 0.8422), "t4-macrorev-20240930-us6": (0.583, 0.6667),
    "t4-postearn-20240201-megacap": (0.735, 0.8516)}
REF = {"D20R": "D:/AGENTHON/t4-local/exp_out/d20_audit/d20", "D30": "D:/AGENTHON/t4-local/d30/out"}


def truth(u):
    t = json.load(open(U + u + "/task.json")); ents = {e["entity_id"]: e for e in t["entities"]}; out = {}
    if u.startswith("t4-macrorev"):
        for e in t["entities"]:
            k = next(k for k in MACRO if e["entity_id"].startswith(k.rsplit("_", 1)[0]) and k.endswith(e["entity_id"][-8:])) if e["entity_id"] not in MACRO else e["entity_id"]
            out[e["entity_id"]] = (MACRO[k], None)
        return t, out
    for r in A[A.unit == u].itertuples():
        y = r.outcome; lab = None
        if u.startswith("t4-credit"): lab = "credit_event" if y >= 0.5 else "no_event"
        elif u.startswith("t4-eps-yoy"): lab = "up" if y > ents[r.entity]["prior_year_q_eps"] else "down"
        elif u.startswith("t4-postearn"):
            th = ents[r.entity].get("flat_threshold_abn_pct", 1.0); lab = "positive_reaction" if y > th else ("negative_reaction" if y < -th else "flat")
        out[r.entity] = (lab, float(y))
    return t, out


def iscore(lo, hi, y): return (hi - lo) + 20 * max(lo - y, 0) + 20 * max(y - hi, 0)


def raw(u, outdir):
    t, tr = truth(u); a = json.load(open(f"{outdir}/{u}/answer.json")); P = {p["entity_id"]: p for p in a["entity_predictions"]}
    ids = list(tr); r = {}
    y = [tr[i][1] for i in ids]
    if y[0] is not None:
        r["mae"] = float(np.mean([abs(P[i]["point_forecast"] - tr[i][1]) for i in ids]))
        r["is"] = float(np.mean([iscore(P[i]["interval"]["lo"], P[i]["interval"]["hi"], tr[i][1]) for i in ids]))
    if t["target"]["type"] == "classification":
        r["acc"] = float(np.mean([P[i].get("label") == tr[i][0] for i in ids]))
    if t["target"]["type"] == "ranking":
        r["rho"] = float(spearmanr([P[i]["point_forecast"] for i in ids], y).correlation)
    return r


def anch(q, a): return 0.5 + 0.5 * (q - a) / (1 - a) if q >= a else 0.5 * q / a


def comp(u, r, p):
    typ = "class" if "acc" in r else ("rank" if "rho" in r else "reg")
    if typ == "reg": q = p["nmae"] / (p["nmae"] + r["mae"]) if (p["nmae"] + r["mae"]) > 0 else 0.5
    elif typ == "rank": q = anch((r["rho"] + 1) / 2, p["anchor"])
    else: q = anch(r["acc"], p["anchor"])
    if "is" not in r or p.get("nis") is None: return q
    iq = p["nis"] / (p["nis"] + r["is"]); return 0.7 * q + 0.3 * min(iq, max(0.5, q))


def calibrate():
    P = {}
    for u, (s20, s30) in OFFICIAL.items():
        r20, r30 = raw(u, REF["D20R"]), raw(u, REF["D30"]); best = None
        if u.startswith(("t4-credit", "t4-macrorev")):
            for a in np.linspace(0.05, 0.95, 181):
                e = abs(comp(u, r20, {"anchor": a}) - s20) + abs(comp(u, r30, {"anchor": a}) - s30)
                if best is None or e < best[0]: best = (e, {"anchor": a, "nis": None})
        else:
            key = "nmae" if "acc" not in r20 and "rho" not in r20 else "anchor"
            grid1 = np.geomspace(max(r20.get("mae", 1) * 0.05, 1e-3), r20.get("mae", 1) * 20, 300) if key == "nmae" else np.linspace(0.05, 0.95, 181)
            for v in grid1:
                for nis in np.geomspace(max(r20["is"] * 0.02, 1e-3), r20["is"] * 50, 300):
                    p = {key: v, "nis": nis}; e = abs(comp(u, r20, p) - s20) + abs(comp(u, r30, p) - s30)
                    if best is None or e < best[0]: best = (e, p)
        P[u] = best[1]; P[u]["fit_err"] = round(best[0], 4)
    return P


if __name__ == "__main__":
    P = calibrate()
    for u, p in P.items(): print(f"  calib {u:32s} " + " ".join(f"{k}={v:.4g}" for k, v in p.items() if v is not None))
    for d in ["D20R", "D30"] + sys.argv[1:]:
        od = REF.get(d, d); tot = 0; line = []
        for u in OFFICIAL:
            s = comp(u, raw(u, od), P[u]); tot += s; line.append(f"{u.split('-')[1][:7]} {s:.3f}")
        print(f"{d[-40:]:40s} sum {tot:.3f} analysis {tot / 10:.4f} LB {-0.27 + 1.27 * tot / 10:.4f} | " + " ".join(line))
