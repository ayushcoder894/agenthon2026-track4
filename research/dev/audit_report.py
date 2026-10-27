"""Row-level D20R diagnostic report (diagnosis only). Reads audit_rows.csv (written by audit_rows.py). Uses simple documented stand-in baselines (the organizers' naive answers are hidden).
Classifies each row: WRONG_LABEL, LARGE_NUM_ERROR (|err| > 1.5x the stand-in baseline error), TOO_WIDE (covered but width > 4|err|), INTERVAL_MISS (outcome outside the interval)."""
import json, re
import numpy as np, pandas as pd

R = "t4-local/exp_out/d20_audit"; U = "track4-analysis-public/units/"
d = pd.read_csv(R + "/audit_rows.csv")
ALPHA = 0.1


def task(u): return json.load(open(U + u + "/task.json"))


# ---- stand-in baselines
base = {}
u = "t4-auction-btc-202411-us7"
for e in task(u)["entities"]:
    txt = json.load(open(U + u + f"/corpus/TDIRECT_AUCTIONS_{e['tenor'].split('-')[0]}Y_20241031.json"))["text"]
    vals = [float(m.group(1)) for m in re.finditer(r"\|\s*(?:new|reopening)\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|", txt)]
    base[(u, e["entity_id"])] = vals[-1] if vals else np.nan                      # last auction carried forward
for e in task("t4-cpicomp-202410-us11")["entities"]: base[("t4-cpicomp-202410-us11", e["entity_id"])] = e["latest_published_mom_pct"]
for uu in ("t4-eps-growth-2024Q3-banks", "t4-fomc-curve-20220728", "t4-fomc-curve-20240918", "t4-cotpos-202411-us10"):
    for e in task(uu)["entities"]: base[(uu, e["entity_id"])] = 0.0
d["base"] = [base.get((r.unit, r.entity), np.nan) for r in d.itertuples()]

# ---- derive true label / numeric truth by unit
def true_label(r):
    if r.unit == "t4-credit-event-2023": return "credit_event" if r.outcome == 1 else "no_event"
    if r.unit == "t4-eps-yoy-2023Q2-mixed": return "up" if r.outcome > r.prior else "down"
    if r.unit == "t4-macrorev-20240930-us6": return "up" if r.outcome > r.prior else ("down" if r.outcome < r.prior else "same")
    if r.unit == "t4-postearn-20240201-megacap": return "positive_reaction" if r.outcome > 1 else ("negative_reaction" if r.outcome < -1 else "flat")
    return None
d["true_label"] = [true_label(r) if pd.notna(r.outcome) else None for r in d.itertuples()]
# numeric truth: for eps-yoy the interval/point refer to the EPS level; macrorev interval refers to the level; others: outcome itself
d["y"] = d.outcome
d["err"] = (d.point - d.y).abs()
d["berr"] = (d.base - d.y).abs()
d["width"] = d.hi - d.lo
d["covered"] = (d.y >= d.lo) & (d.y <= d.hi)
d["IS"] = d.width + (2 / ALPHA) * (np.maximum(d.lo - d.y, 0) + np.maximum(d.y - d.hi, 0))
d["wrong_label"] = [(r.label != r.true_label) if (pd.notna(r.label) and r.true_label is not None) else np.nan for r in d.itertuples()]
d["flag"] = ""
for i, r in d.iterrows():
    if pd.isna(r.y): d.at[i, "flag"] = "PENDING"; continue
    f = []
    if r.wrong_label is True or r.wrong_label == 1: f.append("WRONG_LABEL")
    if pd.notna(r.berr) and r.err > 1.5 * r.berr and r.err > 0: f.append("LARGE_NUM_ERROR")
    if not r.covered: f.append("INTERVAL_MISS")
    elif r.width > 4 * max(r.err, 1e-9) and r.unit not in ("t4-credit-event-2023",): f.append("TOO_WIDE")
    d.at[i, "flag"] = ",".join(f) or "ok"
d.to_csv(R + "/audit_rows_flagged.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
lines = ["# D20R row-level diagnostic (diagnosis only)\n",
         "Outcomes: first-published values from public primary sources (see outcomes/provenance.json). Practice-unit exception per Issue #24: in-sample, not a basis for tuning to these rows.\n"]
for u, g in d.groupby("unit", sort=False):
    g = g.copy(); k = g.y.notna()
    lines.append(f"\n## {u}  (n={len(g)}, outcomes available: {int(k.sum())})")
    if k.sum() == 0: lines.append("outcome pending"); continue
    gg = g[k]
    if gg.label.notna().any():
        lc = gg.dropna(subset=["wrong_label"]); lines.append(f"- label accuracy {int((lc.wrong_label == 0).sum())}/{len(lc)}; wrong: {', '.join(lc[lc.wrong_label == 1].entity)}")
    lines.append(f"- point MAE {gg.err.mean():.3f} vs stand-in baseline MAE {gg.berr.mean():.3f}" if gg.berr.notna().any() else f"- point MAE {gg.err.mean():.3f}")
    lines.append(f"- interval coverage {gg.covered.mean():.2f} (target 0.90); mean width {gg.width.mean():.3f}; mean interval score {gg.IS.mean():.3f}")
    if u == "t4-cotpos-202411-us10":
        lines.append(f"- rank correlation (Spearman) point vs outcome: {gg.point.rank().corr(gg.y.rank()):+.2f}")
    if u == "t4-credit-event-2023":
        lines.append(f"- Brier score of the stated probabilities {np.mean((gg.point - gg.y) ** 2):.3f} (all-0.5 gives 0.25)")
    lines.append(gg[["entity", "label", "true_label", "point", "lo", "hi", "y", "flag"]].round(3).to_string(index=False))
tot = d[d.y.notna()]
lines.append("\n## Failure-type counts (rows with outcomes)")
for fl in ("WRONG_LABEL", "LARGE_NUM_ERROR", "INTERVAL_MISS", "TOO_WIDE"):
    m = tot.flag.str.contains(fl); lines.append(f"- {fl}: {int(m.sum())} rows of {len(tot)}  ({', '.join(sorted(set(tot[m].unit.str.replace('t4-', '').str.split('-').str[0])))})")
open(R + "/audit_report.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
