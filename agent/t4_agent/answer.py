"""Assemble answer.json and validate it locally with the scorer's structural rules (stdlib only)."""
from __future__ import annotations

import json
import math
import os
import re

from .claims import build_claims, evidence_claim, support_numbers, task_row_claim
from .predict import Pred, predict_all
from .unit_io import TASK_DOC_ID

MAX_CLAIM_CHARS = 4000
MAX_SPAN_CHARS = 8000


def _finite(*vals):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals)


def check_claim(unit, eid: str, claim: dict) -> str | None:
    """Reason a claim would be refused or false under the deterministic rules, else None."""
    doc_id, s, e, text = claim.get("doc_id"), claim.get("span_start"), claim.get("span_end"), claim.get("claim")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CLAIM_CHARS:
        return "malformed claim text"
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (s, e)) or s < 0 or e <= s:
        return "bad offsets"
    if e - s > MAX_SPAN_CHARS:
        return "span over cap"
    if doc_id == TASK_DOC_ID:
        lo, hi = unit.task_rows.get(eid, (None, None))
        if lo is None or not (lo <= s < e <= hi):
            return "task span outside own row"
        src = unit.task_text
    else:
        doc = unit.docs.get(doc_id)
        if doc is None:
            return "unresolved doc"
        if not doc.admits(eid):
            return "wrong entity"
        if unit.cutoff is not None and doc.doc_date > unit.cutoff:
            return "post-cutoff"
        src = doc.text
    if e > len(src):
        return "span out of range"
    if src[s:e] != text:
        return "claim is not a verbatim quote of its span"
    return None


def build_answer(unit) -> dict:
    preds = predict_all(unit)
    ranking = unit.target_type == "ranking"
    order = {}
    if ranking:
        ranked = sorted(unit.roster, key=lambda eid: (-preds[eid].point, unit.roster.index(eid)))
        order = {eid: i + 1 for i, eid in enumerate(ranked)}
    entity_predictions = []
    for row in unit.entities:
        eid = row["entity_id"]
        p: Pred = preds[eid]
        claims = [c for c in build_claims(unit, row, numbers=list(support_numbers(row, p)) + list(p.extra_numbers), terms=p.terms) if check_claim(unit, eid, c) is None]
        if p.evidence:
            ev = evidence_claim(unit, eid, p.evidence)
            if ev is not None and check_claim(unit, eid, ev) is None:
                claims = [ev] + [c for c in claims if c["doc_id"] == TASK_DOC_ID][:1]
        if not claims:
            doc_id, s, e, seg = task_row_claim(unit, row)
            claims = [{"doc_id": doc_id, "span_start": s, "span_end": e, "claim": seg}]
        entry = {
            "entity_id": eid,
            "point_forecast": float(p.point),
            "interval": {"level": unit.interval_level, "lo": float(p.lo), "hi": float(p.hi)},
            "claims": claims,
        }
        if p.label is not None:
            entry["label"] = p.label
        if ranking:
            entry["rank"] = order[eid]
        entity_predictions.append(entry)
    answer = {"task_id": unit.task.get("task_id"), "entity_predictions": entity_predictions}
    if os.environ.get("T4_DET_REASONS", "1") == "1":
        try:
            reasons = det_reasons(unit, answer, preds)
            if reasons and not _check_reasons(unit, reasons):
                answer["submitted_reasons"] = reasons
        except Exception:
            pass  # reasons are optional; never let them damage the answer
    return answer


def det_reasons(unit, answer: dict, preds: dict, limit: int = 3) -> list:
    """Deterministic submitted_reasons (no model): up to `limit` reasons on distinct entities.

    premise  = the verbatim corpus span already cited as the entity's first claim (entity-bound, pre-cutoff);
    mechanism= the family's number-free rule description;
    implication = the SUBMITTED label/point/interval for that entity (so every figure is either in the
    cited premise or among the submitted values).  Entities whose cited span quotes a supporting figure go first.
    """
    rows = {r["entity_id"]: r for r in unit.entities}
    cands = []
    for e in answer["entity_predictions"]:
        eid = e["entity_id"]
        corp = [c for c in e["claims"] if c["doc_id"] != TASK_DOC_ID]
        if not corp:
            continue
        c = corp[0]
        p = preds[eid]
        nums = support_numbers(rows[eid], p)
        hit = any(re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d)", c["claim"]) for n in nums)
        cands.append((0 if hit else 1, unit.roster.index(eid), eid, c, p))
    cands.sort(key=lambda t: (t[0], t[1]))
    if os.environ.get("T4_GROUP_REASONS", "1") == "1":
        try:
            grouped = _grouped_reasons(unit, answer, cands, limit)
            if grouped and not _check_reasons(unit, grouped):
                return grouped
        except Exception:
            pass
    out, seen = [], set()
    for _, _, eid, c, p in cands:
        if c["claim"] in seen:
            continue
        seen.add(c["claim"])
        entry = next(x for x in answer["entity_predictions"] if x["entity_id"] == eid)
        iv = entry["interval"]
        what = (f"label {entry['label']}, " if entry.get("label") is not None else "") + f"point forecast {entry['point_forecast']}"
        rule = (p.rule or "a nearby prior-value column is carried forward").strip()
        out.append({
            "reason_id": f"r{len(out) + 1}",
            "premise": c["claim"],
            "mechanism": f"The cited passage gives the recent level for {eid}; under the rule used here, {rule}.",
            "answer_implication": f"For {eid} the submitted answer is {what} with a 90% interval from {iv['lo']} to {iv['hi']}.",
            "scope": {"entities": [eid]},
            "citations": [{"doc_id": c["doc_id"], "span_start": c["span_start"], "span_end": c["span_end"]}],
        })
        if len(out) >= limit:
            break
    return out


def _direction(unit, entry, row) -> str:
    """The submitted answer's group: its label, or the sign of the move from the row's own prior value."""
    if entry.get("label") is not None:
        return str(entry["label"])
    if unit.target_type == "ranking" and entry.get("rank") is not None:
        n = len(unit.roster)
        return "top half" if entry["rank"] <= (n + 1) // 2 else "bottom half"
    pt = entry["point_forecast"]
    base = next((row[k] for k in row if isinstance(row[k], (int, float)) and not isinstance(row[k], bool)
                 and re.search(r"prior|latest|last|start", k) and "yield" not in k), None)
    # same-scale prior value (e.g. latest estimate, last print) -> move relative to it; else the point is itself a change
    is_change = unit.target_type == "regression" and re.search(
        r"change|growth|revision|return|move", str((unit.task.get("target") or {}).get("name", "")) + str(unit.task.get("prompt", ""))[:400], re.I)
    if is_change:
        ref = 0.0  # the point is itself a change (growth %, bps move, MoM %)
    elif base is not None and abs(pt) <= 3 * abs(float(base)):
        ref = float(base)
    else:
        return "point"
    return "increase" if pt > ref else ("decrease" if pt < ref else "no change")


def _grouped_reasons(unit, answer: dict, cands, limit: int) -> list:
    """Up to `limit` reasons, one per group of entities sharing the submitted direction/label; each cites up to 4 entity
    passages, states the method with each entity's computed figures (the derivation), and lists the submitted values."""
    rows = {r["entity_id"]: r for r in unit.entities}
    entries = {e["entity_id"]: e for e in answer["entity_predictions"]}
    groups = {}
    for hit, _, eid, c, p in cands:
        low = c["claim"].lower()
        topical = sum(1 for t in (p.terms or ()) if t in low) + 2 * (1 - hit)  # family phrases + quotes a backing figure
        groups.setdefault(_direction(unit, entries[eid], rows[eid]), []).append((topical, eid, c, p))
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:limit]
    out = []
    for gname, items in order:
        items = [(eid, c, p) for _, eid, c, p in sorted(items, key=lambda x: -x[0])]  # most topical passages are cited first
        members = sorted({eid for eid, _, _ in items}, key=unit.roster.index)
        cited, seen_txt = [], set()
        for eid, c, p in items:
            if c["claim"] not in seen_txt and len(cited) < 4:
                seen_txt.add(c["claim"])
                cited.append((eid, c, p))
        premise = " ".join(f"{eid}: {c['claim'].strip()}" for eid, c, _ in cited[:3])
        if len(premise) > 1200:
            premise = premise[:1200].rsplit(" ", 1)[0]
        rule = (cited[0][2].rule or "a nearby prior-value column is carried forward").strip().rstrip(".")
        figs = "; ".join(f"{eid}: {p.why}" for eid, _, p in cited[:4] if getattr(p, "why", ""))
        mech = f"Method: {rule}." + (f" Applied to the cited figures - {figs}." if figs else "")
        if len(mech) > 1400:
            mech = mech[:1400].rsplit(";", 1)[0] + "."
        vals = []
        for eid in members[:8]:
            e = entries[eid]
            vals.append(f"{eid} " + (f"{e['label']} " if e.get("label") is not None else "")
                        + f"(point {e['point_forecast']}, 90% interval {e['interval']['lo']} to {e['interval']['hi']})")
        what = "" if gname == "point" else f"{gname} "
        impl = (f"Supports the submitted {what}answers for {', '.join(members[:8])}" + (" and others" if len(members) > 8 else "")
                + ": " + "; ".join(vals) + ".")
        out.append({
            "reason_id": f"r{len(out) + 1}",
            "premise": premise,
            "mechanism": mech,
            "answer_implication": impl,
            "scope": {"entities": members},
            "citations": [{"doc_id": c["doc_id"], "span_start": c["span_start"], "span_end": c["span_end"]} for _, c, _ in cited],
        })
    total = sum(len(json.dumps({k: r[k] for k in ("reason_id", "premise", "mechanism", "answer_implication")}).encode()) for r in out)
    while out and total > 6300:
        out.pop()
        total = sum(len(json.dumps({k: r[k] for k in ("reason_id", "premise", "mechanism", "answer_implication")}).encode()) for r in out)
    return out


def validate_answer(unit, answer: dict) -> list:
    """Return a list of problems (empty = locally valid)."""
    problems = []
    if answer.get("task_id") != unit.task.get("task_id"):
        problems.append("task_id mismatch")
    eps = answer.get("entity_predictions")
    if not isinstance(eps, list):
        return problems + ["entity_predictions missing"]
    ids = [e.get("entity_id") for e in eps if isinstance(e, dict)]
    if sorted(ids) != sorted(unit.roster) or len(set(ids)) != len(ids):
        problems.append("roster mismatch")
    labels = (unit.task.get("target") or {}).get("labels")
    ranks = []
    for e in eps:
        eid = e.get("entity_id")
        iv = e.get("interval") or {}
        if not _finite(iv.get("lo"), iv.get("hi")) or iv["lo"] > iv["hi"]:
            problems.append(f"{eid}: bad interval")
        if iv.get("level") != unit.interval_level:
            problems.append(f"{eid}: interval level")
        if unit.target_type in ("regression", "ranking") and not _finite(e.get("point_forecast")):
            problems.append(f"{eid}: point_forecast missing")
        if "point_forecast" in e and not _finite(e["point_forecast"]):
            problems.append(f"{eid}: point_forecast not finite")
        if unit.target_type == "classification" and labels and e.get("label") not in labels:
            problems.append(f"{eid}: label not in vocabulary")
        cl = e.get("claims")
        if not isinstance(cl, list) or not cl:
            problems.append(f"{eid}: no claims")
        else:
            for c in cl:
                why = check_claim(unit, eid, c)
                if why:
                    problems.append(f"{eid}: {why}")
        if "rank" in e:
            ranks.append(e["rank"])
    if ranks and sorted(ranks) != list(range(1, len(unit.roster) + 1)):
        problems.append("rank is not a permutation")
    if "submitted_reasons" in answer:
        problems += _check_reasons(unit, answer["submitted_reasons"])
    return problems


_DENY = ("leaderboard", "canary", "/home/", "units/", "reference/", "outcome.json", "team_id", "team name",
         "participant_id", "participant name", "submission_id", "other submission")


def _check_reasons(unit, reasons) -> list:
    """Schema shape (1-3 reasons, required strings, integer citations), deny list, byte caps."""
    if not isinstance(reasons, list) or not 1 <= len(reasons) <= 3:
        return ["submitted_reasons must be a list of 1-3"]
    problems, total = [], 0
    for i, r in enumerate(reasons):
        if not isinstance(r, dict):
            problems.append(f"reason {i} not an object")
            continue
        for key in ("reason_id", "premise", "mechanism", "answer_implication"):
            if not isinstance(r.get(key), str) or not r[key].strip():
                problems.append(f"reason {i} missing {key}")
        for key in ("mechanism", "answer_implication"):
            low = str(r.get(key, "")).lower()
            if any(d in low for d in _DENY):
                problems.append(f"reason {i} {key} hits deny list")
        low_p = str(r.get("premise", "")).lower()
        if any(d in low_p for d in _DENY):
            doc_texts = [d.text.lower() for d in unit.docs.values()]
            if not any(low_p in t for t in doc_texts):  # a verbatim corpus quote is exempt
                problems.append(f"reason {i} premise hits deny list")
        scope = r.get("scope")
        if scope is not None:
            ents = scope.get("entities") if isinstance(scope, dict) else None
            if not isinstance(ents, list) or not all(isinstance(x, str) for x in ents):
                problems.append(f"reason {i} bad scope")
        for c in r.get("citations") or []:
            if not (isinstance(c, dict) and isinstance(c.get("doc_id"), str)
                    and all(isinstance(c.get(k), int) and not isinstance(c.get(k), bool) and c[k] >= 0
                            for k in ("span_start", "span_end"))):
                problems.append(f"reason {i} bad citation")
            else:
                doc = unit.docs.get(c["doc_id"])
                if doc is None or c["span_end"] > len(doc.text) or c["span_end"] - c["span_start"] > MAX_SPAN_CHARS:
                    problems.append(f"reason {i} citation does not resolve")
        total += len(json.dumps({k: r.get(k) for k in ("reason_id", "premise", "mechanism", "answer_implication")}).encode())
    if total > 6500:
        problems.append("reasons over 6500 bytes")
    return problems


def minimal_answer(unit) -> dict:
    """Last-resort answer from the task alone: roster, naive-ish values, one task-row claim each."""
    labels = (unit.task.get("target") or {}).get("labels")
    eps = []
    for i, row in enumerate(unit.entities):
        doc_id, s, e, seg = task_row_claim(unit, row)
        entry = {
            "entity_id": row["entity_id"],
            "point_forecast": 0.0,
            "interval": {"level": unit.interval_level, "lo": -1.0, "hi": 1.0},
            "claims": [{"doc_id": doc_id, "span_start": s, "span_end": e, "claim": seg}],
        }
        if unit.target_type == "classification" and labels:
            entry["label"] = labels[0]
        if unit.target_type == "ranking":
            entry["rank"] = i + 1
        eps.append(entry)
    return {"task_id": unit.task.get("task_id"), "entity_predictions": eps}
