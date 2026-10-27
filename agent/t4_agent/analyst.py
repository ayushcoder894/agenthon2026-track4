"""LLM-first analyst (variant L): the House model reads the evidence and decides every entity's label, point and interval,
and writes the submitted_reasons. The fitted deterministic answer is used only as (a) the reference forecast shown to the
model, (b) a sanity clip for absurd numbers and (c) the fallback for anything that fails validation.

Budget: <= 12 requests per unit (2 votes per entity group with different evidence orders, plus 1 for reasons),
max_tokens 3500, thinking off, fixed temperature, 540 s unit deadline. The deterministic answer is written first by the CLI
and is only replaced by a candidate that passes the full local validation.
"""
from __future__ import annotations

import copy
import math
import os
import statistics

from .answer import _check_reasons, check_claim, det_reasons
from .claims import task_row_claim
from .decide import _claim_from, _Client, _compact_row, _order, _packs
from .predict import predict_all
from .refine import MAX_REQUESTS, Budget, build_reasons, extract_json, _reasons_prompt

GROUP = 4


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def enabled() -> bool:
    return os.environ.get("T4_ANALYST", "0") == "1" and bool(os.environ.get("MODEL_ENDPOINT", "").strip())


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _prompt(unit, group, packs, by_id, preds, v):
    t = unit.task
    target = t.get("target") or {}
    labels = target.get("labels") or []
    asserts = target.get("label_assertions") or {}
    ttype = unit.target_type or "regression"
    lines = [
        "TASK: " + str(t.get("prompt", ""))[:1800],
        f"CUTOFF DATE: {t.get('cutoff_date')}. Use ONLY the task table and the passages below; do not use knowledge of events after the cutoff.",
        f"TARGET: {target.get('name', '')} ({ttype}).",
    ]
    if labels:
        lines.append("LABELS: " + "; ".join(f"{lab} = {str(asserts.get(lab, ''))[:200]}" for lab in labels))
    show_ref = os.environ.get("T4_L_SHOW_REF", "1") == "1"
    if show_ref:
        lines.append("Each entity has a statistical REFERENCE forecast fitted on historical data. Treat it as a prior: keep it unless the "
                     "passages give a concrete reason to move, and then move as far as the evidence justifies.")
    else:
        lines.append("Form your own forecast for each entity from the task table and the passages; be decisive and give your best estimate.")
    lines.append("")
    for row in group:
        eid = row["entity_id"]
        a = by_id[eid]
        p = preds.get(eid)
        ref = (f"label={a['label']}, " if a.get("label") is not None else "") + \
              f"point={a['point_forecast']}, 90% interval=[{a['interval']['lo']}, {a['interval']['hi']}]"
        lines.append(f"ENTITY {eid}: {_compact_row(row)}")
        if show_ref:
            lines.append(f"  REFERENCE: {ref}" + (f" (rule: {p.rule})" if p is not None and p.rule else ""))
        for c in _order(packs[eid], v):
            lines.append(f"  [{c['id']}] {c['text']}")
        lines.append("")
    item = '{"entity_id":"<id>",' + ('"label":"<one label>",' if labels else "") + \
           '"point":<number>,"lo":<number>,"hi":<number>,"confidence":<0..1>,"evidence_ids":["<ids of this entity>"]}'
    lines += [
        'Return ONE JSON object and nothing else: {"forecasts":[' + item + ', ...]} with one object per entity listed above.',
        "Rules: point, lo and hi are in the same units as the reference; [lo, hi] is your 90% interval and must contain point; "
        "confidence is the probability that your label (or the direction of your move from the reference) is right; "
        "evidence_ids must come from that entity's own passage list.",
    ]
    return "\n".join(lines)


def analyze(unit, answer: dict, deadline: float, log=print):
    """Return a candidate answer (dict) or None."""
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
    client = _Client(endpoint, os.environ.get("MODEL_NAME", "").strip() or "default", os.environ.get("MODEL_TOKEN") or None, deadline, log)
    conf_min = _f("T4_L_CONF", 0.55)
    alpha = min(max(_f("T4_L_ALPHA", 1.0), 0.0), 1.0)          # weight on the model's point (1 = model decides)
    width_floor = min(max(_f("T4_L_WIDTH_FLOOR", 0.5), 0.0), 2.0)  # interval half-width >= floor x reference half-width
    new = copy.deepcopy(answer)
    preds = predict_all(unit)
    by_id = {e["entity_id"]: e for e in new["entity_predictions"]}
    packs = _packs(unit, preds, by_id)
    rows = {r["entity_id"]: r for r in unit.entities}
    labels = (unit.task.get("target") or {}).get("labels") or []
    ents = [r for r in unit.entities]
    n_groups = max(1, math.ceil(len(ents) / GROUP))
    votes = 2 if 2 * n_groups + 1 <= MAX_REQUESTS else 1
    if votes * n_groups + 1 > MAX_REQUESTS:  # many entities: bigger groups so the budget still covers everyone once
        n_groups = MAX_REQUESTS - 1
    size = math.ceil(len(ents) / n_groups)
    groups = [ents[i:i + size] for i in range(0, len(ents), size)]
    got = {r["entity_id"]: [] for r in ents}
    failures = 0
    for v in range(votes):
        for g in groups:
            try:
                reply = client.chat(_prompt(unit, g, packs, by_id, preds, v))
            except Exception as exc:
                failures += 1
                log(f"[analyst] request failed ({type(exc).__name__}): {exc}")
                if failures >= 2 or isinstance(exc, Budget):
                    break
                continue
            obj = extract_json(reply)
            items = obj.get("forecasts") if isinstance(obj, dict) else None
            if not isinstance(items, list):
                log("[analyst] reply not parseable")
                continue
            ids = {r["entity_id"] for r in g}
            for it in items:
                if isinstance(it, dict) and it.get("entity_id") in ids:
                    got[it["entity_id"]].append(it)
        if failures >= 2:
            break
    changed, chosen = 0, {}
    for eid, items in got.items():
        e = by_id[eid]
        ref_pt, ref_lo, ref_hi = e["point_forecast"], e["interval"]["lo"], e["interval"]["hi"]
        ref_half = max((ref_hi - ref_lo) / 2.0, 1e-9)
        pts = [float(i["point"]) for i in items if _num(i.get("point"))]
        if not items or not pts:
            continue
        # label: all votes agree, confident enough, and in the vocabulary
        if labels:
            labs = [i.get("label") for i in items if i.get("label") in labels]
            confs = [float(i["confidence"]) for i in items if _num(i.get("confidence")) and 0 <= float(i["confidence"]) <= 1]
            unanimous = os.environ.get("T4_L_UNANIMOUS", "1") == "1"
            if labs and (statistics.mean(confs) if confs else 0) >= conf_min:
                top, cnt = statistics.mode(labs), labs.count(statistics.mode(labs))
                if (len(set(labs)) == 1 and len(labs) == len(items)) or (not unanimous and cnt * 2 >= len(labs)):
                    e["label"] = top
        # point: the model's estimate, clipped to a generous window around the reference (guards against unit mistakes)
        llm_pt = statistics.median(pts)
        lo_clip, hi_clip = ref_lo - 3 * ref_half, ref_hi + 3 * ref_half
        llm_pt = min(max(llm_pt, lo_clip), hi_clip)
        pt = ref_pt + alpha * (llm_pt - ref_pt)
        # interval: the model's width, never narrower than width_floor x the calibrated reference width, centred on the point
        halves = [(float(i["hi"]) - float(i["lo"])) / 2.0 for i in items if _num(i.get("lo")) and _num(i.get("hi")) and float(i["hi"]) >= float(i["lo"])]
        half = max(statistics.median(halves) if halves else ref_half, width_floor * ref_half)
        e["point_forecast"] = round(pt, 6)
        e["interval"]["lo"], e["interval"]["hi"] = round(pt - half, 6), round(pt + half, 6)
        # cite the passage the model relied on first (when it is a valid claim for this entity)
        valid = {c["id"]: c for c in packs[eid]}
        for i in items:
            for x in (i.get("evidence_ids") or []) if isinstance(i.get("evidence_ids"), list) else []:
                c = valid.get(x)
                if c and check_claim(unit, eid, _claim_from(c)) is None:
                    chosen[eid] = c
                    break
            if eid in chosen:
                break
        if eid in chosen:
            tr = task_row_claim(unit, rows[eid])
            e["claims"] = [_claim_from(chosen[eid]), {"doc_id": tr[0], "span_start": tr[1], "span_end": tr[2], "claim": tr[3]}]
        changed += 1
    log(f"[analyst] entities updated: {changed}/{len(ents)} | requests {client.used}")
    if not changed:
        return None
    # ranking units: ranks follow the model's points
    if unit.target_type == "ranking":
        ranked = sorted(unit.roster, key=lambda eid: (-by_id[eid]["point_forecast"], unit.roster.index(eid)))
        for k, eid in enumerate(ranked, 1):
            by_id[eid]["rank"] = k
    # reasons written by the model (grounded in the chosen passages), else the deterministic ones
    reasons = []
    if chosen:
        try:
            raw = extract_json(client.chat(_reasons_prompt(unit, new, chosen)))
            reasons = build_reasons(unit, new, {c["id"]: c for c in chosen.values()}, raw)
            if reasons and _check_reasons(unit, reasons):
                reasons = []
        except Exception as exc:
            log(f"[analyst] reasons request failed ({type(exc).__name__})")
            reasons = []
    if not reasons:
        try:
            reasons = det_reasons(unit, new, preds)
            if reasons and _check_reasons(unit, reasons):
                reasons = []
        except Exception:
            reasons = []
    if reasons:
        new["submitted_reasons"] = reasons
    else:
        new.pop("submitted_reasons", None)
    return new
