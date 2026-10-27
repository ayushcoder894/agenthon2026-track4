"""Model-written submitted_reasons on top of a FINAL deterministic answer (variant D5R / D7R).

Predictions are never changed here: the model only explains the submitted answer, grounded in verbatim evidence passages.

Reasoning grading compares the reasons with the unit's hidden target reasons (coverage over ALL target reasons, grounding,
inferential link, answer consistency). Target reasons follow the drivers the task statement names ("Reason from ..."), so:
  * the drivers are read from the task statement and each gets its own pool of passages, searched across ALL entities;
  * each entity's driving passage (the claim the deterministic answer cites first) is offered too, with the rule's own
    computed figures (`why`), so the mechanism can show the derivation (growth %, fitted rates);
  * each reason covers ONE driver, may name several entities and cite several passages (the judge reads the cited text);
  * the premise is a verbatim passage unless the model's premise only uses figures found in the cited passages.
One request (two at most if the first reply is unusable); any failure keeps the deterministic reasons. The Development
leaderboard does not score reasons; the reasoning bonus (final = ... + 0.25 x reasoning) is graded on the Final only.
"""
from __future__ import annotations

import json
import os
import re

from .answer import _check_reasons
from .claims import top_candidates
from .decide import _Client
from .predict import predict_all
from .refine import _clean_text, extract_json
from .unit_io import TASK_DOC_ID

MAX_DRIVERS = 6
PER_DRIVER = 4
MAX_CITES = 4
MAX_ENTITY_LINES = 14

_STOPW = {"from", "each", "with", "that", "this", "their", "own", "the", "and", "visible", "level", "into", "under", "over",
          "versus", "much", "already", "what", "fall", "within", "inside", "window", "sets", "while", "most", "more", "pre-cutoff",
          "signals", "dynamics", "history", "schedules", "typical", "how", "far", "sit", "below", "new", "trend", "trends",
          "corpus", "driver", "drivers", "regime", "going", "component-level"}
# family fallbacks when the statement names no drivers (or names them tersely)
_FAMILY_DRIVERS = {
    "_eps_direction": ["diluted earnings per share in the latest quarter versus the prior year", "net income trend", "revenue trend"],
    "_eps_growth": ["diluted earnings per share year-to-date growth", "net interest income", "provision for credit losses"],
    "_postearn": ["latest quarterly results and guidance", "revenue growth", "margin trend"],
    "_credit": ["going concern substantial doubt", "net loss and stockholders deficit", "debt maturities and liquidity"],
    "_cot": ["noncommercial net position level", "change in net position over recent weeks", "open interest",
             "scheduled events in the window"],
    "_macrorev": ["revision history", "benchmark revision", "annual update"],
}


def enabled() -> bool:
    return os.environ.get("T4_LLM_REASONS", "0") == "1" and bool(os.environ.get("MODEL_ENDPOINT", "").strip())


def drivers_from_prompt(prompt: str) -> list:
    """The driver phrases a task statement asks the analysis to reason from (empty when it names none)."""
    m = re.search(r"Reason from (.+?)(?:\.\s+(?:For each|Support|Do NOT|Each|Label)|\.\s*$)", prompt or "", re.S)
    if not m:
        return []
    body = m.group(1)
    body = re.sub(r"\([^)]*\)", lambda x: ", " + x.group(0)[1:-1] + ",", body)  # parenthesised lists are drivers too
    parts = re.split(r",|;|\s[-–—]\s|–|—|\band\b(?= (?:the|from|what|how|position|sensitivity))", body)
    out = []
    for p in parts:
        p = " ".join(p.split()).strip(" .:-")
        p = p.split(":", 1)[1].strip() if ":" in p else p  # "pre-cutoff regime: X" -> "X"
        p = re.sub(r"^(and|the|from|plus|while)\s+", "", p, flags=re.I)
        if not _keywords(p):
            continue
        if len(p) >= 6 and p.lower() not in (x.lower() for x in out):
            out.append(p)
    return out[:MAX_DRIVERS]


def _keywords(phrase: str) -> tuple:
    words = [w for w in re.findall(r"[a-z][a-z\-]{3,}", phrase.lower()) if w not in _STOPW]
    stems = []
    for w in words:
        if w.endswith("ies") and len(w) > 5:
            s = w[:-3]
        elif w.endswith(("sses", "ss", "us")):
            s = w[:-2] if w.endswith("sses") else w
        elif w.endswith("es") and len(w) > 5:
            s = w[:-2]
        elif w.endswith("s") and len(w) > 4:
            s = w[:-1]
        else:
            s = w
        if s not in stems:
            stems.append(s)
    return tuple(stems[:5])


def _answer_text(e) -> str:
    return (f"label={e['label']}, " if e.get("label") is not None else "") + \
        f"point={e['point_forecast']}, 90% interval=[{e['interval']['lo']}, {e['interval']['hi']}]" + \
        (f", rank={e['rank']}" if e.get("rank") is not None else "")


def _hybrid_pools(unit, drivers, seen, n, deadline):
    """D20: per-driver pools from the hybrid BM25 + NeMo Retriever index (RRF + MMR, spread across entities).
    Returns (pools, n) or None to fall back to the keyword pools (no index, nothing found, any error)."""
    if os.environ.get("T4_HYBRID", "1") == "0":
        return None
    try:
        import time as _t

        from . import retrieval as _rt

        idx = _rt.index_for(unit, min(deadline - 150, _t.time() + 240) if deadline else _t.time() + 240)
        if idx is None or not idx.items or not idx.dense:
            return None  # no encoder: the keyword pools (BM25-only pools were worse in a spot check)
        target = str((unit.task.get("target") or {}).get("name", "")).replace("_", " ")
        pools, total = [], 0
        for d in drivers:
            order, _ = _rt.rank_dense(idx, d, eligible=lambda it: (it[1], it[2]) not in seen, top=40)
            lst = []
            for i in idx.mmr(order, PER_DRIVER):
                eid, doc_id, s0, e0, text = idx.items[i]
                n += 1
                lst.append({"id": f"p{n}", "entity_id": eid, "doc_id": doc_id, "s": s0, "e": e0, "text": text})
                seen.add((doc_id, s0))
            total += len(lst)
            pools.append((d, lst))
        if total == 0:
            return None
        _hybrid_pools.dense = idx.dense
        return pools, n
    except Exception:
        return None


def _pools(unit, answer, preds, deadline=None):
    """Driver -> list of passages ({id, entity_id, doc_id, s, e, text}); plus each entity's driving passage."""
    t = unit.task
    fams = {getattr(p, "fn", "") for p in preds.values()}
    if unit.target_type == "ranking":
        fams.add("_cot")
    drivers = drivers_from_prompt(str(t.get("prompt", "")))
    if len(drivers) < 2:
        for f in fams:
            drivers += [d for d in _FAMILY_DRIVERS.get(f, []) if d not in drivers]
    drivers = drivers[:MAX_DRIVERS]
    n = 0
    seen = set()
    by_id = {e["entity_id"]: e for e in answer["entity_predictions"]}
    lead = {}
    for row in unit.entities:  # the passage each prediction already rests on
        eid = row["entity_id"]
        corp = [c for c in by_id[eid]["claims"] if c["doc_id"] != TASK_DOC_ID]
        if corp:
            c = corp[0]
            n += 1
            lead[eid] = {"id": f"p{n}", "entity_id": eid, "doc_id": c["doc_id"], "s": c["span_start"], "e": c["span_end"], "text": c["claim"]}
            seen.add((c["doc_id"], c["span_start"]))
    pools = []
    hyb = _hybrid_pools(unit, drivers, seen, n, deadline)
    if hyb is not None:
        pools, n = hyb
        return drivers, pools, lead
    for d in drivers:
        kw = _keywords(d)
        if not kw:
            continue
        found = []
        for row in unit.entities:
            for c in top_candidates(unit, row, 2, terms=kw):
                low = c["text"].lower()
                hits = sum(1 for k in kw if k in low)
                if hits and (c["doc_id"], c["s"]) not in seen:
                    found.append((hits, c["score"], row["entity_id"], c))
        found.sort(key=lambda x: (-x[0], -x[1]))
        lst, ents_used = [], set()
        for hits, _, eid, c in found:  # spread the pool across entities first
            if eid in ents_used and len(ents_used) < len({f[2] for f in found}):
                continue
            ents_used.add(eid)
            n += 1
            lst.append({"id": f"p{n}", "entity_id": eid, "doc_id": c["doc_id"], "s": c["s"], "e": c["e"], "text": c["text"]})
            seen.add((c["doc_id"], c["s"]))
            if len(lst) >= PER_DRIVER:
                break
        pools.append((d, lst))
    return drivers, pools, lead


def _prompt(unit, answer, preds, pools, lead):
    t = unit.task
    lines = [
        "TASK: " + str(t.get("prompt", ""))[:1800],
        f"CUTOFF DATE: {t.get('cutoff_date')}.",
        "",
        "SUBMITTED ANSWERS (FINAL - explain them, never argue for a different answer):",
    ]
    rows = {r["entity_id"]: r for r in unit.entities}
    for e in answer["entity_predictions"][:MAX_ENTITY_LINES]:
        eid = e["entity_id"]
        p = preds.get(eid)
        name = rows[eid].get("name") or ""
        basis = (getattr(p, "why", "") or "")[:220] if p is not None else ""
        from .answer import _direction

        g = _direction(unit, e, rows[eid])
        dtxt = "" if g == "point" or e.get("label") is not None else f" [submitted direction: {g}]"
        lines.append(f"- {eid}{' (' + name + ')' if name and name != eid else ''}: {_answer_text(e)}{dtxt}" + (f" | basis: {basis}" if basis else ""))
    rule = next((getattr(p, "rule", "") for p in preds.values() if getattr(p, "rule", "")), "")
    if rule:
        lines.append(f"Method behind the answers: {rule[:400]}")
    lines += ["", "EVIDENCE PASSAGES each prediction rests on:"]
    for eid, c in list(lead.items())[:MAX_ENTITY_LINES]:
        lines.append(f"  [{c['id']}] ({eid}) {c['text']}")
    for d, lst in pools:
        if lst:
            lines.append(f"DRIVER '{d}':")
            for c in lst:
                lines.append(f"  [{c['id']}] ({c['entity_id']}) {c['text']}")
    lines += [
        "",
        "Write exactly 3 reasons (fewer only if the passages cannot support 3), each about a DIFFERENT driver the task tells you to "
        "reason from, choosing the drivers that matter most for these answers. Each reason:",
        "  premise: the evidence fact, stated plainly with the figures exactly as the cited passages give them (or a task-table value);",
        "  mechanism: why that fact moves the outcome, including any calculation from the cited figures (e.g. growth = 1.25 / 1.10 - 1 = 13.6%);",
        "  implication: what it means for the named entities' SUBMITTED answers (state their label or point), consistent with them: "
        "never call an entity's outcome higher/lower/positive/negative unless that matches its submitted label or direction; if a "
        "factor points the other way, say it is outweighed or that it widens the interval;",
        "  evidence_ids: 1 to 4 passage ids that state the premise; entities: the entity ids the reason applies to.",
        'Return ONE JSON object and nothing else: {"reasons":[{"driver":"...","premise":"...","mechanism":"<=80 words",'
        '"implication":"<=50 words","evidence_ids":["p.."],"entities":["..."]}]}',
        "Do not mention files, paths, scores, rankings of teams or the evaluation. Do not quote numbers that are not in the cited "
        "passages, the task table or the submitted answers.",
    ]
    return "\n".join(lines)


def _nums(s: str, signed: bool = False) -> set:
    if signed:
        return {(m.group(1) or "") + (m.group(2).replace(",", "").lstrip("0") or "0")
                for m in re.finditer(r"(?<![\w.])(-|−)?(\d[\d,]*(?:\.\d+)?)", s)}
    return {x.replace(",", "").lstrip("0") or "0" for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s)}


_UPW = r"\b(increas\w*|ris(e|es|ing)|higher|climb\w*|up|positive|above)\b"
_DOWNW = r"\b(decreas\w*|declin\w*|fall\w*|lower|drop\w*|down|negative|below)\b"


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.15, 0.02 * abs(b))


def _mechanism_ok(mech: str, src_nums: set) -> bool:
    """Every figure in the mechanism is in the sources, a small count/year, or the correct result of a ratio of source figures
    written out as 'X% (a / b - 1)' or 'a / b = X'."""
    ok = set()
    for m in re.finditer(r"(-?\d[\d,]*(?:\.\d+)?)\s*%?\s*\(\s*(\d[\d,]*(?:\.\d+)?)\s*/\s*(\d[\d,]*(?:\.\d+)?)\s*-\s*1\s*\)", mech):
        res, a, b = (float(x.replace(",", "")) for x in m.groups())
        if b == 0 or {_nums(m.group(2)).pop(), _nums(m.group(3)).pop()} - src_nums or not _close(res, 100 * (a / b - 1)):
            return False
        ok |= _nums(m.group(0))
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*/\s*(\d[\d,]*(?:\.\d+)?)\s*-\s*1\s*=\s*(-?\d[\d,]*(?:\.\d+)?)\s*%?", mech):
        a, b, res = (float(x.replace(",", "")) for x in m.groups())
        if b == 0 or _nums(m.group(1) + " " + m.group(2)) - src_nums or not _close(res, 100 * (a / b - 1)):
            return False
        ok |= _nums(m.group(0))
    for n in _nums(mech) - ok - src_nums:
        try:
            v = float(n)
        except ValueError:
            return False
        if not (v == int(v) and (v <= 12 or 1990 <= v <= 2035)):
            return False
    return True
_SYN = {"up": _UPW, "down": _DOWNW, "beat": r"\bbeat\w*", "miss": r"\bmiss\w*", "inline": r"\bin[- ]?line\b",
        "positive reaction": r"\bpositive\b", "negative reaction": r"\bnegative\b", "flat": r"\bflat\b",
        "credit event": r"\bcredit[ _]event\b", "no event": r"\bno[ _]event\b"}


def _contradicts(unit, impl, ents, by_id, rows, labels) -> bool:
    """True when the implication asserts, for the entities it names, a label or direction other than the submitted one."""
    from .answer import _direction

    low = impl.lower().replace("_", " ")
    if re.search(r"\b(incorrect|wrong|surpris\w*|inconsistent|contrary to|disagree\w*|questionable|doubtful|mistaken)\b", low):
        return True  # a reason must never undermine the submitted answer
    groups = {_direction(unit, by_id[e], rows[e]) for e in ents if e in rows}
    if len(groups) != 1:
        # mixed group: any label or direction the text asserts cannot hold for every named entity
        if labels:
            return any(_asserts(low, lab.replace("_", " ").lower()) for lab in labels)
        return bool(re.search(_UPW, low) or re.search(_DOWNW, low))
    g = groups.pop().replace("_", " ").lower()
    if labels:
        for lab in labels:
            o = lab.replace("_", " ").lower()
            if o == g:
                continue
            if _asserts(low, o):
                return True
        return False
    if g == "increase":
        return bool(re.search(_DOWNW, low))
    if g == "decrease":
        return bool(re.search(_UPW, low))
    if g == "no change":
        if re.search(r"\b(expected to|will|likely to|set to|poised to)\s+(rise|increase|fall|decline|drop|climb|move (higher|lower))", low):
            return True
        return bool(re.search(r"\b(increas\w*|decreas\w*|rise|fall|declin\w*|higher|lower)\b", low)) and not re.search(r"no change|unchanged|stable", low)
    return False


def _asserts(low: str, o: str) -> bool:
    """Does the (lower-cased, underscores as spaces) text assert label `o`?"""
    if o == "credit event":  # "credit event" also appears in sentences about a low probability of one
        return bool(re.search(r"\b(will|likely to|expected to) (experience|have|face|suffer) (a )?credit event"
                              r"|\b(higher|elevated|significant|high|increased|substantial) (probability|risk|likelihood|chance) of (a )?credit event"
                              r"|\blabel (is |of |as )?'?credit event", low))
    if o in ("positive reaction", "negative reaction"):
        return bool(re.search(r"\b" + o.split()[0] + r"\s+(stock\s+|market\s+|price\s+)?reaction", low))
    return bool(re.search(_SYN.get(o, r"\b" + re.escape(o) + r"\b"), low))


def build(unit, answer, preds, cands, raw) -> list:
    out, seen_keys, total, used_premises = [], set(), 0, set()
    reasons = raw.get("reasons") if isinstance(raw, dict) else None
    if not isinstance(reasons, list):
        return out
    roster = set(unit.roster)
    by_id = {e["entity_id"]: e for e in answer["entity_predictions"]}
    rows = {r["entity_id"]: r for r in unit.entities}
    labels = (unit.task.get("target") or {}).get("labels") or []
    for r in reasons[:3]:
        if not isinstance(r, dict):
            continue
        ids = r.get("evidence_ids") if isinstance(r.get("evidence_ids"), list) else [r.get("evidence_id")]
        evs = []
        for x in ids:
            c = cands.get(x) if isinstance(x, str) else None
            if c and c not in evs:
                evs.append(c)
        evs = evs[:MAX_CITES]
        mech = _clean_text(r.get("mechanism"), 700)
        impl = _clean_text(r.get("implication"), 450)
        if not evs or not mech or not impl:
            continue
        ents = [x for x in (r.get("entities") or []) if isinstance(x, str) and x in roster] if isinstance(r.get("entities"), list) else []
        if not ents:
            ents = sorted({c["entity_id"] for c in evs}, key=unit.roster.index)
        evs = [c for c in evs if len(c["text"].split()) >= 6]
        if not evs:
            continue
        if _contradicts(unit, impl, ents, by_id, rows, labels):
            continue
        basis = " ".join(str(getattr(preds.get(e), "why", "")) for e in ents)
        src_all = " ".join(c["text"] for c in evs) + " " + " ".join(json.dumps(rows[e]) for e in ents if e in rows) + " " + basis + " " + \
            " ".join(json.dumps({k: by_id[e].get(k) for k in ("point_forecast", "interval")}) for e in ents)
        if not _mechanism_ok(mech, _nums(src_all)):
            continue
        # figures in the implication must be submitted values of the named entities (else use a neutral implication)
        allowed = set()
        for e in ents:
            a = by_id[e]
            allowed |= _nums(json.dumps({k: a.get(k) for k in ("point_forecast", "interval", "label", "rank")}), signed=True)
            allowed |= _nums(e, signed=True)
        if not _nums(impl, signed=True) <= allowed:
            impl = f"Consistent with the submitted answers for {', '.join(ents[:6])}."
        # premise: the model's statement only when every figure in it appears in a cited passage or the entities' task rows
        src = " ".join(c["text"] for c in evs) + " " + " ".join(json.dumps(rows[e]) for e in ents if e in rows)
        prem = _clean_text(r.get("premise"), 600)
        if not prem or not _nums(prem) <= _nums(src) or not re.search(r"\d", prem) or len(prem.split()) < 6:
            prem = evs[0]["text"]
        if prem in used_premises:  # each reason should rest on a different fact
            alt = [c["text"] for c in evs if c["text"] not in used_premises]
            if not alt:
                continue
            prem = alt[0]
        used_premises.add(prem)
        sub = []
        for e in ents[:6]:
            a = by_id[e]
            sub.append(f"{e}: " + (f"label {a['label']}, " if a.get("label") is not None else "") + f"point {a['point_forecast']}")
        if not any(e in impl for e in ents):
            impl = f"For {', '.join(ents[:6])}: {impl}"
        impl = f"{impl} Submitted answer - " + "; ".join(sub) + "."
        key = (prem, mech, impl)
        if key in seen_keys:
            continue
        item = {
            "reason_id": f"r{len(out) + 1}",
            "premise": prem,
            "mechanism": mech,
            "answer_implication": impl,
            "scope": {"entities": ents},
            "citations": [{"doc_id": c["doc_id"], "span_start": c["s"], "span_end": c["e"]} for c in evs],
        }
        size = len(json.dumps({k: item[k] for k in ("reason_id", "premise", "mechanism", "answer_implication")}).encode())
        if total + size > 6200:
            break
        total += size
        seen_keys.add(key)
        out.append(item)
    return out


def write_reasons(unit, answer: dict, deadline: float, log=print):
    """Return a copy of `answer` with model-written reasons, or None (keep the deterministic reasons)."""
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
    client = _Client(endpoint, os.environ.get("MODEL_NAME", "").strip() or "default", os.environ.get("MODEL_TOKEN") or None, deadline, log)
    preds = predict_all(unit)
    drivers, pools, lead = _pools(unit, answer, preds, deadline)
    cands = {c["id"]: c for c in lead.values()}
    for _, lst in pools:
        cands.update({c["id"]: c for c in lst})
    if not cands:
        return None
    log(f"[reasons] drivers: {drivers} | passages {len(cands)} | retrieval: "
        f"{'dense NeMo Retriever' if getattr(_hybrid_pools, 'dense', False) else 'keyword'}")
    prompt = _prompt(unit, answer, preds, pools, lead)
    for attempt in range(2):
        try:
            raw = extract_json(client.chat(prompt))
        except Exception as exc:
            log(f"[reasons] request failed ({type(exc).__name__})")
            break
        reasons = build(unit, answer, preds, cands, raw)
        if reasons and len(reasons) < 3:  # fewer reasons never score higher: top up with the deterministic grouped reasons
            try:
                from .answer import det_reasons

                for d in det_reasons(unit, answer, preds):
                    trial = reasons + [dict(d, reason_id=f"r{len(reasons) + 1}")]
                    if len(trial) <= 3 and not _check_reasons(unit, trial) and d["premise"] not in {x["premise"] for x in reasons}:
                        reasons = trial
            except Exception:
                pass
        if reasons and not _check_reasons(unit, reasons):
            new = dict(answer)
            new["submitted_reasons"] = reasons
            log(f"[reasons] {len(reasons)} model-written reasons accepted (attempt {attempt + 1})")
            return new
        log(f"[reasons] attempt {attempt + 1}: no valid reasons")
    return None
