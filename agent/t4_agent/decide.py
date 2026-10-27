"""Model-assisted decisions on top of the deterministic answer (variants H1 / H2). Never required.

H1  (T4_LLM_LABELS=1)  label decisions on classification units, by self-consistency voting:
    * evidence pack per entity = up to 6 entity-bound, pre-cutoff, verbatim spans selected deterministically, the task row,
      the label vocabulary (with the task's own label descriptions) and the base-rate label;
    * T4_VOTES (3) independent samples per entity group, each with a different fixed ordering of the evidence;
    * flip away from the deterministic label only if the same different label wins a strict majority (>= 2 of 3),
      the mean confidence of those votes is >= T4_LABEL_CONF (0.75), the label is in the unit's vocabulary, and a cited
      evidence id belongs to that entity and yields a claim that passes the claim rules;
    * applying a flip additionally needs T4_ALLOW_LABEL_FLIP=1 (otherwise the votes are only logged = shadow mode);
    * the numeric point is kept coherent with the new label (relabel_point) and the cited span becomes the first claim.
H2  (T4_LLM_MOVES=1)   bounded numeric nudges on REGRESSION units only: a move needs a cited candidate span, is at most
    T4_MOVE_FRAC x the interval half-width, and the share of same-signed moves is capped at 70% unless each cited span
    quotes the entity's own figure.
Request budget: <= 12 per unit (label votes use <= 9, moves <= 3); any failure leaves the deterministic answer untouched.
"""
from __future__ import annotations

import collections
import copy
import math
import os
import statistics
import time

from .answer import _check_reasons, check_claim, det_reasons
from .claims import support_numbers, task_row_claim, top_candidates
from .predict import predict_all, relabel_point
from .refine import MAX_REQUESTS, Budget, Client, extract_json

N_CAND = 6
MAX_LABEL_GROUPS = 3
MAX_MOVE_GROUPS = 3
SAME_SIGN_CAP = 0.7


def _f(name, default, lo, hi):
    try:
        v = float(os.environ.get(name, default))
    except ValueError:
        v = float(default)
    return min(max(v, lo), hi) if v == v else float(default)


def cfg():
    return dict(
        labels=os.environ.get("T4_LLM_LABELS", "0") == "1",
        moves=os.environ.get("T4_LLM_MOVES", "0") == "1",
        flip=os.environ.get("T4_ALLOW_LABEL_FLIP", "0") == "1",
        conf=_f("T4_LABEL_CONF", 0.75, 0.0, 1.0),
        votes=int(_f("T4_VOTES", 3, 1, 5)),
        move_frac=_f("T4_MOVE_FRAC", 0.25, 0.0, 1.0),
        rationale=os.environ.get("T4_LABEL_RATIONALE", "0") == "1",
        skip_macro=os.environ.get("T4_LABEL_SKIP_MACRO", "1") == "1",
        det_reasons=os.environ.get("T4_DET_REASONS", "1") == "1",
    )


def enabled() -> bool:
    c = cfg()
    return c["labels"] or c["moves"]


def _compact_row(row):
    import json

    return json.dumps({k: v for k, v in row.items() if k != "corpus_ref"}, ensure_ascii=False, separators=(",", ":"))[:600]


def _order(lst, v):
    """Fixed, clock-free orderings of the evidence list for vote number v."""
    if v == 0 or len(lst) < 2:
        return list(lst)
    if v == 1:
        return list(reversed(lst))
    k = (3 * (v - 1)) % len(lst)
    return lst[k:] + lst[:k]


def _dense_packs(unit, preds, by_id, deadline):
    """D21: each entity's evidence pack = its N_CAND spans with the highest NeMo Retriever cosine to the task's own
    question (target name + label meanings + the entity's name), from the GPU index; None without an encoder."""
    try:
        from . import retrieval as _rt

        idx = _rt.index_for(unit, deadline)
        if idx is None or not idx.dense:
            return None
        t = unit.task.get("target") or {}
        asr = t.get("label_assertions") or {}
        base_q = " ".join([str(t.get("name", "")).replace("_", " ")] + [str(v) for v in asr.values()])[:600]
        packs, n = {}, 0
        for row in unit.entities:
            eid = row["entity_id"]
            name = str(row.get("name") or eid)
            order, _ = _rt.rank_dense(idx, f"{name}: {base_q}", eligible=lambda it, e=eid: it[0] == e, top=N_CAND)
            lst = []
            for i in order[:N_CAND]:
                _e, doc_id, s0, e0, text = idx.items[i]
                n += 1
                lst.append({"id": f"e{n}", "entity_id": eid, "doc_id": doc_id, "s": s0, "e": e0, "text": text, "score": 0.0})
            packs[eid] = lst
        if any(packs.values()):
            _dense_packs.used = True
            return packs
        return None
    except Exception:
        return None


def _packs(unit, preds, by_id, deadline=None):
    if deadline is not None:
        dp = _dense_packs(unit, preds, by_id, min(deadline - 200, time.time() + 200))
        if dp is not None:
            return dp
    packs, n = {}, 0
    for row in unit.entities:
        eid = row["entity_id"]
        p = preds[eid]
        nums = support_numbers(row, type("P", (), {"point": by_id[eid]["point_forecast"]})())
        lst = []
        for c in top_candidates(unit, row, N_CAND, numbers=nums, terms=p.terms):
            n += 1
            c["id"] = f"e{n}"
            c["entity_id"] = eid
            lst.append(c)
        packs[eid] = lst
    return packs


def _claim_from(c):
    return {"doc_id": c["doc_id"], "span_start": c["s"], "span_end": c["e"], "claim": c["text"]}


SYSTEM = (
    "You are a careful financial analyst. Reply with ONE JSON object and nothing else: no prose outside JSON, no markdown. "
    "Use only the passages given and the cutoff date; never use knowledge of events after the cutoff. Never invent ids."
)


def _label_prompt(unit, group, packs, by_id, v, rationale):
    t = unit.task
    target = t.get("target") or {}
    labels = target.get("labels") or []
    asserts = target.get("label_assertions") or {}
    lines = [
        "TASK: " + str(t.get("prompt", ""))[:1500],
        f"CUTOFF DATE: {t.get('cutoff_date')}. Decide each entity's label using ONLY the passages below.",
        "LABELS:",
    ]
    for lab in labels:
        lines.append(f"  - {lab}: {str(asserts.get(lab, ''))[:300]}")
    lines.append("")
    for row in group:
        eid = row["entity_id"]
        lines.append(f"ENTITY {eid}: {_compact_row(row)}")
        lines.append(f"  default (base-rate) label: {by_id[eid].get('label')}")
        for c in _order(packs[eid], v):
            lines.append(f"  [{c['id']}] {c['text']}")
        lines.append("")
    form = ('{"verdicts":[{"entity_id":"<id>","label":"<one label>","confidence":<0..1>,"evidence_ids":["<ids of this entity>"]'
            + (',"rationale":"<3 short bullet points>"' if rationale else "") + "}]}")
    lines += [
        "Return JSON exactly of the form: " + form,
        "Rules: pick the label the passages most directly support. If the passages do not clearly point away from the default "
        "label, answer the default label with low confidence. Confidence must reflect how directly the evidence settles the "
        "outcome (0.9+ only for explicit statements). One object per entity listed above; evidence_ids must come from that entity's list.",
    ]
    if rationale:
        lines.append("Before the verdict, think step by step in 3 bullet points inside the JSON field `rationale`.")
    return "\n".join(lines)


class _Client(Client):
    """refine.Client with this module's system prompt."""

    def chat(self, user):  # same body, different system message
        import time as _t
        import urllib.error

        if self.used >= MAX_REQUESTS:
            raise Budget("request budget exhausted")
        remaining = self.deadline - _t.time()
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": 3500,
        }
        if not self.dropped_ctk:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return self._post(body, remaining)
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and "chat_template_kwargs" in body and self.used < MAX_REQUESTS:
                self.dropped_ctk = True
                body.pop("chat_template_kwargs")
                self.log("[decide] HTTP 400; retrying once without chat_template_kwargs")
                return self._post(body, self.deadline - _t.time())
            raise


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def label_pass(unit, new, preds, packs, client, c, log):
    """Self-consistency label voting. Mutates `new`; returns stats dict."""
    by_id = {e["entity_id"]: e for e in new["entity_predictions"]}
    rows = {r["entity_id"]: r for r in unit.entities}
    labels = (unit.task.get("target") or {}).get("labels") or []
    elig = [r for r in unit.entities
            if not (c["skip_macro"] and preds[r["entity_id"]].fn == "_macrorev") and packs[r["entity_id"]]]
    stats = {"eligible": len(elig), "flips": [], "kept": 0, "requests": 0, "votes_valid": 0}
    if not elig or not labels:
        return stats
    groups_n = max(1, min(MAX_LABEL_GROUPS, len(elig), (MAX_REQUESTS - 3) // c["votes"]))
    size = math.ceil(len(elig) / groups_n)
    groups = [elig[i:i + size] for i in range(0, len(elig), size)]
    votes = {r["entity_id"]: [] for r in elig}  # eid -> [(label, conf, [ids])]
    failures = 0
    for v in range(c["votes"]):
        for g in groups:
            try:
                reply = client.chat(_label_prompt(unit, g, packs, by_id, v, c["rationale"]))
            except Exception as exc:
                failures += 1
                log(f"[decide] vote {v} request failed ({type(exc).__name__}): {exc}")
                if failures >= 2 or isinstance(exc, Budget) or (
                        isinstance(exc, OSError) and not hasattr(exc, "code")):
                    log("[decide] endpoint unusable; stopping label votes")
                    return _finish_labels(unit, new, preds, packs, c, votes, by_id, rows, labels, stats, client, log, aborted=True)
                continue
            obj = extract_json(reply)
            items = obj.get("verdicts") if isinstance(obj, dict) else None
            if not isinstance(items, list):
                log(f"[decide] vote {v}: reply not parseable")
                continue
            ids_in_group = {r["entity_id"] for r in g}
            for it in items:
                if not isinstance(it, dict) or it.get("entity_id") not in ids_in_group:
                    continue
                lab, conf = it.get("label"), it.get("confidence")
                if not isinstance(lab, str) or lab not in labels or not _num(conf) or not 0.0 <= float(conf) <= 1.0:
                    continue
                ev = it.get("evidence_ids")
                ev = [x for x in ev if isinstance(x, str)] if isinstance(ev, list) else []
                votes[it["entity_id"]].append((lab, float(conf), ev))
    return _finish_labels(unit, new, preds, packs, c, votes, by_id, rows, labels, stats, client, log, aborted=False)


def _finish_labels(unit, new, preds, packs, c, votes, by_id, rows, labels, stats, client, log, aborted):
    stats["requests"] = client.used
    for eid, vs in votes.items():
        stats["votes_valid"] += len(vs)
        base = by_id[eid].get("label")
        if aborted or not vs:
            stats["kept"] += 1
            continue
        cnt = collections.Counter(l for l, _, _ in vs)
        top = cnt.most_common()
        best_n = top[0][1]
        tied = [l for l, n in top if n == best_n]
        lab = tied[0]
        if len(tied) > 1 or lab == base or best_n < min(2, c["votes"]) or best_n * 2 <= c["votes"]:
            stats["kept"] += 1
            continue
        sel = sorted([v for v in vs if v[0] == lab], key=lambda v: -v[1])
        mean_conf = statistics.mean(v[1] for v in sel)
        if mean_conf < c["conf"]:
            stats["kept"] += 1
            log(f"[decide] KEEP {eid}: {base} (vote for {lab} had mean confidence {mean_conf:.2f} < {c['conf']})")
            continue
        cand = None
        valid_ids = {x["id"]: x for x in packs[eid]}
        for _, _, ids in sel:
            for i in ids:
                x = valid_ids.get(i)
                if x and check_claim(unit, eid, _claim_from(x)) is None:
                    cand = x
                    break
            if cand:
                break
        if cand is None:
            stats["kept"] += 1
            log(f"[decide] KEEP {eid}: {base} (no valid cited evidence for {lab})")
            continue
        stats["flips"].append((eid, base, lab, round(mean_conf, 2), best_n))
        log(f"[decide] {'FLIP' if c['flip'] else 'SHADOW-FLIP'} {eid} {base}->{lab} conf={mean_conf:.2f} votes={best_n}/{c['votes']} evidence={cand['id']}")
        if not c["flip"]:
            continue
        entry = by_id[eid]
        entry["label"] = lab
        rp = relabel_point(unit, rows[eid], preds[eid], lab)
        if rp is not None:
            entry["point_forecast"], entry["interval"]["lo"], entry["interval"]["hi"] = (float(rp[0]), float(rp[1]), float(rp[2]))
        tr = task_row_claim(unit, rows[eid])
        entry["claims"] = [_claim_from(cand), {"doc_id": tr[0], "span_start": tr[1], "span_end": tr[2], "claim": tr[3]}]
    return stats


def _move_prompt(unit, group, packs, by_id):
    t = unit.task
    lines = [
        "TASK: " + str(t.get("prompt", ""))[:1500],
        f"CUTOFF DATE: {t.get('cutoff_date')}. Use ONLY the passages below.",
        "For each entity a baseline forecast is given. Say whether the passages clearly imply the true value will be HIGHER (up) or "
        "LOWER (down) than the baseline, or whether to leave it (none).",
        "",
    ]
    for row in group:
        eid = row["entity_id"]
        a = by_id[eid]
        lines.append(f"ENTITY {eid}: {_compact_row(row)}")
        lines.append(f"  baseline point={a['point_forecast']} interval=[{a['interval']['lo']}, {a['interval']['hi']}]")
        for c in packs[eid]:
            lines.append(f"  [{c['id']}] {c['text']}")
        lines.append("")
    lines += [
        'Return JSON exactly: {"moves":[{"entity_id":"<id>","direction":"up|down|none","strength":<0..1>,"evidence_id":"<id of this entity or null>"}]}',
        "Rules: answer none unless a passage explicitly supports a direction; cite the passage; strength 1 only for explicit statements; "
        "one object per entity listed above.",
    ]
    return "\n".join(lines)


def move_pass(unit, new, preds, packs, client, c, log):
    by_id = {e["entity_id"]: e for e in new["entity_predictions"]}
    rows = {r["entity_id"]: r for r in unit.entities}
    ents = [r for r in unit.entities if packs[r["entity_id"]]]
    stats = {"moves": [], "dropped_for_cap": 0, "requests": 0}
    if not ents or c["move_frac"] <= 0:
        return stats
    groups_n = max(1, min(MAX_MOVE_GROUPS, math.ceil(len(ents) / 6)))
    size = math.ceil(len(ents) / groups_n)
    cand_moves = []  # dict(eid, sign, strength, cand, indiv)
    failures = 0
    for i in range(0, len(ents), size):
        g = ents[i:i + size]
        try:
            reply = client.chat(_move_prompt(unit, g, packs, by_id))
        except Exception as exc:
            failures += 1
            log(f"[decide] move request failed ({type(exc).__name__}): {exc}")
            if failures >= 2 or isinstance(exc, Budget):
                break
            continue
        obj = extract_json(reply)
        items = obj.get("moves") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            continue
        ids = {r["entity_id"] for r in g}
        for it in items:
            if not isinstance(it, dict) or it.get("entity_id") not in ids:
                continue
            d, s = it.get("direction"), it.get("strength")
            if d not in ("up", "down") or not _num(s) or not 0.0 < float(s) <= 1.0:
                continue
            cand = next((x for x in packs[it["entity_id"]] if x["id"] == it.get("evidence_id")), None)
            if cand is None or check_claim(unit, it["entity_id"], _claim_from(cand)) is not None:
                continue
            nums = support_numbers(rows[it["entity_id"]], type("P", (), {"point": by_id[it["entity_id"]]["point_forecast"]})())
            # exempt from the same-sign cap only when the cited span comes from a document bound to THIS entity
            # (not a shared one) and quotes one of the entity's own figures
            doc = unit.docs.get(cand["doc_id"])
            indiv = bool(doc is not None and not doc.shared and any(n in cand["text"] for n in nums))
            cand_moves.append(dict(eid=it["entity_id"], sign=1 if d == "up" else -1, strength=float(s), cand=cand, indiv=indiv))
    # cap the share of same-signed moves (stand-in showed one-directional bias)
    seen = set()
    cand_moves = [m for m in cand_moves if not (m["eid"] in seen or seen.add(m["eid"]))]
    if len(cand_moves) >= 3:
        pos = [m for m in cand_moves if m["sign"] > 0]
        neg = [m for m in cand_moves if m["sign"] < 0]
        maj, mino = (pos, neg) if len(pos) >= len(neg) else (neg, pos)
        droppable = sorted([m for m in maj if not m["indiv"]], key=lambda m: m["strength"])
        # drop the weakest not-individually-supported majority-sign moves until the retained share is <= 70%
        while maj and len(maj) / (len(maj) + len(mino)) > SAME_SIGN_CAP and droppable:
            m = droppable.pop(0)
            maj.remove(m)
            cand_moves.remove(m)
            stats["dropped_for_cap"] += 1
    for m in cand_moves:
        entry = by_id[m["eid"]]
        half = (entry["interval"]["hi"] - entry["interval"]["lo"]) / 2
        d = m["sign"] * min(m["strength"], 1.0) * c["move_frac"] * half
        if abs(d) < 1e-12:
            continue
        entry["point_forecast"] = round(entry["point_forecast"] + d, 6)
        entry["interval"]["lo"] = round(entry["interval"]["lo"] + d, 6)
        entry["interval"]["hi"] = round(entry["interval"]["hi"] + d, 6)
        tr = task_row_claim(unit, rows[m["eid"]])
        entry["claims"] = [_claim_from(m["cand"]), {"doc_id": tr[0], "span_start": tr[1], "span_end": tr[2], "claim": tr[3]}]
        stats["moves"].append((m["eid"], "up" if m["sign"] > 0 else "down", round(d, 6)))
        log(f"[decide] MOVE {m['eid']} {'up' if m['sign'] > 0 else 'down'} by {d:+.6g} evidence={m['cand']['id']} individually_supported={m['indiv']}")
    stats["requests"] = client.used
    return stats


def decide(unit, answer: dict, deadline: float, log=print):
    """Return a modified copy of `answer`, or None if nothing changed / nothing could be done."""
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
    c = cfg()
    if not endpoint or not (c["labels"] or c["moves"]):
        return None
    client = _Client(endpoint, os.environ.get("MODEL_NAME", "").strip() or "default", os.environ.get("MODEL_TOKEN") or None, deadline, log)
    new = copy.deepcopy(answer)
    preds = predict_all(unit)
    by_id = {e["entity_id"]: e for e in new["entity_predictions"]}
    packs = _packs(unit, preds, by_id, deadline)
    log(f"[decide] evidence packs: {sum(len(v) for v in packs.values())} spans ({'dense NeMo Retriever' if getattr(_dense_packs, 'used', False) else 'lexical'})")
    changed = False
    if c["labels"] and unit.target_type == "classification":
        st = label_pass(unit, new, preds, packs, client, c, log)
        log(f"[decide] label pass: eligible={st['eligible']} valid_votes={st['votes_valid']} flips={len(st['flips'])} kept={st['kept']} requests={client.used}")
        changed = changed or (c["flip"] and bool(st["flips"]))
    if c["moves"] and unit.target_type == "regression":
        st = move_pass(unit, new, preds, packs, client, c, log)
        log(f"[decide] move pass: moves={len(st['moves'])} dropped_for_cap={st['dropped_for_cap']} requests={client.used}")
        changed = changed or bool(st["moves"])
    if not changed:
        return None
    if c["det_reasons"]:
        try:
            reasons = det_reasons(unit, new, preds)
            if reasons and not _check_reasons(unit, reasons):
                new["submitted_reasons"] = reasons
            else:
                new.pop("submitted_reasons", None)
        except Exception:
            new.pop("submitted_reasons", None)
    return new
