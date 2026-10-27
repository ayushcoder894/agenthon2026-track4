"""Optional House-model refinement. Runs only when MODEL_ENDPOINT is set; never required.

Contract with the rest of the agent:
  * the deterministic v1 answer has ALREADY been written; this module returns a candidate answer or
    None, and the caller replaces v1 only if the candidate passes the full local validation;
  * any exception, timeout, malformed reply or out-of-range value leaves v1 intact;
  * budget: at most MAX_REQUESTS (12) chat requests per unit (the platform allows 25), max_tokens <= 3500,
    one request each, temperature 0.2, `chat_template_kwargs: {"enable_thinking": false}`; only the
    documented body fields are sent (no `seed`, no `stream`); a single retry happens only after an
    HTTP 400 (refused before admission) and only without chat_template_kwargs; never on 5xx/timeouts;
    deadline-aware timeouts;
  * the model never supplies offsets or free-text claims: it only picks an evidence id from candidate
    spans WE extracted (so every claim stays a verbatim quote of an entity-bound, pre-cutoff passage),
    proposes a bounded adjustment (point within +-25% of the v1 half-width, label only from the task
    vocabulary and only with high confidence), and writes short mechanisms for submitted_reasons.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

import types

from .claims import support_numbers, task_row_claim, top_candidates
from .predict import predict_all
from .unit_io import TASK_DOC_ID

MAX_REQUESTS = 12
MAX_TOKENS = 3500  # platform maximum is 4000 output tokens per request
TEMPERATURE = 0.2


def _move_frac() -> float:
    """Max point move as a fraction of the v1 interval half-width. Baked into the image via
    `ENV T4_MOVE_FRAC` in the Dockerfile (variants: 0 = evidence + reasons only, 0.25 = default)."""
    try:
        v = float(os.environ.get("T4_MOVE_FRAC", "0.25"))
    except ValueError:
        return 0.25
    return min(max(v, 0.0), 1.0) if v == v else 0.25


MOVE_FRAC = _move_frac()
FLIP_CONF = 0.8
# Label flips cannot be validated offline (no outcomes), and the baseline label is the modal/base-rate
# class, so a flip is only accepted when explicitly enabled.
ALLOW_FLIPS = os.environ.get("T4_ALLOW_LABEL_FLIP") == "1"
GROUP_SIZE = 6
N_CAND = 6
DENY = ("leaderboard", "canary", "/home/", "units/", "reference/", "outcome.json", "team_id", "team name",
        "participant_id", "participant name", "submission_id", "other submission")
SYSTEM = (
    "You are a careful quantitative analyst. Reply with ONE JSON object and nothing else: no prose, no "
    "markdown, no code fences. Use only the evidence ids you are given. Never invent numbers or ids."
)


class Budget(Exception):
    pass


class Client:
    def __init__(self, endpoint: str, model: str, token: str | None, deadline: float, log):
        base = endpoint.rstrip("/")
        self.url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
        self.model, self.token, self.deadline, self.log = model, token, deadline, log
        self.used = 0
        self.dropped_ctk = False

    def chat(self, user: str) -> str:
        if self.used >= MAX_REQUESTS:
            raise Budget("request budget exhausted")
        remaining = self.deadline - time.time()
        # Only the documented fields: model, messages, max_tokens, temperature, chat_template_kwargs.
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        if not self.dropped_ctk:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return self._post(body, remaining)
        except urllib.error.HTTPError as exc:
            # A request refused before admission (400) does not consume the platform allowance, so ONE
            # retry without the optional chat_template_kwargs is safe. Never retry 5xx / timeouts.
            if exc.code == 400 and "chat_template_kwargs" in body and self.used < MAX_REQUESTS:
                self.dropped_ctk = True
                body.pop("chat_template_kwargs")
                self.log("[refine] HTTP 400; retrying once without chat_template_kwargs")
                return self._post(body, self.deadline - time.time())
            raise

    def _post(self, body: dict, remaining: float) -> str:
        if remaining < 15:
            raise Budget("deadline too close")
        self.used += 1  # counted locally before sending (the retry above is counted too)
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            cap = float(os.environ.get("T4_HTTP_TIMEOUT", "120"))
        except ValueError:
            cap = 120.0
        with urllib.request.urlopen(req, timeout=max(1.0, min(cap, remaining - 10))) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"] or ""


def extract_json(text: str):
    """First JSON object in the reply (tolerates fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _compact_row(row: dict) -> str:
    return json.dumps({k: v for k, v in row.items() if k != "corpus_ref"}, ensure_ascii=False, separators=(",", ":"))[:600]


def _group_prompt(unit, group, cands, answer_by_id) -> str:
    t = unit.task
    target = t.get("target") or {}
    lines = [
        "TASK: " + str(t.get("prompt", ""))[:1500],
        f"TARGET: type={target.get('type')} name={target.get('name')} labels={target.get('labels')}",
        f"CUTOFF: {t.get('cutoff_date')}. Use only the evidence below.",
        "For each entity you get its table row, the baseline prediction, the ALLOWED point range, and candidate evidence passages.",
        "",
    ]
    for row in group:
        eid = row["entity_id"]
        a = answer_by_id[eid]
        half = (a["interval"]["hi"] - a["interval"]["lo"]) / 2
        p0 = a["point_forecast"]
        lines.append(f"ENTITY {eid}: {_compact_row(row)}")
        lines.append(f"  baseline: label={a.get('label')} point={p0} interval=[{a['interval']['lo']}, {a['interval']['hi']}]")
        lines.append(f"  allowed point range: [{p0 - MOVE_FRAC * half:.6g}, {p0 + MOVE_FRAC * half:.6g}]")
        for c in cands[eid]:
            lines.append(f"  [{c['id']}] {c['text']}")
        lines.append("")
    lines += [
        "Return JSON exactly of the form:",
        '{"entities":[{"entity_id":"<id>","evidence_id":"<one candidate id for this entity or null>",'
        '"label":"<one allowed label or null>","point":<number inside the allowed range or null>,'
        '"confidence":<0..1>,"note":"<=25 words"}]}',
        "Rules: keep the baseline (null) unless the evidence clearly implies otherwise; choose the evidence_id of the "
        "passage that most directly bears on the target for that entity; one object per entity listed above.",
    ]
    return "\n".join(lines)


def _reasons_prompt(unit, answer, chosen) -> str:
    t = unit.task
    lines = [
        "TASK: " + str(t.get("prompt", ""))[:1200],
        "Final answers per entity and the evidence passage chosen for each:",
    ]
    for e in answer["entity_predictions"]:
        ev = chosen.get(e["entity_id"])
        lines.append(f"- {e['entity_id']}: label={e.get('label')} point={e.get('point_forecast')}"
                     + (f" evidence [{ev['id']}]: {ev['text']}" if ev else " (no evidence passage)"))
    lines += [
        "",
        "The answers above are FINAL. Write 1 to 3 reasons that SUPPORT them: each reason explains why its evidence is consistent with "
        "the submitted label/value of the entities it names. Never argue for a different answer. Each reason is grounded in ONE of the evidence ids listed.",
        'Return JSON exactly: {"reasons":[{"evidence_id":"<id from the list>","mechanism":"<why that fact moves the answer, <=60 words>",'
        '"implication":"<what it implies for the answer, naming the entity ids, <=40 words>","entities":["<entity ids>"]}]}',
        "Do not mention files, paths, scores, rankings of teams or the evaluation.",
    ]
    return "\n".join(lines)


def _clean_text(s, limit) -> str | None:
    if not isinstance(s, str):
        return None
    s = " ".join(s.split())
    if not s or len(s) > limit:
        return None
    low = s.lower()
    if any(d in low for d in DENY) or "://" in low or "http" in low or "www." in low:
        return None
    return s


def build_reasons(unit, answer, chosen_by_id, raw) -> list:
    out, seen, total = [], set(), 0
    reasons = raw.get("reasons") if isinstance(raw, dict) else None
    if not isinstance(reasons, list):
        return out
    roster = set(unit.roster)
    for r in reasons[:3]:
        if not isinstance(r, dict):
            continue
        ev = chosen_by_id.get(r.get("evidence_id"))
        mech = _clean_text(r.get("mechanism"), 600)
        impl = _clean_text(r.get("implication"), 400)
        ents = [x for x in (r.get("entities") or []) if isinstance(x, str) and x in roster] if isinstance(r.get("entities"), list) else []
        if not ev or not mech or not impl or ev["doc_id"] == TASK_DOC_ID:
            continue
        if not ents:
            ents = [ev["entity_id"]]
        if not any(e in impl for e in ents):
            impl = f"For {', '.join(ents)}: {impl}"
        # state the submitted values verbatim so the implication can never misreport the answer
        by_id = {e["entity_id"]: e for e in answer["entity_predictions"]}
        sub = []
        for e in ents[:3]:
            a = by_id.get(e)
            if a:
                sub.append(f"{e}: " + (f"label {a['label']}, " if a.get("label") is not None else "") + f"point {a['point_forecast']}")
        if sub:
            impl = f"{impl} Submitted answer - " + "; ".join(sub) + "."
        premise = ev["text"]
        key = (premise, mech, impl)
        if key in seen:
            continue
        item = {
            "reason_id": f"r{len(out) + 1}",
            "premise": premise,
            "mechanism": mech,
            "answer_implication": impl,
            "scope": {"entities": ents},
            "citations": [{"doc_id": ev["doc_id"], "span_start": ev["s"], "span_end": ev["e"]}],
        }
        size = len(json.dumps({k: item[k] for k in ("reason_id", "premise", "mechanism", "answer_implication")}).encode())
        if total + size > 6000:
            break
        total += size
        seen.add(key)
        out.append(item)
    return out


def refine(unit, answer: dict, deadline: float, log=print):
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
    if not endpoint:
        return None
    client = Client(endpoint, os.environ.get("MODEL_NAME", "").strip() or "default",
                    os.environ.get("MODEL_TOKEN") or None, deadline, log)
    new = copy.deepcopy(answer)
    by_id = {e["entity_id"]: e for e in new["entity_predictions"]}
    rows = {r["entity_id"]: r for r in unit.entities}
    labels = (unit.task.get("target") or {}).get("labels") or []
    n_groups = max(1, math.ceil(len(unit.entities) / GROUP_SIZE))
    gsize = GROUP_SIZE if n_groups + 1 <= MAX_REQUESTS else math.ceil(len(unit.entities) / (MAX_REQUESTS - 1))
    try:
        terms_by_id = {k: v.terms for k, v in predict_all(unit).items()}
    except Exception:
        terms_by_id = {}
    cands, evid, diffs = {}, {}, []
    counter = 0
    for row in unit.entities:
        lst = []
        _p = by_id[row["entity_id"]]
        _nums = support_numbers(row, types.SimpleNamespace(point=_p["point_forecast"]))
        for c in top_candidates(unit, row, N_CAND, numbers=_nums, terms=terms_by_id.get(row["entity_id"], ())):
            counter += 1
            c["id"] = f"e{counter}"
            c["entity_id"] = row["entity_id"]
            evid[c["id"]] = c
            lst.append(c)
        cands[row["entity_id"]] = lst
    chosen = {}  # entity_id -> candidate
    failures = 0
    ents = unit.entities
    for gi in range(0, len(ents), gsize):
        group = ents[gi:gi + gsize]
        try:
            reply = client.chat(_group_prompt(unit, group, cands, by_id))
        except Exception as exc:
            failures += 1
            log(f"[refine] group request failed ({type(exc).__name__}): {exc}")
            if failures >= 2 or isinstance(exc, (Budget, urllib.error.URLError)) and not isinstance(exc, urllib.error.HTTPError):
                log("[refine] endpoint unusable; stopping model use for this unit")
                break
            continue
        obj = extract_json(reply)
        items = obj.get("entities") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            log("[refine] group reply not parseable; keeping v1 for these entities")
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            eid = it.get("entity_id")
            if eid not in by_id or eid not in {r["entity_id"] for r in group}:
                continue
            entry, base = by_id[eid], by_id[eid]
            ev = evid.get(it.get("evidence_id"))
            ev_ok = ev is not None and ev["entity_id"] == eid
            if ev_ok:
                tr = task_row_claim(unit, rows[eid])
                entry["claims"] = [
                    {"doc_id": ev["doc_id"], "span_start": ev["s"], "span_end": ev["e"], "claim": ev["text"]},
                    {"doc_id": tr[0], "span_start": tr[1], "span_end": tr[2], "claim": tr[3]},
                ]
                chosen[eid] = ev
                diffs.append(f"{eid}: evidence -> {ev['id']}")
            conf = it.get("confidence")
            conf = float(conf) if _num(conf) else 0.0
            p0 = entry["point_forecast"]
            iv = entry["interval"]
            half = (iv["hi"] - iv["lo"]) / 2
            pt = it.get("point")
            if _num(pt):
                lo_b, hi_b = p0 - MOVE_FRAC * half, p0 + MOVE_FRAC * half
                p1 = min(max(float(pt), lo_b), hi_b)
                if abs(p1 - p0) > 1e-12:
                    d = p1 - p0
                    entry["point_forecast"] = round(p1, 6)
                    iv["lo"], iv["hi"] = round(iv["lo"] + d, 6), round(iv["hi"] + d, 6)
                    diffs.append(f"{eid}: point {p0} -> {entry['point_forecast']}")
            lab = it.get("label")
            if (ALLOW_FLIPS and isinstance(lab, str) and lab in labels and lab != entry.get("label") and ev_ok
                    and conf >= FLIP_CONF):
                diffs.append(f"{eid}: label {entry.get('label')} -> {lab} (conf {conf})")
                entry["label"] = lab
                if any("event" in l for l in labels):  # probability point must stay coherent with the label
                    p = 0.6 if ("event" in lab and not lab.startswith("no")) else 0.03
                    entry["point_forecast"] = p
                    iv["lo"], iv["hi"] = max(0.0, p - 0.25), min(1.0, p + 0.25)
    if unit.target_type == "ranking":
        ranked = sorted(unit.roster, key=lambda x: (-by_id[x]["point_forecast"], unit.roster.index(x)))
        for i, x in enumerate(ranked):
            by_id[x]["rank"] = i + 1
    if chosen:
        try:
            reply = client.chat(_reasons_prompt(unit, new, chosen))
            reasons = build_reasons(unit, new, {c["id"]: c for c in chosen.values()}, extract_json(reply))
            if reasons:
                new["submitted_reasons"] = reasons
                diffs.append(f"{len(reasons)} submitted_reasons")
        except Exception as exc:
            log(f"[refine] reasons request failed ({type(exc).__name__}): {exc}")
    elif failures:
        log("[refine] no usable model reply; deterministic answer kept")
    log(f"[refine] {client.used} requests; changes: " + ("; ".join(diffs[:40]) if diffs else "none"))
    return new if diffs else None
