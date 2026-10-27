"""Claim selection: extractive, verbatim, entity-bound, short.

A claim is the exact text of the span it cites (`claim == text[span_start:span_end]`), so the scorer's
figure check passes whole and the NLI judge is never asked. Spans are short (<= MAX_SPAN chars, far
below the 8,000-char cap and the 400-judge-token cap). Only documents the manifest binds to the
entity (or marks shared) and dated on or before the cutoff are used; the task table (doc_id "task")
is cited for values the task itself gives, inside the entity's own row only.
"""
from __future__ import annotations

import bisect
import re
import time

from .unit_io import TASK_DOC_ID

MAX_SPAN = 320
MIN_SPAN = 24
_STOP = {
    "inc", "corp", "corporation", "company", "co", "ltd", "the", "and", "of", "group", "holdings", "all",
    "items", "less", "futures", "year", "yr", "note", "bond", "auction", "treasury", "constant", "maturity",
    "yield", "plc", "llc", "sa", "nv", "new", "total",
}
_BAD_CHARS = {"​", "‌", "‍", "﻿", "\xa0", " ", " "}
_TOPIC_TERMS = (
    "earnings per share", "diluted", "net income", "going concern", "substantial doubt", "revenue",
    "guidance", "outlook", "net interest income", "bid-to-cover", "bid_to_cover", "first print",
    "three-month average", "net position", "revised", "federal funds",
    "liquidity", "net loss", "covenant", "credit facility", "net sales", "net income per", "total debt",
    "indebtedness", "maturit",
)
_BOILERPLATE = (
    "incorporated by reference", "indicate by check mark", "exhibit", "☑", "☐", "pursuant to section",
    "securities and exchange commission", "agreement, dated", "plan (amended", "forward-looking",
)
_SENT_SPLIT =re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'$])")


def _clean(seg: str) -> bool:
    return all(c.isprintable() and c not in _BAD_CHARS for c in seg)


def candidates(text: str):
    """Yield (start, end) of short single-line spans: whole lines, else sentences of long lines."""
    pos = 0
    for line in text.split("\n"):
        line_start = pos
        pos += len(line) + 1
        if len(line.strip()) < MIN_SPAN:
            continue
        pieces = []
        if len(line) <= MAX_SPAN + 40:
            pieces.append((0, len(line)))
        else:
            last = 0
            for m in _SENT_SPLIT.finditer(line):
                pieces.append((last, m.start()))
                last = m.end()
            pieces.append((last, len(line)))
        for a, b in pieces:
            seg = line[a:b]
            lead = len(seg) - len(seg.lstrip())
            seg_s = seg.strip()
            if len(seg_s) < MIN_SPAN:
                continue
            s = line_start + a + lead
            e = s + len(seg_s)
            if len(seg_s) > MAX_SPAN:
                cut = seg_s.rfind(" ", 0, MAX_SPAN)
                e = s + (cut if cut >= MIN_SPAN else MAX_SPAN)
            yield s, e


def _name_tokens(row: dict):
    toks = set()
    for field in ("name", "entity_id", "series_name", "tenor"):
        v = row.get(field)
        if isinstance(v, str):
            for w in re.findall(r"[A-Za-z0-9&][A-Za-z0-9&\-]+", v.lower()):
                if len(w) >= 3 and w not in _STOP:
                    toks.add(w)
    return toks


FAST_SEGS = 40000  # above this many acceptable segments in one document, score a sample + needle hits
DEADLINE_AT = None  # epoch seconds; set by the CLI so corpus-claim search can never eat the unit clock


def set_deadline(ts):
    global DEADLINE_AT
    DEADLINE_AT = ts


def _over_deadline() -> bool:
    return DEADLINE_AT is not None and time.time() > DEADLINE_AT


def _segments(doc):
    """Acceptable (start, end) spans of a document, computed once and cached on the document."""
    segs = getattr(doc, "_segs", None)
    if segs is None:
        text = doc.text
        segs = []
        for s, e in candidates(text):
            seg = text[s:e]
            if not _clean(seg):
                continue
            # a contentful claim: needs a digit (the scorer treats a digit-free, filler-only claim as
            # content-free even when it is a verbatim quote) and enough words to say something
            if not re.search(r"\d", seg) or len(seg.split()) < 6:
                continue
            letters = [c for c in seg if c.isalpha()]
            if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
                continue  # headings / all-caps boilerplate
            segs.append((s, e))
        doc._segs = segs
        doc._low = text.lower()
        doc._starts = [s for s, _ in segs]
    return segs


def support_numbers(row: dict, pred=None) -> list:
    """Figures that back the prediction: numeric task-row values and the submitted point (several
    roundings), as strings to look for in a candidate span."""
    nums = set()
    for k, v in row.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("maturity_years",):
            nums.add(str(v))
            nums.add(f"{abs(v):g}")
    if pred is not None and getattr(pred, "point", None) is not None:
        p = abs(float(pred.point))
        for fmt in (".2f", ".3f", ".1f", "g"):
            nums.add(format(p, fmt))
    return [n for n in nums if len(n) >= 3 and n not in ("0.0", "0.00", "0.000", "1.0", "1.00")]


def iter_scored(unit, row: dict, numbers=(), terms=()):
    """Yield (sortkey, doc_id, start, end, text) for every acceptable verbatim span for the entity."""
    eid = row["entity_id"]
    num_res = [re.compile(r"(?<![\d.])" + re.escape(n) + r"(?!\d)") for n in numbers]
    toks = _name_tokens(row)
    eid_l = eid.lower()
    name_l = str(row.get("name", "")).lower().strip()
    tick = 0
    for doc in unit.docs_for(eid):
        if _over_deadline():
            return
        segs = _segments(doc)
        text, low = doc.text, doc._low
        bound_bonus = 0.0 if doc.shared else 1.0  # prefer entity-specific documents
        if len(segs) > FAST_SEGS:
            # very large document: score an even sample plus every segment that contains a needle
            needles = set(toks) | {eid_l} | {t for t in terms} | {n.lower() for n in numbers} | set(_TOPIC_TERMS)
            if len(name_l) >= 4:
                needles.add(name_l)
            idx = set(range(0, len(segs), max(1, len(segs) // 400)))
            for nd in needles:
                pos, cnt = low.find(nd), 0
                while pos != -1 and cnt < 1500:
                    i = bisect.bisect_right(doc._starts, pos) - 1
                    if i >= 0 and segs[i][1] > pos:
                        idx.add(i)
                    pos, cnt = low.find(nd, pos + 1), cnt + 1
            order = sorted(idx)
        else:
            order = range(len(segs))
        for i in order:
            tick += 1
            if tick % 2000 == 0 and _over_deadline():
                return
            s, e = segs[i]
            seg = text[s:e]
            segl = low[s:e]
            score = bound_bonus
            score += sum(2.0 for t in toks if t in segl)
            if eid_l in segl:
                score += 3.0
            if len(name_l) >= 4 and name_l in segl:
                score += 4.0
                if segl.startswith(name_l) or segl.startswith("- " + name_l):
                    score += 3.0
            if segl.startswith(("series:", "source:")):
                score -= 2.0
            if any(b in segl for b in _BOILERPLATE):
                score -= 4.0
            for key in ("ref_month", "quarter_reported", "tenor"):
                val = row.get(key)
                if isinstance(val, str) and len(val) >= 4 and val.lower() in segl:
                    score += 2.5
            if "estimate was revised" in segl and str(row.get("ref_month", "~")).lower() in segl:
                score += 2.0
            score += min(3, len(re.findall(r"\d[\d,.]*", seg))) * 0.4
            if num_res and any(r.search(seg) for r in num_res):
                score += 3.0  # the span quotes a figure that backs the submitted prediction
            if terms:
                score += 3.0 * sum(1 for t in terms if t in segl)  # family evidence phrases
            score += sum(1.2 for t in _TOPIC_TERMS if t in segl)
            # table header / boilerplate lines are poor claims
            if seg.count(" | ") > 6 or segl.startswith("source:") or "table of contents" in segl:
                score -= 3.0
            yield (score, doc.doc_date.toordinal(), -s), doc.doc_id, s, e, seg


def best_corpus_claim(unit, row: dict, avoid: set | None = None, numbers=(), terms=()):
    """The best verbatim span for the entity: (doc_id, s, e, text, score) or None."""
    best = None
    for item in iter_scored(unit, row, numbers, terms):
        if avoid and (item[1], item[2]) in avoid:
            continue
        if best is None or item[0] > best[0]:
            best = item
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[0][0]


def top_candidates(unit, row: dict, limit: int = 8, numbers=(), terms=()):
    """Up to `limit` non-overlapping best spans: list of dicts {doc_id, s, e, text, score}."""
    import heapq

    pool = heapq.nlargest(80, iter_scored(unit, row, numbers, terms), key=lambda it: it[0])
    chosen = []
    for key, doc_id, s, e, seg in pool:
        if any(c["doc_id"] == doc_id and s < c["e"] and c["s"] < e for c in chosen):
            continue
        chosen.append({"doc_id": doc_id, "s": s, "e": e, "text": seg, "score": key[0]})
        if len(chosen) >= limit:
            break
    return chosen


def task_row_claim(unit, row: dict):
    """A verbatim fragment of the entity's own row in the task table (doc_id 'task')."""
    eid = row["entity_id"]
    rs, re_ = unit.task_rows[eid]
    line = unit.task_text[rs:re_]
    body_s, body_e = 1, len(line) - 1  # strip '{' '}'
    limit = min(body_e, body_s + 200)
    frag_end = limit
    if limit < body_e:
        cut = line.rfind(', "', body_s, limit)
        frag_end = cut if cut > body_s + 20 else limit
    s, e = rs + body_s, rs + frag_end
    return TASK_DOC_ID, s, e, unit.task_text[s:e]


def build_claims(unit, row: dict, max_claims: int = 2, numbers=(), terms=()):
    out = []
    got = best_corpus_claim(unit, row, numbers=numbers, terms=terms)
    if got:
        doc_id, s, e, seg, _ = got
        out.append({"doc_id": doc_id, "span_start": s, "span_end": e, "claim": seg})
    if len(out) < max_claims:
        doc_id, s, e, seg = task_row_claim(unit, row)
        out.append({"doc_id": doc_id, "span_start": s, "span_end": e, "claim": seg})
    return out


def evidence_claim(unit, eid: str, ev, max_chars: int = 420):
    """A verbatim claim quoting the passage (doc_id, start, end) that drove a prediction, widened within its line (and
    stopping at any unprintable cell separator) to whole words with some context, so it reads as a statement and
    always carries the figures. None when it is not usable."""
    try:
        doc_id, s0, e0 = ev
        doc = unit.docs.get(doc_id)
        if doc is None or not doc.admits(eid):
            return None
        text = doc.text
        if not _clean(text[s0:e0]):
            return None
        line_s = text.rfind(chr(10), 0, s0) + 1
        line_e = text.find(chr(10), e0)
        line_e = len(text) if line_e == -1 else line_e
        bad = lambda c: not _clean(c)
        s = s0
        while s > max(line_s, s0 - 150) and not bad(text[s - 1]):
            s -= 1
        while s < s0 and s > line_s and not text[s - 1].isspace() and not text[s].isspace():
            s += 1  # start on a word boundary
        e = e0
        while e < min(line_e, e0 + 60) and not bad(text[e]):
            e += 1
        while e > e0 and e < line_e and not text[e].isspace():
            e -= 1  # end on a word boundary
        while s < s0 and text[s].isspace():
            s += 1
        while e > e0 and text[e - 1].isspace():
            e -= 1
        if e - s > max_chars:
            s, e = s0, e0
        seg = text[s:e]
        if len(seg.split()) < 6 or not _clean(seg) or not any(c.isdigit() for c in seg):
            return None
        return {"doc_id": doc_id, "span_start": s, "span_end": e, "claim": seg}
    except Exception:
        return None
