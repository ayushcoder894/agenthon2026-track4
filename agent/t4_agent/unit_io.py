"""Load a Track 4 unit (task, corpus index, entity bindings) using only the standard library.

Trust model: entity binding and document existence come from the manifests (the corpus index
`corpus/manifest.json` and, if present beside the task, the unit `manifest.json`), never from the
documents themselves. A document is citable only if it is declared `role: corpus`, carries an entity
label (`entity_ids` or `shared: true`), has a parseable `doc_date` on or before the cutoff and a
text we can resolve exactly as the scorer does (flat `text`, else `spans[].text` joined by one space).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field

DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TASK_DOC_ID = "task"


@dataclass
class Doc:
    doc_id: str
    text: str
    doc_date: dt.date
    entity_ids: tuple | None
    shared: bool

    def admits(self, entity_id: str) -> bool:
        return self.shared or (self.entity_ids is not None and entity_id in self.entity_ids)


@dataclass
class Unit:
    task: dict
    cutoff: dt.date | None
    interval_level: float
    target_type: str | None
    entities: list  # list of row dicts, file order
    roster: list  # entity ids, file order
    docs: dict = field(default_factory=dict)  # doc_id -> Doc (citable only)
    task_text: str = ""
    task_rows: dict = field(default_factory=dict)  # entity_id -> (start, end)
    notes: list = field(default_factory=list)

    def docs_for(self, entity_id: str, shared: bool | None = None) -> list:
        out = []
        for doc in self.docs.values():
            if not doc.admits(entity_id):
                continue
            if shared is not None and doc.shared != shared:
                continue
            out.append(doc)
        return sorted(out, key=lambda d: (d.doc_date, d.doc_id), reverse=True)


def parse_date(value) -> dt.date | None:
    if not isinstance(value, str) or not ISO_DATE_RE.match(value):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def doc_text(doc: dict) -> str | None:
    """The text a CLAIM citation indexes. The track scorer's claim check (`_span_text`) reads ONLY the flat
    `text` field: a citation into a spans-only document resolves to empty text and is out of range, so the
    claim is false (the shared helper's spans fallback is not what the claim rules use). Documents without a
    string `text` are therefore not citable and are skipped."""
    text = doc.get("text")
    return text if isinstance(text, str) else None


def task_table_text(task: dict):
    lines, ranges, offset = [], {}, 0
    for row in task.get("entities", []):
        line = json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
        ranges[row["entity_id"]] = (offset, offset + len(line))
        lines.append(line)
        offset += len(line) + 1
    return "\n".join(lines), ranges


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _manifest_labels(paths) -> dict:
    """doc_id -> (entity_ids tuple | None, shared bool) from every readable manifest (first wins)."""
    labels: dict = {}
    for path in paths:
        manifest = _read_json(path) if os.path.isfile(path) else None
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
            continue
        for entry in manifest["files"]:
            if not isinstance(entry, dict) or entry.get("role") != "corpus":
                continue
            rel = entry.get("path")
            if not isinstance(rel, str):
                continue
            base = os.path.basename(rel)
            if not base.endswith(".json") or base == "manifest.json":
                continue
            doc_id = base[: -len(".json")]
            ids = entry.get("entity_ids")
            if isinstance(ids, list) and all(isinstance(x, str) for x in ids):
                ids = tuple(ids)
            else:
                ids = None
            shared = entry.get("shared") is True
            if doc_id not in labels and (ids is not None or shared):
                labels[doc_id] = (ids, shared)
    return labels


def load_unit(task_path: str, corpus_dir: str) -> Unit:
    task = _read_json(task_path)
    if not isinstance(task, dict):
        raise ValueError("task.json unreadable")
    rows = [r for r in task.get("entities", []) if isinstance(r, dict) and isinstance(r.get("entity_id"), str)]
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    level = task.get("interval_level")
    level = float(level) if isinstance(level, (int, float)) and not isinstance(level, bool) else 0.9
    unit = Unit(
        task=task,
        cutoff=parse_date(task.get("cutoff_date")),
        interval_level=level,
        target_type=target.get("type") if target.get("type") in ("classification", "regression", "ranking") else None,
        entities=rows,
        roster=[r["entity_id"] for r in rows],
    )
    unit.task_text, unit.task_rows = task_table_text({"entities": rows})

    task_dir = os.path.dirname(os.path.abspath(task_path))
    corpus_dir = os.path.abspath(corpus_dir)
    labels = _manifest_labels(
        [
            os.path.join(corpus_dir, "manifest.json"),
            os.path.join(task_dir, "manifest.json"),
            os.path.join(os.path.dirname(corpus_dir), "manifest.json"),
        ]
    )
    if not labels:
        unit.notes.append("no labelled manifest found: corpus documents are not citable")
    try:
        names = sorted(os.listdir(corpus_dir))
    except OSError:
        names = []
    for name in names:
        if not name.endswith(".json") or name == "manifest.json":
            continue
        doc_id = name[: -len(".json")]
        if not DOC_ID_RE.match(doc_id) or doc_id not in labels or doc_id == TASK_DOC_ID:
            continue
        raw = _read_json(os.path.join(corpus_dir, name))
        if not isinstance(raw, dict):
            continue
        text = doc_text(raw)
        date = parse_date(raw.get("doc_date"))
        if text is None or date is None:
            continue
        if unit.cutoff is not None and date > unit.cutoff:
            continue  # embargo: never read, never cite
        ids, shared = labels[doc_id]
        unit.docs[doc_id] = Doc(doc_id, text, date, ids, shared)
    return unit
