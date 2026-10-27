"""Compact fitted artifacts (JSON) built offline from public, licensed, pre-cutoff data (see ARTIFACT_PROVENANCE.md).

`get(name)` returns the artifact dict or None. A missing file, an unreadable file, a wrong schema or `T4_ARTIFACTS=0` all give
None, and every caller then runs the exact v2 code path (byte-identical fallback). Artifacts hold numbers only (coefficients,
multipliers), never raw data and never task answers.
"""
from __future__ import annotations

import json
import os

_CACHE: dict = {}


def _load():
    if "all" in _CACHE:
        return _CACHE["all"]
    data = None
    if os.environ.get("T4_ARTIFACTS", "1") != "0":
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts.json")
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("families"), dict):
                data = None
        except Exception:
            data = None
    _CACHE["all"] = data
    return data


_GUARD = {"cutoff": None}
GUARD_DAYS = 30


def set_task_cutoff(cutoff) -> None:
    """Cutoff guard: fitted artifacts are used only for tasks whose cutoff is at least GUARD_DAYS after the artifacts' data limit."""
    _GUARD["cutoff"] = cutoff


def guard_ok(data_limit: str) -> bool:
    import datetime as _dt
    c = _GUARD["cutoff"]
    if c is None:
        return False          # unknown cutoff: fitted values cannot be shown to predate it
    try:
        lim = _dt.date.fromisoformat(str(data_limit)[:10])
        cd = c if isinstance(c, _dt.date) else _dt.date.fromisoformat(str(c)[:10])
    except Exception:
        return False
    return cd >= lim + _dt.timedelta(days=GUARD_DAYS)


def get(name: str):
    data = _load()
    if not data or not guard_ok(data.get("data_limit", "9999-12-31")):
        return None
    fam = data["families"].get(name)
    return fam if isinstance(fam, dict) and fam.get("enabled", True) else None
