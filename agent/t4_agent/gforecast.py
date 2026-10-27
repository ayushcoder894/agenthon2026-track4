"""Global forecaster at run time: LightGBM trees (exported to JSON, evaluated by gbm.py) trained on hundreds of first-release macro and
never-revised market series, on scale-free features (gfeat.py). Several copies are shipped, each fitted only on data published up to its
data limit; a task uses the latest copy whose data limit is at least 6 months before the task's cutoff (cutoff guard), else none."""
from __future__ import annotations

import datetime as dt
import json
import os

from . import gbm
from .gfeat import features

_M = None
GUARD_DAYS = 183


def _models():
    global _M
    if _M is None:
        try:
            _M = json.load(open(os.path.join(os.path.dirname(__file__), "global_models.json"), encoding="utf-8"))
        except Exception:
            _M = {"models": []}
    return _M


def pick(cutoff):
    """The model copy usable for a task with this cutoff (date or ISO string), or None."""
    try:
        c = cutoff if isinstance(cutoff, dt.date) else dt.date.fromisoformat(str(cutoff)[:10])
    except Exception:
        return None
    ok = [m for m in _models().get("models", []) if dt.date.fromisoformat(m["data_limit"]) + dt.timedelta(days=GUARD_DAYS) <= c]
    return max(ok, key=lambda m: m["data_limit"]) if ok else None


def ppy_of_gap(gap_days):
    if gap_days is None:
        return None
    if 25 <= gap_days <= 35:
        return 12
    if 6 <= gap_days <= 8:
        return 52
    return None


def forecast(vals, h, ppy, cutoff):
    """-> dict(delta, lo, hi, model) in the table's units, relative to the last value, or None."""
    m = pick(cutoff)
    if m is None or ppy not in (12, 52) or h < 1:
        return None
    fs = features(list(vals), int(h), ppy)
    if fs is None:
        return None
    f, s = fs
    d = m["lam"] * gbm.predict(m["pt"], f)
    lo, hi = gbm.predict(m["q05"], f), gbm.predict(m["q95"], f)
    lo, hi = min(lo, d), max(hi, d)
    return dict(delta=d * s, lo=lo * s, hi=hi * s, model=m["data_limit"], scale=s)
