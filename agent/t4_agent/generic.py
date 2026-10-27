"""Generic solver for task families without a dedicated predictor (the held-out set spans families that have no published
counterpart). It infers what the target measures from the task itself and never carries a level forward as a change.

Regression / numeric truth
  * target kind: 'change' (change, growth, return, revision, bps, MoM/YoY, abnormal, surprise ...) or 'level'
  * point: a row column that measures the SAME quantity (shares the target's name tokens and its kind) carried forward;
    else the entity's own history table in the corpus (header matched to the target name or to the entity's identifiers):
    level target -> 0.5 x last + 0.5 x mean of the last 6 values; change target whose table holds the same change -> same;
    change target whose table holds levels -> 0 (the naive no-change point);
    else 0 for a change target, the nearest prior-style column for a level target
  * band: from the history table when one is found (sd of the series, or sd of its period changes scaled by sqrt(horizon
    periods) when the table holds levels of a change target); otherwise a wide default by unit (misses cost 20x their distance)
Classification: status-quo labels first (no / not / none / unchanged / hold / flat / inline / within / stable ...), then the
  historically more frequent direction words (up / beat / positive ...), else the first label.
Ranking: a constant point for every row (the scorer's constant-forecast quality equals the ranking anchor, 0.5), with
  data-driven bands.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import statistics

_CHANGE = re.compile(r"(?i)change|growth|return|revision|revise|delta|surprise|\bbps\b|basis.point|_bps|mom\b|_mom|mom_|yoy|abnormal|spread_change|pct_change|percent.change|reaction|\bmove|shift|impact")
_STATUS_QUO = ("no_", "no ", "not", "none", "neither", "unchanged", "no_change", "hold", "flat", "inline", "within", "stable", "same", "maintain")
_UPWARD = ("up", "increase", "higher", "rise", "above", "positive", "beat", "gain", "expand", "accelerat")
_PRIOR_TAGS = ("latest", "prior", "last", "previous", "trailing", "current", "start", "consensus", "baseline", "ago", "reference",
               "benchmark", "expected", "street", "forecast", "estimate", "guidance", "opening")
_BASE_TAGS = ("consensus", "prior", "last", "latest", "estimate", "current", "ago", "previous", "reference", "benchmark", "expected",
              "street", "forecast", "guidance", "baseline", "opening")
_STOP = {"pct", "the", "of", "a", "an", "to", "in", "for", "and", "rank", "first", "print", "value", "id", "usd", "bn"}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _tokens(s) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t and t not in _STOP and not t.isdigit()}


def _fnum(s):
    try:
        v = float(str(s).replace(",", "").replace("$", "").replace("%", "").replace("+", "").strip())
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def target_kind(unit, row=None) -> str:
    t = unit.task.get("target") or {}
    txt = " ".join([str(t.get("name", "")), str(t.get("unit", "")), str(t.get("units", ""))])
    if row is not None:
        txt += " " + " ".join(str(row.get(k, "")) for k in ("unit", "units"))
    return "change" if _CHANGE.search(txt) else "level"


def _col_kind(name: str) -> str:
    return "change" if _CHANGE.search(name) else "level"


def _tables(text: str):
    """Yield (header_cells, rows) for every pipe table in the text."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if " | " in lines[i]:
            head = [c.strip() for c in lines[i].split("|")]
            rows = []
            j = i + 1
            while j < len(lines) and " | " in lines[j]:
                rows.append([c.strip() for c in lines[j].split("|")])
                j += 1
            if len(rows) >= 4:
                yield head, rows
            i = j
        else:
            i += 1


def _dates(col):
    out = []
    for v in col:
        m = re.match(r"(\d{4})-(\d{2})(?:-(\d{2}))?$", str(v).strip())
        if not m:
            return None
        out.append(dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 15)))
    return out


_CUE = re.compile(r"(?i)month-over-month|year-over-year|quarter-over-quarter|week-over-week|percent changes?|% changes?|\bm/m\b|\by/y\b|\bmom\b|\byoy\b|growth rates?|annualized rate|first prints? of the (?:monthly|quarterly) (?:changes?|growth)")


def _lag1(vals) -> float:
    m = statistics.mean(vals)
    d = [v - m for v in vals]
    den = sum(x * x for x in d)
    return sum(a * b for a, b in zip(d[:-1], d[1:])) / den if den > 0 else 1.0


def history(unit, row):
    """(values, period_days or None) of the entity's own history for the target, or None."""
    t = unit.task.get("target") or {}
    want = _tokens(t.get("name", ""))
    ident = set()
    for k in ("entity_id", "name", "series_fred", "series_id", "tenor", "ticker", "series_name"):
        v = row.get(k)
        if isinstance(v, str) and v:
            ident |= _tokens(v)
            ident.add(v.lower())
    best = None
    for shared in (False, True):
        for doc in unit.docs_for(row["entity_id"], shared=shared):
            for head, rows in _tables(doc.text):
                for ci, cell in enumerate(head[1:], 1):
                    ct = _tokens(cell)
                    score = 2 * len(ct & want) + (3 if cell.lower() in ident else 0) + len(ct & ident)
                    if score <= 0:
                        continue
                    vals = [_fnum(r[ci]) for r in rows if len(r) > ci]
                    vals = [v for v in vals if v is not None]
                    if len(vals) < 4:
                        continue
                    if best is None or score > best[0]:
                        ds = _dates([r[0] for r in rows if len(r) > ci and _fnum(r[ci]) is not None])
                        gap = None
                        if ds and len(ds) >= 3:
                            g = sorted((b - a).days for a, b in zip(ds, ds[1:]) if (b - a).days > 0)
                            gap = g[len(g) // 2] if g else None
                        best = (score, vals, gap, cell, bool(_CUE.search(doc.text[:700])) or bool(_CHANGE.search(cell)))
        if best is not None:
            break
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _horizon_days(unit):
    t = unit.task
    try:
        a = dt.date.fromisoformat(str(t.get("cutoff_date"))[:10])
        b = dt.date.fromisoformat(str(t.get("resolution_date"))[:10])
        return max((b - a).days, 1)
    except ValueError:
        return None


LEVELS_AUTOCORR = 0.0   # D18 experiment switch: a level series also has strongly autocorrelated VALUES (0 = previous behaviour)


def _is_levels(vals) -> bool:
    """A level series (yields, net positions, ratios) vs a series of changes that hovers around zero."""
    m, s = statistics.mean(vals), statistics.pstdev(vals)
    if not (abs(m) > 3 * s and (min(vals) > 0 or max(vals) < 0)):
        return False
    if LEVELS_AUTOCORR > 0 and len(vals) >= 12 and s > 0:
        d = [v - m for v in vals]
        ac = sum(a * b for a, b in zip(d[:-1], d[1:])) / sum(x * x for x in d)
        return ac >= LEVELS_AUTOCORR
    return True


def _default_half(unit, row, point) -> float:
    txt = " ".join(str(x) for x in ((unit.task.get("target") or {}).get("name", ""), row.get("unit", ""), row.get("units", ""))).lower()
    if "bps" in txt or "basis" in txt:
        return 60.0
    if "prob" in txt:
        return 0.3
    if re.search(r"pct|percent|growth|return", txt):
        return max(60.0 if "growth" in txt or "yoy" in txt else 3.0, 2 * abs(point))
    return max(0.5 * abs(point), 1.0)


def regress(unit, row):
    """-> (point, lo, hi, why)"""
    kind = target_kind(unit, row)
    want = _tokens((unit.task.get("target") or {}).get("name", ""))
    nums = [(k, v) for k, v in row.items() if _num(v)]
    def _is_same(k):
        ov = len(_tokens(k) & want)
        tagged = any(t in k.lower() for t in _PRIOR_TAGS)
        return _col_kind(k) == kind and (ov >= 2 or (ov >= 1 and tagged))   # a prior tag plus a shared token, or two shared tokens
    same = [(len(_tokens(k) & want), k, v) for k, v in nums if _is_same(k)]
    same.sort(key=lambda x: (-x[0], 0 if any(t in x[1] for t in _PRIOR_TAGS) else 1))
    h = history(unit, row)
    point, why = None, ""
    if same:
        point, why = float(same[0][2]), f"same-quantity column '{same[0][1]}' carried forward"
    if h is not None and GLOBAL_ON:
        try:
            r = _global_regress(unit, row, kind, h)
        except Exception:
            r = None
        if r is not None:
            return r
    if h is not None and BSC_ON:
        try:
            r = _bsc_regress(unit, row, kind, h)
        except Exception:
            r = None
        if r is not None:
            return r
    if h is not None:
        vals, gap, cell, cue = h
        levels = _is_levels(vals)
        # D18 (exp G40): a table whose own title / header says it holds changes (month-over-month, % change, growth ...) and whose values are not
        # near-unit-root (autocorrelation < 0.97) is a series of changes even when its mean is 3 sd from zero (e.g. core CPI m/m +0.3 +- 0.1)
        if levels and cue and len(vals) >= 12 and _lag1(vals) < 0.97:
            levels = False
        # D18: a history column named like the target IS the target (a YOY / MOM series with a positive mean looks like 'levels' to
        # _is_levels, which would force a zero point and a change-based band); only a differently-named column holds levels to difference.
        own_series = bool(want) and want <= _tokens(cell)
        if kind == "level" or not levels or own_series:
            series_pt = 0.5 * vals[-1] + 0.5 * statistics.mean(vals[-6:])
            why_pt = f"0.5 x last + 0.5 x mean of the last 6 values of '{cell}' in the corpus"
            if kind == "change" and len(vals) >= 12:
                # D18 (exp G17, 30 change-type first-print series, 2013-19 / 2021-25): a mean-reverting AR(1) on the deviation from the
                # last-36 mean beat 0.5 last + 0.5 mean6 on both periods (skill vs last 0.533 -> 0.566 and 0.525 -> 0.546)
                w = vals[-36:]
                mu = statistics.mean(w)
                dev = [x - mu for x in w]
                den = sum(x * x for x in dev[:-1])
                if len(dev) >= 8 and den > 1e-12:
                    phi = min(max(sum(a * b for a, b in zip(dev[:-1], dev[1:])) / den, -0.2), 0.98)
                    # G20: a 50/50 blend with the faster-adapting shipped rule kept the gain on 30 macro series (+4% both periods) and on
                    # 10 CPI components over 1970-2025 splits, and limited the loss in the 2021-25 inflation regime shift (-2.4% vs -6.4% pure AR)
                    series_pt = 0.5 * (mu + phi * dev[-1]) + 0.5 * series_pt
                    why_pt = f"0.5 x AR(1) toward the last-{len(w)} mean (phi {phi:.2f}) + 0.5 x (0.5 last + 0.5 mean6) of '{cell}'"
            if point is None:
                point, why = series_pt, why_pt
            sd = statistics.pstdev(vals[-24:]) if len(vals) >= 4 else abs(point) * 0.2
            n = len(vals[-24:])
            tf = 2.08 if n >= 9 else 2.5
            half = max(tf * sd, 1e-6)
            why += f"; band {tf} x sd {sd:.4g} of {n} history values"
            # D18 (exp G9b/G9d, 53 first-print macro series, validation 2013-19 / test 2021-25, horizons 1-3): the 2.08 x sd band covered
            # 93-98% (target 90%) and lost the interval score; a mean of 1.645 x sd and the 90th percentile of the history's own
            # walk-forward errors for this very rule won in 74-100% of series (level) / 62-89% (change) and cut the scaled score 10-20%.
            hd, per = _horizon_days(unit), 1
            if hd and gap:
                per = max(1, round(hd / gap * (5 / 7 if gap <= 1.5 else 1.0)))
            per = min(per, 6)
            errs = [abs(vals[j + per] - (0.5 * vals[j] + 0.5 * statistics.mean(vals[max(0, j - 5):j + 1])))
                    for j in range(max(11, len(vals) - 1 - per - 24), len(vals) - per)]
            if len(vals) >= 20 + per and len(errs) >= 8:
                q90 = sorted(errs)[min(len(errs) - 1, int(math.ceil(0.9 * len(errs))) - 1)]
                half = max(0.5 * (1.645 * sd + q90), 1e-6)
                why += f"; D18 band = mean of 1.645 x sd {sd:.4g} and the 90th percentile {q90:.4g} of {len(errs)} walk-forward errors at {per} period(s)"
        else:
            if point is None:
                point, why = 0.0, f"change target, '{cell}' holds levels: no-change point"
            diffs = [b - a for a, b in zip(vals, vals[1:])]
            sd = statistics.pstdev(diffs[-60:]) if len(diffs) >= 3 else abs(vals[-1]) * 0.02
            hd = _horizon_days(unit)
            periods = 1.0
            if hd and gap:
                periods = hd / gap * (5 / 7 if gap <= 1.5 else 1.0)
            scale = 100.0 if "bps" in " ".join([str((unit.task.get('target') or {}).get('name', '')), str(row.get('unit', ''))]).lower() and max(abs(v) for v in vals) < 30 else 1.0
            half = 1.96 * sd * math.sqrt(max(periods, 1.0)) * scale
            why += f"; band 1.96 x sd of period changes x sqrt({periods:.0f} periods)" + (" x 100 (percent -> bps)" if scale != 1.0 else "")
        return point, point - half, point + half, why
    if point is None:
        if kind == "change":
            point, why = 0.0, "change target with no same-quantity input: no-change point"
        else:
            hit = next((v for k, v in nums if any(t in k for t in _PRIOR_TAGS)), None)
            point = float(hit) if hit is not None else 0.0
            why = "level target: nearest prior-style column carried forward"
    half = _default_half(unit, row, point)
    return point, point - half, point + half, why + "; wide default band (no history found)"


BSC_ON = False  # G1 (off: no out-of-time support): backtest-select-combine on the unit's own history tables (see bsc.py)


def _bsc_periods(unit, gap):
    hd, per = _horizon_days(unit), 1
    if hd and gap:
        per = max(1, round(hd / gap * (5 / 7 if gap <= 1.5 else 1.0)))
    return per


def _bsc_mode(kind, h):
    vals, gap, cell, cue = h
    return "delta" if (kind == "change" and not cue and not _CHANGE.search(cell)) else "direct"


def _bsc_unit(unit, kind):
    """Unit-level selection, pooled over every entity whose history table matches the target (cached on the unit)."""
    cache = getattr(unit, "_bsc_cache", None)
    if cache is not None:
        return cache
    from . import bsc as B
    hs = {}
    for r in unit.entities:
        try:
            hh = history(unit, r)
        except Exception:
            hh = None
        if hh is not None:
            hs[r["entity_id"]] = hh
    cache = None
    if hs:
        modes = [_bsc_mode(kind, hh) for hh in hs.values()]
        mode = max(set(modes), key=modes.count)
        gaps = sorted(hh[1] for hh in hs.values() if hh[1])
        per = _bsc_periods(unit, gaps[len(gaps) // 2] if gaps else None)
        series = [hh[0] for hh in hs.values() if _bsc_mode(kind, hh) == mode]
        if mode == "direct":
            w, rep = B.select_direct(series, per)
            active = set(w) != {"naive"}
        else:
            w, rep = B.select_delta(series, per)
            active = bool(w)
        cache = dict(mode=mode, per=per, w=w, rep=rep, active=active)
    unit._bsc_cache = cache
    return cache


def _bsc_regress(unit, row, kind, h):
    """BSC forecast for one entity, or None (then the established generic rule applies)."""
    from . import bsc as B
    c = _bsc_unit(unit, kind)
    if not c or not c["active"] or _bsc_mode(kind, h) != c["mode"]:
        return None
    vals, gap, cell, cue = h
    per, w = c["per"], c["w"]
    scale = 1.0
    if c["mode"] == "delta":
        tx = " ".join([str((unit.task.get("target") or {}).get("name", "")), str(row.get("unit", ""))]).lower()
        if "bps" in tx and max(abs(v) for v in vals) < 30:
            scale = 100.0
        fc = lambda v, hh: B.forecast_delta(v, hh, w)
        point = scale * fc(vals, per)
    else:
        fc = lambda v, hh: B.forecast_direct(v, hh, w)
        point = fc(vals, per)
    q = B.band_from_backtest(vals, per, fc, delta=c["mode"] == "delta")
    if q is not None:
        lo, hi = point + scale * q[0], point + scale * q[1]
        bt = f"; band = this entity's 5th-95th percentile walk-forward errors at {per} period(s)"
    else:
        sd = statistics.pstdev(vals[-24:]) if c["mode"] == "direct" else statistics.pstdev([b - a for a, b in zip(vals, vals[1:])]) * math.sqrt(per)
        half = max((2.08 if c["mode"] == "direct" else 1.96) * sd * scale, 1e-6)
        lo, hi = point - half, point + half
        bt = f"; band from the sd of '{cell}'"
    return point, min(lo, point), max(hi, point), c["rep"] + f" ('{cell}', {c['mode']} mode, horizon {per} period(s))" + bt


GLOBAL_ON = True   # G2: global LightGBM forecaster (gforecast.py) on the entity's own history table


def _global_regress(unit, row, kind, h):
    """Global-model forecast for one entity, or None (then the established generic rule applies)."""
    from . import gforecast as GF
    vals, gap, cell, cue = h
    ppy = GF.ppy_of_gap(gap)
    if ppy is None:
        return None
    per = _bsc_periods(unit, gap)
    mode = _bsc_mode(kind, h)
    r = GF.forecast(vals, per, ppy, unit.task.get("cutoff_date"))
    if r is None:
        return None
    scale = 1.0
    if mode == "delta":
        tx = " ".join([str((unit.task.get("target") or {}).get("name", "")), str(row.get("unit", ""))]).lower()
        if "bps" in tx and max(abs(v) for v in vals) < 30:
            scale = 100.0
        point, lo, hi = scale * r["delta"], scale * r["lo"], scale * r["hi"]
    else:
        point, lo, hi = vals[-1] + r["delta"], vals[-1] + r["lo"], vals[-1] + r["hi"]
    why = (f"global forecaster (LightGBM on first-release macro and market series, data limit {r['model']}): "
           f"{'change' if mode == 'delta' else 'next value'} of '{cell}' {per} period(s) ahead from its last {len(vals)} values; "
           f"band = the model's 5th-95th percentile")
    return point, min(lo, point), max(hi, point), why


def class_half(unit, row, base) -> float:
    """Half-width for the numeric leg of a classification unit when no history is found."""
    name = str((unit.task.get("target") or {}).get("name", "")).lower()
    if base is not None and "revis" in name:
        return max(abs(base) * 0.015, 1e-6)          # revisions of a published level are small relative to the level
    thr = next((float(v) for k, v in row.items() if re.search(r"threshold|neutral|flat|band|cutoff|tolerance", k.lower()) and _num(v) and 0 < v <= 50), None)
    if thr is not None:
        return 7.0 * thr                              # outcomes spread several times a 'flat' threshold
    if base is not None:
        return max(abs(base) * 0.3, 1.0)
    return _default_half(unit, row, 0.0)


def choose_label(labels):
    low = [l.lower() for l in labels]
    for i, l in enumerate(low):
        if any(l.startswith(s) or l.replace("-", "_").startswith(s) for s in _STATUS_QUO) or l in ("no", "none", "false"):
            return labels[i]
    for w in _UPWARD:
        for i, l in enumerate(low):
            if l.startswith(w):
                return labels[i]
    return labels[0] if labels else None


_UP_WORDS = re.compile(r"(?i)(?<![A-Za-z0-9_])(higher|above|greater|more|increase|increased|rise|rises|risen|up|accelerat\w*|exceed\w*)(?![A-Za-z0-9_])")
_DOWN_WORDS = re.compile(r"(?i)(?<![A-Za-z0-9_])(lower|below|less|decrease|decreased|fall|falls|fallen|down|declin\w*|decelerat\w*)(?![A-Za-z0-9_])")
_VS_LAST = re.compile(r"(?i)(previous|prior|preceding|last|latest|most recent)\s+(print|release|reading|value|observation|report|month|week|period|quarter|figure|number|publication)|than\s+(the\s+)?(previous|prior|last|preceding)")
_NOT_PLAIN = re.compile(r"(?i)revis|year|consensus|estimate|forecast|expect|\d|%|percent|basis|threshold|at least|by more|exceed")


def direction_label(unit, row):
    """For a two-way up/down label whose assertion compares the NEXT value with the latest published one, and where the entity's own
    history table is a series of changes: the next value is above the latest iff the mean of the last 6 values is above it
    (D18, exp G35: 30 change-type first-print series, accuracy 0.70 / 0.66 vs 0.51 for always-up; level-type series 0.62 / 0.53).
    Returns the label or None when the task is not that shape."""
    try:
        t = unit.task.get("target") or {}
        labels = t.get("labels") or []
        if len(labels) != 2:
            return None
        asr = t.get("label_assertions") or {}
        up = [l for l in labels if _UP_WORDS.search(str(asr.get(l, l))) and not _DOWN_WORDS.search(str(asr.get(l, l)))]
        dn = [l for l in labels if _DOWN_WORDS.search(str(asr.get(l, l))) and not _UP_WORDS.search(str(asr.get(l, l)))]
        if len(up) != 1 or len(dn) != 1:
            return None
        text = " ".join(str(asr.get(l, "")) for l in labels)
        if not _VS_LAST.search(text) or _NOT_PLAIN.search(text):
            return None
        h = history(unit, row)
        if h is None:
            return None
        vals = h[0]
        if len(vals) < 8 or target_kind(unit, row) != "change":
            return None
        return up[0] if statistics.mean(vals[-6:]) > vals[-1] else dn[0]
    except Exception:
        return None
