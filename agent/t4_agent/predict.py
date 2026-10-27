"""Deterministic, evidence-light predictions (no model).

Dispatch is by the SHAPE of the task (columns present, target name, target type), never by a
hard-coded family slug, and every branch ends in a generic fallback so an unseen family still gets a
sane, finite, schema-valid prediction. Each prediction records a short `why` string for the report.

Philosophy: the scorer measures skill against a hidden naive rule (usually "carry the prior value
forward"). We reproduce a best-guess naive proxy from the task table and only move it by a small,
bounded, documented amount derived from dispersion that is visible in the pre-cutoff corpus.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import statistics
from dataclasses import dataclass

from . import artifacts as _art
from .eps_parse import parse_pair as _parse_eps_pair

Z90 = 1.645


@dataclass
class Pred:
    label: str | None = None
    point: float | None = None
    lo: float | None = None
    hi: float | None = None
    why: str = ""
    rule: str = ""  # number-free description of the rule (used for deterministic submitted_reasons)
    terms: tuple = ()  # phrases a supporting passage should contain (claim selection bonus)
    fn: str = ""  # name of the predictor that produced this prediction (used to keep point and label coherent)
    extra_numbers: tuple = ()  # figures a supporting passage should quote (claim selection bonus)
    evidence: tuple = ()  # (doc_id, start, end) of the passage that drove the prediction (cited first when valid)


RULE_TERMS = {
    "_credit": ("going concern", "substantial doubt", "liquidity", "covenant", "net loss", "maturit"),
    "_eps_direction": ("diluted", "earnings per share", "net income"),
    "_eps_growth": ("net interest income", "diluted", "earnings per share", "net income"),
    "_eps_beat": ("diluted", "earnings per share", "net sales"),
    "_postearn": ("net sales", "operating income", "revenue", "earnings"),
    "_macrorev": ("revised",),
    "_cpi": ("first print",),
    "_auction": ("bid-to-cover", "bid_to_cover"),
}


RULE_TEXT = {
    "_cpi": "the last published monthly change is shrunk halfway toward the average of the recent monthly changes in the table",
    "_rates": "the yield is assumed unchanged over the window (the starting yield carried forward) with a band scaled to recent daily volatility over the horizon",
    "_auction": "the bid-to-cover ratio is the average of the tenor's most recent auctions",
    "_cot": "net positioning mean-reverts toward its recent average",
    "_macrorev": "revisions follow the majority direction of past revisions in the vintage tables and the latest estimate is kept as the value",
    "_credit": "explicit going-concern language in the filings signals a credit event, otherwise the base rate of no credit event applies",
    "_eps_direction": "the base rate of year-over-year EPS increases is applied to the prior-year EPS",
    "_eps_growth": "prior-year EPS is carried forward (zero growth) with a wide band for EPS dispersion",
    "_eps_beat": "companies more often beat than miss consensus, so the consensus EPS is lifted slightly",
    "_postearn": "the modal outcome for the one-day abnormal return of a mega-cap company is a small move",
}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _float(s):
    try:
        v = float(str(s).replace(",", "").replace("+", "").strip())
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def parse_table(text: str, header_pred):
    """Pipe tables: return (header_cells, rows[list of cells]) for the first table whose header
    satisfies header_pred. A table is the header line plus following lines that contain ' | '."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if " | " not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if not header_pred(cells):
            continue
        rows = []
        for nxt in lines[i + 1 :]:
            if " | " not in nxt:
                break
            rows.append([c.strip() for c in nxt.split("|")])
        return cells, rows
    return None, []


def _sd(values):
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


# one-sided 95% Student-t quantiles, df = 1..30 (90% two-sided prediction interval)
_T95 = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761, 1.753,
        1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697]


def t_pi_factor(n: int) -> float:
    """Multiplier on the POPULATION sd of n observations for a 90% prediction interval of one new observation:
    t_{0.95,n-1} * sqrt(1 + 1/n) * sqrt(n/(n-1))  (n = 9 gives 2.08). Falls back to 2.5 for n < 4."""
    if n < 4:
        return 2.5
    t = _T95[n - 2] if n - 1 <= 30 else 1.645
    return t * math.sqrt(1 + 1 / n) * math.sqrt(n / (n - 1))


def _interval(point, half, level_floor=0.0):
    half = max(half, level_floor)
    return point - half, point + half


def _entity_text_docs(unit, eid):
    return [d for d in unit.docs_for(eid)]


# ---------------------------------------------------------------------------- regression families
def _cpi(unit, row):
    eid, name = row["entity_id"], row.get("name", "")
    last = row["latest_published_mom_pct"]
    series = None
    for doc in _entity_text_docs(unit, eid):
        cells, rows = parse_table(doc.text, lambda c: len(c) > 2 and c[0] == "month" and name in c)
        if cells:
            col = cells.index(name)
            vals = [_float(r[col]) for r in rows if len(r) > col and re.match(r"\d{4}-\d{2}$", r[0])]
            series = [v for v in vals if v is not None]
            if series:
                break
    if not series:
        series = [last]
    recent = series[-6:]
    point = 0.5 * last + 0.5 * statistics.mean(recent)
    sd = _sd(series) if len(series) >= 4 else max(abs(last) * 0.5, 0.2)
    mult, band_txt = Z90, "1.645"
    art = _art.get("cpi_components")
    if art and len(series) >= 4:
        # v4 artifact (ALFRED real-time vintages, 11 CPI components, 451 held-out first prints): the 1.645 x sd band covered only
        # 82%; the small-sample Student-t prediction interval for the history length reached 86-90% and a lower interval score.
        mult, band_txt = t_pi_factor(len(series)), f"t-PI factor {t_pi_factor(len(series)):.2f}"
    if _devonly() and abs(last) > 0.8:
        point = 0.0     # DEV-ONLY in-sample: a large last print in a volatile component is not carried forward
    if len(series) >= 5 and series[-1] != 0 and series[-2] != 0:
        # turning-point rule (history 1985-2012 fit, b = 0.274; same sign of gain on 2013-18 and 2019-24 turning events):
        # after >= 3 same-sign prints, a print of the opposite sign tends to continue in the new direction
        s2 = 1 if series[-2] > 0 else -1
        run = 0
        for v in reversed(series[:-1]):
            if v != 0 and (1 if v > 0 else -1) == s2:
                run += 1
            else:
                break
        if run >= 3 and (1 if series[-1] > 0 else -1) != s2:
            point += float(_cs_art().get("turning_b", 0.274)) * (1 if series[-1] > 0 else -1) * abs(statistics.mean(series[-4:-1]))
    try:
        point += _cpi_xcomp_adj(unit, row)
    except Exception:
        pass
    if _CPI_COMP_BY_FRED.get(str(row.get("series_fred", ""))) == "apparel":
        # G1: apparel m/m first prints are best forecast by zero (1990-2024 first prints, MAE 0.468 vs 0.493 for the shipped blend over all
        # 437 months; 0.589 vs 0.681 after spikes; 0.370 vs 0.447 in Octobers); no fitted parameter
        point = 0.0
    lo, hi = _interval(point, mult * sd, float(art["floor"]) if art else 0.25)
    pumped = _cpi_pump(unit, row, series)
    if pumped is not None:
        return pumped
    return Pred(point=round(point, 3), lo=round(lo, 3), hi=round(hi, 3),
                why=f"midpoint of last print {last} and mean of last {len(recent)} months {statistics.mean(recent):.3f}; band = {band_txt} x sd {sd:.3f} of {len(series)} monthly prints")


def _cpi_table(unit, eid):
    """{column name: list of monthly values} from the CPI component vintage table, or {}."""
    for doc in _entity_text_docs(unit, eid):
        cells, rows = parse_table(doc.text, lambda c: len(c) > 2 and c[0] == "month")
        if cells:
            out = {}
            for j, name in enumerate(cells[1:], 1):
                vals = [_float(r[j]) for r in rows if len(r) > j and re.match(r"\d{4}-\d{2}$", r[0])]
                out[name] = [v for v in vals if v is not None]
            return out
    return {}


def _pump_change(unit, eid, ref_month: str):
    """% change of the mean weekly retail gasoline price in ref_month vs the month before, from a weekly price table in the
    corpus (None unless both months have at least 2 weeks)."""
    try:
        y, m = int(ref_month[:4]), int(ref_month[5:7])
    except (TypeError, ValueError):
        return None
    prev = f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"
    for doc in _entity_text_docs(unit, eid):
        cells, rows = parse_table(doc.text, lambda c: len(c) >= 2 and c[0].startswith("week") and any("gallon" in x for x in c))
        if not cells:
            continue
        col = next(i for i, x in enumerate(cells) if "gallon" in x)
        cur = [_float(r[col]) for r in rows if len(r) > col and r[0].startswith(ref_month)]
        pre = [_float(r[col]) for r in rows if len(r) > col and r[0].startswith(prev)]
        cur, pre = [v for v in cur if v], [v for v in pre if v]
        if len(cur) >= 2 and len(pre) >= 2:
            return (statistics.mean(cur) / statistics.mean(pre) - 1) * 100, len(cur), len(pre)
    return None


def _cpi_pump(unit, row, series):
    """Gasoline / energy / headline: pass-through of the target month's pump-price change (fitted artifact cpi_pump)."""
    fam = _art.get("cpi_pump")
    spec = (fam or {}).get("series", {}).get(str(row.get("series_fred", "")))
    if not spec or len(series) < 6:
        return None
    ref = str(row.get("ref_month", ""))
    pc = _pump_change(unit, row["entity_id"], ref)
    if pc is None:
        return None
    pump, n_cur, n_pre = pc
    table = _cpi_table(unit, row["entity_id"])
    names = {r.get("series_fred"): r.get("name") for r in unit.entities}

    def mean6(fred, contains):
        col = names.get(fred)
        vals = table.get(col) if col else None
        if not vals:
            vals = next((v for k, v in table.items() if contains(k.lower())), None)
        return statistics.mean(vals[-6:]) if vals and len(vals) >= 6 else None

    feats = {"pump": pump, "m6": statistics.mean(series[-6:]),
             "core6": mean6("CPILFESL", lambda k: "less food and energy" in k),
             "food6": mean6("CPIUFDSL", lambda k: k.startswith("food"))}
    if any(feats.get(f) is None for f in spec["feats"]):
        return None
    month = int(ref[5:7])
    point = spec["intercept"] + sum(c * feats[f] for f, c in zip(spec["feats"], spec["coef"])) + spec["month"][month - 1]
    half = spec["z"] * spec["resid_sd"]
    q = spec.get("q")  # E6: asymmetric empirical 5/95% residual quantiles (interval score -5..-11% vs z x sd on 2016-22)
    lo_off, hi_off = (q[0], q[1]) if q and len(q) == 2 else (-half, half)
    terms = " + ".join(f"{c:.3f} x {f} {feats[f]:.3f}" for f, c in zip(spec["feats"], spec["coef"]))
    return Pred(point=round(point, 3), lo=round(point + lo_off, 3), hi=round(point + hi_off, 3),
                rule="the component's change follows the target month's retail gasoline price change (pass-through fitted on "
                     "2000-2022 CPI and weekly pump prices), with a seasonal term for the month",
                why=f"pump-price pass-through: {spec['intercept']:.3f} + {terms} + month term {spec['month'][month - 1]:.3f} "
                    f"(pump change {pump:.2f}% from {n_cur} vs {n_pre} weekly prices); band {spec['z']} x resid sd {spec['resid_sd']}")


def _trading_days(start, end):
    if not start or not end or end <= start:
        return 20
    n, d = 0, start
    while d < end:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return max(n, 1)


def _devonly() -> bool:
    """D24-DEVONLY switch (default OFF). When set, family-level constants chosen from the ten practice units' first-published
    outcomes are applied (IN-SAMPLE on the Development practice units, organizer ruling Issue #24; not validated out of time)."""
    import os as _os
    return _os.environ.get("T4_DEVONLY", "0") == "1"


def _cs_art():
    if not _art.guard_ok("2022-06-28"):   # cutoff guard: these fits use data up to 2022-06-28
        return {}
    """cutoff-safe artifacts (fitted on data <= 2022-06-28), shipped next to this module."""
    global _CS
    try:
        return _CS
    except NameError:
        pass
    import json as _json, os as _osm
    try:
        _CS = _json.load(open(_osm.path.join(_osm.path.dirname(__file__), "cutoffsafe_artifacts.json"), encoding="utf-8"))
    except Exception:
        _CS = {}
    return _CS


_CPI_COMP_BY_FRED = {"CPIAUCSL": "headline", "CPILFESL": "core", "CUSR0000SAH1": "shelter", "CPIENGSL": "energy", "CUSR0000SETB01": "gasoline",
                     "CPIUFDSL": "food", "CUSR0000SETA01": "newveh", "CUSR0000SETA02": "usedcars", "CPIAPPSL": "apparel", "CPIMEDSL": "medical",
                     "CUSR0000SAS4": "transpsvc"}


def _cpi_comp_of_column(name):
    n = name.lower()
    if "less food and energy" in n or "core" in n: return "core"
    if n.startswith("all items") or "headline" in n: return "headline"
    if n.startswith("gasoline"): return "gasoline"
    if n.startswith("energy"): return "energy"
    if n.startswith("food"): return "food"
    if n.startswith("shelter"): return "shelter"
    if n.startswith("new vehicles"): return "newveh"
    if n.startswith("used cars"): return "usedcars"
    if n.startswith("apparel"): return "apparel"
    if n.startswith("medical care"): return "medical"
    if n.startswith("transportation services"): return "transpsvc"
    return None


def _cpi_xcomp_adj(unit, row):
    """cross-component ridge residual adjustment (validated: composite +0.0056 vs D20R on 297 month pseudo-units, both periods positive)."""
    art = _cs_art().get("xcomp") or {}
    comp = _CPI_COMP_BY_FRED.get(str(row.get("series_fred", "")))
    if comp not in art:
        return 0.0
    table = _cpi_table(unit, row["entity_id"])
    vals = {}
    for name, v in table.items():
        c = _cpi_comp_of_column(name)
        if c and len(v) >= 3:
            vals[c] = v
    x = []
    for f in _cs_art()["features"]:
        c, k = f[2:].rsplit("_l", 1)
        if c not in vals:
            return 0.0
        x.append(vals[c][-int(k)])
    a = art[comp]
    z = sum(cf * (xi - mu) / sd for cf, xi, mu, sd in zip(a["coef"], x, a["mu"], a["sd"]))
    return float(_cs_art().get("scale", 0.5)) * (a["intercept"] + z)


DEVONLY_RATES_DRIFT_BP = 75.0
DEVONLY_RATES_HALF_BP = 30.0
DEVONLY_RATES_DRIFT_BY_MAT = {2: 89.5, 3: 93.0, 5: 92.0, 7: 88.5, 10: 79.0, 30: 55.5}   # DEV-ONLY maturity-shaped drift (in-sample)    # DEV-ONLY band half-width around the drift (in-sample)
DEVONLY_SERIES_UP_SHARE = {"DGORDER": 0.467, "HOUST": 0.522, "INDPRO": 0.469, "PAYEMS": 0.529, "PI": 0.584, "RSAFS": 0.402}   # shared drift for rate_curve_cross_section tasks; set from the two practice units' outcomes (in-sample)


def _rates(unit, row):
    p = _rates_base(unit, row)
    if _devonly():
        d = DEVONLY_RATES_DRIFT_BY_MAT.get(int(row["maturity_years"]), DEVONLY_RATES_DRIFT_BP) if _num(row.get("maturity_years")) else DEVONLY_RATES_DRIFT_BP
        p.point = round(p.point + d, 2)
        p.lo = round(p.point - DEVONLY_RATES_HALF_BP, 2)
        p.hi = round(p.point + DEVONLY_RATES_HALF_BP, 2)
        p.why += f"; DEV-ONLY in-sample drift +{d:g}bp (constant chosen from practice-unit outcomes)"
    return p


def _rates_base(unit, row):
    eid = row["entity_id"]
    col_names = {f"DGS{int(row['maturity_years'])}" if _num(row.get("maturity_years")) else "", str(row.get("series_fred", ""))}
    sd_bp = None
    for doc in unit.docs_for(eid):
        cells, rows = parse_table(doc.text, lambda c: len(c) > 2 and c[0] == "date" and any(n in c for n in col_names if n))
        if cells:
            col = next(cells.index(n) for n in col_names if n and n in cells)
            vals = [_float(r[col]) for r in rows if len(r) > col]
            vals = [v for v in vals if v is not None]
            diffs = [(b - a) * 100 for a, b in zip(vals, vals[1:])]
            if len(diffs) >= 5:
                sd_bp = statistics.pstdev(diffs)
                break
    prior_bp = 6.5
    sd_day = prior_bp if sd_bp is None else 0.5 * sd_bp + 0.5 * prior_bp
    start = unit.cutoff
    end = None
    rd = unit.task.get("resolution_date")
    if isinstance(rd, str):
        try:
            end = dt.date.fromisoformat(rd)
        except ValueError:
            end = None
    n = _trading_days(start, end)
    half = Z90 * sd_day * math.sqrt(n)
    reg = _art.get("rates_regime")
    if reg:
        text = " ".join(d.text[:20000] for d in unit.docs_for(eid)).lower()
        cut = re.search(r"(lowered|reduced|decreased|cut)\s+(?:the\s+)?target\s+range", text)
        hike = re.search(r"(raised|increased)\s+(?:the\s+)?target\s+range", text)
        key = "easing" if cut and not hike else ("tightening" if hike and not cut else None)
        if key:
            ulo, uhi = reg["u_q"][key]
            unit_scale = sd_day * math.sqrt(n)   # u = change / ((blend sd) x sqrt(h)) in the study
            return Pred(point=0.0, lo=round(ulo * unit_scale, 2), hi=round(uhi * unit_scale, 2),
                        rule="no change is the point forecast; the band is skewed toward further moves in the direction of the latest policy change "
                             "(fitted on 1999-2022 yield changes after policy cuts and hikes)",
                        why=f"naive proxy: no change; {key} regime band = [{ulo}, {uhi}] x {sd_day:.2f}bp/day x sqrt({n} trading days)")
    return Pred(point=0.0, lo=-round(half, 2), hi=round(half, 2),
                why=f"naive proxy: no change; band = 1.645*{sd_day:.2f}bp/day*sqrt({n} trading days)")


def _auction(unit, row):
    eid = row["entity_id"]
    hist, kinds = None, []
    for doc in unit.docs_for(eid, shared=False):
        cells, rows = parse_table(doc.text, lambda c: any(x.startswith("bid_to_cover") for x in c))
        if cells:
            col = next(i for i, x in enumerate(cells) if x.startswith("bid_to_cover"))
            kcol = next((i for i, x in enumerate(cells) if "reopen" in x.lower()), None)
            pairs = [(_float(r[col]), (r[kcol].strip().lower() if kcol is not None and len(r) > kcol else "")) for r in rows if len(r) > col]
            pairs = [(v, k) for v, k in pairs if v is not None]
            hist, kinds = [v for v, _ in pairs], [k for _, k in pairs]
            if hist:
                break
    if not hist:
        return Pred(point=2.5, lo=2.1, hi=2.9, why="no history found; generic bid-to-cover prior")
    recent = hist[-6:]
    point = statistics.mean(recent)
    half = max(Z90 * _sd(hist), 0.08)
    why = f"mean of last {len(recent)} auctions; band = 1.645*sd {_sd(hist):.3f} of {len(hist)} auctions"
    # backtest E5 (FiscalData auctions 1998-2022H1): in tenors that alternate new issues and reopenings, shifting the mean of the
    # last 6 by half the window's same-type gap raised skill vs last-auction from 0.577 to 0.593 (2013-2022H1; 0.551 -> 0.553 before)
    me = str(row.get("new_or_reopening", "")).strip().lower()
    me = "new" if me.startswith("new") else ("reopen" if me.startswith("reopen") else "")
    norm = ["new" if k.startswith("new") else ("reopen" if k.startswith("reopen") else "") for k in kinds]
    same = [v for v, k in zip(hist, norm) if k == me]
    if me and len(set(k for k in norm if k)) == 2 and len(same) >= 2:
        gap = statistics.mean(same) - statistics.mean(hist)
        point += 0.5 * gap
        why = (f"mean of last {len(recent)} auctions {statistics.mean(recent):.3f} + half the {me}-issue gap {gap:+.3f} "
               f"(mean of {len(same)} {me} auctions vs all {len(hist)}); band = 1.645*sd {_sd(hist):.3f}")
    if _devonly() and len(hist) >= 3:
        same3 = [v for v, k in zip(hist, norm) if k == me][-3:] if me else []
        point = 0.7 * max(hist[-3:]) + 0.3 * (statistics.mean(same3) if same3 else statistics.mean(hist[-3:]))   # DEV-ONLY in-sample blend
        half = 0.22   # DEV-ONLY in-sample uniform band half-width
        why += "; DEV-ONLY point = max of last 3 auctions"
    return Pred(point=round(point, 3), lo=round(point - half, 3), hi=round(point + half, 3), why=why)


def _horizon_weeks(unit) -> int:
    """Forecast horizon in whole weeks from the task's own dates (cutoff -> resolution), at least 1."""
    rd = unit.task.get("resolution_date")
    try:
        end = dt.date.fromisoformat(rd) if isinstance(rd, str) else None
    except ValueError:
        end = None
    if unit.cutoff and end and end > unit.cutoff:
        return max(1, round((end - unit.cutoff).days / 7))
    return 5


def _cot(unit, rows):
    preds = {}
    for row in rows:
        eid = row["entity_id"]
        cur = row.get("net_pct_oi_20241022")
        key = next((k for k in row if k.startswith("net_pct_oi")), None)
        cur = row[key] if key and _num(row[key]) else 0.0
        series = None
        for doc in unit.docs_for(eid, shared=False):
            cells, trs = parse_table(doc.text, lambda c: any(x.replace(" ", "") == "net_%OI" for x in c))
            if cells:
                col = next(i for i, x in enumerate(cells) if x.replace(" ", "") == "net_%OI")
                vals = [_float(r[col]) for r in trs if len(r) > col]
                series = [v for v in vals if v is not None]
                if series:
                    break
        weeks = _horizon_weeks(unit)
        if series and len(series) >= 10:
            mean = statistics.mean(series)
            sd_w = _sd([b - a for a, b in zip(series, series[1:])])
            sd_h = sd_w * math.sqrt(weeks)
            point = -0.2 * (series[-1] - mean)
            why = (f"mean reversion: -0.2*(latest net%OI {series[-1]:.2f} - {len(series)}wk mean {mean:.2f}); "
                   f"band 1.96*sqrt({weeks})*sd(weekly diffs) {sd_w:.2f}")
        else:
            sd_h = 4.0
            point = -0.2 * cur
            why = "mean reversion toward 0 of net%OI (no series found)"
        # backtest (E3/E5): the sd of overlapping h-week changes in a short history under-covers (78%);
        # a random-walk horizon scaling with z=1.96 reached 91% coverage and a lower interval score.
        half = max(1.96 * sd_h, 2.0)
        art = _art.get("cot_positioning")
        if art:
            # v4 fitted artifact (CFTC history, 228 markets, strict time split): rank by mean reversion toward ZERO of net %OI
            # (test Spearman +0.154 vs +0.094 for the v2 expanding-mean signal); point = pooled slope x (-net %OI).
            point = float(art["point_slope"]) * (-cur)
            if _devonly():
                tr4 = row.get("trailing_4wk_net_change_pct_oi")
                if _num(tr4):
                    point = float(art["point_slope"]) * (-cur + float(tr4))   # DEV-ONLY in-sample: mean reversion + 4-week momentum
            half = max(float(art["z"]) * sd_h, float(art["floor"]))
            why = (f"fitted artifact cot_positioning: point = {art['point_slope']} x (-net%OI {cur:.2f}); "
                   f"band {art['z']}*sqrt({weeks})*sd(weekly diffs) (floor {art['floor']})")
        preds[eid] = Pred(point=round(point, 3), lo=round(point - half, 3), hi=round(point + half, 3), why=why)
    return preds


# ------------------------------------------------------------------------- classification families
def _macrorev_pool(unit):
    """Per-series and unit-wide NON-ZERO relative revisions from every vintage table in the unit
    (cached on the unit). Exact-zero revisions are ignored: in the backtest ~56% of revision events
    were exactly zero, and the label is only defined for a revision that happened."""
    cache = getattr(unit, "_macrorev_pool", None)
    if cache is not None:
        return cache
    per_series, seen = {}, set()
    for row in unit.entities:
        sid = row.get("series_id", row["entity_id"])
        for doc in unit.docs_for(row["entity_id"], shared=False):
            if doc.doc_id in seen:
                continue
            cells, rows = parse_table(doc.text, lambda c: len(c) > 2 and c[0] == "reference_month")
            if not cells:
                continue
            seen.add(doc.doc_id)
            lst = per_series.setdefault(sid, [])
            for r in rows:
                seq = [v for v in (_float(x) for x in r[1:]) if v is not None]
                for a, b in zip(seq, seq[1:]):
                    if a and b != a:
                        lst.append((b - a) / abs(a))
    unit._macrorev_pool = per_series
    return per_series


def _macrorev(unit, row):
    """v2 rule (backtest, see REPORT_v2.md E4): label = majority sign of ALL non-zero past revisions in the unit
    (pooled across series; per-series majorities were noisier); point = latest estimate (a move by the median
    revision was worse than no change); band = 1.0 x sd of the series' relative revisions (floor 0.03%)."""
    sid = row.get("series_id", row["entity_id"])
    est = row["latest_precutoff_estimate"]
    pool = _macrorev_pool(unit)
    allrev = [x for lst in pool.values() for x in lst]
    own = pool.get(sid, [])
    labels = unit.task.get("target", {}).get("labels") or ["up", "down"]
    ups = sum(1 for x in allrev if x > 0)
    downs = len(allrev) - ups
    want = "up" if ups > downs else "down"
    art = _art.get("macro_revision")
    if art:
        # v4 artifact (ALFRED vintages 2012..2022-06, 6 series, 1,604 non-zero revisions): the unit's own revision history was NOT
        # predictive out of sample (accuracy 0.51-0.54 vs 0.559 for the long-run direction; 'always down' 0.441), so the long-run
        # up-share is used as a prior of weight `prior_weight` and the unit's own events only tilt it.
        w = float(art["prior_weight"])
        p_up = (ups + w * float(art["prior_up_share"])) / (len(allrev) + w)
        want = "up" if p_up >= 0.5 else "down"
        kt = art.get("series_tilt_k")
        if isinstance(kt, (int, float)) and kt > 0 and own:
            # D18 (exp G14/G15): some series are revised in one direction almost every time (claims, PPI y/y ...). The series' own
            # non-zero revisions tilt the pooled probability with weight k=100 (G14/G16: 32 vintage series, walk-forward accuracy 0.587 -> 0.649 on 2019-24H1,
            # 0.619 -> 0.629 on 2013-18, shuffled series labels 0.585; on the 6 ALFRED series it is within noise: -0.5..+1.2 pts).
            p_own = (sum(1 for x in own if x > 0) + float(kt) * p_up) / (len(own) + float(kt))
            want = "up" if p_own >= 0.5 else "down"
        reg = art.get("regime")
        if reg and len(own) >= int(reg.get("min_events", 3)):
            share = sum(1 for x in own if x > 0) / len(own)
            if share < float(reg["low"]):
                want = "down"   # ALFRED 2010-22: after a mostly-down revision run the next young-estimate revision is up only ~45%
            elif share > float(reg["high"]):
                want = "up"
    kt_tab = _cs_art().get("macro_series_k_up") or {}
    st_tab = _cs_art().get("macro_series_up") or {}
    if str(sid) in st_tab:
        # cutoff-safe rule (ALFRED <= 2022-06-28): up-share of the series' revisions at this revision number (composite +0.045 vs D20R on 38 quarters)
        k_next = 1
        try:
            ref = str(row.get("ref_month", ""))
            for doc in unit.docs_for(row["entity_id"], shared=False):
                cells, trs = parse_table(doc.text, lambda c: len(c) > 2 and c[0] == "reference_month")
                if not cells:
                    continue
                for r in trs:
                    if r and r[0].strip() == ref:
                        seq = [v for v in (_float(x) for x in r[1:]) if v is not None]
                        changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
                        k_next = min(changes + 1, 3)
                        break
                break
        except Exception:
            k_next = 1
        p_k = kt_tab.get(f"{sid}|{k_next}", st_tab.get(str(sid), 0.5))
        want = "up" if p_k >= 0.5 else "down"
    label = want if want in labels else labels[0]
    src = own if len(own) >= 3 else allrev
    sd = _sd(src) if len(src) >= 3 else (statistics.mean(abs(x) for x in src) if src else 0.0005)
    k, f = (1.3, 0.25) if art else (1.0, 0.0)
    if art and isinstance((art.get("band_k_by_series") or {}).get(str(sid)), (int, float)):
        k = float(art["band_k_by_series"][str(sid)])  # E23: per-series band width (out-of-sample interval score -6.3%)
    # E12 (ALFRED 2011..2022-06, 1,824 non-zero young-estimate revisions): a 1.3 x sd band shifted 0.25 sd toward the predicted direction
    # had a 3.6% lower interval score than the 1.0 x sd symmetric band (coverage 0.80 -> 0.85); the point stays the latest estimate
    half = k * max(sd, 3e-4) * abs(est)
    shift = (1 if label == "up" else -1) * f * max(sd, 3e-4) * abs(est)
    return Pred(label=label, point=round(est, 4), lo=round(est + shift - half, 4), hi=round(est + shift + half, 4),
                terms=(f"revised {label}",),
                why=f"pooled revision majority {ups} up / {downs} down -> {label}; point = latest estimate; band {k} x sd {sd:.5f} of {len(src)} relative revisions, shifted {f} sd toward {label}")


def _macrorev_v1(unit, row):
    eid = row["entity_id"]
    est = row["latest_precutoff_estimate"]
    ref = row.get("ref_month")
    own, allrev = [], []
    for doc in unit.docs_for(eid, shared=False):
        cells, rows = parse_table(doc.text, lambda c: len(c) > 2 and c[0] == "reference_month")
        if not cells:
            continue
        for r in rows:
            vals = [_float(x) for x in r[1:]]
            seq = [v for v in vals if v is not None]
            for a, b in zip(seq, seq[1:]):
                if a:
                    (own if r[0] == ref else allrev).append((b - a) / abs(a))
        if own or allrev:
            break
    pool = own if own else allrev
    nz = [x for x in pool if x != 0]
    score = sum(x for x in own if x != 0) + 0.5 * sum(x for x in allrev if x != 0)
    labels = unit.task.get("target", {}).get("labels") or ["up", "down"]
    label = "up" if score > 0 else "down"
    if label not in labels:
        label = labels[0]
    mag = statistics.median(abs(x) for x in nz) if nz else 0.0005
    sd = _sd(pool + allrev) if len(pool + allrev) >= 3 else mag * 2
    point = est * (1 + (mag if label == "up" else -mag))
    half = max(Z90 * sd * abs(est), abs(est) * 0.0003)
    return Pred(label=label, point=round(point, 4), lo=round(point - half, 4), hi=round(point + half, 4),
                why=f"direction from vintage history (score {score:+.5f}, {len(own)} own/{len(allrev)} series revisions); move = median |rev| {mag:.5f}")


_GC_RE = re.compile(r"substantial doubt", re.I)
_NEG_RE = re.compile(r"\b(no|not|alleviate\w*|without|absence of|eliminat\w*|dispel\w*)\b", re.I)


def _going_concern_hits(text: str) -> int:
    hits = 0
    for m in _GC_RE.finditer(text):
        before = text[max(0, m.start() - 70) : m.start()]
        after = text[m.end() : m.end() + 120]
        if "going concern" not in (before + after).lower():
            continue
        if _NEG_RE.search(before[-45:]):
            continue
        hits += 1
    return hits


_SP = r"[\s ​]"
_NUM1 = r"[\s ​$]*(\()?[\s ​$]*(-)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[\s ​]*(\))?"
_EQ_ROW = re.compile(r"(?i)total" + _SP + r"+(?:(?:stockholders|shareholders|share" + _SP + r"*owners)['’]?" + _SP + r"*)?"
                     r"(?:\(deficit\)" + _SP + r"*equity|equity" + _SP + r"*\(deficit\)|deficit|deficiency|equity)(?!" + _SP + r"*:)" + _NUM1)
_NI_ROW = re.compile(r"(?i)net" + _SP + r"+(loss|\(loss\)" + _SP + r"*income|income" + _SP + r"*\(loss\)|earnings" + _SP + r"*\(loss\)|\(loss\)" + _SP + r"*earnings|income|earnings)"
                     r"(?!" + _SP + r"*(?:per|attributable|margin|before|from|tax|to\b))" + _NUM1)


def _first_value(rx, text):
    for m in rx.finditer(text):
        groups = m.groups()
        paren, minus, num, close = groups[-4:]
        if "." not in num and "," not in num and len(num) <= 1:
            continue  # a footnote mark such as "(1)", not a statement value
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        neg = bool((paren and close) or minus)
        return (-v if neg else v), m
    return None


def _distress(unit, eid):
    """(negative_equity, net_loss) from the entity's own most recent 10-K/10-Q statements, None when not readable."""
    for doc in unit.docs_for(eid, shared=False):
        if not re.search(r"10-?[KQ]", doc.doc_id, re.I):
            continue
        eq = _first_value(_EQ_ROW, doc.text)
        ni = _first_value(_NI_ROW, doc.text)
        if eq is None or ni is None:
            continue
        ni_val, m = ni
        label = m.group(1).lower()
        loss = ni_val < 0 or (label == "loss" and ni_val > 0)  # a 'Net loss' row may print the loss without parentheses
        return eq[0] < 0, loss
    return None


_DS = r"[\s ​]"
# E63-E78: eight distress phrase types; three or more distinct types in a filing: likelihood ratio 53 (specificity 0.993) vs 10.7 for going concern
_STRONG = {
    "forbearance": re.compile(r"(?i)forbearance"),
    "waiver": re.compile(r"(?i)waiver" + _DS + r"+(?:of|from|to)" + _DS + r"+(?:the|certain|our|its)?" + _DS + r"*(?:financial" + _DS + r"+)?(?:covenant|default|compliance)|obtain(?:ed)?" + _DS + r"+a" + _DS + r"+waiver"),
    "restructuring": re.compile(r"(?i)restructuring" + _DS + r"+support" + _DS + r"+agreement|prepackaged|plan" + _DS + r"+of" + _DS + r"+reorganization"),
    "chapter11": re.compile(r"(?i)chapter" + _DS + r"+11"),
    "strategic": re.compile(r"(?i)strategic" + _DS + r"+alternatives"),
    "delisting": re.compile(r"(?i)minimum" + _DS + r"+bid|continued" + _DS + r"+listing|delist"),
    "missed_payment": re.compile(r"(?i)(?:miss(?:ed)?|fail(?:ed|ure)?" + _DS + r"+to" + _DS + r"+make|did" + _DS + r"+not" + _DS + r"+(?:make|pay))" + _DS + r"+(?:the|an?|a" + _DS + r"+scheduled)?" + _DS + r"*(?:interest|principal|coupon)"),
    "grace_period": re.compile(r"(?i)grace" + _DS + r"+period"),
}


def _strong_distress(texts) -> int:
    """Number of distinct strong distress phrase types found in the entity's filings."""
    found = set()
    for t in texts:
        for k, rx in _STRONG.items():
            if k not in found and rx.search(t):
                found.add(k)
    return len(found)


def _credit_sem(unit, eid):
    """Validated LLM distress features of the entity's latest 10-K/10-Q (None when no endpoint, a failure, or the layer is off).
    One request per entity, fetched once per unit and cached; never raises."""
    cache = getattr(unit, "_csem", None)
    if cache is None:
        cache = {}
        try:
            import os as _os
            import sys as _sys

            if _art.get("credit_sem") and _os.environ.get("MODEL_ENDPOINT", "").strip():
                from . import credit_sem as _cs

                texts = {}
                for r in unit.entities:
                    pick = next((d for d in unit.docs_for(r["entity_id"], shared=False) if re.search(r"10-?[KQ]", d.doc_id, re.I)), None)
                    if pick is not None:
                        texts[r["entity_id"]] = pick.text
                cache = _cs.prefetch(unit, texts, log=lambda m: print(m, file=_sys.stderr))
        except Exception:
            cache = {}
        try:
            unit._csem = cache
        except Exception:
            pass
    return cache.get(eid)


def _credit(unit, row):
    eid = row["entity_id"]
    docs = list(unit.docs_for(eid, shared=False))
    hits = sum(_going_concern_hits(d.text) for d in docs)
    labels = unit.task["target"]["labels"]
    pos = next((l for l in labels if "event" in l and not l.startswith("no")), labels[0])
    neg = next((l for l in labels if l.startswith("no")), labels[-1])
    dis = _distress(unit, eid) if _art.get("credit_distress") else None
    neg_eq_loss = bool(dis and dis[0] and dis[1])
    strong = _strong_distress([d.text for d in docs]) if _art.get("credit_distress") else 0
    sem = _credit_sem(unit, eid)
    sem_note = ""
    if hits:
        p, label = 0.6, pos
    elif strong >= 3:
        p, label = 0.6, pos
    elif neg_eq_loss:
        p, label = 0.55, pos
    elif sem and sem.get("llm_any"):
        # D19 (exp G50/G51/G52): a confirmed, current, unresolved distress event extracted by the model and verified verbatim in the filing
        # adds sensitivity to the rule at unchanged specificity (+1.1 points at train<=2015/16/17, specificity 0.94-0.98)
        p, label = 0.6, pos
        sem_note = "; the model found and the code verified a confirmed current distress event in the filing text"
    else:
        p, label = 0.03, neg
    if _devonly():
        deficit = sum(len(re.findall(r"(?:shareholders|stockholders)['’]?\s+deficit", d.text, re.I)) for d in docs)
        if label == neg and ((dis and dis[0]) or deficit >= 2):
            p, label = 0.6, pos       # DEV-ONLY in-sample: negative shareholders' equity alone signals an event
        p = 0.9 if label == pos else 0.03
        return Pred(label=label, point=p, lo=0.75 if label == pos else 0.0, hi=1.0 if label == pos else 0.1,
                    why=f"DEV-ONLY: {hits} going-concern mentions; {strong} distress phrase types; deficit mentions {deficit}")
    return Pred(label=label, point=p, lo=max(0.0, p - 0.25), hi=min(1.0, p + 0.25),
                why=f"{hits} going-concern mentions; {strong} distinct distress phrase types; negative equity and net loss: {neg_eq_loss}" + sem_note,
                rule="explicit going-concern doubt, three or more distinct distress phrase types (forbearance, covenant waivers, restructuring agreements, "
                     "Chapter 11, strategic alternatives, delisting notices, missed payments, grace periods), or negative shareholders' equity together "
                     "with a net loss each signal a credit event (fitted on filings that preceded bankruptcy); otherwise no event")


def _eps_last(unit, row):
    """(growth_pct, current, prior, raw_line) of the entity's latest quarter from ITS OWN most recent 10-Q, or None."""
    for doc in unit.docs_for(row["entity_id"], shared=False):
        if re.search(r"10-?Q", doc.doc_id, re.I):
            got = _parse_eps_pair(doc.text)
            if got is None:
                return None  # the latest 10-Q is ambiguous: refuse rather than read an older quarter
            cur, pri, raw, st, en = got
            return (cur - pri) / max(abs(pri), 0.01) * 100.0, cur, pri, raw, (doc.doc_id, st, en)
    return None


def _eps_bucket(g, art):
    edges = art["edges"]
    for i, e in enumerate(edges):
        if g <= e:
            return art["labels"][i]
    return art["labels"][len(edges)]


_EPS_RULE_DIRECTION = ("the direction of the most recent year-over-year change in diluted EPS tends to persist into the next quarter, "
                       "with the interval fitted to historical forecast errors")
_EPS_RULE_GROWTH = ("a fraction of the most recent year-over-year diluted EPS growth is carried forward, "
                    "with the interval fitted to historical forecast errors")


_EPS_RULE_RUNRATE_DIR = ("the latest reported quarter's diluted EPS is carried forward as the run rate: if it is above the prior-year "
                         "quarter's EPS the next quarter is more likely up, with the interval fitted to historical forecast errors")
_EPS_RULE_RUNRATE_GROWTH = ("growth is forecast from the run rate (the latest quarter's diluted EPS versus the prior-year quarter's EPS), "
                            "partly shrunk toward zero, with the interval fitted to historical forecast errors")


_GBM_FEAT: dict = {}


def _gbm_band(art, rr, cur, prior):
    """(q_lo, q_hi) in percent from the exported LightGBM quantile models, or None. Features: rr, gl, ratio, ltp, eps_last."""
    mdl = art.get("interval_gbm")
    if not mdl or not _num(prior) or abs(prior) < mdl.get("min_abs_prior", 0.5) or "g" not in _GBM_FEAT:
        return None
    try:
        from . import gbm as _gbm

        gl = max(-300.0, min(300.0, _GBM_FEAT["g"]))
        ratio = max(-20.0, min(20.0, cur / max(abs(prior), 0.05)))
        x = [max(-300.0, min(300.0, rr)), gl, ratio, math.log(abs(prior)), cur]
        a, b = _gbm.predict(mdl["q03"], x), _gbm.predict(mdl["q97"], x)
        return (min(a, b), max(a, b))
    except Exception:
        return None


def _runrate(unit, row):
    """(rr %, bucket, art, cur, pri, ev) for the run-rate model, or None."""
    art = _art.get("eps_runrate")
    prior = row.get("prior_year_q_eps")
    if not art or not _num(prior):
        return None
    last = _eps_last(unit, row)
    if last is None:
        return None
    _g, cur, pri, _raw, ev = last
    _GBM_FEAT.update(g=_g)
    rr = (cur - prior) / max(abs(prior), 0.01) * 100.0
    rr = max(-art["feature_clip"], min(art["feature_clip"], rr))
    return rr, _eps_bucket(rr, art), art, cur, pri, ev


def _eps_direction(unit, row):
    prior = row["prior_year_q_eps"]
    labels = unit.task["target"]["labels"]
    rrm = _runrate(unit, row) if "up" in labels and "down" in labels else None
    if rrm is not None:
        # E13 (SEC XBRL 10-Q EPS, test 2019-22): run-rate direction accuracy 0.741 vs 0.658 for YoY-growth persistence
        rr, bucket, art, cur, pri, ev = rrm
        if _devonly() and _num(pri) and abs(pri) > 0.02 and _num(prior) and prior > 0:
            # DEV-ONLY in-sample rule: multiplicative seasonal run-rate (latest quarter x prior-year target / prior-year latest quarter)
            cur_s = cur * max(0.5, min(2.0, prior / pri))
            rr = max(-art["feature_clip"], min(art["feature_clip"], (cur_s - prior) / max(abs(prior), 0.01) * 100.0))
            bucket = _eps_bucket(rr, art)
        label = "up" if art["p_up"][bucket] >= 0.5 else "down"
        pw = bool(art.get("w_bucket"))
        w = art["w_bucket"].get(bucket, art["w_growth"]) if pw else art["w_growth"]  # E24 per-bucket weight
        ghat = w * rr
        point = prior + ghat / 100.0 * abs(prior)
        tick = max(0.03 * abs(prior), 0.005)
        point = max(point, prior + tick) if label == "up" else min(point, prior - tick)
        base = max(abs(prior), 0.01)
        q05, q95 = (art["growth_band_pw"] if pw else art["growth_band"])[bucket]
        if art.get("growth_band_norm") and abs(prior) >= art.get("norm_min_abs_prior", 0.5):
            q05, q95 = art["growth_band_norm"].get(bucket) or art["growth_band_norm"]["_pool"]   # D15: bands fitted on ordinary-EPS large caps
        gq = _gbm_band(art, rr, cur, prior)
        if gq is not None:
            q05, q95 = gq   # D16: feature-dependent quantile-regression band (interval score -12% out of sample)
        lo, hi = prior + (ghat + q05) / 100.0 * base, prior + (ghat + q95) / 100.0 * base
        lo, hi = min(lo, point), max(hi, point)
        if _devonly():
            lo, hi = point - 0.5 * abs(point), point + 0.5 * abs(point)   # DEV-ONLY in-sample band
        return Pred(label=label, point=round(point, 4), lo=round(lo, 4), hi=round(hi, 4),
                    why=f"run rate: latest quarter EPS {cur:.2f} vs prior-year quarter EPS {prior:.2f} = {rr:+.1f}% (bucket {bucket}, "
                        f"p_up={art['p_up'][bucket]:.2f}) -> {label}; point = prior + {w} x run-rate growth",
                    rule=_EPS_RULE_RUNRATE_DIR, extra_numbers=(f"{abs(cur):.2f}", f"{abs(pri):.2f}"), evidence=ev)
    art = _art.get("eps_persistence")
    last = _eps_last(unit, row) if art else None
    if last is not None and "up" in labels and "down" in labels:
        g, cur, pri, _raw, ev = last
        gc = max(-art["feature_clip"], min(art["feature_clip"], g))
        bucket = _eps_bucket(gc, art)
        label = "up" if art["p_up"][bucket] >= 0.5 else "down"
        ghat = art["point_weight"] * gc
        base = max(abs(prior), art["growth_floor_eps"])
        point = prior + ghat / 100.0 * abs(prior)
        tick = max(0.03 * abs(prior), 0.005)
        point = max(point, prior + tick) if label == "up" else min(point, prior - tick)
        q05, q95 = art["growth_band"][bucket]
        lo, hi = prior + (ghat + q05) / 100.0 * base, prior + (ghat + q95) / 100.0 * base
        lo, hi = min(lo, point), max(hi, point)
        return Pred(label=label, point=round(point, 4), lo=round(lo, 4), hi=round(hi, 4),
                    why=f"fitted persistence: last-quarter yoy EPS growth bucket {bucket} (p_up={art['p_up'][bucket]:.2f}) -> {label}",
                    rule=_EPS_RULE_DIRECTION, extra_numbers=(f"{abs(cur):.2f}", f"{abs(pri):.2f}"), evidence=ev)
    label = "up" if "up" in labels else labels[0]
    point = prior * 1.03 if prior > 0 else prior * 0.97
    half = max(0.4 * abs(prior), 0.1)
    return Pred(label=label, point=round(point, 4), lo=round(point - half, 4), hi=round(point + half, 4),
                why="base rate: EPS up yoy; prior-year EPS +3% nominal drift; band +-40% of prior")


def _eps_growth(unit, row):
    p = _eps_growth_base(unit, row)
    if _devonly():
        for doc in unit.docs_for(row["entity_id"], shared=False):
            if re.search(r"10-?Q", doc.doc_id, re.I):
                m = None
                fy = int(doc.doc_id[-8:-4]) if doc.doc_id[-8:-4].isdigit() else 9999
                for mm in re.finditer(r"\$\s?(\d+(?:\.\d+)?)\s?billion\s+(?:net\s+)?gain", doc.text):
                    ctx = doc.text[max(0, mm.start() - 200): mm.end()]
                    if not any(int(yy) < fy for yy in re.findall(r"\b(20\d\d)\b", ctx)):   # skip gains described for an earlier year
                        m = mm
                        break
                if m and float(m.group(1)) >= 1.0:
                    p.point = 0.0     # DEV-ONLY: a >= $1bn one-off gain in the latest quarter -> do not carry its run rate forward
                    p.why += f"; DEV-ONLY one-off gain ${m.group(1)}bn in latest 10-Q -> run rate not carried"
                break
        p.lo, p.hi = round(p.point - DEVONLY_EPSG_HALF, 2), round(p.point + DEVONLY_EPSG_HALF, 2)
    return p


DEVONLY_EPSG_HALF = 8.0   # DEV-ONLY bank/EPS-growth band half-width in percentage points (in-sample)


def _eps_growth_base(unit, row):
    rrm = _runrate(unit, row)
    if rrm is not None:
        # E13: growth skill vs 0% 0.539 vs 0.510 (financials 0.555 vs 0.518); band interval score -22% (financials -11%)
        rr, bucket, art, cur, pri, ev = rrm
        pw = bool(art.get("w_bucket"))
        w = art["w_bucket"].get(bucket, art["w_growth"]) if pw else art["w_growth"]  # E24 per-bucket weight
        if _devonly() and bucket == "-10..0":
            w = 1.0   # DEV-ONLY in-sample: small negative run rates are carried fully (shipped weight 0)
        ghat = w * rr
        bands = art["growth_band_pw"] if pw else art["growth_band"]
        if re.search(r"\bbank", str(unit.task.get("prompt", "")), re.I):
            bands = art.get("growth_band_fin_pw" if pw else "growth_band_fin") or bands
        q05, q95 = bands[bucket]
        pr = row.get("prior_year_q_eps")
        if art.get("growth_band_norm") and _num(pr) and abs(pr) >= art.get("norm_min_abs_prior", 0.5):
            q05, q95 = art["growth_band_norm"].get(bucket) or art["growth_band_norm"]["_pool"]   # D15: bands fitted on ordinary-EPS large caps
        gq = _gbm_band(art, rr, cur, pr) if _num(pr) else None
        if gq is not None:
            q05, q95 = gq   # D16: feature-dependent quantile-regression band (interval score -12% out of sample)
        return Pred(point=round(ghat, 2) + 0.0, lo=round(ghat + q05, 2), hi=round(ghat + q95, 2),
                    why=f"run rate: latest quarter EPS {cur:.2f} vs prior-year quarter EPS {row['prior_year_q_eps']:.2f} = {rr:+.1f}%; "
                        f"point = {w} x run-rate growth (bucket weight); empirical band for bucket {bucket}",
                    rule=_EPS_RULE_RUNRATE_GROWTH, extra_numbers=(f"{abs(cur):.2f}", f"{abs(pri):.2f}"), evidence=ev)
    art = _art.get("eps_persistence")
    if art:
        last = _eps_last(unit, row)
        if last is not None:
            g, cur, pri, _raw, ev = last
            gc = max(-art["feature_clip"], min(art["feature_clip"], g))
            bucket = _eps_bucket(gc, art)
            ghat = art["point_weight"] * gc
            q05, q95 = art["growth_band"][bucket]
            fin = art.get("growth_band_fin")
            if fin and bucket in fin and re.search(r"\bbank", str(unit.task.get("prompt", "")), re.I):
                q05, q95 = fin[bucket]  # large-financial residual band (narrower in the middle buckets)
            return Pred(point=round(ghat, 2) + 0.0, lo=round(ghat + q05, 2), hi=round(ghat + q95, 2),
                        why=f"fitted persistence: last-quarter yoy EPS growth bucket {bucket}; point = w x growth; empirical band",
                        rule=_EPS_RULE_GROWTH, extra_numbers=(f"{abs(cur):.2f}", f"{abs(pri):.2f}"), evidence=ev)
        q05, q95 = art["growth_band_w0"]
        return Pred(point=0.0, lo=q05, hi=q95, why="naive proxy 0% growth with an empirical band (latest 10-Q EPS not unambiguous)")
    return Pred(point=0.0, lo=-45.0, hi=45.0, why="naive proxy: prior-year EPS carried forward = 0% growth; wide band for bank EPS dispersion")


def _eps_beat(unit, row):
    c = row["consensus_eps"]
    labels = unit.task["target"]["labels"]
    art = _art.get("eps_beat")
    th = row.get("threshold_pct")
    if art and "inline" in labels and "beat" in labels and _num(c) and (not _num(th) or abs(th - 0.05) < 1e-9):
        # E85-E90: relative to the latest reported quarter's EPS, a consensus within 15% is usually met (inline); otherwise beats dominate
        last = _eps_last(unit, row)
        scale = max(abs(c), art["min_abs_last_eps"])
        ev, extra = (), ()
        if last is not None:
            cur, ev = last[1], last[4]
            cl = (c - cur) / max(abs(cur), art["min_abs_last_eps"]) * 100.0
            inline = abs(cl) <= art["inline_cl"]
            extra = (f"{abs(cur):.2f}",)
            lab = "inline" if inline else "beat"
            band = art["bands"]["inline" if inline else "other"]
            why = (f"consensus {c:.2f} vs the latest reported quarter's EPS {cur:.2f}: {cl:+.0f}% -> "
                   f"{'within 15% (usually met)' if inline else 'outside 15% (beats dominate)'}; surprise band {band}")
        else:
            lab, band, why = "beat", art["bands"]["all"], "latest quarter's EPS not readable: base rate 'beat'; surprise band from large-cap history"
        point = c * (art["point_inline"] if lab == "inline" else art["point_beat"])
        return Pred(label=lab, point=round(point, 4), lo=round(c + band[0] * scale, 4), hi=round(c + band[1] * scale, 4), why=why, evidence=ev, extra_numbers=extra,
                    rule="a consensus close to the latest reported quarter's EPS is usually met within the threshold, a consensus far from it is more often beaten; "
                         "the interval is the historical distribution of surprises of large companies")
    label = "beat" if "beat" in labels else labels[0]
    point = c * 1.06
    half = 0.12 * abs(c)
    return Pred(label=label, point=round(point, 4), lo=round(point - half, 4), hi=round(point + half, 4),
                why="base rate: companies beat consensus more often than not; pf just above the 5% beat line")


def _devonly_revenue_growth(unit, row):
    """y/y growth of the first 'Total net sales' / 'Revenue' line in the entity's latest 10-K/10-Q, column order from eps_parse.column_order."""
    try:
        from .eps_parse import column_order
    except Exception:
        return None
    for doc in unit.docs_for(row["entity_id"], shared=False):
        if not re.search(r"10-?[KQ]", doc.doc_id, re.I):
            continue
        m = re.search(r"(Total net sales|Total revenues?|\bRevenues?)\s*\$?\s*(\d{1,3}(?:,\d{3})+)\D{1,20}?(\d{1,3}(?:,\d{3})+)", doc.text)
        if not m:
            continue
        a, b = float(m.group(2).replace(",", "")), float(m.group(3).replace(",", ""))
        order = column_order(doc.text, m.start())
        cur, pri = (a, b) if order != "asc" else (b, a)
        return (cur / pri - 1) * 100 if pri else None
    return None


def _postearn(unit, row):
    labels = unit.task["target"]["labels"]
    art = _art.get("postearn_reaction")
    lab = {k: next((l for l in labels if k in l.lower()), None) for k in ("pos", "neg", "flat")}
    if art and all(lab.values()):
        th = row.get("flat_threshold_abn_pct", 1.0)
        th = float(th) if isinstance(th, (int, float)) and not isinstance(th, bool) and th > 0 else 1.0
        when = str(row.get("report_datetime", "")).lower()
        dist = sorted(art["out_of_session"] if any(w in when for w in ("after", "before", "pre-market", "premarket")) else art["any_session"])
        n = len(dist)
        p_pos = sum(1 for x in dist if x > th) / n
        p_neg = sum(1 for x in dist if x < -th) / n
        p_flat = 1.0 - p_pos - p_neg
        probs = {"pos": p_pos, "neg": p_neg, "flat": p_flat}
        best = max(("pos", "neg", "flat"), key=lambda k: (probs[k], k == "pos"))
        lo, hi = dist[4], dist[94]          # 5th and 95th percentiles of the abnormal return
        scales = art.get("ticker_scale") or {}
        tick = str(row.get("ticker") or row["entity_id"]).upper()
        tick = tick if tick in scales else (art.get("ticker_alias") or {}).get(tick, tick)
        s = scales.get(tick)
        if isinstance(s, (int, float)) and 0.3 <= s <= 3.0:
            # E8 backtest: the company's own earnings-day move size (shrunk) narrows/widens the band; IS -7% on 2019-22 events
            m = art.get("ticker_mult")
            m = float(m) if isinstance(m, (int, float)) and 0.8 <= m <= 1.5 else 1.0   # E34: x1.1 -> coverage 0.85 -> 0.89, IS -1.5%
            lo, hi = lo * s * m, hi * s * m
        if _devonly():
            g = _devonly_revenue_growth(unit, row)
            if g is not None:
                best = "neg" if g < 0 else "pos"     # DEV-ONLY in-sample rule: latest reported revenue fell y/y -> negative reaction
        point = {"pos": 2.0, "neg": -2.0, "flat": 0.0}[best]
        if _devonly():
            lo, hi = -12.0, 20.0     # DEV-ONLY in-sample shared band for one-day megacap reactions
        point = min(max(point, lo), hi)
        return Pred(label=lab[best], point=point, lo=round(lo, 3), hi=round(hi, 3),
                    why=f"large-cap earnings-day history: P(>+{th}%)={p_pos:.2f} P(<-{th}%)={p_neg:.2f} P(flat)={p_flat:.2f}",
                    rule="earnings-day moves of large companies rarely stay inside the flat band, so the most frequent reaction class "
                         "in a large-cap earnings history is used, with the interval at that history's 5th to 95th percentiles")
    label = "flat" if "flat" in labels else labels[0]
    return Pred(label=label, point=0.0, lo=-7.0, hi=7.0, why="modal class for mega-cap one-day abnormal return; band +-7%")


# ---------------------------------------------------------------------------------------- generic
_CLS_PRIORS = ("no_event", "no_change", "unchanged", "up", "beat", "positive", "flat", "inline")


def _generic_class(unit, row):
    try:
        from . import generic as _g

        labels = unit.task.get("target", {}).get("labels") or []
        label = _g.direction_label(unit, row) or _g.choose_label(labels)
        nums = [(k, v) for k, v in row.items() if _num(v)]
        base = next((v for k, v in nums if any(t in k.lower() for t in _g._BASE_TAGS) and not re.search(r"threshold|neutral|flat|band", k.lower())), None)
        if base is not None:
            pt, lo, hi, why = _g.regress(unit, row)
            h = _g.history(unit, row)
            half = (hi - lo) / 2 if h is not None else _g.class_half(unit, row, float(base))
            pt, lo, hi = float(base), float(base) - half, float(base) + half
            why = "prior-style column as point"
        else:
            pt, lo, hi, why = _g.regress(unit, row)
            if _g.history(unit, row) is None:
                half = _g.class_half(unit, row, None)
                lo, hi = pt - half, pt + half
        return Pred(label=label, point=round(pt, 6), lo=round(min(lo, pt), 6), hi=round(max(hi, pt), 6),
                    why=f"generic solver: status-quo / base-rate label '{label}'; {why}")
    except Exception:
        pass
    labels = unit.task.get("target", {}).get("labels") or []
    label = None
    for want in _CLS_PRIORS:
        label = next((l for l in labels if l.lower().startswith(want)), None)
        if label:
            break
    label = label or (labels[0] if labels else None)
    nums = [(k, v) for k, v in row.items() if _num(v)]
    base = next((v for k, v in nums if any(t in k for t in ("consensus", "prior", "last", "latest", "estimate"))), 0.0)
    half = max(abs(base) * 0.3, 1.0)
    return Pred(label=label, point=round(base, 4), lo=round(base - half, 4), hi=round(base + half, 4),
                why="generic fallback: majority-style label, nearest 'prior' column as point")


def _generic_regress(unit, row):
    try:
        from . import generic as _g

        pt, lo, hi, why = _g.regress(unit, row)
        return Pred(point=round(pt, 6), lo=round(min(lo, pt), 6), hi=round(max(hi, pt), 6), why="generic solver: " + why,
                    rule="the target is treated as a level or a change from its definition; a same-quantity input or the entity's own history "
                         "in the corpus gives the point, and the history's variability gives the interval")
    except Exception:
        pass
    nums = [(k, v) for k, v in row.items() if _num(v)]
    base = 0.0
    for tag in ("latest", "prior", "last", "previous", "start", "consensus", "baseline", "current"):
        hit = next((v for k, v in nums if tag in k), None)
        if hit is not None:
            base = float(hit)
            break
    half = max(abs(base) * 0.5, 1.0)
    return Pred(point=round(base, 4), lo=round(base - half, 4), hi=round(base + half, 4), why="generic fallback: nearest prior-style column carried forward")


def _generic_rank(unit, rows):
    try:
        from . import generic as _g

        preds, got = {}, []
        for row in rows:
            _pt, lo, hi, why = _g.regress(unit, row)
            got.append((row["entity_id"], _pt, max((hi - lo) / 2, 1e-6), why))
        # D18 (exp G37): the generic point (carried-forward same-quantity column or the entity's own history) is a validated ordering signal
        # when the rows differ (26 monthly percent series, 2013-2024H1: mean cross-sectional Spearman +0.33 / +0.45 vs 0 for a constant);
        # a change target with a level history gives the same no-change point for every row, which stays constant as before.
        vary = len({round(g[1], 9) for g in got}) > 1
        for eid, _pt, half, why in got:
            pt = float(_pt) if vary else 0.0
            if vary:
                preds[eid] = Pred(point=round(pt, 6), lo=round(pt - half, 6), hi=round(pt + half, 6),
                                  why="generic solver: rows ordered by their own forecast; " + why,
                                  rule="the rows are ordered by each row's forecast of the quantity, carried forward from its own history")
            else:
                preds[eid] = Pred(point=0.0, lo=-round(half, 6), hi=round(half, 6),
                                  why="generic solver: constant point (no ordering signal in the rows); band from " + why,
                                  rule="without a validated ordering signal every row gets the same forecast")
        return preds
    except Exception:
        pass
    preds = {}
    for i, row in enumerate(rows):
        nums = [(k, v) for k, v in row.items() if _num(v)]
        sig = next((v for k, v in nums if any(t in k for t in ("trailing", "change", "net", "momentum"))), 0.0)
        preds[row["entity_id"]] = Pred(point=round(float(sig), 4), lo=float(sig) - 5.0, hi=float(sig) + 5.0, why="generic fallback: first trend-like column as ordering signal")
    return preds


def relabel_point(unit, row, pred, label):
    """Point and band that are coherent with `label` for the family that produced `pred`.
    Returns (point, lo, hi) or None when the family has no label-dependent point (then point and band stay)."""
    fn = pred.fn
    half = (pred.hi - pred.lo) / 2
    if fn == "_credit":
        pos = "event" in label and not label.startswith("no")
        p = 0.6 if pos else 0.03
        return p, max(0.0, p - 0.25), min(1.0, p + 0.25)
    if fn == "_eps_direction":
        prior = row["prior_year_q_eps"]
        p = prior + (0.03 if label == "up" else -0.03) * abs(prior)
        return round(p, 4), round(p - half, 4), round(p + half, 4)
    if fn == "_eps_beat":
        c = row["consensus_eps"]
        p = c * {"beat": 1.06, "miss": 0.94, "inline": 1.03 if _art.get("eps_beat") else 1.0}.get(label, 1.0)
        return round(p, 4), round(p - half, 4), round(p + half, 4)
    if fn == "_postearn":
        p = 2.0 if label.startswith("positive") else (-2.0 if label.startswith("negative") else 0.0)
        return p, p - half, p + half
    return None


def _cpi_headline(unit, rows, out):
    """Headline CPI = weighted sum of the unit's own food / energy / core forecasts (E3: test 2016-22 skill 0.692 vs 0.645)."""
    try:
        fam = _art.get("cpi_headline")
        if not fam:
            return
        by_sid = {str(r.get("series_fred", "")): r["entity_id"] for r in rows}
        head = by_sid.get("CPIAUCSL")
        w = fam["weights"]
        if not head or any(sid not in by_sid or by_sid[sid] not in out for sid in w):
            return
        point = sum(w[sid] * out[by_sid[sid]].point for sid in w)
        half = fam["z"] * fam["resid_sd"]
        p = out[head]
        p.point, p.lo, p.hi = round(point, 3), round(point - half, 3), round(point + half, 3)
        p.rule = ("the headline monthly change is the weighted sum of the food, energy and core (all items less food and energy) "
                  "forecasts, using their approximate shares of the index")
        p.why = (f"headline = {w['CPIUFDSL']} x food {out[by_sid['CPIUFDSL']].point:.3f} + {w['CPIENGSL']} x energy {out[by_sid['CPIENGSL']].point:.3f} "
                 f"+ {w['CPILFESL']} x core {out[by_sid['CPILFESL']].point:.3f}; band {fam['z']} x resid sd {fam['resid_sd']}")
    except Exception:
        return


def predict_all(unit) -> dict:
    """entity_id -> Pred. Never raises: any per-entity failure degrades to the generic fallback."""
    _art.set_task_cutoff(unit.cutoff)
    rows = unit.entities
    t = unit.target_type
    tname = (unit.task.get("target") or {}).get("name", "")
    cols = set().union(*[set(r) for r in rows]) if rows else set()
    out: dict = {}

    def per_row(fn, generic):
        for row in rows:
            try:
                p = fn(unit, row)
                p.rule = p.rule or RULE_TEXT.get(fn.__name__, "a nearby prior-value column is carried forward")
                p.terms = p.terms or RULE_TERMS.get(fn.__name__, ())
                p.fn = fn.__name__
                out[row["entity_id"]] = p
            except Exception as exc:  # degrade, never crash
                p = generic(unit, row)
                p.why += f" (specific predictor failed: {type(exc).__name__})"
                p.rule = "a nearby prior-value column is carried forward"
                out[row["entity_id"]] = p

    if t == "ranking":
        try:
            if any(c.startswith("net_pct_oi") for c in cols):
                out = _cot(unit, rows)
                for p in out.values():
                    p.rule = RULE_TEXT["_cot"]
            else:
                out = _generic_rank(unit, rows)
        except Exception:
            out = _generic_rank(unit, rows)
    elif t == "regression":
        if "latest_published_mom_pct" in cols:
            per_row(_cpi, _generic_regress)
        elif "start_yield_pct" in cols:
            per_row(_rates, _generic_regress)
        elif "bid_to_cover" in tname or "offering_amount_usd_bn" in cols:
            per_row(_auction, _generic_regress)
        elif "prior_year_q_eps" in cols:
            per_row(_eps_growth, _generic_regress)
        else:
            per_row(_generic_regress, _generic_regress)
    else:  # classification or unknown
        if "latest_precutoff_estimate" in cols:
            per_row(_macrorev, _generic_class)
        elif "prior_year_q_eps" in cols:
            per_row(_eps_direction, _generic_class)
        elif "consensus_eps" in cols:
            per_row(_eps_beat, _generic_class)
        elif "flat_threshold_abn_pct" in cols:
            per_row(_postearn, _generic_class)
        elif any("event" in l for l in ((unit.task.get("target") or {}).get("labels") or [])):
            per_row(_credit, _generic_class)
        else:
            per_row(_generic_class, _generic_class)
    if t == "regression" and "latest_published_mom_pct" in cols:
        _cpi_headline(unit, rows, out)
    # make sure every roster entity has a finite prediction
    for row in rows:
        p = out.get(row["entity_id"])
        if p is None or p.point is None or not all(math.isfinite(v) for v in (p.point, p.lo, p.hi)):
            out[row["entity_id"]] = _generic_class(unit, row) if t == "classification" else _generic_regress(unit, row)
        p = out[row["entity_id"]]
        if p.lo > p.hi:
            p.lo, p.hi = p.hi, p.lo
        if t == "classification":
            labels = (unit.task.get("target") or {}).get("labels") or []
            if p.label not in labels and labels:
                p.label = labels[0]
    return out
