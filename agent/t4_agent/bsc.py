"""Backtest-Select-Combine (BSC): a self-calibrating generic forecaster for task families without a dedicated predictor.

Rationale (chosen on design grounds, not on any outcome): in the M3/M4 forecasting competitions no single method wins across
unfamiliar series; robust practice is to evaluate a small menu of simple methods on the series' own history with rolling origins at
the task's horizon, keep only what beats the naive carry-forward, and combine rather than pick a single winner. Everything here uses
only the unit's frozen corpus (the entity history tables), so it is cutoff-safe by construction.

Two modes, decided from the task and the table:
  direct  : the history column measures the target itself (a level target, or a table of changes for a change target);
            forecast y[t+h] from y[..t].
  delta   : a change target whose table holds levels (positions, yields, ratios); forecast d = y[t+h] - y[t].
Selection is pooled over all entities of the unit (scaled by each entity's own naive error, MASE-style), so a unit with short
series still has dozens of backtest points.
"""
from __future__ import annotations

import math
import statistics

MIN_POINTS = 12          # pooled backtest points needed before anything other than naive is trusted
MIN_GAIN = 0.03          # a method must beat carry-forward by 3% scaled error to be used at all


def _mean(x):
    return sum(x) / len(x)


def _ar1(w):
    mu = _mean(w)
    dev = [x - mu for x in w]
    den = sum(x * x for x in dev[:-1])
    if len(dev) < 8 or den <= 1e-12:
        return mu, 0.0
    phi = sum(a * b for a, b in zip(dev[:-1], dev[1:])) / den
    return mu, min(max(phi, -0.2), 0.98)


def _ses(w, a=0.3):
    s = w[0]
    for x in w[1:]:
        s = a * x + (1 - a) * s
    return s


# ---- direct-mode candidates: f(history up to t, h) -> forecast of y[t+h]
def _d_naive(v, h): return v[-1]
def _d_mean3(v, h): return _mean(v[-3:])
def _d_mean6(v, h): return _mean(v[-6:])
def _d_mean12(v, h): return _mean(v[-12:])
def _d_med6(v, h): return statistics.median(v[-6:])
def _d_half(v, h): return 0.5 * v[-1] + 0.5 * _mean(v[-6:])
def _d_ses(v, h): return _ses(v[-24:])


def _d_ar1(v, h):
    mu, phi = _ar1(v[-36:])
    return mu + (phi ** h) * (v[-1] - mu)


DIRECT = {"naive": _d_naive, "mean3": _d_mean3, "mean6": _d_mean6, "mean12": _d_mean12, "median6": _d_med6,
          "half_last_mean6": _d_half, "ses0.3": _d_ses, "ar1_mean36": _d_ar1}


# ---- delta-mode candidates: f(levels up to t, h) -> forecast of y[t+h] - y[t]; c is a pooled slope fitted in the backtest
def _x_revert(v, h): return -(v[-1] - _mean(v[-26:]))          # distance from the recent mean (reverts if slope > 0)
def _x_mom(v, h): return v[-1] - v[-1 - h] if len(v) > h else 0.0   # last h-period change (momentum if slope > 0)


DELTA = {"revert": _x_revert, "momentum": _x_mom}


def _origins(n, h, lo=8, keep=40):
    return list(range(max(lo, n - h - keep), n - h))


def select_direct(series_list, h):
    """series_list: list of value lists (one per entity). Returns (weights dict, report)."""
    errs = {k: [] for k in DIRECT}
    for v in series_list:
        js = _origins(len(v), h)
        if not js:
            continue
        scale = _mean([abs(v[j + h] - v[j]) for j in js]) or 1e-9
        for k, f in DIRECT.items():
            errs[k] += [abs(v[j + h] - f(v[:j + 1], h)) / scale for j in js]
    n = len(errs["naive"])
    if n < MIN_POINTS:
        return {"naive": 1.0}, f"BSC: {n} backtest points, too few; carry-forward"
    sc = {k: _mean(e) for k, e in errs.items()}
    base = sc["naive"]
    good = sorted((s, k) for k, s in sc.items() if k != "naive" and s < base * (1 - MIN_GAIN))
    if not good:
        return {"naive": 1.0}, f"BSC: no method beat carry-forward on {n} backtest points (naive scaled error {base:.3f}); carry-forward"
    top = [k for _, k in good[:3]]
    w = {k: 1.0 / len(top) for k in top}
    rep = ", ".join(f"{k} {sc[k]:.3f}" for k in top)
    return w, f"BSC: equal-weight combination of the {len(top)} best of {len(DIRECT)} methods on {n} pooled backtest points ({rep}; carry-forward {base:.3f})"


def select_delta(series_list, h):
    """Fits one pooled slope per signal on scaled backtest pairs; keeps signals whose scaled error beats the zero-change forecast."""
    pairs = {k: [] for k in DELTA}
    for v in series_list:
        js = _origins(len(v), h, lo=max(8, h + 1))
        if not js:
            continue
        scale = _mean([abs(v[j + h] - v[j]) for j in js]) or 1e-9
        for k, f in DELTA.items():
            pairs[k] += [(f(v[:j + 1], h) / scale, (v[j + h] - v[j]) / scale) for j in js]
    n = len(pairs["revert"])
    if n < MIN_POINTS:
        return {}, f"BSC: {n} backtest points, too few; no-change forecast"
    base = _mean([abs(y) for _, y in pairs["revert"]])
    keep, rep = {}, []
    for k, pr in pairs.items():
        # time-ordered split inside the backtest: fit the slope on the first 2/3, judge it on the last 1/3
        cut = max(4, (2 * len(pr)) // 3)
        fit, test = pr[:cut], pr[cut:]
        den = sum(x * x for x, _ in fit)
        if den <= 1e-12 or not test:
            continue
        c = max(0.0, sum(x * y for x, y in fit) / den)          # sign-constrained: a signal is used only in its stated direction
        e_sig = _mean([abs(y - c * x) for x, y in test])
        e_0 = _mean([abs(y) for _, y in test])
        if c > 0 and e_sig < e_0 * (1 - MIN_GAIN):
            cf = max(0.0, sum(x * y for x, y in pr) / sum(x * x for x, _ in pr))
            keep[k] = 0.5 * cf / max(1, len(keep) + 1)          # half the full-sample slope (shrinkage), split across kept signals
            rep.append(f"{k} slope {cf:.2f} (held-out error {e_sig:.3f} vs no-change {e_0:.3f})")
    if not keep:
        return {}, f"BSC: neither mean reversion nor momentum beat no-change on {n} pooled backtest points; no-change forecast"
    return keep, "BSC: " + "; ".join(rep) + "; slopes halved for shrinkage"


def forecast_direct(v, h, w):
    return sum(wt * DIRECT[k](v, h) for k, wt in w.items())


def forecast_delta(v, h, w):
    return sum(c * DELTA[k](v, h) for k, c in w.items())


def band_from_backtest(v, h, fc, delta=False):
    """(q_lo, q_hi) signed error quantiles of this entity's own walk-forward errors for the chosen forecaster, or None."""
    js = _origins(len(v), h)
    if len(js) < 8:
        return None
    if delta:
        e = [(v[j + h] - v[j]) - fc(v[:j + 1], h) for j in js]
    else:
        e = [v[j + h] - fc(v[:j + 1], h) for j in js]
    e.sort()
    k = len(e)
    lo, hi = e[max(0, int(math.floor(0.05 * k)))], e[min(k - 1, int(math.ceil(0.95 * k)) - 1)]
    sd = statistics.pstdev(e)
    # small-sample guard: never narrower than 1.645 sd of the errors
    return min(lo, -1.645 * sd), max(hi, 1.645 * sd)
