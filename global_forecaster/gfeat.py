"""Scale-free features for the global forecaster (pure Python; the SAME file ships in the agent, so training and inference agree).
A history is the list of a table's values, oldest first, at a regular period; h is the horizon in periods; ppy the periods per year.
Target used in training: (v[t+h] - v[t]) / s, where s is the series' own typical one-period move (mean |diff| over the last 24)."""
from __future__ import annotations

import math
import statistics

NAMES = ["h", "ppy", "d1", "d2", "d3", "d6", "d12", "m6", "m12", "m36", "seas1", "seas2", "vr", "ac_diff", "ac_lvl", "levelness", "pos36",
         "trend12", "zero_rel", "streak", "abs_mean_rel", "n_hist"]
MIN_LEN = 8


def scale(v):
    w = v[-25:]
    d = [abs(b - a) for a, b in zip(w[:-1], w[1:])]   # consecutive moves within the last 25 values (fixed: was misaligned for n < 25)
    s = sum(d) / len(d) if d else 0.0
    if s <= 1e-12:
        s = max(abs(v[-1]) * 1e-3, 1e-6)
    return s


def _ac(x):
    if len(x) < 6:
        return 0.0
    m = sum(x) / len(x)
    d = [a - m for a in x]
    den = sum(a * a for a in d)
    return sum(a * b for a, b in zip(d[:-1], d[1:])) / den if den > 1e-12 else 0.0


def features(v, h, ppy):
    """-> (feature list, scale) or None when the history is too short."""
    n = len(v)
    if n < MIN_LEN:
        return None
    s = scale(v)
    last = v[-1]
    def dk(k): return (last - v[-1 - k]) / s if n > k else 0.0
    def mk(k): return (last - sum(v[-k:]) / len(v[-k:])) / s
    seas1 = seas2 = 0.0
    p = int(ppy) if ppy in (4, 12, 52) else 0
    if p and n > p + 1 and h < p:
        seas1 = (v[-1 - p + h] - v[-1 - p]) / s
        if n > 2 * p + 1:
            seas2 = 0.5 * (seas1 + (v[-1 - 2 * p + h] - v[-1 - 2 * p]) / s)
    diffs = [b - a for a, b in zip(v[:-1], v[1:])]
    sd24 = statistics.pstdev(diffs[-24:]) or 1e-12
    vr = (statistics.pstdev(diffs[-6:]) / sd24) if len(diffs) >= 6 else 1.0
    w36 = v[-36:]
    mu36, sd36 = sum(w36) / len(w36), statistics.pstdev(w36)
    lo, hi = min(w36), max(w36)
    pos = (last - lo) / (hi - lo) if hi > lo else 0.5
    x12 = list(range(min(12, n)))
    y12 = v[-len(x12):]
    mx, my = sum(x12) / len(x12), sum(y12) / len(y12)
    den = sum((a - mx) ** 2 for a in x12)
    slope = sum((a - mx) * (b - my) for a, b in zip(x12, y12)) / den if den else 0.0
    streak, sg = 0, 0
    for d in reversed(diffs):
        g = int(d > 0) - int(d < 0)
        if g == 0:
            break
        if sg == 0:
            sg = g
        if g != sg:
            break
        streak += g
    f = [float(h), float(ppy), dk(1), dk(2), dk(3), dk(6), dk(12), mk(6), mk(12), mk(36), seas1, seas2, vr, _ac(diffs[-36:]), _ac(w36),
         min(abs(mu36) / (sd36 + 1e-12), 50.0), pos, slope * h / s, max(-50.0, min(50.0, last / (sd36 + 1e-12))), float(streak),
         min(abs(mu36) / s, 200.0), float(min(n, 120))]
    return [max(-200.0, min(200.0, x)) for x in f], s
