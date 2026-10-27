"""Batch 3 (E26-E28): direction of the next revision of a young estimate (ALFRED vintages, six series, realtime 2011..2022-06).
Rows as in macrorev_band.py (non-zero revisions only, the label is defined for a revision that happened).
E26 pooled long-run prior 'up' (current default) vs per-series long-run up-share (expanding, past only)
E27 regime rule thresholds / windows (own last-12 share; 0.3/0.7 current) and last-revision sign
E28 same reference month's own previous revision sign (the row's own vintage history)"""
import json
import numpy as np
import pandas as pd

exec(open("t4-local/dev/macrorev_band.py").read().split("def IS(")[0])
R = R[R.rel != 0].copy().sort_values("v")
R["up"] = R.rel > 0
# expanding per-series up-share of past rows (strictly before v)
R["series_share"] = R.groupby("s").up.transform(lambda x: x.shift(1).expanding().mean())
R["pooled_share"] = R.up.shift(1).expanding().mean()
print("rows", len(R), "up-share overall", round(R.up.mean(), 3))
for per, S in (("2011-2016", R[R.v.dt.year <= 2016]), ("2017-2022H1", R[R.v.dt.year >= 2017])):
    up = S.up.values
    res = {"always_up": np.mean(up), "per-series share>=.5": np.mean((S.series_share >= 0.5).values == up)}
    for lo, hi in ((0.3, 0.7), (0.25, 0.75), (0.4, 0.6), (0.2, 0.8)):
        lab = np.where(S.share < lo, False, np.where(S.share > hi, True, True))
        res[f"regime {lo}/{hi}"] = np.mean(lab == up)
        lab2 = np.where(S.share < lo, False, np.where(S.share > hi, True, S.series_share >= 0.5))
        res[f"regime {lo}/{hi}+series prior"] = np.mean(lab2 == up)
    res["own share>=.5"] = np.mean((S.share >= 0.5).values == up)
    print(f"=== {per} n={len(S)}")
    for k, v in res.items():
        print(f"   {k:32s} {v:.3f}")
    print("   per-series up-share:", S.groupby("s").up.mean().round(2).to_dict())
