"""D23-1 (revised): 'rebound after a large pre-meeting yield drop' as the shared feature. Historical regression across 168 meetings (2002-2022-05):
 z = mean change of DGS2..30 over the unit window (bp);  x = change in DGS2 (and DGS10) over the 30 trading days up to the cutoff. Slope, t-stat (clustered bootstrap), out-of-sample skill (expanding window)."""
import numpy as np, pandas as pd
src = open("t4-local/dev/fomc_text.py").read().split("TEXT = [")[0]
exec(src)
pos = Y.index.get_indexer(D.cutoff)
D["m30_2"] = (Y.DGS2.values[pos] - Y.DGS2.values[np.maximum(pos - 30, 0)]) * 100
D["m30_10"] = (Y.DGS10.values[pos] - Y.DGS10.values[np.maximum(pos - 30, 0)]) * 100
D["z"] = D[["d_" + m for m in MATS]].mean(axis=1)
d = D.dropna(subset=["m30_2", "z"])
for xcol in ("m30_2", "m30_10"):
    x, y = d[xcol].values, d.z.values
    b = (x * y).sum() / (x * x).sum(); a = y.mean()
    rng = np.random.default_rng(0); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(d), len(d)); bs.append((x[i] * y[i]).sum() / (x[i] ** 2).sum())
    print(f"{xcol}: through-origin slope {b:+.3f} (bootstrap sd {np.std(bs):.3f}, t={b / np.std(bs):+.2f}); corr {np.corrcoef(x, y)[0, 1]:+.3f}; Spearman {pd.Series(x).rank().corr(pd.Series(y).rank()):+.3f}")
# out-of-sample (expanding by year) skill of z_hat = b_hat * m30_2, per maturity-average
A = d[["d_" + m for m in MATS]].values; P = np.full(A.shape, np.nan)
for yr in range(2007, 2023):
    te = d.year == yr; trn = d.res < d.cutoff[te].min() if te.any() else None
    if not te.any() or trn.sum() < 40: continue
    xx, yy = d.loc[trn, "m30_2"].values, d.loc[trn, "z"].values; bh = (xx * yy).sum() / (xx * xx).sum()
    P[te.values] = np.outer(bh * d.loc[te, "m30_2"].values, np.ones(len(MATS)))
ok = ~np.isnan(P).any(axis=1)
for al in (0.5, 1.0, 2.0):
    m0 = np.abs(A[ok]).mean(); m1 = np.abs(A[ok] - al * P[ok]).mean(); print(f"OOS scale x{al}: MAE {m1:.2f} vs zero {m0:.2f}; skill {m0 / (m0 + m1):.4f}; n={ok.sum()}")
b_all = (d.m30_2 * d.z).sum() / (d.m30_2 ** 2).sum()
print("\nfull-sample slope", round(b_all, 4))
for u, m in (("2022-07-28 (2Y -29bp since 2022-06-16)", -29.0), ("2024-09-19 (2Y -118bp since 2024-07-01; 57d window, treat as -118 and -62 30d-equivalent)", -118.0), ("2024 30d-equivalent", -62.0)):
    print(f"  {u}: predicted mean change {b_all * m:+.1f} bp  (actual DGS2..30 mean: 2022 +96, 2024 +70)")
