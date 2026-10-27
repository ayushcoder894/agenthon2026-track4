import sys
import numpy as np, pandas as pd
src = open("t4-local/dev/auction_hier.py").read().split("TEN_IDX = ")[0]
exec(src)
H = 13
hist = {c: [] for c in [f"h{i}" for i in range(1, H + 1)] + [f"i{i}" for i in range(1, H + 1)] + [f"n{i}" for i in range(1, H + 1)]}
Aidx = {t: g.sort_values("date") for t, g in A.groupby("ten")}
for r in D.itertuples():
    mo = pd.Period(r.date, "M"); cutoff = (mo - 1).to_timestamp(how="end").normalize()
    g = Aidx[r.ten]; w = g[(g.date <= cutoff) & (g.date > cutoff - pd.Timedelta(days=380))].tail(H).iloc[::-1]       # most recent first
    b, ind, nw = w.btc.values, w.ind.values, w.new.values.astype(float)
    for i in range(H):
        hist[f"h{i + 1}"].append(b[i] if i < len(b) else np.nan); hist[f"i{i + 1}"].append(ind[i] if i < len(ind) else np.nan); hist[f"n{i + 1}"].append(nw[i] if i < len(nw) else np.nan)
for k, v in hist.items(): D[k] = v
D["split"] = np.where(D.date <= "2012-12-31", "train", np.where(D.date <= "2018-12-31", "val", "final"))
D["block"] = D.date.dt.year * 10 + D.date.dt.quarter
D["tix"] = D.ten.map({t: i for i, t in enumerate(["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"])})
D.drop(columns=["end"]).to_parquet("t4-local/dev/exp500/panels/auc_panel.parquet"); print(D.split.value_counts().to_dict(), D.shape)
