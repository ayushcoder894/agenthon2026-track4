import sys, glob, io, zipfile
import numpy as np, pandas as pd
src = open("t4-local/dev/cot_study.py").read().split("def evaluate")[0]
exec(src)
D = D.sort_values(["mkt", "date"]).copy(); D["rank_oi"] = D.groupby("date").oi.rank(ascending=False)
D = D[(D.rank_oi <= 60) & (D.oi > 0)]
D["split"] = np.where(D.date <= "2014-12-31", "train", np.where(D.date <= "2018-12-31", "val", "final"))
D["block"] = D.date.dt.year * 10 + D.date.dt.quarter
keep = ["mkt", "date", "oi", "L", "S", "net", "npo", "y", "split", "block", "rank_oi"]
D[keep].to_parquet("t4-local/dev/exp500/panels/cot_panel.parquet"); print(D.split.value_counts().to_dict(), D.mkt.nunique(), D.shape)
