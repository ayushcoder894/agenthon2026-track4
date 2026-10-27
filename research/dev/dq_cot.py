"""D21R-Q family 2: COT. Point = D20R's fitted artifact (0.1086 x -net%OI), unchanged -> the RANKING is untouched. Intervals: D20R band (max(1.96 sd(weekly diffs, 26w) sqrt(5), 2.0)) vs E1/E2/E3.
Universe: the 40 markets with the largest open interest in each origin week (the unit has ten major markets). Target: 5-week change in net %OI (origin OI base)."""
import sys, glob, io, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, "t4-local/dev")
src = open("t4-local/dev/cot_study.py").read().split("def evaluate")[0]
exec(src)
from dq_common import run, summarize
D = D.sort_values(["mkt", "date"]).copy(); g = D.groupby("mkt")
D["sd26w"] = g.npo.transform(lambda x: x.diff().rolling(25).std())
D["d13"] = g.npo.diff(13); D["lev_sd"] = g.npo.transform(lambda x: x.rolling(26).std()); D["logoi"] = np.log(D.oi)
D["rank_oi"] = D.groupby("date").oi.rank(ascending=False)
D = D.dropna(subset=["y", "sd26w", "d4", "d13", "mean26", "sd26"])
D = D[(D.rank_oi <= 40) & (D.oi > 0)]
D = D[D.date.dt.dayofweek.notna()]
D["t"] = D.date; D["known_time"] = D.date + pd.Timedelta(days=38)
D["point"] = 0.1086 * (-D.npo)
half = np.maximum(1.96 * D.sd26w * np.sqrt(5), 2.0)
D["lo0"], D["hi0"] = D.point - half, D.point + half
D["scale"] = D.sd26w * np.sqrt(5)
D["dev26"] = D.npo - D.mean26
D["absnpo"] = D.npo.abs()
FE = ["npo", "d1", "d4", "d13", "sd26w", "sd26", "dev26", "absnpo", "logoi"]
D = D[D.date >= "2008-01-01"].iloc[::2]            # every other week-market row (halves overlap, keeps run time modest)
print("rows", len(D), "markets", D.mkt.nunique(), D.date.min().date(), "->", D.date.max().date())
L = run(D, FE, scale_col="scale", first_test_year=2012, min_train=2000)
for per, (a, b) in {"2012-2022": (2012, 2022), "2012-2016": (2012, 2016), "2017-2022": (2017, 2022)}.items():
    print("\n==", per); print(summarize(L[(L.year >= a) & (L.year <= b)]).round(4).to_string(index=False))
L.to_csv("t4-local/exp_out/dq_cot_long.csv", index=False)
