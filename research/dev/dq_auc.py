"""D21R-Q family 3: auctions. Point = D20R's shipped point (m6 + 0.5 x new/reopen gap), unchanged. Intervals: D20R band vs E1/E2/E3."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "t4-local/dev")
src = open("t4-local/dev/auction_hier.py").read().split("TEN_IDX = ")[0]
exec(src)
from dq_common import run, summarize
df = D.copy()
df["t"] = df.date; df["known_time"] = df.date + pd.Timedelta(days=1)
df["point"] = df.shipped; df["lo0"] = df.shipped - df.shipped_half; df["hi0"] = df.shipped + df.shipped_half
df["tix"] = df.ten.map({t: i for i, t in enumerate(["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]).__iter__().__class__ and {t: i for i, t in enumerate(["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"])}.items()})
df["tix"] = df.ten.map({t: i for i, t in enumerate(["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"])})
df["scale"] = df.sd
FE = ["sd", "m6", "last", "m12", "own", "F", "lm6", "gap", "ind3", "tix", "new", "n"]
L = run(df, FE, scale_col="scale", group="ten", first_test_year=2010, min_train=300)
for per, (a, b) in {"2010-2024": (2010, 2024), "2010-2016": (2010, 2016), "2017-2024": (2017, 2024)}.items():
    print("\n==", per); print(summarize(L[(L.year >= a) & (L.year <= b)]).round(4).to_string(index=False))
L.to_csv("t4-local/exp_out/dq_auc_long.csv", index=False)
