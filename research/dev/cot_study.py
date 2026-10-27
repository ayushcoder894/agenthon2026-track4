"""E10: ranking futures markets by the 5-week change in noncommercial net position (% of open interest at the origin).
Data: CFTC Commitments of Traders, legacy futures-only annual files 2006..2022 (public domain), report dates <= 2022-06-28.
Each origin week: cross-section of the 40 markets with the largest open interest; mean Spearman of signal vs outcome."""
import glob
import io
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

fr = []
for f in sorted(glob.glob("data_cot/deacot*.zip")):
    z = zipfile.ZipFile(f)
    d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), low_memory=False)
    fr.append(d)
D = pd.concat(fr)
col = {c.lower(): c for c in D.columns}
def C(*keys):
    return next(col[k] for k in col if all(x in k for x in keys))
D = pd.DataFrame({"mkt": D[C("market and exchange")], "date": pd.to_datetime(D[C("as of date in form yyyy-mm-dd")]),
                  "oi": pd.to_numeric(D[C("open interest (all)")], errors="coerce"),
                  "L": pd.to_numeric(D[C("noncommercial positions-long (all)")], errors="coerce"),
                  "S": pd.to_numeric(D[C("noncommercial positions-short (all)")], errors="coerce")}).dropna()
D = D[D.date <= "2022-06-28"].drop_duplicates(["mkt", "date"]).sort_values(["mkt", "date"])
D["net"] = D.L - D.S
D["npo"] = D.net / D.oi * 100
H = 5
g = D.groupby("mkt")
D["y"] = (g.net.shift(-H) - D.net) / D.oi * 100
D["mean26"] = g.npo.transform(lambda x: x.rolling(26).mean())
D["sd26"] = g.npo.transform(lambda x: x.rolling(26).std())
D["d1"] = g.npo.diff(1)
D["d4"] = g.npo.diff(4)
D["sddiff"] = g.npo.transform(lambda x: x.diff().rolling(26).std())
D["f_zero"] = -D.npo
D["f_mean26"] = -(D.npo - D.mean26)
D["f_z26"] = -(D.npo - D.mean26) / D.sd26
D["f_mom4"] = D.d4
D["f_mom1"] = D.d1
D["f_rev4"] = -D.d4
D["f_zero_scaled"] = -D.npo * D.sddiff  # mean reversion scaled by volatility (larger expected moves)
D = D.dropna(subset=["y", "mean26", "sd26", "d4", "sddiff"])
D = D[D.oi > 0]
SIG = ["f_zero", "f_mean26", "f_z26", "f_mom4", "f_mom1", "f_rev4", "f_zero_scaled"]


def evaluate(S, combos):
    res = {}
    for name, w in combos.items():
        rhos = []
        for t, x in S.groupby("date"):
            x = x.nlargest(40, "oi")
            if len(x) < 15:
                continue
            sig = sum(wi * (x[k] - x[k].mean()) / (x[k].std() + 1e-9) for k, wi in w.items())
            rhos.append(spearmanr(sig, x.y)[0])
        res[name] = (np.nanmean(rhos), np.nanstd(rhos) / np.sqrt(len(rhos) / H), len(rhos))
    return res

combos = {k: {k: 1.0} for k in SIG}
combos.update({"zero+mean26": {"f_zero": 1, "f_mean26": 1}, "zero+rev4": {"f_zero": 1, "f_rev4": 0.5},
               "zero+mom4": {"f_zero": 1, "f_mom4": 0.5}, "mean26+rev4": {"f_mean26": 1, "f_rev4": 0.5},
               "zero_scaled+mean26": {"f_zero_scaled": 1, "f_mean26": 1}})
for per, m in (("2006-2015", D.date.dt.year <= 2015), ("2016-2022H1", D.date.dt.year >= 2016)):
    r = evaluate(D[m], combos)
    print(f"=== {per}")
    for k, (mu, se, n) in sorted(r.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:22s} mean Spearman {mu:+.3f} (se~{se:.3f}, weeks {n})")
