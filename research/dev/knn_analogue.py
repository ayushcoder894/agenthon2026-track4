"""Direction 1: historical-analogue (k-nearest-neighbour) forecasting with a strict as-of split, vs the shipped rules.
EPS direction (features rr, gl, ratio, ltp, sign) and COT (npo, mean26, sd26, d4, sddiff); standardized features; k in {25, 100, 400}."""
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
exec(open("t4-local/dev/ml_eps.py").read().split("# ---- direction")[0])
tr, te = P[P.pe_next <= "2018-12-31"], P[P.pe_next >= "2019-01-01"]
F = ["rr", "gl", "ratio", "ltp", "st", "sc"]
sc = StandardScaler().fit(tr[F]); pu = tr.groupby("b").up_next.mean()
rule = np.mean(te.b.map(lambda b: pu.get(b, 0.55) >= 0.5).values == (te.up == 1).values)
out = [f"rule {rule:.3f}"]
for k in (25, 100, 400):
    m = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(sc.transform(tr[F]), tr.up)
    out.append(f"kNN k={k} {np.mean(m.predict(sc.transform(te[F])) == te.up.values):.3f}")
print("EPS direction (test 2019-22): " + " | ".join(out))
exec(open("t4-local/dev/cot_study.py").read().split("SIG = ")[0])
D = D.dropna(subset=["y", "mean26", "sd26", "d4", "sddiff"]); tr, te = D[D.date.dt.year <= 2015], D[D.date.dt.year >= 2016]
F = ["npo", "mean26", "sd26", "d4", "sddiff"]; sc = StandardScaler().fit(tr[F])
def xs(df, s):
    df = df.assign(s=s); r = []
    for t, x in df.groupby("date"):
        x = x.nlargest(40, "oi")
        if len(x) >= 15: r.append(spearmanr(x.s, x.y)[0])
    return np.nanmean(r)
out = [f"shipped -npo {xs(te, -te.npo):+.3f}"]
for k in (100, 400, 1600):
    m = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(sc.transform(tr[F]), tr.y)
    out.append(f"kNN k={k} {xs(te, m.predict(sc.transform(te[F]))):+.3f}")
print("COT ranking Spearman (test 2016-22): " + " | ".join(out))
