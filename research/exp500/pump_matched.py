"""Matched walk-forward: pump pass-through trained on REVISED vs FIRST-PRINT CPI targets. Identical inputs (monthly mean of EIA weekly regular gasoline
prices, target month vs previous month; EIA retail prices are not revised and all weeks of month t are published by its end, before the CPI release),
identical model (OLS: intercept + pump change + month dummies), identical yearly refits on months before each test year, identical held-out months,
all scored against FIRST-PRINT actuals. Band: point +- 1.645 x training residual sd. Interval score alpha = 0.1."""
import json, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); __file__ = os.path.join(HERE, "cpi_exps.py"); sys.argv = [sys.argv[0]]
exec(open(os.path.join(HERE, "cpi_exps.py"), encoding="utf-8").read().split("fam = Family(")[0])      # -> L (revised SA m/m, current vintage)
FP = pd.read_csv("D:/AGENTHON/data_alfred_cpi/first_print_mom.csv", parse_dates=["month"])
FPW = FP.pivot_table(index="month", columns="series", values="first_mom")
g = pd.DataFrame(json.load(open("D:/AGENTHON/data_alfred_cpi/GASREGW.json"))["observations"]); g = g[g.value != "."]
g["date"] = pd.to_datetime(g.date); g["v"] = g.value.astype(float); g["month"] = g.date.dt.to_period("M").dt.to_timestamp()
pm = g.groupby("month").v.mean(); pump = ((pm / pm.shift(1) - 1) * 100).rename("pump")
MAP = {"gasoline": "CUSR0000SETB01", "energy": "CPIENGSL", "headline": "CPIAUCSL"}
rows = []
for comp, sid in MAP.items():
    rv = L[L.comp == comp].set_index("month").y.rename("rev")
    df = pd.concat([rv, FPW[sid].rename("fp"), pump], axis=1).dropna()
    df = df[(df.index >= "1993-01-01") & (df.index <= "2024-09-01")]; df["comp"] = comp; df["mon"] = df.index.month; rows.append(df)
D = pd.concat(rows); D["year"] = D.index.year
def design(d): return np.c_[np.ones(len(d)), d.pump.values, pd.get_dummies(d.mon).reindex(columns=range(1, 13), fill_value=0).values[:, 1:].astype(float)]
def iscore(y, lo, hi): return (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
out = []
for comp in MAP:
    C = D[D.comp == comp]
    for yr in range(2010, 2025):
        tr, te = C[C.year < yr], C[C.year == yr]
        if len(te) == 0 or len(tr) < 60: continue
        for tgt in ("rev", "fp"):
            Xtr = design(tr); coef = np.linalg.lstsq(Xtr, tr[tgt].values, rcond=None)[0]; sd = float(np.std(tr[tgt].values - Xtr @ coef, ddof=Xtr.shape[1]))
            p = design(te) @ coef; y = te.fp.values
            for i in range(len(te)):
                out.append(dict(comp=comp, year=yr, month=te.index[i], model=tgt, err=abs(y[i] - p[i]), IS=iscore(y[i], p[i] - 1.645 * sd, p[i] + 1.645 * sd)))
O = pd.DataFrame(out)
def summ(s):
    return s.groupby("model").agg(MAE=("err", "mean"), IS=("IS", "mean"))
print("Matched walk-forward vs FIRST-PRINT actuals (identical months):")
for per, m in (("2010-24", O.year >= 2010), ("2010-14", O.year.between(2010, 2014)), ("2015-19", O.year.between(2015, 2019)), ("2020-24", O.year >= 2020)):
    S = O[m]; line = f"  {per}: "
    for comp in ("gasoline", "energy"):
        t = summ(S[S.comp == comp]); line += f"{comp} MAE rev {t.at['rev', 'MAE']:.3f} / fp {t.at['fp', 'MAE']:.3f}, IS rev {t.at['rev', 'IS']:.2f} / fp {t.at['fp', 'IS']:.2f} | "
    t = summ(S[S.comp.isin(['gasoline', 'energy'])]); line += f"combined MAE rev {t.at['rev', 'MAE']:.3f} / fp {t.at['fp', 'MAE']:.3f}"
    print(line)
# year-by-year: does first-print training win consistently?
yy = O[O.comp.isin(["gasoline", "energy"])].pivot_table(index=["year"], columns="model", values="err", aggfunc="mean")
yy["fp_better"] = yy.fp < yy.rev; print("\nyears first-print-trained better (gasoline+energy MAE):", int(yy.fp_better.sum()), "of", len(yy)); print(yy.round(3).to_string())
d = O[O.comp.isin(["gasoline", "energy"])].pivot_table(index=["comp", "month"], columns="model", values="err"); diff = (d.fp - d.rev).values
yrs = np.array([m.year for _, m in d.index]); u = np.unique(yrs); rng = np.random.default_rng(0)
est = [np.concatenate([diff[yrs == x] for x in rng.choice(u, len(u))]).mean() for _ in range(2000)]
print(f"\ncombined MAE difference (fp - rev): {diff.mean():+.4f} (se {np.std(est):.4f}, z {diff.mean() / np.std(est):+.2f}); negative = first-print training better")
print("Corpus check: EIA weekly table in the CPI unit runs through 2024-10-28 (4 October weeks), the same aggregation used here.")
