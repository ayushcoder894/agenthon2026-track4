"""CMS-DD post-earnings evaluation. Outcome: abnormal return of the first earnings release after the filing. Features known at the filing date.
(1) sign: P(abn > 0) with AUC; (2) |abn|: does narrative divergence predict the SIZE of the reaction (Spearman); baseline F (fundamentals composite).
Company-grouped 5-fold CV, shuffled-feature control, company bootstrap."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
src = open("t4-local/dev/cmsdd_eval.py").read().split("auc = lambda")[0]
exec(src)                                      # loads T (credit/eps text), pit helpers, growth(), divergence(), cv_scores(), boot()
auc = lambda y, p: roc_auc_score(y, p)
E0 = pd.read_csv("data_cmsdd_events_text.csv", dtype={"cik": str}); print("event filings with features:", len(E0))
idx_ev = json.load(open("data_sec_events/index.json"))
# growth() reads index.json files of the other sets; patch in the event index entries
def growth2(df):
    df = df.copy(); df["cik"] = df.cik.astype(str).str.zfill(10)
    df["period"] = pd.to_datetime(df.acc.map(lambda a: idx_ev[a]["period"])).astype("datetime64[ns]"); df["filed"] = pd.to_datetime(df.acc.map(lambda a: idx_ev[a]["filed"]))
    df = pd.merge_asof(df.sort_values("period"), pit_use.rename(columns={"period_end": "pe0"}), left_on="period", right_on="pe0", by="cik", tolerance=pd.Timedelta(days=5), direction="nearest")
    df["key"] = (df.period - pd.DateOffset(years=1)).astype("datetime64[ns]")
    df = pd.merge_asof(df.sort_values("key"), pit_use.rename(columns={"period_end": "pe_prior", "revenue": "rev_P", "operating_income": "oi_P", "net_income": "ni_P", "total_equity": "eq_P"}),
                       left_on="key", right_on="pe_prior", by="cik", tolerance=pd.Timedelta(days=20), direction="nearest")
    rel = lambda a, b: ((a - b) / b.abs().clip(lower=1e6)).clip(-2, 2)
    df["rev_g"] = (df.revenue / df.rev_P - 1).clip(-1, 2); df["oi_g"] = rel(df.operating_income, df.oi_P); df["ni_g"] = rel(df.net_income, df.ni_P)
    df["margin"] = df.operating_income / df.revenue; df["dmargin"] = (df.margin - df.oi_P / df.rev_P).clip(-0.5, 0.5)
    return df
E0 = growth2(E0); E0 = divergence(E0, E0)
print("with fundamentals (>=1 component):", int((E0.fund_n >= 1).sum()), "| rows", len(E0))
E0["pos"] = (E0.abn > 0).astype(int); E0["size"] = E0.abn.abs()
NARR = ["D1", "D2", "D3", "adj_z", "hedge_z"]
def sign_test(name, df, basecols):
    df = df.dropna(subset=basecols + NARR).copy(); y = df.pos.values; g = df.cik.values
    mu, sd = df[basecols].mean(), df[basecols].std() + 1e-9; Xb = ((df[basecols] - mu) / sd).values; Xn = ((df[NARR] - df[NARR].mean()) / (df[NARR].std() + 1e-9)).values
    p0 = cv_scores(Xb, y, g); p1 = cv_scores(np.hstack([Xb, Xn]), y, g); ci = boot(auc, y, p0, p1, g, n=1000)
    rng = np.random.default_rng(3); sh = [auc(y, cv_scores(np.hstack([Xb, Xn[rng.permutation(len(Xn))]]), y, g)) - auc(y, p0) for _ in range(20)]
    print(f"{name:40s} n={len(df):4d} pos share {y.mean():.2f} | baseline AUC {auc(y, p0):.4f} -> {auc(y, p1):.4f} delta {auc(y, p1) - auc(y, p0):+.4f} (90% CI {ci[0]:+.4f}, {ci[1]:+.4f}) | shuffled mean {np.mean(sh):+.4f}")
sign_test("sign of the reaction: F baseline", E0, ["F"])
Lg = E0[E0.abn.abs() > 1].assign(pos=lambda d: (d.abn > 1).astype(int)); sign_test("positive vs negative (|abn| > 1%)", Lg, ["F"])
print("--- single features: AUC for the sign (>0.5 = higher feature -> more positive reaction) and Spearman with the size of the reaction")
for f in NARR + ["tone", "topline_share", "F"]:
    d = E0.dropna(subset=[f])
    ci = np.percentile([auc(d.pos.values[s], d[f].values[s]) for s in [np.random.default_rng(k).integers(0, len(d), len(d)) for k in range(400)]], [5, 95])
    print(f"   {f:14s} n={len(d)} AUC {auc(d.pos, d[f]):.3f} (90% CI {ci[0]:.3f}, {ci[1]:.3f}) | Spearman with |abn| {spearmanr(d[f], d['size'])[0]:+.3f}")
