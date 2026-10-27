"""CMS-DD step 2: does narrative-fundamentals divergence add information beyond the shipped predictors? Chronology-safe (all features known at the filing date).
Datasets: credit (last statement before a bankruptcy vs controls), EPS direction (next quarter's YoY direction), post-earnings (next earnings-day abnormal return sign).
Models: regularized logistic regression, company-grouped 5-fold CV; compare baseline vs baseline + divergence features vs baseline + SHUFFLED features. 1,000 company bootstraps for CIs."""
import glob, json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, "t4-local/agent")

T = pd.read_csv("data_cmsdd_text.csv", dtype={"cik": str}); T = T[T.found].copy()
cols = ["accession_number", "cik", "period_end", "quarters", "revenue", "operating_income", "net_income", "total_equity", "knowledge_date", "revision"]
pit = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
pit = pit.sort_values(["accession_number", "revision"]).drop_duplicates("accession_number")
pit["period_end"] = pd.to_datetime(pit.period_end).astype("datetime64[ns]")
pit_use = pit[pit.quarters.isin([1, 4])].sort_values("quarters").drop_duplicates(["cik", "period_end"])[["cik", "period_end", "revenue", "operating_income", "net_income", "total_equity"]].sort_values("period_end")
def growth(df):
    idx_all = {}
    for d in ("data_sec", "data_sec_credit"): idx_all.update(json.load(open(f"{d}/index.json")))
    df = df.copy(); df["cik"] = df.cik.astype(str).str.zfill(10)
    df["period"] = pd.to_datetime(df.acc.map(lambda a: idx_all[a]["period"])).astype("datetime64[ns]"); df["filed"] = pd.to_datetime(df.acc.map(lambda a: idx_all[a]["filed"] if "filed" in idx_all[a] else idx_all[a]["knowledge"]))
    df = pd.merge_asof(df.sort_values("period"), pit_use.rename(columns={"period_end": "pe0"}), left_on="period", right_on="pe0", by="cik", tolerance=pd.Timedelta(days=5), direction="nearest")
    df["key"] = (df.period - pd.DateOffset(years=1)).astype("datetime64[ns]")
    df = pd.merge_asof(df.sort_values("key"), pit_use.rename(columns={"period_end": "pe_prior", "revenue": "rev_P", "operating_income": "oi_P", "net_income": "ni_P", "total_equity": "eq_P"}),
                       left_on="key", right_on="pe_prior", by="cik", tolerance=pd.Timedelta(days=20), direction="nearest")
    rel = lambda a, b: ((a - b) / b.abs().clip(lower=1e6)).clip(-2, 2)
    df["rev_g"] = (df.revenue / df.rev_P - 1).clip(-1, 2); df["oi_g"] = rel(df.operating_income, df.oi_P); df["ni_g"] = rel(df.net_income, df.ni_P)
    df["margin"] = df.operating_income / df.revenue; df["dmargin"] = (df.margin - df.oi_P / df.rev_P).clip(-0.5, 0.5)
    df["neg_eq"] = (df.total_equity < 0).astype(int); df["loss"] = (df.net_income < 0).astype(int)
    return df
def divergence(df, ref):
    z = lambda s, r: ((s - r.mean()) / (r.std() + 1e-9)).clip(-3, 3)
    comps = pd.concat([z(df.rev_g, ref.rev_g), z(df.oi_g, ref.oi_g), z(df.ni_g, ref.ni_g)], axis=1)
    df["fund_n"] = comps.notna().sum(axis=1)
    F = comps.mean(axis=1).fillna(0.0); rg, ng = comps.iloc[:, 0].fillna(F), comps.iloc[:, 2].fillna(F)
    tz = z(df.tone, ref.tone)
    df["F"] = F; df["D1"] = tz - F
    emph = df.topline_share * rg + (1 - df.topline_share) * ng
    df["D2"] = emph - F
    df["adj_z"] = z(df.adjusted_rate, ref.adjusted_rate); df["hedge_z"] = z(df.hedge_rate, ref.hedge_rate)
    df["D3"] = df.adj_z * (-F)
    for c in ("rev_g", "dmargin", "ni_g"): df[c] = df[c].fillna(0.0)
    return df
NARR = ["D1", "D2", "D3", "adj_z", "hedge_z"]
def cv_scores(X, y, groups, seed=0):
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        m = LogisticRegression(C=0.3, max_iter=3000).fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]
    return oof
def boot(metric, y, p0, p1, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; out = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        try: out.append(metric(y[s], p1[s]) - metric(y[s], p0[s]))
        except ValueError: pass
    return np.percentile(out, [5, 95])
def run(name, df, basecols, y, groups, metric, hib=True):
    df = df.dropna(subset=basecols + NARR + ["tone"]).copy()
    if len(df) < 60:
        print(f"{name}: too few rows ({len(df)})")
        return df
    yy = df[y].values.astype(int); g = df.cik.values
    mu, sd = df[basecols].mean(), df[basecols].std() + 1e-9; Xb = ((df[basecols] - mu) / sd).values; Xn = ((df[NARR] - df[NARR].mean()) / (df[NARR].std() + 1e-9)).values
    p0 = cv_scores(Xb, yy, g); p1 = cv_scores(np.hstack([Xb, Xn]), yy, g)
    m0, m1 = metric(yy, p0), metric(yy, p1); ci = boot(metric, yy, p0, p1, g)
    rng = np.random.default_rng(5); sh = []
    for k in range(20):
        Xs = Xn[rng.permutation(len(Xn))]; sh.append(metric(yy, cv_scores(np.hstack([Xb, Xs]), yy, g)) - m0)
    print(f"{name:34s} n={len(df):4d} pos={int(yy.sum()):4d} | baseline {m0:.4f} | +divergence {m1:.4f} | delta {m1 - m0:+.4f} (90% CI {ci[0]:+.4f}, {ci[1]:+.4f}) | shuffled control mean delta {np.mean(sh):+.4f} (sd {np.std(sh):.4f})")
    return df


auc = lambda y, p: roc_auc_score(y, p)
nll = lambda y, p: -log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

# ---------------- credit: baseline = the 16 phrase flags + negative equity + loss + fundamentals controls
ph = pd.read_csv("data_sec_credit/phrases.csv"); flag = [c for c in ph.columns if c[:2].isdigit()]
ph = ph.rename(columns={c: "p" + c.split(" ")[0] for c in flag}); flag = [c for c in ph.columns if c.startswith("p") and c[1:].isdigit()]
for c in flag: ph[c] = ph[c].astype(int)
C = T[T.set == "credit"].merge(ph[["acc", "label"] + flag], on="acc", how="inner")
C = growth(C); C = divergence(C, C)
print("=== CREDIT (207 pre-bankruptcy statements vs controls), AUC")
run("credit: flags+equity/loss+fundamentals", C, flag + ["neg_eq", "loss", "rev_g", "dmargin", "ni_g", "F"], "label", "cik", auc)

# ---------------- EPS direction: baseline = run-rate and last-quarter growth signals
exec(open("t4-local/dev/eps_runrate_fit.py").read().split("# direction")[0])
idx_e = json.load(open("data_sec/index.json"))
E = T[T.set == "eps"].copy(); E["period"] = pd.to_datetime(E.acc.map(lambda a: idx_e[a]["period"])).astype("datetime64[ns]"); E["filed"] = pd.to_datetime(E.acc.map(lambda a: idx_e[a]["filed"]))
Pj = P[["cik", "period_end", "rr", "g_last", "up_next", "growth_next", "eps_last", "eps_prior_next"]].copy(); Pj["period_end"] = pd.to_datetime(Pj.period_end).astype("datetime64[ns]"); Pj["cik"] = Pj.cik.astype(str).str.zfill(10)
E["cik"] = E.cik.astype(str).str.zfill(10)
E = pd.merge_asof(E.sort_values("period"), Pj.sort_values("period_end").rename(columns={"period_end": "pe_j"}), left_on="period", right_on="pe_j", by="cik", tolerance=pd.Timedelta(days=5), direction="nearest").dropna(subset=["up_next"])
E = E.drop(columns=["period", "filed"]); E = growth(E); E = divergence(E, E)
E["srr"] = np.sign(E.rr.clip(-300, 300)) * np.log1p(E.rr.clip(-300, 300).abs()); E["sgl"] = np.sign(E.g_last.clip(-300, 300)) * np.log1p(E.g_last.clip(-300, 300).abs()); E["st"] = np.sign(E.eps_prior_next); E["sc"] = np.sign(E.eps_last)
E["up_next"] = E.up_next.astype(int)
print("=== EPS DIRECTION (next quarter up vs down; the shipped rule's accuracy ~0.74), AUC and -log-loss")
run("eps: run-rate + growth signals (AUC)", E, ["srr", "sgl", "st", "sc", "F"], "up_next", "cik", auc)
run("eps: run-rate + growth signals (-logloss)", E, ["srr", "sgl", "st", "sc", "F"], "up_next", "cik", nll)

# ---------------- post-earnings: next earnings-day abnormal return after the filing
ev = pd.read_parquet("t4-local/dev/postearn_events.parquet"); ev["day"] = pd.to_datetime(ev.day)
em = pd.read_parquet("data_earn/earnings.parquet", columns=["cik", "ticker"]).dropna().drop_duplicates("cik"); em["cik"] = em.cik.astype(str).str.zfill(10)
tk = dict(zip(em.cik, em.ticker))
X = E.copy(); X["ticker"] = X.cik.map(tk); X = X.dropna(subset=["ticker"])
rows = []
for r in X.itertuples():
    nxt = ev[(ev.ticker == r.ticker) & (ev.day > r.filed + pd.Timedelta(days=3))].sort_values("day")
    if len(nxt) and (nxt.day.iloc[0] - r.filed).days <= 140: rows.append((r.Index, nxt.abn.iloc[0]))
X = X.loc[[i for i, _ in rows]].assign(abn=[a for _, a in rows]); X["pos"] = (X.abn > 0).astype(int)
print("=== POST-EARNINGS (next earnings-day abnormal return > 0), AUC; baseline = fundamentals composite F")
run("postearn: F only", X, ["F"], "pos", "cik", auc)
Xl = X[X.abn.abs() > 1]; Xl = Xl.assign(pos=(Xl.abn > 1).astype(int))
run("postearn: positive vs negative (|abn|>1%)", Xl, ["F"], "pos", "cik", auc)
