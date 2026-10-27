"""EPS decomposition study (Track 4, eps_growth family). Does richer accounting information add skill on top of the SHIPPED run-rate rule?
Same target construction as eps_benchmark.py / eps_runrate_study.py (first-reported 10-Q diluted EPS, data <= 2022-06-28, origin = end(Q)+14d).
Baseline point = shipped per-bucket weight x run-rate rr. Features come only from filings known at the origin.
Residual model: ridge / LightGBM-L1 on (y - baseline), shrunk by alpha. Splits: (A) train outcomes <= 2018 -> test 2019-22; (B) expanding window, test year by year 2016-2022;
(C) company-disjoint: fit on half the companies, test on the other half (time-separated). Skill = mae0/(mae0+MAE), mae0 = MAE of the 0% forecast.
Significance: company-clustered bootstrap of the per-company mean absolute-error improvement. Control: same model with feature rows shuffled across companies within quarter."""
import glob, json, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

src = open("t4-local/dev/eps_runrate_study.py").read().split("for name, S0 in")[0]
exec(src)   # -> P with cik, period_end (last quarter), pe_next, eps_last, eps_prior_next, rr, gl, y, fin, big, up_next, origin ...

ART = json.load(open("t4-local/agent/t4_agent/artifacts.json"))["families"]["eps_runrate"]
EDGES = ART["edges"]; LABS = ART["labels"]
P["b"] = pd.cut(P.rr, bins=[-np.inf] + EDGES + [np.inf], labels=LABS).astype(str)
P["base"] = P.b.map(ART["w_bucket"]).astype(float) * P.rr
P["abs_prior"] = P.eps_prior_next.abs()

# ---------- extended quarterly feature table (first reported) ----------
FL = ["cik", "period_end", "quarters", "period_kind", "form", "knowledge_date", "revision", "eps_diluted", "revenue", "pretax_income",
      "income_tax_expense", "net_income", "shares_diluted_weighted", "operating_income", "interest_expense", "total_assets", "total_equity"]
pit = pd.concat([pd.read_parquet(f, columns=FL) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))], ignore_index=True)
pit = pit[(pit.knowledge_date <= LIMIT) & pit.form.astype(str).str.startswith("10-Q")]
pit["period_end"] = pd.to_datetime(pit.period_end).astype("datetime64[ns]")
flow = pit[(pit.period_kind == "quarter") & (pit.quarters == 1)].sort_values(["cik", "period_end", "knowledge_date", "revision"])
flow = flow.groupby(["cik", "period_end"], as_index=False).first()[["cik", "period_end", "eps_diluted", "revenue", "pretax_income", "income_tax_expense",
                                                                    "net_income", "shares_diluted_weighted", "operating_income", "interest_expense"]]
bal = pit[pit.total_assets.notna() | pit.total_equity.notna()].sort_values(["cik", "period_end", "knowledge_date", "revision"])
bal = bal.groupby(["cik", "period_end"], as_index=False).first()[["cik", "period_end", "total_assets", "total_equity"]]
E = flow.merge(bal, on=["cik", "period_end"], how="left").sort_values(["cik", "period_end"])
E["period_end"] = E.period_end.astype("datetime64[ns]")


def lag(k, cols):
    """value of `cols` for the quarter k*91 days before each row (same company), tolerance 12 days."""
    a = E[["cik", "period_end"]].copy(); a["key"] = (a.period_end - pd.Timedelta(days=91 * k)).astype("datetime64[ns]")
    b = E[["cik", "period_end"] + cols].rename(columns={"period_end": "pe_l", **{c: f"{c}_l{k}" for c in cols}}).sort_values("pe_l")
    m_ = pd.merge_asof(a.sort_values("key"), b, left_on="key", right_on="pe_l", by="cik", tolerance=pd.Timedelta(days=12), direction="nearest")
    m_ = m_.drop(columns=["key", "pe_l"]).drop_duplicates(["cik", "period_end"])
    return m_


F = E.copy()
for k in (1, 2, 3, 4):
    F = F.merge(lag(k, ["eps_diluted"] + (["revenue", "pretax_income", "income_tax_expense", "shares_diluted_weighted", "net_income", "operating_income"] if k == 4 else [])),
                on=["cik", "period_end"], how="left")
eps_cols = ["eps_diluted"] + [f"eps_diluted_l{k}" for k in (1, 2, 3, 4)]
rev = F.revenue.abs().clip(lower=1.0)
F["f_qoq"] = ((F.eps_diluted - F.eps_diluted_l1) / F.eps_diluted_l1.abs().clip(lower=0.05)).clip(-5, 5)
hist = F[[f"eps_diluted_l{k}" for k in (1, 2, 3, 4)]]
mu, sd = hist.mean(axis=1), hist.std(axis=1)
F["f_jump"] = ((F.eps_diluted - mu) / mu.abs().clip(lower=0.05)).clip(-5, 5)
F["f_z"] = ((F.eps_diluted - mu) / (sd + 0.1 * mu.abs() + 0.02)).clip(-6, 6)
F["f_trend"] = ((F.eps_diluted_l1 - F.eps_diluted_l4) / F.eps_diluted_l4.abs().clip(lower=0.05)).clip(-5, 5)
F["f_pm"] = (F.pretax_income / rev).clip(-2, 2)
F["f_pm_yoy"] = F.f_pm - (F.pretax_income_l4 / F.revenue_l4.abs().clip(lower=1.0)).clip(-2, 2)
etr = (F.income_tax_expense / F.pretax_income.where(F.pretax_income.abs() > 1e-6)).clip(-0.5, 0.8)
F["f_etr"] = etr
etr4 = (F.income_tax_expense_l4 / F.pretax_income_l4.where(F.pretax_income_l4.abs() > 1e-6)).clip(-0.5, 0.8)
F["f_etr_dev"] = etr - etr4
F["f_sh_yoy"] = (F.shares_diluted_weighted / F.shares_diluted_weighted_l4 - 1).clip(-0.5, 0.5)
F["f_rev_yoy"] = (F.revenue / F.revenue_l4.abs().clip(lower=1.0) - 1).clip(-1, 3)
F["f_nonop"] = ((F.pretax_income - F.operating_income) / rev).clip(-1, 1)
F["f_ni_rev"] = (F.net_income / rev).clip(-2, 2)
F["f_int_rev"] = (F.interest_expense / rev).clip(0, 3)
F["f_roa"] = (F.net_income * 4 / F.total_assets.abs().clip(lower=1.0)).clip(-0.5, 0.5)
F["f_eq_assets"] = (F.total_equity / F.total_assets.abs().clip(lower=1.0)).clip(-1, 1)
F["f_ni_yoy"] = ((F.net_income - F.net_income_l4) / F.net_income_l4.abs().clip(lower=1.0)).clip(-5, 5)
SEQ = ["f_qoq", "f_jump", "f_z", "f_trend"]
BRIDGE = ["f_pm", "f_pm_yoy", "f_etr", "f_etr_dev", "f_sh_yoy", "f_rev_yoy", "f_nonop", "f_ni_rev", "f_int_rev", "f_ni_yoy"]
FINF = ["f_roa", "f_eq_assets"]
D = P.merge(F[["cik", "period_end"] + SEQ + BRIDGE + FINF], on=["cik", "period_end"], how="left")
D = D[D.pe_next.notna()].copy()
D["yy"] = D.pe_next.dt.year
D["resid"] = D.y - D.base
print("examples", len(D), "companies", D.cik.nunique(), "| with sequence feats", int(D.f_jump.notna().sum()), "| fin", int(D.fin.sum()))

SETS = {"seq": SEQ, "bridge": BRIDGE, "all": SEQ + BRIDGE + FINF}
SUBS = {"ALL": lambda d: d, "LARGE": lambda d: d[(d.rev_last >= 1e9) & (d.abs_prior >= 0.5)], "FIN": lambda d: d[d.fin]}


def skill(y, pred):
    m0 = np.abs(y).mean(); return m0 / (m0 + np.abs(y - pred).mean())


def fit_predict(tr, te, cols, kind, seed=0):
    Xtr, Xte = tr[cols].fillna(0.0).values, te[cols].fillna(0.0).values
    ytr = tr.resid.clip(-300, 300).values
    if kind == "ridge":
        mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
        r = Ridge(alpha=2000.0).fit((Xtr - mu_) / sd_, ytr)
        return r.predict((Xte - mu_) / sd_)
    m = lgb.LGBMRegressor(objective="l1", n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=50.0, verbose=-1, random_state=seed)
    m.fit(Xtr, ytr); return m.predict(Xte)


def cluster_boot(te, dmae, n=500, seed=1):
    g = pd.DataFrame({"c": te.cik.values, "d": dmae}).groupby("c").d.agg(["sum", "count"])
    rng = np.random.default_rng(seed); idx = np.arange(len(g)); est = []
    for _ in range(n):
        s = rng.choice(idx, len(idx)); est.append(g["sum"].values[s].sum() / g["count"].values[s].sum())
    return float(np.mean(dmae)), float(np.std(est))


def evaluate(tr, te, label, rows):
    for sname, cols in SETS.items():
        for kind in ("ridge", "gbm"):
            raw = fit_predict(tr, te, cols, kind)
            for alpha in (0.5, 1.0):
                pred = te.base.values + alpha * raw
                for sub, f in SUBS.items():
                    msk = f(te).index
                    t = te.loc[msk]; p = pd.Series(pred, index=te.index).loc[msk].values
                    y = t.y.clip(-1000, 1000).values
                    dmae = np.abs(y - t.base.values) - np.abs(y - p)   # >0 = model better than shipped
                    mean_d, se = cluster_boot(t, dmae)
                    rows.append(dict(split=label, feats=sname, model=kind, alpha=alpha, sub=sub, n=len(t), skill_base=skill(y, t.base.values), skill_model=skill(y, p),
                                     d_mae=mean_d, se=se, z=mean_d / se if se > 0 else 0.0))


rows = []
# (A) train <= 2018, test 2019-22
evaluate(D[D.pe_next <= "2018-12-31"], D[D.pe_next >= "2019-01-01"], "A_2019-22", rows)
# (B) expanding window test years 2016..2022 pooled
for yr in range(2016, 2023):
    evaluate(D[D.yy < yr], D[D.yy == yr], f"B_{yr}", rows)
# (C) company-disjoint: random half of companies for fit, time-separated test (2019-22 on the other half)
rng = np.random.default_rng(3); ciks = D.cik.unique(); half = set(rng.choice(ciks, len(ciks) // 2, replace=False))
evaluate(D[D.cik.isin(half) & (D.pe_next <= "2018-12-31")], D[~D.cik.isin(half) & (D.pe_next >= "2019-01-01")], "C_unseen_cos", rows)
R = pd.DataFrame(rows); R.to_csv("t4-local/exp_out/eps_decomp_results.csv", index=False)

pd.set_option("display.width", 200)
print(R[R.split.isin(["A_2019-22", "C_unseen_cos"])].round(4).sort_values(["split", "sub", "feats", "model", "alpha"]).to_string(index=False))
Bp = R[R.split.str.startswith("B_")].groupby(["feats", "model", "alpha", "sub"]).apply(
    lambda g: pd.Series(dict(n=g.n.sum(), d_mae=np.average(g.d_mae, weights=g.n), years_better=int((g.d_mae > 0).sum()), of=len(g)))).reset_index()
print("\nExpanding-window 2016-22 (pooled over years):"); print(Bp.round(4).sort_values(["sub", "d_mae"], ascending=[True, False]).to_string(index=False))
