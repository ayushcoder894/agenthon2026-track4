"""D21R-Q family 1: bank / large-cap EPS growth. Point = D20R's w_bucket x run-rate (unchanged). Baselines: D20R (frozen LightGBM q03/q97 artifact on |prior|>=0.5; fitted on data <= 2022-06, so IN-SAMPLE on this history = favourable to D20R)
and D20R-style refit (bucket residual quantiles, expanding). Challengers: E1 residual quantiles, E2 LightGBM quantile (+conformal), E3 vol-scaled."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "t4-local/dev"); sys.path.insert(0, "t4-local/agent")
src = open("t4-local/dev/eps_decomp.py").read().split("SETS = {")[0]
exec(src)
from t4_agent import gbm as G
from dq_common import run, summarize, q_resid
art = json.load(open("t4-local/agent/t4_agent/artifacts.json"))["families"]["eps_runrate"]; mdl = art["interval_gbm"]
d = D[(D.rev_last >= 1e9) & (D.abs_prior >= 0.5)].copy()
d["gl"] = d.g_last.clip(-300, 300); d["rrc"] = d.rr.clip(-300, 300)
d["ratio"] = (d.eps_last / d.abs_prior.clip(lower=0.05)).clip(-20, 20); d["labs"] = np.log(d.abs_prior)
d["y"] = d.y.clip(-300, 300)                       # winsorise outcomes at +-300% (heavy tails otherwise dominate the interval score)
d["point"] = d.base
q = []
for x in d[["rrc", "gl", "ratio", "labs", "eps_last"]].values:
    a, b = G.predict(mdl["q03"], list(x)), G.predict(mdl["q97"], list(x)); q.append((min(a, b), max(a, b)))
q = np.array(q); d["lo0"], d["hi0"] = d.point + q[:, 0], d.point + q[:, 1]
d = d.sort_values(["cik", "pe_next"]); d["vol8"] = d.groupby("cik").g_last.transform(lambda s: s.clip(-300, 300).rolling(8, min_periods=4).std())
d["scale"] = d.vol8.fillna(d.vol8.median()).clip(lower=8.0)
d["t"] = d.pe_next; d["known_time"] = d.kd_next.dt.tz_localize(None) if getattr(d.kd_next.dt, "tz", None) is not None else d.kd_next
d["finf"] = d.fin.astype(int)
FE = ["rrc", "gl", "ratio", "labs", "eps_last", "finf", "f_jump", "f_z", "f_qoq", "f_pm_yoy", "f_etr_dev", "f_sh_yoy", "f_rev_yoy"]
print("rows", len(d), "fin", int(d.fin.sum()), d.t.min().date(), "->", d.t.max().date())


def refit_bucket(known, te):
    out = {}
    r = known.y - known.point
    pool = (r.quantile(0.03), r.quantile(0.97)); bq = {}
    for b, g in known.groupby("b"):
        rr_ = g.y - g.point
        bq[b] = (rr_.quantile(0.03), rr_.quantile(0.97)) if len(rr_) >= 80 else pool
    lo = np.array([bq.get(b, pool)[0] for b in te.b]); hi = np.array([bq.get(b, pool)[1] for b in te.b])
    return {"D20R_refit_bucket": (te.point.values + lo, te.point.values + hi)}


L = run(d, FE, scale_col="scale", group="finf", first_test_year=2012, min_train=1500, extra=refit_bucket)
L["fin"] = L.finf.astype(int) if "finf" in L else 0
for nm, sub in (("BANKS/FIN (|prior|>=0.5)", L[L.finf == 1]), ("ALL LARGE CAPS", L)):
    for per, (a, b) in {"2012-2022": (2012, 2022), "2012-2016": (2012, 2016), "2017-2022": (2017, 2022)}.items():
        print(f"\n== {nm} {per}"); print(summarize(sub[(sub.year >= a) & (sub.year <= b)]).round(4).to_string(index=False))
L.to_csv("t4-local/exp_out/dq_eps_long.csv", index=False)
