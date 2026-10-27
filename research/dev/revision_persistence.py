"""Does the direction of a series' recent revisions predict the direction of its NEXT revision?
Data: ZipLime/us-economic-events (Apache-2.0), observations with every vintage. Data used: knowledge_date <= 2022-06-28.
Level revisions: payrolls (sum of the revisions to monthly changes up to the reference month), housing starts (levels),
retail sales (MoM revision as the level-direction proxy)."""
import numpy as np
import pandas as pd

LIMIT = pd.Timestamp("2022-06-28", tz="UTC")
o = pd.read_parquet("data_econ/observations.parquet")
o = o[o.knowledge_date <= LIMIT]


def level_revisions(series):
    """One row per (release time, reference period) with the sign of the revision of that period's LEVEL in that release."""
    d = o[(o.series_id == series) & o.is_revision & o.revision.notna()].copy()
    d["t"] = d.vintage_at
    rows = []
    for t, g in d.groupby("t"):
        g = g.sort_values("reference_period")
        if series == "NONFARM_PAYROLLS_CHANGE":
            cum = g.revision.cumsum()          # the level of month m moves by the sum of the change revisions up to m
            for (_, r), c in zip(g.iterrows(), cum):
                rows.append((t, r.reference_period, float(c)))
        else:
            for _, r in g.iterrows():
                rows.append((t, r.reference_period, float(r.revision)))
    R = pd.DataFrame(rows, columns=["t", "ref", "rev"])
    return R[R.rev != 0].sort_values("t")


def evaluate(series, k_list=(1, 3, 6)):
    R = level_revisions(series)
    rel = R.groupby("t").rev.apply(lambda x: np.sign(x.sum())).rename("rel_sign").reset_index()   # direction of each release's revisions
    out = []
    for i in range(len(rel)):
        t = rel.t.iloc[i]
        past = rel.iloc[:i]
        if len(past) < 6:
            continue
        targets = R[R.t == t]
        for k in k_list:
            s = np.sign(past.rel_sign.iloc[-k:].sum())
            for _, tg in targets.iterrows():
                out.append({"t": t, "k": k, "pred_up": s > 0 if s != 0 else True, "up": tg.rev > 0})
    E = pd.DataFrame(out)
    E["year"] = E.t.dt.year
    print(f"\n=== {series}: revision events {len(R)}, up-share {np.mean(R.rev > 0):.2f}")
    for split, lab in ((E.year <= 2018, "train <=2018"), (E.year >= 2019, "test 2019-2022H1")):
        S = E[split]
        if S.empty:
            continue
        base = S[S.k == 1].up.mean()
        line = f"  {lab:18s} n={len(S[S.k == 1]):4d} always-up {base:.3f} always-down {1 - base:.3f}"
        for k in k_list:
            Sk = S[S.k == k]
            line += f" | persist last {k}: {np.mean(Sk.pred_up == Sk.up):.3f}"
        print(line)
    return E


for s in ("NONFARM_PAYROLLS_CHANGE", "HOUSING_STARTS", "RETAIL_SALES_MOM", "PRIVATE_PAYROLLS_CHANGE", "BUILDING_PERMITS", "NEW_HOME_SALES"):
    try:
        evaluate(s)
    except Exception as exc:
        print(s, "failed:", type(exc).__name__, exc)
