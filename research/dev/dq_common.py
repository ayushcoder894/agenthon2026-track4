"""D21R-Q common harness: interval estimators around a FIXED point forecast, expanding-window validation.
Input frame columns: t (origin/outcome-known time as Timestamp), y, point, plus feature columns; optional `scale` (vol proxy) and baseline band (lo0, hi0).
Estimators (all trained only on rows whose outcome was known before the test block starts):
  E1 resid_q     : pooled (optionally grouped) 5%/95% quantiles of y - point
  E2 lgbm_q      : LightGBM quantile regression (0.05, 0.95) on features, target y - point; E2c = + conformal (CQR) offset fitted on the latest calibration slice
  E3 volscale    : point + scale x quantiles of standardised residuals (y - point)/scale
Metric: interval score IS = width + 20 (L-y)+ + 20 (y-U)+ (alpha=0.1), coverage, width; paired IS difference vs the baseline with time-block bootstrap (by test year)."""
import numpy as np, pandas as pd
import lightgbm as lgb


def interval_score(y, lo, hi, alpha=0.1):
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def q_resid(tr, group=None, lo=0.05, hi=0.95, min_n=40, fallback=None):
    """returns function(te)->(dlo, dhi) offsets from pooled/grouped residual quantiles."""
    r = tr.y - tr.point
    pool = (float(r.quantile(lo)), float(r.quantile(hi)))
    if group is None:
        return lambda te: (np.full(len(te), pool[0]), np.full(len(te), pool[1]))
    d = {}
    for k, g in tr.groupby(group):
        rr = g.y - g.point
        if len(rr) >= min_n:
            w = len(rr) / (len(rr) + min_n)       # shrink group quantiles toward the pooled ones
            d[k] = (w * float(rr.quantile(lo)) + (1 - w) * pool[0], w * float(rr.quantile(hi)) + (1 - w) * pool[1])
    return lambda te: (np.array([d.get(k, pool)[0] for k in te[group]]), np.array([d.get(k, pool)[1] for k in te[group]]))


def lgbm_q(tr, feats, seed=0, n_est=200, leaves=7, min_child=100):
    X, r = tr[feats].values, (tr.y - tr.point).values
    ms = []
    for a in (0.05, 0.95):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=n_est, learning_rate=0.03, num_leaves=leaves, min_child_samples=min_child,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=20.0, verbose=-1, random_state=seed)
        ms.append(m.fit(X, r))
    def f(te):
        a, b = ms[0].predict(te[feats].values), ms[1].predict(te[feats].values)
        return np.minimum(a, b), np.maximum(a, b)
    return f


def cqr_offset(model_fn, cal, level=0.90):
    """conformalised quantile regression offset: smallest e s.t. coverage of [lo-e, hi+e] on the calibration slice >= level (e may be negative)."""
    lo, hi = model_fn(cal); y = (cal.y - cal.point).values
    s = np.maximum(lo - y, y - hi)
    n = len(s); k = int(np.ceil((n + 1) * level)); k = min(max(k, 1), n)
    return float(np.sort(s)[k - 1])


def run(df, feats, scale_col=None, group=None, first_test_year=2012, last_test_year=2100, cal_years=1, min_train=300, extra=None):
    """expanding-window by calendar year. Returns long frame with columns: year, model, lo, hi (plus y, point). Baseline columns lo0/hi0 -> model 'D20R'."""
    out = []
    years = sorted(df.t.dt.year.unique())
    for yr in years:
        if yr < first_test_year or yr > last_test_year: continue
        te = df[df.t.dt.year == yr]
        known = df[df.known_time < pd.Timestamp(f"{yr}-01-01")]           # outcomes already published before this test block
        if len(te) == 0 or len(known) < min_train: continue
        res = {}
        if "lo0" in te: res["D20R"] = (te.lo0.values, te.hi0.values)
        f1 = q_resid(known, group=None);                         a, b = f1(te); res["E1_pooled"] = (te.point.values + a, te.point.values + b)
        if group:
            f1g = q_resid(known, group=group);                   a, b = f1g(te); res["E1_grouped"] = (te.point.values + a, te.point.values + b)
        # recent-window residual quantiles (last 5 years)
        rec = known[known.known_time >= pd.Timestamp(f"{yr - 5}-01-01")]
        if len(rec) > min_train:
            a, b = q_resid(rec)(te); res["E1_recent5y"] = (te.point.values + a, te.point.values + b)
        if feats:
            tr2 = known[known.known_time < pd.Timestamp(f"{yr - cal_years}-01-01")]; cal = known[known.known_time >= pd.Timestamp(f"{yr - cal_years}-01-01")]
            m = lgbm_q(known, feats); a, b = m(te); res["E2_lgbm"] = (te.point.values + a, te.point.values + b)
            if len(tr2) > min_train and len(cal) > 50:
                m2 = lgbm_q(tr2, feats); e = cqr_offset(m2, cal); a, b = m2(te); res["E2c_lgbm_cqr"] = (te.point.values + a - e, te.point.values + b + e)
        if scale_col:
            kn = known[known[scale_col] > 0]; z = ((kn.y - kn.point) / kn[scale_col])
            qa, qb = float(z.quantile(0.05)), float(z.quantile(0.95)); res["E3_volscale"] = (te.point.values + qa * te[scale_col].values, te.point.values + qb * te[scale_col].values)
            if feats:
                pass
        if extra: res.update(extra(known, te))
        for k, (lo, hi) in res.items():
            o = te[["t", "y", "point"]].copy(); o["year"] = yr; o["model"] = k; o["lo"], o["hi"] = lo, hi
            for c in ([group] if group else []): o[c] = te[c].values
            out.append(o)
    L = pd.concat(out)
    L["IS"] = interval_score(L.y.values, L.lo.values, L.hi.values); L["cov"] = ((L.y >= L.lo) & (L.y <= L.hi)).astype(float); L["w"] = L.hi - L.lo
    return L


def summarize(L, base="D20R", n_boot=600, seed=0):
    rows = []
    b = L[L.model == base].reset_index(drop=True) if base in set(L.model) else None
    for md, g in L.groupby("model", sort=False):
        g = g.reset_index(drop=True)
        r = dict(model=md, n=len(g), IS=g.IS.mean(), cov=g["cov"].mean(), width=g.w.mean())
        if b is not None and md != base and len(b) == len(g):
            d = (g.IS.values - b.IS.values); yrs = g.year.values; u = np.unique(yrs); rng = np.random.default_rng(seed)
            sums = np.array([d[yrs == y].sum() for y in u]); cnt = np.array([(yrs == y).sum() for y in u])
            est = []
            for _ in range(n_boot):
                s = rng.integers(0, len(u), len(u)); est.append(sums[s].sum() / cnt[s].sum())
            r.update(dIS=float(d.mean()), rel=float(g.IS.mean() / b.IS.mean()), se=float(np.std(est)), z=float(d.mean() / np.std(est)) if np.std(est) > 0 else 0.0,
                     years_better=int(sum(1 for y in u if d[yrs == y].mean() < 0)), years=len(u))
        rows.append(r)
    return pd.DataFrame(rows)
