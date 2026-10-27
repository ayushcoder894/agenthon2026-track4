"""Experiment C: FOMC regularised yield-curve model with NEW inputs = statement language (inflation / growth / forward-guidance / risk-balance, and the change vs the previous
statement) + curve shape + policy move + horizon. Pre-registered fixed lexicon (no tuning on outcomes). Strong ridge shrinkage; forecasts shrunk toward 0 by alpha.
Unit emulation: origin = close of the first trading day after the statement date (the unit's cutoff), outcome = yield change to the close of the trading day before the NEXT statement date.
Training only on meetings whose outcome was known before the test meeting. Tests: expanding window from 2008; plus block cycle-out (train two eras, test the third).
Models: direct (per-maturity ridge), onefactor (reduced rank: one common factor z predicted from features, yield_m change = loading_m x z)."""
import json, re, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge


def load(s):
    d = json.load(open(f"data_fred/{s}.json"))["observations"]
    return pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in d})


MATS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
Y = pd.DataFrame({m: load(m) for m in MATS + ["DGS1"]}).dropna(); Y = Y[Y.index >= "2000-01-01"]
tgt = load("DFEDTAR").dropna(); mid = ((load("DFEDTARU") + load("DFEDTARL")) / 2).dropna()
pol = pd.concat([tgt[tgt.index < mid.index.min()], mid]).sort_index().reindex(Y.index, method="ffill")
S = [json.loads(l) for l in open("data_fomc/statements.jsonl", encoding="utf-8")]
S = sorted(S, key=lambda r: r["date"])

# ---------------- pre-registered lexicon ----------------
def near(txt, a, b, w=90):
    n = 0
    for m in re.finditer(a, txt):
        if re.search(b, txt[max(0, m.start() - w): m.end() + w]): n += 1
    return n


def feats(txt):
    t = txt.lower(); L = max(len(t.split()), 1) / 100.0
    infl_hi = near(t, r"inflation", r"elevated|high|persist|upward|pressure|firmer|above|exceed|accelerat|remain")
    infl_lo = near(t, r"inflation", r"low|subdued|below|declin|slow|muted|undesirabl|disinflation|decelerat")
    g_str = near(t, r"activity|growth|employment|labor|job|spending|output", r"strong|solid|expand|robust|improv|firm|gain|stronger|pick")
    g_wk = near(t, r"activity|growth|employment|labor|job|spending|output", r"weak|soft|slow|contract|declin|deteriorat|downturn|slackening|moderat|fell|decreas")
    tight = len(re.findall(r"firming|tighten|removal of|further increases|ongoing increases|raise|raising|normaliz|reduc\w+ its holdings|restrain|some further|additional (?:policy )?firming|increase(?:s)? in the target range", t))
    easy = len(re.findall(r"patient|considerable (?:period|time)|extended period|measured pace|accommodat|lowest|near zero|exceptionally low|ease|easing|support|asset purchases|forward guidance|reduc\w+ the target|lower(?:ed)? the target", t))
    rb = 0
    if re.search(r"risks?[^.]{0,80}(?:roughly )?balanced|balanced[^.]{0,40}risks?", t): rb = 0
    if re.search(r"(?:downside risks?|risks?[^.]{0,60}weighted toward (?:weakness|economic weakness)|weighted toward weakness)", t): rb = -1
    if re.search(r"(?:upside risks?|risks?[^.]{0,60}(?:weighted|tilted) (?:toward|to) (?:higher|inflation))|balance of risks[^.]{0,60}inflation", t): rb = 1
    return dict(infl_hi=infl_hi / L, infl_lo=infl_lo / L, g_str=g_str / L, g_wk=g_wk / L, tight=tight / L, easy=easy / L, rb=rb, nwords=L * 100)


F = pd.DataFrame([dict(date=pd.Timestamp(r["date"]), **feats(r["text"])) for r in S])
F["infl_net"] = F.infl_hi - F.infl_lo; F["growth_net"] = F.g_str - F.g_wk; F["fwd_net"] = F.tight - F.easy
F["tone"] = F.infl_net + F.growth_net + F.fwd_net
for c in ("tone", "infl_net", "growth_net", "fwd_net"): F[c + "_chg"] = F[c].diff().fillna(0)
print("statements", len(F), F.date.min().date(), F.date.max().date())

# ---------------- build meeting examples ----------------
idx = Y.index
rows = []
for i in range(len(F) - 1):
    d0, d1 = F.date[i], F.date[i + 1]
    pos0 = idx.searchsorted(d0, side="right")          # first trading day AFTER the statement date = cutoff
    pos1 = idx.searchsorted(d1, side="left") - 1       # last trading day before the next statement date = resolution
    if pos0 >= len(idx) or pos1 >= len(idx) or pos1 <= pos0 + 3: continue
    c, r = idx[pos0], idx[pos1]
    if (r - c).days > 110 or (c - d0).days > 5: continue
    pm = (pol.iloc[pos0] - pol.iloc[max(idx.searchsorted(d0, side="left") - 1, 0)]) * 100       # policy move at the meeting (bp), known at cutoff
    base = dict(date=d0, cutoff=c, res=r, hdays=pos1 - pos0, year=d0.year, pmove=pm,
                slope=(Y.DGS10.iloc[pos0] - Y.DGS2.iloc[pos0]) * 100, gap=(Y.DGS1.iloc[pos0] - pol.iloc[pos0]) * 100,
                lvl10=Y.DGS10.iloc[pos0], mom10=(Y.DGS10.iloc[pos0] - Y.DGS10.iloc[max(pos0 - 10, 0)]) * 100,
                mom2=(Y.DGS2.iloc[pos0] - Y.DGS2.iloc[max(pos0 - 10, 0)]) * 100)
    for m in MATS: base["d_" + m] = (Y[m].iloc[pos1] - Y[m].iloc[pos0]) * 100
    base.update({k: F.loc[i, k] for k in ("tone", "infl_net", "growth_net", "fwd_net", "rb", "tone_chg", "infl_net_chg", "growth_net_chg", "fwd_net_chg")})
    rows.append(base)
D = pd.DataFrame(rows)
print("meeting examples", len(D), D.date.min().date(), "->", D.date.max().date(), "| mean horizon days", round(D.hdays.mean(), 1),
      "| mean |dy| bp by maturity", D[["d_" + m for m in MATS]].abs().mean().round(1).to_dict())

TEXT = ["tone", "infl_net", "growth_net", "fwd_net", "rb", "tone_chg", "infl_net_chg", "growth_net_chg", "fwd_net_chg"]
CURVE = ["slope", "gap", "lvl10", "mom10", "mom2", "pmove"]
SETS = {"text_only": TEXT, "curve_only": CURVE, "text+curve": TEXT + CURVE}
LOAD = np.array([1.0] * len(MATS))
ALPHA = 60.0


def fit_pred(tr, te, cols, kind):
    Xtr, Xte = tr[cols].fillna(0).values.astype(float), te[cols].fillna(0).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ytr = tr[["d_" + m for m in MATS]].clip(-120, 120).values
    if kind == "direct":
        return np.column_stack([Ridge(alpha=ALPHA).fit(Xtr, Ytr[:, k]).predict(Xte) for k in range(len(MATS))])
    z = Ytr.mean(1); ld = np.array([(Ytr[:, k] @ z) / (z @ z) for k in range(len(MATS))])         # loading of each maturity on the common factor
    zp = Ridge(alpha=ALPHA).fit(Xtr, z).predict(Xte)
    return np.outer(zp, ld)


def skill(a, p):
    m0 = np.abs(a).mean(); return m0 / (m0 + np.abs(a - p).mean())


def cboot(a, p, n=600, seed=0):
    d = (np.abs(a) - np.abs(a - p)).mean(axis=1); rng = np.random.default_rng(seed); est = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.std(est))


A = D[["d_" + m for m in MATS]].values
out = []; PRED = {}
for sname, cols in SETS.items():
    for kind in ("direct", "onefactor"):
        P = np.full(A.shape, np.nan)
        for yr in range(2007, 2023):
            te = D[D.year == yr]
            tr = D[D.res < te.cutoff.min()]
            if te.empty or len(tr) < 40: continue
            P[te.index] = fit_pred(tr, te, cols, kind)
        PRED[(sname, kind)] = P
        ok = ~np.isnan(P).any(axis=1)
        for al in (0.25, 0.5, 1.0):
            for per, sel in (("2007-2013", D.year.between(2007, 2013)), ("2014-2022", D.year.between(2014, 2022)), ("ALL", D.year.between(2007, 2022))):
                m = ok & sel.values
                if m.sum() < 8: continue
                dm, se = cboot(A[m], P[m] * al)
                out.append(dict(feats=sname, model=kind, alpha=al, period=per, n=int(m.sum()), skill=skill(A[m], P[m] * al), d_mae=dm, se=se, z=dm / se))
R = pd.DataFrame(out); pd.set_option("display.width", 200)
print("\nExpanding-window (train on meetings resolved before each test year); skill vs zero change (0.500 = equal):")
print(R.round(4).to_string(index=False))
# era cross-validation (leave-era-out)
eras = {"2002-2007": (2002, 2007), "2008-2015": (2008, 2015), "2016-2022": (2016, 2022)}
print("\nLeave-era-out (train on the other two eras; look-ahead across eras is deliberate, a stress test of regime stability):")
for sname, cols in SETS.items():
    for kind in ("direct", "onefactor"):
        line = []
        for en, (a, b) in eras.items():
            te = D[D.year.between(a, b)]; tr = D[~D.year.between(a, b)]
            if len(te) < 8: continue
            P = fit_pred(tr, te, cols, kind) * 0.5
            Ay = te[["d_" + m for m in MATS]].values; dm, se = cboot(Ay, P)
            line.append(f"{en}: {skill(Ay, P):.3f} (z {dm / se:+.1f})")
        print(f"{sname:11s} {kind:9s} a=0.5 | " + " | ".join(line))
# directly informative: simple correlations between text tone and the subsequent mean yield change
z = D[["d_" + m for m in MATS]].mean(axis=1)
print("\nSpearman(mean subsequent yield change, feature):", {c: round(float(D[c].rank().corr(z.rank())), 3) for c in TEXT + CURVE})
R.to_csv("t4-local/exp_out/fomc_text_results.csv", index=False)
