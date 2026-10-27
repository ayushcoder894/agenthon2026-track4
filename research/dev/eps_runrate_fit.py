"""Fit + validate the run-rate EPS artifact. rr = (EPS_last - EPS_prior_target) / |EPS_prior_target| x 100, clipped +-300.
Direction: P(up | rr bucket). Growth unit: point = w_g x rr, band = 5/95% residual quantiles by rr bucket (all / financials).
Level unit: point = prior + w_l x (EPS_last - prior), band = residual quantiles of (EPS_next - point) / |prior| by rr bucket.
Train <= 2018, test 2019-2022H1; final artifact refit on everything <= 2022-06-28."""
import json
import numpy as np
import pandas as pd

exec(open("t4-local/dev/eps_runrate_study.py").read().split("for name, S0 in")[0])
EDGES = [-50, -10, 0, 10, 30]
LABS = ["<=-50", "-50..-10", "-10..0", "0..10", "10..30", ">30"]
P["b"] = pd.cut(P.rr, bins=[-np.inf] + EDGES + [np.inf], labels=LABS).astype(str)
P["lev_y"] = (P.eps_next - P.eps_prior_next) / P.eps_prior_next.abs().clip(lower=0.01)  # relative level change (= growth/100)
P["lev_rr"] = (P.eps_last - P.eps_prior_next) / P.eps_prior_next.abs().clip(lower=0.01)
tr, te = P[P.pe_next <= "2018-12-31"], P[P.pe_next >= "2019-01-01"]


def IS(y, lo, hi):
    return np.mean((hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0))

# direction
pu = tr.groupby("b").up_next.mean()
print("train P(up | rr bucket):", pu.round(3).to_dict())
lab_te = te.b.map(lambda b: pu.get(b, 0.55) >= 0.5)
print(f"test accuracy bucket-argmax {np.mean(lab_te.values == (te.up_next == 1).values):.3f} | rr>0 {np.mean((te.rr > 0).values == (te.up_next == 1).values):.3f}")
# growth point weight
for w in (0.5, 0.75, 0.9, 1.0):
    for S, nm in ((te, "ALL"), (te[te.fin], "FIN")):
        mae0 = np.abs(S.y).mean()
        print(f"growth w={w} {nm}: skill {mae0 / (mae0 + np.abs(S.y - w * S.rr).mean()):.3f}", end=" | ")
    print()
# level point weight (relative units)
for w in (0.5, 0.75, 0.9, 1.0):
    a = np.abs(te.lev_y.clip(-10, 10)).mean()
    b = np.abs(te.lev_y.clip(-10, 10) - w * te.lev_rr).mean()
    print(f"level w={w}: skill vs prior {a / (a + b):.3f}")
W_G, W_L = 0.75, 0.75


def bands(S, col, pred, qlo=5, qhi=95):
    out = {}
    for b, g in S.groupby("b"):
        r = g[col] - pred(g)
        out[b] = [float(np.percentile(r, qlo)), float(np.percentile(r, qhi))]
    return out

# growth bands, test IS vs the current artifact (g-based point and g-bucket band)
art = json.load(open("t4-local/agent/t4_agent/artifacts.json"))["families"]["eps_persistence"]
for nm, Str, Ste in (("ALL", tr, te), ("FIN", tr[tr.fin], te[te.fin])):
    bd = bands(Str, "y", lambda g: W_G * g.rr)
    lo = W_G * Ste.rr + Ste.b.map(lambda b: bd[b][0]); hi = W_G * Ste.rr + Ste.b.map(lambda b: bd[b][1])
    gb = Ste.g_last.clip(-300, 300)
    gbk = pd.cut(gb, bins=[-np.inf] + EDGES + [np.inf], labels=LABS).astype(str)
    band_key = "growth_band_fin" if nm == "FIN" else "growth_band"
    lo0 = 0.25 * gb + gbk.map(lambda b: art[band_key][b][0]); hi0 = 0.25 * gb + gbk.map(lambda b: art[band_key][b][1])
    print(f"{nm} growth band test: run-rate IS {IS(Ste.y, lo, hi):.1f} cov {np.mean((Ste.y >= lo) & (Ste.y <= hi)):.2f} | "
          f"current IS {IS(Ste.y, lo0, hi0):.1f} cov {np.mean((Ste.y >= lo0) & (Ste.y <= hi0)):.2f}")
# level bands
bl = bands(tr, "lev_y", lambda g: W_L * g.lev_rr)
y = te.lev_y.clip(-10, 10)
lo = W_L * te.lev_rr + te.b.map(lambda b: bl[b][0]); hi = W_L * te.lev_rr + te.b.map(lambda b: bl[b][1])
print(f"level band test (relative to |prior|): IS {IS(y, lo, hi):.3f} cov {np.mean((y >= lo) & (y <= hi)):.2f}")

# final refit on everything
ALL = P
final = {
    "edges": EDGES, "feature_clip": 300, "w_growth": W_G, "w_level": W_L,
    "p_up": {k: round(float(v), 4) for k, v in ALL.groupby("b").up_next.mean().items()},
    "growth_band": {k: [round(x, 1) for x in v] for k, v in bands(ALL, "y", lambda g: W_G * g.rr).items()},
    "growth_band_fin": {k: [round(x, 1) for x in v] for k, v in bands(ALL[ALL.fin], "y", lambda g: W_G * g.rr).items()},
    "level_band": {k: [round(x, 4) for x in v] for k, v in bands(ALL, "lev_y", lambda g: W_L * g.lev_rr).items()},
    "n": int(len(ALL)), "n_fin": int(ALL.fin.sum()),
}
print(json.dumps(final, indent=1))
json.dump(final, open("t4-local/dev/eps_runrate_fit.json", "w"), indent=1)
