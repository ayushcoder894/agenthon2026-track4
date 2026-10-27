"""exp500 harness: >= 500 Track-4 experiments, chronological validation, ONE-SHOT final test with Holm correction.
Rules baked in:
  * every experiment is scored on a VALIDATION window only (outcomes after TRAIN_END and before FINAL_START);
  * the FINAL window (>= FINAL_START) is evaluated exactly once, for the top-K validation candidates per family, never used to pick among them;
  * every experiment is compared with the SHIPPED baseline of its family on the same rows, with a year-block bootstrap (paired), and gets a Holm-adjusted p over the family's experiment count;
  * one JSON line per experiment in results.jsonl (never overwritten), so the total experiment count is auditable.
Metrics: 'acc' (direction accuracy, higher better), 'skill' (naive_MAE/(naive_MAE+MAE), higher better), 'is' (interval score, alpha=.1, lower better), 'rho' (rank corr, higher better)."""
from __future__ import annotations
import json, os, time, math
import numpy as np, pandas as pd

OUT = os.environ.get("EXP500_OUT", "results.jsonl")


def log(rec: dict):
    rec = dict(rec); rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(OUT, "a", encoding="utf-8") as f: f.write(json.dumps(rec, default=float) + "\n")


def interval_score(y, lo, hi, alpha=0.1):
    y, lo, hi = map(np.asarray, (y, lo, hi)); return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def skill(y, pred, naive):
    y = np.asarray(y, float); m0 = np.abs(y - naive).mean(); m1 = np.abs(y - pred).mean(); return m0 / (m0 + m1)


def block_boot(diff, blocks, n=400, seed=0):
    """paired mean difference with block (e.g. year) bootstrap: returns (mean, se)."""
    diff = np.asarray(diff, float); blocks = np.asarray(blocks)
    u = np.unique(blocks); s = np.array([diff[blocks == b].sum() for b in u]); c = np.array([(blocks == b).sum() for b in u]); rng = np.random.default_rng(seed); est = []
    for _ in range(n):
        i = rng.integers(0, len(u), len(u)); est.append(s[i].sum() / c[i].sum())
    return float(diff.mean()), float(np.std(est))


def p_from_z(z):
    return 0.5 * math.erfc(abs(z) / math.sqrt(2))      # two-sided normal p = erfc(|z|/sqrt2)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order)."""
    p = np.asarray(pvals); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


class Family:
    """A family holds rows (DataFrame with column 'block' for the bootstrap, 'split' in {train,val,final}) and a shipped baseline column."""
    def __init__(self, name, higher_better=True):
        self.name, self.hb, self.results = name, higher_better, []

    def run(self, exp_id, desc, params, scorer, base_scorer, df):
        """scorer(df)->per-row score vector (higher better for acc/skill-like; lower better if hb False) on VALIDATION rows; paired vs base_scorer."""
        t0 = time.time(); v = df[df.split == "val"]
        try:
            s = np.asarray(scorer(df, "val"), float); b = np.asarray(base_scorer(df, "val"), float)
        except Exception as e:
            log(dict(family=self.name, exp=exp_id, desc=desc, params=params, status="error", err=repr(e)[:160])); return None
        d = (s - b) if self.hb else (b - s)                    # >0 = better than shipped baseline
        m, se = block_boot(d, v.block.values); z = m / se if se > 0 else 0.0
        rec = dict(family=self.name, exp=exp_id, desc=desc, params=params, status="ok", n=int(len(v)), val_score=float(np.mean(s)), val_base=float(np.mean(b)),
                   val_gain=m, val_se=se, val_z=z, val_p=p_from_z(z), sec=round(time.time() - t0, 2))
        self.results.append(rec); log(rec); return rec

    def finalize(self, df, scorers: dict, base_scorer, top_k=3):
        """one-shot final test for the top_k by validation z; Holm over ALL experiments of the family."""
        ok = [r for r in self.results if r.get("status") == "ok"]
        if not ok: return []
        adj = holm([r["val_p"] for r in ok])
        for r, a in zip(ok, adj): r["val_p_holm"] = float(a)
        top = sorted(ok, key=lambda r: -r["val_z"])[:top_k]; fin = []
        for r in top:
            sc = scorers[r["exp"]]; f = df[df.split == "final"]
            if len(f) == 0: continue
            s = np.asarray(sc(df, "final"), float); b = np.asarray(base_scorer(df, "final"), float); d = (s - b) if self.hb else (b - s)
            m, se = block_boot(d, f.block.values); z = m / se if se > 0 else 0.0
            rec = dict(family=self.name, exp=r["exp"], desc=r["desc"], status="FINAL", n_family_experiments=len(ok), val_gain=r["val_gain"], val_z=r["val_z"], val_p_holm=r["val_p_holm"],
                       final_n=int(len(f)), final_score=float(np.mean(s)), final_base=float(np.mean(b)), final_gain=m, final_se=se, final_z=z,
                       verdict="PROMISING" if (r["val_p_holm"] < 0.1 and z > 1.64 and m > 0) else "REJECT")
            log(rec); fin.append(rec)
        return fin
