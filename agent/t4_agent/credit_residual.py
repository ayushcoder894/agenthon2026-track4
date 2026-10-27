"""D22: NeMo semantic residual for the credit rule's probability (vendored, predict-only, numpy only).

THEORY-SPECIFIED PRIOR (mode "roster_relative_prior"): no parameter is fitted to any outcome. For each entity the 7 standing risk queries give the mean-of-top-3 cosine
similarity of its latest filing's shortlisted passages; each query's similarity is z-scored ACROSS THE UNIT'S ROSTER (so no training-set scale is needed), and
S = sum_k w_k z_k with fixed signs/weights from the economics (going concern, liquidity, funding, covenants, restructuring, operating stress raise distress;
"resolution" lowers it). The rule's probability p0 is then moved on the logit scale by c*S (c = 0.15) and the label is event iff p >= 0.5. With p0 in {0.6, 0.55, 0.03}
this can only REMOVE a rule-positive whose filing reads clearly milder than its peers (0.55 -> needs S < -1.3; 0.6 -> S < -2.7); it can never create a positive
(p0 = 0.03 would need S > 24). Requires >= 5 entities with features. Unvalidated by design: see ARTIFACT_PROVENANCE.md.

Chunking / shortlisting / pooled-similarity features follow the external research package D21R_semantic_residual_candidate (read and tested before use);
the learned part is an ordinary L2-regularised logistic residual on top of the rule's own probability, stored as JSON coefficients in
artifacts.json -> credit_residual. It is enabled ONLY when the pre-registered chronological gate (cli.py deployment_gate) passed on historical filings
(fit data: SEC filings on or before 2022-06-28, label availability <= training cutoff). Any failure returns None and the caller keeps the rule.
"""
from __future__ import annotations

import math
import re
import time

from . import artifacts as _art
from . import embed as _embed

CREDIT_QUERIES = {
    "liquidity": "The company may have inadequate cash and liquidity to meet near-term financial obligations.",
    "funding": "The issuer faces difficulty refinancing debt, securing lending facilities, or raising funds.",
    "covenants": "The borrower risks breaching debt covenants, loan agreements, or mandatory debt payments.",
    "going_concern": "There is substantial doubt about the business continuing as a going concern.",
    "operating_stress": "Operating losses, declining revenues, or negative cash flows threaten the company.",
    "restructuring": "The company is restructuring liabilities, negotiating with creditors, or facing insolvency.",
    "resolution": "The company has resolved its outstanding debt obligations, secured financing, or improved liquidity.",
}
FINANCE_TERMS = ("debt", "liquidity", "loan", "cash flow", "covenant", "credit", "borrow", "restructur", "default", "going concern", "maturit", "refinanc",
                 "interest expense", "financing", "insolvenc", "working capital", "operating loss", "payment", "funding", "risk factor")


def split_chunks(text, max_chars=1500, overlap=160):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, pos = [], 0
    while pos < len(text) and text:
        end = min(len(text), pos + max_chars)
        if end < len(text):
            last = text.rfind(" ", max(pos + max_chars // 2, pos + 1), end)
            if last > pos:
                end = last
        if end <= pos:
            end = min(len(text), pos + max_chars)
        chunks.append(text[pos:end])
        if end == len(text):
            break
        pos = max(pos + 1, end - overlap)
    return chunks


def shortlist(chunks, max_passages=128):
    import numpy as np

    if len(chunks) <= max_passages:
        return chunks
    wanted = max(1, 3 * max_passages // 4)
    ranking = sorted(range(len(chunks)), key=lambda i: (-sum(t in chunks[i].lower() for t in FINANCE_TERMS), i))
    top = set(ranking[:wanted])
    top.update(int(i) for i in np.linspace(0, len(chunks) - 1, max_passages - wanted).astype(int))
    if len(top) < max_passages:
        top.update(i for i in ranking if i not in top and len(top) < max_passages)
    return [chunks[i] for i in sorted(top)[:max_passages]]


def features(passage_emb, query_emb, keys):
    """passage_emb (n, d), query_emb (7, d) L2-normalised float arrays -> the package's 7 x 5 + 2 pooled features (no prior filing: delta_max = 0)."""
    import numpy as np

    sc = query_emb @ passage_emb.T
    f = {"doc:log_passages": float(np.log1p(passage_emb.shape[0])), "doc:has_prior": 0.0}
    for i, key in enumerate(keys):
        order = np.sort(sc[i, :])
        n = order.size
        best = float(order[-1])
        f[f"sem:{key}:max"] = best
        f[f"sem:{key}:top3"] = float(order[-min(3, n):].mean())
        f[f"sem:{key}:topdecile"] = float(order[-max(1, int(np.ceil(n / 10))):].mean())
        f[f"sem:{key}:concentration"] = float(best - np.median(sc[i, :]))
        f[f"sem:{key}:delta_max"] = 0.0
    return f


def predict(model, p0, sig):
    import numpy as np

    x = np.array([float(sig.get(k, float("nan"))) for k in model["feature_keys"]], dtype=float)
    m, s = np.asarray(model["medians"]), np.asarray(model["scale"])
    x = np.where(np.isfinite(x), x, m)
    x = np.clip((x - m) / s, -6.0, 6.0)
    z = math.log(p0 / (1 - p0)) + model["intercept"] + float(x @ np.asarray(model["weights"]))
    return 1.0 / (1.0 + math.exp(-z))


PRIOR_W = {"going_concern": 0.9, "liquidity": 0.6, "funding": 0.5, "covenants": 0.5, "restructuring": 0.6, "operating_stress": 0.4, "resolution": -0.5}
PRIOR_C = 0.15
PRIOR_THRESHOLD = 0.5


def roster_prior(feats, rule_p):
    """feats: {entity: feature dict}; returns {entity: probability} using the roster-relative z-scores of sem:<q>:top3."""
    import numpy as np

    ents = [e for e in feats if feats[e]]
    if len(ents) < 5:
        return {}
    z = {e: 0.0 for e in ents}
    for q, w in PRIOR_W.items():
        v = np.array([feats[e].get(f"sem:{q}:top3", np.nan) for e in ents], dtype=float)
        if not np.isfinite(v).all() or float(v.std()) < 1e-6:
            continue
        zz = np.clip((v - v.mean()) / v.std(), -2.5, 2.5)
        for e, x in zip(ents, zz):
            z[e] += w * float(x)
    out = {}
    for e in ents:
        p0 = float(rule_p[e])
        out[e] = 1.0 / (1.0 + math.exp(-(math.log(p0 / (1 - p0)) + PRIOR_C * max(-8.0, min(8.0, z[e])))))
    return out


def prefetch(unit, entity_texts, rule_p, log=print, wall=90.0):
    """{entity_id: new probability} for entities whose filing text could be encoded; {} when disabled, no GPU, or on any failure.
    entity_texts: {entity_id: latest 10-K/10-Q text}; rule_p: {entity_id: the rule's probability}."""
    art = _art.get("credit_residual")
    prior_mode = bool(art) and art.get("mode") == "roster_relative_prior"
    if not art or not art.get("enabled") or (not prior_mode and not art.get("model", {}).get("enabled")):
        return {}
    try:
        import numpy as np

        import os as _os

        t_start = float(_os.environ.get("T4_T0", time.time()))
        deadline = min(time.time() + wall, t_start + 330.0)   # hard cap: the answer file is written after this layer, the unit limit is 600 s
        if deadline - time.time() < 40:
            log("[credit_residual] skipped: not enough time left in the unit budget")
            return {}
        enc = _embed.get(min(deadline - 60, time.time() + 120))
        if enc is None:
            return {}
        keys = sorted(CREDIT_QUERIES)
        q = enc.encode([CREDIT_QUERIES[k] for k in keys], "query", deadline=deadline)
        if q is None:
            return {}
        q = q.numpy().astype("float64")
        old_len = enc.max_len
        enc.max_len = 512          # the historical features were built with 512-token chunks
        out = {}
        for eid, text in entity_texts.items():
            if time.time() > deadline - 20:
                break
            chunks = shortlist(split_chunks(text), 128)
            if not chunks:
                continue
            emb = enc.encode(chunks, "passage", deadline=deadline)
            if emb is None:
                break
            sig = features(emb.numpy().astype("float64"), q, keys)
            if prior_mode:
                out[eid] = sig
            else:
                out[eid] = predict(art["model"], float(rule_p[eid]), sig)
        enc.max_len = old_len
        if prior_mode:
            out = roster_prior(out, rule_p)
        log(f"[credit_residual] semantic residual applied to {len(out)} of {len(entity_texts)} entities")
        return out
    except Exception as exc:
        log(f"[credit_residual] unavailable ({type(exc).__name__})")
        return {}
