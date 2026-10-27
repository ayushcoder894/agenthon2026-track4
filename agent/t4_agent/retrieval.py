"""Hybrid evidence retrieval (D20): BM25 over the agent's citable verbatim spans + NeMo Retriever dense embeddings
(t4_agent.embed, GPU-first), fused by reciprocal rank fusion (k = 60) and diversified with maximal marginal relevance.

Candidates are exactly the spans the claim builder can cite (claims._segments: single-line, contentful, digit-bearing,
cutoff-safe documents admitted for the entity), so every retrieved passage is a verbatim, citable quote. Without a
usable encoder (no GPU, load failure, deadline) the ranking is BM25 alone. Never raises; returns [] on any failure.
"""
from __future__ import annotations

import math
import re
import time

from . import embed as _embed
from .claims import _segments

RRF_K = 60
_TOK = re.compile(r"[a-z][a-z0-9]{2,}|\d+(?:\.\d+)?%?")   # hyphenated words split: "going-concern" matches "going concern"
_STOP = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has", "have", "had", "its", "their", "our",
         "which", "will", "would", "could", "may", "been", "such", "into", "than", "also", "other", "any", "all", "each", "per",
         "not", "but", "these", "those", "there", "what", "how", "own"}
MAX_CANDS = 60000          # spans embedded per unit at most (B200: well under a minute)
DENSE_WAIT = 150.0         # seconds the first caller waits for the encoder to load


def _toks(s: str):
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


class Index:
    """All citable spans of the given entities (entity-specific documents first, then shared ones)."""

    def __init__(self, unit, entity_ids, deadline: float):
        self.unit = unit
        self.items = []          # (entity_id, doc_id, s, e, text)
        seen = set()
        for eid in entity_ids:
            for doc in unit.docs_for(eid):
                for s, e in _segments(doc):
                    key = (eid, doc.doc_id, s)
                    if key in seen:
                        continue
                    seen.add(key)
                    self.items.append((eid, doc.doc_id, s, e, doc.text[s:e]))
                    if len(self.items) >= MAX_CANDS:
                        break
        self.toks = [_toks(it[4]) for it in self.items]
        n = len(self.toks)
        df = {}
        for tt in self.toks:
            for t in set(tt):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.avgdl = (sum(len(t) for t in self.toks) / n) if n else 1.0
        self.tf = [{} for _ in range(n)]
        for i, tt in enumerate(self.toks):
            d = self.tf[i]
            for t in tt:
                d[t] = d.get(t, 0) + 1
        self.emb = None
        self.dense = False
        enc = _embed.get(min(deadline, time.time() + DENSE_WAIT)) if self.items else None
        if enc is not None:
            try:
                emb = enc.encode([it[4] for it in self.items], "passage", deadline=deadline)
                if emb is not None:
                    self.emb, self.dense, self.enc = emb, True, enc
            except Exception:
                self.emb, self.dense = None, False

    def bm25(self, query: str, k1: float = 1.2, b: float = 0.75):
        q = set(_toks(query))
        out = []
        for i, tf in enumerate(self.tf):
            sc = 0.0
            dl = len(self.toks[i]) or 1
            for t in q:
                f = tf.get(t)
                if f:
                    sc += self.idf.get(t, 0.0) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
            out.append(sc)
        return out

    def rank(self, query: str, eligible=None, top: int = 40):
        """Indices of the best spans for `query` by RRF of BM25 and dense ranks (BM25 alone without an encoder)."""
        idx = [i for i in range(len(self.items)) if eligible is None or eligible(self.items[i])]
        if not idx:
            return [], {}
        bm = self.bm25(query)
        bm_order = sorted(idx, key=lambda i: -bm[i])
        fused = {}
        for r, i in enumerate(bm_order[:400]):
            if bm[i] > 0:
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        dense_sim = {}
        if self.dense:
            try:
                qv = self.enc.encode([query], "query")[0]
                sims = (self.emb[idx] @ qv).tolist()
                dense_sim = dict(zip(idx, sims))
                for r, i in enumerate(sorted(idx, key=lambda j: -dense_sim[j])[:400]):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
            except Exception:
                dense_sim = {}
        order = sorted(fused, key=lambda i: -fused[i])[:top]
        return order, dense_sim

    def mmr(self, order, k: int, lam: float = 0.75, spread_entities: bool = True):
        """Pick k of `order` (best first) balancing relevance and novelty (dense cosine, else token Jaccard)."""
        chosen = []
        rel = {i: 1.0 - r / max(1, len(order)) for r, i in enumerate(order)}
        ents_all = {self.items[i][0] for i in order}
        while order and len(chosen) < k:
            best, best_sc = None, -1e9
            used_ents = {self.items[c][0] for c in chosen}
            for i in order:
                if i in chosen:
                    continue
                if spread_entities and self.items[i][0] in used_ents and len(used_ents) < len(ents_all):
                    continue
                if chosen:
                    if self.dense:
                        sim = max(float(self.emb[i] @ self.emb[c]) for c in chosen)
                    else:
                        a = set(self.toks[i])
                        sim = max(len(a & set(self.toks[c])) / max(1, len(a | set(self.toks[c]))) for c in chosen)
                else:
                    sim = 0.0
                sc = lam * rel[i] - (1 - lam) * sim
                if sc > best_sc:
                    best, best_sc = i, sc
            if best is None:
                break
            chosen.append(best)
        return chosen


DENSE_FLOOR = 0.20         # cosine below this is not evidence for a driver (relevant ~0.5, boilerplate ~0.0-0.2)


def rank_dense(idx, query: str, eligible=None, top: int = 40):
    """Indices by dense cosine alone (exp G62: dense beat the BM25+dense mix for evidence selection), above DENSE_FLOOR."""
    if not idx.dense:
        return [], {}
    ids = [i for i in range(len(idx.items)) if eligible is None or eligible(idx.items[i])]
    if not ids:
        return [], {}
    qv = idx.enc.encode([query], "query")[0]
    sims = dict(zip(ids, (idx.emb[ids] @ qv).tolist()))
    order = [i for i in sorted(ids, key=lambda j: -sims[j]) if sims[i] >= DENSE_FLOOR][:top]
    return order, sims


_CACHE = {}


def index_for(unit, deadline: float):
    """One index per unit (all entities), built once."""
    key = id(unit)
    if key not in _CACHE:
        try:
            _CACHE[key] = Index(unit, [r["entity_id"] for r in unit.entities], deadline)
        except Exception:
            _CACHE[key] = None
    return _CACHE[key]
