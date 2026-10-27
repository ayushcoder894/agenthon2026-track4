"""Dense sentence encoder: NVIDIA NeMo Retriever embedding model nvidia/llama-nemotron-embed-1b-v2 (formerly
llama-3.2-nv-embedqa-1b-v2), baked into the image under /agent/models/nemo_embed (revision and checksum in
ARTIFACT_PROVENANCE.md). Nothing is downloaded at run time.

GPU-first: the encoder loads only when CUDA is available (the scoring cards grant a GPU) unless T4_EMBED_CPU=1.
Pooling = mean over all tokens (prefix included) then L2 normalisation, exactly as the model's sentence-transformers
configuration (modules.json: Transformer -> Pooling(mean) -> Normalize); queries take the prefix "query: ", passages
"passage: ". Loading runs in a background thread with a deadline: any failure, a missing library or a slow load means
`get()` returns None and every caller falls back to lexical retrieval. Never raises.
"""
from __future__ import annotations

import os
import sys
import threading
import time

_DIR = os.environ.get("T4_EMBED_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nemo_embed")
_STATE = {"enc": None, "tried": False, "error": None, "thread": None}
_LOCK = threading.Lock()


def _log(msg):
    print(f"[embed] {msg}", file=sys.stderr)


class Encoder:
    def __init__(self, model_dir: str):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import PreTrainedTokenizerFast

        from ._nemo_bidir import LlamaBidirectionalConfig, LlamaBidirectionalModel

        cpu_ok = os.environ.get("T4_EMBED_CPU") == "1"
        if not torch.cuda.is_available() and not cpu_ok:
            raise RuntimeError("no CUDA device")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = PreTrainedTokenizerFast.from_pretrained(model_dir)  # tokenizer_config: PreTrainedTokenizerFast (no config lookup)
        cfg = LlamaBidirectionalConfig.from_pretrained(model_dir)
        try:
            model = LlamaBidirectionalModel.from_pretrained(model_dir, config=cfg, dtype=self.dtype)
        except TypeError:
            model = LlamaBidirectionalModel.from_pretrained(model_dir, config=cfg, torch_dtype=self.dtype)
        self.model = model.to(self.device).eval()
        self.max_len = int(os.environ.get("T4_EMBED_MAXLEN", "160"))
        self.batch = int(os.environ.get("T4_EMBED_BATCH", "256" if self.device == "cuda" else "16"))
        name = torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"
        _log(f"loaded on {name} ({self.dtype}), max_len {self.max_len}, batch {self.batch}")

    def encode(self, texts, kind: str = "passage", deadline: float | None = None):
        """L2-normalised float32 numpy-free list-of-lists is avoided: returns a torch tensor on CPU (n, 2048), or None
        when the deadline passes first."""
        torch = self.torch
        prefix = "query: " if kind == "query" else "passage: "
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        with torch.inference_mode():
            for b in range(0, len(order), self.batch):
                if deadline is not None and time.time() > deadline:
                    return None
                idx = order[b:b + self.batch]
                enc = self.tok([prefix + texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hid = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
                emb = (hid * m).sum(1) / m.sum(1).clamp(min=1)
                emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu()
                for j, i in enumerate(idx):
                    out[i] = emb[j]
        return torch.stack(out) if out else torch.zeros((0, 2048))


def _load():
    try:
        t0 = time.time()
        enc = Encoder(_DIR)
        _STATE["enc"] = enc
        _log(f"ready in {time.time() - t0:.1f}s")
    except Exception as exc:  # missing torch, no GPU, bad weights: lexical retrieval only
        import traceback

        _STATE["error"] = f"{type(exc).__name__}: {exc}"[:200]
        if os.environ.get("T4_EMBED_DEBUG") == "1":
            traceback.print_exc()
        _log(f"unavailable ({_STATE['error']})")


def start():
    """Begin loading in the background (idempotent)."""
    if os.environ.get("T4_EMBED", "1") == "0" or not os.path.isdir(_DIR):
        return
    with _LOCK:
        if _STATE["thread"] is None:
            th = threading.Thread(target=_load, name="t4-embed-load", daemon=True)
            _STATE["thread"] = th
            th.start()


def get(wait_until: float | None = None):
    """The loaded encoder, waiting for the background load until `wait_until` (epoch seconds); None if unavailable."""
    if os.environ.get("T4_EMBED", "1") == "0" or not os.path.isdir(_DIR):
        return None
    start()
    th = _STATE["thread"]
    if th is not None and th.is_alive():
        th.join(timeout=max(0.0, (wait_until or time.time() + 90) - time.time()))
    return _STATE["enc"]
