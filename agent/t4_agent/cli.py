"""`analyze --task T --corpus C --out O` — always exits 0 and always leaves a valid answer.json.

The deterministic answer is written FIRST. Anything that follows (optional model refinement) may
only replace it with an answer that passes the same local validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .answer import build_answer, minimal_answer, validate_answer
from .unit_io import load_unit


def _write(path: str, answer: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    data = json.dumps(answer, ensure_ascii=True, indent=1)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(data)


def run(task_path: str, corpus_dir: str, out_path: str, log=print) -> dict:
    t0 = time.time()
    os.environ["T4_T0"] = repr(t0)  # agent start: optional layers inside the deterministic build cap themselves against it
    if os.environ.get("MODEL_ENDPOINT"):
        try:  # D20: start loading the NeMo Retriever encoder (GPU) in the background while the deterministic answer is built
            from . import embed as _embed

            _embed.start()
        except Exception:
            pass
    unit = load_unit(task_path, corpus_dir)
    from . import claims as _claims

    _claims.set_deadline(t0 + 240)  # corpus-claim search may use at most ~4 minutes; task-row claims remain
    answer = None
    try:
        answer = build_answer(unit)
        problems = validate_answer(unit, answer)
        if problems:
            log(f"[t4-agent] deterministic answer failed local validation: {problems[:5]}")
            answer = None
    except Exception as exc:  # never crash
        log(f"[t4-agent] deterministic build failed: {type(exc).__name__}: {exc}")
    if answer is None:
        answer = minimal_answer(unit)
        problems = validate_answer(unit, answer)
        if problems:
            log(f"[t4-agent] minimal answer also has problems: {problems[:5]}")
    _write(out_path, answer)
    log(f"[t4-agent] wrote {out_path} in {time.time() - t0:.1f}s")

    # optional model refinement (Phase 2) — never allowed to damage the answer above
    try:
        if not os.environ.get("MODEL_ENDPOINT"):
            return answer
        from . import reasons_llm as _rl  # noqa: WPS433
        from . import decide as _decide0  # noqa: WPS433

        if _decide0.enabled() and _rl.enabled():  # D21: model label votes / bounded moves first, then reasons on the final answer
            decided = _decide0.decide(unit, answer, deadline=t0 + 380, log=log)
            if decided is not None:
                problems = validate_answer(unit, decided)
                if problems:
                    log(f"[t4-agent] decided answer rejected by local validation: {problems[:3]}")
                else:
                    _write(out_path, decided)
                    log("[t4-agent] decided answer accepted")
                    answer = decided
            cand = _rl.write_reasons(unit, answer, deadline=t0 + 540, log=log)
            if cand is not None and not validate_answer(unit, cand):
                _write(out_path, cand)
                log("[t4-agent] model-written reasons accepted")
                answer = cand
            return answer
        if _rl.enabled():  # variant D5R: predictions stay deterministic; the model only writes the submitted_reasons
            cand = _rl.write_reasons(unit, answer, deadline=t0 + 540, log=log)
            if cand is not None and not validate_answer(unit, cand):
                _write(out_path, cand)
                log("[t4-agent] model-written reasons accepted")
                answer = cand
            return answer
        from . import analyst as _analyst  # noqa: WPS433

        if _analyst.enabled():  # variant L: the model is the analyst; the fitted answer is the prior and the fallback
            cand = _analyst.analyze(unit, answer, deadline=t0 + 540, log=log)
            if cand is not None:
                problems = validate_answer(unit, cand)
                if problems:
                    log(f"[t4-agent] analyst answer rejected by local validation: {problems[:3]}")
                else:
                    _write(out_path, cand)
                    log("[t4-agent] analyst answer accepted")
                    answer = cand
            return answer
        from . import decide as _decide  # noqa: WPS433

        if _decide.enabled():  # variants H1 / H2: label votes and guarded nudges (replaces the evidence/reasons layer)
            decided = _decide.decide(unit, answer, deadline=t0 + 540, log=log)
            if decided is not None:
                problems = validate_answer(unit, decided)
                if problems:
                    log(f"[t4-agent] decided answer rejected by local validation: {problems[:3]}")
                else:
                    _write(out_path, decided)
                    log("[t4-agent] decided answer accepted")
                    answer = decided
            return answer
        if os.environ.get("T4_NO_REFINE") == "1":
            return answer
        from .refine import refine  # noqa: WPS433

        refined = refine(unit, answer, deadline=t0 + 540, log=log)
        if refined is not None and not validate_answer(unit, refined):
            _write(out_path, refined)
            log("[t4-agent] refined answer accepted")
    except ImportError:
        pass
    except Exception as exc:
        log(f"[t4-agent] refinement ignored: {type(exc).__name__}: {exc}")
    return answer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="analyze")
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    try:
        args, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        # argparse already printed usage; still exit 0 only if we can find our inputs
        return 0
    try:
        run(args.task, args.corpus, args.out, log=lambda m: print(m, file=sys.stderr))
    except BaseException as exc:  # absolutely never non-zero
        print(f"[t4-agent] fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            try:
                task = json.load(open(args.task, encoding="utf-8"))
            except BaseException:
                task = {}
            eps = [{"entity_id": r["entity_id"], "point_forecast": 0.0,
                    "interval": {"level": 0.9, "lo": -1.0, "hi": 1.0}, "claims": []}
                   for r in task.get("entities", []) if isinstance(r, dict) and "entity_id" in r]
            _write(args.out, {"task_id": task.get("task_id", ""), "entity_predictions": eps})
        except BaseException:
            pass
    return 0
