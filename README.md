# Agenthon 2026 — Track 4 (Explainability / Analysis): team 662

An evidence-grounded financial forecasting agent for the NeurIPS 2026 Agenthon, Track 4. Each task ("unit") gives a table of
entities, a target to forecast for each one, and a **frozen corpus** of documents dated on or before the task's cutoff. The agent
must answer as of the cutoff, with no internet access (only the organizers' House LLM through an audited proxy), and must back every
claim with a citation into the corpus.

## The challenge

- **Targets:** regression (e.g. next CPI component print, Treasury auction bid-to-cover, bank EPS growth, yield changes after an FOMC
  meeting), classification (EPS up/down, credit event within 12 months, macro revision direction, post-earnings reaction) and ranking
  (change in speculative positioning, COT).
- **Scoring (scorer 5.3.0):** per unit, `0.7 x q + 0.3 x min(iq, max(0.5, q))` where `q = naive_MAE / (naive_MAE + MAE)` and
  `iq = naive_IS / (naive_IS + IS)` (90% interval score, 20x miss penalty). Classification and ranking are anchored so the unit's hidden
  naive rule scores 0.5. Leaderboard = `-0.27 + 1.27 x analysis` (+ `0.25 x reasoning` in the Final, graded against hidden target reasons).
  A pure naive answer therefore shows **0.365**.
- **Rules that shaped the design:** no data after a task's cutoff (for inputs, fitting or model selection); no extra neural checkpoints
  (only the House LLM and NeMo Retriever embeddings); no stored task-to-answer lookups; fitted non-neural models allowed if trained on
  data available before the cutoff and disclosed.
- **Development set:** 10 public practice units. **Final:** a larger hidden set, most families with no public counterpart, ≤ 12 entities
  per unit, 600 s per unit, 12 h per run.

## Approach

1. **Specialised family predictors** (`agent/t4_agent/predict.py`), each fitted and validated out of time on public history
   (ALFRED vintages, SEC XBRL, CFTC, Treasury FiscalData, EIA): CPI components (recent-average blend, gasoline pass-through,
   turning-point and cross-component signals, apparel → 0), EPS run-rate direction and growth with LightGBM quantile bands, COT
   mean reversion, auction recent-average with new-issue/reopening adjustment, credit distress rules (going concern, distress-phrase
   types, negative equity with losses), macro revision tendencies, calibrated no-change bands for rates.
2. **Global LightGBM forecaster for unseen families** (`global_forecaster/`, runtime in `agent/t4_agent/gforecast.py` + `gfeat.py`):
   one model trained on 1.8M forecast cases from 367 FRED/ALFRED series (first-published values only, dated by publication) plus
   never-revised market series, on 22 scale-free features (recent moves, mean reversion, seasonality, volatility, autocorrelation,
   history length). Shipped as JSON trees evaluated in pure Python, in five copies with data limits 2005-2022; a task uses the latest
   copy whose data ends ≥ 6 months before its cutoff.
3. **Cutoff guard** on every fitted artifact, calibrated 90% intervals, and verbatim-quote claims to pass the faithfulness checks.
4. **Research harness** (`research/exp500/common.py`): chronological train/validation/one-shot-final splits, quarter/year-block
   bootstrap, Holm correction — 600+ experiments across nine families.

## Results

| Build | Dev analysis | Dev leaderboard |
|---|---|---|
| Naive rule | 0.500 | 0.365 |
| D20R (first general build, official) | 0.619 | 0.516 |
| **Final general model** (local scorer, reproduces official scores exactly) | **0.623** | **0.522** |
| Unseen-family proxy (all units forced through the generic path), before → after global model | 0.460 → 0.483 | 0.315 → 0.343 |

Global forecaster, out of time (pseudo-units scored with the scorer's point-quality formula; 0.5 = carry-forward):

| Window | Previous generic rule | Global model | Interval score vs random-walk band |
|---|---|---|---|
| 2013-17 (validation) | 0.495-0.506 | **0.535** | −16% |
| 2018-22 (test) | 0.486-0.499 | **0.535** | −21% |
| 2023-26 (test) | 0.485-0.495 | **0.521** | −21% |

### What we learned
- Development-tuned constants (our in-sample "DEV-ONLY" builds reached 0.75-0.77 on the Development board, 0.745 official) do **not**
  generalise: on history the same rules lost (e.g. auction max-of-3 blend −0.068 composite, z −16; FOMC drift: zero skill over 168
  meetings). The Final uses none of them.
- The Dev outcomes were dominated by one-off post-cutoff events (FOMC yield jumps of 2-3σ, hurricane-driven used-car prices, a
  bankruptcy after a profitable 10-K) that no pre-cutoff signal predicts; the honest ceiling on those ten units is ≈ 0.52.
- Training on revised vs first-published CPI: revised-trained pass-through wins even when scored on first prints (+0.077 MAE, z 3.5).
- Score-formula tricks (bold labels, edge-dependent band width) are worth ≈ +0.001; calibrated quantile bands are already near-optimal.
- Most remaining Final upside is in reasoning: task prompts list the drivers the authors want ("Reason from …"), which likely mirror
  the hidden target reasons.

## Repository layout

```
agent/                  Final agent package (t4_agent) and Dockerfile (built FROM our D20R base image)
global_forecaster/      fetch (FRED/ALFRED API), panel build, training/evaluation, export, band-policy test
research/exp500/        family experiments (CPI, auctions, COT, EPS, macro, FOMC, post-earnings) and the experiment harness
research/dev/           local scorer (localscore.py), host prediction harness (hostpred.py), audits and earlier studies
ARTIFACT_PROVENANCE_FINAL.md   data sources, data limits and the cutoff guard
```

Not included: downloaded datasets (FRED/ALFRED, SEC, CFTC, Treasury, EIA — fetch them with the scripts and your own API keys), the
fitted `artifacts.json` and `global_models.json` (regenerate with the scripts), embedding model weights, and all credentials.

## Reproduce the global forecaster
```
python global_forecaster/fetch_catalog.py      # needs a free FRED API key in D:/AGENTHON/fred_key.txt (path set in the script)
python global_forecaster/build_panel.py
python global_forecaster/train_eval.py 2012
python global_forecaster/export_models.py 2005 2010 2015 2020 2022
```
