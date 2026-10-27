# Artifact provenance — Track 4 Final submission (team 662)

Image: `ghcr.io/ayushcoder894/agenthon-t4-agent@sha256:cd25e3d5b20e9454411257b75fb6833573e72612652c3e52140ed8b6dd28ed76`
Base: the D20R Development image (`agenthon-t4-agent:d20r-local`) with `/agent/t4_agent/` replaced. `T4_DEVONLY=0`: no Development-only (in-sample) rule is active.

## Neural models
Only the House model (`nvidia/nemotron-3-super-120b-a12b`, through the organizer proxy) and the NeMo Retriever embedding model (`nvidia/llama-nemotron-embed-1b-v2`). No other neural checkpoint.

## Fitted non-neural artifacts (numbers only, no task answers)
| File | Content | Data limit | Use |
|---|---|---|---|
| `artifacts.json` | family coefficients, quantile bands, LightGBM quantile trees (D20R) | 2022-06-28 | only for tasks whose cutoff is >= 30 days after the data limit (cutoff guard in `artifacts.py`) |
| `cutoffsafe_artifacts.json` | CPI turning-point coefficient and cross-component ridge; macro revision up-shares by series x revision number | 2022-06-28 | same guard (`predict._cs_art`) |
| `global_models.json` | global LightGBM forecaster (point = L1, bands = 5%/95% quantile trees), five copies | 2005-12-31, 2010-12-31, 2015-12-31, 2020-12-31, 2022-12-31 | a task uses the latest copy whose data limit is >= 183 days before its cutoff; none otherwise (`gforecast.py`) |

### Global forecaster training data
367 series from the FRED/ALFRED API (St. Louis Fed): 267 revised macro series taken as **initial-release (first-published) values only** (ALFRED `output_type=4`, with each value's publication date), and 100 never-revised market series (H.15 rates, H.10 FX, spreads, commercial paper, commodity and gasoline prices). Training rows are dated by publication date; each copy uses only rows published on or before its data limit. Features are scale-free (`gfeat.py`); target is the h-period change divided by the series' typical one-period move. Retrieved 2026-10-21/22.

### Selection
Design and hyper-parameters were chosen on rationale and on historical out-of-time tests (train <= 2012, validation 2013-2017, tests 2018-2022 and 2023-2026). The ten Development practice units' outcomes were used for diagnosis only (practice-unit exception, Issue #24); no constant in this image was set from them. One change (CPI apparel point = 0) has no fitted parameter and was supported by 1990-2024 first-print history.
