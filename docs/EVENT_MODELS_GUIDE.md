# WeatherGPT event models — user and developer guide

Twenty small LightGBM specialists that turn the newest NOAA **GFS** + **GEFS** run into calibrated answers for any **land** point in India
(6–38 °N, 67–98 °E) up to **10 days** ahead. 19 pass the admission gate and are served. Design and verbatim results: `docs/AGENTIC_ARCHITECTURE.md`.
Hands-on: `notebooks/weathergpt_events_implementation.ipynb`.

> Research prototype for SIH 2026. Not an official forecast; do not use it for safety-critical decisions. Official warnings stay with IMD / CAP.

## 1. What it answers

| model | question | horizon |
|---|---|---|
| `thunderstorm`, `fog`, `strong_wind`, `rain_3h`, `dust` | chance of the event in each 3-hour window (METAR-defined: thunder, visibility < 1 km, wind ≥ 25 kt, rain, dust) | 10 d, 52 steps |
| `temperature_range`, `wind_range`, `humidity_range` | hourly 10th / 50th / 90th percentile plus an 80 % conformal interval | 10 d |
| `hot_day`, `cold_night`, `heatwave_imd` | chance the IST-day max ≥ 40 °C / min ≤ 5 °C / an IMD-style heat-wave day | 9 d |
| `tmax_range`, `tmin_range` | the daily maximum / minimum as a range | 9 d |
| `rain_curve` | for each UTC day: P(rain ≥ t) at 16 thresholds, P(≥1 mm), P(≥2.5 mm = IMD rainy day), IMD class probabilities, amount **if** it rains (10/50/90th) | 10 d |
| `rain_3day_any_2p5mm`, `rain_3day_any_15mm`, `rain_7day_any_2p5mm` | chance of at least that much on one or more of the next 3 / 7 days | 10 d |
| `rain_3day_total`, `rain_7day_total` | the 3- / 7-day total as an exceedance curve ("how much rain this week") | 10 d |
| `coldwave_imd` | **refused by the gate** (worse than the global base rate) | — |

## 2. Use it

```python
from weathergpt_events.loader import load_engine
from weathergpt_events.service import ForecastService, summarise

engine = load_engine("Arko007/weathergpt-events", token=None)       # HF repo id (token while private) or a local artifact folder
service = ForecastService.live(engine, cache_dir="run_cache")      # newest 00Z GFS+GEFS run; ~1.5 min the first time, then cached
result = service.forecast(28.61, 77.21, targets=["rain_curve", "tmax_range"], horizon_days=3)
print(summarise(result))
```

```bash
python -m weathergpt_events --source Arko007/weathergpt-events status
python -m weathergpt_events --source ./artifacts forecast --lat 13.08 --lon 80.27 --targets rain_curve,tmax_range --horizon 3 [--json]
```
`HF_TOKEN` (environment) is used for a private repo. Exit code 2 means "no answer available" (refused place or no fresh run).

```python
from weathergpt_events.api import create_router        # FastAPI: GET /status /catalogue /forecast, POST /tool
app.include_router(create_router(service), prefix="/events")
```

Local-machine note: `load_engine("<repo id>")` downloads the models; on a small laptop point it at a folder that already holds them, or run it on Kaggle / Colab.

## 3. Reading an answer

`forecast()` returns `{"available", "run", "location": {lat, lon, land_fraction, coastal}, "targets": {name: {...}}}`.

* **`available: false` + `reason`** — a refusal is an answer: sea, outside the box, a cell with too little land, a stale run (> 54 h), an unknown or unserved model.
* Each target has `zone` and `entries`. **An entry with `validated: false` carries no numbers**, only `note` (the model has no held-out skill in that zone / lead).
  `include_unvalidated=True` shows them for debugging only.
* Probability models: `probability`. Range models: `q10 q50 q90` and `lo hi` (the 80 % interval) with `nominal_coverage`.
* `rain_curve` / `*_total`: `exceedance` (`{"<mm>": P(≥ mm)}`, monotone), and `amount_if_wet_mm` (`None` when rain is too unlikely for a conditional amount to mean
  anything). `amount_lower_bound` marks quantiles beyond the last threshold. The target also carries **`thresholds_without_skill_mm`**: thresholds (about 90 mm/day and
  up) at which held-out data show no skill over climatology — returned, but not validated.
* Times: step models use `valid_time_utc` (the 3-hour window is centred on it); station-day models use `ist_day`; rain uses the UTC day; windows give `window_start_utc_day` and `window_days`.

## 4. How far to trust it (honest limits)

Skill is measured on **places never seen** (zone-stratified 4°×4° blocks) in the **future** (from 2025-01-06), beside zone × month climatology and a raw-GFS baseline;
tables are in `docs/AGENTIC_ARCHITECTURE.md` §5.3. Beyond that:
single run, **one seed, no confidence intervals**; the extension-vs-base-feature comparison is **not row-paired**; rain truth (CHIRPS) is itself noisy against METAR;
heat/cold-wave labels approximate IMD criteria from METAR normals; `tmin_range` beats raw GFS by only about 2 %; 00Z cycle only; no nowcast from live observations.

## 5. Plugging into the WeatherGPT backend / an orchestrator LLM

1. **Advertise**: `service.catalogue()` returns one entry per model (what it answers, horizon, `validated_zones`, JSON-schema `parameters`) — built from the loaded registry, so a model that
   failed the gate is never offered.
2. **Execute**: `service.call_tool(name, {"lat":…, "lon":…, "horizon_days":…})` validates the call (unknown tool, extra / non-numeric arguments, horizon outside 1–10 raise `ToolCallError`) and runs it.
3. **Evidence**: `weathergpt_events.ceo_bridge.to_ceos(result)` converts only validated entries into `app.schemas.ceo.CanonicalEvidenceObject`s (probability stays probability; rain amounts are flagged
   `conditional_on_wet`; thresholds without skill ride along in `extra`).
4. **HTTP**: mount `create_router(service)`; a refusal is HTTP 200 with `available: false`, a malformed tool call is 422.

The planner / validator / executor inside `app/` is the backend team's part; the interfaces above are what it should call.

## 6. Reproduce the pipeline (Kaggle only — no Modal)

All compute runs as single-file Kaggle kernels produced by `event_models/bundle.py` (it inlines the `event_models` and `weathergpt_events` modules and bakes environment variables).

| stage | command / script | notes |
|---|---|---|
| truth, GFS base, GFS extension | `collect_truth.py`, `collect_gfs.py`, `collect_gfs_ext.py` (bundled kernels in `backup/event_models/`) | public AWS / IEM / CHIRPS sources, no keys |
| train | `python3 event_models/make_train_kernels.py --push` (all), `--only=train-rain-day,train-rain-window`, `--base` (USE_EXT=0 baselines) | 5 train kernels; max 5 concurrent CPU sessions; the push retries every 60 s while slots are full |
| results | `python3 backup/event_models/kaggle_log.py <owner>/weathergpt-<slug>` then `python3 backup/event_models/results_from_logs.py LOG… --base LOG…` | read logs through the API, never `kaggle kernels output` |
| live smoke test | `python3 event_models/make_train_kernels.py --serve --push` | needs internet; prints a PASS/FAIL line and `SERVE_SMOKE_BEGIN…END` JSON |
| publish to Hugging Face | `python3 event_models/make_train_kernels.py --publish --push` | creates / updates a **private** repo; token is read from the private Kaggle dataset `asanaai-conf-creds`, never printed or committed |

Memory: a Kaggle CPU kernel has 30 GB. The rain kernels need `ROW_STRIDE=4` and the per-file float32 loader (`dataset.read_thinned`); both were OOM-killed before that. Leakage rules
(forecast-run features only, 3-hour label windows, uncovered windows dropped, 5-day gaps between periods, spatial hold-out) live in `dataset.py` / `features_events.py` and are covered by tests.

## 7. Tests

```bash
python3 -m pytest -q tests -p no:cacheprovider          # whole repo (backend + ML), about 4 minutes
python3 -m pytest -q tests -k "event or events or rain or notebook"
```
Last full run (2026-10-07): **147 passed in 3 min 44 s** (backend + ML; 135 existing tests plus 12 for the real-world layer and notebook).
The first run of `tests/test_event_pipeline.py` builds a synthetic 10-day fixture (about 9 minutes on a laptop, cached under `/tmp/weathergpt_pipeline_fixture_v4`; bump `FIXTURE_VERSION` if the generators change).

| area | file | what it proves |
|---|---|---|
| labels, dataset joins, features, extension fields | `test_event_labels.py`, `test_event_dataset.py`, `test_event_features.py`, `test_event_ext.py` | METAR parsing, leakage rules, window labels, IMD labels, climatology reference, bundler env |
| trainer | `test_event_train.py`, `test_rain_products.py` | planted signals are found and calibrated, monotone rain curves, coherent rain products |
| serving | `test_events_registry.py`, `test_events_static.py` | admission gate (incl. the global-base-rate rule), coastal rule, serving reproduces the trainer's own metrics |
| end to end | `test_event_pipeline.py` | all 20 targets train on a synthetic fixture; **serving features equal training features row for row**; engine answers are bounded, monotone, honour the horizon and withhold unvalidated numbers; real engine output converts to valid evidence objects |
| real-world layer | `test_events_service.py` | loader on real trained artifacts, run cache, tool-call validation, CLI, HTTP router, CEO bridge |
| notebook | `test_notebook.py` | every notebook code cell executes offline against a canned engine |

**Not covered by unit tests:** the live network fetch (exercised once, successfully, by the Kaggle `serve-smoke` run on the real 2026-10-06 run, which found and fixed a real bug), real-data forecast
quality beyond the held-out metrics, and the notebook against the live Hugging Face repo.

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no fresh GFS/GEFS run is published…` | NOAA is late or down; the newest complete run is older than 54 h. Retry later — the service will not serve a stale run |
| `not over land (land fraction 0.xx)` | the 0.25° GFS cell is mostly sea (below the training stations' 5th percentile of land fraction) |
| `no model passed the admission gate` when loading | artifact folder is incomplete or its `metrics.json` lacks provenance |
| 401 / 404 from Hugging Face | the repo is private: pass `token=` or set `HF_TOKEN` |
| `eccodes` missing | `pip install eccodes` (the live fetch also tries this automatically) |
| Kaggle kernel `Killed` | out of memory: raise `ROW_STRIDE`, see §6 |

## 9. Repo map

`event_models/` collection + training + bundler + Kaggle kernel builders · `weathergpt_events/` inference (`registry` gate, `models`, `static`, `features`, `engine`, `live`, `tools`, `loader`, `service`, `api`, `ceo_bridge`, `__main__`) ·
`notebooks/` implementation notebook · `docs/` this guide + architecture · `backup/event_models/` exact kernel bundles that were pushed, logs helpers, `SESSION_HANDOFF_2026-10-06.md` (state and history).
