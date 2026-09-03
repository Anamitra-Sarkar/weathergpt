# WeatherGPT — session handoff, 2026-09-03

Written mid-flight so work survives a terminal close. Everything below is
verified, not planned, unless marked TODO.

Approved plan: `~/.claude/plans/https-github-com-anamitra-sarkar-weather-bubbly-sphinx.md`

---

## 1. Why this rebuild exists — audit findings (all verified on disk)

The FastAPI evidence pipeline (`app/schemas/ceo.py`, `app/services/retrieval.py`,
`app/rade/v2.py`) is genuinely good. **The ML layer did not exist.**

| Doc claim | Reality found |
|---|---|
| M1 semantic classifier f1 ~0.90 | No weights. `metrics.json` = `{"note": "dry-run, no model trained"}`. Corpus: 1500 rows / **231 unique**; 64 rows labelled `temperature_*` but suffixed `(mm)`. |
| M2 bias correction, 14 400 real pairs, `rmse_t 1.32` | `best.pt` (37 852 B) was fitted to `np.random.randn(500,5)` for 1 epoch by `train_bias_correction.py --dry-run`, which **overwrites the real artifact**. No `scaler.pkl`. In `matched_pairs.csv`, `obs_t2m_c ≡ gfs_t2m_k − 273.15` to float precision and `obs_apcp_mm − gfs_apcp_mm == 0.0` for all 4800 rows → **the regression target is identically zero**. `lead_hours` is a 0–239 row index. |
| M3 intent parser, 2000 Groq rows | No weights. 1200 rows, **85.75 % one class** (majority baseline 0.857 beats the documented 0.45), 196 dupes, 0 Groq rows, `decision` was `random.choice()` for ~2/3 of rows. |
| ML is part of the system | **Nothing under `app/` imports torch/transformers.** `MODEL_DIR` is read by no code. |
| Evaluation | `scripts/backtest.py` is a 41-line skeleton, ignores `valid_from`/`valid_to`, no baseline, never run. |

Also: no frontend, CORS off by default, `groq_client.py` imported by nothing, the
user-facing "answer" is `_synthesize()` string concatenation, `app/api/v1/main.py`
is dead divergent code, two RADE engines can disagree in one response.

---

## 2. Live-verified data facts (probed this session, keep these)

- `historical-forecast-api` + `&models=gfs_seamless` vs `archive-api/era5` → **real** bias:
  mean **+2.34 °C**, sd 2.14, max 9.7. The shipped degenerate file lacked `&models=`.
- **`<var>_previous_day1..7` works** on historical-forecast and returns per-model
  suffixed columns → **genuine forecast lead ages**, killing the `i % 72` fabrication.
- **Requests must be small**: 4 models × 4 vars × 8 lead-ages × 92 days = 128 columns
  → Open-Meteo answers **502**. One model × 46 days (32 columns) → 200, 181 KB, 2.3 s.
- **`era5_land` returns 100 % nulls for precipitation and wind** through this API.
  Use **`era5_seamless`** (ERA5-Land temperature + ERA5 precipitation/wind). This cost
  one full D1 and D2 rebuild.
- **The ensemble API serves real members only for ~the last 4 days**, despite accepting
  `past_days=93`. Members start ≈ 2026-08-30; ERA5 truth ends ≈ 2026-08-28 → **zero
  overlap**, so a verified 31-member GFS training corpus is impossible from this API.
  → M4 must train on D1's **multi-model** ensemble (4 models = 4 members) and transfer
  to live GFS members through summary statistics. Record as a model-card assumption.
- SACHET/NDMA CAP feed is live: 99 real IMD/CWC alerts, real CAP 1.2 XML, **real polygons**.
- NOAA GFS GRIB2 `.idx` byte-range works (`noaa-gfs-bdp-pds.s3` and NOMADS both 200).
- IMD's own APIs return **401 "your IP needs to be whitelisted"** — genuinely unavailable.
- Groq models `qwen/qwen3.8-27b`, `qwen/qwen3.6-27b`, `openai/gpt-oss-20b/120b` all exist.
- Modal workspace `anamitrasarslsn10ab`; secret `groq-api-key` already present.

---

## 3. What is DONE and committed

### `app/services/field_taxonomy.py` (new, 46 canonical variables)
Single source of truth for native-field semantics, imported by the Modal builders
*and* the serving path. Labels are derived from a source table's own unit / level /
statistical-processing metadata, never from the abbreviation, so `TMAX (mm)` is
**structurally impossible** — `classify_native_field` abstains on a unit contradiction.
Abstention is a first-class return value.

### `modal_jobs/` — the data foundry (runs remotely; nothing downloads locally)
- `common.py` — app, 3 images, volumes `weathergpt-data` / `weathergpt-models` / `weathergpt-hfcache`.
  All images carry `.add_local_python_source("modal_jobs", "app")` (required, else `ModuleNotFoundError`).
- `locations.py` — 128 real Indian place names spanning every state and climate zone.
- `contracts.py` — dataset contracts that abort a build. Includes the one that would
  have caught the whole disaster: `max |forecast − truth| > 1e-3`.
- `build_corpora.py` — D1 (multi-model MOS) + D2 (ensemble) + location resolution.
- `build_fields.py` — D3, harvested from authoritative tables.
- `build_queries.py` — D4, multilingual queries with exact slot spans.
- `features_d1.py` — shared feature + split definition for M2/M4/M5 (defined once).
- `train_field_mapper.py` — M1. **Trained, done.**
- `train_calibration.py` — M4. Needs rework onto D1 (see §5).
- `train_mos.py` — M2. Written, not yet run (waits on D1).

### Artifacts on the `weathergpt-data` volume
- `locations.json` — **127 locations resolved**, 35 admin1 regions, elevation 2–3502 m,
  1 failure (Cherrapunji has no IN geocoder hit).
- `d3_fields.parquet` + `.stats.json` — **15 243 rows, 4 564 labelled, 44 classes**, from
  7 source tables: CF standard names 5071, NCEP GFS idx 3711, eccodes GRIB2 3191,
  BUFR 1766, WRF Registry 1429, IMD products 34, Open-Meteo 29, live SACHET CAP 12.
  Split **by source table**: train = CF + GRIB2 + CAP, test = WRF/NCEP/BUFR/OpenMeteo/IMD.
- `d2_ensemble/` — 268 224 rows, 0 failures, **but the member columns are all NaN**
  (see §2). Not usable as a training corpus. Keep the fetch code for live inference.

### M1 — RESULT (real, on schemas never trained on)

| metric | M1 | dict-registry baseline | majority |
|---|---|---|---|
| zero-shot macro-F1 | **0.729** | 0.160 | 0.017 |
| zero-shot accuracy | **0.767** | 0.593 | 0.550 |
| statistic accuracy | **0.976** | — | — |
| evidence-class accuracy | **0.996** | — | — |
| **misassignment rate** | **0.0032** | — | — |
| hallucination rate | 0.027 | — | — |
| val macro-F1 / accuracy | 0.841 / 0.964 | 0.034 | — |

Abstention threshold 0.5564 (calibrated on val). Per-source zero-shot accuracy:
IMD 0.853, BUFR 0.933, WRF 0.871, Open-Meteo 0.759, NCEP idx 0.648.
Weak spots: `mapped_accuracy` 0.543 and level accuracy 0.674 zero-shot →
TODO: rerun with `intfloat/multilingual-e5-large` and 12 epochs.

---

## 4. Modal jobs RUNNING right now (detached — survive the terminal closing)

Check with `modal app list | grep weathergpt`, logs at modal.com.

- **D1 rebuild** — `modal run --detach modal_jobs/build_corpora.py --what d1 --n-shards 8`
  127 locations × 4 models × 8 lead-ages × 12 months → `/data/d1_mos/*.parquet`.
  Was 15/127 with 0 misses at checkpoint time. Expect ~1–2 h.
- **D4 build** — `modal run --detach modal_jobs/build_queries.py`
  676 base rows × 13 languages via Groq → `/data/d4_queries.parquet`.

`max_containers` is capped at 8 (6 for D4) so other agents' Modal work is not crowded.
**Never touched any non-weathergpt app.** A `modal_eval_all.py` app from another
project was seen running and deliberately left alone.

---

## 5. NEXT STEPS, in order

1. **Verify D1 finished**: `modal app list | grep weathergpt`; then run M2:
   `modal run --detach modal_jobs/train_mos.py --epochs 30`.
   It runs the contracts first and will refuse to train on a degenerate corpus.
2. **Rework M4 onto D1** (`train_calibration.py`): replace the D2 loader with
   `features_d1.build_features`; the 4 models are the ensemble members. Keep the CSGD
   + closed-form-CRPS head, the isotonic exceedance calibration and the full
   verification suite (Brier per threshold, reliability, rank histogram, PIT).
   Document the multi-model → GFS-member transfer assumption in the model card.
3. **M3** `train_intent.py` — JointBERT on `google/muril-base-cased`: intent head +
   BIO slot tagging + multi-label variable head. Report per-language breakdown and
   generalisation to held-out districts + held-out template families
   (`HELD_OUT_FAMILIES = {"sow", "heat", "storm"}`).
4. **M5** `train_trust_ranker.py` — LightGBM `lambdarank` over (variable, lead,
   location, season) groups, relevance = rank by `|forecast − truth|` from D1.
   Compare NDCG@1/@3 against the hand-tuned weights in `app/services/ranker.py:8-50`.
5. **Phase 3 — serving** (`app/ml/`): `registry.py` with the metrics-contract gate
   (refuse an artifact lacking `dataset_sha256`/`split`/baselines), inference clients,
   every ML output emitted as a **derived CEO** with `parent_ids` +
   `transformation` + `algorithm_version`. Primary backend a Modal GPU endpoint so
   this 3.7 GB-RAM machine never hosts weights.
6. **Phase 4 — real sources**: SACHET CAP adapter with point-in-polygon matching and
   Update/Cancel lifecycle; GFS GRIB2 via `.idx` byte-range decoded on Modal;
   Open-Meteo geocoding fallback in `location_resolver` (the 8-city gazetteer 404s
   almost every live query); wire `groq_client` into `run_explanation_agent`.
7. **Phase 4 — delete the duplicates** (user asked explicitly): `app/api/v1/`,
   `services/fusion.py`, `services/disagreement.py`, `schemas/wio_v2.py`,
   `decoders/grib2_placeholder.py`, `rade/{enumerator,policy,utility}.py`,
   `kaggle_kernel*/` → `legacy/`, and the committed
   `weathergpt-2026-09-01.{zip,tar.gz}`.
8. **Phase 5** — `eval/` harness honouring `valid_from`/`valid_to`, demo UI at
   `app/static/index.html` (no npm), new tests, docs regenerated from measured results.

## 6. Traps already paid for — do not repeat

- `train_bias_correction.py --dry-run` **overwrites the real checkpoint**. Fix before running.
- Modal cannot parse `list[str] | None` or `list` annotations on a remote function parameter.
- Modal images need `.add_local_python_source(...)` or the `modal_jobs` package will not import.
- Open-Meteo 502s on large multi-model requests; keep them to one model / ~46 days.
- `era5_land` has no precipitation; `era5_seamless` does.
- `pkill -f "<pattern>"` matches this agent's own shell. Use `pgrep`+`kill` on an exact path.
