# WeatherGPT event models — session handoff (2026-10-05 → 2026-10-06)

Read this first. Everything here was verified from logs/tests unless marked **UNVERIFIED**.
Companion docs: `docs/AGENTIC_ARCHITECTURE.md` (design + rain edge-case table), `event_models/README.md` (pipeline map).

## 1. What the user wants (their words, condensed)

* An **agentic architecture**: an orchestrator LLM picks only the needed tools (live weather APIs + **small specialist
  models, one per weather target**) instead of querying everything. M1 (field mapper), M3 (intent), M5 (trust ranker)
  are **not needed** (kept, not deleted). "Train best-ever models", "rain yes/no, if yes how much", **ranges**,
  **whole India**, **"all sorts of features, no compromise, fully logical"**, forecast horizon "5 days not enough"
  (→ now 10 days). "Treat this as your own project." Wants speed ("do fast bro").
* Standing rules: **never download datasets/checkpoints locally** (3.7 GB RAM; logs only); HF token never committed;
  **nothing committed/pushed to GitHub this session** (user will decide); give honest, hedged numbers.

## 2. Hard constraints discovered

* **NO MODAL from now on (user decision, 2026-10-06; its workspace had also exceeded its spend limit)** → all compute is **Kaggle** (account of the stored CLI creds:
  **`anamitrasarkar007`**; the earlier demo notebook is under `arkosarkarhehe`).
* Kaggle: **max 5 concurrent CPU sessions** ("Maximum batch CPU session count of 5 reached"); pushing a GPU session
  failed with "Maximum batch GPU session count of 2 reached" (something else on the account may hold GPU sessions).
  **No cancel API in the CLI** — a running kernel cannot be stopped from here.
* Read kernel logs WITHOUT downloading outputs: `python3 backup/event_models/kaggle_log.py anamitrasarkar007/weathergpt-<slug> [tail_lines]`
  (never `kaggle kernels output`).
* Open-Meteo free API is billed by variables×days → bulk data came from **NOAA GFS/GEFS on AWS** (no limit, GRIB byte-range).

## 3. Data collected (Kaggle kernel outputs; slugs are `anamitrasarkar007/weathergpt-<slug>`)

| slug | content | state |
|---|---|---|
| `truth-collect` | METAR (IEM `IN__ASOS`): **142 stations ok, 7,914,643 reports, 4,570,272 station-hours, 264,764 station-days**; events: 105,100 thunderstorm / 95,702 fog / 29,813 strong-wind / 312,069 rain / 13,929 dust hours, 12,072 hot days. CHIRPS daily 0.05° whole India window **2016-01-01→2026-08-31, 3,896 days** | COMPLETE |
| `gfs-s1`,`s3`,`s4`,`s5`,`s2b` + `s2` (partial) | GFS base fields, 18 fields, 00Z runs, 32 steps (days 0–4), **2,012 run dates 2021-04-01→2026-10-03**. `s2` crashed on a corrupt NOAA file but saved May–Oct 2022; `s2b` (2022-11-01→2023-05-31) fills the gap | all COMPLETE |
| `gfs-l1`, `gfs-l2` | base fields for **days 5–9** (leads 126–240 h, 6-hourly), every 2nd run date (ordinal-parity 0): 433 + 573 = **1,006 dates**, 0 bad messages | COMPLETE |
| `ext-e1..e4` | **extension predictors**: GFS upper-air dynamics + surface energy + neighbourhoods + terrain, **GEFS mean/spread**, all **52 steps (days 0–9)**, every 2nd run date. E1 2021-04-01→2022-08-16, E2 2022-08-17→2024-01-01, E3 2024-01-02→2025-05-18, E4 2025-05-19→2026-10-03 | **RUNNING** (≈4–5 h each, started ≈23:07–00:30 on 2026-10-05/06) |

Counts: GFS station-step rows 136×32×2,012 = **8,756,224**; point-day rows 2,709×5×2,012 = **27,252,540** (+ days 5–9 and ext
for every 2nd run). Points = 2,573 land nodes (0.5°, box 6–38°N × 67–98°E) + 136 stations. Old D1 corpus was 9.58M rows
(127 places, 1 year) — not comparable row-for-row (see architecture doc). Exact post-join training counts are printed by
`build_tables` (`[tables] ...` lines) — **not known yet**.

Probe/smoke kernels (logs only, keep for provenance): `source-probe`, `bulk-probe`, `gfs-aws-probe`, `gefs-probe`,
`gfs-smoke`, `ext-smoke`, `inspect-ext`, `validate-smoke`.

## 4. Verified findings (from logs)

* ERA5 has **no thunderstorm/fog/visibility** (CAPE, visibility 100% null via Open-Meteo archive) → METAR is the only honest truth.
* METAR `p01i` reads 100% valid but is **zero-filled** (Indian METARs don't report it) → not used.
* CHIRPS has **no nodata flag; ocean = −9999** → masked explicitly.
* GFS archive on AWS: 2021-01-01→now; `atmos/` path from GFS v16 (2021-03); 54 MB/s, 25 ms/message decode. GEFS mean/spread
  (`pgrb2sp25`, `geavg`/`gespr`) available from 2017. **NOAA corrupt data: 2022-11-30 00Z, 9 steps unreadable** (27,33,36,45,54,63,66,84,102 h).
* GFS vs METAR at airports (2-date smoke): temp bias +0.9 °C d1–2 (RMSE ~3), RH bias −11…−12 %, wind bias +0.9…+1.5 m/s;
  raw CAPE AUC for thunderstorm only 0.60 (lots of headroom), rain AUC 0.74 (apcp), fog 0.74, gust 0.74.
  GFS cell rain vs CHIRPS corr 0.48 (day 0) → 0.25 (day 4); AUC for ≥2.5 mm 0.80 → 0.72.
* **Label noise (CHIRPS vs METAR rain at 185,664 station-days):** CHIRPS cell-mean ≥1 mm on 69.5 % of airport-rain days and on
  15.1 % of airport-dry days; AUC 0.81 overall, **0.71 in monsoon**. Probabilities are calibrated to CHIRPS, not a gauge.
* Extension smoke (3 + 2 dates incl. leads to 240 h): **no field absent from either archive**, no bad messages; 95 step cols +
  102 day cols, no all-NaN, ranges physical; ~48–63 s/date. Found+fixed `soilw` sentinel 9999.

## 5. Bugs found and fixed this session (each would have corrupted results or wasted a run)

`vis_km` forecast/observed column collision (merge `_x/_y`, fog model would have lost its best feature) → `obs_` prefix + hard guard ·
truncated GRIB ranges crashed a shard → framing check + 4 retries + skip · eccodes not installed on Kaggle · `soilw` 9999 sentinel
blended into coastal points → per-field validity masks + **NaN-aware bilinear sampler** · station normals counted days not years ·
bundler `--env=K=V` form baked a bogus var (every kernel would have trained ALL targets) → strict parser + header test · window
table crashed with only 5 forecast days → clear skip · gate admitted a pure-noise model (BSS>0 by chance) → `MIN_BSS 0.005`, `MIN_AUC 0.55`,
1 % MAE gain · step anomalies keyed by station would have mismatched serving → **one nearest-node climatology** · double derivation
of ext fields per date (wasted CPU) · 3-hourly METAR fixture made every label window uncovered (test artifact).

## 5a. FIRST REAL-DATA RESULTS (kernel `base-events`, base features only, USE_EXT=0, OLD hash-only hold-out)

Step table (METAR-labelled station-steps): **4,352,178 rows** — train 1,714,858 · val 553,173 · test_time 1,348,831 · test_space 285,493
(449,823 dropped in the gaps between periods); p1 rows before labelling 10,690,552. Skill is vs **zone × month climatology** (the
honest reference); `test_space` = places the model never saw, future dates (from 2025-01-06):

| target | test_time AUC | test_time BSS | test_space AUC | test_space BSS |
|---|---|---|---|---|
| thunderstorm | 0.879 | +0.099 | 0.819 | +0.066 |
| fog | 0.914 | +0.165 | 0.859 | +0.069 |
| strong_wind | 0.899 | +0.087 | 0.831 | +0.051 |
| rain_3h | 0.879 | +0.180 | 0.823 | +0.091 |

(All positive over local climatology on unseen places; the raw single-feature rule AUCs, per-zone/lead breakdowns and reliability are
in each `metrics.json` inside the kernel output; the log only shows the headline.)  **These numbers used the FLAWED split below —
re-run `base-events` / `base-day` (bundles already rebuilt) before quoting them as the baseline for the extension ablation.**

**Split flaw found from this run and FIXED:** hash-only 4°×4° block hold-out (460 of 2,709 points held out) left **three whole zones
with ZERO held-out points** (north-west arid 227 pts, west coast 118, south interior 76) — their skill could never be verified, so the
serving gate would have withheld every forecast there. Now `holdout_blocks(..., zone=...)` is **zone-stratified** (≥1 block per
multi-block zone, one global md5 block rank so overlapping zones pick the same blocks; ≈28 % of points held out on a realistic layout,
every zone covered). Unit test added. The tiny 12-point test fixture bypasses stratification on purpose (documented in the test).

## 5b. Bugs found by the parity/engine tests at the very end (fixed, regression tests added)

* climatology artifact lost its `bin` index-level name after smoothing → saved column was `level_1`, serving could not load it
  (training worked because it joins positionally). Fixed in `fit_climatology`; test `test_saved_climatology_artifact_has_named_key_columns_for_serving`.
* conformal margin can be **negative** (over-covering validation) → `lo` above `q10`, possibly inverted / excluding its own
  median. New shared `train.calibrated_interval` (clamped around the median) used by BOTH the evaluation and the served interval.
* live-run freshness rule was muddled → pure `live.run_is_fresh` (≤ 54 h, never "from the future") with a test.
* docs/card: HF repo is **public** (docs said private) and the package is not on PyPI (card said `pip install`); fixed in repo
  docs and `modal_jobs/export_models.py` — **the already-published HF card still has the old text** (re-publish WITHOUT Modal: a Kaggle kernel with the HF token stored as a Kaggle secret, or upload the README by hand).

## 6. Code map (all uncommitted)

`event_models/` — training/collection: `labels.py` (METAR → labels), `collect_truth.py`, `collect_gfs.py` (GFS fetch; env
`GFS_LEADSET=short|long|all`, `SHARD_START/END/STRIDE/OFFSET`), `collect_gfs_ext.py`, `dataset.py` (joins, CHIRPS truth, windows,
IMD heat/cold-wave labels, station normals), `features_events.py` (zones, splits, climatology, anomalies, time/solar features),
`train.py` (binary / xent / quantile / **curve** trainers + all metrics), `rain_products.py`, `run_training.py` (18 targets,
`build_tables`, provenance), `bundle.py` (inlines modules into one Kaggle script), `make_train_kernels.py`, `validate_smoke.py`, `inspect_ext.py`.
`weathergpt_events/` — inference: `registry.py` (admission gate + per-zone/lead skill), `models.py`, `static.py`, `features.py`
(same functions as training), `engine.py` (no numbers without validated skill), `live.py` (RunStore, static grids builder), `tools.py`.
Kernel bundles + metadata live under `backup/event_models/<slug>/`. `docs/AGENTIC_ARCHITECTURE.md` updated.

Tests: **81 passed** (`python3 -m pytest tests -q -k "event or events or rain"`, ≈ 3 min with the cached fixture): `tests/test_event_{labels,dataset,features,ext,train}.py`,
`test_events_{static,registry}.py`, `test_rain_products.py`, and `test_event_pipeline.py` (builds a synthetic 10-day fixture,
cached at `/tmp/weathergpt_pipeline_fixture_v4`, first run ≈ 9 min; bump `FIXTURE_VERSION` if the generators change). That file
covers: tables with/without extension, every one of the 20 targets training end to end, short-horizon skip, **train/serve
parity** (serving features == training features, row for row, for step / IST-day / rain-day / 3-day / 7-day tables), the engine
(structure, monotone curves, horizon, no numbers without validated skill, refusals) and the tool catalogue.
NOT covered by any test: the live network fetch in `weathergpt_events/live.py` and real-data skill (needs Kaggle).

## 7. Models (20 targets; all trained on real data on 2026-10-06, see the latest update at the bottom)

step events: thunderstorm, fog, strong_wind, rain_3h, dust · step ranges: temperature_range, wind_range, humidity_range ·
station-day: hot_day, cold_night, heatwave_imd, coldwave_imd, tmax_range, tmin_range · rain: **rain_curve** (16-threshold
exceedance curve, monotone in threshold; "yes/no + how much" derived from it), rain_3day_any_2p5mm, rain_3day_any_15mm,
rain_3day_total, rain_7day_any_2p5mm, rain_7day_total. Evaluation, edge cases and gating are documented in the architecture doc.

Training kernels: **`base-events` (USE_EXT=0; thunderstorm, fog, strong_wind, rain_3h) is RUNNING** — first real numbers.
`base-day` was NOT pushed (GPU-session limit; CPU slots full). Bundles for `train-events`, `train-ranges`, `train-day`,
`train-rain-day`, `train-rain-window` are built in `backup/event_models/train-*/` but **not pushed** (they need the ext shards).

## 8. Resume checklist (in order)

1. `for k in ext-e1 ext-e2 ext-e3 ext-e4 base-events; do kaggle kernels status anamitrasarkar007/weathergpt-$k; done`
2. `base-events` is DONE (results in §5a). Re-run it and `base-day` with the fixed split: `python3 event_models/make_train_kernels.py --base --push` (CPU slots permitting; GPU push failed with "max GPU session count 2").
3. When all `ext-e*` COMPLETE: read each `EXT_REPORT` (look at `field_absence`, `bad_messages`, `absent_runs`); re-run any failed range.
4. `python3 event_models/make_train_kernels.py --push` (5 train kernels; retries until slots free). Bundles in `backup/event_models/train-*/` and `base-*/` were rebuilt AFTER the hold-out fix (they are regenerated on every call anyway).
5. Local: `python3 -m pytest tests -q -k "event or events or rain" --ignore=tests/test_event_pipeline.py` (≈90 s), then `tests/test_event_pipeline.py` (parity + engine). **Do not run these while Kaggle polling scripts hog the laptop.**
6. After training: paste verbatim metrics into `docs/AGENTIC_ARCHITECTURE.md` §5; ablation = same targets with `USE_EXT=0` vs ext; build static grids kernel (`live.build_and_save_static`); serving smoke kernel against the latest live run; publish models (no Modal: the HF token must be a Kaggle secret, or upload by hand).
7. Not built: planner/validator/executor in `app/` (teammate's area; design in the architecture doc), nowcast with live observations, 12Z cycle, odd run dates.
8. Git: committed locally on `main` on 2026-10-06 (3 commits, **not pushed** — the user decides when to push). Never commit tokens.

## 9. Risks / unknowns

Extension shards may take longer than 5 h (47–63 s/date measured; 4 shards × ~250 dates × 52 leads) · LightGBM time on 3M×~230 features is an
estimate (~45 min/model/quantile; ranges train 3 quantiles each) · rain tables are huge: `ROW_STRIDE=2` thins them for memory (30 GB) ·
test period for rain ends 2026-08-31 (CHIRPS lag) · live fetch in `live.py` is untested on the network (no local downloads allowed; test on Kaggle) ·
IEM/NOAA URLs were stable during this session but are third-party.

## Update 2026-10-06 ~19:20 IST (after compaction)
- ext-e1..e4 COMPLETE, EXT_REPORT clean (field_absence {}, bad_messages {}).
- Pushed 5 train kernels at 19:15 IST (account anamitrasarkar007, CPU): train-events, train-ranges, train-day, train-rain-day, train-rain-window — all RUNNING.
- `make_train_kernels.py --push` (background, started 19:15) is retrying base-events / base-day every 60 s until a CPU slot frees (limit 5). If that process died, re-run `python3 event_models/make_train_kernels.py --base --push`.
- Next: read logs via `python3 backup/event_models/kaggle_log.py anamitrasarkar007/weathergpt-<slug>`; read metrics.json; base-vs-ext ablation on the fixed split; static grids + serving smoke kernel; HF publish without Modal. Not pushed to GitHub. No Modal.
- A second kaggle.json may exist for GPU if ever needed (user said so); not needed so far.

## Update 2026-10-06 ~19:50 IST (second compaction)
- COMPLETE: train-events, train-day, base-events (fixed split), base-day. RUNNING: train-ranges. train-rain-day and train-rain-window were **OOM-killed** ("Killed" right after the `[tables] points` line, before `[tables] p2 rows`): the full-width p2 + p2x tables were read, merged and only then thinned. Fixed with `dataset.read_thinned` / `dataset.run_dates` (thin to the same kept run dates and store float32 per file; synthetic check = identical keys, max diff 9e-8; 14 unit tests pass). Both re-pushed 19:40 via `python3 event_models/make_train_kernels.py --only=train-rain-day,train-rain-window --push` (new `--only=` flag).
- Results are printed between `TRAIN_SUMMARY_BEGIN`/`TRAIN_SUMMARY_END` in each kernel log (JSON, per target -> metrics -> val/test_time/test_space).
- test_space AUC / BSS vs zone-month climatology, ext vs base (fixed split): thunderstorm .849/.075 vs .842/.072; fog .855/.084 vs .877/.112 (**ext worse**); strong_wind .890/.078 vs .885/.071; rain_3h .858/.182 vs .852/.169; hot_day .965/.339 vs .963/.334; heatwave_imd .865/.060 vs .846/-.021 (ext clearly better). Others (ext only): dust .888/.094; cold_night .975/.260; coldwave_imd AUC .779 but BSS .667 -> almost surely a tiny-positive-count artefact, do NOT quote.
- **Caveat: base vs ext are not row-paired** (ext does an inner merge, so n differs, e.g. tmax_range 59.7k vs 86.4k test_space rows). Treat differences of a few thousandths as noise; a paired ablation needs base re-scored on the ext row set.
- tmax_range test_space: median MAE 1.85 vs raw GFS 2.43; conformal coverage .739 (band .70-.90). tmin_range: MAE 2.00 vs raw GFS 2.04 (barely better), coverage .779.

## Update 2026-10-07 ~01:00 IST (third block; supersedes the older "pending" lines above)
- **There are 20 targets, not 18** (5 step events, 3 step ranges, 6 station-day, rain_curve, 5 rain windows) -- the "18" in older text was a miscount.
- ALL 5 train kernels COMPLETE and base-events/base-day (fixed split) COMPLETE. Both rain kernels needed memory fixes: `read_thinned` (per-file thinning + float32), truth columns trimmed per product, `point_day_rain_table(inplace=True)`, `ROW_STRIDE=4` for those two kernels (`GROUP_ENV` in `make_train_kernels.py`). Final tables: rain 6.37 M rows, rain3 5.03 M, rain7 2.43 M.
- Verbatim results are in `docs/AGENTIC_ARCHITECTURE.md` section 5.3, generated by `backup/event_models/results_from_logs.py LOG... --base LOG...` (logs fetched with `kaggle_log.py`).
- **Gate fix (real finding):** `coldwave_imd` had BSS +0.667 vs zone x month but -0.461 vs the global base rate; the gate now also requires BSS vs global >= MIN_BSS. Applying the gate to the real summaries: 19 of 20 served, `coldwave_imd` refused. New `registry.unskilled_thresholds` + engine key `thresholds_without_skill_mm` flag rain-curve tail thresholds (>= ~90 mm daily) with no held-out skill.
- New: `event_models/serve_smoke.py` (static grids + live GFS/GEFS run + gate + engine on 12 places incl. sea/out-of-domain refusals; `--serve` flag of `make_train_kernels.py`), `event_models/publish_hf.py` (private HF repo `<user>/weathergpt-events`, card generated from metrics; token from the private Kaggle dataset `asanaai-conf-creds`, never printed), bundler now inlines `weathergpt_events` too. 12/12 pipeline tests + registry/static tests pass locally.
- `serve-smoke` pushed 00:51 IST; next: read `SERVE_SMOKE_BEGIN..END` from its log (`kaggle_log.py anamitrasarkar007/weathergpt-serve-smoke`), fix anything it finds, then build/push `publish-hf` (sources: `serve-smoke` + dataset `asanaai-conf-creds`; needs `enable_internet`).
- Nothing committed since the 3 original commits (user did not ask); nothing pushed.

## Update 2026-10-07 ~01:30 IST (final block for this session)
- **serve-smoke PASS (0 problems)** on the live 2026-10-06 00Z GFS+GEFS run: static grids built from GFS orography/land mask, run fetched in ~100 s (52+52 steps), 20 artifacts through the gate (19 served, `coldwave_imd` refused), engine ~12 s per place, 9 inland + 5 coastal cities answered, open sea / London refused. Real bug found by it and fixed: `live.fetch_static_arrays` ran `idx_task` outside the worker pool (HTTP client is None there). Plausibility samples (log lines `[serve-sample]`): Delhi 6 Oct tmax day-1 q10/q50/q90 = 33.5/35.5/35.7 C, P(any rain) 0.026. Not verified against observations; Cherrapunji tmax (~29.6 C) looks high for a ~1,300 m town -- probably GFS cell elevation vs the actual site (unchecked).
- **Coastal rule is data-driven now**: 13 of 136 training stations have land fraction < 0.5 (5th percentile 0.27), so `FeatureBuilder.min_land = min(0.5, 5th pct of station land_frac)`; results carry `location.land_fraction` and `coastal`. Chennai (0.29) is served; Puri (0.24) is refused.
- **Published (PRIVATE) to https://huggingface.co/Arko007/weathergpt-events** by Kaggle kernel `publish-hf` (token read from the private dataset `asanaai-conf-creds`, never printed; 111 files: 19 served model dirs, `models_not_served/coldwave_imd/metrics.json`, `static_grids.npz`, `points.parquet`, `climatology_day.parquet`, `live_smoke_report.json`, `src/` inference + shared feature code, generated `README.md` card). Make it public from the HF settings page when ready. The old public `Arko007/weathergpt-models` card is untouched (still says `pip install weathergpt-models`).
- Tests: 12/12 `tests/test_event_pipeline.py` (parity + engine) pass with all of today's changes; registry/static/features tests pass. NOT covered by tests: the live network fetch (covered only by the Kaggle smoke run).
- **Uncommitted** (the user has not asked for a commit): `event_models/{dataset,run_training,make_train_kernels,bundle,serve_smoke,publish_hf}.py`, `weathergpt_events/{registry,engine,features,live}.py`, tests, `docs/AGENTIC_ARCHITECTURE.md`, `backup/event_models/*` (bundles + `results_from_logs.py` + this file). Nothing pushed to GitHub.
- Still open / v1.1: nowcast with live observations, 12Z cycle, odd run dates, paired (row-matched) base-vs-extension ablation, per-model seeds/CIs, the `app/` planner/validator/executor (teammate's area), `tmin_range` barely beats raw GFS (2 %), fog may be better served by base features.
