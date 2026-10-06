# WeatherGPT event models — handoff (state as of 2026-10-07, after the final cleanup)

Read this first. Everything here was verified from logs / tests unless marked **UNVERIFIED**. The old layered "update" blocks were folded into this file; nothing below is stale.
Companion docs: `docs/EVENT_MODELS_GUIDE.md` (use, reproduce, test), `docs/AGENTIC_ARCHITECTURE.md` (design, rain edge cases, **verbatim results in §5.3**), `event_models/README.md`, `weathergpt_events/README.md`,
`notebooks/weathergpt_events_implementation.ipynb`.

## 0. State at a glance

* **20 targets trained on real data (Kaggle CPU), 19 pass the admission gate, `coldwave_imd` is refused** (BSS +0.667 vs zone × month but −0.461 vs the global base rate; the gate now also requires skill over the global rate).
* **Live end-to-end smoke test PASSED** on the real 2026-10-06 00Z GFS+GEFS run (kernel `serve-smoke`): 13 cities answered (9 original incl. Chennai/Mumbai/Kolkata + Kochi, Visakhapatnam, Panaji, Mangaluru), Puri (24 % land), open sea and London refused, 0 structural problems, ~85 s run download, ~12 s per place.
* **Published, PRIVATE:** https://huggingface.co/Arko007/weathergpt-events (kernel `publish-hf`, 111 files: 19 served model folders, `models_not_served/coldwave_imd`, `static_grids.npz`, `points.parquet`, `climatology_day.parquet`, `live_smoke_report.json`, `src/`, generated card). Make it public from the HF settings page when ready.
  **That publish predates** `loader/service/api/ceo_bridge/__main__` and the new card usage text → re-run `publish-hf` once (needs the Kaggle account) to refresh `src/` and the card.
* **Code is committed and pushed** up to commit `90b4ed3` (GitHub `Anamitra-Sarkar/weathergpt` `main`, commits re-authored as `anamitrasarslsn10ab@gmail.com` at the user's request; the older commit `bcd1ee8` on origin keeps `anamitrasarkar13@gmail.com`).
  The cleanup commit on top of it ("Real-world layer, implementation notebook, guide and refreshed docs…": loader/service/CLI/API/CEO bridge, notebook, guide, tests, this file) was **local and NOT pushed when this was written** — check `git log origin/main..main`. The notebook clones its code from GitHub, so it only works for outsiders after that push. Never push unasked.
* **Kaggle account `anamitrasarkar007` is idle** (no running kernels). The user's other agent may switch accounts — it must use its own `KAGGLE_CONFIG_DIR`, not overwrite `~/.kaggle/kaggle.json`. **Do not use Kaggle until the user says that agent is done.**

## 1. What the user wants (their words, condensed)

* An **agentic architecture**: an orchestrator LLM picks only the needed tools (live weather APIs + **small specialist models, one per weather target**). M1 (field mapper), M3 (intent), M5 (trust ranker) are not needed (kept). "Train best-ever models", "rain yes/no, if yes how much", **ranges**, **whole India**, **10-day horizon**, "all sorts of features, no compromise", "treat this as your own project", **speed**, **no mistakes**, honest hedged numbers.
* This is for the **SIH 2026 hackathon**: needs real, usable code, an implementation notebook, a guide, documentation and tests (all now in the repo).
* Standing rules: **never download datasets/checkpoints/JSON locally** (3.7 GB RAM; read logs only); **no Modal ever** (user decision 2026-10-06); HF token never committed or printed; **never commit/push unasked**; commit trailers = `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01XGNoEJFLtggPrupUB7aLwM`; keep the private Kaggle dataset `anamitrasarkar007/asanaai-conf-creds` (holds `hf_token`).
* The planner / validator / executor inside `app/` belongs to the teammate; this repo provides the model-side interfaces (`tools.catalogue`, `ForecastService.call_tool`, `ceo_bridge.to_ceos`, `api.create_router`).

## 2. Hard constraints discovered

* Kaggle: **max 5 concurrent CPU sessions**; a GPU push failed ("max batch GPU session count of 2") — none needed (tabular LightGBM); **no kernel cancel in the CLI**; CPU kernels have **30 GB RAM** and are silently `Killed` beyond it.
* Read logs only through the API: `python3 backup/event_models/kaggle_log.py anamitrasarkar007/weathergpt-<slug>` (never `kaggle kernels output`). Results tables: `backup/event_models/results_from_logs.py`.
* NOAA GFS/GEFS on AWS via `.idx` byte-range + eccodes is the bulk source (Open-Meteo is billed per variable-day and its variables differ from what is trained on).

## 3. Data collected (Kaggle kernel outputs; slugs are `anamitrasarkar007/weathergpt-<slug>`; all COMPLETE)

| slug | content |
|---|---|
| `truth-collect` | METAR (IEM `IN__ASOS`): 142 stations, 7,914,643 reports, 4,570,272 station-hours, 264,764 station-days; events 105,100 thunderstorm / 95,702 fog / 29,813 strong-wind / 312,069 rain / 13,929 dust hours. CHIRPS daily 0.05° 2016-01-01→2026-08-31 (3,896 days) |
| `gfs-s1,s2,s2b,s3,s4,s5` | GFS base fields, 18 fields, 00Z runs, 2,012 run dates 2021-04-01→2026-10-03, days 0–4 (`s2` partial after a corrupt NOAA file, `s2b` fills the gap) |
| `gfs-l1`, `gfs-l2` | base fields for days 5–9, every 2nd run date (1,006 dates) |
| `ext-e1..e4` | extension predictors (upper-air dynamics, surface energy, neighbourhoods, terrain, GEFS mean/spread), 52 steps (days 0–9), every 2nd run date, clean reports |
| `train-events`, `train-ranges`, `train-day`, `train-rain-day`, `train-rain-window` | the 20 trained targets (+ `climatology_day.parquet`) |
| `base-events`, `base-day` | USE_EXT=0 baselines on the fixed split |
| `serve-smoke`, `publish-hf` | live smoke test (writes `deploy/`), HF publisher |

Points = 2,573 land nodes (0.5°, box 6–38°N × 67–98°E) + 136 stations. Probe / smoke kernels kept for provenance: `source-probe`, `bulk-probe`, `gfs-aws-probe`, `gefs-probe`, `gfs-smoke`, `ext-smoke`, `inspect-ext`, `validate-smoke`.
Final training tables (extension features, from the kernel logs): step events 2,727,007 rows (train 919,150 · val 295,618 · test_time 733,145 · test_space 276,025 · 503,069 dropped in the gaps), station-day 675,397, rain-day 6,366,257, rain 3-day 5,031,963, rain 7-day 2,425,498 (rain kernels thinned with `ROW_STRIDE=4`; the others with 2). The base-feature runs keep more rows (4.35 M step rows on the old split) because they need no extension join.
`backup/event_models/<slug>/` holds the exact bundles that were pushed (the `train-events|ranges|day` bundles predate the lean loader — same results, it only matters for the rain tables).

## 4. Verified findings (from logs)

* ERA5 has no thunderstorm/fog/visibility → METAR is the only honest truth. METAR `p01i` is zero-filled → unused. CHIRPS has no nodata flag, ocean = −9999 → masked.
* GFS on AWS: 2021-01-01→now (`atmos/` path from v16); GEFS mean/spread from 2017. **NOAA corrupt data: 2022-11-30 00Z, 9 steps unreadable.**
* GFS vs METAR (smoke): temp bias +0.9 °C d1–2, RH bias −11…−12 %, wind +0.9…+1.5 m/s; raw CAPE AUC for thunderstorm only 0.60 (headroom); GFS rain vs CHIRPS corr 0.48 (day 0) → 0.25 (day 4).
* **Label noise (CHIRPS vs METAR rain, 185,664 station-days):** AUC 0.81 overall, 0.71 in the monsoon → rain probabilities are calibrated to CHIRPS, not a gauge.
* Extension smoke: no field absent from either archive; found + fixed `soilw` sentinel 9999.
* 13 of 136 training stations have GFS land fraction < 0.5 (5th percentile 0.27) → the coastal rule (`FeatureBuilder.min_land`) follows the stations.
* Live smoke samples (not checked against observations): Delhi 6 Oct tmax day-1 q10/q50/q90 = 33.5/35.5/35.7 °C, P(any rain) 0.026. **UNVERIFIED / suspicious:** Cherrapunji day-1 tmax ≈ 29.6 °C looks high for a ~1,300 m town (likely GFS cell elevation vs the real site).

## 5. Results (full tables: `docs/AGENTIC_ARCHITECTURE.md` §5.3; copied from the kernel logs by `results_from_logs.py`)

test_space = places never seen (zone-stratified 4°×4° blocks), future dates (from 2025-01-06), beside zone × month climatology / raw GFS. Headlines: thunderstorm AUC 0.849 / BSS +0.075, fog 0.855 / +0.084, strong_wind 0.890 / +0.078,
rain_3h 0.858 / +0.182, dust 0.888 / +0.094, hot_day 0.965 / +0.339, cold_night 0.975 / +0.260, heatwave_imd 0.865 / +0.060; ranges beat raw GFS (tmax MAE 1.85 vs 2.43, temperature 1.63 vs 2.06, wind 1.09 vs 1.39, humidity 7.6 vs 13.4; tmin only 2.00 vs 2.04);
rain_curve BSS +0.20…+0.26 up to 15.6 mm/day and none at ≥ ~90 mm (flagged by the engine); windows BSS +0.15…+0.48. **Caveats:** one seed, no CIs; extension-vs-base is not row-paired (extension helps heatwave_imd, slightly rain_3h, not fog); CHIRPS label noise.

## 6. Bugs found and fixed (each would have corrupted results or wasted a run)

Earlier: `vis_km` forecast/observed column collision · truncated GRIB ranges crashed a shard · eccodes missing on Kaggle · `soilw` 9999 sentinel · station normals counted days not years · bundler `--env=K=V` form baked a bogus var · window table crashed with 5 forecast days · gate admitted a pure-noise model
(`MIN_BSS 0.005`, `MIN_AUC 0.55`, 1 % MAE gain) · step anomalies keyed by station vs serving (now one nearest-node climatology) · hash-only hold-out left three zones with zero held-out points (now zone-stratified) · climatology artifact lost its `bin` name · negative conformal margin
(shared `train.calibrated_interval`) · muddled freshness rule (`live.run_is_fresh`, ≤ 54 h).
This session: both rain kernels OOM-killed (full-width tables merged before thinning) → `read_thinned` per-file float32, per-product truth columns, `inplace` merge, `ROW_STRIDE=4` · **gate accepted `coldwave_imd`** (worse than the global rate) → global-rate rule · rain-curve tail thresholds had no skill but looked validated → `thresholds_without_skill_mm` ·
**live static fetch ran `idx_task` outside the worker pool** (HTTP client is None) → fixed (found by the live smoke run) · coastal cities (Chennai 0.29 land) wrongly refused → data-driven `min_land` · `api.py` used postponed annotations so FastAPI could not resolve the local request model (every `/tool` call 422) → fixed (found by its test) ·
test for short-horizon data patched `read_all` but p2 is now read via `read_thinned` → test updated.

## 7. Code map

`event_models/` — `labels.py`, `collect_truth.py`, `collect_gfs.py` (env `GFS_LEADSET`, `SHARD_*`), `collect_gfs_ext.py`, `dataset.py` (joins, CHIRPS truth, windows, IMD labels, `read_thinned`, `run_dates`), `features_events.py`, `train.py`, `rain_products.py`,
`run_training.py` (20 targets, `build_tables`, provenance), `bundle.py` (inlines `event_models` + `weathergpt_events`), `make_train_kernels.py` (`--push --only= --base --serve --publish`, `GROUP_ENV`), `serve_smoke.py`, `publish_hf.py`, `validate_smoke.py`, `inspect_ext.py`.
`weathergpt_events/` — `registry.py` (gate, `unskilled_thresholds`), `models.py`, `static.py`, `features.py` (`min_land`), `engine.py`, `live.py` (`RunStore`), `tools.py`, **`loader.py`, `service.py`, `api.py`, `ceo_bridge.py`, `__main__.py`**, `README.md`.
`notebooks/weathergpt_events_implementation.ipynb`; `docs/EVENT_MODELS_GUIDE.md`, `docs/AGENTIC_ARCHITECTURE.md`; `backup/event_models/` bundles, `kaggle_log.py`, `results_from_logs.py`, this file.

## 8. Testing (see the guide §7 for the table)

Run `python3 -m pytest -q tests -p no:cacheprovider` (≈ 4 min; first `test_event_pipeline.py` run builds a ≈ 9 min synthetic fixture cached in `/tmp/weathergpt_pipeline_fixture_v4`; don't run it while heavy processes hog the 3.7 GB laptop).
Last full run (2026-10-07, whole repo incl. the backend's tests): **147 passed in 3 min 44 s**. Covered: labels, leakage rules, trainer, **train/serve parity**, gate, engine, service, CLI, router, CEO bridge, notebook (offline, canned engine).
**Not covered by unit tests:** the live network fetch (exercised once by the Kaggle smoke run), forecast quality beyond held-out metrics, the notebook against the live HF repo.

## 9. Open items, in priority order

1. **When the user says the other agent is done with Kaggle:** (a) run `python3 event_models/make_train_kernels.py --publish --push` to refresh the HF `src/` + card; (b) execute the implementation notebook for real (push it as a Kaggle notebook with internet + the HF token) and keep its outputs; (c) optional: re-run the smoke test.
2. Ask the user whether to push the local cleanup commit and whether to make the HF repo public. The old public `Arko007/weathergpt-models` card still says `pip install weathergpt-models` (not on PyPI) — a README-only fix via a Kaggle kernel, only if the user wants it.
3. Science, optional: row-paired base-vs-extension ablation (re-score base on the extension rows), several seeds + confidence intervals, check Cherrapunji-type high-terrain temperature bias, fog may be better served by base features, `tmin_range` is marginal.
4. v1.1: nowcast from live observations, 12Z cycle, odd run dates, gridded temperature truth. The `app/` planner/validator/executor is the teammate's.

## 10. Risks / unknowns

Single seed / no CIs; CHIRPS label noise; IMD wave labels are METAR-normals approximations; one live run tested (06 Oct, post-monsoon withdrawal — monsoon-season behaviour of the live path is untested); NOAA / IEM / AWS URLs are third-party and could change.
