# event_models — per-target weather event / range / rain-curve models (training + collection)

Design and rationale: `docs/AGENTIC_ARCHITECTURE.md`; how to use / reproduce / test: `docs/EVENT_MODELS_GUIDE.md`. Current state and resume steps:
`backup/event_models/SESSION_HANDOFF_2026-10-06.md`. Inference side: `weathergpt_events/`.

Runs on **Kaggle only** (Modal is not used). Each kernel is ONE file produced by `bundle.py`, which inlines the
`event_models.*` and `weathergpt_events.*` modules it imports and bakes environment variables:

    python event_models/bundle.py event_models/collect_gfs.py out.py --env SHARD_START=2021-04-01 --env SHARD_END=2022-04-30

| stage | script | needs | writes |
|---|---|---|---|
| truth | `collect_truth.py` | internet | METAR obs/hourly/daily labels, CHIRPS arrays |
| base forecasts | `collect_gfs.py` | internet, eccodes (auto-installed) | `p1_station_steps_*`, `p2_point_days_*`, `points.parquet` |
| extension predictors | `collect_gfs_ext.py` | internet | `p1x_*`, `p2x_*`, `points_ext.parquet` (GFS dynamics + GEFS) |
| training | `run_training.py` (via `make_train_kernels.py`) | the collectors' outputs as kernel inputs | per-target `model*.txt`, calibrators, `metrics.json` (with provenance), `climatology_day.parquet` |
| serving smoke test | `serve_smoke.py` (`make_train_kernels.py --serve`) | trained outputs + internet | static grids, live run, gate + engine on 17 places (9 inland, 5 coastal, 3 that must be refused), PASS/FAIL |
| publish | `publish_hf.py` (`make_train_kernels.py --publish`) | serve-smoke output + private creds dataset | private Hugging Face repo with a generated model card |

Key environment variables: `GFS_LEADSET` (short|long|all), `SHARD_START/END/STRIDE/OFFSET`, `TARGETS`, `ROUNDS`,
`USE_EXT` (auto|0|1), `ROW_STRIDE` (the two rain kernels run with 4: `GROUP_ENV` in `make_train_kernels.py`). Leakage rules live in `dataset.py` / `features_events.py` (forecast-run features only,
3-hour label windows around the valid time, uncovered windows dropped, gaps between periods, 4°×4° spatial hold-out).
Tests: `tests/test_event*.py`, `tests/test_events_*.py`, `tests/test_rain_products.py`.

`make_train_kernels.py` flags: `--push`, `--only=a,b`, `--base`, `--serve`, `--publish`. Results tables from logs: `backup/event_models/results_from_logs.py`.
