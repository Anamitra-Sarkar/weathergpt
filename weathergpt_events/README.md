# weathergpt_events — inference package for the event / range / rain-curve models

Turns the newest GFS + GEFS run into validated answers for a point in India. Full guide: `docs/EVENT_MODELS_GUIDE.md`; design and results: `docs/AGENTIC_ARCHITECTURE.md`.

| module | role |
|---|---|
| `registry.py` | admission gate (provenance, skill on unseen places over zone × month climatology **and** the global rate, coverage band, monotone curves) and per-(zone, lead) skill lookup |
| `models.py` | loads the four artifact kinds (binary / cross-entropy / quantile / exceedance curve); shares prediction code with the trainer |
| `static.py` | terrain / land / coast descriptors at any lat-lon, defined exactly as in training; domain check |
| `features.py` | the same feature tables the trainer builds (parity-tested), for one query point |
| `engine.py` | `EventEngine.forecast(lat, lon, run, …)` — no numbers without validated skill; refuses sea / out-of-domain |
| `live.py` | `RunStore`: newest complete GFS+GEFS run from NOAA's AWS archive (byte-range + GRIB), cached; static-grid builder |
| `tools.py` | tool catalogue for an orchestrator, built from the loaded registry |
| `loader.py` | `load_engine(folder_or_hf_repo_id)` |
| `service.py` | `ForecastService`: run cache + freshness refusal, validated `call_tool`, `summarise` |
| `api.py` | optional FastAPI router (`create_router`, `create_app`) |
| `ceo_bridge.py` | validated answers → `app.schemas.ceo.CanonicalEvidenceObject`s |
| `__main__.py` | CLI: `python -m weathergpt_events {status,catalogue,forecast}` |

Imports only numpy / pandas / scipy / lightgbm (plus `eccodes`, `httpx` for the live fetch; `huggingface_hub` for repo ids; `fastapi` for the router). The feature code is the SAME code the models were trained with (`event_models`).
Tests: `tests/test_events_*.py`, `tests/test_event_pipeline.py`, `tests/test_notebook.py`.
