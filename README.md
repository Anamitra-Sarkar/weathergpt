# WeatherGPT

WeatherGPT is a modular FastAPI weather-intelligence backend. It normalizes source data into Canonical Evidence Objects (CEOs), applies deterministic semantic/temporal/spatial checks, builds a provenance-bearing WIO, and can run a risk-aware decision policy.

## Implemented

- CEO schema with source, valid window, statistic, accumulation window, provenance, retrieval time, and derived-evidence lineage.
- Deterministic location and time resolution. Unknown or ambiguous locations return structured errors; they are never mapped to a default city.
- Open-Meteo forecast and Open-Meteo ERA5 reanalysis adapters; NASA POWER historical adapter; CAP decoder and configured CAP feed adapter.
- Explicitly isolated IMD and GRIB2 integrations: they report unavailable until a configured, supported endpoint/decoder exists.
- Concurrent retrieval with per-source failure reporting and freshness-aware in-memory cache.
- Semantic compatibility gate: precipitation amount, rate, and probability cannot be mixed; accumulation windows must match.
- Warning evidence remains categorical and separate from numeric fusion.
- Structured deterministic agents, evidence-ID reviewer, WIO response, evidence lookup, SQLite user-context and feedback storage, and RADE v2.
- Versioned and compatibility API endpoints: `/health`, `/wio/query`, `/query`, `/decision`, `/rade/advise`, `/context`, `/feedback`, `/forecast`, `/warnings/active`, `/evidence/{id}`, `/metrics`.

## Experimental or unavailable

- Member-level ensemble output is accepted only when Open-Meteo returns actual member fields. A forecast mean is not presented as an ensemble.
- IMD requires a real compatible `IMD_API_KEY` and endpoint. CAP requires `CAP_FEED_URL`.
- GRIB2 is unavailable without `eccodes` and `cfgrib`; it is not used as a fallback.
- The checked-in ML artifacts and historical Kaggle claims are **not validated production metrics**. Training scripts must use a real, versioned dataset for reportable results.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
pytest -q
```

Example:

```bash
curl -X POST http://localhost:8001/wio/query \
  -H 'content-type: application/json' \
  -d '{"question":"Will it rain in Nagpur tomorrow afternoon?"}'
```

`GROQ_API_KEY`, `IMD_API_KEY`, and `CAP_FEED_URL` are optional configuration. Secrets are read only from environment variables and must not be committed.

See [architecture](docs/ARCHITECTURE.md), [API notes](docs/API.md), and [verification](docs/VERIFICATION.md).
