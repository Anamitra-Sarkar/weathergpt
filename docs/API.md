# API

All request bodies are Pydantic-validated. Errors use `{error:{code,message,details,request_id}}`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, readiness, source configuration, cache and model-loading status. |
| `POST /wio/query` | Validated evidence retrieval and WIO only. |
| `POST /query` | WIO plus deterministic evidence-grounded synthesis. |
| `POST /decision`, `POST /rade/advise` | WIO plus RADE v2 result. |
| `GET /evidence/{id}` | A CEO generated in this running process. |
| `GET /forecast?location=Nagpur` | Convenience WIO forecast view. |
| `GET /warnings/active?location=Nagpur` | Active normalized warnings for a location query. |
| `POST /context`, `POST /feedback` | User-scoped SQLite context and outcome records. |
| `GET /metrics` | Process metrics and cache hit rate. |

The equivalent `/api/v1/` query, health, decision, context, and feedback routes are available where listed by OpenAPI. Location must be supplied explicitly or occur unambiguously in the question.
