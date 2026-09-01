# WeatherGPT — API

Base `http://localhost:8001` — see `/docs` (OpenAPI)

## Health

`GET /health` → `{"status":"ok","uptime":123,"mock_mode":False,"timestamp":"..."}`
`GET /` → `{"message":"WeatherGPT API — see /docs"}`

## Plan (debug)

`GET /plan?q=Will it rain tomorrow afternoon in Pune?&location=Pune`
→ `{"location":{...},"valid_from":"2026-09-01T12:00+05:30","horizon":"short","evidence_classes":["observation","forecast","warning","radar"]}`

## WIO (no LLM)

`POST /wio/query`
```json
{"question":"Will it rain in Nagpur tomorrow afternoon?","location":{"raw":"Nagpur"},"lang":"en"}
```
→ `{"wio":{"query":{...},"weather":{"summary":"Forecast precipitation 0.0 mm","rain":{...}},"official_warning":{"active":false},"agreement":{"status":"full_agreement"},"evidence":[...],"disagreements":[]},"evidence_count":20,"warnings":[]}`

## Query (WIO + RADE + LLM)

`POST /query` same body → `{"answer":"**Rain forecast 12–18 IST** 2.4mm …","wio":{...},"evidence_count":18,"warnings":[],"lang":"en"}`

* `WIO only` goes to LLM, never raw GRIB. If `GROQ_API_KEY` missing, mock `Advice: best — note`.

## RADE

`POST /rade/advise` same body → `{"wio":{...},"best_action":"irrigate","scores":{"spray":-13,"wait":1.6,"irrigate":10,...},"scenarios":[...],"explanation":"Irrigation recommended..."}`

## Evidence

`GET /evidence/{id}` (stub, future) — returns `CEO` by `evidence_id`.

## Warnings

`GET /warnings/active?district=Nagpur` → list of `WIOWarning` (from `CAP` + `IMD`).

## cURL

```bash
curl http://localhost:8001/health
curl "http://localhost:8001/plan?q=Will%20it%20rain%20tomorrow&location=Nagpur"
curl -X POST http://localhost:8001/wio/query -H "Content-Type: application/json" -d '{"question":"Will it rain in Nagpur tomorrow afternoon?","location":{"raw":"Nagpur"}}' | jq
curl -X POST http://localhost:8001/query -H "Content-Type: application/json" -d '{"question":"Will it rain in Nagpur tomorrow afternoon and should I spray pesticide?","location":{"raw":"Nagpur"}}' | jq
```
