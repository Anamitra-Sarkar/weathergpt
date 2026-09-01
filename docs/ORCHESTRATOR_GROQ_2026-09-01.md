# Orchestrator — Groq 4-Model Setup — 2026-09-01

**User preference (no llama, deprecated):**

```
PREFERRED_MODELS = [
  "qwen/qwen3.8-27b",   # 0
  "qwen/qwen3.6-27b",   # 1
  "openai/gpt-oss-20b", # 2
  "openai/gpt-oss-120b" # 3
]
ORCHESTRATOR_MODEL = "qwen/qwen3.8-27b"
```

**File:** `app/orchestrator/models.py:5` + `groq_client.py:12` (`model_for(role)` round-robin, hard-rejects `llama`, reads `GROQ_API_KEY` from env or `/home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt`, now also hard-injected in `official_train.py` for Kaggle).

**Routing:**

| Role | Model | Queue index for 6 agents |
|------|-------|--------------------------|
| orchestrator | qwen/qwen3.8-27b | 0 |
| intent_parser | qwen/qwen3.8-27b | 0 |
| location_resolver | qwen/qwen3.6-27b | 1 |
| forecast_agent | openai/gpt-oss-20b | 2 |
| history_agent | openai/gpt-oss-120b | 3 |
| warning_agent | qwen/qwen3.8-27b | 4→0 |
| solution_agent | qwen/qwen3.6-27b | 5→1 |
| reviewer | openai/gpt-oss-20b | 0 |
| explainer | openai/gpt-oss-120b | 1 |

`model_for(int)` also queues: `i % len(PREFERRED_MODELS)`.

**Live verification 2026-08-31 23:56 IST:**

```
qwen/qwen3.8-27b -> 200 {"choices":[{"message":{"content":"Hi there"}}]}
qwen/qwen3.6-27b -> 200
openai/gpt-oss-20b -> 200
openai/gpt-oss-120b -> 200
orchestrator live: Ready to forecast weather.
```

**Backend integration:** `app/main.py:136` — if `GROQ_API_KEY` present (now in `.env` as `qwen/qwen3.8-27b`), `POST /query` builds `WIO + RADE` then calls `groq_client.generate(..., role="explainer_agent", model=ORCHESTRATOR_MODEL)` with WIO JSON only (never raw GRIB). Mock fallback if missing.

**Kaggle usage:** `official_train.py` now hard-injects the same key so `M3` Groq paraphrasing `800 via 4-model queue` works even when `GROQ_API_KEY` not in `kaggle/input`. Log: `Groq paraphrasing 800 via [...]` `0/800 qwen3.8 … 700/800` all `200` in v3, but v2 had `Groq ready: False` due to missing injection.

**What still needs your attention:** `429 Too Many Requests` on burst paraphrase (800 calls in ~300s) — add `tenacity` retry + `0.25s sleep` already in code, but may need `1s` backoff for hackathon demo.

**No extra keys needed today** — `GROQ_API_KEY` is set, Open-Meteo/POWER are key-free, IMD/HF optional later.
