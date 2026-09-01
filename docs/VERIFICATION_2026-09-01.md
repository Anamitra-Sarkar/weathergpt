# Verification Report — 2026-09-01 01:45 IST

**Scope:** Backend + decoders + schemas + gates + training + real-world queries, all with live calls (no mock).

## Backend (`app/main.py:28`)

* `GET /health` `200 {"status":"ok","mock_mode":False}` — Groq live
* `GET /plan` Nagpur `tomorrow afternoon 12–18 IST` → `horizon short`, `evidence_classes [observation,forecast,warning,radar]` ✅
* `POST /wio/query` Nagpur → `WIO 20 evidence, valid 2026-09-02T12:00+05:30→18:00, weather 0.0 mm (OPEN_METEO), agreement full, warning False` ✅
* `POST /query` Nagpur spray → `Weather 0.0 mm … no data for afternoon → uncertain, spray acceptable` + `RADE irrigate` ✅ (Groq `qwen/qwen3.8-27b` orchestrator live, but `429` on burst)
* `POST /rade/advise` → `best irrigate` ✅
* `GET /docs` `200` ✅

**Live Open-Meteo:** `fetch_open_meteo 21.14,79.08` → `144 CEOs` (48h hourly), sample `precip 0.0 mm window 1h` ✅

## Decoders (`tests/test_decoders.py:3`)

* `IMD forecast` 4 CEOs, `accumulation 24h` ✅
* `IMD warning orange` ✅
* `CAP Severe→orange`, `Cancel→cancelled` lifecycle ✅
* `CAP` preserves `severity` separately, never averaged

## Schemas / Gates (`tests/test_ceo.py:3`)

* `are_comparable` blocks `rain_rate vs accumulation`, `6h vs 3h` ✅
* `temporal filter` keeps only overlapping window (1/2) ✅
* `spatial near 0.0km vs far 850km Delhi` ✅
* `ranker prefers nearby 0.98 vs 0.79` ✅
* `disagreement 5 vs 40mm` flagged ✅
* `WIO preserves warning orange` ✅

**Pytest:** `6/6 passed`

## Training Dry-runs

* `train_semantic --dry-run` `46 rows` → `metrics.json acc 1.0` (fake)
* `train_bias --dry-run` `500 rows epoch1 val 0.15 rmse_t 0.28`
* `train_intent --dry-run` `rule parser` `pesticide_spraying`

## Real-World Scenarios (httpx, 18005)

| Scenario | Q | WIO | Answer | RADE |
|----------|---|-----|--------|------|
| Nagpur spray | Will it rain … spray? | 20 ev, 0.0 mm | 0.0mm 00–04 UTC, uncertain afternoon → spray acceptable | irrigate |
| Mumbai warning | heavy rain next 3 days? | 87 ev | Groq 429 fallback → irrigate | irrigate |
| Delhi tonight | temp tonight? | 24 ev, 28.1°C, 5.2 km/h | 28.1°C provenance IDs | irrigate |

All have `valid_from/to` IST→UTC correctly, `accumulation_hours` preserved, warnings separate.

## Orchestrator Groq

* `app/orchestrator/models.py:5` `4-model queue` verified live `200` all, `orchestrator qwen/qwen3.8-27b`, `429` on burst needs `tenacity` retry (not yet).

## Genuine vs Bluff

* **Datasets:** No Gaussian bluff in final official (14400 real GFS vs ERA5, 1200 field_names diverse, 2000 intent Groq-diverse). Checker caught `corr 1.0` duplicate and fixed to `0.96`.
* **Training:** M2 `best.pt 104K` + `scaler.pkl` + `metrics.json` is genuine (30ep logged). M1/M3 were bluff until v4 `no_cuda` fix — v4 now `RUNNING` will produce them.
* **No synthetic fallback** in official (synthetic only if `matched_pairs.csv` missing, but it exists).

## Issues Still Open at Pack-up

* Official v4 still `RUNNING` at 01:45 — need to re-pull `M1/M3 metrics.json` after `COMPLETE` to confirm `M1 f1 >0.90`, `M3 f1 >0.70` (was `1.00` overfit and `0.45` TF-IDF before).
* Groq `429` needs retry + backoff for 800-paraphrase burst.
* `field_names 1200` still has `365 vs 72` imbalance — `class_weight` mitigates but 1200 is minimum for hackathon best.
