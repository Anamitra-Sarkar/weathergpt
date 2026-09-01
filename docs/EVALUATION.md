# WeatherGPT — Evaluation

## Metrics (per your brief)

* **Forecast Skill:** `Brier Score` for `rain>0.5mm` binary (`0 best`), `RMSE`/`CRPS` for amounts
* **Decision Accuracy:** `RADE best` vs expert/empirical outcome
* **Calibration:** `70%` forecast → `~70%` observed
* **Timeliness:** `p50 <800ms` `WIO` (LLM separate), `p95 <2s` end-to-end
* **Robustness:** `answered / total` (vs missing context)
* **Trust:** field survey `clarity + provenance`

## Training Metrics (Kaggle)

**M1 Semantic 9-label (TF-IDF fallback on T4 40s):** `acc 1.00 f1 1.00` on 165 rows → **overfit, not best** (needs DistilBERT 8ep on 1200, expect `0.92/0.90`)

**M2 Bias Deep MLP (genuine):**

| Run | Rows | Device | Epochs | Best Val | `rmse_t` | `rmse_p` | Bias |
|-----|------|--------|--------|----------|----------|----------|------|
| T4 40.5s | 4800 (10d) | P100→CPU | 20 | 0.817 | 1.27 | 0.046 | — |
| Official v1 30d | 14400 | P100→CPU | 30 | 0.990 | 1.39 | 0.196 | -1.41±2.05 |
| Official v2 30d | 14400 | P100→CPU | 30 | 0.892 | 1.32 | 0.197 | -1.41 |
| **Official v3 30d (current, fixed)** | **14400** | **CPU (P100) / T4 x2 GPU** | **30** | **0.89–0.99** | **1.32–1.41** | **0.19** | **-1.41** |

*Synthetic Gaussian 10k would have `rmse ~0.28` but is **not** used in official — 14.4k is real GFS vs ERA5.*

**M3 Intent 5-way:** `TF-IDF acc 0.45 f1 0.45` on 1200 template → **weak** (needs DistilBERT 5ep on 2000 Groq-diverse, expect `>0.75`)

## Backend Verification (live, `mock_mode False`)

* `pytest 6/6` — `comparable_gate`, `ceo_roundtrip`, `wio_preserves_warning`, `imd_city`, `immd_warning`, `cap`
* `GET /health 200`, `POST /wio/query` Nagpur `20ev`, `POST /query` Nagpur spray `2.4mm` + `provenance`, `Mumbai 87ev`, `Delhi 28.1°C` — all `agreement full` until `429` burst
* `GET /plan` `horizon short` correctly

## Backtest (planned)

```bash
python scripts/backtest.py --wio-endpoint http://localhost:8001/wio/query --csv datasets/historical_2024.csv
# CSV: question,location_raw,valid_from,valid_to,observed_rain_mm
# → Brier + RMSE reported, compare vs baseline "always IMD warning"
```

Run after `official v4` DistilBERT finishes (now `RUNNING` with `use_cpu` fix).
