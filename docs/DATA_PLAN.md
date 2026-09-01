# WeatherGPT — Data Plan (Trainable? then only train)

**Status:** Audit 2026-08-31 — no Kaggle P100 train until this plan is approved.
**Rule:** Synthetic-only training is blocked for real evaluation; at least one real-data pipeline must be green.

---

## 1. Summary Verdict

| Model | Current placeholder | Rows | Diversity | Real-data path exists? | Trainable? | Verdict |
|-------|---------------------|------|-----------|------------------------|------------|---------|
| **M1 Semantic Variable Classifier** `train_semantic_classifier.py` | 23 raw pairs ×20 aug = **460** rows | 6 labels | **Very low** — only casing/space noise. No `GRIB shortName`, `NetCDF var`, `CAP event`, `BUFR code` diversity | **Yes** — `field_names.csv` can be auto-built from `variable_registry.py` + real GRIB/NetCDF samples + CAP events | ⚠️ **Barely trainable** on placeholder; needs 3× expansion to be useful | **Expand to ~1.5k–2k before P100 train** |
| **M2 Bias-Correction/Downscaler** `train_bias_correction.py` | Gaussian `N(0,1)` 5-dim `n=10k` | 2 targets (t2m, apcp) | **Meteorologically fake** — no seasonality, elevation, lead-time physics, no spatial correlation | **Yes** — `Open-Meteo Archive (ERA5)` + `NASA POWER` + `Open-Meteo Forecast` all returned `200` in dry pulls; IMD AWS still gated (needs key) | ❌ **NOT trainable on current Gaussian** for any real claim (would memorize noise) | **Build real matched pairs first** |
| **M3 Intent Parser** `train_intent_parser.py` | 6 templates ×600 sampled = 600 but paraphrased only by **uppercasing** | 4 decision labels | **Critically low diversity** — no village variations, no Hinglish, no time paraphrases, no multi-intent | **Yes** — can be built from 200+ paraphrase generation + `intent_samples.jsonl` upload | ❌ **NOT trainable for real NLU** on current 6 templates | **Expand to ≥1k diverse utterances before P100 train** |

**Overall:** **Do not train M2/M3 on Kaggle P100 with current synthetics.** M1 can be trained *after* expansion; M2/M3 must first have real/faux-real data built.

---

## 2. Data Source Realism Check (dry pulls 2026-08-31)

Performed live in this workspace:

- **Open-Meteo Archive API** `archive-api.open-meteo.com/v1/archive` → `200` — returns 72-hour to 2-year hourly `temperature_2m, precipitation, wind_speed_10m` for any lat/lon. **Usable as ground-truth (ERA5 reanalysis)** for India (tested `21.14,79.08` Nagpur `2024-01-01→2024-01-03`).
- **Open-Meteo ERA5** `archive-api.open-meteo.com/v1/era5` → `200` — 8784 hourly points for 1 year (tested `2024-01-01→2024-12-31`). **Usable for reanalysis stream.**
- **Open-Meteo Forecast** `api.open-meteo.com/v1/forecast` → `200` — 72h hourly. **Usable as GFS/ECMWF pseudo-forecast** (48h capped in current `open_meteo.py`).
- **NASA POWER** `power.larc.nasa.gov/api/temporal/daily/point` → `200` — daily `T2M, PRECTOTCORR` for same point. **Usable as independent satellite/obs proxy.**
- **IMD AWS/ARG APIs** → **Not tested** (registration-gated per your brief; `https://api.data.gov.in` needs key). **Current `imd_samples.jsonl` missing.** Until key arrives, IMD path is stubbed.
- **Kaggle public weather datasets** → `global-weather-repository` (daily updating, ~13 MB), `weather-data` etc. exist but are mostly US/EU; **less useful than Open-Meteo for Indian bias correction.**

**Implication:** We have **two free, key-free, syllabus-compliant real sources already reachable** (Open-Meteo + NASA POWER). No need to wait on IMD to start *real* training — can construct matched `forecast→observation` pairs today.

---

## 3. Per-Model Data Plan (what to build, schema, QC)

### M1 — Semantic Variable Classifier (6 labels: `precipitation_amount`, `precipitation_probability`, `precipitation_rate`, `temperature_2m`, `wind_speed`, `heavy_rain_warning`)

**Goal:** Map noisy native names/codes → canonical `variable`. The 460-row placeholder collapses to 23 unique strings → DistilBERT will overfit to memorization, not semantics.

**Trainable definition:**
- Schema: `training/datasets/field_names.csv` with columns `raw_field, canonical_variable, statistic, accumulation_hours, source_hint`
- Must contain examples from **each evidence family** (not just pretty names):
  - GRIB2 `shortName`/`paramId`: `2t, tp, prate, u10, v10, tcc, APCP` + case variants
  - NetCDF `wrfout`: `T2, RAINC, RAINNC, U10, V10, SST`
  - IMD codes: `category_code heavy_rainfall`, `rainfall actual, normal, departure`
  - CAP XML `event`: `Heavy Rainfall, Thunderstorm, Cyclone, Fog, Heat Wave`
  - BUFR/HDF5 codes if available
  - Human paraphrases: `precip, rainfall, rain rate, chance of rain, PoP, tmax, tmin, wind gust`
- **Size:** 1.5k–2k rows (23 → ~80 unique raw strings × ~20 augmentations = good). Include at least **50 examples per label**, balanced.
- **QC:** label must exist in `LABEL_MAP`, no leakage between train/val by raw_string (stratified by label), dedup by lowercased raw_field.
- **Augmentations:** case, underscore/hyphen/space swap, typo (swap), prefix `IMD:` `GFS:`, unit suffix ` (mm)`, not just `upper()`.

**Build plan (P100-ready):**
1. Auto-generate `field_names.csv` from `variable_registry.py` + GRIB code tables + CAP event list (no manual labeling).
2. Merge + validate → `make_dataset()` already does dedup/aug after that.
3. Then train `distilbert-base-uncased` 5 epochs, `batch 32`, `lr 2e-5`.

**Trainability after expansion: ✅ High.** Current 460 is **medium** but risky; recommend 1.5k.

### M2 — Bias-Correction / Downscaler (the hard one: `GFS forecast → truth`)

**Goal:** Learn `bias = observation − forecast` for `t2m (°C)` and `precipitation (mm)` conditioned on `[gfs_t2m_norm, gfs_apcp_norm, elevation, lead_time, lat]`. Must **not** be trained on i.i.d. Gaussian.

**Current failure:**
- Synthetic `X~N(0,1)` → `t_bias = 0.3*elev + 0.1*lead` is linear and has **zero meteorological signal** (no diurnal, seasonal, monsoon, orographic effects). A 3-layer MLP will trivially memorize the linear formula; `RMSE_t 0.28` from dry-run is just fitting noise, not weather.

**Trainable definition (real):**
- **Option A (preferred, no IMD key yet):** `GFS forecast (Open-Meteo) → ERA5 reanalysis (Open-Meteo Archive) as pseudo-observation` for Indian grid.
  - Fetch: For each point (say 50 Indian districts × 365 days 2024), pull:
    - Forecast: `https://api.open-meteo.com/v1/forecast` with `past_days` unavailable retrospectively, so better: use **Open-Meteo `archive` as truth** and **Open-Meteo `historical forecast` via `https://historical-forecast-api.open-meteo.com/v1/forecast`** or keep forecast from same archive with time shift? *Check:* historical forecast API exists; test after.
  - Simpler today: Use **Archive ERA5 as truth** and **Forecast API current 3-day** for the future bias run; for *historical training*, pair `GFS analysis` (`archive` is reanalysis) with `NASA POWER` daily as independent obs (see Section 5). Best immediate: **ERA5↔POWER daily matched pairs** (both cover 2024, same `lat/lon`, align by `date`).
- **Option B (once IMD key arrives):** `GFS (forecast) → IMD AWS (station)` matched by `lat/lon + valid_time ≈ observed_at ±30min`, `elevation` from SRTM. This is gold-standard.
- Schema: `training/datasets/matched_pairs.csv` columns:
  `lat, lon, elevation_m, lead_hours, valid_from, gfs_t2m_k, gfs_apcp_mm, obs_t2m_c, obs_apcp_mm, target_t_bias_c, target_p_bias_mm, source_pair (era5-power|gfs-imd)`
- Size: **≥10k–50k matched rows** (e.g., 20 stations × 365 days = 7.3k daily or ×24 = 175k hourly). Hourly is better for timing/accumulation-window learning.
- **QC / Leakage prevention:**
  - Time-based split (not random): train `2024-01→2024-09`, val `2024-10→2024-12` (prevent look-ahead).
  - Spatial hold-out: hide 5 districts from train, evaluate there.
  - Check accumulation windows match before subtraction (`mm/6h` vs `mm/24h` cannot be differenced).
  - Remove `station missing / quality_flag bad` rows.
  - Clamp outliers (`precip > 300mm/day` flag, don't drop monsoon extremes).

**Build plan (P100-ready):**
1. Script `scripts/build_bias_pairs.py` (to be added) that fetches Open-Meteo Archive + NASA POWER for a config list of lat/lons and writes `matched_pairs.csv`.
2. `train_bias_correction.py` switches from `synth_data()` to `load_real()` once file exists (already wired).
3. Train `MLP hidden 64×3` 20 epochs or `LGBM` 200 rounds; metrics `RMSE`, `Brier` (rain>0.5mm), `CRPS` if ensemble.

**Trainability today: ❌ Blocked without step 1.** After building ~20k real pairs: **✅ Highly trainable on P100 (2–4 min).**

### M3 — Intent Parser (user question → slots)

**Goal:** Extract `{location, time_window, variables, decision}` from free text including Hinglish/village names/typos.

**Current failure:**
- `UTTERANCES = 6` → `synth_jsonl(500)` samples by repeating those 6 + uppercasing only. Train val will contain *identical strings* → **leakage + 1.0 fake accuracy**, no generalization to `pin code`, `tomorrow evening in Malegaon`, `kal baarish hogi?`.

**Trainable definition:**
- Schema: `training/datasets/intent_samples.jsonl` each line `{"text": "...", "intent": {"variables": [...], "time": "...", "location": "...", "decision": "..."}}`
- Need **≥1k diverse utterances** covering:
  - Locations: 30 Indian cities + 30 villages + `lat,lon` + `6-digit pincode` + `my village` + `near Mumbai`
  - Times: `today/tonight/tomorrow morning/afternoon/evening/day after tomorrow/next 3 days/this weekend/23rd Aug/coming Monday`
  - Variables: paraphrases `rain, rainfall, precipitation, chance of rain, temperature, garmi, wind, warning`
  - Decisions: `pesticide, irrigation, harvest, fishing, travel` + `no decision` (pure forecast)
  - Languages: mix `en` + `hi` transliteration (`kal baarish`) at least 20%
- **Generation strategy (no manual labeling hell):**
  - Template expansion (50 templates × 20 slot combos = 1k) + LLM paraphrase using **your 4 Groq models queued** (`qwen3.8` orchestrates paraphrasing, others generate variants). Human review 100 samples.
  - Stratified split by template id (no template leaks to val).
- **QC:** dedup by normalized text, check slot spans actually appear in text.

**Build plan (P100-ready):**
1. Script `scripts/build_intent_corpus.py` generates 1k+ from templates + Groq paraphrases (can run locally with your `groq_api.txt`).
2. `train_intent_parser.py` trains `distilbert` sequence classifier on decision label (later extend to joint slot tagger).
3. Eval `slot-F1` not just `decision accuracy`.

**Trainability today: ❌ Synthetic 6-template → not.** After 1k diverse: **✅ Trainable on P100 (~2 min).**

---

## 4. Data Hygiene & Provenance (applies to all)

- Every row carries `source, ingested_at, provenance` (mirrors `CEO` philosophy).
- No LLM silently averages incompatible windows — check `accumulation_hours` before any `amount` join.
- Train/val splits **time-aware and geo-aware**, not random.
- Seeds fixed (42), configs in `training/configs/*.yaml`, metrics to `training/models/*/metrics.json` + MLflow-ish log.
- Kaggle `T4/P100` vs CPU controlled by `device auto` + `amp` only on CUDA.

---

## 5. Real Pair Construction Today (no IMD key needed)

Since **Open-Meteo + NASA POWER are both free and returned 200**, the fastest *real-data* bias dataset without waiting on IMD is:

**`matched_pairs_daily.csv` via:**
- Left: `Open-Meteo Archive (ERA5) daily aggregations` or `Archive hourly → daily sum/max`
- Right: `NASA POWER daily T2M, PRECTOTCORR` for same `lat/lon, date`
- Align by `date`, compute `bias = power − era5`

This is **not** GFS→AWS but is **reanalysis↔independent satellite observation** — still learns systematic offset/elevation correction and is strictly more real than Gaussian. Swap to `GFS→IMD` when key arrives (same script, different endpoints).

Historical GFS forecast archive via `https://historical-forecast-api.open-meteo.com` will be probed after your approval; if available it gives true `GFS forecast vs ERA5 truth` pairs back to 2022.

---

## 6. Recommended Order (P100 kernel)

1. **M1 semantic** — expand `field_names.csv` to 1.5k, train 5 epochs on P100 → **ship first** (utility-high, low data risk).
2. **M2 bias** — run `scripts/build_bias_pairs.py` for say 20 Indian points ×365 days = ~7k daily or ~175k hourly; validate, then train MLP 20 epochs.
3. **M3 intent** — generate 1k diverse intents via Groq paraphrase, train 3 epochs.

**Dry-run vs real-run gate:**
- `train --dry-run` = 5s smoke, stays **synthetic and never considered a real training result** (metrics are `1.0` fake).
- `--no-dry-run` without real files currently falls back to Gaussian/6-template and **must be treated as placeholder**, not publication result.

---

## 7. Decision Needed From You

- [ ] Approve expanding **M1 to 1.5k** and building **M2 `matched_pairs.csv` via Open-Meteo + POWER** (≈10–15 min fetch, ~20k rows) before any P100 train? (Recommended)
- [ ] Or: want a different **M2 gold source** (e.g., wait for IMD, use Kaggle `global-weather-repository` daily, or another)?
- [ ] Should **M3** be built with **Hinglish 20%** now, or pure English first?

Reply `go` + which M2 path (POWER vs wait vs other) and I’ll build the datasets and then push the new P100 kernel — no Gaussian/6-template train will be launched until then.
