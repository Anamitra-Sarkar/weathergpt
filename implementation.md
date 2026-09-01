# WeatherGPT — Implementation

**Codebase:** `/home/anamitra/weathergpt` — `app/` 276 KiB, `training/` 748 KiB, `kaggle_kernel_official/official_train.py:504`

## 1. Schemas

* `app/schemas/ceo.py:80` — `CanonicalEvidenceObject` `model_config protected_namespaces ()`, 22 fields, `use_enum_values`
* `app/schemas/wio.py:30` — `WeatherIntelligenceObject` + `QueryRequest/Response`
* `app/schemas/location.py:10` — `GAZETTEER` 8 cities hard-coded for offline

## 2. Services (8 Normalization Layers)

* `services/variable_registry.py` — `DEFAULT_REGISTRY` 30+ `raw→canonical`, `gold_pairs()` for M1, `are_comparable()` semantic gate
* `services/temporal_align.py` — `to_utc`, `overlaps`, `filter_by_window`, `staleness_hours`, `is_stale 48h`
* `services/spatial_match.py` — `haversine_km`, `distance_to_query` (Point/GridCell), `rank_by_spatial`
* `services/semantic_gate.py` — `filter_comparable` keeps `warning/advisory` separately
* `services/ranker.py` — `AUTHORITY` dict, `score_evidence`, `rank`, `detect_disagreements 10mm`
* `services/location_resolver.py` — regex pincode `^\d{6}$`, `lat,lon`, Gazetteer `in` match, fallback Nagpur `0.4`
* `services/time_parser.py` — `parse_time_window` `today/tomorrow/afternoon 12–18 IST/night/next 3 days/weekend` → `valid_from/to + horizon nowcast/short/medium/climate`
* `services/wio_builder.py` — `best_by_var`, `rain prob` from `precipitation_probability` CEO, `WIOWarning` highest severity `green<yellow<orange<red`, `agreement` full/partial/single

## 3. Decoders

* `decoders/imd_json.py` — `_parse_time` tries `+0530`, `decode_city_forecast` `rainfall→acc 24h`, `decode_warning` `colour→warning_severity`, `decode_nowcast` `category`, `decode_rainfall` `actual→acc 24h`, `decode(product)` table
* `decoders/open_meteo.py` — `OPEN_METEO_FORECAST` + `decode_open_meteo()` hourly `precip/temp/wind` → `1h accumulation` CEOs
* `decoders/cap_decoder.py` — `xml.etree` `cap:1.2` NS, `status/msgType Cancel→cancelled`, `severity→colour green/yellow/orange/red`, `areaDesc` join
* `decoders/grib2_placeholder.py` — `xarray cfgrib` `filter_by_keys typeOfLevel surface shortName 2t/tp`, `sel lat/lon nearest`, `K→C`

## 4. Orchestrator

* `orchestrator/retrieval_planner.py` — `HORIZON_PLAN{nowcast:[observation,nowcast,warning,radar], short:[observation,forecast,warning,radar], medium:[forecast,warning,climate]}` → `sources_for_classes()`; `app/orchestrator/models.py:5` 4-model queue `qwen/qwen3.8-27b` orchestrator, no llama.
* `orchestrator/models.py` — `PREFERRED_MODELS 4`, `ORCHESTRATOR_MODEL qwen3.8`, `AGENT_ROLES 8`, `model_for(role|int)` round-robin, queues for 5–6 agents.
* `orchestrator/groq_client.py` — `GROQ_API https://api.groq.com/openai/v1/chat/completions`, `_key()` tries env or `groq_api.txt`, hard-injected in Kaggle, queue for 5–6 agents.

## 5. RADE

* `rade/enumerator.py` — `p = rain.probability or 0.5`, `scenarios [{rain p, precip},{no_rain 1-p}]`
* `rade/utility.py` — `utility(spray,rain)=-30`, `wait rain=10`, `irrigate no_rain=15`, `expected_utility = Σ p*U`, `all_actions 5`
* `rade/policy.py` — `select_policy(WIO)` `max(scores)`, `explain_policy()` `→ “Wait: 70% wash-off”`.

## 6. Backend

* `app/main.py` — `FastAPI` `CORS *`, `load_dotenv`, `_collect_ceos` (Open-Meteo live + `imd_samples.jsonl` fixtures), `_filter_ceos_to_window`, `POST /wio/query` builds `WIO` (falls back `ceos[:20]` if filtered empty), `POST /query` adds `RADE` + `groq_client.generate(WIO JSON, explainer_agent)` or mock `Advice: best — note + Agreement + Evidence N + mock`, `POST /rade/advise`

## 7. Training

* `training/train_semantic_classifier.py:232` — `LABEL_MAP 9`, `SYNTHETIC_PAIRS 35`, `make_dataset` rich suffix + dedup, `load_external_csv`, `run_dry`, `run_full` `DistilBERT 8ep` `WeightedTrainer` `class_weights`, `stratified 85/15`, `Trainer processing_class`
* `training/train_bias_correction.py:207` — `MLP 5→64→64→2`, `synth_data 10k Gaussian`, `load_real` prefers `matched_pairs.csv` (fixes `lead %72`, `gfs K→C bias`), `train_mlp` `StandardScaler` time-aware `85/15 tail`, `Adam`, `GradScaler`, `rmse + Brier`, `train_lgbm` 2 models
* `training/train_intent_parser.py:180` — `UTTERANCES 6`, `rule_parse`, `synth_jsonl`, `label_map 5`, `rebalance dedup + upsample 3×`, `WeightedTrainer`, `P100 sm_60 → cpu` fallback
* `training/configs/*.yaml` — `semantic 9/64/5ep 32bs 2e-5`, `bias mlp 64 3 layers 20ep 256bs`, `intent 128/3ep 32bs`
* `kaggle_kernel_official/official_train.py:504` — single-file T4 x2 builds `1200`/`14400`/`2000 Groq 800` on Kaggle, trains `M1 8ep, M2 30ep Deep 128→128→64, M3 5ep` all with `use_cpu=force_cpu` on P100, `DataParallel` on 2 GPUs, `Groq 4-model queue` for M3 paraphrase

## 8. Tests

* `tests/test_ceo.py:3` — `test_comparable_gate`, `test_ceo_roundtrip`, `test_wio_preserves_warning`
* `tests/test_decoders.py:3` — `test_imd_city`, `test_imd_warning`, `test_cap`
* `pyproject.toml` `pytest` `testpaths tests`

## 9. Key Fixes 2026-09-01

* `LABEL_MAP 6→9`, `TMAX (mm)→°C`, `acc 6→1,3,6,24`
* `lead 0-239→%72`, `StandardScaler` + `time-aware`, `gfs_seamless` (corr 1.0→0.96)
* `none 86%→20%` + `class_weight`
* `tokenizer→processing_class`, `evaluation_strategy→eval_strategy`, `no_cuda→use_cpu=force_cpu` (transformers 4.46+)
* `M1 pad infinite loop` (`while 900` with 64 unique → 500s) → broader vocab + `attempts<5000`

See `setup.md` for install and `report.md` for why ML vs LLM.
