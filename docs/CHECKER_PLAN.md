# Checker Findings -> Processing Plan (Kaggle P100, env analysed)

## Environment (checker kernel on P100)
- Python 3.12.13, torch 2.10.0+cu128, CUDA available True, GPU Tesla P100-PCIE-16GB 16384 MiB, capability (6,0)
- **Critical:** P100 sm_60 is INCOMPATIBLE with torch 2.10 (needs sm_70+). All training must be forced to CPU (`FORCE_CPU=1`, `--device cpu`). No GPU speedup, but still trainable (MLP small). To use GPU, downgrade torch to 2.0+cu118 or use `pytorch-light`? For now we accept CPU.

## Datasets Found / Rebuilt
- No pre-existing datasets in /kaggle/input, so checker rebuilt all 3 fresh on Kaggle (no local download, as requested).
- field_names 1500 rows, matched_pairs 4800 rows, intent 1200 rows — same as our earlier P100 run, but now audited.

## Verdict per Dataset (from report.json)

### M1 field_names.csv: NEEDS PROCESSING
- dup_rows 1269 / 1500 (84% duplicates) due to 8 variants per raw_field repeated many times with just case/space noise.
- extra_labels_vs_expected_6: wind_gust, temperature_min, temperature_max (3 extra) vs LABEL_MAP 6.
- sample_TMAX_mm 6 (unit error), acc_hours only 6 (no 1,3,24)
- per_canonical 411 vs 37 imbalance

### M2 matched_pairs.csv: NEEDS PROCESSING (minor)
- rows 4800, corr_t2m 0.96 (good, not 1.0 after fix with models=gfs_seamless), corr_precip 0.42 (real)
- bias_t_mean -1.41°C, std 2.05°C (real bias, not zero)
- lead_hours 0-239 sequential is time-index not forecast lead — must recompute.
- No missing, no scaling yet.

### M3 intent_samples.jsonl: CHECKER BUG (M3_error name 'ch' is not defined) — checker script had bug in pincode line, but from earlier local audit we know:
- none 86% imbalance, need upsampling.

## Processing Plan (to be executed ON KAGGLE before training)

### M1: Dedup + Label Fix + Rebalance + Augment
1. Dedup on lower(raw_field) keep first, expect ~231 unique -> expand with richer augmentations to reach ~800-1000 deduped (not 1500 with dups).
2. Fix LABEL_MAP: expand to 9 labels in code + config (num_labels 9) OR collapse tmax/tmin->temperature_2m and wind_gust->wind_speed. Decision: expand to 9 (more faithful to WeatherGPT).
3. Fix TMAX (mm) -> TMAX (°C) and add accumulation variants.
4. Stratified split, class_weight in loss.

### M2: Lead Fix + Scaling + Time-aware Split
1. Recompute lead_hours = (valid_from - init_time) where init_time is historical forecast init (we approximated as day start, need to store init). Simplest: lead = hour_of_day % 24? But checker shows 0-239 is just index; we should set lead = i % 24 or i % 72 (GFS 0-72h). For now set lead = i % 72.
2. StandardScaler fit on train only (5 features), save scaler.pkl.
3. Time-aware split: train 2024-01-01:07, val 2024-01-08:10 (not random), spatial hold-out 4 points for test.
4. Keep elevation scaling.

### M3: Rebalance + Dedup
1. Fix checker bug, then upsample minority decisions (pesticide/marine/irrigation/harvest) 3x, dedup texts, keep 1200 but balanced ~40% none, 60% others.
2. Use class_weight in Trainer.

## Next: Push processor kernel that applies these fixes ON KAGGLE, then trains.

