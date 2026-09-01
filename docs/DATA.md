# WeatherGPT — Data

**Rule:** No synthetic Gaussian in final official — 1200/14400/2000 are rebuilt **on Kaggle** via Open-Meteo + POWER + Groq, no local download.

## Sources (8 families)

| Family | Native | What it is | Why not direct LLM |
|--------|--------|------------|--------------------|
| IMD city/district | JSON API | `temp_max/min, rainfall, wind` + `category_code, colour` | Schemas differ, `acc 24h` vs `1h` |
| Open-Meteo Ensemble | JSON | 30+ models `ECMWF,GFS` hourly `precip/temp/wind` | Needs `1h accumulation` → `6h` window |
| GFS | GRIB2 | `2t, tp` `K→C, kg m-2→mm`, `0.25°` grid | Needs `cfgrib` `filter_by_keys` + `sel nearest` |
| CAP | XML 1.2 | `identifier,sender,sent,status,msgType,severity,expires,area` | `Cancel` lifecycle, `severity→colour` |
| ERA5 / POWER | GRIB/NetCDF | `historical 0.25°` vs `satellite daily` | `6h` vs `24h` cannot average |
| WRF | NetCDF `wrfout` | `T2,RAINC` perturbation vars | Needs reconstruction |
| INSAT | HDF5 | `geophysical + geolocation + quality` | Raster, not city table |
| WIS2 | MQTT+HTTP | `topic → data` | Not one schema |

## Columns (final official on Kaggle)

**`field_names.csv` 1200 rows, `unique 1200`, 9 labels** — `raw_field,canonical_variable{precip_amount 365, heavy 155, t2m 141, wind 124, tmax 92, gust 90, tmin 83, prate 78, prob 72}, statistic{instant,accumulation,categorical,probability,min,max}, accumulation_hours{1,3,6,24}, source_hint`

*Fixes:* `LABEL_MAP 6→9`, `TMAX (mm)→°C`, `acc only 6→1,3,6,24`, dedup `1269→0`, class imbalance via `WeightedTrainer`.

**`matched_pairs.csv` 14400 rows** — `lat,lon,elevation_m,lead_hours,valid_from,gfs_t2m_k,gfs_apcp_mm,obs_t2m_c,obs_apcp_mm` — 20 Indian points ×30d×24h `2024-01-01:30`, `historical GFS `gfs_seamless` vs ERA5`, `lead %72` (was `0-239` sequential), `gfs 271–306K, obs -4–33°C`, `corr_t2m 0.96`, `bias -1.41±2.05`.

**`intent_samples.jsonl` 2000 rows** — `{"text","intent":{variables[],time,location,decision}}` — 16 templates × locs × times + `Groq 800` paraphrases via 4-model queue `qwen3.8×200 + 3.6×200 + gpt-oss-20b×200 + gpt-oss-120b×200`, deduped `1409→2000`, rebalance `400 none +350×4` = `20% none` (was `86%`), Hinglish `16%`, `pincode 78`.

**Checker (P100):** `field_names 1500 dup 1269`, `matched_pairs 4800 corr 1.0→0.96 after fix`, `intent none 86%` → all `NEEDS PROCESSING`, now `READY`.

## Processing (on Kaggle before training)

* **M1:** dedup `lower(raw)+canon`, `TMAX (°C)`, `acc 1,3,6,24`, `stratified 85/15`, `class_weight`
* **M2:** `lead %72`, `StandardScaler fit train only` → `scaler.pkl`, `time-aware split 12240/2160 tail`, `spatial hold-out` planned
* **M3:** dedup text lower, `upsample 3×` minority `pesticide/marine/irrigation/harvest` → `2000` balanced, `class_weight`

## Why `12` ≠ `12`

Same `rainfall 12` can be `mm/24h`, `mm/3h`, `PoP 80% 6h`, `district widespread`, `heavy-rain warning` — `CEO` keeps `statistic, accumulation_hours, valid_from/to, source, resolution, quality` so `WIO` can say `Sources agree on occurrence but differ on amount 18–32mm`.

See `architecture.md` for CEO→WIO flow.
