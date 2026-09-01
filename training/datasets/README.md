# Datasets — drop real data here (git-ignored)

All training scripts **work without any real data** (synthetic fallback). Drop any of these files to train on real weather instead:

| File | What it should contain |
|------|------------------------|
| `imd_aws.csv` | `station_id, lat, lon, observed_at, t2m_c, apcp_mm, ...` — IMD AWS/ARG station obs |
| `gfs_history.csv` | `lat, lon, model_init, valid_from, t2m_k, apcp_kgm2, elevation, lead_hours` — GFS coarse history for bias correction |
| `field_names.csv` | `raw_field, canonical_variable, statistic, accumulation_hours` — e.g. `APCP,precipitation_amount,accumulation,6` |
| `intent_samples.jsonl` | `{"text":"Will it rain...","intent":{"variables":["precipitation_amount"],"time":"tomorrow afternoon","location":"Nagpur","decision":"pesticide_spraying"}}` |
| `imd_samples.jsonl` | IMD fixture lines: each line a JSON record with `_product` e.g. `{"_product":"warning","district":"Nagpur","hazard":"heavy rainfall","colour":"orange","valid_from":"2026-09-01T00:00:00+0530",...}` |

If files are absent, each `train_*.py` generates synthetic data so the Kaggle notebook never fails on first run.

Tip for Kaggle: `Add Data → Upload → training/datasets/` or `!wget` your own storage.
