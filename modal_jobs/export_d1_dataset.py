"""Publish the D1 corpus (M2/M4/M5's training data) to the Hugging Face Hub.

D1 is `weathergpt-data/d1_mos/*.parquet` on Modal -- 127 per-location shards,
one row per (location, valid_time), with each of 4 NWP models' forecasts and
ERA5-Land truth for 4 variables. M2 (bias correction) and M4 (calibration)
both trained on this exact corpus. M5 (trust ranker) trained a day earlier on
a slightly smaller snapshot (9,507,456 rows vs the current 9,582,912) that
was since overwritten by a later rebuild -- this upload is the corpus M2/M4
trained on, and is not byte-identical to what produced M5, which the README
discloses rather than papers over.
"""
from __future__ import annotations

import json

import modal

from modal_jobs.common import DATA_DIR, DATA_IMAGE, VOLUMES, app

HF_SECRET = modal.Secret.from_name("arko007-hf-token")
EXPORT_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "httpx==0.28.1", "pandas==2.2.3", "pyarrow==18.1.0", "numpy==2.1.3",
    "huggingface_hub==0.26.5",
).add_local_python_source("modal_jobs", "app", "weathergpt_models")

README = """---
license: cc-by-4.0
tags:
  - weather
  - meteorology
  - nwp
  - india
  - forecasting
  - tabular
size_categories:
  - 1M<n<10M
---

# WeatherGPT D1 — multi-model NWP forecasts vs. ERA5-Land truth (India)

The training corpus for **M2** (distributional bias correction) and **M4**
(precipitation calibration) in the [WeatherGPT](https://github.com/Anamitra-Sarkar/weathergpt)
project (SIH 2026). One row per (location, valid time): four NWP models'
forecasts at that point in time, and what ERA5-Land actually observed there.

**M5** (trust ranker) trained a day earlier on a slightly smaller snapshot
of this same pipeline (9,507,456 rows vs. this upload's 9,582,912) that was
overwritten by a later rebuild before this dataset was captured -- so this
is the exact corpus M2 and M4 trained on, and is extremely similar to but
not byte-identical to what produced M5. Disclosed here rather than glossed
over.

## What's in it

- **9,582,912 rows**, one parquet file per location, **127 locations**
  across India (real geocoded district headquarters), **127 shards** total.
- Source: [Open-Meteo](https://open-meteo.com)'s historical-forecast API for
  the 4 NWP models, and its ERA5-Land archive for truth. Real API calls,
  real forecast/truth pairs -- not synthetic.

| column | meaning |
|---|---|
| `loc_id`, `lat`, `lon`, `elevation_m`, `admin1` | the location |
| `valid_time`, `hour_utc`, `doy`, `month` | when this forecast/observation is for |
| `lead_hours`, `lead_age_days` | how far ahead the forecast was made |
| `fc_<variable>_<model>` | forecast value, for `variable` in `{{temperature_2m, precipitation, wind_speed_10m, relative_humidity_2m}}` and `model` in `{{gfs_seamless, ecmwf_ifs025, icon_seamless, gem_seamless}}` (16 columns) |
| `truth_<variable>` | ERA5-Land observed value for that variable (4 columns) |
| `chunk_misses` | how many API chunks failed for this location during the build (counted, not silently dropped) |

## How M2/M4 actually used it

Loaded whole (`modal_jobs/features_d1.py::load_d1`), then per target variable
(`temperature_2m`, `precipitation`, `wind_speed_10m`):

- **Split**: chronological 70% cutoff (`{cutoff}`) for train/val, AND 20% of
  locations (`{n_held_out}` of 127) held out entirely for a genuinely unseen
  spatial test set -- so the reported skill isn't just "interpolating a
  location it already trained on with different timestamps."
- **Per-target usable rows** (a row is usable only if that variable's truth
  and all 4 forecasts are present): train **{train_rows:,}**,
  val **{val_rows:,}**, test (spatially held out) **{test_rows:,}** --
  identical across all 3 target variables since they're reshaped from the
  same row set.
- **30 input features** per row, built by
  [`weathergpt_models/features.py::assemble_features`](https://github.com/Anamitra-Sarkar/weathergpt/blob/main/weathergpt_models/features.py)
  -- the same function training and inference both call, so there's no
  train/serve skew: the target variable's 4 raw forecasts, 7 ensemble summary
  stats (mean/sd/min/max/median/spread ratio/wet fraction), mean+sd of the 3
  other variables as cross-predictors, 9 time/location features (lead hours,
  lead age, cyclic hour-of-day and day-of-year, elevation in km, lat, lon),
  and 4 per-model missingness flags.
- **Target**: the corresponding `truth_<variable>`, with precipitation fit in
  cube-root space (exact quantile-preserving transform, not an approximation)
  because hourly Indian rainfall is a spike at zero with a long monsoon tail.

## Quickstart

```python
import pandas as pd
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="Arko007/weathergpt-d1-mos-dataset", repo_type="dataset")
import glob
frame = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{{path}}/d1_mos/*.parquet"))
print(frame.shape, frame.columns.tolist())
```

## Full context

Model cards, held-out metrics and baselines for M2/M4 (trained on this data)
are at [`Arko007/weathergpt-models`](https://huggingface.co/Arko007/weathergpt-models).
Integration and real-world-usability notes are in
[`docs/MODEL_REGISTRY_INTEGRATION.md`](https://github.com/Anamitra-Sarkar/weathergpt/blob/main/docs/MODEL_REGISTRY_INTEGRATION.md)
and [`docs/REAL_WORLD_READINESS.md`](https://github.com/Anamitra-Sarkar/weathergpt/blob/main/docs/REAL_WORLD_READINESS.md).
"""


@app.function(image=EXPORT_IMAGE, volumes=VOLUMES, secrets=[HF_SECRET],
              timeout=60 * 30)
def export(repo_id: str, private: bool = False) -> dict:
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    from modal_jobs.features_d1 import build_features, load_d1, split_masks

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_ARKO007")
    if not token:
        raise RuntimeError("no HF token found in the arko007-hf-token secret")

    root = Path(DATA_DIR) / "d1_mos"
    shard_paths = sorted(root.glob("*.parquet"))
    if not shard_paths:
        raise RuntimeError(f"no shards found at {root}")

    frame, files = load_d1(DATA_DIR)
    masks = split_masks(frame, seed=42)
    X, y, names, members, keep = build_features(frame, "temperature_2m")
    train_rows = int((masks["train"] & keep).sum())
    val_rows = int((masks["val"] & keep).sum())
    test_rows = int((masks["test"] & keep).sum())

    readme = README.format(
        cutoff=str(masks["cutoff"]), n_held_out=len(masks["held_out_locations"]),
        train_rows=train_rows, val_rows=val_rows, test_rows=test_rows)

    api = HfApi(token=token)
    who = api.whoami()
    print(f"[export_d1] uploading as {who.get('name')}, {len(shard_paths)} shards, "
          f"{len(frame):,} total rows")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="dataset")
    api.upload_folder(folder_path=str(root), repo_id=repo_id, repo_type="dataset",
                      path_in_repo="d1_mos", commit_message="Upload D1 MOS corpus")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset", commit_message="Dataset card")

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[export_d1] published {url}")
    return {"url": url, "account": who.get("name"), "total_rows": int(len(frame)),
            "n_shards": len(shard_paths), "train_rows": train_rows,
            "val_rows": val_rows, "test_rows": test_rows}


@app.local_entrypoint()
def main_export_d1(repo_id: str = "Arko007/weathergpt-d1-mos-dataset", private: bool = False):
    result = export.remote(repo_id, private)
    print(json.dumps(result, indent=2))
