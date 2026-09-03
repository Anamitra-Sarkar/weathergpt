"""Shared Modal app, images and volumes for the WeatherGPT data foundry.

Everything in `modal_jobs/` runs remotely.  The local machine never downloads a
dataset or a checkpoint; artifacts live on the two Modal Volumes below.
"""
from __future__ import annotations

import modal

APP_NAME = "weathergpt"

# --- volumes -----------------------------------------------------------------
DATA_VOL = modal.Volume.from_name("weathergpt-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("weathergpt-models", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("weathergpt-hfcache", create_if_missing=True)

DATA_DIR = "/data"
MODEL_DIR = "/models"
HF_CACHE_DIR = "/hfcache"

VOLUMES = {DATA_DIR: DATA_VOL, MODEL_DIR: MODEL_VOL}
TRAIN_VOLUMES = {DATA_DIR: DATA_VOL, MODEL_DIR: MODEL_VOL, HF_CACHE_DIR: HF_CACHE_VOL}

# --- images ------------------------------------------------------------------
# Data building: pure-python + arrow.  No torch, so the image stays small and
# cold-starts fast for the long network-bound fetch jobs.
DATA_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "httpx==0.28.1",
        "pandas==2.2.3",
        "pyarrow==18.1.0",
        "numpy==2.1.3",
        "lxml==5.3.0",
        "pydantic==2.10.3",
        "pyyaml==6.0.2",
    )
    .add_local_python_source("modal_jobs", "app")
)

# Training: torch + transformers + gradient boosting.  Pinned so a rerun in a
# week reproduces the same numbers.
TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface-hub==0.26.5",
        "sentence-transformers==3.3.1",
        "datasets==3.1.0",
        "accelerate==1.2.1",
        "seqeval==1.2.2",
        "scikit-learn==1.5.2",
        "lightgbm==4.5.0",
        "scipy==1.14.1",
        "pandas==2.2.3",
        "pyarrow==18.1.0",
        "numpy==2.1.3",
        "matplotlib==3.9.3",
        "onnx==1.17.0",
        "onnxruntime==1.20.1",
        "httpx==0.28.1",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("modal_jobs", "app")
)

# GRIB2 decoding needs the ECMWF eccodes C library; only this image carries it.
GRIB_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libeccodes0", "libeccodes-dev")
    .pip_install(
        "eccodes==2.38.3",
        "cfgrib==0.9.14.1",
        "xarray==2024.11.0",
        "numpy==2.1.3",
        "httpx==0.28.1",
        "fastapi[standard]==0.115.6",
    )
    .add_local_python_source("modal_jobs", "app")
)

app = modal.App(APP_NAME)

GROQ_SECRET = modal.Secret.from_name("groq-api-key")
