# WeatherGPT — Meteorological Interoperability + Agentic Retrieval Layer

> **Name: WeatherGPT** (general weather intelligence, not farmer-only). This repo implements the **AI/ML + Backend** core of the WeatherGPT problem: a query-driven interoperability layer that turns heterogeneous point/grid/raster/polygon, observation/forecast/warning/satellite/radar/reanalysis sources into provenance-preserving, temporally-aligned, spatially-matched, uncertainty-aware *evidence* that an LLM can safely explain.

Built from your two briefs:
- **Survey of Recent Advances (2024-2026)** — agentic hub-and-spoke, GenCast ensembles/Brier, Indic ASR/TTS, offline-first, privacy
- **WeatherGPT Data Fragmentation** — 8 normalization problems + Canonical Evidence Object + Weather Intelligence Object

---

## 1) What Gets Trained (and Why Not a Global Weather Model)

We **do not** train a new global NWP. We train small, auditable models that make heterogeneous sources **comparable + trustworthy**:

| # | Model | Task | Input → Output | Trainable on Kaggle T4 / CPU |
|---|-------|------|----------------|------------------------------|
| M1 | **Semantic Variable Classifier** (`training/train_semantic_classifier.py`) | Map raw source field names/descriptions (`APCP`, `rainfall`, `rain_rate`, `PoP`, `heavy-rain warning`) → canonical `variable + statistic + accumulation_window` | text → canonical label | ✅ BERT-mini + synthetic data, 5 min on T4, CPU also fine |
| M2 | **Bias-Correction / Downscaler** (`training/train_bias_correction.py`) | Learn `GFS coarse grid → IMD AWS station` residual for temperature & precipitation; also handles unit/accumulation mismatch | `[gfs_t2m, gfs_apcp, elevation, lead_hours, lat/lon]` → `bias` | ✅ MLP (PyTorch) or LightGBM, 2–10 min |
| M3 | **Intent / Time-Window Parser** (`training/train_intent_parser.py`) | User utterance → structured intent `{variables, evidence_classes, valid_from/to, decision}` | text → JSON slots | ✅ DistilBERT + rules, fine-tunable |

All three have **CPU fallback** — they auto-detect `cuda` and fall back to `cpu`. Kaggle notebook runs end-to-end on `T4 x2` or plain CPU.

---

## 2) Quick Start

### Local (CPU)

```bash
git clone <this-repo> weathergpt
cd weathergpt

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# optional: heavier scientific stack only if you need GRIB/NetCDF/HDF5 decoding
pip install -r requirements-full.txt

# run backend (no keys needed — mock mode)
uvicorn app.main:app --reload --port 8001
# → http://localhost:8001/docs
# → http://localhost:8001/health

# run training smoke tests (no download, synthetic data)
python training/train_semantic_classifier.py --dry-run --epochs 1
python training/train_bias_correction.py --dry-run --epochs 1 --model mlp
python training/train_intent_parser.py --dry-run --epochs 1
```

### Kaggle — T4 x2 (recommended)

1. Create new Kaggle Notebook → Settings → Accelerator: **GPU T4 x2**, Internet: **On**, Persistent: **On**
2. Upload this repo as dataset OR `!git clone https://github.com/<you>/weathergpt.git`
3. Open `training/notebooks/weathergpt_training.ipynb` → **Run All**
4. Outputs saved to `/kaggle/working/weathergpt_outputs/` + `training/models/`
5. Download artifacts: `File → Download` or copy to Kaggle Dataset

See `training/notebooks/weathergpt_training.ipynb` for the exact cell order.

### Docker

```bash
docker build -t weathergpt:latest .
docker run -p 8001:8001 --env-file .env weathergpt:latest
```

---

## 3) Environment Variables

Copy `.env.example` → `.env`:

```bash
cp .env.example .env
```

| Var | Required | What |
|-----|----------|------|
| `GROQ_API_KEY` | for LLM explainer only | Groq `openai/gpt-oss-120b` or `llama-3.3-70b` — if missing, backend runs in mock-explainer mode |
| `GROQ_MODEL` | no | default `llama-3.3-70b-versatile` |
| `OPENWEATHER_API_KEY` | no | legacy fallback only |
| `IMD_API_KEY` / `IMD_API_BASE` | no | when you have IMD portal access — decoders work with fixtures until then |
| `HF_TOKEN` | no | only for optional HF model pulls |

**Mock mode is the default.** Backend boots and returns `WIO` without any keys. Set keys when ready — no code change needed.

---

## 4) API — Backend

```
GET  /health
POST /wio/query        → Weather Intelligence Object (no LLM)
POST /query            → WIO + LLM explanation + warnings
POST /rade/advise      → RADE decision {action, expected utility}
GET  /evidence/{id}
GET  /warnings/active?district=Nagpur
POST /train/semantic   # optional: trigger training via API (background job)
```

Example:

```bash
curl -X POST http://localhost:8001/wio/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Will it rain in Nagpur tomorrow afternoon and should I spray pesticide?",
       "location":{"raw":"Nagpur"},
       "lang":"en"}' | jq
```

Response is a `WeatherIntelligenceObject` (see `app/schemas/wio.py`) — **the only thing the LLM ever sees**.

Full docs: `http://localhost:8001/docs` (OpenAPI).

---

## 5) Training — Kaggle T4 Ready

### What Kaggle runs

`training/notebooks/weathergpt_training.ipynb` runs three pipelines sequentially:

1. **Cell 1–3**: env check (`torch.cuda.is_available()`), install `requirements-kaggle.txt` if needed, set `device`
2. **Semantic Classifier** — synthetic + real field-name corpus → `distilbert-base` finetune → `training/models/semantic_classifier/`
3. **Bias Correction** — synthetic `GFS→AWS` pairs (replace with `training/datasets/` when you upload real IMD/ERA5) → `MLP` on T4 → `training/models/bias_correction/`
4. **Intent Parser** — utterance→slots dataset → `DistilBERT` token classifier → `training/models/intent_parser/`

Each script supports:

```bash
python training/train_semantic_classifier.py --epochs 5 --batch-size 32 --device auto
python training/train_bias_correction.py --model mlp --epochs 20 --batch-size 256 --device cuda
python training/train_intent_parser.py --epochs 3 --device auto

# dry-run (no HF download, 5s smoke test — used in CI)
python training/train_semantic_classifier.py --dry-run
```

Outputs:
- `training/models/<model>/best.pt` + `config.json` + `metrics.json`
- `weathergpt_outputs/` summary for Kaggle download

### Using Your Own Data on Kaggle

Upload any of these to `training/datasets/` and the scripts auto-detect them:

- `training/datasets/imd_aws.csv` — `station_id, lat, lon, observed_at, t2m_c, apcp_mm`
- `training/datasets/gfs_history.csv` — `lat, lon, model_init, valid_from, t2m_k, apcp_kgm2`
- `training/datasets/field_names.csv` — `raw_field, canonical_variable, statistic, accumulation_hours`
- `training/datasets/intent_samples.jsonl` — `{"text":"Will it rain...","intent":{"variables":["precipitation_amount"],...}}`

If absent, synthetic data is generated (so the notebook never fails on first run).

### CPU vs T4

All scripts:

- `--device auto` (default) → `cuda` if available else `cpu`
- Kaggle `T4 x2` uses `DataParallel` automatically when `torch.cuda.device_count()>1`
- Mixed precision (`--amp`) available for T4, auto-disabled on CPU
- CPU run is ~3–5× slower but completes; reduce `--epochs` for smoke tests

---

## 6) Project Structure

```
weathergpt/
  app/
    main.py                 # FastAPI entry
    schemas/{ceo,wio,advice,location}.py
    decoders/{imd_json,grib2,netcdf,hdf5,cap,wis2_mqtt}.py
    services/{variable_registry,temporal_align,spatial_match,semantic_gate,ranker,wio_builder,location_resolver,time_parser}.py
    orchestrator/{retrieval_planner,agent_manager}.py
    rade/{enumerator,utility,policy,revision}.py
  training/
    train_semantic_classifier.py
    train_bias_correction.py
    train_intent_parser.py
    configs/{semantic,bias_correction,intent}.yaml
    datasets/               # drop your real data here (git-ignored)
    models/                 # trained artifacts (git-ignored)
    notebooks/weathergpt_training.ipynb  # Kaggle T4 entry point
  tests/
  requirements*.txt
  Dockerfile / docker-compose.yml
  .env.example
```

---

## 7) Metrics (per your brief)

- **Semantic classifier**: accuracy / F1 per canonical variable, confusion on `rain_rate vs accumulation`
- **Bias correction**: `RMSE`, `MAE`, `Brier Score` (for `rain>0.5mm`), `CRPS` for ensemble
- **Intent parser**: slot-F1 for `location/time/variables`
- Backend: `/metrics` exposes them from `metrics.json`

---

## 8) Installation Guide — Full Dependency Notes

**Minimal install** (`requirements.txt`): FastAPI, Pydantic, torch (CPU), transformers, scikit-learn, pandas, pint, shapely — enough for training + API without scientific decoders.

**Full scientific stack** (`requirements-full.txt`): adds `cfgrib`, `eccodes`, `netCDF4`, `h5py`, `pyproj`, `paho-mqtt`. Install only if you need GRIB2/NetCDF/HDF5 decoding locally; Kaggle image has them preinstalled.

If `eccodes` fails locally, use `conda`:

```bash
conda install -c conda-forge eccodes cfgrib netcdf4 h5py
```

---

## 9) Next Steps After Kaggle Train

1. Copy `training/models/*` back to local `training/models/`
2. Backend auto-loads them if present: `MODEL_DIR=training/models uvicorn app.main:app --reload`
3. Run backtest: `python scripts/backtest.py --wio-endpoint http://localhost:8001/wio/query`

---

## License
MIT — for SIH 2026 / internal research use. Replace with your org license before public release.
