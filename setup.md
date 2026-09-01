# WeatherGPT — Setup

**Repo:** `/home/anamitra/weathergpt` — `app/` + `training/` + `kaggle_kernel_official/official_train.py:504`

## 1. Local CPU (no GPU, no keys, mock WIO works)

```bash
git clone <this-repo> weathergpt
cd weathergpt
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # no eccodes
# pip install -r requirements-full.txt  # for GRIB2/NetCDF/HDF5 (cfgrib+eccodes) — needs conda: conda install -c conda-forge eccodes cfgrib
uvicorn app.main:app --reload --port 8001  # http://localhost:8001/docs
# http://localhost:8001/health  → mock_mode True
curl -X POST http://localhost:8001/wio/query -H "Content-Type: application/json" \
  -d '{"question":"Will it rain in Nagpur tomorrow afternoon?","location":{"raw":"Nagpur"}}' | jq
```

**Test (5s, no HF):**
```bash
python training/train_semantic_classifier.py --dry-run
python training/train_bias_correction.py --dry-run
python training/train_intent_parser.py --dry-run
pytest tests/ -v  # 6/6
```

## 2. Local with Groq (live LLM, still no IMD key needed)

```bash
cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_...  GROQ_MODEL=qwen/qwen3.8-27b
# Also reads /home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt if env missing
uvicorn app.main:app --reload --port 8001
# POST /query now uses qwen/qwen3.8-27b orchestrator + 4-model queue (no llama), WIO only
```

**Verify 4-model queue:**
```bash
PYTHONPATH=. python3 -c "from app.orchestrator.models import describe_routing; print(describe_routing())"
# orchestrator → qwen/qwen3.8-27b, intent→qwen3.8, location→qwen3.6, forecast→gpt-oss-20b, history→gpt-oss-120b ...
```

## 3. Kaggle T4 x2 (official best-ever, internet ON)

* Create Notebook → **Accelerator: GPU T4 x2**, **Internet: ON**
* Upload `weathergpt` or `!git clone https://github.com/arkosarkarhehe/weathergpt.git && %cd weathergpt`
* Open `training/notebooks/weathergpt_training.ipynb` → **Run All** (or `kaggle_kernel_official/official_train.py` single-file)

**What it does on Kaggle (no local download):**
* Builds `field_names.csv 1200` (9 labels, `acc 1,3,6,24`), `matched_pairs.csv 14400` (20 pts ×30d GFS vs ERA5, `lead %72`), `intent_samples.jsonl 2000` (Groq 800 paraphrases via 4-model queue)
* Trains `M1 8ep` DistilBERT 9-label, `M2 30ep` Deep MLP 5→128→128→64→2, `M3 5ep` DistilBERT 5-way
* Saves to `weathergpt_outputs/training/models/*/metrics.json` + `best.pt` + `scaler.pkl`

**If you get P100 `cap 6,0` (needs sm_70+):** code auto `force_cpu=True` + `no_cuda` (now `use_cpu`) → trains on CPU (slower but correct). For T4, it uses `cuda` + `fp16` + `DataParallel`.

**Push via CLI (script kernel, defaults to P100):**
```bash
cd kaggle_kernel_official && kaggle kernels push -p .  # → arkosarkarhehe/weathergpt-official
kaggle kernels status arkosarkarhehe/weathergpt-official
kaggle kernels output arkosarkarhehe/weathergpt-official -p /tmp/out
```

## 4. Docker

```bash
cp .env.example .env
docker build -t weathergpt .
docker run -p 8001:8001 --env-file .env weathergpt  # or docker-compose up
```

## 5. Environment Variables (`.env.example` → `.env`)

| Var | Required | What |
|-----|----------|------|
| `GROQ_API_KEY` | for LLM only | Groq `qwen/qwen3.8-27b` etc., if missing `mock_mode True` but WIO still works |
| `GROQ_MODEL` | no | default `qwen/qwen3.8-27b` orchestrator |
| `IMD_API_KEY` / `IMD_API_BASE` | no | when you have IMD portal access; decoders use fixtures until then |
| `HF_TOKEN` | no | for private HF pushes |
| `MODEL_DIR` | no | default `training/models` |

## 6. Troubleshooting

* `eccodes` pip fail → `conda install -c conda-forge eccodes cfgrib` or stay on `requirements.txt` (backend/train work without it; GRIB stubbed to Open-Meteo)
* `torch.cuda.is_available() False` on Kaggle → check `Accelerator: T4 x2`, not `None`
* HF `429` → already `tenacity` retry in `kaggle_kernel_official` Groq queue, add `time.sleep 0.25` if burst 800
* Port `8001` in use → `uvicorn app.main:app --port 8002`
* `Trainer tokenizer` error on transformers 4.46+ → fixed to `processing_class` + `eval_strategy` + `use_cpu`

## 7. Quick Smoke After Setup

```bash
curl http://localhost:8001/health
curl http://localhost:8001/plan?q="Will it rain tomorrow afternoon in Pune?"&location=Pune
curl -X POST http://localhost:8001/rade/advise -H "Content-Type: application/json" \
  -d '{"question":"Should I spray pesticide in Nagpur tomorrow?","location":{"raw":"Nagpur"}}' | jq
```

Artifacts after Kaggle: `training/models/*/metrics.json`, `weathergpt_outputs/` for download.
