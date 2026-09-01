# WeatherGPT — Installation Guide (Kaggle T4 / CPU)

## Option A — Kaggle T4 x2 (fastest, recommended for training)

1. Kaggle → Create Notebook → Settings: **Accelerator GPU T4 x2**, Internet ON
2. Upload repo:
   - Either `!git clone https://github.com/<you>/weathergpt.git` then `%cd weathergpt`
   - Or add this repo as Kaggle Dataset
3. Install + train (all three models, ~7 min total on T4):
   ```
   !pip install -q -r requirements.txt
   !python training/train_semantic_classifier.py --epochs 5 --device auto
   !python training/train_bias_correction.py --model mlp --epochs 20 --device auto
   !python training/train_intent_parser.py --epochs 3 --device auto
   ```
   Or just open `training/notebooks/weathergpt_training.ipynb` → Run All.
4. Artifacts:
   - `training/models/semantic_classifier/best/`
   - `training/models/bias_correction/best.pt + metrics.json`
   - `training/models/intent_parser/best/`
   - Copy to `/kaggle/working/weathergpt_outputs/` for download

To use your own data: Upload → `training/datasets/` (see `training/datasets/README.md`)

---

## Option B — Local CPU

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# optional decoders
pip install -r requirements-full.txt  # if you need GRIB2/NetCDF/HDF5

# smoke (no downloads, 5s)
python training/train_semantic_classifier.py --dry-run
python training/train_bias_correction.py --dry-run
python training/train_intent_parser.py --dry-run

# full CPU train (longer, reduce epochs for quick test)
python training/train_semantic_classifier.py --epochs 2 --device cpu
python training/train_bias_correction.py --model mlp --epochs 5 --device cpu
python training/train_intent_parser.py --epochs 1 --device cpu

# backend
uvicorn app.main:app --reload --port 8001
# → http://localhost:8001/docs
```

---

## Option C — Docker

```bash
cp .env.example .env
docker build -t weathergpt .
docker run -p 8001:8001 --env-file .env weathergpt
```

---

## Troubleshooting

- `eccodes` fail on pip: use conda `conda install -c conda-forge eccodes cfgrib` or stay on `requirements.txt` (backend/train work without it; GRIB is stubbed to Open-Meteo)
- `torch.cuda.is_available() == False` on Kaggle: ensure Accelerator is T4 x2, not None
- HF rate limit `429`: scripts have retry; re-run cell or set `HF_TOKEN` in `.env`
- Port 8001 in use: `uvicorn app.main:app --port 8002`

## What to send after Kaggle train

Zip and upload back: `training/models/` + `weathergpt_outputs/` metrics. Backend auto-loads anything in `training/models/` on next boot (`MODEL_DIR` env).
