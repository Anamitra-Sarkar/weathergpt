"""
Semantic Variable Classifier — raw field name → canonical variable.

Trainable on Kaggle T4 (distilbert) or CPU. Dry-run mode uses synthetic-only + no HF download.
Usage:
  python training/train_semantic_classifier.py --dry-run --epochs 1
  python training/train_semantic_classifier.py --epochs 5 --batch-size 32 --device auto
  # Kaggle:
  python training/train_semantic_classifier.py --epochs 5 --device cuda --amp
"""
from __future__ import annotations
import argparse, json, random, os
from pathlib import Path
import yaml

# auto device
import torch

# registry as synthetic source
LABEL_MAP = {
    "precipitation_amount": 0,
    "precipitation_probability": 1,
    "precipitation_rate": 2,
    "temperature_2m": 3,
    "temperature_max": 4,
    "temperature_min": 5,
    "wind_speed": 6,
    "wind_gust": 7,
    "heavy_rain_warning": 8,
}
INV_LABEL = {v:k for k,v in LABEL_MAP.items()}

# synthetic pairs: raw_field → label
SYNTHETIC_PAIRS = [
    ("APCP", "precipitation_amount"), ("tp", "precipitation_amount"), ("rainfall", "precipitation_amount"),
    ("precipitation", "precipitation_amount"), ("rain", "precipitation_amount"), ("total_precipitation", "precipitation_amount"),
    ("RAINC", "precipitation_amount"), ("RAINNC", "precipitation_amount"),
    ("PoP", "precipitation_probability"), ("precipitation_probability", "precipitation_probability"), ("rain_probability", "precipitation_probability"), ("chance_of_rain", "precipitation_probability"),
    ("prate", "precipitation_rate"), ("rain_rate", "precipitation_rate"), ("precipitation_rate", "precipitation_rate"),
    ("t2m", "temperature_2m"), ("2t", "temperature_2m"), ("temperature", "temperature_2m"), ("temp", "temperature_2m"),
    ("TMAX", "temperature_max"), ("tmax", "temperature_max"), ("temperature_max", "temperature_max"),
    ("TMIN", "temperature_min"), ("tmin", "temperature_min"), ("temperature_min", "temperature_min"),
    ("wind_speed", "wind_speed"), ("wind", "wind_speed"), ("u10", "wind_speed"), ("10m wind", "wind_speed"),
    ("wind_gust", "wind_gust"), ("gust", "wind_gust"), ("10m_gust", "wind_gust"),
    ("heavy rainfall", "heavy_rain_warning"), ("heavy_rain_warning", "heavy_rain_warning"), ("extremely heavy rain", "heavy_rain_warning"), ("Thunderstorm", "heavy_rain_warning"), ("Cyclone", "heavy_rain_warning"),
]

def make_dataset(num_dup: int = 20):
    rows = []
    for raw, canon in SYNTHETIC_PAIRS:
        for _ in range(num_dup):
            # augment with richer variants, avoid TMAX (mm) unit error for temp
            r = raw
            # unit suffix depends on variable: temp -> C, precip -> mm, wind -> m/s, warning -> no unit
            if canon in ("temperature_2m","temperature_max","temperature_min"):
                suffix = random.choice(["", " (°C)", " (C)"]) if random.random()<0.3 else ""
            elif canon in ("precipitation_amount",):
                suffix = random.choice(["", " (mm)", " (kg m-2)"]) if random.random()<0.3 else ""
            elif canon in ("wind_speed","wind_gust"):
                suffix = random.choice(["", " (m/s)"]) if random.random()<0.3 else ""
            else:
                suffix = ""
            r = r + suffix
            if random.random() < 0.2:
                r = r.upper()
            if random.random() < 0.1:
                r = f" {r} "
            # extra: underscore/hyphen/space variant
            if random.random() < 0.15:
                r = r.replace("_"," ").replace(" ","_") if "_" in r else r.replace(" ","_")
            rows.append((r, LABEL_MAP[canon]))
    random.shuffle(rows)
    # dedup on lower after generation
    seen=set()
    dedup=[]
    for raw,label in rows:
        key=(raw.strip().lower(), label)
        if key not in seen:
            seen.add(key)
            dedup.append((raw,label))
    # if dedup too small, pad back with new augmentations (but keep unique)
    return dedup

def load_external_csv(path: str):
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("raw_field") or row.get("field") or ""
            canon = row.get("canonical_variable") or row.get("canonical") or ""
            if canon in LABEL_MAP:
                rows.append((raw, LABEL_MAP[canon]))
    return rows

def run_dry():
    rows = make_dataset(2)
    print(f"[dry-run] synthetic rows: {len(rows)} labels: {LABEL_MAP}")
    print("[dry-run] no model artifact or metric is written")
    return

def run_full(args):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    # config
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    model_name = cfg.get("model_name", "distilbert-base-uncased")
    max_length = int(cfg.get("max_length", 64))
    output_dir = args.output_dir or cfg.get("output_dir", "training/models/semantic_classifier")
    epochs = args.epochs or int(cfg.get("epochs", 5))
    bs = args.batch_size or int(cfg.get("batch_size", 32))
    lr = float(cfg.get("lr", 2e-5))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[semantic] device={device} cuda_count={torch.cuda.device_count()} model={model_name}")

    # data
    rows = []
    # A reportable classifier must be trained on a versioned external corpus.
    p = Path("training/datasets/field_names.csv")
    if p.exists():
        try:
            ext = load_external_csv(str(p))
            rows.extend(ext)
            print(f"[semantic] loaded external {len(ext)} rows from {p}")
        except Exception as e:
            print(f"[semantic] external csv failed: {e}")
    if not rows:
        raise RuntimeError("No training/datasets/field_names.csv found; refusing synthetic semantic-classifier metrics.")
    print(f"[semantic] total rows {len(rows)}")

    # split with stratification: ensure each label in both splits
    from collections import Counter, defaultdict
    import random as _rnd
    # stratified: group by label then split each
    by_label=defaultdict(list)
    for r in rows:
        by_label[r[1]].append(r)
    train_rows, val_rows = [], []
    for label, lst in by_label.items():
        _rnd.shuffle(lst)
        s=int(len(lst)*0.85)
        train_rows.extend(lst[:s])
        val_rows.extend(lst[s:])
    random.shuffle(train_rows); random.shuffle(val_rows)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_ds = Dataset.from_dict({"text": [r[0] for r in train_rows], "labels": [r[1] for r in train_rows]})
    val_ds = Dataset.from_dict({"text": [r[0] for r in val_rows], "labels": [r[1] for r in val_rows]})
    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    val_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LABEL_MAP))

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}

    # class imbalance handling: compute class weights from train distribution
    from collections import Counter
    import torch as _t
    cnt=Counter(r[1] for r in train_rows)
    total=sum(cnt.values())
    weights=[total/cnt[i] if cnt[i]>0 else 1.0 for i in range(len(LABEL_MAP))]
    # normalise
    wsum=sum(weights); weights=[w/wsum*len(weights) for w in weights]
    class_weights=_t.tensor(weights, dtype=_t.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = _t.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        learning_rate=lr, evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        logging_steps=10, seed=42, fp16=(device=="cuda" and args.amp), report_to="none"
    )
    TrainerClass = WeightedTrainer if 'WeightedTrainer' in locals() else Trainer
    trainer = TrainerClass(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print(f"[semantic] metrics {metrics}")

    # save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)
    print(f"[semantic] saved to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no HF download, 2s smoke")
    ap.add_argument("--config", default="training/configs/semantic.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()
    if args.dry_run:
        run_dry()
    else:
        run_full(args)
