"""M1 — cross-schema field semantic mapper.

The task: given a native field name as it appears in some meteorological
standard (`APCP`, `RAINNC`, `tp`, `Max Temp`, `prate`), optionally with a unit
and a description, decide which canonical variable it means, at what statistic,
level and accumulation window — or refuse.

Why a label-embedding bi-encoder rather than a softmax head:
  * the official-warning classes have single-digit support (IMD's real CAP event
    vocabulary is small), and a softmax row learned from three examples is
    worthless, whereas scoring against the *text* of the label transfers meaning
    from the pretrained encoder;
  * new canonical variables can be added later by writing one sentence, without
    retraining a fixed-width classifier.

Refusal is a first-class prediction: `other` is a real class with its own label
text, and a similarity margin threshold calibrated on validation converts
low-confidence predictions into an explicit abstention.  Fabricating a mapping
is the failure mode this project exists to prevent.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime

from modal_jobs.common import (DATA_DIR, HF_CACHE_DIR, MODEL_DIR, TRAIN_IMAGE,
                               TRAIN_VOLUMES, app)

BASE_MODEL = "intfloat/multilingual-e5-base"
MAX_LEN = 64
ALGORITHM_VERSION = "m1_field_mapper_v1"


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, gpu="A10G", timeout=60 * 90)
def train(epochs: int = 6, batch_size: int = 64, lr: float = 2e-5,
          seed: int = 42, temperature: float = 0.05) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import f1_score
    from transformers import AutoModel, AutoTokenizer

    from app.services.field_taxonomy import (CANONICAL_VARIABLES, EVIDENCE_CLASSES,
                                             LABEL_DESCRIPTIONS, STATISTICS,
                                             VERTICAL_LEVELS)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = f"{DATA_DIR}/d3_fields.parquet"
    frame = pd.read_parquet(data_path)
    data_sha = hashlib.sha256(open(data_path, "rb").read()).hexdigest()

    var_index = {name: i for i, name in enumerate(CANONICAL_VARIABLES)}
    stat_index = {name: i for i, name in enumerate(STATISTICS)}
    level_index = {name: i for i, name in enumerate(VERTICAL_LEVELS)}
    class_index = {name: i for i, name in enumerate(EVIDENCE_CLASSES)}

    frame = frame[frame["canonical_variable"].isin(var_index)].reset_index(drop=True)
    frame["y_var"] = frame["canonical_variable"].map(var_index)
    frame["y_stat"] = frame["statistic"].map(lambda s: stat_index.get(s, stat_index["instant"]))
    frame["y_level"] = frame["vertical_level"].map(lambda s: level_index.get(s, level_index["other"]))
    frame["y_class"] = frame["evidence_class"].map(lambda s: class_index.get(s, class_index["forecast"]))

    train_df = frame[frame["split"] == "train"].reset_index(drop=True)
    val_df = frame[frame["split"] == "val"].reset_index(drop=True)
    test_df = frame[frame["split"] == "test_zeroshot"].reset_index(drop=True)
    print(f"[m1] train={len(train_df)} val={len(val_df)} zeroshot_test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE_DIR)
    encoder = AutoModel.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE_DIR).to(device)

    def render(row: dict, *, train_mode: bool) -> str:
        """Serialise one field.  In training, unit and description are dropped at
        random so the model cannot lean on metadata that is often missing at
        inference time — a bare `APCP` must still resolve."""
        name = str(row["raw_field"])
        unit = row.get("unit")
        description = str(row.get("description") or "")
        level = str(row.get("level_text") or "")
        window = str(row.get("time_range_text") or "")
        if train_mode:
            if random.random() < 0.45:
                unit = None
            if random.random() < 0.35:
                description = ""
            if random.random() < 0.30:
                level = ""
            if random.random() < 0.30:
                window = ""
            if random.random() < 0.20:
                name = name.upper()
            elif random.random() < 0.20:
                name = name.lower()
        parts = [f"field: {name}"]
        if unit is not None and str(unit) != "nan" and str(unit):
            parts.append(f"unit: {unit}")
        if description:
            parts.append(f"means: {description}")
        if level:
            parts.append(f"level: {level}")
        if window:
            parts.append(f"period: {window}")
        return "query: " + " | ".join(parts)

    label_texts = ["passage: " + LABEL_DESCRIPTIONS[name] for name in CANONICAL_VARIABLES]

    class Mapper(nn.Module):
        """Shared encoder, mean pooled, with a label-embedding scorer and three
        light auxiliary heads for the rest of the semantics."""

        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            hidden = backbone.config.hidden_size
            self.stat_head = nn.Linear(hidden, len(STATISTICS))
            self.level_head = nn.Linear(hidden, len(VERTICAL_LEVELS))
            self.class_head = nn.Linear(hidden, len(EVIDENCE_CLASSES))

        def embed(self, input_ids, attention_mask):
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return pooled

    model = Mapper(encoder).to(device)

    def tokenize(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN,
                          return_tensors="pt")
        return {k: v.to(device) for k, v in batch.items()}

    label_batch = tokenize(label_texts)

    def label_embeddings():
        pooled = model.embed(label_batch["input_ids"], label_batch["attention_mask"])
        return F.normalize(pooled, dim=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_df) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(1, epochs * steps_per_epoch), pct_start=0.1)

    # `other` dominates (most entries in any standard are not weather variables),
    # so weight it down rather than letting it eat the loss.
    counts = np.bincount(train_df["y_var"].to_numpy(), minlength=len(CANONICAL_VARIABLES))
    weights = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    weights = weights / weights[weights > 0].mean()
    var_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    @torch.no_grad()
    def evaluate(df, name: str, threshold: float | None = None) -> dict:
        model.eval()
        label_vectors = label_embeddings()
        preds, sims, gold = [], [], []
        stats_pred, levels_pred, classes_pred = [], [], []
        for start in range(0, len(df), 128):
            chunk = df.iloc[start:start + 128]
            batch = tokenize([render(row, train_mode=False) for _, row in chunk.iterrows()])
            pooled = model.embed(batch["input_ids"], batch["attention_mask"])
            normalized = F.normalize(pooled, dim=-1)
            similarity = normalized @ label_vectors.T
            best = similarity.argmax(-1)
            preds += best.tolist()
            sims += similarity.max(-1).values.tolist()
            gold += chunk["y_var"].tolist()
            stats_pred += model.stat_head(pooled).argmax(-1).tolist()
            levels_pred += model.level_head(pooled).argmax(-1).tolist()
            classes_pred += model.class_head(pooled).argmax(-1).tolist()

        preds_array = np.array(preds)
        gold_array = np.array(gold)
        sims_array = np.array(sims)
        other_id = var_index["other"]
        if threshold is not None:
            preds_array = np.where(sims_array < threshold, other_id, preds_array)

        mapped = gold_array != other_id
        result = {
            f"{name}_accuracy": float((preds_array == gold_array).mean()),
            f"{name}_macro_f1": float(f1_score(gold_array, preds_array, average="macro",
                                               zero_division=0)),
            f"{name}_mapped_accuracy": float((preds_array[mapped] == gold_array[mapped]).mean())
                if mapped.any() else 0.0,
            f"{name}_statistic_accuracy": float((np.array(stats_pred) == df["y_stat"].to_numpy()).mean()),
            f"{name}_level_accuracy": float((np.array(levels_pred) == df["y_level"].to_numpy()).mean()),
            f"{name}_evidence_class_accuracy": float((np.array(classes_pred) == df["y_class"].to_numpy()).mean()),
            f"{name}_n": int(len(df)),
        }
        # Abstention quality: of the fields we refused, how many were genuinely
        # unmappable, and of the genuinely unmappable, how many did we refuse.
        refused = preds_array == other_id
        unmappable = gold_array == other_id
        result[f"{name}_abstain_precision"] = float(unmappable[refused].mean()) if refused.any() else 0.0
        result[f"{name}_abstain_recall"] = float(refused[unmappable].mean()) if unmappable.any() else 0.0
        # The cardinal sin: confidently mapping a field to the WRONG variable.
        wrong = (~refused) & (~unmappable) & (preds_array != gold_array)
        hallucinated = (~refused) & unmappable
        result[f"{name}_misassignment_rate"] = float(wrong.mean())
        result[f"{name}_hallucination_rate"] = float(hallucinated.mean())
        model.train()
        return result, sims_array, preds_array, gold_array

    order = np.arange(len(train_df))
    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        np.random.shuffle(order)
        model.train()
        running = 0.0
        for step in range(steps_per_epoch):
            index = order[step * batch_size:(step + 1) * batch_size]
            if len(index) == 0:
                continue
            chunk = train_df.iloc[index]
            batch = tokenize([render(row, train_mode=True) for _, row in chunk.iterrows()])
            pooled = model.embed(batch["input_ids"], batch["attention_mask"])
            normalized = F.normalize(pooled, dim=-1)
            label_vectors = label_embeddings()
            logits = normalized @ label_vectors.T / temperature

            y_var = torch.tensor(chunk["y_var"].to_numpy(), device=device)
            loss = F.cross_entropy(logits, y_var, weight=var_weights)
            loss = loss + 0.3 * F.cross_entropy(
                model.stat_head(pooled), torch.tensor(chunk["y_stat"].to_numpy(), device=device))
            loss = loss + 0.2 * F.cross_entropy(
                model.level_head(pooled), torch.tensor(chunk["y_level"].to_numpy(), device=device))
            loss = loss + 0.2 * F.cross_entropy(
                model.class_head(pooled), torch.tensor(chunk["y_class"].to_numpy(), device=device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss)

        metrics, _, _, _ = evaluate(val_df, "val")
        print(f"[m1] epoch {epoch + 1}/{epochs} loss={running / steps_per_epoch:.4f} "
              f"val_macro_f1={metrics['val_macro_f1']:.4f} val_acc={metrics['val_accuracy']:.4f}")
        if metrics["val_macro_f1"] > best_val:
            best_val = metrics["val_macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- calibrate the abstention threshold on validation --------------------
    val_metrics, val_sims, val_preds, val_gold = evaluate(val_df, "val")
    other_id = var_index["other"]
    best_threshold, best_score = 0.0, -1.0
    for candidate in np.quantile(val_sims, np.linspace(0.0, 0.6, 61)):
        adjusted = np.where(val_sims < candidate, other_id, val_preds)
        score = f1_score(val_gold, adjusted, average="macro", zero_division=0)
        if score > best_score:
            best_score, best_threshold = float(score), float(candidate)
    print(f"[m1] abstention threshold {best_threshold:.4f} (val macro-F1 {best_score:.4f})")

    val_metrics, _, _, _ = evaluate(val_df, "val", threshold=best_threshold)
    test_metrics, _, test_preds, test_gold = evaluate(test_df, "test_zeroshot",
                                                      threshold=best_threshold)

    # --- baselines -----------------------------------------------------------
    from app.services.variable_registry import normalize_field

    def registry_baseline(df) -> dict:
        predicted = []
        for _, row in df.iterrows():
            entry = normalize_field(str(row["raw_field"]))
            name = entry["canonical"] if entry else "other"
            predicted.append(var_index.get(name, other_id))
        predicted = np.array(predicted)
        gold = df["y_var"].to_numpy()
        mapped = gold != other_id
        return {
            "accuracy": float((predicted == gold).mean()),
            "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
            "mapped_accuracy": float((predicted[mapped] == gold[mapped]).mean()) if mapped.any() else 0.0,
            "mapped_coverage": float((predicted[mapped] != other_id).mean()) if mapped.any() else 0.0,
        }

    baselines = {
        "dict_registry_zeroshot": registry_baseline(test_df),
        "dict_registry_val": registry_baseline(val_df),
        "majority_class_zeroshot": {
            "accuracy": float((test_df["y_var"].to_numpy() == other_id).mean()),
            "macro_f1": float(f1_score(test_df["y_var"].to_numpy(),
                                       np.full(len(test_df), other_id),
                                       average="macro", zero_division=0)),
        },
    }

    # --- per-source zero-shot breakdown --------------------------------------
    by_source = {}
    for source in sorted(test_df["source_table"].unique()):
        mask = (test_df["source_table"] == source).to_numpy()
        gold = test_gold[mask]
        predicted = test_preds[mask]
        mapped = gold != other_id
        by_source[source] = {
            "n": int(mask.sum()),
            "accuracy": float((predicted == gold).mean()),
            "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
            "mapped_accuracy": float((predicted[mapped] == gold[mapped]).mean()) if mapped.any() else None,
        }

    out_dir = f"{MODEL_DIR}/{ALGORITHM_VERSION}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "base_model": BASE_MODEL}, f"{out_dir}/model.pt")
    tokenizer.save_pretrained(out_dir)
    with torch.no_grad():
        vectors = label_embeddings().cpu().numpy()
    np.save(f"{out_dir}/label_embeddings.npy", vectors)

    metrics = {
        "algorithm_version": ALGORITHM_VERSION,
        "model_kind": "label-embedding bi-encoder + multitask heads",
        "base_model": BASE_MODEL,
        "dataset_kind": "d3_authoritative_parameter_tables",
        "dataset_path": data_path,
        "dataset_sha256": data_sha,
        "split": "by_source_table (train: CF+GRIB2+CAP, test: WRF/NCEP/BUFR/OpenMeteo/IMD)",
        "n_train": int(len(train_df)), "n_val": int(len(val_df)), "n_test": int(len(test_df)),
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
        "abstention_threshold": best_threshold,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        **val_metrics, **test_metrics,
        "baselines": baselines,
        "zeroshot_by_source": by_source,
        "label_space": list(CANONICAL_VARIABLES),
    }
    with open(f"{out_dir}/metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(f"{out_dir}/config.json", "w") as handle:
        json.dump({"base_model": BASE_MODEL, "max_len": MAX_LEN,
                   "abstention_threshold": best_threshold,
                   "canonical_variables": list(CANONICAL_VARIABLES),
                   "statistics": list(STATISTICS), "levels": list(VERTICAL_LEVELS),
                   "evidence_classes": list(EVIDENCE_CLASSES)}, handle, indent=2)

    from modal_jobs.common import MODEL_VOL
    MODEL_VOL.commit()
    print(json.dumps({k: v for k, v in metrics.items() if k != "label_space"}, indent=2))
    return metrics


@app.local_entrypoint()
def main(epochs: int = 6):
    train.remote(epochs=epochs)
