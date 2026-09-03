"""M3 — joint multilingual intent, slot and variable parser.

The predecessor was a 5-way sentence classifier on `distilbert-base-uncased`,
which cannot represent Devanagari, Bengali, Tamil or Telugu at all — every
non-Latin character became `[UNK]` — and it was trained on a corpus whose label
was `random.choice()` for two thirds of rows.  Its slots were never learned;
they came from a hardcoded nine-city regex.

This is a JointBERT-style head on MuRIL, which was pretrained on 17 Indian
languages *and* their romanised transliterations, so Hinglish and Banglish are
in-distribution rather than out of vocabulary.  Three heads share the encoder:

  * intent            -> which decision the user is actually trying to make,
                         in exactly RADE's domain vocabulary
  * BIO slot tagging   -> the location, time expression and crop spans, so the
                         downstream geocoder and time parser get the substring
                         instead of guessing from the whole sentence
  * variable multi-label -> which canonical variables must be retrieved, which
                         is what prunes the retrieval plan

Reported against the rule-based parser it replaces, per language, and on a test
set that holds out both whole template families and whole districts.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from modal_jobs.common import (DATA_DIR, HF_CACHE_DIR, MODEL_DIR, TRAIN_IMAGE,
                               TRAIN_VOLUMES, app)

ALGORITHM_VERSION = "m3_intent_v1"
MAX_LEN = 64


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, gpu="A10G", timeout=60 * 120)
def train(base_model: str = "google/muril-base-cased", epochs: int = 8,
          batch_size: int = 32, lr: float = 3e-5, seed: int = 42) -> dict:
    import hashlib

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import f1_score
    from transformers import AutoModel, AutoTokenizer

    from app.services.field_taxonomy import CANONICAL_VARIABLES
    from modal_jobs.build_queries import BIO_LABELS, INTENTS

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = f"{DATA_DIR}/d4_queries.parquet"
    frame = pd.read_parquet(data_path)
    data_sha = hashlib.sha256(open(data_path, "rb").read()).hexdigest()
    print(f"[m3] {len(frame):,} rows, {frame['language'].nunique()} languages, "
          f"splits={frame['split'].value_counts().to_dict()}")

    intent_index = {name: i for i, name in enumerate(INTENTS)}
    tag_index = {name: i for i, name in enumerate(BIO_LABELS)}
    variable_index = {name: i for i, name in enumerate(CANONICAL_VARIABLES)}

    frame["y_intent"] = frame["intent"].map(intent_index)
    if frame["y_intent"].isna().any():
        raise RuntimeError(f"unknown intents: "
                           f"{sorted(set(frame.loc[frame['y_intent'].isna(), 'intent']))}")

    def variable_vector(names) -> np.ndarray:
        vector = np.zeros(len(CANONICAL_VARIABLES), dtype="float32")
        for name in names:
            if name in variable_index:
                vector[variable_index[name]] = 1.0
        return vector

    frame["y_variables"] = frame["variables"].map(variable_vector)

    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=HF_CACHE_DIR)
    encoder = AutoModel.from_pretrained(base_model, cache_dir=HF_CACHE_DIR).to(device)

    def encode(rows):
        """Word-level BIO -> subword labels.  Only the first subword of a word
        carries the tag; continuation pieces get -100 so the loss ignores them
        and a multi-piece word cannot be counted several times."""
        words = [list(row["tokens"]) for _, row in rows.iterrows()]
        batch = tokenizer(words, is_split_into_words=True, padding="max_length",
                          truncation=True, max_length=MAX_LEN, return_tensors="pt")
        labels = torch.full((len(words), MAX_LEN), -100, dtype=torch.long)
        for i, (_, row) in enumerate(rows.iterrows()):
            word_ids = batch.word_ids(i)
            tags = list(row["tags"])
            previous = None
            for position, word_id in enumerate(word_ids):
                if word_id is None or word_id == previous:
                    continue
                previous = word_id
                if word_id < len(tags):
                    labels[i, position] = tag_index.get(tags[word_id], 0)
        return batch, labels

    class JointParser(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            hidden = backbone.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.intent_head = nn.Linear(hidden, len(INTENTS))
            self.slot_head = nn.Linear(hidden, len(BIO_LABELS))
            self.variable_head = nn.Linear(hidden, len(CANONICAL_VARIABLES))

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            output = self.backbone(**kwargs)
            sequence = self.dropout(output.last_hidden_state)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (sequence * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.intent_head(pooled), self.slot_head(sequence), self.variable_head(pooled)

    model = JointParser(encoder).to(device)

    splits = {name: frame[frame["split"] == name].reset_index(drop=True)
              for name in ("train", "val", "test")}
    cached = {name: encode(part) for name, part in splits.items()}

    counts = np.bincount(splits["train"]["y_intent"].to_numpy(), minlength=len(INTENTS))
    intent_weights = torch.tensor(
        np.where(counts > 0, len(splits["train"]) / (len(INTENTS) * np.maximum(counts, 1)), 0.0),
        dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max(1, epochs * (len(splits["train"]) // batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps,
                                                    pct_start=0.1)

    def batches(name, size, shuffle):
        batch, labels = cached[name]
        part = splits[name]
        order = np.arange(len(part))
        if shuffle:
            np.random.shuffle(order)
        for start in range(0, len(order), size):
            index = order[start:start + size]
            yield (index,
                   {k: v[index].to(device) for k, v in batch.items()},
                   labels[index].to(device),
                   torch.tensor(np.stack(part["y_intent"].to_numpy()[index]), device=device),
                   torch.tensor(np.stack(part["y_variables"].to_numpy()[index]), device=device))

    @torch.no_grad()
    def evaluate(name: str) -> dict:
        from seqeval.metrics import f1_score as seq_f1
        from seqeval.metrics import precision_score as seq_precision
        from seqeval.metrics import recall_score as seq_recall

        model.eval()
        part = splits[name]
        intent_predictions, intent_gold = [], []
        variable_predictions, variable_gold = [], []
        tag_true, tag_pred = [], []
        for index, batch, labels, y_intent, y_variables in batches(name, 128, False):
            intent_logits, slot_logits, variable_logits = model(**batch)
            intent_predictions += intent_logits.argmax(-1).tolist()
            intent_gold += y_intent.tolist()
            variable_predictions.append((torch.sigmoid(variable_logits) > 0.5).cpu().numpy())
            variable_gold.append(y_variables.cpu().numpy())
            best = slot_logits.argmax(-1).cpu().numpy()
            gold = labels.cpu().numpy()
            for row in range(len(index)):
                keep = gold[row] != -100
                if keep.sum() == 0:
                    continue
                tag_true.append([BIO_LABELS[t] for t in gold[row][keep]])
                tag_pred.append([BIO_LABELS[t] for t in best[row][keep]])

        variable_predictions = np.concatenate(variable_predictions)
        variable_gold = np.concatenate(variable_gold)
        model.train()
        return {
            "n": int(len(part)),
            "intent_accuracy": float(np.mean(np.array(intent_predictions) == np.array(intent_gold))),
            "intent_macro_f1": float(f1_score(intent_gold, intent_predictions,
                                              average="macro", zero_division=0)),
            "slot_f1": float(seq_f1(tag_true, tag_pred, zero_division=0)),
            "slot_precision": float(seq_precision(tag_true, tag_pred, zero_division=0)),
            "slot_recall": float(seq_recall(tag_true, tag_pred, zero_division=0)),
            "variable_micro_f1": float(f1_score(variable_gold, variable_predictions,
                                                average="micro", zero_division=0)),
            "variable_macro_f1": float(f1_score(variable_gold, variable_predictions,
                                                average="macro", zero_division=0)),
            "variable_exact_set_match": float(np.mean(
                (variable_predictions == variable_gold).all(axis=1))),
        }

    best_score, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        for index, batch, labels, y_intent, y_variables in batches("train", batch_size, True):
            intent_logits, slot_logits, variable_logits = model(**batch)
            loss = F.cross_entropy(intent_logits, y_intent, weight=intent_weights)
            loss = loss + F.cross_entropy(slot_logits.reshape(-1, len(BIO_LABELS)),
                                          labels.reshape(-1), ignore_index=-100)
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(variable_logits, y_variables)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss) * len(index)
            seen += len(index)
        metrics = evaluate("val")
        score = (metrics["intent_macro_f1"] + metrics["slot_f1"]) / 2
        print(f"[m3] epoch {epoch + 1}/{epochs} loss={total / max(seen, 1):.4f} "
              f"val_intent_f1={metrics['intent_macro_f1']:.4f} "
              f"val_slot_f1={metrics['slot_f1']:.4f} "
              f"val_var_micro_f1={metrics['variable_micro_f1']:.4f}")
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    val_metrics = evaluate("val")
    test_metrics = evaluate("test")

    # --- per-language breakdown on the held-out test set ---------------------
    per_language = {}
    original = splits["test"]
    original_cache = cached["test"]
    for language in sorted(original["language"].unique()):
        mask = (original["language"] == language).to_numpy()
        if mask.sum() < 25:
            continue
        splits["test"] = original[mask].reset_index(drop=True)
        cached["test"] = (original_cache[0][mask], original_cache[1][mask])
        per_language[language] = evaluate("test")
    splits["test"], cached["test"] = original, original_cache

    # --- baselines: the rule parser this replaces ---------------------------
    from app.orchestrator.retrieval_planner import build_retrieval_plan

    decision_to_intent = {"pesticide_spraying": "spray", "irrigation": "irrigate",
                          "harvest": "harvest", "marine": "marine", "travel": "travel",
                          "sowing": "sow", None: "none", "": "none"}

    def rule_baseline(part) -> dict:
        predicted = []
        for _, row in part.iterrows():
            try:
                plan = build_retrieval_plan(row["text"], "short")
                context = getattr(plan, "decision_context", None)
            except Exception:
                context = None
            name = decision_to_intent.get(context, "none")
            predicted.append(intent_index.get(name, intent_index["none"]))
        gold = part["y_intent"].to_numpy()
        return {
            "intent_accuracy": float(np.mean(np.array(predicted) == gold)),
            "intent_macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        }

    majority = int(np.bincount(splits["train"]["y_intent"].to_numpy()).argmax())
    baselines = {
        "rule_based_retrieval_planner_test": rule_baseline(splits["test"]),
        "majority_class_test": {
            "intent_accuracy": float(np.mean(splits["test"]["y_intent"].to_numpy() == majority)),
            "intent_macro_f1": float(f1_score(splits["test"]["y_intent"].to_numpy(),
                                              np.full(len(splits["test"]), majority),
                                              average="macro", zero_division=0)),
        },
    }

    out_dir = f"{MODEL_DIR}/{ALGORITHM_VERSION}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "base_model": base_model}, f"{out_dir}/model.pt")
    tokenizer.save_pretrained(out_dir)
    with open(f"{out_dir}/config.json", "w") as handle:
        json.dump({"base_model": base_model, "max_len": MAX_LEN, "intents": list(INTENTS),
                   "bio_labels": list(BIO_LABELS),
                   "canonical_variables": list(CANONICAL_VARIABLES)}, handle, indent=2)

    metrics = {
        "algorithm_version": ALGORITHM_VERSION,
        "model_kind": "JointBERT: intent + BIO slots + multi-label variables on a shared encoder",
        "base_model": base_model,
        "dataset_kind": "d4_multilingual_templated_queries_with_exact_slot_spans",
        "dataset_path": data_path, "dataset_sha256": data_sha,
        "split": "held-out template families {sow,heat,storm} AND 20% held-out districts",
        "n_train": int(len(splits["train"])), "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "val": val_metrics, "test_heldout": test_metrics,
        "per_language_test": per_language,
        "baselines": baselines,
    }
    with open(f"{out_dir}/metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    from modal_jobs.common import MODEL_VOL
    MODEL_VOL.commit()
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_language_test"},
                     indent=2, default=str))
    print("per_language_test:", json.dumps(per_language, indent=2, default=str))
    return metrics


@app.local_entrypoint()
def main(base_model: str = "google/muril-base-cased", epochs: int = 8):
    train.remote(base_model=base_model, epochs=epochs)
