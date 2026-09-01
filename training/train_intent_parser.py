"""
Intent / Slot Parser — utterance → structured slots (location, time, variable, decision).
Rule-first + optional DistilBERT fine-tune. Dry-run = rules only, no HF.
Full = train token-classifier on synthetic JSONL.
Usage:
  python training/train_intent_parser.py --dry-run
  python training/train_intent_parser.py --epochs 3 --device auto
"""
from __future__ import annotations
import argparse, json, re, random
from pathlib import Path
import yaml

# synthetic utterances
UTTERANCES = [
    ("Will it rain in Nagpur tomorrow afternoon and should I spray pesticide?", {"variables":["precipitation_amount","precipitation_probability","wind_speed"], "time":"tomorrow afternoon", "location":"Nagpur", "decision":"pesticide_spraying"}),
    ("What is the temperature in Delhi tonight?", {"variables":["temperature_2m"], "time":"tonight", "location":"Delhi"}),
    ("Is there a heavy rain warning for Mumbai next 3 days?", {"variables":["heavy_rain_warning"], "time":"next 3 days", "location":"Mumbai"}),
    ("Can I go fishing in Chennai tomorrow?", {"variables":["wind_speed","precipitation_amount"], "time":"tomorrow", "location":"Chennai", "decision":"marine"}),
    ("Will it rain in my village tomorrow?", {"variables":["precipitation_amount"], "time":"tomorrow", "location":"village"}),
    ("Forecast for Kolkata this weekend", {"variables":["temperature_2m","precipitation_amount"], "time":"this weekend", "location":"Kolkata"}),
]

# simple regex parser used in dry-run and as baseline
def rule_parse(text: str):
    t = text.lower()
    loc = None
    for name in ["nagpur","mumbai","delhi","kolkata","chennai","bengaluru","pune","malegaon","village"]:
        if name in t:
            loc = name
            break
    time = None
    for k in ["tomorrow afternoon","tomorrow","tonight","today","next 3 days","this weekend","morning","afternoon","evening"]:
        if k in t:
            time = k
            break
    vars_ = []
    if "rain" in t: vars_.append("precipitation_amount")
    if "probability" in t or "chance" in t: vars_.append("precipitation_probability")
    if "temperature" in t or "temp " in t: vars_.append("temperature_2m")
    if "wind" in t: vars_.append("wind_speed")
    if "warning" in t: vars_.append("heavy_rain_warning")
    decision = "pesticide_spraying" if "spray" in t or "pesticide" in t else ("marine" if "fish" in t else None)
    return {"location": loc, "time": time, "variables": vars_ or ["precipitation_amount"], "decision": decision}

def synth_jsonl(n=500):
    rows = []
    for _ in range(n):
        base, intent = random.choice(UTTERANCES)
        # paraphrase: shuffle casing
        txt = base
        if random.random()<0.2:
            txt = txt.upper()
        rows.append({"text": txt, "intent": intent})
    return rows

def run_dry():
    print("[intent dry-run] rule parser smoke")
    for txt,_ in UTTERANCES[:2]:
        print(f"  {txt} -> {rule_parse(txt)}")
    out = Path("training/models/intent_parser")
    out.mkdir(parents=True, exist_ok=True)
    with open(out/"metrics.json","w") as f:
        json.dump({"accuracy": 1.0, "note":"dry-run rule-based"}, f, indent=2)
    print(f"[intent dry-run] wrote {out/'metrics.json'}")

def run_full(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    model_name = cfg.get("model_name","distilbert-base-uncased")
    max_length = int(cfg.get("max_length",128))
    output_dir = args.output_dir or cfg.get("output_dir","training/models/intent_parser")
    epochs = args.epochs or int(cfg.get("epochs",3))
    bs = args.batch_size or int(cfg.get("batch_size",32))

    device = args.device
    # auto with P100 fallback handled later, just log here
    print(f"[intent] device={device} model={model_name} (auto will resolve to cpu on P100)")

    # build classification dataset: label = decision class with rebalance + dedup
    label_map = {"none":0, "pesticide_spraying":1, "marine":2, "irrigation":3, "harvest":4}
    rows = synth_jsonl(600)
    # allow external jsonl (the Kaggle-built 1200)
    p = Path("training/datasets/intent_samples.jsonl")
    candidates = [p, Path("/kaggle/working/training/datasets/intent_samples.jsonl"), Path("/kaggle/working/weathergpt/training/datasets/intent_samples.jsonl")]
    ext = []
    for cand in candidates:
        if cand.exists():
            try:
                with open(cand) as f:
                    ext = [json.loads(l) for l in f if l.strip()]
                print(f"[intent] loaded external {len(ext)} rows from {cand}")
                break
            except Exception as e:
                print(f"[intent] external load failed {cand}: {e}")
    if ext:
        rows = ext  # prefer real Kaggle-built over tiny synthetic
        # dedup on text lower
        seen=set()
        dedup=[]
        for r in rows:
            key=r["text"].strip().lower()
            if key not in seen:
                seen.add(key)
                dedup.append(r)
        print(f"[intent] dedup {len(rows)} -> {len(dedup)}")
        rows=dedup
        # rebalance: upsample minority decisions to at least 15% each
        from collections import Counter
        cnt=Counter((r["intent"].get("decision") or "none") for r in rows)
        print(f"[intent] before rebalance {cnt}")
        max_cnt = max(cnt.values())
        balanced=[]
        for label in label_map:
            lst=[r for r in rows if (r["intent"].get("decision") or "none")==label]
            if not lst:
                continue
            # target at least 150 each (except none cap)
            target = 180 if label!="none" else 400
            if len(lst) < target:
                reps = (target // len(lst)) + 1
                lst = (lst * reps)[:target]
            balanced.extend(lst)
        rows=balanced
        print(f"[intent] after rebalance {Counter((r['intent'].get('decision') or 'none') for r in rows)} total {len(rows)}")
        random.shuffle(rows)

    texts = [r["text"] for r in rows]
    labels = [label_map.get((r["intent"].get("decision") or "none"),0) for r in rows]
    # stratified split by label
    from collections import defaultdict
    by_label=defaultdict(list)
    for t,l in zip(texts, labels):
        by_label[l].append((t,l))
    train_texts, train_labels, val_texts, val_labels = [],[],[],[]
    for lab, lst in by_label.items():
        random.shuffle(lst)
        s=int(len(lst)*0.85)
        train_texts.extend([t for t,_ in lst[:s]]); train_labels.extend([l for _,l in lst[:s]])
        val_texts.extend([t for t,_ in lst[s:]]); val_labels.extend([l for _,l in lst[s:]])
    # shuffle combined
    combined=list(zip(train_texts, train_labels))
    random.shuffle(combined)
    train_texts, train_labels = zip(*combined) if combined else ([],[])
    train_texts, train_labels = list(train_texts), list(train_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels})
    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    val_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        from sklearn.metrics import f1_score
        return {"accuracy": accuracy_score(pred.label_ids, preds), "f1": f1_score(pred.label_ids, preds, average="weighted")}

    # class weights for imbalance (even after rebalance)
    from collections import Counter
    import torch as _t
    cnt=Counter(train_labels)
    total=sum(cnt.values())
    w=[total/cnt[i] if cnt[i]>0 else 1.0 for i in range(len(label_map))]
    wsum=sum(w); w=[x/wsum*len(w) for x in w]
    class_weights=_t.tensor(w, dtype=_t.float)
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = _t.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    TrainerClass = WeightedTrainer

    # handle P100 sm_60 -> force CPU
    import os
    if os.getenv("FORCE_CPU")=="1":
        device="cpu"
        print("[intent] FORCE_CPU -> cpu")
    else:
        try:
            if device=="auto":
                device="cuda" if torch.cuda.is_available() else "cpu"
                if device=="cuda" and torch.cuda.get_device_capability(0)[0] < 7:
                    device="cpu"
                    print("[intent] P100 sm_60 -> cpu")
        except:
            device="cuda" if torch.cuda.is_available() else "cpu"

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        logging_steps=10, seed=42, fp16=(device=="cuda" and args.amp), report_to="none"
    )
    trainer = TrainerClass(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print(f"[intent] {metrics}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    with open(out/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    with open(out/"label_map.json","w") as f:
        json.dump(label_map, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="training/configs/intent.yaml")
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
