"""Convert Muse Spark's hand-generated multilingual corpus into d4_queries.parquet.

Muse Spark (opencode's free contributor model) generated the corpus directly —
no Groq, no rate limits — because five attempts at the original Groq-based
pipeline could not clear the on_demand tier's queue latency within a session.
This script applies the exact same contracts the original pipeline did: BIO
spans computed from character offsets (never trusted from the source), a
slot-substring re-verification (belt and suspenders — the generating model was
asked to self-verify, this re-checks independently), dedup, and a split by
template family AND location so nothing in test was seen in train.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime

from modal_jobs.build_queries import HELD_OUT_FAMILIES, _spans_to_bio
from modal_jobs.common import DATA_DIR, DATA_IMAGE, VOLUMES, app


@app.function(image=DATA_IMAGE, volumes=VOLUMES, timeout=60 * 10)
def import_and_assemble(rows: list, seed: int = 42) -> dict:
    import pandas as pd

    from modal_jobs.contracts import check_label_corpus

    verified, rejected = [], {"missing_field": 0, "slot_not_substring": 0, "span_align": 0}
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            rejected["missing_field"] += 1
            continue
        spans = []
        crop = row.get("crop_value")
        wanted = [("loc_value", "LOC"), ("time_value", "TIME")] + (
            [("crop_value", "CROP")] if crop else [])
        ok = True
        for field, slot in wanted:
            value = str(row.get(field) or "").strip()
            at = text.find(value)
            if not value or at < 0:
                rejected["slot_not_substring"] += 1
                ok = False
                break
            spans.append((at, at + len(value), slot))
        if not ok:
            continue
        bio = _spans_to_bio(text, spans)
        if bio is None:
            rejected["span_align"] += 1
            continue
        verified.append({
            "text": text, "family": row.get("family", "unknown"),
            "intent": row.get("intent", "none"),
            "variables": list(row.get("variables") or []),
            "language": row.get("language", "en"),
            "loc_id": row.get("loc_id", "unknown"),
            "origin": row.get("origin", "muse_spark"),
            "tokens": [t for t, _ in bio], "tags": [g for _, g in bio],
        })

    print(f"[muse_spark_import] {len(rows)} rows in, {len(verified)} verified, "
          f"rejected {rejected}")
    if not verified:
        raise RuntimeError("nothing survived verification — check the source file")

    frame = pd.DataFrame(verified)
    frame["norm"] = frame["text"].str.strip().str.lower()
    before = len(frame)
    frame = frame.drop_duplicates(subset="norm").reset_index(drop=True)

    all_locs = sorted(frame["loc_id"].unique())
    rng = random.Random(seed)
    held_out_locs = set(rng.sample(all_locs, max(1, len(all_locs) // 5)))

    def assign(row) -> str:
        if row["family"] in HELD_OUT_FAMILIES or row["loc_id"] in held_out_locs:
            return "test"
        digest = int(hashlib.sha1(row["norm"].encode()).hexdigest()[:8], 16) % 100
        return "val" if digest < 15 else "train"

    frame["split"] = frame.apply(assign, axis=1)
    frame = frame.drop(columns=["norm"])

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = f"{DATA_DIR}/d4_queries.parquet"
    frame.to_parquet(out_path, index=False, compression="zstd")

    report = check_label_corpus(frame.assign(key=frame["text"]), name="d4_queries_muse_spark",
                                label_column="intent", split_column="split",
                                key_column="key", min_rows=100)
    stats = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "source": "muse_spark_1.3 (opencode contributor-free), hand-generated, no Groq",
        "rows_before_dedup": int(before), "rows": int(len(frame)),
        "by_split": frame["split"].value_counts().to_dict(),
        "by_language": frame["language"].value_counts().to_dict(),
        "by_intent": frame["intent"].value_counts().to_dict(),
        "rejected_at_import": rejected,
        "majority_intent_baseline": float(frame["intent"].value_counts().iloc[0] / len(frame)),
        "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
        "contracts": report.summary(),
    }
    with open(f"{DATA_DIR}/d4_queries.stats.json", "w") as handle:
        json.dump(stats, handle, indent=2, default=str)
    from modal_jobs.common import DATA_VOL
    DATA_VOL.commit()
    print(json.dumps(stats, indent=2, default=str)[:4000])
    return stats


@app.local_entrypoint()
def import_d4(jsonl_path: str = "scratch_d4_muse_spark.jsonl", seed: int = 42):
    rows = []
    with open(jsonl_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[muse_spark_import] read {len(rows)} rows from {jsonl_path}")
    import_and_assemble.remote(rows, seed)
