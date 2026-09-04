"""Print the metrics of every trained artifact on the model volume."""
from __future__ import annotations

import json

from modal_jobs.common import MODEL_DIR, TRAIN_IMAGE, TRAIN_VOLUMES, app


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, timeout=60 * 10)
def show(keys: str = "") -> dict:
    import os

    out = {}
    for name in sorted(os.listdir(MODEL_DIR)):
        path = f"{MODEL_DIR}/{name}/metrics.json"
        if not os.path.exists(path):
            continue
        metrics = json.loads(open(path).read())
        out[name] = metrics
        wanted = [k.strip() for k in keys.split(",") if k.strip()]
        print(f"\n=== {name} ===")
        if wanted:
            for key in wanted:
                if key in metrics:
                    print(f"  {key:44s} {metrics[key]}")
        else:
            print(json.dumps({k: v for k, v in metrics.items()
                              if k not in ("held_out_locations", "label_space")},
                             indent=2, default=str)[:6000])
    return out


@app.local_entrypoint()
def main_show_metrics(keys: str = ""):
    show.remote(keys)
