"""Kaggle entry point: publish the served models, static grids, source and a generated model card to Hugging Face.

Reads the `serve-smoke` kernel output (deploy/ folder + serve_smoke.json) and the private credentials dataset
(`hf_token`).  The card is generated from the artifacts' own metrics.json -- no number in it is typed by hand.
The repo is created PRIVATE; make it public from the Hugging Face settings page when you are happy with it.
Never prints the token.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from pathlib import Path

# one import per line: the bundler inlines exactly the modules named here (and what they import), and republishes them as ./src
from event_models import dataset
from weathergpt_events import engine
from weathergpt_events import features
from weathergpt_events import live
from weathergpt_events import models
from weathergpt_events import registry as reg_mod
from weathergpt_events import static
from weathergpt_events import tools

_INLINED = (dataset, engine, features, live, models, static)     # imported only so the bundle carries (and republishes) them

INPUT = "/kaggle/input"
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("publish_out")
STAGE = WORK / "hf"

INIT_EVENT_MODELS = '"""Training and shared feature code for the WeatherGPT event models."""\n'
INIT_WEATHERGPT_EVENTS = (
    '"""Inference interface for the WeatherGPT event / range / rain-curve models."""\nimport os\n\n'
    '# The collectors read the lead set at import time.  Serving always wants the full 10-day horizon (52 steps).\n'
    'os.environ.setdefault("GFS_LEADSET", "all")\n\n'
    'from weathergpt_events.registry import EventRegistry  # noqa: E402,F401\n')


def fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return "n/a" if x is None else str(x)


def headline_text(head: dict) -> str:
    flat = []
    for k, v in head.items():
        if isinstance(v, dict):
            flat.append(f"{k}: " + ", ".join(f"{a}={fmt(b)}" for a, b in v.items()))
        else:
            flat.append(f"{k}={fmt(v)}")
    return "; ".join(flat)


def build_card(catalogue: list, registry, smoke: dict, repo: str) -> str:
    any_metrics = next(iter(registry._metrics.values()))
    prov = any_metrics.get("provenance", {})
    split = prov.get("split", {})
    served = [c for c in catalogue if c["available"]]
    refused = [c for c in catalogue if not c["available"]]
    rows = "\n".join(f"| `{c['name']}` | {c['kind']} | {c['horizon_days']} d | {headline_text(c['headline_held_out_skill'])} | "
                     f"{', '.join(c['validated_zones']) or 'none'} |" for c in served)
    refused_rows = "\n".join(f"| `{c['name']}` | {c['why_not']} |" for c in refused) or "| (none) | |"
    tails = "\n".join(f"- `{c['name']}`: no held-out skill over climatology at thresholds (mm) {reg_mod.unskilled_thresholds(registry._metrics[c['name']])}; "
                      f"those exceedance values are returned but flagged `thresholds_without_skill_mm`."
                      for c in served if c["kind"] == "curve" and reg_mod.unskilled_thresholds(registry._metrics[c["name"]]))
    answers = "\n".join(f"- **`{c['name']}`** - {c['answers']}" for c in served)
    return f"""---
license: other
library_name: lightgbm
tags: [weather, forecasting, india, lightgbm, nowcasting, sih-2026]
---

# WeatherGPT event models (India, 10-day horizon)

Small per-target LightGBM specialists that turn the newest NOAA GFS / GEFS run into calibrated answers for any land
point in India (6-38 N, 67-98 E): chance of thunderstorm / fog / strong wind / rain, rain amount, temperature /
wind / humidity ranges, heat- and cold-wave days, and multi-day rain windows.  An orchestrating LLM is meant to call only
the models a question needs; this repo is the ML layer, not a chatbot.

**Status: research prototype.  Not an official forecast and not for safety-critical decisions.**

## How a model is allowed to answer
Every artifact must pass an admission gate (`weathergpt_events/registry.py`) that reads its own `metrics.json`:
it must have provenance, beat a baseline on **places it never saw** (zone-stratified 4x4 degree blocks held out) in the
**future** (`test_space`), and, per climate zone, show held-out evidence there.  Otherwise the engine returns
`validated: false` and **no numbers** for that zone / lead.  Queries over the sea or outside the box are refused.

## Served models (passed the gate) - held-out skill
Skill is measured beside zone x month climatology (binary / rain-curve models) or the raw GFS value (ranges), never alone.

| model | kind | horizon | held-out headline (test_space) | zones with held-out evidence |
|---|---|---|---|---|
{rows}

{answers}

## Not served (failed the gate or untrained)
| model | reason |
|---|---|
{refused_rows}

## Data and protocol
- Inputs: NOAA GFS 0.25 deg and GEFS 0.25 deg mean/spread (00Z runs, 52 steps to 240 h) from the public AWS archive.
- Labels: IEM METAR observations (stations in India) for event / range / station-day targets; CHIRPS-2.0 daily 0.05 deg for rain.
- Split (from the training provenance): `{json.dumps(split)}`; trained at {prov.get('trained_at')}.
- Training rows are thinned by the `row_stride` recorded in each model's `metrics.json` provenance (adjacent runs are near-duplicates and the full tables do not fit in memory).

## Known limitations (read before use)
{tails}
- CHIRPS is a satellite-gauge product, not a gauge truth: in our own check it agreed with METAR rain reports only moderately (AUC about 0.81 overall, about 0.71 in the monsoon),
  so rain skill is measured against a noisy label.
- Heat- and cold-wave labels approximate the IMD criteria using station normals computed from the METAR record; they are not IMD declarations.
- Skill differs by zone and lead time; the gate hides numbers where no held-out evidence exists, so some zones will return fewer answers.
- 00Z cycle only; no nowcast from live observations yet; the forecasts are as good as GFS / GEFS on the day.

## Live end-to-end check ({smoke.get('run')})
The served models were run against the newest published run on {len(smoke.get('places', {}))} places (including three sea / out-of-domain queries that must be refused);
problems found: **{len(smoke.get('problems', []))}**.  Details: `live_smoke_report.json`.

## Use
```python
# pip install lightgbm pandas numpy scipy eccodes ; add ./src to PYTHONPATH
from weathergpt_events import static, registry, engine, features, live
import pandas as pd
grids = static.load_static_grids("static_grids.npz")
reg = registry.EventRegistry.from_dir("models")
clim, points = pd.read_parquet("climatology_day.parquet"), pd.read_parquet("points.parquet")
eng = engine.EventEngine(reg, features.FeatureBuilder(grids, clim, points))
run = live.RunStore("cache").load(live.RunStore("cache").latest())      # needs internet (AWS)
print(eng.forecast(28.61, 77.21, run, targets=["rain_curve", "tmax_range"]))
```
Repo: `{repo}`
"""


def main():
    from huggingface_hub import HfApi
    deploy = Path(next(iter(glob.glob(f"{INPUT}/**/deploy/static_grids.npz", recursive=True)))).parent
    smoke = json.loads(Path(next(iter(glob.glob(f"{INPUT}/**/serve_smoke.json", recursive=True)))).read_text())
    token = Path(next(iter(glob.glob(f"{INPUT}/**/hf_token", recursive=True)))).read_text().strip()
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = os.environ.get("HF_REPO") or f"{user}/weathergpt-events"
    print(f"[publish] authenticated as {user}; target repo {repo} (private)", flush=True)

    reg = reg_mod.EventRegistry.from_dir(deploy / "models")
    catalogue = tools.catalogue(reg)
    shutil.rmtree(STAGE, ignore_errors=True)
    (STAGE / "models").mkdir(parents=True)
    (STAGE / "models_not_served").mkdir()
    for name, gate in reg.gates.items():
        src = deploy / "models" / name
        if gate.passed:
            shutil.copytree(src, STAGE / "models" / name)
        else:
            (STAGE / "models_not_served" / name).mkdir()
            shutil.copy(src / "metrics.json", STAGE / "models_not_served" / name / "metrics.json")
    for f in ("static_grids.npz", "points.parquet", "climatology_day.parquet"):
        shutil.copy(deploy / f, STAGE / f)
    (STAGE / "live_smoke_report.json").write_text(json.dumps(smoke, indent=1))
    for rel, text in sys._bundled_sources.items():
        (STAGE / "src" / rel).parent.mkdir(parents=True, exist_ok=True)
        (STAGE / "src" / rel).write_text(text)
    (STAGE / "src" / "event_models" / "__init__.py").write_text(INIT_EVENT_MODELS)
    (STAGE / "src" / "weathergpt_events" / "__init__.py").write_text(INIT_WEATHERGPT_EVENTS)
    (STAGE / "README.md").write_text(build_card(catalogue, reg, smoke, repo))

    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=repo, repo_type="model", commit_message="WeatherGPT event models: served set, static grids, source, generated card")
    files = api.list_repo_files(repo, repo_type="model")
    print(f"[publish] uploaded {len(files)} files to https://huggingface.co/{repo} (private); served {len(reg.models)}, not served {len(reg.gates) - len(reg.models)}", flush=True)
    print("PUBLISH_FILES_BEGIN\n" + "\n".join(sorted(files)) + "\nPUBLISH_FILES_END", flush=True)


if __name__ == "__main__":
    main()
