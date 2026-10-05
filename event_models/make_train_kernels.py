"""Create (and optionally push) the Kaggle training kernels.

Each kernel bundles event_models/run_training.py, attaches every collection kernel as an input, and trains one
group of targets.  Groups are split by memory footprint: the point-day rain tables are far bigger than the
station tables, so the daily curve and the multi-day windows get their own kernels.

    python event_models/make_train_kernels.py            # write the bundles + metadata
    python event_models/make_train_kernels.py --push     # ... and push, retrying until a CPU slot frees
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "anamitrasarkar007"
SOURCES = ["truth-collect", "gfs-s1", "gfs-s2", "gfs-s2b", "gfs-s3", "gfs-s4", "gfs-s5", "gfs-l1", "gfs-l2",
           "ext-e1", "ext-e2", "ext-e3", "ext-e4"]
GROUPS = {
    "train-events": ["thunderstorm", "fog", "strong_wind", "rain_3h", "dust"],
    "train-ranges": ["temperature_range", "wind_range", "humidity_range"],
    "train-day": ["hot_day", "cold_night", "heatwave_imd", "coldwave_imd", "tmax_range", "tmin_range"],
    "train-rain-day": ["rain_curve"],
    "train-rain-window": ["rain_3day_any_2p5mm", "rain_3day_any_15mm", "rain_3day_total", "rain_7day_any_2p5mm",
                          "rain_7day_total"],
}


# Base-only variants: no extension predictors (USE_EXT=0).  They give the first real-data numbers while the extension
# collection is still running, and are the "before" half of the feature-family ablation.
BASE_SOURCES = [x for x in SOURCES if not x.startswith("ext-")]
BASE_GROUPS = {
    "base-events": ["thunderstorm", "fog", "strong_wind", "rain_3h"],
    "base-day": ["hot_day", "heatwave_imd", "tmax_range"],
}


def build(name: str, targets: list, extra_env: dict) -> Path:
    out = ROOT / "backup" / "event_models" / name
    out.mkdir(parents=True, exist_ok=True)
    env = {"TARGETS": ",".join(targets), **extra_env}
    args = [sys.executable, str(ROOT / "event_models" / "bundle.py"), str(ROOT / "event_models" / "run_training.py"),
            str(out / f"{name}.py")] + [a for k, v in env.items() for a in ("--env", f"{k}={v}")]
    subprocess.run(args, check=True, capture_output=True)
    meta = {"id": f"{OWNER}/weathergpt-{name}", "title": f"WeatherGPT {name}", "code_file": f"{name}.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "keywords": ["weather", "india"], "dataset_sources": [],
            "kernel_sources": [f"{OWNER}/weathergpt-{s}" for s in (BASE_SOURCES if extra_env.get("USE_EXT") == "0" else SOURCES)],
            "competition_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def push(path: Path):
    while True:
        result = subprocess.run(["kaggle", "kernels", "push", "-p", str(path)], capture_output=True, text=True)
        last = (result.stdout + result.stderr).strip().splitlines()[-1]
        if "successfully pushed" in last:
            print(f"{time.strftime('%H:%M:%S')} pushed {path.name}", flush=True)
            return
        print(f"{time.strftime('%H:%M:%S')} {path.name}: {last[:100]} -- retrying", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    only_base = "--base" in sys.argv
    paths = {}
    if not only_base:
        paths.update({n: build(n, t, {"ROUNDS": "1500"}) for n, t in GROUPS.items()})
    paths.update({n: build(n, t, {"ROUNDS": "1500", "USE_EXT": "0"}) for n, t in BASE_GROUPS.items()})
    print("built:", ", ".join(paths))
    if "--push" in sys.argv:
        for name, path in paths.items():
            if only_base == name.startswith("base-") or not only_base:
                push(path)
