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


# The rain groups read the widest, longest tables (millions of point-days).  A 4x run-date thinning keeps ~240 run dates x
# 2,709 points x 10 days, which is still millions of rows, and fits the 30 GB kernel.
GROUP_ENV = {"train-rain-day": {"ROW_STRIDE": "4"}, "train-rain-window": {"ROW_STRIDE": "4"}}


# Base-only variants: no extension predictors (USE_EXT=0).  They give the first real-data numbers while the extension
# collection is still running, and are the "before" half of the feature-family ablation.
BASE_SOURCES = [x for x in SOURCES if not x.startswith("ext-")]
BASE_GROUPS = {
    "base-events": ["thunderstorm", "fog", "strong_wind", "rain_3h"],
    "base-day": ["hot_day", "heatwave_imd", "tmax_range"],
}


def build(name: str, targets: list, extra_env: dict, entry: str = "run_training.py", sources: list | None = None,
          internet: bool = False, datasets: list | None = None) -> Path:
    out = ROOT / "backup" / "event_models" / name
    out.mkdir(parents=True, exist_ok=True)
    env = {"TARGETS": ",".join(targets), **extra_env} if targets else dict(extra_env)
    args = [sys.executable, str(ROOT / "event_models" / "bundle.py"), str(ROOT / "event_models" / entry),
            str(out / f"{name}.py")] + [a for k, v in env.items() for a in ("--env", f"{k}={v}")]
    subprocess.run(args, check=True, capture_output=True)
    if sources is None:
        sources = BASE_SOURCES if extra_env.get("USE_EXT") == "0" else SOURCES
    meta = {"id": f"{OWNER}/weathergpt-{name}", "title": f"WeatherGPT {name}", "code_file": f"{name}.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": internet, "keywords": ["weather", "india"], "dataset_sources": list(datasets or []),
            "kernel_sources": [f"{OWNER}/weathergpt-{s}" for s in sources],
            "competition_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def build_publish() -> Path:
    """Publish to Hugging Face: the serve-smoke output + the private credentials dataset (hf_token, never printed)."""
    return build("publish-hf", [], {"GFS_LEADSET": "all"}, entry="publish_hf.py", sources=["serve-smoke"], internet=True,
                 datasets=[f"{OWNER}/asanaai-conf-creds"])


def build_serve() -> Path:
    """Serving smoke test: trained models + one data shard (points table) + the live network."""
    return build("serve-smoke", [], {"GFS_LEADSET": "all"}, entry="serve_smoke.py", sources=["gfs-s1"] + list(GROUPS), internet=True)


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
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)   # e.g. --only=train-rain-day,train-rain-window
    paths = {}
    if "--publish" in sys.argv:
        paths["publish-hf"] = build_publish()
    elif "--serve" in sys.argv:
        paths["serve-smoke"] = build_serve()
    elif only:
        paths.update({n: build(n, GROUPS[n], {"ROUNDS": "1500", **GROUP_ENV.get(n, {})}) for n in only})
    elif not only_base:
        paths.update({n: build(n, t, {"ROUNDS": "1500", **GROUP_ENV.get(n, {})}) for n, t in GROUPS.items()})
    if not only and "--serve" not in sys.argv and "--publish" not in sys.argv:
        paths.update({n: build(n, t, {"ROUNDS": "1500", "USE_EXT": "0"}) for n, t in BASE_GROUPS.items()})
    print("built:", ", ".join(paths))
    if "--push" in sys.argv:
        for name, path in paths.items():
            if only or "--serve" in sys.argv or "--publish" in sys.argv or only_base == name.startswith("base-") or not only_base:
                push(path)
