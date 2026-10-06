"""Kaggle entry point: serve the TRAINED models against the latest LIVE GFS/GEFS run (needs internet).

What it proves, end to end, with real artifacts and no mocks:
  1. the GFS orography / land mask download and the static grids build (also saved for deployment);
  2. the newest published run is fetched with the collectors' own code (52 steps, 10 days);
  3. every trained artifact goes through the admission gate (which are served, which are refused and why);
  4. the engine answers for real places across India and refuses sea / out-of-domain queries;
  5. structural sanity of every number it returns: probabilities in [0, 1], ranges ordered, curves monotone.

Inputs (kernel sources): the data kernels (points tables) and the train-* kernels (models + climatology_day.parquet).
Output: /kaggle/working/deploy/{static_grids.npz, points.parquet, climatology_day.parquet, models/<target>/...}
        /kaggle/working/serve_smoke.json     SERVE_SMOKE_BEGIN / SERVE_SMOKE_END in the log carries the summary.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from event_models import dataset
from weathergpt_events import engine as engine_mod
from weathergpt_events import live
from weathergpt_events import registry as reg_mod
from weathergpt_events import static as static_mod
from weathergpt_events import tools
from weathergpt_events.features import FeatureBuilder

INPUT = "/kaggle/input"
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("serve_smoke_out")
DEPLOY = WORK / "deploy"

PLACES = {                                   # name -> (lat, lon, expectation)
    "Delhi": (28.61, 77.21, "serve"), "Mumbai": (19.08, 72.88, "serve"), "Jaisalmer": (26.92, 70.92, "serve"),
    "Cherrapunji": (25.30, 91.70, "serve"), "Chennai": (13.08, 80.27, "serve"), "Leh": (34.15, 77.58, "serve"),
    "Bengaluru": (12.97, 77.59, "serve"), "Kolkata": (22.57, 88.36, "serve"), "Nagpur": (21.15, 79.09, "serve"),
    "Kochi": (9.93, 76.27, "either"), "Visakhapatnam": (17.69, 83.22, "either"), "Panaji": (15.50, 73.83, "either"),
    "Mangaluru": (12.91, 74.86, "either"), "Puri": (19.81, 85.83, "either"),      # coastal: informational (see land-fraction note)
    "Arabian Sea": (15.0, 68.5, "refuse"), "Bay of Bengal": (12.0, 88.0, "refuse"), "London": (51.5, -0.1, "refuse"),
}


def collect_models() -> dict:
    """Copy every trained target (a folder with metrics.json) from the train-* kernel outputs into one tree."""
    models = DEPLOY / "models"
    models.mkdir(parents=True, exist_ok=True)
    found = {}
    for metrics in sorted(glob.glob(f"{INPUT}/**/event_models/*/metrics.json", recursive=True)):
        src = Path(metrics).parent
        if src.name in found:
            raise RuntimeError(f"target {src.name} appears twice ({found[src.name]} and {src}); attach only one kernel per target")
        found[src.name] = src
        shutil.copytree(src, models / src.name, dirs_exist_ok=True)
    return found


def check_entry(kind: str, values: dict) -> list:
    """Structural problems of one answered entry (empty list == fine)."""
    bad = []
    if kind in ("binary", "xent"):
        p = values.get("probability")
        if p is None or not 0.0 <= p <= 1.0:
            bad.append(f"probability {p}")
    elif kind == "quantile":
        lo, hi = values.get("lo"), values.get("hi")
        qs = [values[k] for k in sorted(values) if k.startswith("q") and values[k] is not None]
        if lo is None or hi is None or lo > hi + 1e-6 or any(q < lo - 1e-6 or q > hi + 1e-6 for q in qs):
            bad.append(f"range lo={lo} hi={hi} q={qs}")
        if qs != sorted(qs):
            bad.append(f"quantiles not ordered {qs}")
    elif kind == "curve":
        curve = [values["exceedance"][k] for k in sorted(values["exceedance"], key=float)]
        if any(c is None or not 0.0 <= c <= 1.0 for c in curve) or any(b > a + 1e-9 for a, b in zip(curve, curve[1:])):
            bad.append("exceedance curve not a monotone probability curve")
    return bad


def main():
    t0 = time.time()
    now = datetime.now(timezone.utc)
    found = collect_models()
    print(f"[serve] {len(found)} trained targets found: {sorted(found)}", flush=True)

    points = pd.read_parquet(dataset.find_dir(INPUT, "points.parquet") / "points.parquet")
    nodes = points[points["kind"] == "node"].reset_index(drop=True)
    stations = points[points["kind"] == "station"]
    print("[serve] training stations land_frac quantiles (0, 5, 10, 25, 50 %):",
          [round(float(stations["land_frac"].quantile(q)), 2) for q in (0, 0.05, 0.1, 0.25, 0.5)],
          "| stations below 0.5:", int((stations["land_frac"] < 0.5).sum()), "of", len(stations),
          "| node land_frac min:", round(float(nodes["land_frac"].min()), 2), flush=True)
    clim_path = glob.glob(f"{INPUT}/**/climatology_day.parquet", recursive=True)
    if not clim_path:
        raise FileNotFoundError("climatology_day.parquet not found in any train kernel output")
    climatology = pd.read_parquet(clim_path[0])
    shutil.copy(clim_path[0], DEPLOY / "climatology_day.parquet")
    points.to_parquet(DEPLOY / "points.parquet", index=False)      # nodes AND stations: the engine needs both (reference node + coastal rule)
    print(f"[serve] points {len(points)} (nodes {len(nodes)}) | climatology rows {len(climatology)} | {time.time() - t0:.0f}s", flush=True)

    store = live.RunStore(WORK / "cache", workers=8)
    run_date = store.latest(now)
    if run_date is None:
        raise RuntimeError("no fresh GFS+GEFS run is published (or the newest is older than MAX_RUN_AGE_H); nothing to serve")
    print(f"[serve] newest fresh run: {run_date} | {time.time() - t0:.0f}s", flush=True)

    grids = live.build_and_save_static(run_date, DEPLOY / "static_grids.npz")
    print(f"[serve] static grids built: {sorted(grids)} | land fraction {float(grids['land'].mean()):.2f} | {time.time() - t0:.0f}s", flush=True)
    run = store.load(run_date)
    print(f"[serve] run loaded: {len(run.base)} base steps, {len(run.ext)} ext steps | {time.time() - t0:.0f}s", flush=True)

    registry = reg_mod.EventRegistry.from_dir(DEPLOY / "models")
    print("[serve] admission gate:", flush=True)
    for row in registry.status():
        print(f"   {row['target']:22s} {'SERVED ' if row['loaded'] else 'REFUSED'} {'' if row['loaded'] else row['reason'][:150]}", flush=True)
    engine = engine_mod.EventEngine(registry, FeatureBuilder(grids, climatology, points))
    catalogue = tools.catalogue(registry)

    report, problems = {"run": run_date.isoformat(), "served": sorted(registry.models), "refused": {
        n: g.reason for n, g in registry.gates.items() if not g.passed}, "places": {}}, []
    for name, (lat, lon, expect) in PLACES.items():
        t1 = time.time()
        out = engine.forecast(lat, lon, run)
        entry = {"available": out["available"], "seconds": round(time.time() - t1, 1)}
        if expect == "either":
            entry["reason"] = out.get("reason")
        elif expect == "refuse":
            if out["available"]:
                problems.append(f"{name}: should have been refused but was answered")
            entry["reason"] = out.get("reason")
        if expect != "refuse":
            if not out["available"]:
                if expect == "serve":
                    problems.append(f"{name}: refused ({out.get('reason')})")
            else:
                summary = {}
                for target, res in out["targets"].items():
                    if not res["available"]:
                        summary[target] = {"available": False, "reason": res["reason"][:120]}
                        continue
                    answered = [e for e in res["entries"] if e["validated"]]
                    for e in answered:
                        for bad in check_entry(res["kind"], e):
                            problems.append(f"{name}/{target}: {bad}")
                    peak = None
                    if answered and res["kind"] in ("binary", "xent"):
                        peak = round(max(e["probability"] for e in answered), 3)
                    summary[target] = {"zone": res["zone"], "entries": len(res["entries"]), "validated": len(answered), "peak_probability": peak}
                entry["zone"] = next((v["zone"] for v in summary.values() if "zone" in v), None)
                entry["targets"] = summary
        report["places"][name] = entry
        print(f"[serve] {name:14s} available={entry['available']} {entry['seconds']}s zone={entry.get('zone')} {entry.get('reason', '')}", flush=True)
    # one full answer, for the record (rain, Delhi)
    full = engine.forecast(*PLACES["Delhi"][:2], run, targets=[t for t in ("rain_curve", "tmax_range") if t in registry.models])
    (WORK / "serve_smoke_example_delhi.json").write_text(json.dumps(full)[:400000])
    # a few real numbers in the log, for a human plausibility check against the weather actually forecast
    for place in ("Delhi", "Cherrapunji"):
        sample = engine.forecast(*PLACES[place][:2], run, targets=[t for t in ("rain_curve", "tmax_range", "rain_7day_total") if t in registry.models])
        for target, res in sample["targets"].items():
            for e in [x for x in res.get("entries", []) if x["validated"]][:3]:
                brief = {k: e[k] for k in ("forecast_day", "start_forecast_day", "q10", "q50", "q90", "lo", "hi", "p_any_rain", "p_rainy_day",
                                          "p_heavy_rain", "amount_if_wet_mm") if k in e}
                print(f"[serve-sample] {place} {target} {brief}", flush=True)
            if res.get("thresholds_without_skill_mm"):
                print(f"[serve-sample] {place} {target} thresholds_without_skill_mm={res['thresholds_without_skill_mm']}", flush=True)
    report["catalogue"] = [{k: c[k] for k in ("name", "available", "horizon_days", "validated_zones")} for c in catalogue]
    report["problems"], report["seconds"] = problems, round(time.time() - t0)
    (WORK / "serve_smoke.json").write_text(json.dumps(report, indent=1))
    print("SERVE_SMOKE_BEGIN\n" + json.dumps(report, indent=1) + "\nSERVE_SMOKE_END", flush=True)
    print(f"[serve] {'PASS' if not problems else 'FAIL: ' + str(len(problems)) + ' problems'} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
