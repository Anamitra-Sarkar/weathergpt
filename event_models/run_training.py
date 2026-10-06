"""Kaggle entry point: build the training tables and fit the selected event models.

Inputs  (kernel sources): weathergpt-truth-collect (METAR + CHIRPS) and weathergpt-gfs-s1..s5.
Env     TARGETS   comma list of target names, or "all"   (lets several kernels split the work)
        ROUNDS    max boosting rounds per model (default 1500)
Output  /kaggle/working/event_models/<target>/{model*.txt, calibrator.json|interval.json, features.json, metrics.json}
        /kaggle/working/event_models/summary_<TAGS>.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from event_models import dataset
from event_models import features_events as fe
from event_models.train import Target, fit_target

INPUT = "/kaggle/input"
OUT = Path("/kaggle/working/event_models") if Path("/kaggle").exists() else Path("event_models_out")

TARGETS = {t.name: t for t in [
    # --- hourly-step events at METAR stations (3 h label window around the valid time) ---
    Target("thunderstorm", "step", "binary", "w_ts_any", primary="cape_ml",
           notes="TS or VCTS reported in the 3 h window around the forecast time"),
    Target("fog", "step", "binary", "w_fog", primary="vis_km", direction=-1, notes="visibility < 1 km"),
    Target("strong_wind", "step", "binary", "w_strong_wind", primary="gust", notes="peak gust-or-mean >= 25 kt"),
    Target("rain_3h", "step", "binary", "w_rain_any", primary="apcp_cell_mean",
           notes="RA/DZ/SHRA/TSRA at the station in the 3 h window"),
    Target("dust", "step", "binary", "w_dust", primary="gust", notes="DU/SA/SS/DS/PO reported; rare"),
    # --- hourly ranges at stations, observed METAR truth ---
    Target("temperature_range", "step", "quantile", "obs_temp_c", point_forecast="t2m_c"),
    Target("wind_range", "step", "quantile", "obs_wind_ms", point_forecast="wind10"),
    Target("humidity_range", "step", "quantile", "obs_rh", point_forecast="rh2m"),
    # --- daily temperature at stations (IST day) ---
    Target("hot_day", "day", "binary", "hot_day", primary="t2m_max", notes="IST-day Tmax >= 40 C"),
    Target("cold_night", "day", "binary", "cold_night", primary="t2m_min", direction=-1, notes="IST-day Tmin <= 5 C"),
    Target("heatwave_imd", "day", "binary", "heatwave", primary="t2m_max",
           notes="IMD heat wave: plains Tmax>=40 C and +4.5 C over the station normal (or >=45 C); hills >=30 C and +4.5 C"),
    Target("coldwave_imd", "day", "binary", "coldwave", primary="t2m_min", direction=-1,
           notes="IMD cold wave: plains Tmin<=10 C and -4.5 C under the normal (or <=4 C); hills <=0 C and -4.5 C"),
    Target("tmax_range", "day", "quantile", "tmax_c", point_forecast="t2m_max"),
    Target("tmin_range", "day", "quantile", "tmin_c", point_forecast="t2m_min"),
    # --- whole-India rain: ONE coherent exceedance curve P(share of 0.5 deg cell >= t) over 16 thresholds
    #     (monotone in t by construction); "rain yes/no" and "how much" are both derived from it ---
    Target("rain_curve", "rain", "curve", dataset.thr_col(1.0),
           labels=tuple(dataset.thr_col(t) for t in dataset.RAIN_THRESHOLDS), thresholds=dataset.RAIN_THRESHOLDS,
           score_low="rain_cell_mean_mm", score_high="rain_cell_max_mm",
           notes="P(random point in the 0.5 deg cell >= t mm in the UTC day); t = 0.1 ... 204.5 incl. IMD boundaries"),
    # --- multi-day windows are NOT products of daily probabilities: the union and the total are learned directly ---
    Target("rain_3day_any_2p5mm", "rain3", "xent", dataset.thr_col(2.5, "frac_any3_ge_"), primary="sum3_rain_cell_mean_mm",
           notes="P(point reaches >= 2.5 mm on at least one of 3 consecutive UTC days)"),
    Target("rain_3day_any_15mm", "rain3", "xent", dataset.thr_col(15.6, "frac_any3_ge_"), primary="max3_rain_cell_max_mm",
           notes="P(point reaches >= 15.6 mm on at least one of 3 consecutive UTC days)"),
    Target("rain_3day_total", "rain3", "curve", dataset.thr_col(10.0, "frac_sum3_ge_"),
           labels=tuple(dataset.thr_col(t, "frac_sum3_ge_") for t in dataset.WINDOW_SPECS[3]["sum"]),
           thresholds=dataset.WINDOW_SPECS[3]["sum"], score_low="sum3_rain_cell_mean_mm", score_high="max3_rain_cell_max_mm",
           key_thresholds=(10.0, 25.0, 50.0), wet=2.5,
           notes="P(3-day total at a point >= t mm); exceedance curve, monotone in t"),
    Target("rain_7day_any_2p5mm", "rain7", "xent", dataset.thr_col(2.5, "frac_any7_ge_"), primary="sum7_rain_cell_mean_mm",
           notes="P(point reaches >= 2.5 mm on at least one of 7 consecutive UTC days)"),
    Target("rain_7day_total", "rain7", "curve", dataset.thr_col(25.0, "frac_sum7_ge_"),
           labels=tuple(dataset.thr_col(t, "frac_sum7_ge_") for t in dataset.WINDOW_SPECS[7]["sum"]),
           thresholds=dataset.WINDOW_SPECS[7]["sum"], score_low="sum7_rain_cell_mean_mm", score_high="max7_rain_cell_max_mm",
           key_thresholds=(25.0, 50.0, 100.0), wet=10.0,
           notes="P(7-day total at a point >= t mm); 'how much rain this week' as one coherent distribution"),
]}


lead_bucket = fe.lead_bucket


ARTIFACTS: dict = {}          # serving needs these (forecast climatologies); main() writes them next to the models
ROW_STRIDE = int(os.environ.get("ROW_STRIDE", "2"))   # thin the huge point-day tables further (adjacent runs are near-duplicates)


def _ext_present() -> bool:
    mode = os.environ.get("USE_EXT", "auto")
    if mode == "0":
        return False
    found = bool(glob.glob(f"{INPUT}/**/p1x_station_steps_*.parquet", recursive=True))
    if mode == "1" and not found:
        raise FileNotFoundError("USE_EXT=1 but no extension tables under the kernel inputs")
    return found


def _thin_dates(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1:
        return frame
    keep = sorted(pd.to_datetime(frame["run_date"]).unique())[::stride]
    return frame[pd.to_datetime(frame["run_date"]).isin(keep)]


def build_tables(needed: set):
    t0 = time.time()
    truth_dir = dataset.find_dir(INPUT, "stations.parquet")
    chirps_dir = dataset.find_dir(INPUT, "grid.json")
    points = pd.read_parquet(dataset.find_dir(INPUT, "points.parquet") / "points.parquet")
    use_ext = _ext_present()
    if use_ext:
        points = points.merge(dataset.read_all(INPUT, "points_ext.parquet", ["point_id"]), on="point_id", how="left")
    terrain = fe.terrain_columns(points)
    points["zone"] = fe.region_of(points["lat"], points["lon"], points["elevation_m"], points["dist_coast_km"])
    points["held_out"] = fe.holdout_blocks(points["lat"].to_numpy(), points["lon"].to_numpy(), zone=points["zone"].to_numpy())
    print("[tables] points", len(points), "| extension:", use_ext, "| terrain cols:", len(terrain), "| held out:",
          int(points["held_out"].sum()), "| by zone:", points.groupby("zone")["held_out"].agg(["sum", "size"]).to_dict("index"), flush=True)
    held = dict(zip(points["point_id"], points["held_out"]))
    ref_map = fe.nearest_node_ids(points)
    tables, features = {}, {}
    keys2 = ["point_id", "run_date", "day_k"]

    # ---- ONE forecast climatology, from the UTC-day aggregates of every node and station (train-period forecasts only).
    # Every place -- node, station, or an arbitrary query at serving time -- is referenced to its nearest grid node.
    light = dataset.read_all(INPUT, "p2_point_days_*.parquet", keys2, columns=keys2 + [c for c in fe.DAY_CLIM_COLS if c != "rain_mm"] + ["rain_mm"])
    clim = fe.fit_climatology(light, fe.DAY_CLIM_COLS, "day")
    ARTIFACTS["climatology_day"] = clim.reset_index()
    anom_cols = [f"{c}_anom" for c in clim.columns]
    station_anoms = fe.apply_climatology(light[light["point_id"].str.startswith("S:")], clim, "day", ref_map)
    del light

    if needed & {"step", "day"}:
        p1 = dataset.read_all(INPUT, "p1_station_steps_*.parquet", ["point_id", "run_date", "lead_h"])
        ext_step_cols = []
        if use_ext:
            keys = ["point_id", "run_date", "lead_h"]
            p1x = dataset.read_all(INPUT, "p1x_station_steps_*.parquet", keys)
            ext_step_cols = dataset.ext_columns(p1x, keys)
            p1 = dataset.merge_ext(p1, p1x, keys, how="inner")      # only rows that have ALL their inputs
            del p1x
        print("[tables] p1 rows", len(p1), "| extension columns", len(ext_step_cols), flush=True)
    if "step" in needed:
        day_anom = station_anoms[keys2 + [f"{c}_anom" for c in fe.STEP_DAY_ANOM_COLS if f"{c}_anom" in station_anoms]].copy()
        day_anom = day_anom.rename(columns={f"{c}_anom": f"day_{c}_anom" for c in fe.STEP_DAY_ANOM_COLS}).rename(columns={"day_k": "day_k_valid"})
        p1s = p1.assign(day_k_valid=((p1["lead_h"] - 1) // 24).astype("int8"))
        p1s = p1s.merge(day_anom, on=["point_id", "run_date", "day_k_valid"], how="left").drop(columns=["day_k_valid"])
        hourly = dataset.load_metar_hourly(truth_dir)
        table = dataset.station_step_table(fe.step_context(p1s), hourly, half_width=1, min_cover=2)
        table = fe.build_step_features(table, points)
        table["month"] = table["valid_time"].dt.month
        table["lead_bucket"] = lead_bucket(table["lead_h"])
        table["split"] = fe.split_labels(table["run_date"], table["point_id"].map(held).to_numpy(bool))
        tables["step"] = table
        features["step"] = fe.step_feature_names(ext_step_cols, fe.step_anom_names(), terrain)
        del p1s
        print("[tables] step", len(table), dict(pd.Series(table["split"]).value_counts()), f"{time.time() - t0:.0f}s", flush=True)
    if "day" in needed:
        daily = dataset.load_metar_daily(truth_dir)
        normals = dataset.station_normals(daily)
        elev = points.set_index("point_id")["station_elev_m"].fillna(points.set_index("point_id")["elevation_m"]) \
            if "station_elev_m" in points else points.set_index("point_id")["elevation_m"]
        labelled = dataset.imd_wave_labels(daily, normals, elev)
        agg = fe.apply_climatology(dataset.ist_day_aggregates(p1), clim, "day", ref_map)
        table = agg.merge(labelled.drop(columns=["doy"]), on=["point_id", "ist_day"], how="inner")
        table = dataset.filter_complete_ist_days(table)
        for flag in ("hot_day", "severe_heat", "cold_night"):
            table[flag] = table[flag].astype(float)
        table = fe.build_day_features(table, points, "ist_day")
        table["month"] = pd.to_datetime(table["ist_day"]).dt.month
        table["lead_bucket"] = table["day_k"].map(lambda d: f"day{d}")
        table["split"] = fe.split_labels(table["run_date"], table["point_id"].map(held).to_numpy(bool))
        day_ext = [c for c in sorted(ext_agg_columns(ext_step_cols)) if c in table.columns]
        tables["day"] = table
        features["day"] = fe.day_feature_names("station_day", day_ext, [a for a in anom_cols if a in table.columns], terrain)
        print("[tables] day", len(table), dict(pd.Series(table["split"]).value_counts()),
              f"| ext {len(day_ext)} anom {len(anom_cols)}", f"{time.time() - t0:.0f}s", flush=True)
    if needed & {"rain", "rain3", "rain7"}:
        # The point-day tables are wide and huge (millions of rows): thin to the kept run dates and drop to float32 while
        # reading, file by file.  The kept dates are exactly what thinning the merged table used to keep.
        dates = dataset.run_dates(INPUT, "p2_point_days_*.parquet")
        if use_ext:
            dates &= dataset.run_dates(INPUT, "p2x_point_days_*.parquet")
        keep = set(sorted(dates)[::ROW_STRIDE]) if use_ext and ROW_STRIDE > 1 else None
        p2 = dataset.read_thinned(INPUT, "p2_point_days_*.parquet", keys2, keep)
        ext_day_cols = []
        if use_ext:
            p2x = dataset.read_thinned(INPUT, "p2x_point_days_*.parquet", keys2, keep)
            ext_day_cols = dataset.ext_columns(p2x, keys2)
            p2 = dataset.merge_ext(p2, p2x, keys2, how="inner")
            del p2x
        truth = dataset.chirps_cell_truth(chirps_dir, points)
        # keep only the label columns the requested products use (the 3/7-day window columns are most of the width)
        window_cols = [c for c in truth.columns if c.startswith(("frac_any", "frac_sum"))]
        if not needed & {"rain3", "rain7"}:
            truth = truth.drop(columns=window_cols)
        elif "rain" not in needed:
            truth = truth.drop(columns=[c for c in truth.columns if c.startswith("frac_ge_")])
        print("[tables] p2 rows", len(p2), "| cols", p2.shape[1], f"| {p2.memory_usage().sum() / 1e9:.1f} GB", "| chirps truth rows", len(truth), f"{time.time() - t0:.0f}s", flush=True)
    if "rain" in needed:
        p2a = fe.apply_climatology(p2, clim, "day", ref_map)
        if not needed & {"rain3", "rain7"}:
            del p2                                  # nothing else needs it: free it before the big merge
        table = dataset.point_day_rain_table(p2a, truth, inplace=True)
        table = table[table["rain_buckets"] >= 4].copy()
        table = fe.build_day_features(table, points, "valid_date")
        table["month"] = pd.to_datetime(table["valid_date"]).dt.month
        table["lead_bucket"] = table["day_k"].map(lambda d: f"day{d}")
        table["split"] = fe.split_labels(table["run_date"], table["point_id"].map(held).to_numpy(bool))
        tables["rain"] = table
        features["rain"] = fe.day_feature_names("rain", ext_day_cols, anom_cols, terrain)
        del p2a
        print("[tables] rain", len(table), dict(pd.Series(table["split"]).value_counts()), f"{time.time() - t0:.0f}s", flush=True)
    for name, window in (("rain3", 3), ("rain7", 7)):
        if name not in needed:
            continue
        if int(p2["day_k"].max()) < window - 1:
            print(f"[tables] {name}: skipped, the forecast data only reaches day {int(p2['day_k'].max())} "
                  f"(a {window}-day window needs day {window - 1})", flush=True)
            continue
        day_cols = dataset.window_day_columns(p2.columns)
        table = dataset.rain_window_table(p2, truth, window=window, day_cols=day_cols)
        table = fe.build_day_features(table, points, "valid_date")
        table["month"] = pd.to_datetime(table["valid_date"]).dt.month
        table["lead_bucket"] = table["day_k"].map(lambda d: f"start_day{d}")
        table["split"] = fe.split_labels(table["run_date"], table["point_id"].map(held).to_numpy(bool))
        tables[name] = table
        features[name] = (dataset.window_feature_names(window, day_cols) + ["day_k", "sin_doy", "cos_doy"]
                          + fe.STATIC + terrain + ["zone_code"])
        print(f"[tables] {name}", len(table), dict(pd.Series(table["split"]).value_counts()), f"{time.time() - t0:.0f}s", flush=True)
    return tables, features


def ext_agg_columns(ext_step_cols: list) -> set:
    """Names the IST-day aggregator produces from the extension step columns."""
    out = set()
    for names, suffix in ((dataset.IST_EXT_MEAN, "mean"), (dataset.IST_EXT_MAX, "max"), (dataset.IST_EXT_MIN, "min")):
        out |= {f"{c}_{suffix}" for c in names if c in ext_step_cols}
    if "rh700" in ext_step_cols:
        out.add("rh700_min")
    for label in ("tmax", "tmin"):
        if f"avg_{label}2m_c" in ext_step_cols:
            out |= {f"gefs_{label}_mean_c", f"gefs_{label}_spread_c"}
    return out


ALGORITHM_VERSION = "events-v1"


def brief(report: dict) -> dict:
    """One-line honest headline per target for the kernel log (test_space = places never seen, future dates)."""
    if "skipped" in report:
        return {"skipped": report["skipped"]}
    out = {}
    for split in ("test_time", "test_space"):
        m = report.get("metrics", {}).get(split, {})
        if report["kind"] == "curve":
            keys = report.get("metrics", {}).get("test_space", {}).get("detail_key_thresholds", {}).keys()
            out[split] = {f"t{k}": {x: m["thresholds"][k].get(x) for x in ("bss_vs_zone_month", "bss_vs_gfs_calibrated", "auc_any_pixel")}
                          for k in list(keys)[:3] if k in m.get("thresholds", {})}
            if "amount_if_wet_median" in m:
                out[split]["amount_within_x2"] = m["amount_if_wet_median"]["within_factor_2"]
        elif report["kind"] == "quantile":
            out[split] = {x: m.get(x) for x in ("coverage_conformal", "median_mae", "gfs_raw_mae")}
        else:
            out[split] = {x: m.get(x) for x in ("n", "auc", "bss_vs_zone_month", "bss_vs_global")}
    return out


def dataset_fingerprint() -> dict:
    """What the models learned from: a hash over the names and sizes of every input data file, plus counts.
    (Names+sizes, not contents: the inputs are tens of GB; any re-collection changes at least one size.)"""
    files = sorted(glob.glob(f"{INPUT}/**/*.parquet", recursive=True) + glob.glob(f"{INPUT}/**/*.npz", recursive=True))
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{os.path.relpath(path, INPUT)}:{os.path.getsize(path)}\n".encode())
    return {"sha256": digest.hexdigest(), "n_files": len(files), "total_bytes": int(sum(os.path.getsize(f) for f in files))}


def provenance_block() -> dict:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "dataset_kind": "gfs_gefs_forecasts_vs_observed_metar_chirps",
        "dataset_sha256": (fp := dataset_fingerprint())["sha256"], "dataset_files": fp["n_files"],
        "split": {"train_end": str(fe.TRAIN_END.date()), "val": [str(fe.VAL_START.date()), str(fe.VAL_END.date())],
                  "test_start": str(fe.TEST_START.date()), "spatial_holdout": "20% of 4x4 degree blocks",
                  "gap_days": 5, "row_stride": ROW_STRIDE},
        "sources": ["IEM METAR (IN__ASOS)", "CHIRPS-2.0 daily p05", "NOAA GFS 0.25 (AWS)", "NOAA GEFS 0.25 mean/spread (AWS)"],
        "use_extension": _ext_present(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    selected = os.environ.get("TARGETS", "all")
    names = list(TARGETS) if selected == "all" else [n.strip() for n in selected.split(",")]
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown targets: {unknown}; known: {list(TARGETS)}")
    rounds = int(os.environ.get("ROUNDS", "1500"))
    OUT.mkdir(parents=True, exist_ok=True)
    tables, features = build_tables({TARGETS[n].table for n in names})
    for artifact, frame in ARTIFACTS.items():            # forecast climatologies the serving side needs
        frame.to_parquet(OUT / f"{artifact}.parquet", index=False)
    summary = {}
    provenance = provenance_block()
    for name in names:
        target = TARGETS[name]
        if target.table not in tables:
            summary[name] = {"skipped": f"table '{target.table}' unavailable for the data collected so far"}
            print(f"[run] {name}: table '{target.table}' unavailable", flush=True)
            continue
        frame, feats = tables[target.table], features[target.table]
        missing = [c for c in feats + [target.label] if c not in frame.columns]
        if missing:
            summary[name] = {"error": f"missing columns {missing[:6]}"}
            print(f"[run] {name}: missing columns {missing[:6]}", flush=True)
            continue
        started = time.time()
        report = fit_target(target, frame, feats, frame["split"].to_numpy(), OUT / name, rounds=rounds)
        report["seconds"] = round(time.time() - started)
        if "skipped" not in report:
            report["provenance"] = provenance
            (OUT / name / "metrics.json").write_text(json.dumps(report, indent=1, default=float))   # provenance travels with the artifact
        summary[name] = report
        print(f"[run] {name}: {brief(report)}", flush=True)
    tag = "_".join(names)[:60]
    (OUT / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    print("TRAIN_SUMMARY_BEGIN")
    print(json.dumps(summary, indent=1, default=float))
    print("TRAIN_SUMMARY_END")


if __name__ == "__main__":
    main()
