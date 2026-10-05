from __future__ import annotations
import os as _os
import sys as _sys, types as _types
_pkg = _types.ModuleType('event_models'); _pkg.__path__ = []; _sys.modules['event_models'] = _pkg
def _load(name, src):
    m = _types.ModuleType('event_models.' + name); _sys.modules['event_models.' + name] = m
    setattr(_pkg, name, m); exec(compile(src, 'event_models/' + name + '.py', 'exec'), m.__dict__)
_load('dataset', '"""Join GFS forecasts (collect_gfs) with observed truth (collect_truth) into training tables.\n\nThree tables, one per kind of target:\n\n  station_step_table   (station, run, lead)  hourly-step events + continuous truth, METAR\n  station_day_table    (station, run, IST day) Tmax/Tmin + hot-day / cold-night, METAR\n  point_day_rain_table (point, run, day_k)   areal rain truth from CHIRPS, whole India\n\nLeakage rules that live HERE so every model inherits them:\n  * features only ever come from the forecast run (issued at 00Z of `run_date`);\n    nothing observed after the run is a feature;\n  * labels are windows around the forecast valid time, never around the run time;\n  * a label whose window has no observation is MISSING (row dropped), never "no event".\n"""\nfrom __future__ import annotations\n\nimport glob\nimport json\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nIST = pd.Timedelta(hours=5, minutes=30)\nEVENT_FLAGS = ("ts_any", "ts_at", "fog", "dense_fog", "strong_wind", "gale", "rain_any",\n               "rain_heavy", "dust", "squall")\nCHIRPS_THRESHOLDS = (1.0, 2.5, 15.6, 64.5)   # mm/day: wet, IMD rainy day, moderate, heavy\n\n\ndef find_dir(root: str, marker: str) -> Path:\n    """Locate a kernel-output folder under /kaggle/input whatever the mount nesting is."""\n    hits = glob.glob(f"{root}/**/{marker}", recursive=True)\n    if not hits:\n        raise FileNotFoundError(f"{marker} not found under {root}")\n    return Path(hits[0]).parent\n\n\ndef read_all(root: str, pattern: str, dedupe_on: list | None = None) -> pd.DataFrame:\n    """Concatenate every file matching `pattern` anywhere under `root` (all shard kernels\' outputs)."""\n    files = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))\n    if not files:\n        raise FileNotFoundError(f"no files match {pattern} under {root}")\n    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)\n    return frame.drop_duplicates(subset=dedupe_on) if dedupe_on else frame\n\n\ndef read_parts(folder: Path, pattern: str) -> pd.DataFrame:\n    parts = sorted(folder.glob(pattern))\n    if not parts:\n        raise FileNotFoundError(f"no files match {pattern} in {folder}")\n    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)\n\n\n# ----------------------------------------------------------------- METAR side\ndef load_metar_hourly(truth_dir: Path) -> pd.DataFrame:\n    frames = []\n    for path in sorted((truth_dir / "metar_hourly").glob("*.parquet")):\n        frame = pd.read_parquet(path)\n        frame["point_id"] = "S:" + path.stem\n        frames.append(frame)\n    out = pd.concat(frames, ignore_index=True)\n    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True).dt.tz_localize(None)\n    return out\n\n\ndef window_labels(hourly: pd.DataFrame, half_width: int = 1) -> pd.DataFrame:\n    """Per station: label = OR of the flag over hours [T-h, T+h]; `cover` = hours observed in it.\n\n    Operates on the full hourly grid so a gap in reporting shrinks `cover` instead\n    of silently shifting the window.\n    """\n    flags = [c for c in EVENT_FLAGS if c in hourly.columns]\n    out = []\n    for point_id, g in hourly.groupby("point_id", sort=False):\n        g = g.set_index("hour_utc").sort_index()\n        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))\n        observed = full["n_obs"].fillna(0) > 0\n        size = 2 * half_width + 1\n        res = pd.DataFrame(index=full.index)\n        res["cover"] = observed.astype(int).rolling(size, center=True, min_periods=1).sum()\n        for flag in flags:\n            res[f"w_{flag}"] = (full[flag].astype("float64").fillna(0.0)\n                                .rolling(size, center=True, min_periods=1).max())\n        # exact-hour continuous truth\n        for col in ("temp_c", "rh", "wind_ms", "peak_wind_ms", "vis_km"):\n            if col in full:\n                res[f"obs_{col}"] = full[col]   # prefixed: forecast tables have their own vis_km etc.\n        res["n_obs_hour"] = full["n_obs"].fillna(0)\n        res["point_id"] = point_id\n        out.append(res.rename_axis("valid_hour").reset_index())\n    return pd.concat(out, ignore_index=True)\n\n\ndef station_step_table(p1: pd.DataFrame, hourly: pd.DataFrame, half_width: int = 1,\n                       min_cover: int = 2) -> pd.DataFrame:\n    """Forecast step rows + METAR labels at the valid time.  Rows without enough observed hours are dropped."""\n    labels = window_labels(hourly, half_width)\n    p1 = p1.copy()\n    p1["valid_time"] = pd.to_datetime(p1["run_date"]) + pd.to_timedelta(p1["lead_h"].astype(int), unit="h")\n    p1["valid_hour"] = p1["valid_time"].dt.floor("h")\n    clash = (set(p1.columns) & set(labels.columns)) - {"point_id", "valid_hour"}\n    if clash:   # pandas would silently rename both to _x/_y and a feature would vanish\n        raise ValueError(f"forecast/observation column collision: {sorted(clash)}")\n    merged = p1.merge(labels, on=["point_id", "valid_hour"], how="inner")\n    merged = merged[merged["cover"] >= min_cover].copy()\n    return merged\n\n\n# ----------------------------------------------------------- daily temperature\ndef ist_day_aggregates(p1: pd.DataFrame) -> pd.DataFrame:\n    p1 = p1.copy()\n    valid = pd.to_datetime(p1["run_date"]) + pd.to_timedelta(p1["lead_h"].astype(int), unit="h")\n    p1["ist_day"] = (valid + IST).dt.floor("D")\n    p1["valid_hour_utc"] = valid.dt.hour\n    grouped = p1.groupby(["point_id", "run_date", "ist_day"], sort=False)\n    agg = grouped.agg(\n        n_steps=("lead_h", "size"),\n        t2m_max=("t2m_c", "max"), t2m_min=("t2m_c", "min"), t2m_mean=("t2m_c", "mean"),\n        dpt_mean=("dpt2m_c", "mean"), rh_min=("rh2m", "min"), rh_mean=("rh2m", "mean"),\n        wind_mean=("wind10", "mean"), gust_max=("gust", "max"), cape_max=("cape", "max"),\n        tcc_mean=("tcc", "mean"), pwat_mean=("pwat", "mean"), hpbl_max=("hpbl", "max"),\n        mslp_mean=("mslp_hpa", "mean"), lead_min=("lead_h", "min"), lead_max=("lead_h", "max"),\n    ).reset_index()\n    agg["day_k"] = (agg["ist_day"] - pd.to_datetime(agg["run_date"])).dt.days\n    return agg\n\n\ndef station_day_table(p1: pd.DataFrame, daily_truth: pd.DataFrame) -> pd.DataFrame:\n    agg = ist_day_aggregates(p1)\n    truth = daily_truth.copy()\n    truth["ist_day"] = pd.to_datetime(truth["ist_day"])\n    return agg.merge(truth, on=["point_id", "ist_day"], how="inner")\n\n\ndef load_metar_daily(truth_dir: Path) -> pd.DataFrame:\n    frames = []\n    for path in sorted((truth_dir / "metar_daily").glob("*.parquet")):\n        frame = pd.read_parquet(path)\n        frame["point_id"] = "S:" + path.stem\n        frames.append(frame)\n    return pd.concat(frames, ignore_index=True)\n\n\n# ---------------------------------------------------------------- CHIRPS truth\ndef chirps_cell_truth(chirps_dir: Path, points: pd.DataFrame, half_cell_deg: float = 0.25) -> pd.DataFrame:\n    """Areal rain truth for each point\'s cell (+-half_cell_deg box = 10x10 CHIRPS pixels).\n\n    Per (point, UTC day): cell mean, cell max, and the FRACTION of the cell\'s pixels at or above\n    each threshold.  That fraction is the probability that a random location in the cell sees\n    that much rain -- the quantity a "will it rain where I am" model should be calibrated to.\n    """\n    grid = json.loads((chirps_dir / "grid.json").read_text())\n    step, top, left = grid["step"], grid["lat_north"], grid["lon_west"]\n    half = int(round(half_cell_deg / step))\n    rows = np.round((top - points["lat"].to_numpy()) / step).astype(int)\n    cols = np.round((points["lon"].to_numpy() - left) / step).astype(int)\n    out = []\n    for path in sorted(chirps_dir.glob("chirps_india_*.npz")):\n        blob = np.load(path)\n        data, dates = blob["data"].astype("float32"), pd.to_datetime(blob["dates"])\n        n_days, n_rows, n_cols = data.shape\n        for i, point_id in enumerate(points["point_id"].to_numpy()):\n            r0, r1 = max(rows[i] - half, 0), min(rows[i] + half, n_rows)\n            c0, c1 = max(cols[i] - half, 0), min(cols[i] + half, n_cols)\n            if r1 <= r0 or c1 <= c0:\n                continue\n            cell = data[:, r0:r1, c0:c1].reshape(n_days, -1)\n            valid = np.isfinite(cell)\n            n_valid = valid.sum(axis=1)\n            if n_valid.max() < 0.5 * cell.shape[1]:   # mostly ocean / outside CHIRPS coverage\n                continue\n            filled = np.where(valid, cell, 0.0)\n            row = {"point_id": point_id, "valid_date": dates, "cell_valid_px": n_valid,\n                   "chirps_mean_mm": np.where(n_valid > 0, filled.sum(axis=1) / np.maximum(n_valid, 1), np.nan),\n                   "chirps_max_mm": np.where(n_valid > 0, np.where(valid, cell, -1).max(axis=1), np.nan)}\n            for t in CHIRPS_THRESHOLDS:\n                row[f"frac_ge_{str(t).replace(\'.\', \'p\')}"] = np.where(\n                    n_valid > 0, ((cell >= t) & valid).sum(axis=1) / np.maximum(n_valid, 1), np.nan)\n            out.append(pd.DataFrame(row))\n    return pd.concat(out, ignore_index=True)\n\n\ndef point_day_rain_table(p2: pd.DataFrame, truth: pd.DataFrame, min_valid_frac: float = 0.5) -> pd.DataFrame:\n    p2 = p2.copy()\n    p2["valid_date"] = pd.to_datetime(p2["run_date"]) + pd.to_timedelta(p2["day_k"].astype(int), unit="D")\n    merged = p2.merge(truth, on=["point_id", "valid_date"], how="inner")\n    merged = merged[merged["cell_valid_px"] >= min_valid_frac * 100].copy()\n    return merged\n')
"""Validate the collected GFS forecasts against REAL observations before scaling up.

Reads the outputs of `weathergpt-truth-collect` (METAR + CHIRPS) and
`weathergpt-gfs-smoke` (3 requested run dates, 2 of which exist) as kernel inputs, and checks that the
forecasts we collected are physically consistent with what happened:

  1. temperature / wind / dew point at stations vs METAR, by lead  (bias, RMSE, corr)
  2. thunderstorm label vs GFS CAPE / lifted index / reflectivity   (AUC, base rate)
  3. rain label vs GFS rain bucket                                  (AUC)
  4. whole-India daily rain: GFS cell rain vs CHIRPS cell rain      (corr, bias)
  5. table health: row counts, NaN rates, value ranges, join survival

Prints a report; any check that fails is printed as FAIL rather than raising, so
one broken field does not hide the rest.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from event_models import dataset

warnings.filterwarnings("ignore")
INPUT = "/kaggle/input"


def auc(y, score):
    from sklearn.metrics import roc_auc_score
    y, score = np.asarray(y), np.asarray(score)
    ok = np.isfinite(score)
    if y[ok].min() == y[ok].max():
        return float("nan")
    return float(roc_auc_score(y[ok], score[ok]))


def main():
    truth_dir = dataset.find_dir(INPUT, "stations.parquet")
    gfs_dir = dataset.find_dir(INPUT, "points.parquet")
    chirps_dir = dataset.find_dir(INPUT, "grid.json")
    print("truth_dir", truth_dir, "| gfs_dir", gfs_dir, "| chirps_dir", chirps_dir)
    report: dict = {}

    points = pd.read_parquet(gfs_dir / "points.parquet")
    p1 = dataset.read_parts(gfs_dir, "p1_station_steps_*.parquet")
    p2 = dataset.read_parts(gfs_dir, "p2_point_days_*.parquet")
    report["tables"] = {"points": len(points), "p1_rows": len(p1), "p2_rows": len(p2),
                        "run_dates": sorted(pd.to_datetime(p1["run_date"]).dt.strftime("%Y-%m-%d").unique().tolist())}
    nan_rate = p1.drop(columns=["point_id", "run_date"]).isna().mean().round(3)
    report["p1_nan_rate_nonzero"] = nan_rate[nan_rate > 0].to_dict()
    ranges = p1[["t2m_c", "rh2m", "wind10", "gust", "cape", "vis_km", "mslp_hpa", "pwat", "apcp_mm"]].describe().loc[["min", "50%", "max"]].round(2)
    report["p1_ranges"] = ranges.to_dict()

    hourly = dataset.load_metar_hourly(truth_dir)
    table = dataset.station_step_table(p1, hourly, half_width=1, min_cover=2)
    report["station_step_rows"] = {"forecast_rows": len(p1), "after_join": len(table),
                                   "survival": round(len(table) / max(len(p1), 1), 3),
                                   "stations_joined": int(table["point_id"].nunique())}

    # 1 continuous variables by lead bucket
    cont = {}
    table["lead_bucket"] = pd.cut(table["lead_h"], [0, 24, 48, 72, 120], labels=["d1", "d2", "d3", "d4-5"])
    for name, fc, ob in (("temp_c", "t2m_c", "obs_temp_c"), ("wind_ms", "wind10", "obs_wind_ms"), ("rh", "rh2m", "obs_rh")):
        ok = table[[fc, ob, "lead_bucket"]].dropna()
        for bucket, g in ok.groupby("lead_bucket", observed=True):
            err = g[fc] - g[ob]
            cont[f"{name}|{bucket}"] = {"n": len(g), "bias": round(float(err.mean()), 2),
                                        "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
                                        "corr": round(float(np.corrcoef(g[fc], g[ob])[0, 1]), 3)}
    report["continuous_vs_metar"] = cont

    # 2/3 event labels vs physically-motivated forecast scores
    ev = {}
    base = table.dropna(subset=["w_ts_any"])
    ev["ts"] = {"n": len(base), "base_rate": round(float(base["w_ts_any"].mean()), 4),
                "auc_cape": round(auc(base["w_ts_any"], base["cape"]), 3),
                "auc_cape_ml": round(auc(base["w_ts_any"], base["cape_ml"]), 3),
                "auc_neg_lftx": round(auc(base["w_ts_any"], -base["lftx"]), 3),
                "auc_refc": round(auc(base["w_ts_any"], base["refc"]), 3)}
    ev["rain"] = {"base_rate": round(float(base["w_rain_any"].mean()), 4),
                  "auc_apcp": round(auc(base["w_rain_any"], base["apcp_mm"]), 3),
                  "auc_apcp_cell": round(auc(base["w_rain_any"], base["apcp_cell_mean"]), 3),
                  "auc_pwat": round(auc(base["w_rain_any"], base["pwat"]), 3)}
    ev["fog"] = {"base_rate": round(float(base["w_fog"].mean()), 4),
                 "auc_neg_vis": round(auc(base["w_fog"], -base["vis_km"]), 3),
                 "auc_rh": round(auc(base["w_fog"], base["rh2m"]), 3)}
    ev["strong_wind"] = {"base_rate": round(float(base["w_strong_wind"].mean()), 4),
                         "auc_gust": round(auc(base["w_strong_wind"], base["gust"]), 3)}
    report["events_vs_forecast_scores"] = ev

    # 4 whole-India rain vs CHIRPS
    truth_rain = dataset.chirps_cell_truth(chirps_dir, points)
    rain = dataset.point_day_rain_table(p2, truth_rain)
    out = {"rows": len(rain), "points": int(rain["point_id"].nunique())}
    for k, g in rain.groupby("day_k"):
        g = g.dropna(subset=["rain_cell_mean_mm", "chirps_mean_mm"])
        if len(g) < 100:
            continue
        out[f"day{k}"] = {"n": len(g),
                          "corr_mean": round(float(np.corrcoef(g["rain_cell_mean_mm"], g["chirps_mean_mm"])[0, 1]), 3),
                          "gfs_mean_mm": round(float(g["rain_cell_mean_mm"].mean()), 2),
                          "chirps_mean_mm": round(float(g["chirps_mean_mm"].mean()), 2),
                          "auc_rain_ge2p5": round(auc(g["frac_ge_2p5"] >= 0.5, g["rain_cell_mean_mm"]), 3),
                          "chirps_wet_frac_2p5": round(float((g["frac_ge_2p5"]).mean()), 3)}
    report["rain_vs_chirps"] = out

    # 6 label noise: is CHIRPS (rain truth) consistent with independent METAR rain reports at the same stations?
    h = hourly.copy()
    h["day"] = pd.to_datetime(h["hour_utc"]).dt.floor("D")
    daily_metar = h.groupby(["point_id", "day"]).agg(metar_rain=("rain_any", "max"), hours=("n_obs", "size")).reset_index()
    daily_metar = daily_metar[daily_metar["hours"] >= 12]            # a day with <12 observed hours cannot say "dry"
    both = daily_metar.merge(truth_rain.rename(columns={"valid_date": "day"}), on=["point_id", "day"])
    both["metar_rain"] = both["metar_rain"].astype(bool)
    both["month"] = both["day"].dt.month
    agree = {"station_days": len(both), "metar_rain_day_rate": round(float(both["metar_rain"].mean()), 4),
             "auc_chirps_frac_ge1_vs_metar": round(auc(both["metar_rain"], both["frac_ge_1p0"]), 3),
             "p_chirps_mean_ge1_given_metar_rain": round(float((both.loc[both["metar_rain"], "chirps_mean_mm"] >= 1).mean()), 3),
             "p_chirps_mean_ge1_given_metar_dry": round(float((both.loc[~both["metar_rain"], "chirps_mean_mm"] >= 1).mean()), 3),
             "p_chirps_max_ge2p5_given_metar_dry": round(float((both.loc[~both["metar_rain"], "chirps_max_mm"] >= 2.5).mean()), 3)}
    for name, months in (("monsoon_JJAS", [6, 7, 8, 9]), ("other", [1, 2, 3, 4, 5, 10, 11, 12])):
        sub = both[both["month"].isin(months)]
        if len(sub) > 500:
            agree[name] = {"n": len(sub), "metar_rain_rate": round(float(sub["metar_rain"].mean()), 4),
                           "auc": round(auc(sub["metar_rain"], sub["frac_ge_1p0"]), 3)}
    report["chirps_vs_metar_label_noise"] = agree

    print("VALIDATION_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("VALIDATION_END")


if __name__ == "__main__":
    main()
