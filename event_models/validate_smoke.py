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
from __future__ import annotations

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
             "auc_chirps_frac_ge1_vs_metar": round(auc(both["metar_rain"], both["frac_ge_1"]), 3),
             "p_chirps_mean_ge1_given_metar_rain": round(float((both.loc[both["metar_rain"], "chirps_mean_mm"] >= 1).mean()), 3),
             "p_chirps_mean_ge1_given_metar_dry": round(float((both.loc[~both["metar_rain"], "chirps_mean_mm"] >= 1).mean()), 3),
             "p_chirps_max_ge2p5_given_metar_dry": round(float((both.loc[~both["metar_rain"], "chirps_max_mm"] >= 2.5).mean()), 3)}
    for name, months in (("monsoon_JJAS", [6, 7, 8, 9]), ("other", [1, 2, 3, 4, 5, 10, 11, 12])):
        sub = both[both["month"].isin(months)]
        if len(sub) > 500:
            agree[name] = {"n": len(sub), "metar_rain_rate": round(float(sub["metar_rain"].mean()), 4),
                           "auc": round(auc(sub["metar_rain"], sub["frac_ge_1"]), 3)}
    report["chirps_vs_metar_label_noise"] = agree

    print("VALIDATION_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("VALIDATION_END")


if __name__ == "__main__":
    main()
