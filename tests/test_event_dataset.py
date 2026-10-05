"""Join-logic tests on tiny synthetic frames: window labels, coverage drops, IST days, CHIRPS cells."""
import json

import numpy as np
import pandas as pd
import pytest

from event_models import dataset


def _hourly(point="S:TEST"):
    hours = pd.to_datetime(["2025-06-01 05:00", "2025-06-01 06:00", "2025-06-01 07:00",
                            "2025-06-01 12:00"])  # 08:00-11:00 unobserved
    return pd.DataFrame({"hour_utc": hours, "n_obs": [2, 2, 2, 2], "temp_c": [30.0, 31.0, 32.0, 36.0],
                         "rh": 50.0, "wind_ms": 3.0, "peak_wind_ms": 4.0, "vis_km": 8.0,
                         "ts_any": [False, False, True, False], "ts_at": False, "fog": False,
                         "dense_fog": False, "strong_wind": False, "gale": False, "rain_any": False,
                         "rain_heavy": False, "dust": False, "squall": False, "point_id": point})


def test_window_label_spreads_to_neighbours_and_tracks_coverage():
    labels = dataset.window_labels(_hourly()).set_index("valid_hour")
    # TS reported at 07:00 -> hours 06,07,08 windows see it
    assert labels.loc["2025-06-01 06:00", "w_ts_any"] == 1.0
    assert labels.loc["2025-06-01 07:00", "w_ts_any"] == 1.0
    assert labels.loc["2025-06-01 08:00", "w_ts_any"] == 1.0
    assert labels.loc["2025-06-01 05:00", "w_ts_any"] == 0.0
    # 09:00's window is 08,09,10: nothing observed -> cover 0 (so the row gets dropped, not labelled "no event")
    assert labels.loc["2025-06-01 09:00", "cover"] == 0
    assert labels.loc["2025-06-01 06:00", "cover"] == 3


def test_station_step_table_drops_uncovered_valid_times_and_uses_run_plus_lead():
    p1 = pd.DataFrame({"point_id": "S:TEST", "run_date": pd.Timestamp("2025-06-01"),
                       "lead_h": [6, 9, 12], "t2m_c": 30.0})
    table = dataset.station_step_table(p1, _hourly())
    # valid 06:00 (covered), 09:00 (cover 0 -> dropped), 12:00 (covered by the 12:00 report only -> cover 1 < 2 -> dropped)
    assert table["lead_h"].tolist() == [6]
    assert table.iloc[0]["w_ts_any"] == 1.0
    assert table.iloc[0]["obs_temp_c"] == 31.0   # exact-hour truth at 06:00


def test_ist_day_groups_by_india_calendar_day():
    # run 00Z Jun 1; lead 21h = 21:00 UTC Jun 1 = 02:30 IST Jun 2 ; lead 18h = 18:00 UTC = 23:30 IST Jun 1
    p1 = pd.DataFrame({"point_id": "S:T", "run_date": pd.Timestamp("2025-06-01"),
                       "lead_h": [18, 21, 24, 33], "t2m_c": [30.0, 20.0, 18.0, 41.0], "dpt2m_c": 10.0,
                       "rh2m": 40.0, "wind10": 2.0, "gust": 5.0, "cape": 0.0, "tcc": 10.0, "pwat": 20.0,
                       "hpbl": 500.0, "mslp_hpa": 1000.0})
    agg = dataset.ist_day_aggregates(p1).set_index("ist_day")
    assert agg.loc["2025-06-01", "n_steps"] == 1 and agg.loc["2025-06-01", "t2m_max"] == 30.0
    assert agg.loc["2025-06-02", "n_steps"] == 3          # 21h, 24h, 33h(=09:00 UTC Jun 2 -> 14:30 IST)
    assert agg.loc["2025-06-02", "t2m_max"] == 41.0 and agg.loc["2025-06-02", "t2m_min"] == 18.0
    assert agg.loc["2025-06-02", "day_k"] == 1


def test_chirps_cell_truth_fractions(tmp_path):
    rows, cols = 40, 40
    data = np.full((2, rows, cols), np.nan, "float16")
    data[0, 10:20, 10:20] = 0.0
    data[0, 10:15, 10:20] = 5.0       # half the 10x10 cell is wet (>=2.5, >=1) on day 0
    data[1, 10:20, 10:20] = 70.0      # whole cell heavy on day 1
    np.savez_compressed(tmp_path / "chirps_india_2025.npz", data=data, dates=np.array(["2025-06-01", "2025-06-02"]))
    (tmp_path / "grid.json").write_text(json.dumps({"step": 0.05, "lat_north": 38.0, "lon_west": 67.0}))
    # cell centre pixel (15,15) -> lat = 38 - 15*0.05, lon = 67 + 15*0.05
    points = pd.DataFrame({"point_id": ["G:x"], "lat": [38.0 - 15 * 0.05], "lon": [67.0 + 15 * 0.05]})
    truth = dataset.chirps_cell_truth(tmp_path, points).set_index("valid_date")
    d0, d1 = truth.loc["2025-06-01"], truth.loc["2025-06-02"]
    assert d0["frac_ge_2p5"] == pytest.approx(0.5) and d0["frac_ge_15p6"] == 0.0
    assert d0["chirps_mean_mm"] == pytest.approx(2.5, abs=1e-3)
    assert d1["frac_ge_64p5"] == 1.0 and d1["chirps_max_mm"] == pytest.approx(70.0, abs=0.1)


def test_forecast_and_observed_visibility_do_not_collide():
    """Regression: forecast vis_km and METAR vis_km used to be silently suffixed _x/_y by the merge."""
    p1 = pd.DataFrame({"point_id": "S:TEST", "run_date": pd.Timestamp("2025-06-01"), "lead_h": [6],
                       "vis_km": 24.0, "t2m_c": 30.0})
    table = dataset.station_step_table(p1, _hourly())
    assert table.iloc[0]["vis_km"] == 24.0                       # the forecast feature survives
    assert table.iloc[0]["obs_vis_km"] == pytest.approx(8.0)     # observed truth under its own name
    p1_bad = p1.assign(w_ts_any=1.0)
    with pytest.raises(ValueError, match="collision"):
        dataset.station_step_table(p1_bad, _hourly())


def test_rain_thresholds_cover_imd_boundaries_and_columns_are_stable():
    for imd in (2.5, 15.6, 64.5, 115.6, 204.5):
        assert imd in dataset.RAIN_THRESHOLDS
    assert list(dataset.RAIN_THRESHOLDS) == sorted(dataset.RAIN_THRESHOLDS)
    assert dataset.thr_col(1.0) == "frac_ge_1" and dataset.thr_col(2.5) == "frac_ge_2p5"
    assert dataset.thr_col(64.5) == "frac_ge_64p5" and dataset.thr_col(150.0) == "frac_ge_150"
    assert dataset.thr_col(2.5, "frac_any3_ge_") == "frac_any3_ge_2p5"


def test_three_day_union_is_not_the_product_of_daily_fractions(tmp_path):
    rows, cols, days = 40, 40, 5
    data = np.full((days, rows, cols), np.nan, "float16")
    data[:, 10:20, 10:20] = 0.0
    data[0, 10:20, 10:15] = 5.0        # day 0: left half wet
    data[1, 10:20, 15:20] = 5.0        # day 1: right half wet (disjoint pixels)
    data[2, 10:20, 10:15] = 5.0        # day 2: left half again (overlaps day 0)
    np.savez_compressed(tmp_path / "chirps_india_2025.npz", data=data,
                        dates=np.array([f"2025-06-0{d + 1}" for d in range(days)]))
    (tmp_path / "grid.json").write_text(json.dumps({"step": 0.05, "lat_north": 38.0, "lon_west": 67.0}))
    points = pd.DataFrame({"point_id": ["G:x"], "lat": [38.0 - 15 * 0.05], "lon": [67.0 + 15 * 0.05]})
    truth = dataset.chirps_cell_truth(tmp_path, points).set_index("valid_date")
    first = truth.loc["2025-06-01"]
    assert first["frac_ge_2p5"] == pytest.approx(0.5)               # one day alone: half the cell
    assert first["frac_any3_ge_2p5"] == pytest.approx(1.0)          # left | right | left = whole cell
    assert first["frac_any3_ge_15p6"] == 0.0
    # the last two days of the array have no complete window -> NaN, not a guess
    assert np.isnan(truth.loc["2025-06-04", "frac_any3_ge_2p5"]) and np.isnan(truth.loc["2025-06-05", "frac_any3_ge_2p5"])


def test_rain_window_table_pivots_days_and_drops_incomplete_ones():
    def day(k, mean, buckets=4):
        return {"point_id": "G:x", "run_date": pd.Timestamp("2025-06-01"), "day_k": k, "rain_buckets": buckets,
                "rain_cell_mean_mm": mean, "rain_cell_max_mm": mean * 2, "rain_mm": mean, "pwat_mean": 50.0,
                "cape_max": 100.0, "refc_max": 20.0, "tcc_mean": 60.0, "wind_mean": 3.0, "t2m_max": 33.0}
    p2 = pd.DataFrame([day(0, 1.0), day(1, 2.0), day(2, 4.0), day(3, 8.0, buckets=2), day(4, 16.0)])
    truth = pd.DataFrame({"point_id": "G:x", "valid_date": pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-03"]),
                          "cell_valid_px": 100, "frac_any3_ge_2p5": [0.9, 0.5, 0.2], "frac_any3_ge_15p6": 0.0})
    table = dataset.rain_window_table(p2, truth)
    # k0=0 uses days 0,1,2 (complete); k0=1 uses 1,2,3 (day 3 incomplete -> dropped); k0=2 uses 2,3,4 -> dropped
    assert table["day_k"].tolist() == [0]
    row = table.iloc[0]
    assert row["d0_rain_cell_mean_mm"] == 1.0 and row["d2_rain_cell_mean_mm"] == 4.0
    assert row["sum3_rain_cell_mean_mm"] == 7.0 and row["max3_rain_cell_max_mm"] == 8.0
    assert row["frac_any3_ge_2p5"] == 0.9


# ------------------------------------------------------------ IMD heat / cold wave labels
def _daily_history(point="S:T", years=range(2016, 2024), tmax_by_doy=lambda d: 30.0, tmin_by_doy=lambda d: 15.0):
    days = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    return pd.DataFrame({"point_id": point, "ist_day": days,
                         "tmax_c": [tmax_by_doy(min(d.dayofyear, 365)) for d in days],
                         "tmin_c": [tmin_by_doy(min(d.dayofyear, 365)) for d in days], "n_temp": 24})


def test_station_normals_recover_the_seasonal_cycle_and_need_enough_years():
    hist = _daily_history(tmax_by_doy=lambda d: 30 + 10 * np.sin(2 * np.pi * (d - 100) / 365))
    normals = dataset.station_normals(hist).set_index("doy")
    assert normals.loc[190, "tmax_norm"] == pytest.approx(30 + 10 * np.sin(2 * np.pi * 90 / 365), abs=0.6)
    thin = dataset.station_normals(_daily_history(years=range(2022, 2024)))      # 2 years: not enough behind each day
    assert thin["tmax_norm"].isna().all()


def test_imd_heatwave_rules_plains_hills_and_missing_normal():
    normals = dataset.station_normals(_daily_history("S:P", tmax_by_doy=lambda d: 36.0, tmin_by_doy=lambda d: 14.0))
    hills = dataset.station_normals(_daily_history("S:H", tmax_by_doy=lambda d: 24.0, tmin_by_doy=lambda d: 5.0))
    normals = pd.concat([normals, hills])
    elev = pd.Series({"S:P": 200.0, "S:H": 2200.0})
    day = pd.Timestamp("2025-05-20")
    rows = pd.DataFrame([
        {"point_id": "S:P", "ist_day": day, "tmax_c": 41.0, "tmin_c": 28.0},      # >=40 and +5 over normal  -> heat wave
        {"point_id": "S:P", "ist_day": day, "tmax_c": 40.5, "tmin_c": 28.0},      # >=40 but only +4.5 -> boundary: true
        {"point_id": "S:P", "ist_day": day, "tmax_c": 39.9, "tmin_c": 28.0},      # <40: no, however large the departure
        {"point_id": "S:P", "ist_day": day, "tmax_c": 45.2, "tmin_c": 30.0},      # absolute >=45 -> heat wave regardless
        {"point_id": "S:H", "ist_day": day, "tmax_c": 30.0, "tmin_c": 12.0},      # hills: >=30 and +6 -> heat wave
        {"point_id": "S:H", "ist_day": day, "tmax_c": 28.0, "tmin_c": 12.0},      # hills: +4 only -> no
    ])
    out = dataset.imd_wave_labels(rows, normals, elev)
    assert out["heatwave"].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert out.loc[0, "severe_heatwave"] == 0.0 and dataset.imd_wave_labels(
        rows.assign(tmax_c=[47.5, 41, 41, 41, 31, 31]), normals, elev).loc[0, "severe_heatwave"] == 1.0


def test_cold_wave_rules_and_no_label_without_a_normal():
    normals = dataset.station_normals(_daily_history("S:P", tmax_by_doy=lambda d: 25.0, tmin_by_doy=lambda d: 14.0))
    elev = pd.Series({"S:P": 200.0, "S:NONORM": 200.0})
    day = pd.Timestamp("2025-01-10")
    rows = pd.DataFrame([
        {"point_id": "S:P", "ist_day": day, "tmax_c": 20.0, "tmin_c": 9.0},        # <=10 and -5 -> cold wave
        {"point_id": "S:P", "ist_day": day, "tmax_c": 20.0, "tmin_c": 11.0},       # not <=10 -> no
        {"point_id": "S:P", "ist_day": day, "tmax_c": 20.0, "tmin_c": 3.5},        # <=4 absolute -> yes
        {"point_id": "S:NONORM", "ist_day": day, "tmax_c": 38.0, "tmin_c": 8.0},   # no normal, not absolute -> NaN, NOT False
        {"point_id": "S:NONORM", "ist_day": day, "tmax_c": 46.0, "tmin_c": 3.0},   # absolute thresholds decide without a normal
    ])
    out = dataset.imd_wave_labels(rows, normals, elev)
    assert out["coldwave"].tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(out.loc[3, "coldwave"]) and np.isnan(out.loc[3, "heatwave"])
    assert out.loc[4, "coldwave"] == 1.0 and out.loc[4, "heatwave"] == 1.0


def test_merge_ext_refuses_silent_column_collisions():
    base_t = pd.DataFrame({"point_id": ["a"], "run_date": [1], "lead_h": [3], "cape": [1.0]})
    good = pd.DataFrame({"point_id": ["a"], "run_date": [1], "lead_h": [3], "wind850": [5.0]})
    assert "wind850" in dataset.merge_ext(base_t, good, ["point_id", "run_date", "lead_h"]).columns
    bad = good.assign(cape=2.0)
    with pytest.raises(ValueError, match="collision"):
        dataset.merge_ext(base_t, bad, ["point_id", "run_date", "lead_h"])
