"""Unit tests for the METAR-derived event labels (synthetic rows, no network)."""
import numpy as np
import pandas as pd
import pytest

from event_models import labels


def _frame(rows):
    cols = ["valid", "tmpf", "dwpf", "sknt", "gust", "vsby", "wxcodes"]
    return pd.DataFrame(rows, columns=cols)


def test_wxcodes_distinguish_at_station_from_vicinity():
    wx = pd.Series(["TSRA", "VCTS", "+TSRA BR", "-RA", "FG", "VCSH", None, "BLDU", "+SHRA"])
    out = labels.parse_wxcodes(wx)
    assert out["ts_at"].tolist() == [True, False, True, False, False, False, False, False, False]
    assert out["ts_vc"].tolist() == [False, True, False, False, False, False, False, False, False]
    assert out["ts_any"].tolist() == [True, True, True, False, False, False, False, False, False]
    # VCSH is a shower near the airport, not rain at it
    assert out["rain_any"].tolist() == [True, False, True, True, False, False, False, False, True]
    assert out["rain_heavy"].tolist() == [False, False, True, False, False, False, False, False, True]
    assert out["dust"].tolist()[7] is True or out["dust"].iloc[7]
    assert out["fog_code"].tolist()[4]


def test_clean_applies_physical_bounds_without_dropping_the_row():
    frame = _frame([["2025-06-01 06:00", 300.0, 70.0, 5, None, 3.0, "TSRA"],   # absurd 300F
                    ["2025-06-01 06:30", 95.0, 77.0, 8, 30, 0.1, ""]])
    obs = labels.clean_observations(frame)
    assert len(obs) == 2
    assert np.isnan(obs.loc[0, "temp_c"])           # bad sensor -> NaN
    assert bool(obs.loc[0, "ts_at"])                # but the thunderstorm report survives
    assert obs.loc[1, "temp_c"] == pytest.approx(35.0, abs=0.01)
    assert obs.loc[1, "vis_km"] == pytest.approx(0.1609, abs=1e-3)
    assert obs.loc[1, "peak_kt"] == 30


def test_rh_clipped_when_dewpoint_exceeds_temperature():
    rh = labels.rh_from_dewpoint(pd.Series([20.0]), pd.Series([20.2]))
    assert rh.iloc[0] == 100.0


def test_hourly_labels_or_over_the_hour_and_thresholds():
    frame = _frame([
        ["2025-06-01 06:05", 86.0, 77.0, 5, None, 5.0, ""],
        ["2025-06-01 06:35", 86.0, 77.0, 8, 27, 0.3, "TSRA"],   # TS + fog-vis + gust 27kt
        ["2025-06-01 07:05", 86.0, 77.0, 4, None, 4.0, "HZ"],
    ])
    hourly = labels.hourly_labels(labels.clean_observations(frame))
    assert len(hourly) == 2
    first = hourly.iloc[0]
    assert first["n_obs"] == 2 and first["ts_at"] and first["rain_any"]
    assert first["fog"] and not first["dense_fog"]
    assert first["strong_wind"] and not first["gale"]
    assert first["vis_km"] == pytest.approx(0.3 * labels.MILE_KM, abs=1e-6)
    second = hourly.iloc[1]
    assert not second["ts_any"] and not second["fog"] and not second["strong_wind"]
    assert hourly["ts_at"].dtype == bool


def test_daily_extremes_use_ist_day_and_min_obs():
    # 00:30 UTC on 2 Jun is 06:00 IST on 2 Jun: the cold morning belongs to the 2nd, not the 1st.
    times = pd.date_range("2025-06-01 18:30", periods=24, freq="h")  # = 2 Jun 00:00..23:00 IST
    temps_f = [60.0] * 12 + [113.0] * 12                              # 15.6C morning, 45C afternoon
    frame = _frame([[t.strftime("%Y-%m-%d %H:%M"), f, 50.0, 3, None, 5.0, ""]
                    for t, f in zip(times, temps_f)])
    daily = labels.daily_temperature_labels(labels.clean_observations(frame))
    assert len(daily) == 1 and str(daily.loc[0, "ist_day"].date()) == "2025-06-02"
    assert daily.loc[0, "tmax_c"] == pytest.approx(45.0, abs=0.01)
    assert daily.loc[0, "hot_day"] and daily.loc[0, "severe_heat"] and not daily.loc[0, "cold_night"]
    sparse = labels.daily_temperature_labels(labels.clean_observations(frame.head(3)))
    assert sparse.empty
