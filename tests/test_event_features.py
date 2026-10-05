"""Zones, splits and step-context features (synthetic inputs)."""
import numpy as np
import pandas as pd
import pytest

from event_models import features_events as fe


def test_region_of_known_cities():
    # name: (lat, lon, elevation_m, distance_to_coast_km, expected zone)
    cities = {
        "Delhi": (28.6, 77.2, 216, 1000, "indo_gangetic"), "Patna": (25.6, 85.1, 53, 700, "indo_gangetic"),
        "Jaisalmer": (26.9, 70.9, 230, 300, "northwest_arid"), "Mumbai": (19.1, 72.9, 8, 5, "west_coast"),
        "Chennai": (13.1, 80.3, 10, 5, "east_coast"), "Bengaluru": (13.0, 77.6, 900, 300, "south_interior"),
        "Nagpur": (21.1, 79.1, 310, 500, "central"), "Shillong": (25.6, 91.9, 1500, 400, "northeast"),
        "Guwahati": (26.1, 91.7, 55, 400, "northeast"), "Leh": (34.1, 77.6, 3500, 1500, "himalaya_north"),
        "Darjeeling": (27.0, 88.3, 2050, 500, "himalaya_north"), "PortBlair": (11.6, 92.7, 4, 3, "islands"),
        "Kochi": (10.0, 76.3, 3, 4, "west_coast"),
    }
    lat, lon, elev, dist, want = (np.array([c[i] for c in cities.values()]) for i in range(5))
    got = fe.region_of(lat, lon, elev, dist)
    for name, g, w in zip(cities, got, want):
        assert g == w, f"{name}: {g} != {w}"


def test_holdout_is_deterministic_and_about_the_requested_fraction():
    ids = [f"G:{i}" for i in range(5000)]
    a, b = fe.is_holdout(ids), fe.is_holdout(ids)
    assert (a == b).all() and 0.17 < a.mean() < 0.23
    lat, lon = np.repeat(np.arange(8, 36), 30), np.tile(np.arange(68, 98), 28)
    blocks = fe.holdout_blocks(lat, lon)
    # every cell in the same 4x4 block shares the flag
    keyed = pd.DataFrame({"b": [f"{int(a // 4)}_{int(o // 4)}" for a, o in zip(lat, lon)], "h": blocks})
    assert (keyed.groupby("b")["h"].nunique() == 1).all()


def test_split_labels_gap_and_holdout_semantics():
    dates = pd.to_datetime(["2023-06-01", "2024-03-31", "2024-04-02", "2024-06-01", "2025-01-02",
                            "2025-03-01"])
    held = np.array([False, False, False, False, False, False])
    labels = fe.split_labels(pd.Series(dates), held)
    assert labels.tolist() == ["train", "train", "drop", "val", "drop", "test_time"]
    held_all = np.ones(len(dates), bool)
    labels = fe.split_labels(pd.Series(dates), held_all)
    assert labels.tolist() == ["drop", "drop", "drop", "drop", "drop", "test_space"]   # held-out never trains or validates


def test_step_context_diffs_within_a_run_only():
    rows = []
    for run in ("2025-06-01", "2025-06-02"):
        for lead, cape in ((3, 100.0), (6, 300.0), (9, 250.0)):
            rows.append({"point_id": "S:A", "run_date": pd.Timestamp(run), "lead_h": lead, "cape": cape,
                         "t2m_c": 30.0, "mslp_hpa": 1000.0, "rh2m": 50.0, "refc": 0.0, "tcc": 10.0,
                         "pwat": 30.0, "apcp_cell_mean": 0.0})
    out = fe.step_context(pd.DataFrame(rows))
    first = out[out["run_date"] == "2025-06-01"].set_index("lead_h")
    assert np.isnan(first.loc[3, "d_cape_prev"]) and first.loc[6, "d_cape_prev"] == 200.0
    assert first.loc[3, "d_cape_next"] == 200.0 and np.isnan(first.loc[9, "d_cape_next"])
    # the second run's first step must NOT see the first run's last step
    second = out[out["run_date"] == "2025-06-02"].set_index("lead_h")
    assert np.isnan(second.loc[3, "d_cape_prev"])


# ---------------------------------------------------------------- climatology and local solar time
def _p1_for_clim():
    rows = []
    for run in pd.date_range("2022-01-01", "2025-12-31", freq="D"):
        for lead in (3, 6, 9):
            hot_year = 10.0 if run.year == 2025 else 0.0             # 2025 is in the TEST period and is much hotter
            rows.append({"point_id": "S:A", "run_date": run, "lead_h": lead, "t2m_c": 25.0 + hot_year, "pwat": 40.0})
    return pd.DataFrame(rows)


def test_forecast_climatology_ignores_dates_after_training_ends():
    p1 = _p1_for_clim()
    clim = fe.fit_climatology(p1, ["t2m_c", "pwat"], "step")
    assert clim["t2m_c"].round(6).eq(25.0).all()                     # the hot 2025 test year did not leak into the reference
    anom = fe.apply_climatology(p1, clim, "step")
    test_rows = anom[pd.to_datetime(anom["run_date"]) >= "2025-01-06"]
    assert test_rows["t2m_c_anom"].round(3).eq(10.0).all()           # a +10 C anomaly is visible, not absorbed
    train_rows = anom[pd.to_datetime(anom["run_date"]) <= "2024-03-31"]
    assert train_rows["t2m_c_anom"].abs().max() < 1e-4


def test_day_climatology_is_smoothed_circularly_across_new_year():
    rows = []
    for run in pd.date_range("2021-04-01", "2024-03-31", freq="D"):
        doy = run.dayofyear
        for k in (0, 1):
            rows.append({"point_id": "P", "run_date": run, "day_k": k, "t2m_max": 40.0 if (doy >= 360 or doy <= 5) else 20.0})
    clim = fe.fit_climatology(pd.DataFrame(rows), ["t2m_max"], "day")["t2m_max"]
    by_bin = clim.xs("P", level="point_id")
    # the cold-vs-hot step at the year boundary is smoothed on BOTH sides (bin 36 and bin 0 are neighbours)
    assert by_bin.loc[0] > 20.0 and by_bin.loc[36] > 20.0
    assert by_bin.loc[36] != by_bin.loc[20]


def test_missing_climatology_gives_nan_anomaly_not_zero():
    p1 = _p1_for_clim()
    clim = fe.fit_climatology(p1, ["t2m_c"], "step")
    other = p1.assign(point_id="S:UNSEEN")
    anom = fe.apply_climatology(other, clim, "step")
    assert anom["t2m_c_anom"].isna().all()                           # an unseen place has no reference: NaN, never a fake 0


def test_local_solar_time_shifts_with_longitude():
    f = pd.DataFrame({"valid_time": pd.Timestamp("2025-06-01 06:30"), "lon": [67.0, 97.0]})
    t = fe.time_features(f, "valid_time")
    # 06:30 UTC is 10:58 local solar at 67E and 12:58 at 97E -> the two longitudes see different parts of the day
    hour = lambda r: (np.degrees(np.arctan2(r["sin_lst"], r["cos_lst"])) % 360) / 15
    assert hour(t.iloc[0]) == pytest.approx(6.5 + 67 / 15, abs=0.01)
    assert hour(t.iloc[1]) == pytest.approx(6.5 + 97 / 15, abs=0.01)
    assert hour(t.iloc[1]) - hour(t.iloc[0]) == pytest.approx(2.0, abs=0.01)


def test_bundler_env_parsing_is_strict_and_accepts_both_spellings():
    from event_models import bundle
    assert bundle.parse_env(["--env", "A=1", "--env=B=x=y"]) == {"A": "1", "B": "x=y"}
    for bad in (["--env"], ["A=1"], ["--env", "novalue"], ["--env", "--oops=1"]):
        with pytest.raises(SystemExit):
            bundle.parse_env(bad)


def test_bundled_kernel_header_carries_exactly_the_requested_environment(tmp_path):
    from event_models import bundle
    text = bundle.bundle(__import__("pathlib").Path("event_models/run_training.py"), {"TARGETS": "fog,thunderstorm", "USE_EXT": "0"})
    head = "\n".join(text.splitlines()[:6])
    assert "_os.environ['TARGETS'] = 'fog,thunderstorm'" in head and "_os.environ['USE_EXT'] = '0'" in head
    assert "'--env'" not in head


def _points_for_ref():
    return pd.DataFrame({"point_id": ["G:a", "G:b", "G:c", "S:near_a", "S:near_c"], "kind": ["node", "node", "node", "station", "station"],
                         "lat": [20.0, 20.0, 25.0, 20.1, 24.8], "lon": [75.0, 76.0, 75.0, 75.1, 75.1]})


def test_nearest_node_reference_for_stations_and_queries():
    pts = _points_for_ref()
    ref = fe.nearest_node_ids(pts)
    assert ref["G:a"] == "G:a" and ref["G:c"] == "G:c"                    # nodes reference themselves
    assert ref["S:near_a"] == "G:a" and ref["S:near_c"] == "G:c"          # a station never references itself
    assert fe.nearest_node_id(pts, 20.05, 75.9) == "G:b"                  # an arbitrary query point -> nearest node


def test_station_rows_are_anomalised_against_their_nearest_node_not_themselves():
    pts = _points_for_ref()
    rows = []
    for run in pd.date_range("2022-01-01", "2023-12-31", freq="D"):
        for pid, base_t in (("G:a", 30.0), ("S:near_a", 33.0)):          # the station runs 3 C warmer than its reference node
            rows.append({"point_id": pid, "run_date": run, "day_k": 0, "t2m_max": base_t})
    frame = pd.DataFrame(rows)
    clim = fe.fit_climatology(frame, ["t2m_max"], "day")
    out = fe.apply_climatology(frame, clim, "day", fe.nearest_node_ids(pts))
    station = out[out["point_id"] == "S:near_a"]
    # self-reference would give anomaly 0; the honest serving-time reference gives the +3 C offset
    assert station["t2m_max_anom"].round(3).eq(3.0).all()
    assert out[out["point_id"] == "G:a"]["t2m_max_anom"].abs().max() < 1e-4


def test_saved_climatology_artifact_has_named_key_columns_for_serving():
    """Regression: smoothing dropped the 'bin' level name, so reset_index() wrote 'level_1' and serving could not load it."""
    rows = [{"point_id": "P", "run_date": run, "day_k": 0, "t2m_max": 30.0}
            for run in pd.date_range("2021-04-01", "2024-03-31", freq="D")]
    clim = fe.fit_climatology(pd.DataFrame(rows), ["t2m_max"], "day")
    assert list(clim.index.names) == ["point_id", "bin"]
    artifact = clim.reset_index()
    assert {"point_id", "bin", "t2m_max"} <= set(artifact.columns) and "level_1" not in artifact.columns


def test_stratified_holdout_gives_every_multi_block_zone_held_out_places():
    """Regression from the first real run: hash-only block hold-out left north-west arid, west coast and south interior with
    zero held-out points, so their skill could never be verified."""
    rng = np.random.default_rng(3)
    zones = np.array(fe.ZONES)
    lat = np.concatenate([rng.uniform(8, 36, 400) for _ in range(1)])
    lon = rng.uniform(68, 97, 400)
    zone = rng.choice(zones[:7], 400)                      # seven zones, scattered over many 4-degree blocks
    held = fe.holdout_blocks(lat, lon, zone=zone)
    blocks = pd.Series([f"{int(a // 4)}_{int(o // 4)}" for a, o in zip(lat, lon)])
    for z in zones[:7]:
        n_blocks = blocks[zone == z].nunique()
        if n_blocks >= 2:
            assert held[zone == z].any(), f"zone {z} has {n_blocks} blocks but nothing held out"
    again = fe.holdout_blocks(lat, lon, zone=zone)
    assert (held == again).all()                            # deterministic
    # blocks stay whole: every point in a block shares the flag
    assert (pd.DataFrame({"b": blocks, "h": held}).groupby("b")["h"].nunique() == 1).all()
    assert 0.1 < held.mean() < 0.6                          # a held-out share that is neither empty nor most of the data
