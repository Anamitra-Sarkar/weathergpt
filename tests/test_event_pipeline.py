"""End-to-end plumbing: collector outputs + truth outputs -> build_tables -> every target's fit.

Synthetic but structurally real: forecast tables come from the actual collect_gfs / collect_gfs_ext functions
(10 forecast days, 52 steps), METAR labels from the actual labels module, CHIRPS from the actual npz layout.
The point is to catch stage-to-stage mismatches (a renamed or colliding column, a label string that does not
exist, an extension column that never reaches the feature list) locally instead of after a multi-hour Kaggle run.
"""
import json
import time

import numpy as np
import pandas as pd
import pytest

import test_event_ext as ext_fixtures
from event_models import collect_gfs as cg
from event_models import collect_gfs_ext as ext
from event_models import features_events as fe_module
from event_models import labels
from event_models import run_training as rt
from event_models import train as T

ALL_LEADS = list(range(3, 73, 3)) + list(range(78, 241, 6))
STATIONS = {"S:VAAA": (24.5, 73.2), "S:VBBB": (26.0, 75.4), "S:VCCC": (25.2, 74.1), "S:VDDD": (27.1, 76.0)}
NODES = [(23.5 + 0.5 * i, 72.5 + 0.5 * j) for i in range(2) for j in range(4)]


def _points():
    rows = []
    for pid, (lat, lon) in STATIONS.items():
        rows.append({"point_id": pid, "kind": "station", "lat": lat, "lon": lon, "elevation_m": 200.0,
                     "land_frac": 1.0, "dist_coast_km": 400.0, "station_elev_m": 210.0})
    for lat, lon in NODES:
        rows.append({"point_id": f"G:{lat:.2f}:{lon:.2f}", "kind": "node", "lat": lat, "lon": lon,
                     "elevation_m": 250.0, "land_frac": 1.0, "dist_coast_km": 420.0, "station_elev_m": np.nan})
    return pd.DataFrame(rows)


def _metar_raw(rng, start="2016-01-01", end="2026-09-05"):
    times = pd.date_range(start, end, freq="h")                        # hourly, like real airports: the 3 h label window needs >=2 observed hours
    n = len(times)
    doy, hour = times.dayofyear.to_numpy(), times.hour.to_numpy()
    temp_c = 24 + 18 * np.sin(2 * np.pi * (doy - 100) / 365) + 5 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 3, n)
    dew_c = temp_c - rng.uniform(1, 12, n)
    r1, r2, r3, r4 = (rng.random(n) for _ in range(4))
    wx = np.where(r1 < 0.04, "TSRA", np.where(r2 < 0.08, "-RA", np.where(r3 < 0.02, "FG", np.where(r4 < 0.005, "DU", ""))))
    vsby = np.where(wx == "FG", 0.3, np.where(rng.random(n) < 0.01, 0.4, 3.0))
    gust = np.where(rng.random(n) < 0.03, rng.uniform(26, 40, n), np.nan)
    return pd.DataFrame({"valid": times.strftime("%Y-%m-%d %H:%M"), "tmpf": temp_c * 9 / 5 + 32, "dwpf": dew_c * 9 / 5 + 32,
                         "sknt": rng.uniform(0, 15, n), "gust": gust, "vsby": vsby, "wxcodes": wx})


RUN_DATES = list(pd.date_range("2021-04-02", "2026-08-20", freq="28D"))


def _tilt(arrays: dict, scale: float) -> None:
    """Add a smooth spatial gradient to every field so interpolation at off-grid points is exercised."""
    grad = (np.add.outer(np.arange(cg.NR) * 0.004, np.arange(cg.NC) * 0.007)).astype("float32")
    for fields in arrays.values():
        for name, arr in fields.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and name not in ("apcp",):
                fields[name] = (arr * (1.0 + scale * grad / (1 + grad.mean()))).astype("float32")


def run_arrays(i: int):
    """Both array sets for run number i -- reproducible, so tests can rebuild exactly what the fixture wrote."""
    rng = np.random.default_rng(1000 + i)
    base_arrays = _fake_base_run(rng)
    ext_arrays = ext_fixtures._fake_arrays(rain_bucket=float(rng.exponential(2.0)))
    _tilt(base_arrays, 0.05)
    _tilt(ext_arrays, 0.05)
    return base_arrays, ext_arrays


def _fake_base_run(rng):
    """arrays[lead][field] for one run in the shapes and units collect_gfs produces."""
    base_vals = {"t2m": 300.0, "rh2m": 60.0, "dpt2m": 293.0, "u10": 3.0, "v10": 2.0, "gust": 7.0, "cape": 300.0,
                 "cape_ml": 250.0, "cin": -50.0, "lftx": -1.0, "vis": 20000.0, "tcc": 50.0, "pwat": 45.0, "cwat": 0.3,
                 "mslp": 100300.0, "hpbl": 600.0, "refc": 10.0, "apcp": 0.5}
    arrays = {}
    for lead in ALL_LEADS:
        f = {name: np.full((cg.NR, cg.NC), v * (1 + 0.2 * rng.standard_normal()), "float32") for name, v in base_vals.items()}
        f["apcp"] = np.full((cg.NR, cg.NC), rng.exponential(1.5), "float32")
        f["apcp_width"] = 6 if lead % 6 == 0 else 3
        arrays[lead] = f
    return arrays


FIXTURE_VERSION = "v4"      # bump whenever the synthetic generators above change: the inputs are cached on disk between runs


@pytest.fixture(scope="module", autouse=True)
def ten_day_leadset():
    """The collectors read their lead set at import time; the tests need the 10-day (52-step) set for EVERY test in
    this module -- building the fixture, the training tables, and the serving builder alike."""
    mp = pytest.MonkeyPatch()
    mp.setattr(cg, "LEADS", ALL_LEADS)
    mp.setattr(cg, "DAY_STEPS", {k: [h for h in ALL_LEADS if 24 * k < h <= 24 * k + 24] for k in range(10)})
    # The 12-point fixture cannot support zone-stratified hold-out (two 4-degree blocks): stratification would hold out most of
    # it and leave too few training rows.  Production has 2,709 points; stratification has its own unit test
    # (test_stratified_holdout_gives_every_multi_block_zone_held_out_places) and was checked on a realistic layout.
    original = fe_module.holdout_blocks
    mp.setattr(fe_module, "holdout_blocks", lambda lat, lon, fraction=0.2, block_deg=4.0, zone=None: original(lat, lon, fraction, block_deg))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def synthetic_inputs():
    from pathlib import Path
    root = Path(f"/tmp/weathergpt_pipeline_fixture_{FIXTURE_VERSION}")
    if (root / ".complete").exists():
        yield root
        return
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    rng = np.random.default_rng(11)
    truth, gfs = root / "truth", root / "gfs"
    for d in ("metar_hourly", "metar_daily", "chirps"):
        (truth / d).mkdir(parents=True)
    gfs.mkdir()
    points = _points()
    points.drop(columns=["station_elev_m"]).to_parquet(gfs / "points.parquet", index=False)
    pd.DataFrame({"sid": [p[2:] for p in STATIONS]}).to_parquet(truth / "stations.parquet", index=False)

    for pid in STATIONS:
        obs = labels.clean_observations(_metar_raw(rng))
        labels.hourly_labels(obs).to_parquet(truth / "metar_hourly" / f"{pid[2:]}.parquet", index=False)
        labels.daily_temperature_labels(obs, min_obs=4).to_parquet(truth / "metar_daily" / f"{pid[2:]}.parquet", index=False)

    (truth / "chirps" / "grid.json").write_text(json.dumps({"step": 0.05, "lat_north": 28.0, "lon_west": 72.0}))
    for year in range(2021, 2027):
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31" if year < 2026 else "2026-09-30")
        cell_rain = rng.gamma(0.4, 6.0, size=(len(days), 1, 1)) * (rng.random((len(days), 1, 1)) < 0.35)
        data = (cell_rain * rng.gamma(1.0, 1.0, size=(len(days), 100, 100))).astype("float16")
        np.savez_compressed(truth / "chirps" / f"chirps_india_{year}.npz", data=data,
                            dates=days.strftime("%Y-%m-%d").to_numpy().astype("U10"))   # same dtype collect_truth writes

    stn = points[points["kind"] == "station"]
    terrain = {"dzdx": np.zeros((cg.NR, cg.NC)), "dzdy": np.zeros((cg.NR, cg.NC))}
    stn_sampler = cg.Sampler(stn["lat"].to_numpy(), stn["lon"].to_numpy())
    all_sampler = cg.Sampler(points["lat"].to_numpy(), points["lon"].to_numpy())
    p1, p2, p1x, p2x = [], [], [], []
    for i, run in enumerate(RUN_DATES):
        base_arrays, ext_arrays = run_arrays(i)
        p1.append(cg.station_step_rows(run.date(), base_arrays, None, stn["point_id"].to_numpy(), stn_sampler))
        p2.append(cg.point_day_rows(run.date(), base_arrays, points["point_id"].to_numpy(), all_sampler))
        derived = ext.derive_all(ext_arrays, terrain)
        p1x.append(ext.station_step_ext(run.date(), ext_arrays, terrain, stn["point_id"].to_numpy(), stn_sampler, derived))
        p2x.append(ext.point_day_ext(run.date(), ext_arrays, terrain, points["point_id"].to_numpy(), all_sampler, derived))
    pd.concat(p1).to_parquet(gfs / "p1_station_steps_all.parquet", index=False)
    pd.concat(p2).to_parquet(gfs / "p2_point_days_all.parquet", index=False)
    pd.concat(p1x).to_parquet(gfs / "p1x_station_steps_all.parquet", index=False)
    pd.concat(p2x).to_parquet(gfs / "p2x_point_days_all.parquet", index=False)
    pd.DataFrame({"point_id": points["point_id"], "terrain_hgt": 200.0, "terrain_slope": 0.01,
                  "terrain_elev_std_5": 5.0, "terrain_elev_std_9": 8.0, "terrain_relief_9": 40.0,
                  "terrain_elev_minus_nb9": 0.0}).to_parquet(gfs / "points_ext.parquet", index=False)
    (root / ".complete").write_text("ok")
    yield root


def _check_tables(tables, features):
    for name, table in tables.items():
        suffixed = [c for c in table.columns if c.endswith("_x") or c.endswith("_y")]
        assert not suffixed, f"{name}: merge produced suffixed columns {suffixed}"
        missing = [c for c in features[name] if c not in table.columns]
        assert not missing, f"{name}: missing features {missing}"
        dupes = [c for c in set(features[name]) if features[name].count(c) > 1]
        assert not dupes, f"{name}: duplicated feature names {dupes}"
        counts = pd.Series(table["split"]).value_counts().to_dict()
        assert counts.get("train", 0) > 500 and counts.get("val", 0) > 100 and counts.get("test_time", 0) > 150, (name, counts)


def test_tables_with_extension_carry_every_declared_feature(synthetic_inputs, monkeypatch):
    monkeypatch.setattr(rt, "INPUT", str(synthetic_inputs))
    monkeypatch.setattr(rt, "ROW_STRIDE", 1)
    tables, features = rt.build_tables({"step", "day", "rain", "rain3", "rain7"})
    assert set(tables) == {"step", "day", "rain", "rain3", "rain7"}
    _check_tables(tables, features)
    # extension predictors, anomalies, terrain and local solar time all reached the feature lists
    assert {"wind850", "avg_crain", "spr_cape", "mfc850_e7", "day_t2m_mean_anom", "day_pwat_mean_anom", "sin_lst", "terrain_relief_9"} <= set(features["step"])
    assert {"gefs_rain_spread_mm", "rain_frac1_nb9", "avg_pwat_mean", "t2m_max_anom", "terrain_hgt"} <= set(features["rain"])
    assert {"gefs_tmax_mean_c", "gefs_tmax_spread_c", "t2m_max_anom", "k_index_max"} <= set(features["day"])
    # forecast visibility (feature) and observed visibility (truth) coexist under different names
    assert "vis_km" in tables["step"].columns and "obs_vis_km" in tables["step"].columns
    # forecast days reach 9 and the 7-day windows only start where a full week fits
    assert tables["rain"]["day_k"].max() == 9 and set(tables["rain7"]["day_k"]) == {0, 1, 2, 3}
    assert tables["day"]["day_k"].between(1, 9).all()
    # the IMD labels exist; rows with no normal are NaN rather than False
    day = tables["day"]
    assert {"heatwave", "coldwave", "tmax_norm"} <= set(day.columns) and day["heatwave"].notna().sum() > 0


def test_tables_without_extension_still_build(synthetic_inputs, monkeypatch):
    monkeypatch.setattr(rt, "INPUT", str(synthetic_inputs))
    monkeypatch.setenv("USE_EXT", "0")
    tables, features = rt.build_tables({"step", "rain"})
    _check_tables(tables, features)
    assert "wind850" not in features["step"] and "gefs_rain_spread_mm" not in features["rain"]
    assert "day_t2m_mean_anom" in features["step"]               # anomalies need only the base forecasts


def test_every_target_trains_end_to_end(synthetic_inputs, monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "INPUT", str(synthetic_inputs))
    monkeypatch.setattr(rt, "ROW_STRIDE", 1)
    tables, features = rt.build_tables({t.table for t in rt.TARGETS.values()})
    problems = {}
    for name, target in rt.TARGETS.items():
        frame, feats = tables[target.table], features[target.table]
        assert target.label in frame.columns, f"{name}: label {target.label} not in the {target.table} table"
        for attr in ("primary", "point_forecast", "score_low", "score_high"):
            col = getattr(target, attr)
            if col:
                assert col in frame.columns, f"{name}: {attr}={col}"
        if target.labels:
            assert all(c in frame.columns for c in target.labels), f"{name}: curve labels"
        if target.kind == "curve":
            rep = T.fit_curve(target, frame, feats, frame["split"].to_numpy(), tmp_path / name, rounds=10, time_limit_s=60,
                              max_stacked=200_000, eval_cap=3_000, extra_params={"min_data_in_leaf": 30}, min_rows=150)
        else:
            rep = rt.fit_target(target, frame, feats, frame["split"].to_numpy(), tmp_path / name, rounds=10, time_limit_s=60,
                                extra_params={"min_data_in_leaf": 20}, min_rows=150)
        if "skipped" in rep:
            problems[name] = rep["skipped"]
        else:
            assert (tmp_path / name / "metrics.json").exists(), name
            assert set(rep["metrics"]) == {"val", "test_time", "test_space"}, (name, list(rep["metrics"]))
    assert not problems, f"targets skipped for lack of rows: {problems}"


def test_short_horizon_data_skips_window_tables_with_a_clear_message(synthetic_inputs, monkeypatch, capsys):
    """Only 5 forecast days collected (the first collection): the 7-day table must be skipped, not crash."""
    monkeypatch.setattr(rt, "INPUT", str(synthetic_inputs))
    monkeypatch.setattr(rt, "ROW_STRIDE", 1)
    real = rt.dataset.read_all

    def short_only(root, pattern, dedupe_on=None, columns=None):
        frame = real(root, pattern, dedupe_on, columns)
        return frame[frame["day_k"] <= 4] if "day_k" in frame.columns and "point_days" in pattern else frame
    real_thinned = rt.dataset.read_thinned

    def short_thinned(root, pattern, dedupe_on, keep_dates=None, columns=None):
        frame = real_thinned(root, pattern, dedupe_on, keep_dates, columns)
        return frame[frame["day_k"] <= 4]
    monkeypatch.setattr(rt.dataset, "read_all", short_only)
    monkeypatch.setattr(rt.dataset, "read_thinned", short_thinned)      # the point-day tables are read through this one now
    tables, _ = rt.build_tables({"rain3", "rain7"})
    assert "rain3" in tables and "rain7" not in tables
    assert "rain7: skipped" in capsys.readouterr().out



# ------------------------------------------------------------------------------- train / serve parity
def _station_static(root, point_id):
    pts = pd.read_parquet(root / "gfs" / "points.parquet").merge(pd.read_parquet(root / "gfs" / "points_ext.parquet"), on="point_id")
    return pts[pts["point_id"] == point_id].reset_index(drop=True)


def _compare(train_rows: pd.DataFrame, serve_rows: pd.DataFrame, features: list, on: list, label: str):
    joined = train_rows.merge(serve_rows, on=on, suffixes=("_train", "_serve"))
    assert len(joined) >= 3, f"{label}: no overlapping rows to compare ({len(train_rows)} train, {len(serve_rows)} serve)"
    bad = {}
    for col in features:
        if col in on:
            continue
        a, b = joined[f"{col}_train"].to_numpy("float64"), joined[f"{col}_serve"].to_numpy("float64")
        same = np.isclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)
        if not same.all():
            bad[col] = (float(np.nanmax(np.abs(a - b))), int((~same).sum()))
    assert not bad, f"{label}: serving features differ from training features: {dict(list(bad.items())[:8])}"
    return len(joined)


@pytest.fixture(scope="module")
def parity_setup(synthetic_inputs):
    from weathergpt_events import static as static_mod
    from weathergpt_events.features import FeatureBuilder, RunArrays
    mp = pytest.MonkeyPatch()
    mp.setattr(rt, "INPUT", str(synthetic_inputs))
    mp.setattr(rt, "ROW_STRIDE", 1)
    tables, features = rt.build_tables({"step", "day", "rain", "rain3", "rain7"})
    i = 20
    base_arrays, ext_arrays = run_arrays(i)
    points = pd.read_parquet(synthetic_inputs / "gfs" / "points.parquet")
    land = np.ones((cg.NR, cg.NC), "float32")
    land[:, :6] = 0.0                                                      # a strip of "sea" on the western edge
    grids = static_mod.build_static_grids(land, np.full((cg.NR, cg.NC), 200.0, "float32"))   # flat: matches the fixture's zero terrain
    builder = FeatureBuilder(grids, rt.ARTIFACTS["climatology_day"], points)
    yield {"tables": tables, "features": features, "builder": builder, "run_date": RUN_DATES[i].date(), "root": synthetic_inputs,
           "run": RunArrays(RUN_DATES[i].date(), base_arrays, ext_arrays), "points": points}
    mp.undo()


def test_serving_step_features_equal_training_features(parity_setup):
    s = parity_setup
    pid = "S:VAAA"
    lat, lon = STATIONS[pid]
    serve = s["builder"].step_table(s["run"], lat, lon, point_id=pid, static=_station_static(s["root"], pid))
    train = s["tables"]["step"]
    train = train[(train["point_id"] == pid) & (pd.to_datetime(train["run_date"]).dt.date == s["run_date"])]
    n = _compare(train, serve, s["features"]["step"], ["lead_h"], "step")
    assert n >= 10


def test_serving_day_features_equal_training_features(parity_setup):
    s = parity_setup
    pid = "S:VBBB"
    lat, lon = STATIONS[pid]
    serve = s["builder"].station_day_table(s["run"], lat, lon, point_id=pid, static=_station_static(s["root"], pid))
    train = s["tables"]["day"]
    train = train[(train["point_id"] == pid) & (pd.to_datetime(train["run_date"]).dt.date == s["run_date"])]
    _compare(train, serve, s["features"]["day"], ["day_k"], "day")


def test_serving_rain_features_equal_training_features(parity_setup):
    s = parity_setup
    pid = f"G:{NODES[3][0]:.2f}:{NODES[3][1]:.2f}"
    lat, lon = NODES[3]
    static = _station_static(s["root"], pid)
    serve = s["builder"].rain_day_table(s["run"], lat, lon, point_id=pid, static=static)
    train = s["tables"]["rain"]
    train = train[(train["point_id"] == pid) & (pd.to_datetime(train["run_date"]).dt.date == s["run_date"])]
    _compare(train, serve, s["features"]["rain"], ["day_k"], "rain")
    for name, window in (("rain3", 3), ("rain7", 7)):
        serve_w = s["builder"].window_table(s["run"], lat, lon, window, point_id=pid, static=static)
        train_w = s["tables"][name]
        train_w = train_w[(train_w["point_id"] == pid) & (pd.to_datetime(train_w["run_date"]).dt.date == s["run_date"])]
        _compare(train_w, serve_w, s["features"][name], ["day_k"], name)


def test_serving_references_an_off_grid_station_to_its_nearest_node(parity_setup):
    """A query that is not a collected point gets the nearest node's climatology and its own interpolated forecast."""
    s = parity_setup
    lat, lon = 24.9, 73.4                                                    # between nodes, not a station
    serve = s["builder"].rain_day_table(s["run"], lat, lon, static=_station_static(s["root"], "G:23.50:72.50").assign(lat=lat, lon=lon))
    assert len(serve) == 10 and serve["point_id"].eq("Q:query").all()
    assert serve["rain_mm_anom"].notna().all()                              # a reference exists -> anomalies are real numbers



# ----------------------------------------------------------------------------------------- engine
def _engine(parity_setup, tmp_path, monkeypatch, skill=(True, "ok")):
    """Registry over three really-trained (synthetic-data) models; the gate is bypassed because the fixture's labels are
    random, and the skill lookup is stubbed so both branches of the safety default can be tested."""
    from weathergpt_events import registry as reg_mod
    from weathergpt_events.engine import EventEngine
    from weathergpt_events.models import load_model
    s = parity_setup
    reg = reg_mod.EventRegistry()
    for name in ("thunderstorm", "tmax_range", "rain_curve"):
        target = rt.TARGETS[name]
        frame, feats = s["tables"][target.table], s["features"][target.table]
        if target.kind == "curve":
            T.fit_curve(target, frame, feats, frame["split"].to_numpy(), tmp_path / name, rounds=8, time_limit_s=60,
                        max_stacked=100_000, eval_cap=2_000, extra_params={"min_data_in_leaf": 30}, min_rows=100)
        else:
            rt.fit_target(target, frame, feats, frame["split"].to_numpy(), tmp_path / name, rounds=8, time_limit_s=60,
                          extra_params={"min_data_in_leaf": 20}, min_rows=100)
        metrics = json.loads((tmp_path / name / "metrics.json").read_text())
        reg.models[name] = load_model(tmp_path / name)
        reg.gates[name] = reg_mod.GateResult(name, True, "ok")
        reg._metrics[name] = metrics
    monkeypatch.setattr(reg_mod.EventRegistry, "skill", lambda self, target, zone, lead=None: skill)
    return EventEngine(reg, s["builder"])


def test_engine_answers_are_structured_coherent_and_bounded(parity_setup, tmp_path, monkeypatch):
    s = parity_setup
    engine = _engine(parity_setup, tmp_path, monkeypatch)
    pid, (lat, lon) = "S:VAAA", STATIONS["S:VAAA"]
    out = engine.forecast(lat, lon, s["run"], point_id=pid, static=_station_static(s["root"], pid), horizon_days=10)
    assert out["available"] and set(out["targets"]) == {"thunderstorm", "tmax_range", "rain_curve"}
    storm = out["targets"]["thunderstorm"]["entries"]
    assert len(storm) == len(ALL_LEADS) and all(0.0 <= e["probability"] <= 1.0 for e in storm)
    assert storm[0]["lead_h"] == 3 and storm[-1]["lead_h"] == 240 and storm[0]["valid_time_utc"] < storm[-1]["valid_time_utc"]
    tmax = out["targets"]["tmax_range"]["entries"]
    assert [e["forecast_day"] for e in tmax] == list(range(1, 10))                      # IST days 1..9
    assert all(e["q10"] <= e["q50"] <= e["q90"] and e["lo"] <= e["q50"] <= e["hi"] and e["lo"] <= e["hi"] for e in tmax)
    rain = out["targets"]["rain_curve"]["entries"]
    assert len(rain) == 10
    for e in rain:
        ex = [e["exceedance"][k] for k in sorted(e["exceedance"], key=float)]
        assert all(a >= b - 1e-9 for a, b in zip(ex, ex[1:]))                           # monotone exceedance curve
        assert e["p_any_rain"] >= e["p_rainy_day"] >= e["p_heavy_rain"]
        assert abs(sum(e["imd_classes"].values()) - e["exceedance"]["0.1"]) < 1e-3       # classes sum to P(>=0.1 mm)
    # horizon is honoured
    short = engine.forecast(lat, lon, s["run"], targets=["rain_curve"], point_id=pid, static=_station_static(s["root"], pid), horizon_days=3)
    assert len(short["targets"]["rain_curve"]["entries"]) == 3


def test_engine_withholds_numbers_without_validated_skill(parity_setup, tmp_path, monkeypatch):
    s = parity_setup
    engine = _engine(parity_setup, tmp_path, monkeypatch, skill=(False, "no skill over local climatology in zone 'x'"))
    pid, (lat, lon) = "S:VAAA", STATIONS["S:VAAA"]
    static = _station_static(s["root"], pid)
    out = engine.forecast(lat, lon, s["run"], targets=["thunderstorm"], point_id=pid, static=static)
    entry = out["targets"]["thunderstorm"]["entries"][0]
    assert entry["validated"] is False and "probability" not in entry and "no skill" in entry["note"]
    debug = engine.forecast(lat, lon, s["run"], targets=["thunderstorm"], point_id=pid, static=static, include_unvalidated=True)
    assert "probability" in debug["targets"]["thunderstorm"]["entries"][0]               # numbers only on explicit request


def test_engine_refuses_the_sea_the_outside_and_unknown_or_refused_targets(parity_setup, tmp_path, monkeypatch):
    s = parity_setup
    engine = _engine(parity_setup, tmp_path, monkeypatch)
    sea = engine.forecast(24.5, 67.5, s["run"])                                          # inside the box, on the sea strip
    assert not sea["available"] and "land-only" in sea["reason"]
    assert not engine.forecast(51.5, -0.1, s["run"])["available"]                        # London
    pid, (lat, lon) = "S:VAAA", STATIONS["S:VAAA"]
    out = engine.forecast(lat, lon, s["run"], targets=["nonexistent", "fog"], point_id=pid, static=_station_static(s["root"], pid))
    assert out["targets"]["nonexistent"]["reason"] == "unknown target"
    assert "fog" not in engine.registry.gates or not out["targets"]["fog"]["available"]


def test_tool_catalogue_only_advertises_models_that_passed_the_gate(parity_setup, tmp_path, monkeypatch):
    from weathergpt_events import registry as reg_mod, tools
    engine = _engine(parity_setup, tmp_path, monkeypatch)
    engine.registry.gates["fog"] = reg_mod.GateResult("fog", False, "does not beat local climatology")
    engine.registry._metrics["fog"] = {"kind": "binary", "table": "step", "notes": ""}
    rows = {r["name"]: r for r in tools.catalogue(engine.registry)}
    assert rows["rain_curve"]["available"] and rows["rain_curve"]["horizon_days"] == 10
    assert not rows["fog"]["available"] and rows["fog"]["why_not"] and rows["fog"]["validated_zones"] == []
    assert "lat" in rows["rain_curve"]["parameters"]["properties"]
