"""Serving-side static geography must equal training's definition (`collect_gfs.build_points`)."""
import numpy as np
import pandas as pd
import pytest

from event_models import collect_gfs as base
from weathergpt_events import static


def _fields(seed=0):
    rng = np.random.default_rng(seed)
    land = np.zeros((base.NR, base.NC), "float32")
    land[:, 30:] = 1.0                                           # a coastline: sea on the west, land to the east
    hgt = (200 + 60 * np.sin(np.arange(base.NC) / 9.0)[None, :] + 40 * rng.random((base.NR, base.NC))).astype("float32")
    return land, hgt


def test_static_sampling_matches_training_points_at_nodes():
    land, hgt = _fields()
    grids = static.build_static_grids(land, hgt)
    # reproduce what build_points does for nodes, with stations stubbed out (no network in tests)
    import event_models.collect_gfs as cg
    stub = pd.DataFrame(columns=["point_id", "kind", "lat", "lon", "name", "state", "station_elev_m"])
    original = cg.station_points
    cg.station_points = lambda: stub
    try:
        training = cg.build_points(land, hgt)
    finally:
        cg.station_points = original
    sample = training.sample(40, random_state=1)
    serving = static.point_static(grids, sample["lat"].to_numpy(), sample["lon"].to_numpy())
    for col in ("elevation_m", "land_frac", "dist_coast_km"):
        assert np.allclose(serving[col].to_numpy(), sample[col].to_numpy(), atol=1e-3), col


def test_domain_checks_refuse_ocean_and_out_of_box():
    land, hgt = _fields()
    grids = static.build_static_grids(land, hgt)
    lat = base.LAT_TOP - 60 * base.STEP
    assert static.in_domain(lat, base.LON_LEFT + 80 * base.STEP, grids)[0]
    ok, why = static.in_domain(lat, base.LON_LEFT + 5 * base.STEP, grids)       # in the box but over the "sea" columns
    assert not ok and "land-only" in why
    ok, why = static.in_domain(51.5, -0.1, grids)                                # London
    assert not ok and "outside" in why


def test_coast_distance_is_zero_ish_at_the_coast_and_grows_inland():
    land, hgt = _fields()
    grids = static.build_static_grids(land, hgt)
    lat = base.LAT_TOP - 60 * base.STEP
    near = static.point_static(grids, [lat], [base.LON_LEFT + 31 * base.STEP])["dist_coast_km"][0]
    far = static.point_static(grids, [lat], [base.LON_LEFT + 100 * base.STEP])["dist_coast_km"][0]
    assert near < 60 and far > 1500


def test_run_freshness_rule():
    from datetime import date, datetime, timezone
    from weathergpt_events import live
    now = datetime(2026, 10, 6, 6, 0, tzinfo=timezone.utc)
    assert live.run_is_fresh(date(2026, 10, 6), now)                       # today's 00Z run, 6 h old
    assert live.run_is_fresh(date(2026, 10, 5), now)                       # yesterday's, 30 h old: still the newest if NOAA is late
    assert live.run_is_fresh(date(2026, 10, 4), now)                       # 54 h old: the limit
    assert not live.run_is_fresh(date(2026, 10, 3), now)                   # 78 h old: stale, refuse
    assert not live.run_is_fresh(date(2026, 10, 7), now)                   # a run "from the future" is a clock/date bug, not fresh
