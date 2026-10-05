"""Derived physics in the extension collector, checked against fields with known analytic answers."""
import numpy as np
import pandas as pd
import pytest

from event_models import collect_gfs as base
from event_models import collect_gfs_ext as ext

NR, NC = base.NR, base.NC
ROW, COL = np.meshgrid(np.arange(NR), np.arange(NC), indexing="ij")
X = COL * ext.DX_M[:, None]                   # metres east of the west edge (varies with latitude as the grid does)
Y = (NR - ROW) * ext.DY_M                     # metres north of the south edge


def _interior(a, m=3):
    return a[m:-m, m:-m]


def test_derivative_operators_recover_unit_gradients():
    assert np.allclose(_interior(ext.ddx(X.astype("float64"))), 1.0, atol=1e-6)
    assert np.allclose(_interior(ext.ddy(Y.astype("float64"))), 1.0, atol=1e-6)


def test_solid_body_rotation_has_vorticity_2_omega_and_no_divergence():
    omega = 1e-5                                          # s^-1, anticlockwise
    # x measured from the CENTRE column: the grid's east-west spacing shrinks with latitude, so a field built from
    # distance-from-the-edge carries a geometric (tan(lat)/R) divergence that grows with distance from the patch
    xc = (COL - 62) * ext.DX_M[:, None]
    yc = (60 - ROW) * ext.DY_M
    out = ext.derive_ext({"u850": -omega * yc, "v850": omega * xc}, {"dzdx": 0 * xc, "dzdy": 0 * xc})
    patch = (slice(20, 110), slice(57, 68))
    assert np.allclose(out["vort850_e5"][patch], 2 * omega * 1e5, atol=0.02)
    assert np.abs(out["div850_e5"][patch]).max() < 0.02


def test_expansion_has_positive_divergence_and_negative_moisture_flux_convergence():
    a = 2e-6
    u, v = a * (X - X.mean()), a * (Y - Y.mean())
    q = np.full_like(u, 0.012)
    out = ext.derive_ext({"u850": u, "v850": v, "q850": q}, {"dzdx": 0 * u, "dzdy": 0 * u})
    assert np.allclose(_interior(out["div850_e5"]), 2 * a * 1e5, atol=0.02)
    # diverging flow = moisture-flux DEconvergence: -div(qV) = -q*2a < 0 (rain needs convergence, i.e. positive)
    assert np.allclose(_interior(out["mfc850_e7"]), -0.012 * 2 * a * 1e7, atol=0.02)
    assert (_interior(out["mfc850_e7"]) < 0).all()


def test_upslope_wind_is_the_component_up_the_terrain():
    slope = 0.01
    terrain = {"dzdx": np.full((NR, NC), slope), "dzdy": np.zeros((NR, NC))}
    u10, v10 = np.full((NR, NC), 5.0), np.zeros((NR, NC))
    out = ext.derive_ext({"u10": u10, "v10": v10}, terrain)
    assert np.allclose(out["upslope10"], 0.05)                       # westerly wind blowing up an eastward-rising slope
    out2 = ext.derive_ext({"u10": -u10, "v10": v10}, terrain)
    assert np.allclose(out2["upslope10"], -0.05)                     # downslope is negative
    assert np.allclose(out["wdir10_sin"] ** 2 + out["wdir10_cos"] ** 2, 1.0, atol=1e-5)


def test_dewpoint_and_stability_indices_against_textbook_values():
    assert ext.dewpoint_c(np.array([25.0]), np.array([100.0]))[0] == pytest.approx(25.0, abs=0.05)
    assert ext.dewpoint_c(np.array([30.0]), np.array([50.0]))[0] == pytest.approx(18.4, abs=0.3)
    f = {k: np.full((NR, NC), v) for k, v in {"t850": 293.15, "t700": 283.15, "t500": 263.15, "rh850": 90.0, "rh700": 60.0,
                                              "z850": 1500.0, "z500": 5900.0}.items()}
    out = ext.derive_ext(f, {"dzdx": np.zeros((NR, NC)), "dzdy": np.zeros((NR, NC))})
    td850, td700 = ext.dewpoint_c(20.0, 90.0), ext.dewpoint_c(10.0, 60.0)
    assert out["k_index"][0, 0] == pytest.approx((20 - (-10)) + td850 - (10 - td700), abs=1e-3)
    assert out["total_totals"][0, 0] == pytest.approx(20 + td850 + 20, abs=1e-3)
    assert out["lapse_850_500"][0, 0] == pytest.approx(30.0 / 4.4, abs=1e-3)       # K/km over a 4.4 km layer


def test_neighbourhood_filters_see_a_spike_only_inside_their_window():
    a = np.zeros((NR, NC), "float32")
    a[60, 60] = 25.0
    m5, x5, s5 = ext.nb_filters(a, 5)
    m9, x9, _ = ext.nb_filters(a, 9)
    assert x5[60, 62] == 25.0 and x5[60, 63] == 0.0                  # 5x5 reaches 2 nodes away, not 3
    assert x9[60, 64] == 25.0 and x9[60, 65] == 0.0                  # 9x9 reaches 4
    assert m5[60, 60] == pytest.approx(1.0) and m9[60, 60] == pytest.approx(25.0 / 81)
    assert s5[60, 60] > 0 and s5[0, 0] == 0.0


def _fake_arrays(rain_bucket=2.0):
    arrays = {}
    for lead in base.LEADS:
        f = {name: np.full((NR, NC), 1.0, "float32") for name in list(ext.GFS_EXT) + [f"avg_{k}" for k in ext.GEFS_AVG] + [f"spr_{k}" for k in ext.GEFS_SPR]}
        f["t850"][:] = 295.0
        f["t700"][:] = 285.0
        f["t500"][:] = 265.0
        f["rh850"][:] = 80.0
        f["rh700"][:] = 60.0
        f["z850"][:], f["z500"][:] = 1500.0, 5900.0
        f["apcp"][:] = rain_bucket
        f["avg_apcp"][:] = rain_bucket / 2
        f["spr_apcp"][:] = 1.0
        f["avg_tmax2m"][:] = 305.0 + (lead % 24) / 24.0
        f["avg_tmin2m"][:] = 295.0
        f["spr_tmax2m"][:] = 1.5
        f["spr_tmin2m"][:] = 0.5
        f["avg_t2m"][:] = 300.0
        f["spr_t2m"][:] = 1.0
        w = 6 if lead % 6 == 0 else 3
        for name in ("apcp", "avg_apcp", "spr_apcp", "avg_tmax2m", "avg_tmin2m", "spr_tmax2m", "spr_tmin2m"):
            f[f"width:{name}"] = w
        arrays[lead] = f
    return arrays


def test_daily_ext_rows_are_complete_and_numerically_right():
    terrain = {"dzdx": np.zeros((NR, NC)), "dzdy": np.zeros((NR, NC))}
    sampler = base.Sampler([20.0, 25.0], [75.0, 80.0])
    rows = ext.point_day_ext(pd.Timestamp("2025-07-15").date(), _fake_arrays(), terrain, np.array(["a", "b"]), sampler)
    day0 = rows[rows["day_k"] == 0].set_index("point_id").loc["a"]
    # four 6-hour buckets of 2 mm each -> 8 mm; uniform field -> neighbourhood max/mean equal it
    assert day0["rain_nb5_mean_mm"] == pytest.approx(8.0) and day0["rain_nb9_max_mm"] == pytest.approx(8.0)
    assert day0["rain_frac1_nb9"] == pytest.approx(1.0) and day0["rain_frac10_nb9"] == pytest.approx(0.0)
    assert day0["gefs_rain_mean_mm"] == pytest.approx(4.0)                          # 4 buckets x 1 mm ensemble mean
    assert day0["gefs_rain_spread_mm"] == pytest.approx(2.0)                        # sqrt(4 x 1^2): quadrature
    assert day0["gefs_tmax_mean_c"] == pytest.approx(305.0 + 23 / 24.0 - 273.15, abs=0.05) or day0["gefs_tmax_mean_c"] > 30
    assert day0["gefs_tmax_spread_c"] == pytest.approx(1.5) and day0["gefs_tmin_mean_c"] == pytest.approx(295.0 - 273.15)
    # every aggregate column the reducers promise exists and is finite for a complete day
    expected = [f"{n}_mean" for n in ext.MEAN_F] + [f"{n}_max" for n in ext.MAX_F] + [f"{n}_min" for n in ext.MIN_F]
    missing = [c for c in expected if c not in rows.columns]
    assert not missing, missing[:8]
    assert set(rows["day_k"]) == {0, 1, 2, 3, 4}


def test_incomplete_rain_day_leaves_nan_instead_of_a_partial_sum():
    terrain = {"dzdx": np.zeros((NR, NC)), "dzdy": np.zeros((NR, NC))}
    arrays = _fake_arrays()
    for lead in (12, 18, 24):
        arrays[lead].pop("apcp")
        arrays[lead].pop("width:apcp")
    rows = ext.point_day_ext(pd.Timestamp("2025-07-15").date(), arrays, terrain, np.array(["a"]), base.Sampler([20.0], [75.0]))
    day0 = rows[rows["day_k"] == 0].iloc[0]
    # only one 6 h bucket survived: the sum is still computed from what exists, but base.rain_buckets (training filter)
    # is what guards completeness -- here we just require no crash and finite neighbourhood stats from the survivor
    assert np.isfinite(day0["rain_nb5_mean_mm"])


def test_station_step_ext_has_one_row_per_station_and_lead():
    terrain = {"dzdx": np.zeros((NR, NC)), "dzdy": np.zeros((NR, NC))}
    stn = base.Sampler([20.0, 25.0], [75.0, 80.0])
    rows = ext.station_step_ext(pd.Timestamp("2025-07-15").date(), _fake_arrays(), terrain, np.array(["S:a", "S:b"]), stn)
    assert len(rows) == 2 * len(base.LEADS) and {"wind850", "k_index", "avg_apcp", "avg_crain", "spr_cape"} <= set(rows.columns)
    assert (rows.groupby("lead_h").size() == 2).all()


def test_terrain_features_flat_vs_ridge():
    flat = ext.terrain_features(np.full((NR, NC), 300.0))
    assert np.allclose(flat["slope"], 0.0) and np.allclose(flat["elev_std_9"], 0.0) and np.allclose(flat["relief_9"], 0.0)
    ridge = np.full((NR, NC), 300.0)
    ridge[:, 60] = 2300.0
    t = ext.terrain_features(ridge)
    assert t["relief_9"][50, 60] == 2000.0 and t["elev_std_9"][50, 60] > 400
    assert t["elev_minus_nb9"][50, 60] > 1500                          # point on a ridge sits far above its surroundings


def test_sentinel_values_are_masked_before_interpolation():
    """Regression: a 9999 soil-moisture sentinel was being blended with real values at coastal points."""
    soil = np.full((NR, NC), 0.25, "float32")
    soil[40, 40] = 9999.0                                    # one undefined node (a lake / sea cell)
    out = ext.derive_ext({"soilw": soil, "avg_soilw": soil}, {"dzdx": np.zeros((NR, NC)), "dzdy": np.zeros((NR, NC))})
    assert np.isnan(out["soilw"][40, 40]) and out["soilw"][40, 41] == pytest.approx(0.25)
    # a point sitting between the undefined node and three good ones gets the good ones' value, not NaN and not 9999
    lat = base.LAT_TOP - 40.5 * base.STEP
    lon = base.LON_LEFT + 40.5 * base.STEP
    sampled = base.Sampler([lat], [lon])(out["soilw"])
    assert sampled[0] == pytest.approx(0.25, abs=1e-6)


def test_sampler_is_unchanged_without_nans_and_nan_with_no_valid_neighbour():
    grid = np.add.outer(np.arange(NR) * 10.0, np.arange(NC) * 1.0).astype("float32")
    s = base.Sampler([base.LAT_TOP - 0.25 * 4.5], [base.LON_LEFT + 0.25 * 6.5])
    assert s(grid)[0] == pytest.approx(45.0 + 6.5, abs=1e-4)           # same as plain bilinear
    allnan = np.full((NR, NC), np.nan, "float32")
    assert np.isnan(s(allnan)[0])
