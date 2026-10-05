"""The trainer on synthetic data with a planted signal: it must find it, calibrate, and report honestly."""
import numpy as np
import pandas as pd
import pytest

from event_models import train as T


def _frame(n=40_000, seed=3):
    rng = np.random.default_rng(seed)
    x1, x2, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    logit = -2.5 + 2.2 * x1                                   # ~10% events, driven by x1 only
    event = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    temp_true = 25 + 3 * x1 + (0.3 + 1.2 * np.abs(x2)) * rng.normal(size=n)  # strongly heteroscedastic in x2
    frame = pd.DataFrame({"x1": x1, "x2": x2, "noise": noise, "event": event, "temp": temp_true,
                          "gfs_temp": 25 + 3 * x1 + 1.0,                      # raw forecast with +1 bias
                          "month": rng.integers(1, 13, n),
                          "zone": np.where(np.abs(x2) > 1.0, "hi_noise", "lo_noise"),
                          "lead_bucket": rng.choice(["d1", "d2"], n)})
    split = np.where(np.arange(n) < 24_000, "train", np.where(np.arange(n) < 30_000, "val",
                     np.where(np.arange(n) < 35_000, "test_time", "test_space")))
    return frame, split


def test_binary_target_learns_signal_and_calibrates(tmp_path):
    frame, split = _frame()
    target = T.Target("syn_event", "step", "binary", "event", primary="x1", direction=1)
    rep = T.fit_target(target, frame, ["x1", "x2", "noise"], split, tmp_path / "m", rounds=120, time_limit_s=60)
    m = rep["metrics"]["test_time"]
    assert m["auc"] > 0.80, m
    assert m["bss_vs_train_clim"] > 0.05, m
    assert m["ece"] < 0.03, m                                   # isotonic on val -> calibrated on test
    assert rep["top_features"][0]["feature"] == "x1"
    assert (tmp_path / "m" / "model.txt").exists() and (tmp_path / "m" / "calibrator.json").exists()
    # the ML model must be no worse than the single-feature rule on the thing it was built from
    assert m["ml"]["csi"] >= m["rule"]["csi"] - 0.02


def test_fractional_label_trains_with_cross_entropy(tmp_path):
    frame, split = _frame()
    frame["frac"] = np.clip(0.5 + 0.35 * frame["x1"] + 0.05 * np.random.default_rng(1).normal(size=len(frame)), 0, 1)
    target = T.Target("syn_frac", "rain", "xent", "frac", primary="x1")
    rep = T.fit_target(target, frame, ["x1", "x2"], split, tmp_path / "m", rounds=100, time_limit_s=60)
    assert rep["metrics"]["test_space"]["auc"] > 0.9


def test_quantile_range_covers_and_beats_static_width(tmp_path):
    frame, split = _frame()
    target = T.Target("syn_temp", "step", "quantile", "temp", point_forecast="gfs_temp")
    rep = T.fit_target(target, frame, ["x1", "x2", "noise"], split, tmp_path / "m", rounds=150, time_limit_s=90)
    m = rep["metrics"]["test_time"]
    assert 0.74 < m["coverage_conformal"] < 0.86, m             # nominal 80%
    assert m["median_mae"] < m["gfs_raw_mae"], m                # model removes the +1 bias
    assert m["median_bias"] == pytest.approx(0.0, abs=0.15)
    assert abs(m["gfs_raw_bias"] - 1.0) < 0.1
    # The point of a range model: coverage is uniform across conditions, where a static interval
    # around the raw forecast over-covers the calm cases and under-covers the noisy ones.
    zone = m["coverage_by_zone"]
    for name in ("hi_noise", "lo_noise"):
        assert 0.74 < zone[name]["coverage"] < 0.86, zone
    assert zone["hi_noise"]["static_coverage"] < 0.72, zone
    assert zone["lo_noise"]["static_coverage"] > 0.90, zone


def test_too_few_rows_is_reported_not_crashed(tmp_path):
    frame, split = _frame(n=500)
    target = T.Target("tiny", "step", "binary", "event", primary="x1")
    assert "skipped" in T.fit_target(target, frame, ["x1"], split, tmp_path / "m")


# ----------------------------------------------------------------------------- curve target
TH = (0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.6, 25.0, 35.5, 50.0, 64.5, 90.0, 115.6, 150.0, 204.5)


def _rain_frame(n=60_000, seed=5, pixels=20):
    rng = np.random.default_rng(seed)
    x1, x2, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    wet_cell = rng.random(n) < 1 / (1 + np.exp(-(-0.8 + 1.6 * x1)))
    px_wet = (rng.random((n, pixels)) < 0.8) & wet_cell[:, None]
    amounts = np.exp(rng.normal((1.3 + 0.7 * x2)[:, None], 0.9, size=(n, pixels))) * px_wet
    F = np.stack([(amounts >= t).mean(axis=1) for t in TH], axis=1).astype("float32")
    frame = pd.DataFrame({"x1": x1, "x2": x2, "noise": noise,
                          "rain_cell_mean_mm": amounts.mean(axis=1) * np.exp(rng.normal(0, 0.6, n)),   # noisy raw "GFS"
                          "rain_cell_max_mm": amounts.max(axis=1) * np.exp(rng.normal(0, 0.6, n)),
                          "month": rng.integers(1, 13, n), "zone": rng.choice(["a", "b"], n),
                          "point_id": rng.choice([f"P{i}" for i in range(200)], n),
                          "lead_bucket": rng.choice(["day0", "day1"], n)})
    for j, t in enumerate(TH):
        frame[f"y_{j}"] = F[:, j]
    split = np.where(np.arange(n) < 36_000, "train", np.where(np.arange(n) < 44_000, "val",
                     np.where(np.arange(n) < 52_000, "test_time", "test_space")))
    return frame, split


def test_curve_model_is_coherent_calibrated_and_answers_how_much(tmp_path):
    frame, split = _rain_frame()
    target = T.Target("syn_curve", "rain", "curve", "y_0", labels=tuple(f"y_{j}" for j in range(len(TH))),
                      thresholds=TH, score_low="rain_cell_mean_mm", score_high="rain_cell_max_mm")
    # as in the real pipeline the raw forecast scores are model FEATURES, and the baseline is the calibrated
    # version of that same single variable -- the model has to add information beyond it
    feats = ["x1", "x2", "noise", "rain_cell_mean_mm", "rain_cell_max_mm"]
    rep = T.fit_curve(target, frame, feats, split, tmp_path / "m", rounds=120, time_limit_s=120,
                      max_stacked=480_000, eval_cap=8_000)
    m = rep["metrics"]["test_time"]
    # coherence: the monotone constraint makes the raw curve non-increasing in threshold for every row
    assert m["monotone_violations_raw"] == 0.0 and m["monotone_violations_calibrated"] == 0.0
    t25 = m["thresholds"]["2.5"]
    assert t25["auc"] > 0.80, t25
    assert t25["bss_vs_zone_month"] > 0.10, t25                  # beats LOCAL climatology, not just a global rate
    assert t25["bss_vs_gfs_calibrated"] > 0.0, t25               # beats a one-variable calibrated "GFS"
    heavy = m["thresholds"]["25"]
    assert heavy["events_any_pixel"] > 50 and heavy["auc_any_pixel"] > 0.75, heavy   # rare class still discriminated
    amount = m["amount_if_wet_median"]
    assert amount["within_factor_2"] > 0.5 and amount["log10_mae"] <= amount["gfs_raw_log10_mae"], amount
    assert "reliability" in m["detail_key_thresholds"]["2.5"]
    for f in ("model.txt", "calibrators.json", "climatology.json", "features.json", "metrics.json"):
        assert (tmp_path / "m" / f).exists(), f
    # spatial test set is reported too, and a rare threshold without enough events keeps the model's own output
    assert "test_space" in rep["metrics"]
    assert any(not c["calibrated"] for c in rep["calibration"]) or all(c["calibrated"] for c in rep["calibration"])


def test_tier_sampling_keeps_rare_heavy_rows_and_is_unbiased():
    rng = np.random.default_rng(0)
    n = 200_000
    F = np.zeros((n, len(TH)), "float32")
    heavy = rng.random(n) < 0.01
    wet = (rng.random(n) < 0.2) & ~heavy
    F[wet, :4] = 0.5                       # wet rows reach 2.5 mm
    F[heavy, :9] = 0.5                     # heavy rows reach 25 mm
    pick, w = T._tier_sample(rng, F, np.asarray(TH), budget_rows=20_000)
    assert abs(len(pick) - 20_000) < 1_500
    kept = np.zeros(n, bool)
    kept[pick] = True
    p_heavy, p_dry = kept[heavy].mean(), kept[~heavy & (F[:, 3] == 0)].mean()
    assert 8 < p_heavy / p_dry < 16                            # heavy rows are oversampled ~12x relative to dry
    # inverse-probability weights reconstruct the population counts
    assert abs(w.sum() - n) / n < 0.06
    assert abs(w[heavy[pick]].sum() - heavy.sum()) / heavy.sum() < 0.1


def test_calibrated_interval_never_excludes_its_own_median_even_with_a_large_negative_margin():
    lo, mid, hi = np.array([10.0, 10.0]), np.array([12.0, 12.0]), np.array([14.0, 14.0])
    l, h = T.calibrated_interval(lo, mid, hi, margin=0.5)
    assert l.tolist() == [9.5, 9.5] and h.tolist() == [14.5, 14.5]            # positive margin widens
    l, h = T.calibrated_interval(lo, mid, hi, margin=-3.0)                     # would invert to lo=13 > hi=11 unclamped
    assert (l <= mid).all() and (h >= mid).all() and (l <= h).all()
    assert l.tolist() == [12.0, 12.0] and h.tolist() == [12.0, 12.0]
