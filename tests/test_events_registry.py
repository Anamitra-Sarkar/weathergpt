"""Serving models reproduce the trainer's own reported metrics; the admission gate refuses what it should."""
import json

import numpy as np
import pandas as pd
import pytest

import test_event_train as tt
from event_models import train as T
from weathergpt_events import registry
from weathergpt_events.models import load_model

PROVENANCE = {"algorithm_version": "events-v1", "dataset_kind": "synthetic", "dataset_sha256": "ab" * 32,
              "split": {"train_end": "x"}, "trained_at": "2026-10-06T00:00:00+00:00"}


def _stamp(directory):
    path = directory / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["provenance"] = PROVENANCE
    path.write_text(json.dumps(metrics))
    return metrics


def test_binary_serving_reproduces_the_trainers_test_brier(tmp_path):
    frame, split = tt._frame()
    target = T.Target("syn_event", "step", "binary", "event", primary="x1")
    feats = ["x1", "x2", "noise"]
    rep = T.fit_target(target, frame, feats, split, tmp_path / "syn_event", rounds=80, time_limit_s=60)
    _stamp(tmp_path / "syn_event")
    model = load_model(tmp_path / "syn_event")
    rows = frame[split == "test_time"]
    p = model.predict(rows)["probability"]
    brier = float(np.mean((p - rows["event"].to_numpy()) ** 2))
    assert brier == pytest.approx(rep["metrics"]["test_time"]["brier"], abs=1e-6)
    assert ((p >= 0) & (p <= 1)).all()


def test_range_serving_reproduces_the_trainers_coverage_and_mae(tmp_path):
    frame, split = tt._frame()
    target = T.Target("syn_temp", "step", "quantile", "temp", point_forecast="gfs_temp")
    rep = T.fit_target(target, frame, ["x1", "x2", "noise"], split, tmp_path / "syn_temp", rounds=60, time_limit_s=60)
    _stamp(tmp_path / "syn_temp")
    model = load_model(tmp_path / "syn_temp")
    rows = frame[split == "test_time"]
    out = model.predict(rows)
    y = rows["temp"].to_numpy()
    cov = float(((y >= out["lo"]) & (y <= out["hi"])).mean())
    assert cov == pytest.approx(rep["metrics"]["test_time"]["coverage_conformal"], abs=1e-6)
    assert float(np.mean(np.abs(y - out["q50"]))) == pytest.approx(rep["metrics"]["test_time"]["median_mae"], abs=1e-4)  # trainer rounds to 4 dp
    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()


def test_curve_serving_reproduces_the_trainers_brier_and_is_monotone(tmp_path):
    frame, split = tt._rain_frame(n=30_000)
    n = len(frame)
    split = np.where(np.arange(n) < 0.6 * n, "train", np.where(np.arange(n) < 0.72 * n, "val",
                     np.where(np.arange(n) < 0.86 * n, "test_time", "test_space")))
    labels = tuple(f"y_{j}" for j in range(len(tt.TH)))
    target = T.Target("syn_curve", "rain", "curve", "y_0", labels=labels, thresholds=tt.TH,
                      score_low="rain_cell_mean_mm", score_high="rain_cell_max_mm")
    feats = ["x1", "x2", "noise", "rain_cell_mean_mm", "rain_cell_max_mm"]
    rep = T.fit_curve(target, frame, feats, split, tmp_path / "syn_curve", rounds=40, time_limit_s=60,
                      max_stacked=300_000, eval_cap=10 ** 6)               # eval_cap above n: the trainer scores every row
    _stamp(tmp_path / "syn_curve")
    model = load_model(tmp_path / "syn_curve")
    rows = frame[split == "test_time"]
    out = model.predict(rows)
    S = out["exceedance"]
    assert (np.diff(S, axis=1) <= 1e-9).all()
    j = list(tt.TH).index(2.5)
    brier = float(np.mean((S[:, j] - rows[f"y_{j}"].to_numpy()) ** 2))
    assert brier == pytest.approx(rep["metrics"]["test_time"]["thresholds"]["2.5"]["brier"], abs=1e-6)


def test_registry_loads_models_that_pass_and_explains_refusals(tmp_path):
    frame, split = tt._frame()
    good = T.Target("syn_event", "step", "binary", "event", primary="x1")
    T.fit_target(good, frame, ["x1", "x2", "noise"], split, tmp_path / "syn_event", rounds=60, time_limit_s=60)
    _stamp(tmp_path / "syn_event")
    # an unstamped copy: trained fine, but cannot say what it learned from
    T.fit_target(good, frame, ["x1", "x2", "noise"], split, tmp_path / "no_provenance", rounds=20, time_limit_s=60)
    # a model fitted to pure noise must not pass the margin
    noisy = frame.assign(event=np.random.default_rng(0).integers(0, 2, len(frame)).astype(float))
    T.fit_target(T.Target("noise_event", "step", "binary", "event", primary="noise"), noisy, ["x1", "x2", "noise"], split,
                 tmp_path / "noise_event", rounds=40, time_limit_s=60)
    _stamp(tmp_path / "noise_event")
    reg = registry.EventRegistry.from_dir(tmp_path)
    status = {row["target"]: row for row in reg.status()}
    assert status["syn_event"]["loaded"] and "syn_event" in reg.models
    assert not status["no_provenance"]["loaded"] and "provenance" in status["no_provenance"]["reason"]
    assert not status["noise_event"]["loaded"]                        # fitted to coin flips: no meaningful margin, no discrimination
    assert "margin" in status["noise_event"]["reason"] or "discrimination" in status["noise_event"]["reason"]


def _gate_metrics(**space):
    return {"target": "t", "kind": space.pop("kind"), "provenance": PROVENANCE, "metrics": {"test_space": space}}


def test_gate_rules_for_each_kind():
    ok = registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=0.1, auc=0.8, auc_rule_feature=0.7))
    assert ok.passed
    assert not registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=-0.01, auc=0.8, auc_rule_feature=0.7)).passed
    assert not registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=0.002, auc=0.8, auc_rule_feature=0.7)).passed   # chance-level BSS
    assert "discrimination" in registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=0.05, auc=0.52, auc_rule_feature=0.5)).reason
    assert "raw forecast feature" in registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=0.1, auc=0.6, auc_rule_feature=0.7)).reason
    # within sampling noise of the raw feature is acceptable; the 0.01 tolerance is the only slack
    assert registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=0.1, auc=0.695, auc_rule_feature=0.70)).passed
    assert not registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=0.1, auc=0.685, auc_rule_feature=0.70)).passed
    # a rare-event model can "beat" a noisy zone x month table while being worse than a constant (real coldwave_imd run:
    # BSS +0.667 vs zone x month, -0.461 vs the global rate): it must not be served
    worse_than_constant = registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=0.667, bss_vs_global=-0.461, auc=0.78, auc_rule_feature=0.68))
    assert not worse_than_constant.passed and "global base rate" in worse_than_constant.reason
    assert registry.gate(_gate_metrics(kind="binary", bss_vs_zone_month=0.06, bss_vs_global=0.016, auc=0.865, auc_rule_feature=0.865)).passed
    assert registry.gate(_gate_metrics(kind="quantile", median_mae=1.0, gfs_raw_mae=1.4, coverage_conformal=0.8)).passed
    assert "outside" in registry.gate(_gate_metrics(kind="quantile", median_mae=1.0, gfs_raw_mae=1.4, coverage_conformal=0.55)).reason
    assert not registry.gate(_gate_metrics(kind="quantile", median_mae=1.5, gfs_raw_mae=1.4, coverage_conformal=0.8)).passed
    assert not registry.gate(_gate_metrics(kind="quantile", median_mae=1.395, gfs_raw_mae=1.4, coverage_conformal=0.8)).passed  # <1% gain
    assert not registry.gate({"target": "t", "kind": "binary", "metrics": {}}).passed          # no provenance at all


def test_zone_and_lead_skill_lookup():
    m = {"kind": "binary", "metrics": {"test_space": {
        "by_zone": {"west_coast": {"n": 4000, "bss_vs_local_clim": 0.2}, "northwest_arid": {"n": 3000, "bss_vs_local_clim": -0.05},
                    "islands": {"n": 40, "bss_vs_local_clim": 0.9}},
        "by_lead": {"h000-024": {"n": 3000, "bss_vs_local_clim": 0.3}, "h168-240": {"n": 3000, "bss_vs_local_clim": -0.02}}}}}
    assert registry.zone_lead_skill(m, "west_coast", "h000-024") == (True, "ok")
    ok, why = registry.zone_lead_skill(m, "west_coast", "h168-240")
    assert not ok and "lead" in why
    assert not registry.zone_lead_skill(m, "northwest_arid", None)[0]
    ok, why = registry.zone_lead_skill(m, "islands", None)
    assert not ok and "no held-out evidence" in why                  # 40 rows is not evidence, however good it looks
    assert not registry.zone_lead_skill(m, "himalaya_north", None)[0]


def test_unskilled_thresholds_lists_tail_thresholds_without_held_out_skill():
    from weathergpt_events.registry import unskilled_thresholds
    metrics = {"metrics": {"test_space": {"thresholds": {
        "25": {"threshold_mm": 25.0, "bss_vs_zone_month": 0.148},
        "90": {"threshold_mm": 90.0, "bss_vs_zone_month": 0.0},
        "115.6": {"threshold_mm": 115.6, "bss_vs_zone_month": -0.026},
        "204.5": {"threshold_mm": 204.5, "bss_vs_zone_month": None}}}}}
    assert unskilled_thresholds(metrics) == [90.0, 115.6, 204.5]
    assert unskilled_thresholds({}) == []
