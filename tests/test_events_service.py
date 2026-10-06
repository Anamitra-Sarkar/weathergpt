"""The real-world layer: artifact loader, service (run cache + validated tool calls), CLI, HTTP router and the CEO bridge."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import test_event_train as tt
from event_models import collect_gfs as cg
from event_models import train as T
from weathergpt_events import loader, registry as reg_mod, service as svc, static as static_mod, tools
from weathergpt_events.features import RunArrays

PROVENANCE = {"algorithm_version": "events-v1", "dataset_kind": "synthetic", "dataset_sha256": "ab" * 32,
              "split": {"train_end": "x"}, "trained_at": "2026-10-06T00:00:00+00:00"}


# ------------------------------------------------------------------------------------------------ helpers
def _gate_registry():
    """A registry with one passing binary model, one passing range model and one refused model (metrics only, no boosters)."""
    reg = reg_mod.EventRegistry()
    good = {"target": "thunderstorm", "kind": "binary", "table": "step", "provenance": PROVENANCE, "notes": "x",
            "metrics": {"test_space": {"bss_vs_zone_month": 0.1, "bss_vs_global": 0.1, "auc": 0.85, "auc_rule_feature": 0.7,
                                       "by_zone": {"central": {"n": 5000, "bss_vs_local_clim": 0.1}}}}}
    rng = {"target": "tmax_range", "kind": "quantile", "table": "day", "provenance": PROVENANCE,
           "metrics": {"test_space": {"median_mae": 1.0, "gfs_raw_mae": 1.5, "coverage_conformal": 0.8, "coverage_by_zone": {}}}}
    bad = {"target": "dust", "kind": "binary", "table": "step", "provenance": PROVENANCE,
           "metrics": {"test_space": {"bss_vs_zone_month": 0.0, "bss_vs_global": 0.0, "auc": 0.9, "auc_rule_feature": 0.8}}}
    for name, m in (("thunderstorm", good), ("tmax_range", rng), ("dust", bad)):
        reg._metrics[name], reg.gates[name] = m, reg_mod.gate(m)
        if reg.gates[name].passed:
            reg.models[name] = object()                      # the service only reads the names
    return reg


class FakeEngine:
    def __init__(self):
        self.registry, self.calls = _gate_registry(), []

    def forecast(self, lat, lon, run, targets=None, horizon_days=10, include_unvalidated=False, **_):
        self.calls.append({"lat": lat, "lon": lon, "targets": targets, "horizon_days": horizon_days, "run": run.run_date})
        return {"available": True, "run": run.run_date.isoformat(), "location": {"lat": lat, "lon": lon, "land_fraction": 1.0, "coastal": False},
                "targets": {}}


def _run(day=date(2026, 10, 6)):
    return RunArrays(day, {}, {})


def _service(provider=None, ttl=900.0, clock=None):
    engine = FakeEngine()
    return svc.ForecastService(engine, provider or (lambda: _run()), ttl_s=ttl, **({"clock": clock} if clock else {})), engine


# ------------------------------------------------------------------------------------------------ loader
@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    frame, split = tt._frame()
    target = T.Target("syn_event", "step", "binary", "event", primary="x1")
    T.fit_target(target, frame, ["x1", "x2", "noise"], split, root / "models" / "syn_event", rounds=60, time_limit_s=60)
    path = root / "models" / "syn_event" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["provenance"] = PROVENANCE
    path.write_text(json.dumps(metrics))
    static_mod.save_static_grids(static_mod.build_static_grids(np.ones((cg.NR, cg.NC), "float32"), np.full((cg.NR, cg.NC), 100.0, "float32")),
                                 root / "static_grids.npz")
    pd.DataFrame({"point_id": ["G:20.00:78.00", "S:XXXX"], "kind": ["node", "station"], "lat": [20.0, 20.1], "lon": [78.0, 78.1],
                  "land_frac": [1.0, 1.0]}).to_parquet(root / "points.parquet", index=False)
    pd.DataFrame({"point_id": ["G:20.00:78.00"], "bin": [1], "t2m_c": [25.0]}).to_parquet(root / "climatology_day.parquet", index=False)
    return root


def test_load_engine_from_a_folder_serves_the_models_that_pass_the_gate(artifact_dir):
    engine = loader.load_engine(artifact_dir)
    assert "syn_event" in engine.registry.models and engine.registry.gates["syn_event"].passed
    assert engine.builder.min_land == 0.5 and set(engine.builder.grids) >= {"land", "hgt", "dzdx", "dzdy"}


def test_load_engine_names_what_is_missing(tmp_path, artifact_dir):
    with pytest.raises(FileNotFoundError, match="points.parquet"):
        loader.check_layout(tmp_path)
    broken = tmp_path / "broken"
    (broken / "models").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="static_grids.npz"):
        loader.load_engine(broken)
    with pytest.raises(FileNotFoundError, match="neither a folder nor a Hugging Face repo id"):
        loader.resolve_source("not-a-folder-and-not-a-repo")


def test_load_engine_refuses_when_no_model_passes_the_gate(artifact_dir, tmp_path):
    import shutil
    root = tmp_path / "nothing"
    shutil.copytree(artifact_dir, root)
    path = root / "models" / "syn_event" / "metrics.json"
    metrics = json.loads(path.read_text())
    del metrics["provenance"]
    path.write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="no model passed the admission gate"):
        loader.load_engine(root)


# ------------------------------------------------------------------------------------------------ service
def test_run_is_fetched_once_per_ttl_and_a_missing_run_is_a_clear_refusal():
    calls, now = [], [0.0]

    def provider():
        calls.append(1)
        return _run()
    service, engine = _service(provider, ttl=100.0, clock=lambda: now[0])
    for _ in range(3):
        service.forecast(28.6, 77.2)
    assert len(calls) == 1
    now[0] = 101.0
    service.forecast(28.6, 77.2)
    assert len(calls) == 2
    stale, _ = _service(lambda: None)
    out = stale.forecast(28.6, 77.2)
    assert out["available"] is False and "stale" in out["reason"]
    assert stale.status()["run"] is None


def test_tool_call_dispatch_validates_everything_the_catalogue_promises():
    service, engine = _service()
    out = service.call_tool("thunderstorm", {"lat": 19.08, "lon": 72.88, "horizon_days": 3})
    assert out["available"] and engine.calls[-1] == {"lat": 19.08, "lon": 72.88, "targets": ["thunderstorm"], "horizon_days": 3, "run": date(2026, 10, 6)}
    assert service.call_tool("thunderstorm", {"lat": 19, "lon": 72})["available"]                # ints are numbers; default horizon is 10
    assert engine.calls[-1]["horizon_days"] == 10
    refused = service.call_tool("dust", {"lat": 19.0, "lon": 72.0})                              # known but not served: an answer, not an exception
    assert refused["available"] is False and "not served" in refused["reason"]
    assert "tools" in service.call_tool("catalogue") and "served" in service.call_tool("status")
    bad_calls = [("nonexistent", {"lat": 1, "lon": 2}), ("thunderstorm", {"lat": "x", "lon": 2}), ("thunderstorm", {"lat": 1}),
                 ("thunderstorm", {"lat": float("nan"), "lon": 2}), ("thunderstorm", {"lat": True, "lon": 2}),
                 ("thunderstorm", {"lat": 1, "lon": 2, "horizon_days": 11}), ("thunderstorm", {"lat": 1, "lon": 2, "horizon_days": 0}),
                 ("thunderstorm", {"lat": 1, "lon": 2, "horizon_days": 2.5}), ("thunderstorm", {"lat": 1, "lon": 2, "units": "metric"}),
                 ("catalogue", {"x": 1})]
    for name, args in bad_calls:
        with pytest.raises(svc.ToolCallError):
            service.call_tool(name, args)
    assert len(engine.calls) == 2                                                                # none of the bad calls reached the engine


def test_summarise_is_honest_about_refusals_and_unvalidated_zones():
    assert "NOT AVAILABLE" in svc.summarise({"available": False, "reason": "outside the modelled domain"})
    result = {"available": True, "run": "2026-10-06", "location": {"lat": 13.08, "lon": 80.27, "land_fraction": 0.29, "coastal": True},
              "targets": {
                  "thunderstorm": {"available": True, "zone": "east_coast", "kind": "binary", "entries": [
                      {"valid_time_utc": "2026-10-06T03:00:00", "lead_h": 3, "validated": True, "probability": 0.0612}]},
                  "dust": {"available": True, "zone": "east_coast", "kind": "binary", "entries": [
                      {"valid_time_utc": "2026-10-06T03:00:00", "lead_h": 3, "validated": False, "note": "no held-out evidence for zone 'east_coast'"}]},
                  "tmax_range": {"available": True, "zone": "east_coast", "kind": "quantile", "entries": [
                      {"ist_day": "2026-10-07", "forecast_day": 1, "validated": True, "q10": 31.0, "q50": 32.4, "q90": 33.1, "lo": 30.6, "hi": 33.5}]},
                  "rain_curve": {"available": True, "zone": "east_coast", "kind": "curve", "thresholds_without_skill_mm": [90.0], "entries": [
                      {"utc_day": "2026-10-07", "forecast_day": 1, "validated": True, "p_any_rain": 0.4, "p_rainy_day": 0.3,
                       "amount_if_wet_mm": {"p10": 1.5, "p50": 6.0, "p90": 20.0}}]},
                  "fog": {"available": False, "reason": "not served: x"}}}
    text = svc.summarise(result)
    assert "[coastal cell]" in text and "P = 0.061" in text and "31.0 / 32.4 / 33.1" in text and "6.0" in text
    assert "no validated answer in zone 'east_coast'" in text and "not available: not served: x" in text and "[90.0] mm" in text


# ------------------------------------------------------------------------------------------------ CLI
def test_cli_prints_catalogue_status_and_forecast(monkeypatch, capsys):
    from weathergpt_events import __main__ as cli
    service, engine = _service()
    monkeypatch.setattr(cli, "load_engine", lambda *a, **k: engine)
    monkeypatch.setattr(cli.ForecastService, "live", classmethod(lambda cls, eng, cache, **k: service))
    assert cli.main(["--source", "x", "catalogue"]) == 0
    out = capsys.readouterr().out
    assert "served  thunderstorm" in out and "refused dust" in out
    assert cli.main(["--source", "x", "status"]) == 0
    assert "newest fresh run: 2026-10-06" in capsys.readouterr().out
    assert cli.main(["--source", "x", "--json", "forecast", "--lat", "28.6", "--lon", "77.2", "--targets", "thunderstorm,tmax_range", "--horizon", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["available"] is True
    assert engine.calls[-1]["targets"] == ["thunderstorm", "tmax_range"] and engine.calls[-1]["horizon_days"] == 2
    stale = svc.ForecastService(engine, lambda: None)
    monkeypatch.setattr(cli.ForecastService, "live", classmethod(lambda cls, eng, cache, **k: stale))
    assert cli.main(["--source", "x", "forecast", "--lat", "1", "--lon", "2"]) == 2          # exit code 2 = no answer available


# ------------------------------------------------------------------------------------------------ HTTP router
def test_router_exposes_the_service_and_maps_bad_tool_calls_to_422():
    from fastapi.testclient import TestClient
    from weathergpt_events.api import create_app
    service, engine = _service()
    client = TestClient(create_app(service))
    assert client.get("/status").json()["served"] == ["thunderstorm", "tmax_range"]
    assert {t["name"] for t in client.get("/catalogue").json()["tools"]} == {"thunderstorm", "tmax_range", "dust"}
    r = client.get("/forecast", params={"lat": 28.6, "lon": 77.2, "targets": "thunderstorm", "horizon_days": 2})
    assert r.status_code == 200 and engine.calls[-1]["targets"] == ["thunderstorm"]
    assert client.get("/forecast", params={"lat": 128.6, "lon": 77.2}).status_code == 422             # impossible latitude rejected by the schema
    assert client.get("/forecast", params={"lat": 28.6, "lon": 77.2, "horizon_days": 11}).status_code == 422
    assert client.post("/tool", json={"name": "thunderstorm", "arguments": {"lat": 19.0, "lon": 72.9}}).status_code == 200
    bad = client.post("/tool", json={"name": "nope", "arguments": {}})
    assert bad.status_code == 422 and "unknown tool" in bad.json()["detail"]
    assert client.post("/tool", json={"name": "dust", "arguments": {"lat": 19.0, "lon": 72.9}}).json()["available"] is False


# ------------------------------------------------------------------------------------------------ CEO bridge
def _answer():
    step = {"valid_time_utc": "2026-10-06T06:00:00", "lead_h": 6, "validated": True, "probability": 0.2, "lead_bucket": "h4_12"}
    return {"available": True, "run": "2026-10-06", "location": {"lat": 28.6, "lon": 77.2, "land_fraction": 1.0, "coastal": False}, "targets": {
        "thunderstorm": {"available": True, "kind": "binary", "zone": "indo_gangetic", "entries": [step, {**step, "validated": False, "probability": None, "note": "x"}]},
        "tmax_range": {"available": True, "kind": "quantile", "zone": "indo_gangetic", "entries": [
            {"ist_day": "2026-10-07", "forecast_day": 1, "validated": True, "q10": 33.0, "q50": 35.0, "q90": 35.7, "lo": 32.6, "hi": 36.1, "nominal_coverage": 0.8}]},
        "rain_curve": {"available": True, "kind": "curve", "zone": "indo_gangetic", "thresholds_without_skill_mm": [90.0], "entries": [
            {"utc_day": "2026-10-07", "forecast_day": 1, "validated": True, "exceedance": {"0.1": 0.3, "1": 0.2, "2.5": 0.1}, "wet_mm": 1.0,
             "p_any_rain": 0.2, "p_rainy_day": 0.1, "p_heavy_rain": 0.0, "imd_classes": {"light": 0.1}, "amount_if_wet_mm": {"p10": 1.0, "p50": 4.0, "p90": 9.0},
             "amount_lower_bound": [False, False, False]},
            {"utc_day": "2026-10-08", "forecast_day": 2, "validated": True, "exceedance": {"0.1": 0.01}, "wet_mm": 1.0, "p_any_rain": 0.01, "p_rainy_day": 0.0,
             "p_heavy_rain": 0.0, "imd_classes": {}, "amount_if_wet_mm": {"p10": None, "p50": None, "p90": None}, "amount_lower_bound": [False] * 3}]},
        "rain_7day_total": {"available": True, "kind": "curve", "zone": "indo_gangetic", "entries": [
            {"window_start_utc_day": "2026-10-06", "start_forecast_day": 0, "window_days": 7, "validated": True, "exceedance": {"10": 0.3}, "wet_mm": 10.0,
             "amount_if_wet_mm": {"p10": 11.0, "p50": 17.0, "p90": 31.0}, "amount_lower_bound": [False] * 3}]},
        "dust": {"available": False, "reason": "not served: x"}}}


def test_ceo_bridge_emits_only_validated_evidence_and_never_mixes_amount_with_probability():
    pytest.importorskip("app.schemas.ceo")
    from app.schemas.ceo import CanonicalEvidenceObject
    from weathergpt_events.ceo_bridge import to_ceos
    ceos = to_ceos(_answer())
    assert all(isinstance(c, CanonicalEvidenceObject) for c in ceos)
    by = {}
    for c in ceos:
        by.setdefault(c.model_name.split(":")[1], []).append(c)
    assert len(by["thunderstorm"]) == 1 and by["thunderstorm"][0].probability == 0.2 and by["thunderstorm"][0].variable.value == "thunderstorm_probability"
    assert by["thunderstorm"][0].valid_to - by["thunderstorm"][0].valid_from == pd.Timedelta(hours=3).to_pytimedelta()
    tmax = by["tmax_range"][0]
    assert tmax.variable.value == "temperature_max" and tmax.value == 35.0 and tmax.extra["q10"] == 33.0 and tmax.extra["value_is"] == "median"
    assert tmax.valid_from.isoformat().startswith("2026-10-06T18:30")                       # IST day starts 18:30 UTC the evening before
    rain = by["rain_curve"]
    kinds = [(c.variable.value, c.statistic.value) for c in rain]
    assert ("precipitation_probability", "probability") in kinds and ("rainfall_distribution", "probability") in kinds
    amounts = [c for c in rain if c.variable.value == "precipitation_amount"]
    assert len(amounts) == 1 and amounts[0].extra["conditional_on_wet"] is True and amounts[0].value == 4.0     # day 2 has no meaningful amount -> none emitted
    assert all(c.probability is None for c in amounts) and all(c.value is None for c in rain if c.variable.value != "precipitation_amount")
    assert any(c.extra.get("thresholds_without_skill_mm") == [90.0] for c in rain)
    week = [c for c in by["rain_7day_total"] if c.variable.value == "precipitation_amount"][0]
    assert week.accumulation_window_hours == 168.0 and week.extra["conditional_on_wet"] is True
    assert "dust" not in by and all(c.quality_flag == "validated" and c.model_initialization_time.isoformat().startswith("2026-10-06T00:00") for c in ceos)
    assert to_ceos({"available": False, "reason": "x"}) == []
