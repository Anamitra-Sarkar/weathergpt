"""The implementation notebook must run top to bottom.  Offline: the model loader and the live-run fetch are replaced by a canned engine that
returns answers with the real engine's exact shapes, so every table, plot, tool call, evidence conversion and HTTP call in the notebook executes.
(This proves the notebook's code paths, not forecast quality; the live path is covered by the Kaggle serve-smoke run.)"""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import nbformat
import pytest

from weathergpt_events import registry as reg_mod
from weathergpt_events.features import RunArrays

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "weathergpt_events_implementation.ipynb"
RUN_DAY = date(2026, 10, 6)
THRESHOLDS = [0.1, 0.5, 1, 2.5, 5, 7.5, 10, 15.6, 25, 35.5, 50, 64.5, 90, 115.6, 150, 204.5]
PROV = {"algorithm_version": "events-v1", "dataset_kind": "t", "dataset_sha256": "ab", "split": {}, "trained_at": "now"}
SEA, OUTSIDE = (15.0, 68.5), (51.5, -0.1)


def _registry():
    reg = reg_mod.EventRegistry()
    spec = {"thunderstorm": ("binary", "step", {"bss_vs_zone_month": .075, "auc": .85, "auc_rule_feature": .72}),
            "rain_3day_any_2p5mm": ("xent", "rain3", {"bss_vs_zone_month": .35, "auc": .89, "auc_rule_feature": .83}),
            "tmax_range": ("quantile", "day", {"median_mae": 1.85, "gfs_raw_mae": 2.43, "coverage_conformal": .74}),
            "rain_curve": ("curve", "rain", {}), "rain_7day_total": ("curve", "rain7", {}),
            "dust": ("binary", "step", {})}
    for name, (kind, table, head) in spec.items():
        passed = name != "dust"
        reg._metrics[name] = {"target": name, "kind": kind, "table": table, "provenance": PROV, "notes": name, "metrics": {"test_space": {}}}
        reg.gates[name] = reg_mod.GateResult(name, passed, "ok" if passed else "no skill over local climatology", head)
        if passed:
            reg.models[name] = object()
    return reg


class CannedEngine:
    """Same call signature and result shapes as `EventEngine`, with plausible made-up numbers."""

    def __init__(self):
        self.registry = _registry()

    def forecast(self, lat, lon, run, targets=None, horizon_days=10, include_unvalidated=False, **_):
        if (lat, lon) in (SEA, OUTSIDE) or abs(lat - 19.81) < 0.01:
            return {"available": False, "reason": "not over land (land fraction 0.00); these models are land-only", "run": run.run_date.isoformat()}
        zone = "indo_gangetic" if lon > 75 else "west_coast"
        out = {"available": True, "run": run.run_date.isoformat(), "location": {"lat": lat, "lon": lon, "land_fraction": 1.0, "coastal": False}, "targets": {}}
        init = datetime(run.run_date.year, run.run_date.month, run.run_date.day, tzinfo=timezone.utc)
        for name in targets or list(self.registry.gates):
            gate = self.registry.gates[name]
            if not gate.passed:
                out["targets"][name] = {"available": False, "reason": f"not served: {gate.reason}"}
                continue
            kind, entries = self.registry._metrics[name]["kind"], []
            if name == "thunderstorm":
                for lead in [3 * i for i in range(1, 41)] + [120 + 6 * i for i in range(1, 13)]:
                    if lead <= horizon_days * 24:
                        entries.append({"valid_time_utc": (init + timedelta(hours=lead)).isoformat(), "lead_h": lead, "validated": True,
                                        "probability": round(0.05 + 0.04 * ((lead // 3) % 5), 3)})
            elif name == "tmax_range":
                for k in range(1, min(horizon_days, 9) + 1):
                    entries.append({"ist_day": str(run.run_day if hasattr(run, "run_day") else run.run_date + timedelta(days=k)), "forecast_day": k, "validated": True,
                                    "q10": 33.0 - .1 * k, "q50": 35.0 - .1 * k, "q90": 35.7, "lo": 32.6 - .1 * k, "hi": 36.1, "nominal_coverage": .8})
            elif name == "rain_3day_any_2p5mm":
                for k in range(0, min(horizon_days, 8)):
                    entries.append({"window_start_utc_day": str(run.run_date + timedelta(days=k)), "start_forecast_day": k, "window_days": 3,
                                    "validated": True, "probability": round(0.1 + 0.05 * k, 3)})
            elif name == "rain_curve":
                for k in range(0, horizon_days):
                    p = max(0.02, 0.5 - 0.04 * k)
                    ex = {f"{t:g}": round(p * 0.8 ** i, 5) for i, t in enumerate(THRESHOLDS)}
                    entries.append({"utc_day": str(run.run_date + timedelta(days=k)), "forecast_day": k, "validated": True, "exceedance": ex, "wet_mm": 1.0,
                                    "p_any_rain": ex["1"], "p_rainy_day": ex["2.5"], "p_heavy_rain": ex["64.5"], "imd_classes": {"light": 0.2, "moderate": 0.1},
                                    "amount_if_wet_mm": {"p10": 1.5, "p50": 6.0, "p90": 20.0} if p > 0.1 else {"p10": None, "p50": None, "p90": None},
                                    "amount_lower_bound": [False, False, False]})
            elif name == "rain_7day_total":
                for k in range(0, min(horizon_days, 4)):
                    ex = {f"{t:g}": round(0.6 * 0.75 ** i, 5) for i, t in enumerate([10, 25, 50, 100, 150, 200])}
                    entries.append({"window_start_utc_day": str(run.run_date + timedelta(days=k)), "start_forecast_day": k, "window_days": 7, "validated": True,
                                    "exceedance": ex, "wet_mm": 10.0, "amount_if_wet_mm": {"p10": 11.0, "p50": 17.0, "p90": 31.0}, "amount_lower_bound": [False] * 3})
            res = {"available": True, "kind": kind, "zone": zone, "entries": entries}
            if kind == "curve":
                res["thresholds_without_skill_mm"] = [90.0, 115.6, 150.0, 204.5]
            out["targets"][name] = res
        return out


def _code_cells():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            cells.append("\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("%", "!"))))
    return cells


def test_notebook_is_valid_and_has_the_expected_sections():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    headings = [line for c in nb.cells if c.cell_type == "markdown" for line in c.source.splitlines() if line.startswith("## ")]
    assert len(headings) >= 12 and any("admission gate" in h for h in headings) and any("evidence objects" in h.lower() for h in headings)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")          # committed clean; real outputs come from a live run


def test_every_code_cell_runs_offline_with_the_canned_engine(monkeypatch, tmp_path, capsys):
    pytest.importorskip("app.schemas.ceo")
    import matplotlib
    matplotlib.use("Agg")
    from weathergpt_events import loader, service as svc
    engine = CannedEngine()
    monkeypatch.setattr(loader, "load_engine", lambda *a, **k: engine)
    monkeypatch.setattr(svc.ForecastService, "live", classmethod(lambda cls, eng, cache_dir, **k: cls(eng, lambda: RunArrays(RUN_DAY, {}, {}))))
    monkeypatch.chdir(tmp_path)
    shown = []
    ns = {"__name__": "notebook", "display": shown.append}
    for i, source in enumerate(_code_cells()):
        try:
            exec(compile(source, f"<notebook cell {i}>", "exec"), ns)
        except Exception as exc:                                                       # name the failing cell
            raise AssertionError(f"notebook code cell {i} failed: {exc!r}\n{source[:300]}") from exc
    out = capsys.readouterr().out
    assert "2 of 6 artifacts served" not in out and "5 of 6 artifacts served" in out      # 5 served + 1 refused (dust)
    assert "Arabian Sea" in out and "London" in out and "land-only" in out                  # refusals shown with their reason
    assert "rejected: unknown tool 'make_it_rain'" in out and "horizon_days must be an integer from 1 to 10" in out
    assert ns["delhi"]["available"] and len(ns["ceos"]) > 0 and ns["r"].status_code == 200
    assert {c.variable.value for c in ns["ceos"]} >= {"precipitation_probability", "rainfall_distribution", "temperature_max"}
    assert len(shown) == 2                                                                 # catalogue table + evidence table (other tables are a cell's last expression)
