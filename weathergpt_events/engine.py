"""Turn a live run into answers for one location.

Safety default: when a model has no demonstrated skill for the user's climate zone / lead time (from its own held-out
breakdowns), the entry carries NO numbers -- only `validated: false` and the reason.  `include_unvalidated=True`
returns the numbers anyway, for debugging.  Nothing here invents a value: a dry forecast has no conditional amount
(NaN -> None), an over-range tail is flagged as a lower bound, a point over the sea or outside the box is refused.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from event_models import rain_products as rp
from weathergpt_events import static as static_mod
from weathergpt_events.features import FeatureBuilder, RunArrays
from weathergpt_events.registry import EventRegistry

WINDOW_OF = {"rain3": 3, "rain7": 7}


def _py(x):
    """numpy / pandas scalars -> JSON-friendly python (NaN -> None)."""
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


class EventEngine:
    def __init__(self, registry: EventRegistry, builder: FeatureBuilder):
        self.registry, self.builder = registry, builder

    # ------------------------------------------------------------------ tables
    def _table(self, kind: str, run: RunArrays, lat, lon, point_id, static, cache: dict) -> pd.DataFrame:
        if kind not in cache:
            if kind == "step":
                cache[kind] = self.builder.step_table(run, lat, lon, point_id, static)
            elif kind == "day":
                cache[kind] = self.builder.station_day_table(run, lat, lon, point_id, static)
            elif kind == "rain":
                cache[kind] = self.builder.rain_day_table(run, lat, lon, point_id, static)
            elif kind in WINDOW_OF:
                cache[kind] = self.builder.window_table(run, lat, lon, WINDOW_OF[kind], point_id, static)
            else:
                raise ValueError(f"unknown feature table {kind!r}")
        return cache[kind]

    @staticmethod
    def _when(kind: str, row) -> dict:
        if kind == "step":
            return {"valid_time_utc": _py(row.valid_time), "lead_h": int(row.lead_h)}
        if kind == "day":
            return {"ist_day": str(pd.Timestamp(row.ist_day).date()), "forecast_day": int(row.day_k)}
        if kind == "rain":
            return {"utc_day": str(pd.Timestamp(row.valid_date).date()), "forecast_day": int(row.day_k)}
        return {"window_start_utc_day": str(pd.Timestamp(row.valid_date).date()), "start_forecast_day": int(row.day_k),
                "window_days": WINDOW_OF[kind]}

    # ------------------------------------------------------------------ predictions
    def _values(self, model, out: dict, i: int) -> dict:
        if model.kind in ("binary", "xent"):
            return {"probability": _py(out["probability"][i])}
        if model.kind == "quantile":
            keys = [k for k in out if k.startswith("q")] + ["lo", "hi"]
            return {**{k: _py(out[k][i]) for k in keys}, "nominal_coverage": _py(out["nominal_coverage"])}
        S = out["exceedance"][i:i + 1]
        prods = rp.rain_products(S, out["thresholds"], levels=(0.1, 0.5, 0.9)) if model.name == "rain_curve" else None
        values = {"exceedance": {f"{t:g}": _py(S[0, j]) for j, t in enumerate(out["thresholds"])}, "wet_mm": out["wet"]}
        if prods is not None:
            amounts = prods["amount_if_wet"][0]
            values.update({
                "p_any_rain": _py(prods["p_any_rain"][0]), "p_rainy_day": _py(prods["p_rainy_day"][0]),
                "p_heavy_rain": _py(prods["p_heavy"][0]),
                "imd_classes": {k: _py(v[0]) for k, v in prods["class_probs"].items()},
                "amount_if_wet_mm": {f"p{int(l * 100)}": _py(a) for l, a in zip(prods["amount_levels"], amounts)},
                "amount_lower_bound": [bool(c) for c in prods["amount_capped"][0]]})
        else:                                   # window totals: conditional amounts given the model's own wet level
            amounts, capped = rp.amount_quantiles(S, out["thresholds"], levels=(0.1, 0.5, 0.9), wet=out["wet"])
            values["amount_if_wet_mm"] = {f"p{p}": _py(a) for p, a in zip((10, 50, 90), amounts[0])}
            values["amount_lower_bound"] = [bool(c) for c in capped[0]]
        return values

    def forecast(self, lat: float, lon: float, run: RunArrays, targets: list | None = None, point_id: str | None = None,
                 static: pd.DataFrame | None = None, horizon_days: int = 10, include_unvalidated: bool = False) -> dict:
        ok, why = static_mod.in_domain(lat, lon, self.builder.grids)
        if not ok:
            return {"available": False, "reason": why, "run": run.run_date.isoformat()}
        result = {"available": True, "run": run.run_date.isoformat(), "location": {"lat": lat, "lon": lon}, "targets": {}}
        cache: dict = {}
        for name in targets or list(self.registry.gates):
            gate = self.registry.gates.get(name)
            if gate is None:
                result["targets"][name] = {"available": False, "reason": "unknown target"}
                continue
            if name not in self.registry.models:
                result["targets"][name] = {"available": False, "reason": f"not served: {gate.reason}"}
                continue
            model = self.registry.models[name]
            kind = self.registry._metrics[name]["table"]
            frame = self._table(kind, run, lat, lon, point_id, static, cache)
            if frame.empty:
                result["targets"][name] = {"available": False, "reason": "no complete forecast rows in this run for this target"}
                continue
            frame = frame[self._within(frame, kind, horizon_days)].reset_index(drop=True)
            out = model.predict(frame)
            zone = str(frame["zone"].iloc[0])
            entries = []
            for i, row in enumerate(frame.itertuples()):
                validated, note = self.registry.skill(name, zone, getattr(row, "lead_bucket", None))
                entry = {**self._when(kind, row), "validated": validated}
                if validated or include_unvalidated:
                    entry.update(self._values(model, out, i))
                if not validated:
                    entry["note"] = note
                entries.append(entry)
            result["targets"][name] = {"available": True, "kind": model.kind, "zone": zone, "entries": entries}
        return result

    @staticmethod
    def _within(frame: pd.DataFrame, kind: str, horizon_days: int) -> np.ndarray:
        limit = {"step": frame.get("lead_h"), "day": frame.get("day_k"), "rain": frame.get("day_k")}.get(kind)
        if limit is None:
            return np.ones(len(frame), bool)
        return (limit <= (horizon_days * 24 if kind == "step" else horizon_days - 1)).to_numpy()
