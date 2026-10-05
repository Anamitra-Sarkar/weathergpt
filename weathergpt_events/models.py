"""Loaders and predictors for the four artifact kinds the trainer writes (binary / xent / quantile / curve).

Prediction code is shared with training wherever possible (`event_models.train._predict_curve`,
`event_models.rain_products`) so the two cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from event_models import rain_products as rp
from event_models import train as training


def _booster(path: Path):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(path))


def _isotonic(p: np.ndarray, x: list, y: list) -> np.ndarray:
    """sklearn IsotonicRegression(out_of_bounds='clip').predict is exactly a clipped linear interpolation."""
    return np.interp(p, np.asarray(x), np.asarray(y))


class _Base:
    kind = ""

    def __init__(self, directory: Path, metrics: dict):
        self.directory, self.metrics = Path(directory), metrics
        self.name = metrics["target"]
        self.features = json.loads((self.directory / "features.json").read_text())

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise KeyError(f"{self.name}: serving frame lacks {len(missing)} training features, e.g. {missing[:5]}")
        return frame[self.features].to_numpy("float32")


class ProbabilityModel(_Base):
    """binary or cross-entropy target -> calibrated probability."""

    def __init__(self, directory, metrics):
        super().__init__(directory, metrics)
        self.kind = metrics["kind"]
        self.booster = _booster(self.directory / "model.txt")
        self.cal = json.loads((self.directory / "calibrator.json").read_text())

    def predict(self, frame: pd.DataFrame) -> dict:
        raw = self.booster.predict(self.matrix(frame))
        p = np.clip(_isotonic(raw, self.cal["x"], self.cal["y"]), 0.0, 1.0)
        return {"probability": p, "decision_threshold": self.cal.get("threshold_ml")}


class RangeModel(_Base):
    """q10 / q50 / q90 with the conformal widening fitted on validation."""

    kind = "quantile"

    def __init__(self, directory, metrics):
        super().__init__(directory, metrics)
        spec = json.loads((self.directory / "interval.json").read_text())
        self.taus, self.transform, self.margin = spec["quantiles"], spec["transform"], spec["conformal_margin"]
        self.boosters = [_booster(self.directory / f"model_q{int(round(t * 100)):02d}.txt") for t in self.taus]

    def predict(self, frame: pd.DataFrame) -> dict:
        X = self.matrix(frame)
        raw = np.sort(np.stack([b.predict(X) for b in self.boosters], axis=1), axis=1)      # monotone quantiles
        q = np.clip(raw, 0, None) ** 3 if self.transform == "cbrt" else raw
        out = {f"q{int(round(t * 100)):02d}": q[:, i] for i, t in enumerate(self.taus)}
        mid = q[:, len(self.taus) // 2]
        out["lo"], out["hi"] = training.calibrated_interval(q[:, 0], mid, q[:, -1], self.margin)   # same rule as the evaluation
        if self.transform == "cbrt":
            out["lo"] = np.clip(out["lo"], 0, None)
        out["nominal_coverage"] = self.taus[-1] - self.taus[0]
        return out


class CurveModel(_Base):
    """Exceedance curve S(t) over `thresholds`, coherent (non-increasing in t) and calibrated per threshold."""

    kind = "curve"

    def __init__(self, directory, metrics):
        super().__init__(directory, metrics)
        self.booster = _booster(self.directory / "model.txt")
        spec = json.loads((self.directory / "calibrators.json").read_text())
        self.thresholds, self.calibrators = np.asarray(spec["thresholds"]), spec["calibrators"]
        self.climatology = json.loads((self.directory / "climatology.json").read_text())
        self.wet = float(metrics.get("wet", 1.0))

    def predict(self, frame: pd.DataFrame, wet: float | None = None) -> dict:
        X = self.matrix(frame)
        raw = training._predict_curve(self.booster, X, np.log(self.thresholds))
        cal = np.stack([raw[:, j] if c is None else _isotonic(raw[:, j], c["x"], c["y"])
                        for j, c in enumerate(self.calibrators)], axis=1)
        S = rp.enforce_monotone(cal)
        return {"exceedance": S, "thresholds": self.thresholds, "wet": wet or self.wet}


def load_model(directory: str | Path):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    kind = metrics["kind"]
    if kind in ("binary", "xent"):
        return ProbabilityModel(directory, metrics)
    if kind == "quantile":
        return RangeModel(directory, metrics)
    if kind == "curve":
        return CurveModel(directory, metrics)
    raise ValueError(f"unknown artifact kind {kind!r} in {directory}")
