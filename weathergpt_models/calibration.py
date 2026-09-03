"""M4 inference — event probabilities that have been checked against reality."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np

from weathergpt_models.features import MODELS, assemble_features, members_from_mapping
from weathergpt_models.types import CalibratedProbability


class _DistributionalHead:
    def __init__(self, torch, in_dim: int, family: str, hidden: int = 192):
        nn = torch.nn
        self.family = family
        self._torch = torch
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, 3 if family == "csgd" else 2))

    def __call__(self, x):
        torch = self._torch
        raw = self.net(x)
        if self.family == "csgd":
            return (torch.nn.functional.softplus(raw[:, 0]) + 0.05,
                    torch.nn.functional.softplus(raw[:, 1]) + 0.05,
                    raw[:, 2].clamp(-25.0, 5.0))
        return raw[:, 0], torch.nn.functional.softplus(raw[:, 1]) + 0.05


class ProbabilityCalibrator:
    """Turns ensemble spread into a probability whose reliability was measured.

    Precipitation uses a censored shifted gamma distribution — a point mass at
    zero for the dry hours and a long right tail for monsoon bursts — fitted by
    minimising CRPS, then refined per threshold by isotonic regression so the
    reliability curve is monotone.  Temperature and wind use a Gaussian EMOS
    head.

    Reliability curves and Brier skill scores per threshold are in
    `metrics.json`, measured on locations the model never trained on.
    """

    def __init__(self, directory: str | Path, *, device: str = "cpu"):
        import torch

        self.directory = Path(directory)
        self.device = device
        self._torch = torch
        metrics = json.loads((self.directory / "metrics.json").read_text())
        self.algorithm_version = metrics.get("algorithm_version", "m4")
        self.metrics = metrics
        self.thresholds = tuple(metrics.get("thresholds_mm", (0.1, 1.0, 5.0, 10.0, 25.0, 50.0)))
        self.transfer_assumption = metrics.get("transfer_assumption", "")

        payload = torch.load(self.directory / "heads.pt", map_location="cpu", weights_only=False)
        self._heads, self._scalers = {}, {}
        for variable, block in payload.items():
            head = _DistributionalHead(torch, block["in_dim"], block["family"])
            head.net.load_state_dict(
                {k[len("net."):]: v for k, v in block["state_dict"].items()
                 if k.startswith("net.")})
            head.net.to(device).eval()
            self._heads[variable] = head
            self._scalers[variable] = (np.asarray(block["mean"]), np.asarray(block["std"]))

        isotonic_path = self.directory / "isotonics.pkl"
        self._isotonics = (pickle.loads(isotonic_path.read_bytes())
                           if isotonic_path.exists() else {})
        self._grid = torch.linspace(0.0, 1.0, 161, device=device) ** 2 * 150.0

    def supports(self, variable: str) -> bool:
        return variable in self._heads

    def _features(self, variable, forecasts, context, other_forecasts):
        members = members_from_mapping(forecasts)
        others = {name: members_from_mapping(values)
                  for name, values in (other_forecasts or {}).items()}
        context = {key: np.asarray([value], dtype="float64") for key, value in context.items()}
        X = assemble_features(variable, members, others, context)
        mean, std = self._scalers[variable]
        return members, self._torch.tensor((X - mean) / std, dtype=self._torch.float32,
                                           device=self.device)

    def exceedance_probability(self, variable: str, threshold: float, *,
                               forecasts: dict, context: dict,
                               other_forecasts: dict | None = None) -> CalibratedProbability:
        if variable not in self._heads:
            raise KeyError(f"{variable} was not trained; available: {sorted(self._heads)}")
        torch = self._torch
        members, tensor = self._features(variable, forecasts, context, other_forecasts)
        head = self._heads[variable]

        with torch.no_grad():
            parameters = head(tensor)
            if head.family == "csgd":
                shape, scale, shift = parameters
                gamma = torch.distributions.Gamma(shape.unsqueeze(-1), 1.0 / scale.unsqueeze(-1))
                argument = (self._grid - shift.unsqueeze(-1)).clamp(min=1e-6)
                density = torch.exp(gamma.log_prob(argument))
                widths = torch.diff(self._grid, prepend=self._grid[:1])
                cdf = torch.cumsum(density * widths, dim=-1).clamp(0.0, 1.0)
                position = int(torch.searchsorted(
                    self._grid, torch.tensor([threshold], device=self.device)
                ).clamp(0, len(self._grid) - 1))
                raw_probability = float(1.0 - cdf[0, position])
            else:
                mu, sigma = parameters
                normal = torch.distributions.Normal(mu, sigma)
                raw_probability = float(1.0 - normal.cdf(torch.full_like(mu, threshold))[0])

        method = "csgd" if head.family == "csgd" else "gaussian"
        probability = raw_probability
        isotonic = self._isotonics.get(threshold) or self._isotonics.get(float(threshold))
        if isotonic is not None:
            probability = float(np.clip(isotonic.predict([raw_probability])[0], 0.0, 1.0))
            method = f"{method}_isotonic"

        with np.errstate(invalid="ignore"):
            live = np.isfinite(members)
            raw_frequency = (float((members[live] > threshold).mean())
                             if live.any() else float("nan"))

        return CalibratedProbability(
            variable=variable, threshold=float(threshold), probability=probability,
            raw_ensemble_frequency=raw_frequency, method=method,
            algorithm_version=self.algorithm_version,
            parents=[model for model in MODELS if forecasts.get(model) is not None])

    def probability_curve(self, variable: str, *, forecasts: dict, context: dict,
                          other_forecasts: dict | None = None) -> list:
        """Calibrated probability at every verified threshold, for a decision engine."""
        return [self.exceedance_probability(variable, threshold, forecasts=forecasts,
                                            context=context, other_forecasts=other_forecasts)
                for threshold in self.thresholds]
