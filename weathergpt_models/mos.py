"""M2 inference — bias-corrected value with a calibrated interval."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weathergpt_models.features import MODELS, assemble_features, members_from_mapping
from weathergpt_models.types import CorrectedForecast


class _QuantileNet:
    """Rebuilt to match the training architecture exactly."""

    def __init__(self, torch, in_dim: int, n_quantiles: int, hidden: int = 256):
        nn = torch.nn
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.SiLU())
        self.head = nn.Linear(hidden // 2, n_quantiles)
        self._torch = torch

    def modules(self):
        return {"body": self.body, "head": self.head}

    def __call__(self, x):
        torch = self._torch
        raw = self.head(self.body(x))
        # cumulative softplus: the quantiles are monotone by construction, so a
        # 90th percentile can never come back below the 10th
        return raw[:, :1] + torch.cat(
            [torch.zeros_like(raw[:, :1]),
             torch.cumsum(torch.nn.functional.softplus(raw[:, 1:]), dim=1)], dim=1)


class MOSCorrector:
    """Post-processes raw NWP output into a corrected value plus an interval.

    The interval is conformalised: the width adjustment was fitted on a
    held-out slice so the nominal 80% band really contains the truth about 80%
    of the time, with no distributional assumption. The measured coverage on the
    spatially held-out test set is in `metrics.json`.
    """

    QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

    def __init__(self, directory: str | Path, *, device: str = "cpu"):
        import torch

        self.directory = Path(directory)
        self.device = device
        self._torch = torch
        metrics = json.loads((self.directory / "metrics.json").read_text())
        self.algorithm_version = metrics.get("algorithm_version", "m2")
        self.metrics = metrics

        payload = torch.load(self.directory / "quantile_nets.pt", map_location="cpu",
                             weights_only=False)
        self.variables = list(payload.keys())
        self._nets, self._scalers, self._blend, self._served = {}, {}, {}, {}
        self._boosters = {}
        for variable, block in payload.items():
            net = _QuantileNet(torch, block["in_dim"], len(self.QUANTILES))
            state = block["state_dict"]
            net.body.load_state_dict(
                {k[len("body."):]: v for k, v in state.items() if k.startswith("body.")})
            net.head.load_state_dict(
                {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})
            for module in net.modules().values():
                module.to(device).eval()
            self._nets[variable] = net
            self._scalers[variable] = (np.asarray(block["mean"]), np.asarray(block["std"]),
                                       float(block["conformal_q"]))
            self._blend[variable] = float(block.get("blend_weight", 1.0))
            self._served[variable] = block.get("served_head", "quantile_net")

        # The gradient-boosted head is only loaded when it is actually part of
        # what won on the spatial holdout, so a pure-network deployment does not
        # pay for lightgbm at all.
        if any(head in ("lightgbm", "blend") for head in self._served.values()):
            import lightgbm as lgb

            for variable in self.variables:
                if self._served[variable] == "quantile_net":
                    continue
                boosters = {}
                for level in self.QUANTILES:
                    path = self.directory / f"lgbm_{variable}_q{int(level * 100):02d}.txt"
                    if path.exists():
                        boosters[level] = lgb.Booster(model_file=str(path))
                if len(boosters) == len(self.QUANTILES):
                    self._boosters[variable] = boosters
                else:
                    # fall back rather than serve a partial quantile ladder
                    self._served[variable] = "quantile_net"

    def supports(self, variable: str) -> bool:
        return variable in self._nets

    def correct(self, variable: str, *, forecasts: dict, context: dict,
                other_forecasts: dict | None = None) -> CorrectedForecast:
        """`forecasts` maps NWP model name to its raw value for this variable.

        `context` needs lead_hours, lead_age_days, hour_utc, doy, elevation_m,
        lat, lon.  `other_forecasts` maps the other canonical variables to their
        own {model: value} dicts; omit any you do not have and the corresponding
        cross-variable features are zeroed.
        """
        if variable not in self._nets:
            raise KeyError(f"{variable} was not trained; available: {sorted(self._nets)}")
        torch = self._torch
        members = members_from_mapping(forecasts)
        others = {name: members_from_mapping(values)
                  for name, values in (other_forecasts or {}).items()}
        context = {key: np.asarray([value], dtype="float64") for key, value in context.items()}

        X = assemble_features(variable, members, others, context)
        mean, std, conformal = self._scalers[variable]
        tensor = torch.tensor((X - mean) / std, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            net_prediction = self._nets[variable](tensor).cpu().numpy()[0]

        head = self._served[variable]
        if head == "quantile_net" or variable not in self._boosters:
            predicted = net_prediction
        else:
            gbm = np.array([self._boosters[variable][level].predict(X)[0]
                            for level in self.QUANTILES])
            weight = self._blend[variable] if head == "blend" else 0.0
            predicted = weight * net_prediction + (1 - weight) * gbm
        # independently fitted quantiles can cross; re-sorting is what the
        # training-time evaluation did, so serving must do the same
        predicted = np.sort(predicted)

        quantiles = {level: float(predicted[i]) for i, level in enumerate(self.QUANTILES)}
        median = quantiles[0.5]
        with np.errstate(invalid="ignore"):
            raw_mean = float(np.nanmean(members))
        if variable in ("precipitation",):
            median = max(0.0, median)
            quantiles = {level: max(0.0, value) for level, value in quantiles.items()}

        return CorrectedForecast(
            variable=variable, value=median, quantiles=quantiles,
            interval_low=float(quantiles[0.1] - conformal),
            interval_high=float(quantiles[0.9] + conformal),
            interval_coverage_nominal=0.80,
            raw_ensemble_mean=raw_mean,
            correction=float(median - raw_mean) if np.isfinite(raw_mean) else float("nan"),
            algorithm_version=f"{self.algorithm_version}:{head}",
            parents=[model for model in MODELS if forecasts.get(model) is not None])
