"""M4 inference — event probabilities that have been checked against reality."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from weathergpt_models.features import MODELS, assemble_features, members_from_mapping
from weathergpt_models.types import CalibratedProbability


class _DistributionalHead:
    """Rebuilt to match `train_calibration.py`'s anchored architecture exactly.

    This class previously diverged from the trainer after the anchoring change
    (a 3rd hidden layer and hidden=192->256 were added there but not here), and
    the mismatch surfaced as a `RuntimeError` from `load_state_dict` the first
    time anything actually called `registry.calibration` -- the admission gate
    only checks `metrics.json`, so a shape mismatch in the weights themselves
    was invisible until real use.  Exactly the failure mode this project is
    built to prevent, caught in a real-world smoke test rather than in training.
    """

    def __init__(self, torch, in_dim: int, family: str, hidden: int = 256):
        nn = torch.nn
        self.family = family
        self._torch = torch
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 4 if family == "csgd" else 2))

    def __call__(self, x, anchor_mean, anchor_spread):
        torch = self._torch
        F = torch.nn.functional
        raw = self.net(x)
        if self.family == "csgd":
            # (P(wet), gamma shape, conditional wet mean, censoring shift), must
            # match train_calibration.py's DistributionalHead exactly. The wet
            # mean is anchored on the ensemble mean so the gamma does not have
            # to discover the scale of rainfall from scratch.
            p_wet = torch.sigmoid(raw[:, 0] + 1.0).clamp(1e-4, 1 - 1e-4)
            shape = F.softplus(raw[:, 1] + 0.5) + 0.05
            wet_mean = F.softplus(raw[:, 2] + 1.0) * (anchor_mean.clamp(min=0.0) + 0.2)
            scale = (wet_mean / shape).clamp(min=1e-3)
            shift = -F.softplus(raw[:, 3]).clamp(max=25.0)
            return p_wet, shape, scale, shift
        mu = anchor_mean + raw[:, 0]
        sigma = F.softplus(raw[:, 1] + 0.5) * (anchor_spread + 0.3) + 0.05
        return mu, sigma


class ProbabilityCalibrator:
    """Turns ensemble spread into a probability whose reliability was measured.

    Precipitation only.  A hurdle model: an explicit probability that the hour
    is wet at all, times a censored shifted gamma for how much falls if it is.
    The two questions a rainfall forecast is really answering are separated, and
    the predictive distribution gets a genuine atom at zero -- without which a
    continuous distribution loses to the raw ensemble on the dry hours, which
    are most of them.  Fitted by minimising CRPS, then refined per threshold by
    isotonic regression so the reliability curve is monotone.

    Temperature and wind are not served here.  A first version fit Gaussian EMOS
    heads for both and measured them losing to the raw four-model ensemble's
    fair CRPS by -37.9% and -24.7% -- worse even at the first training epoch,
    which is what a discrete four-point ensemble drawn from genuinely skillful
    models does to a two-parameter symmetric summary of it.  M2 already beats
    that same baseline on both variables using the real forecast distribution;
    use `registry.mos` for a corrected temperature or wind value and its
    interval.

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

    def supports(self, variable: str) -> bool:
        return variable in self._heads

    def _features(self, variable, forecasts, context, other_forecasts):
        members = members_from_mapping(forecasts)
        others = {name: members_from_mapping(values)
                  for name, values in (other_forecasts or {}).items()}
        context = {key: np.asarray([value], dtype="float64") for key, value in context.items()}
        X = assemble_features(variable, members, others, context)
        mean, std = self._scalers[variable]
        tensor = self._torch.tensor((X - mean) / std, dtype=self._torch.float32,
                                    device=self.device)
        with np.errstate(invalid="ignore", all="ignore"):
            anchor_mean = float(np.nan_to_num(np.nanmean(members), nan=0.0))
            anchor_spread = float(np.nan_to_num(np.nanstd(members), nan=0.0))
        anchor_mean_t = self._torch.tensor([anchor_mean], dtype=self._torch.float32,
                                           device=self.device)
        anchor_spread_t = self._torch.tensor([anchor_spread], dtype=self._torch.float32,
                                             device=self.device)
        return members, tensor, anchor_mean_t, anchor_spread_t

    def exceedance_probability(self, variable: str, threshold: float, *,
                               forecasts: dict, context: dict,
                               other_forecasts: dict | None = None) -> CalibratedProbability:
        if variable not in self._heads:
            raise KeyError(f"{variable} was not trained; available: {sorted(self._heads)}")
        torch = self._torch
        members, tensor, anchor_mean, anchor_spread = self._features(
            variable, forecasts, context, other_forecasts)
        head = self._heads[variable]

        with torch.no_grad():
            parameters = head(tensor, anchor_mean, anchor_spread)
            if head.family == "csgd":
                p_wet, shape, scale, shift = parameters
                # hurdle: F(t) = (1 - p) + p * P(max(0, X + shift) <= t)
                point = torch.full_like(shape, float(threshold))
                wet_cdf = torch.special.gammainc(shape,
                                                 ((point - shift) / scale).clamp(min=0.0))
                cdf = (1 - p_wet) + p_wet * wet_cdf
                raw_probability = float(1.0 - cdf[0])
            else:
                mu, sigma = parameters
                normal = torch.distributions.Normal(mu, sigma)
                raw_probability = float(1.0 - normal.cdf(torch.full_like(mu, threshold))[0])

        method = "hurdle_csgd" if head.family == "csgd" else "gaussian"
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
