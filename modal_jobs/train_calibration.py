"""M4 — ensemble calibration: turning model spread into honest probabilities.

RADE converts a rain probability into an irrigation or spraying recommendation.
If the probability is over-confident the recommendation is wrong with a straight
face, so this is the model that decides whether the decision layer can be
trusted at all.

Counting how many ensemble members exceed a threshold gives a number between 0
and 1, but a small NWP ensemble is systematically under-dispersed: it says 0%
and it rains, it says 100% and it does not.  This fits a proper predictive
distribution instead:

  * precipitation -> a **censored shifted gamma distribution**, the standard
    parametric family for precipitation post-processing.  It has a point mass at
    zero (most hours are dry) and a long right tail (monsoon bursts), neither of
    which a Gaussian can represent.  Parameters come from a small network fitted
    by **directly minimising CRPS**, integrated on a fixed quadrature grid so
    the objective is differentiable and stable in float32.
  * temperature and wind -> a Gaussian EMOS head, also CRPS-fitted.
  * exceedance probabilities at the thresholds that change a farmer's decision
    (0.1, 1, 5, 10, 25, 50 mm), each refined by isotonic regression fitted on
    validation so the reliability curve is monotone.

Three estimators are compared honestly at every threshold — raw ensemble
frequency, the fitted distribution, and the isotonic-refined distribution — and
scored with Brier, Brier skill score against climatology, reliability curves,
rank histograms and PIT.

**Transfer assumption, stated because it is load-bearing.**  Training members
are the four independent NWP models in D1; at request time the live GFS
ensemble supplies thirty-one.  The model consumes only *summary statistics*
(mean, sd, quantiles, wet fraction, spread ratio), never the members
themselves, so the interface is identical — but the mapping from a 4-member
spread to a 31-member spread is an assumption, not a measurement, and it is
recorded in the model card.  It could not be measured: the ensemble API serves
real members only for roughly the last four days while ERA5 truth lags six, so
the two windows do not overlap.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

from modal_jobs.common import DATA_DIR, MODEL_DIR, TRAIN_IMAGE, TRAIN_VOLUMES, app

ALGORITHM_VERSION = "m4_calibration_v1"
THRESHOLDS = (0.1, 1.0, 5.0, 10.0, 25.0, 50.0)
TARGETS = (("precipitation", "csgd"), ("temperature_2m", "gaussian"),
           ("wind_speed_10m", "gaussian"))


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, gpu="A10G",
              timeout=60 * 150, memory=32768)
def train(epochs: int = 40, batch_size: int = 8192, lr: float = 2e-3, seed: int = 42) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.isotonic import IsotonicRegression

    from modal_jobs.contracts import check_forecast_truth_corpus, check_lead_time_signal
    from modal_jobs.features_d1 import (MODELS, build_features, dataset_sha256, load_d1,
                                        split_masks)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frame, files = load_d1(DATA_DIR)
    print(f"[m4] {len(frame):,} rows, {frame['loc_id'].nunique()} locations")

    pairs = [(f"fc_{variable}_{model}", f"truth_{variable}")
             for variable, _ in TARGETS for model in MODELS]
    report = check_forecast_truth_corpus(frame, name="d1_mos", pairs=pairs, min_rows=200_000)
    check_lead_time_signal(frame, report=report, lead_column="lead_age_days",
                           forecast_column="fc_precipitation_gfs_seamless",
                           truth_column="truth_precipitation")
    failures = report.summary()["failed"]
    print("[m4] contracts:", json.dumps(failures, indent=2) if failures else "all passed")

    masks = split_masks(frame, seed=seed)
    print(f"[m4] cutoff={masks['cutoff']} held_out_locations={len(masks['held_out_locations'])}")

    SQRT_PI = math.sqrt(math.pi)
    CRPS_SAMPLES = 96

    def gaussian_crps(mu, sigma, y):
        """Closed-form CRPS of a Gaussian predictive distribution."""
        z = (y - mu) / sigma
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return sigma * (z * (2 * normal.cdf(z) - 1)
                        + 2 * torch.exp(normal.log_prob(z)) - 1.0 / SQRT_PI)

    def csgd_cdf(shape, scale, shift, t):
        """Exact CDF of max(0, X + shift), X ~ Gamma(shape, scale).

        P(Y <= t) = P(X <= t - shift), which is the regularised lower incomplete
        gamma.  Used for evaluation and for exceedance probabilities, where no
        gradient is needed -- `gammainc` has no derivative with respect to the
        shape parameter, which is why the training loss below takes a different
        route.
        """
        return torch.special.gammainc(shape, ((t - shift) / scale).clamp(min=0.0))

    def csgd_crps(shape, scale, shift, y, samples: int = CRPS_SAMPLES):
        """CRPS of the censored shifted gamma, from reparameterised samples.

        Numerical quadrature of the predictive CDF was the obvious approach and
        it is wrong here: a CSGD fitted to mostly-dry precipitation drifts to a
        very small shape parameter, where the gamma density has an integrable
        singularity at zero that no polynomial grid resolves -- a 161-point grid
        recovered 8% of the mass at shape 0.05.  `torch.special.gammainc` is
        exact but carries no gradient in the shape.

        `Gamma.rsample` does have reparameterised gradients in both parameters,
        so the CRPS is estimated from samples using the fair (unbiased) form of
        the energy identity, evaluated through the sorted-sample expression
        rather than an m-by-m pairwise matrix.
        """
        gamma = torch.distributions.Gamma(shape, 1.0 / scale)
        draws = (gamma.rsample((samples,)).transpose(0, 1)
                 + shift.unsqueeze(-1)).clamp(min=0.0)
        ordered, _ = torch.sort(draws, dim=1)
        first = (ordered - y.unsqueeze(-1)).abs().mean(1)
        weights = (2 * torch.arange(1, samples + 1, device=y.device, dtype=y.dtype)
                   - samples - 1)
        second = (weights * ordered).sum(1) / (samples * (samples - 1))
        return first - second

    def crps_ensemble(members: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Fair (unbiased) CRPS of a finite ensemble."""
        members = np.where(np.isfinite(members), members,
                           np.nanmedian(members, axis=1, keepdims=True))
        n = members.shape[1]
        ordered = np.sort(members, axis=1)
        term1 = np.abs(ordered - observation[:, None]).mean(1)
        weights = 2 * np.arange(1, n + 1) - n - 1
        term2 = (weights * ordered).sum(1) / (n * (n - 1))
        return term1 - term2

    class DistributionalHead(nn.Module):
        def __init__(self, in_dim: int, family: str, hidden: int = 192):
            super().__init__()
            self.family = family
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
                nn.Linear(hidden, 3 if family == "csgd" else 2),
            )

        def forward(self, x):
            raw = self.net(x)
            if self.family == "csgd":
                # The shift is the censoring point and is non-positive in the
                # standard CSGD form.  Allowing it positive would push grid
                # points below the shift into the clamped tail of a Gamma with
                # shape < 1, where the density diverges and the integrated CDF
                # saturates at 1 with no usable gradient.
                return (F.softplus(raw[:, 0]) + 0.05,
                        F.softplus(raw[:, 1]) + 0.05,
                        -F.softplus(raw[:, 2]).clamp(max=25.0))
            return raw[:, 0], F.softplus(raw[:, 1]) + 0.05

    results: dict = {}
    artifacts: dict = {}

    for variable, family in TARGETS:
        X, y, names, members, keep = build_features(frame, variable)
        tr = masks["train"] & keep
        va = masks["val"] & keep
        te = masks["test"] & keep
        print(f"[m4:{variable}] train={tr.sum():,} val={va.sum():,} test={te.sum():,}")
        if min(tr.sum(), va.sum(), te.sum()) < 5000:
            raise RuntimeError(f"{variable}: a split is too small to report on")

        mean = X[tr].mean(0)
        std = X[tr].std(0)
        std[std < 1e-6] = 1.0
        to_tensor = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32)  # noqa: E731

        Xtr = to_tensor(X[tr])
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        Xva = to_tensor(X[va]).to(device)
        Xte = to_tensor(X[te]).to(device)
        yva_np, yte_np = y[va], y[te]

        head = DistributionalHead(X.shape[1], family).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        def crps_of(output, target):
            return (csgd_crps(*output, target) if family == "csgd"
                    else gaussian_crps(*output, target))

        n = len(ytr)
        best_val, best_state = float("inf"), None
        for epoch in range(epochs):
            head.train()
            perm = torch.randperm(n)
            total = 0.0
            for start in range(0, n, batch_size):
                index = perm[start:start + batch_size]
                xb = Xtr[index].to(device, non_blocking=True)
                yb = ytr[index].to(device, non_blocking=True)
                loss = crps_of(head(xb), yb).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                total += float(loss) * len(index)
            scheduler.step()
            head.eval()
            with torch.no_grad():
                chunks = [float(crps_of(head(Xva[i:i + 65536]),
                                        torch.tensor(yva_np[i:i + 65536], dtype=torch.float32,
                                                     device=device)).sum())
                          for i in range(0, len(Xva), 65536)]
            val_crps = sum(chunks) / len(yva_np)
            if val_crps < best_val:
                best_val = val_crps
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"[m4:{variable}] epoch {epoch + 1}/{epochs} "
                      f"train_crps={total / n:.4f} val_crps={val_crps:.4f}")
        head.load_state_dict(best_state)
        head.eval()

        @torch.no_grad()
        def sample(x_tensor, size: int = 199) -> np.ndarray:
            out = []
            for i in range(0, len(x_tensor), 32768):
                parameters = head(x_tensor[i:i + 32768])
                if family == "csgd":
                    shape, scale, shift = parameters
                    draws = torch.distributions.Gamma(shape, 1.0 / scale).sample((size,)).T
                    draws = (draws + shift.unsqueeze(-1)).clamp(min=0.0)
                else:
                    mu, sigma = parameters
                    draws = torch.distributions.Normal(mu, sigma).sample((size,)).T
                out.append(draws.cpu().numpy())
            return np.concatenate(out)

        @torch.no_grad()
        def exceedance(x_tensor, threshold: float) -> np.ndarray:
            out = []
            for i in range(0, len(x_tensor), 32768):
                parameters = head(x_tensor[i:i + 32768])
                if family == "csgd":
                    shape, scale, shift = parameters
                    point = torch.full_like(shape, float(threshold))
                    out.append((1.0 - csgd_cdf(shape, scale, shift, point)).cpu().numpy())
                else:
                    mu, sigma = parameters
                    normal = torch.distributions.Normal(mu, sigma)
                    out.append((1.0 - normal.cdf(torch.full_like(mu, threshold))).cpu().numpy())
            return np.concatenate(out)

        scores = {}
        for split_name, mask, x_tensor, observed in (("val", va, Xva, yva_np),
                                                     ("test_spatial_holdout", te, Xte, yte_np)):
            model_samples = sample(x_tensor)
            raw = members[mask]
            model_crps = float(crps_ensemble(model_samples, observed).mean())
            raw_crps = float(crps_ensemble(raw, observed).mean())
            climatology = np.tile(np.quantile(y[tr], np.linspace(0.0025, 0.9975, 199)),
                                  (int(mask.sum()), 1))
            clim_crps = float(crps_ensemble(climatology, observed).mean())

            def flatness(matrix):
                ranks = (np.where(np.isfinite(matrix), matrix, np.inf) < observed[:, None]).sum(1)
                histogram = np.bincount(ranks, minlength=matrix.shape[1] + 1).astype(float)
                histogram /= max(histogram.sum(), 1)
                # 0 is perfectly flat; larger means the ensemble is mis-dispersed
                return float(((histogram - 1 / len(histogram)) ** 2).sum() * len(histogram))

            pit = np.mean(model_samples < observed[:, None], axis=1)
            scores[split_name] = {
                "n": int(mask.sum()),
                "crps_model": model_crps,
                "crps_raw_multi_model_ensemble": raw_crps,
                "crps_climatology": clim_crps,
                "crpss_vs_raw_ensemble": 1 - model_crps / raw_crps,
                "crpss_vs_climatology": 1 - model_crps / clim_crps,
                "rank_histogram_deviation_raw": flatness(raw),
                "rank_histogram_deviation_model": flatness(model_samples),
                "pit_mean": float(pit.mean()), "pit_sd": float(pit.std()),
            }

        if variable == "precipitation":
            isotonics, exceedance_report = {}, {}
            for threshold in THRESHOLDS:
                model_val = exceedance(Xva, threshold)
                observed_val = (yva_np > threshold).astype(float)
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(model_val, observed_val)
                isotonics[threshold] = iso

                raw_test = (members[te] > threshold).mean(1)
                model_test = exceedance(Xte, threshold)
                calibrated = iso.predict(model_test)
                observed_test = (yte_np > threshold).astype(float)
                base_rate = float(observed_test.mean())

                entry = {
                    "base_rate": base_rate,
                    "brier_raw_ensemble_frequency": float(((raw_test - observed_test) ** 2).mean()),
                    "brier_csgd": float(((model_test - observed_test) ** 2).mean()),
                    "brier_csgd_isotonic": float(((calibrated - observed_test) ** 2).mean()),
                    "brier_climatology": float(((base_rate - observed_test) ** 2).mean()),
                }
                for key, label in (("brier_raw_ensemble_frequency", "raw"),
                                   ("brier_csgd", "csgd"),
                                   ("brier_csgd_isotonic", "csgd_isotonic")):
                    entry[f"bss_{label}_vs_climatology"] = (
                        1 - entry[key] / entry["brier_climatology"]
                        if entry["brier_climatology"] > 0 else None)
                curve = []
                bins = np.clip((calibrated * 10).astype(int), 0, 9)
                for b in range(10):
                    selected = bins == b
                    if selected.sum() >= 30:
                        curve.append({"bin": round(b / 10, 1), "n": int(selected.sum()),
                                      "mean_forecast": float(calibrated[selected].mean()),
                                      "observed_frequency": float(observed_test[selected].mean())})
                entry["reliability_test"] = curve
                entry["reliability_max_abs_gap"] = (
                    max(abs(p["mean_forecast"] - p["observed_frequency"]) for p in curve)
                    if curve else None)
                exceedance_report[str(threshold)] = entry
                print(f"[m4:precip>{threshold}mm] base={base_rate:.4f} "
                      f"brier raw={entry['brier_raw_ensemble_frequency']:.5f} "
                      f"csgd={entry['brier_csgd']:.5f} "
                      f"iso={entry['brier_csgd_isotonic']:.5f} "
                      f"(clim {entry['brier_climatology']:.5f})")
            scores["exceedance"] = exceedance_report
            artifacts["isotonics"] = isotonics

        results[variable] = scores
        artifacts[variable] = {"head": head, "mean": mean, "std": std, "family": family,
                               "in_dim": int(X.shape[1]), "feature_names": names}
        print(f"[m4:{variable}] TEST crps={scores['test_spatial_holdout']['crps_model']:.4f} "
              f"raw={scores['test_spatial_holdout']['crps_raw_multi_model_ensemble']:.4f} "
              f"CRPSS={scores['test_spatial_holdout']['crpss_vs_raw_ensemble']:+.4f}")

    out_dir = f"{MODEL_DIR}/{ALGORITHM_VERSION}"
    os.makedirs(out_dir, exist_ok=True)
    import pickle

    torch.save({variable: {"state_dict": artifacts[variable]["head"].state_dict(),
                           "mean": artifacts[variable]["mean"],
                           "std": artifacts[variable]["std"],
                           "family": artifacts[variable]["family"],
                           "in_dim": artifacts[variable]["in_dim"],
                           "feature_names": artifacts[variable]["feature_names"]}
                for variable, _ in TARGETS}, f"{out_dir}/heads.pt")
    with open(f"{out_dir}/isotonics.pkl", "wb") as handle:
        pickle.dump(artifacts.get("isotonics", {}), handle)

    metrics = {
        "algorithm_version": ALGORITHM_VERSION,
        "model_kind": "CSGD (precipitation) + Gaussian EMOS (temperature, wind), CRPS-fitted, "
                      "with isotonic-refined exceedance probabilities",
        "dataset_kind": "d1_multi_model_nwp_vs_era5_seamless",
        "dataset_shards": len(files), "dataset_rows": int(len(frame)),
        "dataset_sha256": dataset_sha256(files),
        "split": "chronological 70% cutoff AND 20% spatially held-out locations",
        "split_cutoff": masks["cutoff"], "held_out_locations": masks["held_out_locations"],
        "ensemble_members_in_training": list(MODELS),
        "transfer_assumption": (
            "Trained on a 4-member multi-model ensemble; served against the 31-member GFS "
            "ensemble. Only summary statistics cross the boundary, so the interface is "
            "identical, but the spread mapping is assumed rather than measured: the ensemble "
            "API serves members only for ~the last 4 days while ERA5 truth lags ~6, so the "
            "two windows never overlap and the transfer could not be verified."),
        "thresholds_mm": list(THRESHOLDS),
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
        "contracts": report.summary(),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }
    with open(f"{out_dir}/metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=str)

    from modal_jobs.common import MODEL_VOL
    MODEL_VOL.commit()
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("held_out_locations", "results")}, indent=2, default=str))
    return metrics


@app.local_entrypoint()
def main(epochs: int = 40):
    train.remote(epochs=epochs)
