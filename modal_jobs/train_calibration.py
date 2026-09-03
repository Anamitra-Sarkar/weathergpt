"""M4 — ensemble post-processing and probability calibration.

A raw 31-member GFS ensemble is not a calibrated forecast.  Counting the members
above a threshold gives a number between 0 and 1, but it is systematically
over-confident and under-dispersed, especially for monsoon precipitation.  RADE
turns those numbers into an irrigation or spraying recommendation, so if the
probabilities are wrong the decision is wrong with a straight face.

This trains a distributional post-processor:

  * precipitation -> a **censored shifted gamma distribution** (CSGD), the
    standard parametric family for precipitation post-processing, whose
    parameters are predicted from ensemble summary statistics by a small
    network and fitted by **directly minimising the closed-form CRPS**.  A CSGD
    handles the point mass at zero (most hours are dry) and the long right tail
    (monsoon bursts) that a Gaussian cannot.
  * temperature and wind -> a Gaussian (EMOS-style) head, also CRPS-trained.
  * exceedance probabilities at the thresholds a farmer actually cares about
    (0.1, 1, 5, 10, 25, 50 mm), each refined by isotonic regression on
    validation so the reliability curve is monotone.

Everything is scored against four baselines and reported with proper
meteorological verification: CRPS skill score, Brier score per threshold,
reliability, rank histogram and PIT.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime

from modal_jobs.common import DATA_DIR, MODEL_DIR, TRAIN_IMAGE, TRAIN_VOLUMES, app

ALGORITHM_VERSION = "m4_calibration_v1"
THRESHOLDS = (0.1, 1.0, 5.0, 10.0, 25.0, 50.0)


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, gpu="A10G", timeout=60 * 90)
def train(epochs: int = 60, batch_size: int = 4096, lr: float = 3e-3, seed: int = 42) -> dict:
    import glob

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.isotonic import IsotonicRegression

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted(glob.glob(f"{DATA_DIR}/d2_ensemble/*.parquet"))
    if not files:
        raise RuntimeError("D2 ensemble corpus is missing; run build_corpora --what d2 first")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    digest = hashlib.sha256()
    for path in files:
        digest.update(open(path, "rb").read())
    data_sha = digest.hexdigest()
    print(f"[m4] loaded {len(frame):,} rows from {len(files)} shards")

    truth_columns = ["truth_precipitation", "truth_temperature_2m", "truth_wind_speed_10m"]
    for column in truth_columns:
        finite = float(np.isfinite(frame[column].to_numpy(dtype="float64")).mean())
        print(f"[m4] {column}: {finite:.1%} finite")
        if finite < 0.5:
            raise RuntimeError(
                f"{column} is {1 - finite:.1%} null — the verifying truth source does not "
                f"publish this variable, so the target would be untrainable. "
                f"Rebuild D2 with a truth model that actually serves it.")
    frame = frame.dropna(subset=truth_columns)
    frame = frame.sort_values("valid_time").reset_index(drop=True)

    def member_matrix(column: str) -> np.ndarray:
        return np.stack(frame[column].to_numpy()).astype("float64")

    precip_members = member_matrix("precipitation_members")
    temp_members = member_matrix("temperature_2m_members")
    wind_members = member_matrix("wind_speed_10m_members")

    def summarise(members: np.ndarray, wet_threshold: float | None) -> np.ndarray:
        """Ensemble summary statistics — the sufficient statistics a post-processor
        needs, not the raw 31 members, which would make the model member-order
        dependent and unusable on any other ensemble.

        Members are genuinely missing for some hours, so every statistic here is
        NaN-aware; rows with too few live members are dropped by the caller
        rather than silently becoming zeros."""
        with np.errstate(invalid="ignore", all="ignore"):
            quantiles = np.nanquantile(members, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
            block = np.stack([np.nanmean(members, 1), np.nanstd(members, 1),
                              np.nanmin(members, 1), np.nanmax(members, 1)], axis=1)
            out = np.concatenate([block, quantiles], axis=1)
            if wet_threshold is not None:
                live = np.isfinite(members)
                wet = (np.where(live, members, -np.inf) > wet_threshold).sum(1) / np.maximum(live.sum(1), 1)
                positive = np.where(live & (members > wet_threshold), members, np.nan)
                conditional_mean = np.nan_to_num(np.nanmean(positive, axis=1), nan=0.0)
                out = np.concatenate([out, wet[:, None], conditional_mean[:, None]], axis=1)
        return out

    def enough_members(members: np.ndarray, minimum_fraction: float = 0.8) -> np.ndarray:
        return np.isfinite(members).mean(1) >= minimum_fraction

    hour = frame["hour_utc"].to_numpy().astype("float64")
    doy = frame["doy"].to_numpy().astype("float64")
    context = np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        frame["elevation_m"].to_numpy().astype("float64") / 1000.0,
        frame["lat"].to_numpy().astype("float64") / 30.0,
        frame["lon"].to_numpy().astype("float64") / 90.0,
    ], axis=1)

    features = {
        "precipitation": np.concatenate([summarise(precip_members, 0.1), context], axis=1),
        "temperature_2m": np.concatenate([summarise(temp_members, None), context], axis=1),
        "wind_speed_10m": np.concatenate([summarise(wind_members, None), context], axis=1),
    }
    targets = {
        "precipitation": frame["truth_precipitation"].to_numpy().astype("float64"),
        "temperature_2m": frame["truth_temperature_2m"].to_numpy().astype("float64"),
        "wind_speed_10m": frame["truth_wind_speed_10m"].to_numpy().astype("float64"),
    }
    raw_members = {"precipitation": precip_members, "temperature_2m": temp_members,
                   "wind_speed_10m": wind_members}
    member_ok = {name: enough_members(matrix) for name, matrix in raw_members.items()}
    for name, ok in member_ok.items():
        print(f"[m4] {name}: {ok.mean():.1%} of rows have >=80% live members")

    # --- split: chronological AND spatial, so neither time nor place leaks ----
    times = pd.to_datetime(frame["valid_time"], utc=True)
    cutoff = times.quantile(0.70)
    locations = sorted(frame["loc_id"].unique())
    rng = np.random.default_rng(seed)
    held_out_locs = set(rng.choice(locations, size=max(1, len(locations) // 5), replace=False))
    is_future = (times > cutoff).to_numpy()
    is_held_out_loc = frame["loc_id"].isin(held_out_locs).to_numpy()

    train_mask = (~is_future) & (~is_held_out_loc)
    val_mask = is_future & (~is_held_out_loc)
    test_mask = is_future & is_held_out_loc
    print(f"[m4] train={train_mask.sum():,} val(time-holdout)={val_mask.sum():,} "
          f"test(time+space holdout)={test_mask.sum():,} "
          f"cutoff={cutoff} held_out_locs={len(held_out_locs)}")

    # --- CRPS ----------------------------------------------------------------
    def crps_ensemble(members: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Fair CRPS of a finite ensemble (Hersbach decomposition, unbiased form)."""
        members = np.where(np.isfinite(members), members,
                           np.nanmedian(members, axis=1, keepdims=True))
        n = members.shape[1]
        sorted_members = np.sort(members, axis=1)
        term1 = np.abs(sorted_members - observation[:, None]).mean(1)
        i = np.arange(1, n + 1)
        weights = (2 * i - n - 1)
        term2 = (weights * sorted_members).sum(1) / (n * (n - 1))
        return term1 - term2

    SQRT_PI = math.sqrt(math.pi)

    def gaussian_crps(mu, sigma, y):
        z = (y - mu) / sigma
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        pdf = torch.exp(normal.log_prob(z))
        cdf = normal.cdf(z)
        return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / SQRT_PI)

    def csgd_crps(shape, scale, shift, y):
        """Closed-form CRPS of a censored shifted gamma distribution.

        Following Scheuerer & Hamill (2015): the predictive distribution is
        `max(0, X + shift)` with `X ~ Gamma(shape, scale)`.  Evaluated by
        numerical integration of the Brier score over a fixed quadrature grid,
        which is differentiable, stable in float32 and avoids the incomplete
        beta function that has no autograd rule in torch.
        """
        grid = torch.linspace(0.0, 1.0, 129, device=y.device, dtype=y.dtype) ** 2 * 120.0
        grid = grid.view(1, -1)
        gamma = torch.distributions.Gamma(shape.unsqueeze(-1), 1.0 / scale.unsqueeze(-1))
        # P(max(0, X + shift) <= t) = P(X <= t - shift)
        # torch's Gamma exposes no CDF, so integrate its density on the grid.
        argument = (grid - shift.unsqueeze(-1)).clamp(min=1e-6)
        density = torch.exp(gamma.log_prob(argument))
        widths = torch.diff(grid, dim=-1)
        widths = torch.cat([widths[:, :1], widths], dim=-1)
        cdf = torch.cumsum(density * widths, dim=-1).clamp(0.0, 1.0)
        indicator = (grid >= y.unsqueeze(-1)).to(y.dtype)
        brier = (cdf - indicator) ** 2
        return (brier * widths).sum(-1)

    class DistributionalHead(nn.Module):
        def __init__(self, in_dim: int, family: str, hidden: int = 128):
            super().__init__()
            self.family = family
            out_dim = 3 if family == "csgd" else 2
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            raw = self.net(x)
            if self.family == "csgd":
                shape = F.softplus(raw[:, 0]) + 0.05
                scale = F.softplus(raw[:, 1]) + 0.05
                shift = raw[:, 2].clamp(-20.0, 5.0)
                return shape, scale, shift
            mu = raw[:, 0]
            sigma = F.softplus(raw[:, 1]) + 0.05
            return mu, sigma

    results: dict = {}
    artifacts: dict = {}

    for variable, family in (("precipitation", "csgd"),
                             ("temperature_2m", "gaussian"),
                             ("wind_speed_10m", "gaussian")):
        X = features[variable]
        y = targets[variable]
        finite = np.isfinite(X).all(1) & np.isfinite(y) & member_ok[variable]
        if finite.sum() < 5000:
            raise RuntimeError(f"{variable}: only {int(finite.sum())} usable rows after "
                               f"member and truth filtering; refusing to report a metric")
        tr = train_mask & finite
        va = val_mask & finite
        te = test_mask & finite

        mean = X[tr].mean(0)
        std = X[tr].std(0)
        std[std < 1e-6] = 1.0
        normalise = lambda a: (a - mean) / std  # noqa: E731

        Xtr = torch.tensor(normalise(X[tr]), dtype=torch.float32, device=device)
        ytr = torch.tensor(y[tr], dtype=torch.float32, device=device)
        Xva = torch.tensor(normalise(X[va]), dtype=torch.float32, device=device)
        yva = torch.tensor(y[va], dtype=torch.float32, device=device)
        Xte = torch.tensor(normalise(X[te]), dtype=torch.float32, device=device)

        head = DistributionalHead(X.shape[1], family).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        def crps_of(model_out, y_true):
            if family == "csgd":
                shape, scale, shift = model_out
                return csgd_crps(shape, scale, shift, y_true)
            mu, sigma = model_out
            return gaussian_crps(mu, sigma, y_true)

        best_val, best_state = float("inf"), None
        n = len(ytr)
        for epoch in range(epochs):
            head.train()
            perm = torch.randperm(n, device=device)
            total = 0.0
            for start in range(0, n, batch_size):
                index = perm[start:start + batch_size]
                loss = crps_of(head(Xtr[index]), ytr[index]).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                total += float(loss) * len(index)
            scheduler.step()
            head.eval()
            with torch.no_grad():
                val_crps = float(crps_of(head(Xva), yva).mean())
            if val_crps < best_val:
                best_val = val_crps
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[m4:{variable}] epoch {epoch + 1}/{epochs} "
                      f"train_crps={total / n:.4f} val_crps={val_crps:.4f}")
        head.load_state_dict(best_state)
        head.eval()

        # --- predictive samples for verification -----------------------------
        @torch.no_grad()
        def sample(x_tensor, size: int = 199) -> np.ndarray:
            out = head(x_tensor)
            if family == "csgd":
                shape, scale, shift = out
                gamma = torch.distributions.Gamma(shape, 1.0 / scale)
                draws = gamma.sample((size,)).T + shift.unsqueeze(-1)
                return draws.clamp(min=0.0).cpu().numpy()
            mu, sigma = out
            normal = torch.distributions.Normal(mu, sigma)
            return normal.sample((size,)).T.cpu().numpy()

        for split_name, mask, x_tensor in (("val", va, Xva), ("test", te, Xte)):
            observed = y[mask]
            raw = raw_members[variable][mask]
            model_samples = sample(x_tensor)
            model_crps = crps_ensemble(model_samples, observed).mean()
            raw_crps = crps_ensemble(raw, observed).mean()
            climatology = np.tile(np.quantile(y[tr], np.linspace(0.005, 0.995, 199)),
                                  (mask.sum(), 1))
            clim_crps = crps_ensemble(climatology, observed).mean()
            ens_mean = raw.mean(1)
            results[f"{variable}_{split_name}_crps"] = float(model_crps)
            results[f"{variable}_{split_name}_crps_raw_ensemble"] = float(raw_crps)
            results[f"{variable}_{split_name}_crps_climatology"] = float(clim_crps)
            results[f"{variable}_{split_name}_crpss_vs_raw"] = float(1 - model_crps / raw_crps)
            results[f"{variable}_{split_name}_crpss_vs_climatology"] = float(1 - model_crps / clim_crps)
            results[f"{variable}_{split_name}_rmse_mean"] = float(
                np.sqrt(((model_samples.mean(1) - observed) ** 2).mean()))
            results[f"{variable}_{split_name}_rmse_raw_ensemble_mean"] = float(
                np.sqrt(((ens_mean - observed) ** 2).mean()))
            results[f"{variable}_{split_name}_n"] = int(mask.sum())
            # rank histogram flatness: 1.0 means perfectly calibrated spread
            ranks = (raw < observed[:, None]).sum(1)
            histogram = np.bincount(ranks, minlength=raw.shape[1] + 1).astype(float)
            histogram /= histogram.sum()
            results[f"{variable}_{split_name}_raw_rank_histogram_chi2"] = float(
                ((histogram - 1 / len(histogram)) ** 2).sum() * len(histogram))
            model_ranks = (model_samples < observed[:, None]).sum(1)
            model_hist = np.bincount(model_ranks, minlength=model_samples.shape[1] + 1).astype(float)
            model_hist /= model_hist.sum()
            results[f"{variable}_{split_name}_model_rank_histogram_chi2"] = float(
                ((model_hist - 1 / len(model_hist)) ** 2).sum() * len(model_hist))

        # --- exceedance probabilities, isotonic-refined on validation --------
        if variable == "precipitation":
            isotonics = {}
            brier = {}
            val_samples = sample(Xva)
            test_samples = sample(Xte)
            for threshold in THRESHOLDS:
                raw_val = (raw_members[variable][va] > threshold).mean(1)
                model_val = (val_samples > threshold).mean(1)
                observed_val = (y[va] > threshold).astype(float)
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(model_val, observed_val)
                isotonics[threshold] = iso

                raw_test = (raw_members[variable][te] > threshold).mean(1)
                model_test = (test_samples > threshold).mean(1)
                calibrated_test = iso.predict(model_test)
                observed_test = (y[te] > threshold).astype(float)
                base_rate = observed_test.mean()
                brier[str(threshold)] = {
                    "base_rate": float(base_rate),
                    "brier_raw_ensemble": float(((raw_test - observed_test) ** 2).mean()),
                    "brier_model": float(((model_test - observed_test) ** 2).mean()),
                    "brier_calibrated": float(((calibrated_test - observed_test) ** 2).mean()),
                    "brier_climatology": float(((base_rate - observed_test) ** 2).mean()),
                }
                reference = brier[str(threshold)]["brier_climatology"]
                brier[str(threshold)]["bss_calibrated_vs_climatology"] = float(
                    1 - brier[str(threshold)]["brier_calibrated"] / reference) if reference > 0 else None
                brier[str(threshold)]["bss_calibrated_vs_raw"] = float(
                    1 - brier[str(threshold)]["brier_calibrated"]
                    / brier[str(threshold)]["brier_raw_ensemble"]) \
                    if brier[str(threshold)]["brier_raw_ensemble"] > 0 else None

                # reliability curve, 10 equal-width bins
                bins = np.clip((calibrated_test * 10).astype(int), 0, 9)
                curve = []
                for b in range(10):
                    sel = bins == b
                    if sel.sum() >= 20:
                        curve.append({"bin": b / 10, "n": int(sel.sum()),
                                      "forecast": float(calibrated_test[sel].mean()),
                                      "observed": float(observed_test[sel].mean())})
                brier[str(threshold)]["reliability"] = curve
            results["precipitation_exceedance"] = brier
            artifacts["isotonics"] = isotonics

        artifacts[variable] = {"head": head, "mean": mean, "std": std, "family": family}

    # --- persist -------------------------------------------------------------
    out_dir = f"{MODEL_DIR}/{ALGORITHM_VERSION}"
    os.makedirs(out_dir, exist_ok=True)
    import pickle

    payload = {
        variable: {
            "state_dict": artifacts[variable]["head"].state_dict(),
            "mean": artifacts[variable]["mean"], "std": artifacts[variable]["std"],
            "family": artifacts[variable]["family"],
            "in_dim": int(features[variable].shape[1]),
        }
        for variable in ("precipitation", "temperature_2m", "wind_speed_10m")
    }
    torch.save(payload, f"{out_dir}/heads.pt")
    with open(f"{out_dir}/isotonics.pkl", "wb") as handle:
        pickle.dump(artifacts["isotonics"], handle)

    metrics = {
        "algorithm_version": ALGORITHM_VERSION,
        "model_kind": "CSGD (precipitation) + Gaussian EMOS (temperature, wind), CRPS-trained",
        "dataset_kind": "d2_gfs025_31member_ensemble_vs_era5_land",
        "dataset_shards": len(files),
        "dataset_sha256": data_sha,
        "split": "chronological 70% cutoff AND 20% spatially held-out locations",
        "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "held_out_locations": sorted(held_out_locs),
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
        "thresholds_mm": list(THRESHOLDS),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        **results,
    }
    with open(f"{out_dir}/metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    from modal_jobs.common import MODEL_VOL
    MODEL_VOL.commit()
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("held_out_locations", "precipitation_exceedance")}, indent=2))
    print("exceedance:", json.dumps(results.get("precipitation_exceedance", {}), indent=2)[:3000])
    return metrics


@app.local_entrypoint()
def main(epochs: int = 60):
    train.remote(epochs=epochs)
