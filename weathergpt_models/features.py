"""Feature construction shared by training and inference.

This module is the contract between the two.  `modal_jobs/features_d1.py`
extracts arrays from the training corpus and calls `assemble_features`; the
inference classes in this package build the same arrays from a request and call
the identical function.  A feature defined twice is a feature that will
eventually be defined differently in the two places, and the failure is silent —
the model simply gets worse and nobody can see why.

The ensemble is represented only by *summary statistics*, never by raw members.
That is deliberate: it makes the models independent of member count and member
ordering, so a post-processor fitted on the four-member multi-model ensemble in
the training corpus can be applied to the thirty-one-member GFS ensemble at
request time through exactly this interface.
"""
from __future__ import annotations

import numpy as np

MODELS = ("gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless")
VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m")
ENSEMBLE_STATS = ("mean", "sd", "min", "max", "median", "spread_ratio", "wet_fraction")

WET_THRESHOLD_MM = 0.1
CONTEXT_KEYS = ("lead_hours", "lead_age_days", "hour_utc", "doy",
                "elevation_m", "lat", "lon")


def wet_threshold_for(variable: str) -> float | None:
    return WET_THRESHOLD_MM if variable == "precipitation" else None


def ensemble_summary(values: np.ndarray, *, wet_threshold: float | None = None) -> np.ndarray:
    """NaN-aware summary of an (n_rows, n_members) forecast matrix -> (n_rows, 7).

    Members genuinely go missing — a model does not always publish every hour —
    so every statistic is NaN-aware and the caller decides how few live members
    is too few.
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim == 1:
        values = values[None, :]
    with np.errstate(invalid="ignore", all="ignore"):
        live = np.isfinite(values)
        mean = np.nanmean(values, axis=1)
        sd = np.nanstd(values, axis=1)
        minimum = np.nanmin(values, axis=1)
        maximum = np.nanmax(values, axis=1)
        median = np.nanmedian(values, axis=1)
        spread_ratio = sd / (np.abs(mean) + 1.0)
        if wet_threshold is None:
            wet = np.zeros_like(mean)
        else:
            wet = ((np.where(live, values, -np.inf) > wet_threshold).sum(1)
                   / np.maximum(live.sum(1), 1))
    return np.stack([mean, sd, minimum, maximum, median, spread_ratio, wet], axis=1)


def feature_names(target_variable: str) -> list:
    names = [f"fc_{target_variable}_{model}" for model in MODELS]
    names += [f"ens_{stat}" for stat in ENSEMBLE_STATS]
    for other in VARIABLES:
        if other == target_variable:
            continue
        names += [f"x_{other}_mean", f"x_{other}_sd"]
    names += ["lead_hours", "lead_age_days", "sin_hour", "cos_hour", "sin_doy", "cos_doy",
              "elevation_km", "lat", "lon"]
    names += [f"missing_{model}" for model in MODELS]
    return names


def assemble_features(target_variable: str, members: np.ndarray,
                      other_members: dict, context: dict) -> np.ndarray:
    """Build the model input matrix.

    `members`        (n, len(MODELS)) forecasts of the target variable
    `other_members`  {variable: (n, len(MODELS))} for the cross-variable predictors
    `context`        arrays of length n, keyed by CONTEXT_KEYS

    Missing members are imputed with the row's live-member mean and flagged by a
    trailing indicator column per model, so the network is told a value was
    imputed rather than being quietly handed a plausible number.
    """
    members = np.asarray(members, dtype="float64")
    if members.ndim == 1:
        members = members[None, :]
    n = len(members)

    with np.errstate(invalid="ignore", all="ignore"):
        row_mean = np.nanmean(members, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    imputed = np.where(np.isfinite(members), members, row_mean[:, None])

    blocks = [imputed, ensemble_summary(members,
                                        wet_threshold=wet_threshold_for(target_variable))]

    for other in VARIABLES:
        if other == target_variable:
            continue
        matrix = other_members.get(other)
        if matrix is None:
            blocks.append(np.zeros((n, 2)))
            continue
        summary = ensemble_summary(matrix, wet_threshold=wet_threshold_for(other))
        blocks.append(summary[:, [0, 1]])

    hour = np.asarray(context["hour_utc"], dtype="float64").reshape(n)
    doy = np.asarray(context["doy"], dtype="float64").reshape(n)
    blocks.append(np.stack([
        np.asarray(context["lead_hours"], dtype="float64").reshape(n),
        np.asarray(context["lead_age_days"], dtype="float64").reshape(n),
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        np.asarray(context["elevation_m"], dtype="float64").reshape(n) / 1000.0,
        np.asarray(context["lat"], dtype="float64").reshape(n),
        np.asarray(context["lon"], dtype="float64").reshape(n),
    ], axis=1))

    blocks.append((~np.isfinite(members)).astype("float64"))

    matrix = np.concatenate(blocks, axis=1)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def members_from_mapping(values: dict) -> np.ndarray:
    """{model_name: value} -> a (1, len(MODELS)) row, missing models as NaN."""
    return np.array([[float(values.get(model, np.nan)) if values.get(model) is not None
                      else np.nan for model in MODELS]], dtype="float64")
