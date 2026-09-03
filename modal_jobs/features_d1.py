"""D1-specific loading, feature extraction and splitting.

The feature *definition* lives in `weathergpt_models.features` so training and
inference cannot drift apart.  This module only knows how to get the arrays out
of the training corpus and how to draw the split.

On lead time: `lead_age_days` is the age of the model run a forecast came from
(Open-Meteo's `previous_dayN`), so `lead_hours = 24 * N + hour_utc` is a real
forecast lead, not a row index.  `contracts.check_lead_time_signal` asserts that
mean error actually grows with it; if it stops growing, the column has silently
stopped meaning what it says.
"""
from __future__ import annotations

from weathergpt_models.features import (  # re-exported for the trainers
    CONTEXT_KEYS, ENSEMBLE_STATS, MODELS, VARIABLES, assemble_features,
    ensemble_summary, feature_names,
)

__all__ = ["MODELS", "VARIABLES", "ENSEMBLE_STATS", "CONTEXT_KEYS", "ensemble_summary",
           "assemble_features", "feature_names", "load_d1", "dataset_sha256",
           "build_features", "split_masks"]


def load_d1(data_dir: str, *, columns: list | None = None):
    import glob

    import pandas as pd

    files = sorted(glob.glob(f"{data_dir}/d1_mos/*.parquet"))
    if not files:
        raise RuntimeError("D1 corpus missing; run `modal run modal_jobs/build_corpora.py "
                           "--what d1` first")
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in files],
                      ignore_index=True)
    return frame, files


def dataset_sha256(files: list) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(files):
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def build_features(frame, target_variable: str):
    """-> (X, y, feature_names, member_matrix, keep_mask).

    `member_matrix` is returned alongside X so the verification code can score
    the raw ensemble on exactly the rows the model was scored on.
    """
    import numpy as np

    members = frame[[f"fc_{target_variable}_{model}" for model in MODELS]].to_numpy(dtype="float64")
    other_members = {
        other: frame[[f"fc_{other}_{model}" for model in MODELS]].to_numpy(dtype="float64")
        for other in VARIABLES if other != target_variable
    }
    context = {
        "lead_hours": frame["lead_hours"].to_numpy(dtype="float64"),
        "lead_age_days": frame["lead_age_days"].to_numpy(dtype="float64"),
        "hour_utc": frame["hour_utc"].to_numpy(dtype="float64"),
        "doy": frame["doy"].to_numpy(dtype="float64"),
        "elevation_m": frame["elevation_m"].to_numpy(dtype="float64"),
        "lat": frame["lat"].to_numpy(dtype="float64"),
        "lon": frame["lon"].to_numpy(dtype="float64"),
    }
    X = assemble_features(target_variable, members, other_members, context)
    y = frame[f"truth_{target_variable}"].to_numpy(dtype="float64")

    # A row is usable when the truth exists and at least two of the four models
    # produced a value: a one-member "ensemble" has no spread and would teach a
    # calibrator nothing about uncertainty.
    live = np.isfinite(members).sum(1)
    keep = np.isfinite(y) & (live >= 2)
    return X, y, feature_names(target_variable), members, keep


def split_masks(frame, *, seed: int = 42, time_quantile: float = 0.70,
                spatial_fraction: float = 0.2):
    """Chronological AND spatial holdout.

    `val` is future time at seen locations, which measures ordinary forecast
    degradation.  `test` is future time at locations never trained on, which is
    the only honest way to claim the model works for a district that was not in
    the corpus -- and a user asking about their village is exactly that case.
    """
    import numpy as np
    import pandas as pd

    times = pd.to_datetime(frame["valid_time"], utc=True)
    cutoff = times.quantile(time_quantile)
    locations = sorted(frame["loc_id"].unique())
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(locations, size=max(1, int(len(locations) * spatial_fraction)),
                              replace=False).tolist())

    future = (times > cutoff).to_numpy()
    unseen_place = frame["loc_id"].isin(held_out).to_numpy()
    return {
        "train": (~future) & (~unseen_place),
        "val": future & (~unseen_place),
        "test": future & unseen_place,
        "cutoff": str(cutoff),
        "held_out_locations": sorted(held_out),
    }
