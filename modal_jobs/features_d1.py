"""Shared feature construction over the D1 multi-model corpus.

One module, used by three trainers (M2 MOS, M4 calibration, M5 trust ranker), so
the feature definition and — more importantly — the split rule exist in exactly
one place.  A split defined twice is a split that will eventually disagree with
itself and leak.

What a row is: one (location, valid_time, lead_age) triple.  At that triple we
hold four independent NWP models' forecasts, which together form a genuine
four-member multi-model ensemble, plus the ERA5 truth that verifies them.

On lead time: `lead_age_days` is the age of the model run the forecast came
from (Open-Meteo's `previous_dayN`), so `lead_hours = 24 * N + hour_utc` is a
real forecast lead, not a row index.  The contract in `contracts.py` asserts
that mean error actually grows with it; if it ever stops growing, the column has
silently stopped meaning what it says.
"""
from __future__ import annotations

MODELS = ("gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless")
VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m")

# Ensemble summary statistics are the interface between training and serving.
# The corpus gives four multi-model members; at request time the live GFS
# ensemble gives thirty-one.  Because the model consumes *summary statistics*
# and never the members themselves, the same post-processor applies to both —
# and the model card records that as an explicit transfer assumption.
ENSEMBLE_STATS = ("mean", "sd", "min", "max", "median", "spread_ratio", "wet_fraction")


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
    for path in files:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def ensemble_summary(values, *, wet_threshold: float | None = None):
    """NaN-aware summary of a (n_rows, n_members) forecast matrix."""
    import numpy as np

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
            wet = (np.where(live, values, -np.inf) > wet_threshold).sum(1) / np.maximum(live.sum(1), 1)
    return np.stack([mean, sd, minimum, maximum, median, spread_ratio, wet], axis=1)


def build_features(frame, target_variable: str):
    """-> (X, y, feature_names, member_matrix, keep_mask).

    `member_matrix` is returned alongside X so the verification code can score
    the raw ensemble on exactly the rows the model was scored on.
    """
    import numpy as np

    wet_threshold = 0.1 if target_variable == "precipitation" else None
    member_columns = [f"fc_{target_variable}_{model}" for model in MODELS]
    members = frame[member_columns].to_numpy(dtype="float64")

    blocks = [members]
    names = list(member_columns)

    blocks.append(ensemble_summary(members, wet_threshold=wet_threshold))
    names += [f"ens_{stat}" for stat in ENSEMBLE_STATS]

    # Cross-variable predictors: humidity and wind carry real information about
    # a precipitation error, and a raw MOS that ignores them leaves skill behind.
    for other in VARIABLES:
        if other == target_variable:
            continue
        other_columns = [f"fc_{other}_{model}" for model in MODELS]
        other_members = frame[other_columns].to_numpy(dtype="float64")
        summary = ensemble_summary(other_members,
                                   wet_threshold=0.1 if other == "precipitation" else None)
        blocks.append(summary[:, [0, 1]])
        names += [f"x_{other}_mean", f"x_{other}_sd"]

    lead_hours = frame["lead_hours"].to_numpy(dtype="float64")
    hour = frame["hour_utc"].to_numpy(dtype="float64")
    doy = frame["doy"].to_numpy(dtype="float64")
    context = np.stack([
        lead_hours,
        frame["lead_age_days"].to_numpy(dtype="float64"),
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        frame["elevation_m"].to_numpy(dtype="float64") / 1000.0,
        frame["lat"].to_numpy(dtype="float64"),
        frame["lon"].to_numpy(dtype="float64"),
    ], axis=1)
    blocks.append(context)
    names += ["lead_hours", "lead_age_days", "sin_hour", "cos_hour", "sin_doy", "cos_doy",
              "elevation_km", "lat", "lon"]

    X = np.concatenate(blocks, axis=1)
    y = frame[f"truth_{target_variable}"].to_numpy(dtype="float64")

    # A row is usable when the truth exists, the context is finite, and at least
    # two of the four models actually produced a value — a one-member "ensemble"
    # has no spread and would teach the calibrator nothing.
    live_members = np.isfinite(members).sum(1)
    keep = np.isfinite(y) & np.isfinite(context).all(1) & (live_members >= 2)
    return X, y, names, members, keep


def split_masks(frame, *, seed: int = 42, time_quantile: float = 0.70,
                spatial_fraction: float = 0.2):
    """Chronological AND spatial holdout.

    `val` is future time at seen locations, so it measures ordinary forecast
    degradation.  `test` is future time at locations never trained on, which is
    the only honest way to claim the model works for a district that was not in
    the corpus — and that is exactly what a user asking about their village is.
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
