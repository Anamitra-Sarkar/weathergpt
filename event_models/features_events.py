"""Feature construction, climate zones and split rules for the event models.

Shared by training and (later) serving so a feature is defined exactly once.
Inputs are the tables produced by `dataset.py`; every feature is computed from the
forecast run only -- nothing observed after the run time can reach a feature.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

STEP_BASE = ["t2m_c", "dpt2m_c", "rh2m", "wind10", "gust", "cape", "cape_ml", "cin", "lftx", "vis_km",
             "tcc", "pwat", "cwat", "mslp_hpa", "hpbl", "refc", "apcp_mm", "apcp_cell_mean",
             "apcp_cell_max", "apcp_width_h"]
CONTEXT_FIELDS = ["cape", "t2m_c", "mslp_hpa", "rh2m", "refc", "tcc", "pwat", "apcp_cell_mean"]
STATIC = ["lat", "lon", "elevation_m", "dist_coast_km", "land_frac"]
ZONES = ["himalaya_north", "northeast", "northwest_arid", "indo_gangetic", "west_coast",
         "east_coast", "south_interior", "central", "islands"]

# ----- split policy (one place, so every model is evaluated the same way) -----
TRAIN_END = pd.Timestamp("2024-03-31")
VAL_START, VAL_END = pd.Timestamp("2024-04-06"), pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-06")
HOLDOUT_FRACTION = 0.2


LEAD_BUCKET_EDGES = [0, 24, 48, 72, 120, 168, 240]
LEAD_BUCKET_LABELS = ["h000-024", "h024-048", "h048-072", "h072-120", "h120-168", "h168-240"]


def lead_bucket(series):
    """Lead-time buckets used for the per-lead skill breakdowns (and the per-lead gate at serving time)."""
    return pd.cut(series, LEAD_BUCKET_EDGES, labels=LEAD_BUCKET_LABELS)


def region_of(lat, lon, elev, dist_coast) -> np.ndarray:
    """Coarse climate zone from geography.  `rules` is in priority order: the first match wins."""
    lat, lon, elev, dist = (np.asarray(x, float) for x in (lat, lon, elev, dist_coast))
    zone = np.full(lat.shape, "central", dtype=object)       # default: peninsular interior
    rules = [
        ("islands", ((lon >= 92) & (lat < 14)) | ((lon >= 71) & (lon <= 74.5) & (lat >= 8) & (lat <= 12.5))),
        ("himalaya_north", (lat >= 30.5) | ((lat >= 27) & (elev >= 1500))),
        ("northeast", (lat >= 21.5) & (lon >= 89.5)),
        ("northwest_arid", (lat >= 23.5) & (lat < 30.5) & (lon < 75.5)),
        ("indo_gangetic", (lat >= 23.5) & (lat < 30.5) & (lon >= 75.5) & (lon < 89.5)),
        ("west_coast", (lon <= 77.5) & (dist <= 120) & (lat < 23.5)),
        ("east_coast", (lon > 77.5) & (dist <= 120) & (lat < 23.5)),
        ("south_interior", lat < 16),
    ]
    for name, mask in reversed(rules):    # assign lowest priority first so higher-priority rules overwrite
        zone[mask] = name
    return zone


def is_holdout(point_ids, fraction: float = HOLDOUT_FRACTION) -> np.ndarray:
    """Deterministic spatial hold-out by hashing the point id (stable across runs and machines)."""
    cut = int(fraction * 10_000)
    return np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % 10_000 < cut for p in point_ids])


def holdout_blocks(lat, lon, fraction: float = HOLDOUT_FRACTION, block_deg: float = 4.0, zone=None) -> np.ndarray:
    """Hold out whole 4x4 degree blocks so neighbouring grid cells cannot leak across the split.

    With `zone` given the choice is STRATIFIED: every climate zone that has at least two blocks gets about `fraction` of
    its blocks (never fewer than one) held out.  Plain hashing once left three whole zones (north-west arid, west coast,
    south interior) with no held-out places at all, so the skill gate had no evidence there and would have withheld every
    forecast in those regions.  Deterministic (md5 rank), so the split is identical across runs and machines.
    """
    ids = [f"B{int(np.floor(a / block_deg))}_{int(np.floor(b / block_deg))}" for a, b in zip(lat, lon)]
    if zone is None:
        flags = {i: bool(is_holdout([i], fraction)[0]) for i in set(ids)}
        return np.array([flags[i] for i in ids])
    by_zone: dict = {}
    for block, z in zip(ids, zone):
        by_zone.setdefault(z, set()).add(block)
    held = set()
    for z, blocks in by_zone.items():
        # ONE global rank per block (not per zone): zones that share blocks then pick the same low-rank blocks, which keeps
        # the union of held-out blocks near `fraction` instead of growing with the number of zones
        ranked = sorted(blocks, key=lambda b: int(hashlib.md5(b.encode()).hexdigest(), 16))
        if len(ranked) >= 2:
            held.update(ranked[:max(1, round(fraction * len(ranked)))])
    return np.array([b in held for b in ids])


def split_labels(run_date: pd.Series, held_out: np.ndarray) -> np.ndarray:
    """-> 'train' | 'val' | 'test_time' | 'test_space' | 'drop'.

    Held-out units never appear in train/val.  Rows in the gaps between periods are
    dropped (forecast errors are autocorrelated for days, so adjacent days leak).
    """
    d = pd.to_datetime(run_date).to_numpy()
    out = np.full(len(d), "drop", dtype=object)
    train_ok = d <= TRAIN_END.to_datetime64()
    val_ok = (d >= VAL_START.to_datetime64()) & (d <= VAL_END.to_datetime64())
    test_ok = d >= TEST_START.to_datetime64()
    out[train_ok & ~held_out] = "train"
    out[val_ok & ~held_out] = "val"
    out[test_ok & ~held_out] = "test_time"
    out[test_ok & held_out] = "test_space"
    return out


# ------------------------------------------------------------------ feature builders
def time_features(frame: pd.DataFrame, valid_col: str) -> pd.DataFrame:
    """UTC hour, LOCAL SOLAR hour (diurnal cycles follow the sun; India spans 30 deg of longitude = 2 h), day of year."""
    valid = pd.to_datetime(frame[valid_col])
    hour = valid.dt.hour + valid.dt.minute / 60.0
    doy = valid.dt.dayofyear
    out = pd.DataFrame(index=frame.index)
    out["sin_hour"], out["cos_hour"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    if "lon" in frame:
        lst = (hour + frame["lon"].to_numpy() / 15.0) % 24
        out["sin_lst"], out["cos_lst"] = np.sin(2 * np.pi * lst / 24), np.cos(2 * np.pi * lst / 24)
    out["sin_doy"], out["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return out.astype("float32")


EXT_CONTEXT_FIELDS = ["avg_crain", "avg_cape", "avg_apcp", "w700", "mfc850_e7", "rh700", "wind850", "k_index", "hlcy"]


def step_context(p1: pd.DataFrame, fields: list | None = None) -> pd.DataFrame:
    """Tendencies along the forecast run: change from the previous step and to the next step."""
    fields = [f for f in (fields or CONTEXT_FIELDS + EXT_CONTEXT_FIELDS) if f in p1.columns]
    p1 = p1.sort_values(["point_id", "run_date", "lead_h"]).reset_index(drop=True)
    grouped = p1.groupby(["point_id", "run_date"], sort=False)
    p1["dt_prev_h"] = grouped["lead_h"].diff().astype("float32")
    for field in fields:
        p1[f"d_{field}_prev"] = grouped[field].diff().astype("float32")
        p1[f"d_{field}_next"] = (-grouped[field].diff(-1)).astype("float32")
    return p1


def derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["t_dpt_spread"] = frame["t2m_c"] - frame["dpt2m_c"]
    out["gust_factor"] = frame["gust"] / (frame["wind10"] + 1.0)
    out["log_cape"] = np.log1p(frame["cape"].clip(lower=0))
    out["log_cape_ml"] = np.log1p(frame["cape_ml"].clip(lower=0))
    out["storm_idx"] = out["log_cape"] * frame["pwat"] / 50.0
    out["neg_lftx"] = -frame["lftx"]
    return out.astype("float32")


def terrain_columns(points: pd.DataFrame) -> list:
    return [c for c in points.columns if c.startswith("terrain_")]


def attach_static(frame: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    keep = points[["point_id", *STATIC, *terrain_columns(points)]].copy()
    keep["zone"] = region_of(points["lat"], points["lon"], points["elevation_m"], points["dist_coast_km"])
    keep["zone_code"] = keep["zone"].map({z: i for i, z in enumerate(ZONES)}).astype("int8")
    return frame.merge(keep, on="point_id", how="left")


# Step rows carry the DAY-level anomalies of their valid UTC day (airmass anomaly: PWAT, temperature, CAPE ...).  The
# climatology exists for every node, so serving can compute it anywhere; a step-level climatology would only exist
# where step forecasts were collected (the stations).
STEP_DAY_ANOM_COLS = ["t2m_mean", "t2m_max", "t2m_min", "dpt_mean", "rh_mean", "pwat_mean", "cape_max", "wind_mean",
                      "mslp_mean", "tcc_mean"]
DAY_CLIM_COLS = ["t2m_max", "t2m_min", "t2m_mean", "dpt_mean", "rh_mean", "pwat_mean", "cape_max", "wind_mean",
                 "mslp_mean", "tcc_mean", "rain_mm"]


def _doy_bin(dates) -> np.ndarray:
    return np.minimum((pd.to_datetime(dates).dt.dayofyear.to_numpy() - 1) // 10, 36)


def climatology_keys(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Keys of the forecast climatology: step -> (point, 3-hour slot, month); day -> (point, 10-day bin of the VALID day)."""
    keys = pd.DataFrame({"point_id": frame["point_id"].to_numpy()})
    if kind == "step":
        valid = pd.to_datetime(frame["run_date"]) + pd.to_timedelta(frame["lead_h"].astype(int), unit="h")
        keys["slot"], keys["month"] = (valid.dt.hour // 3).to_numpy(), valid.dt.month.to_numpy()
    else:
        valid = pd.to_datetime(frame["run_date"]) + pd.to_timedelta(frame["day_k"].astype(int), unit="D")
        keys["bin"] = _doy_bin(valid)
    return keys


def fit_climatology(frame: pd.DataFrame, cols: list, kind: str, until: pd.Timestamp = TRAIN_END) -> pd.DataFrame:
    """Forecast climatology per place, from FORECASTS ONLY issued up to `until` (the end of training).

    No observation enters it, so it carries no label information; restricting it to the training period keeps
    validation and test dates out of the features' reference.  Leads are pooled (a forecast's climate barely
    depends on lead).  Day-of-year bins are smoothed circularly over neighbours.
    """
    cols = [c for c in cols if c in frame.columns]
    use = pd.to_datetime(frame["run_date"]) <= until
    keys = climatology_keys(frame[use], kind)
    data = pd.concat([keys.reset_index(drop=True), frame.loc[use, cols].reset_index(drop=True)], axis=1)
    key_cols = [c for c in keys.columns]
    clim = data.groupby(key_cols)[cols].mean()
    if kind == "day":
        smoothed = []
        for c in cols:
            wide = clim[c].unstack("bin").reindex(columns=range(37))
            padded = pd.concat([wide.iloc[:, -1:], wide, wide.iloc[:, :1]], axis=1)
            sm = padded.T.rolling(3, center=True, min_periods=1).mean().T.iloc[:, 1:-1]
            sm.columns = range(37)
            smoothed.append(sm.stack(future_stack=True).rename(c))
        clim = pd.concat(smoothed, axis=1)
        clim.index = clim.index.set_names(["point_id", "bin"])   # the smoothing drops the level name; the saved artifact needs it
    return clim


def nearest_node_ids(points: pd.DataFrame) -> dict:
    """point_id -> id of the grid NODE used as its climatology reference.

    Nodes reference themselves; a station references its nearest node (never itself).  Serving does the same for an
    arbitrary latitude/longitude, so training sees exactly the reference error serving will have (a few tens of km),
    instead of a flattering self-reference that only exists for places that already had forecasts collected.
    """
    nodes = points[points["kind"] == "node"]
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    ids = nodes["point_id"].to_numpy()
    out = {}
    for pid, la, lo, kind in zip(points["point_id"], points["lat"], points["lon"], points["kind"]):
        if kind == "node":
            out[pid] = pid
        else:
            out[pid] = ids[int(np.argmin((lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2))]
    return out


def nearest_node_id(points: pd.DataFrame, lat: float, lon: float) -> str:
    nodes = points[points["kind"] == "node"]
    d = (nodes["lat"].to_numpy() - lat) ** 2 + ((nodes["lon"].to_numpy() - lon) * np.cos(np.deg2rad(lat))) ** 2
    return nodes["point_id"].to_numpy()[int(np.argmin(d))]


def apply_climatology(frame: pd.DataFrame, clim: pd.DataFrame, kind: str, ref_map: dict | None = None) -> pd.DataFrame:
    """Add `<col>_anom` = forecast value minus the reference place's typical forecast for that time of year / day.
    `ref_map` maps each row's point_id to the node whose climatology is the reference (default: itself)."""
    keys = climatology_keys(frame, kind)
    if ref_map is not None:
        keys["point_id"] = frame["point_id"].map(ref_map).to_numpy()
    key_cols = list(keys.columns)
    ref = keys.join(clim, on=key_cols)
    out = {}
    for c in clim.columns:
        if c in frame.columns:
            out[f"{c}_anom"] = (frame[c].to_numpy() - ref[c].to_numpy()).astype("float32")
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def step_anom_names() -> list:
    return [f"day_{c}_anom" for c in STEP_DAY_ANOM_COLS]


def step_feature_names(ext_cols=(), anom_cols=(), terrain_cols=()) -> list:
    ctx_fields = CONTEXT_FIELDS + [f for f in EXT_CONTEXT_FIELDS if f in ext_cols]
    ctx = ["dt_prev_h"] + [f"d_{f}_{s}" for f in ctx_fields for s in ("prev", "next")]
    derived = ["t_dpt_spread", "gust_factor", "log_cape", "log_cape_ml", "storm_idx", "neg_lftx"]
    time = ["sin_hour", "cos_hour", "sin_lst", "cos_lst", "sin_doy", "cos_doy"]
    return (STEP_BASE + list(ext_cols) + list(anom_cols) + ctx + derived + time + ["lead_h"] + STATIC
            + list(terrain_cols) + ["zone_code"])


def build_step_features(table: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """`table` = station_step_table output (forecast columns + labels); returns it with features added."""
    frame = attach_static(table, points)
    frame = pd.concat([frame, time_features(frame, "valid_time"), derived_features(frame)], axis=1)
    return frame


DAY_BASE = ["t2m_max", "t2m_min", "t2m_mean", "dpt_mean", "rh_min", "rh_mean", "wind_mean", "gust_max",
            "cape_max", "tcc_mean", "pwat_mean", "hpbl_max", "mslp_mean", "n_steps"]
RAIN_DAY_BASE = ["t2m_max", "t2m_min", "t2m_mean", "rh_mean", "rh_min", "dpt_mean", "wind_mean", "gust_max",
                 "cape_max", "cape_ml_max", "cin_mean", "lftx_min", "vis_min_km", "tcc_mean", "pwat_mean",
                 "pwat_max", "mslp_mean", "hpbl_max", "refc_max", "cwat_max", "n_steps", "rain_buckets",
                 "rain_mm", "rain_cell_mean_mm", "rain_cell_max_mm", "rain_max6_mm"]


def day_feature_names(kind: str, ext_cols=(), anom_cols=(), terrain_cols=()) -> list:
    base = DAY_BASE if kind == "station_day" else RAIN_DAY_BASE
    return base + list(ext_cols) + list(anom_cols) + ["day_k", "sin_doy", "cos_doy"] + STATIC + list(terrain_cols) + ["zone_code"]


def build_day_features(table: pd.DataFrame, points: pd.DataFrame, date_col: str) -> pd.DataFrame:
    frame = attach_static(table, points)
    frame = pd.concat([frame, time_features(frame, date_col)[["sin_doy", "cos_doy"]]], axis=1)
    return frame
