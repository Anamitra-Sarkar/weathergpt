"""Join GFS forecasts (collect_gfs) with observed truth (collect_truth) into training tables.

Three tables, one per kind of target:

  station_step_table   (station, run, lead)  hourly-step events + continuous truth, METAR
  station_day_table    (station, run, IST day) Tmax/Tmin + hot-day / cold-night, METAR
  point_day_rain_table (point, run, day_k)   areal rain truth from CHIRPS, whole India

Leakage rules that live HERE so every model inherits them:
  * features only ever come from the forecast run (issued at 00Z of `run_date`);
    nothing observed after the run is a feature;
  * labels are windows around the forecast valid time, never around the run time;
  * a label whose window has no observation is MISSING (row dropped), never "no event".
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

IST = pd.Timedelta(hours=5, minutes=30)
EVENT_FLAGS = ("ts_any", "ts_at", "fog", "dense_fog", "strong_wind", "gale", "rain_any",
               "rain_heavy", "dust", "squall")
# mm/day thresholds for the rain exceedance curve.  They include every IMD class boundary
# (2.5 rainy day, 15.6 moderate, 64.5 heavy, 115.6 very heavy, 204.5 extremely heavy) plus finer
# steps so conditional amounts can be interpolated between them.
RAIN_THRESHOLDS = (0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.6, 25.0, 35.5, 50.0, 64.5, 90.0, 115.6, 150.0, 204.5)
UNION_THRESHOLDS = (2.5, 15.6)      # "at least one such day in the window" labels
UNION_DAYS = 3
# Multi-day windows.  "any" = share of the cell's pixels that reached the daily threshold on AT LEAST ONE day;
# "sum" = share whose window TOTAL reached the threshold (a weekly-rainfall exceedance curve).
WINDOW_SPECS = {
    3: {"union": (2.5, 15.6), "sum": (2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0)},
    7: {"union": (2.5, 15.6), "sum": (5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0)},
}
CHIRPS_THRESHOLDS = RAIN_THRESHOLDS  # backwards-compatible alias


def thr_col(t: float, prefix: str = "frac_ge_") -> str:
    return prefix + f"{t:g}".replace(".", "p")


def find_dir(root: str, marker: str) -> Path:
    """Locate a kernel-output folder under /kaggle/input whatever the mount nesting is."""
    hits = glob.glob(f"{root}/**/{marker}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{marker} not found under {root}")
    return Path(hits[0]).parent


def read_all(root: str, pattern: str, dedupe_on: list | None = None, columns: list | None = None) -> pd.DataFrame:
    """Concatenate every file matching `pattern` anywhere under `root` (all shard kernels' outputs).
    `columns` projects at read time -- the point-day tables are wide, and most callers need a handful of columns."""
    files = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not files:
        raise FileNotFoundError(f"no files match {pattern} under {root}")
    frame = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    return frame.drop_duplicates(subset=dedupe_on) if dedupe_on else frame


def read_thinned(root: str, pattern: str, dedupe_on: list, keep_dates=None, columns: list | None = None) -> pd.DataFrame:
    """Like read_all, but memory-lean for the huge point-day tables: each file is cut to `keep_dates` (run dates) and its
    float64 columns are stored as float32 BEFORE it joins the pile, so the full-width table never exists in memory."""
    files = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not files:
        raise FileNotFoundError(f"no files match {pattern} under {root}")
    parts = []
    for f in files:
        part = pd.read_parquet(f, columns=columns)
        if keep_dates is not None:
            part = part[pd.to_datetime(part["run_date"]).isin(keep_dates)].copy()
        wide = [c for c in part.columns if part[c].dtype == np.float64 and c not in dedupe_on]
        part[wide] = part[wide].astype(np.float32)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=dedupe_on)


def run_dates(root: str, pattern: str) -> set:
    """Every run date present in the files matching `pattern` (reads one column)."""
    files = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return set(pd.to_datetime(pd.concat([pd.read_parquet(f, columns=["run_date"])["run_date"] for f in files])).unique())


def read_parts(folder: Path, pattern: str) -> pd.DataFrame:
    parts = sorted(folder.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"no files match {pattern} in {folder}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


# ----------------------------------------------------------------- METAR side
def load_metar_hourly(truth_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((truth_dir / "metar_hourly").glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["point_id"] = "S:" + path.stem
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True).dt.tz_localize(None)
    return out


def window_labels(hourly: pd.DataFrame, half_width: int = 1) -> pd.DataFrame:
    """Per station: label = OR of the flag over hours [T-h, T+h]; `cover` = hours observed in it.

    Operates on the full hourly grid so a gap in reporting shrinks `cover` instead
    of silently shifting the window.
    """
    flags = [c for c in EVENT_FLAGS if c in hourly.columns]
    out = []
    for point_id, g in hourly.groupby("point_id", sort=False):
        g = g.set_index("hour_utc").sort_index()
        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        observed = full["n_obs"].fillna(0) > 0
        size = 2 * half_width + 1
        res = pd.DataFrame(index=full.index)
        res["cover"] = observed.astype(int).rolling(size, center=True, min_periods=1).sum()
        for flag in flags:
            res[f"w_{flag}"] = (full[flag].astype("float64").fillna(0.0)
                                .rolling(size, center=True, min_periods=1).max())
        # exact-hour continuous truth
        for col in ("temp_c", "rh", "wind_ms", "peak_wind_ms", "vis_km"):
            if col in full:
                res[f"obs_{col}"] = full[col]   # prefixed: forecast tables have their own vis_km etc.
        res["n_obs_hour"] = full["n_obs"].fillna(0)
        res["point_id"] = point_id
        out.append(res.rename_axis("valid_hour").reset_index())
    return pd.concat(out, ignore_index=True)


def station_step_table(p1: pd.DataFrame, hourly: pd.DataFrame, half_width: int = 1,
                       min_cover: int = 2) -> pd.DataFrame:
    """Forecast step rows + METAR labels at the valid time.  Rows without enough observed hours are dropped."""
    labels = window_labels(hourly, half_width)
    p1 = p1.copy()
    p1["valid_time"] = pd.to_datetime(p1["run_date"]) + pd.to_timedelta(p1["lead_h"].astype(int), unit="h")
    p1["valid_hour"] = p1["valid_time"].dt.floor("h")
    clash = (set(p1.columns) & set(labels.columns)) - {"point_id", "valid_hour"}
    if clash:   # pandas would silently rename both to _x/_y and a feature would vanish
        raise ValueError(f"forecast/observation column collision: {sorted(clash)}")
    merged = p1.merge(labels, on=["point_id", "valid_hour"], how="inner")
    merged = merged[merged["cover"] >= min_cover].copy()
    return merged


# ----------------------------------------------------------- daily temperature
# extension columns aggregated over the IST day, named like the point-day table's (<col>_mean / _max / _min)
IST_EXT_MEAN = ["wind850", "wind200", "shear_850_200", "rh850", "rh700", "rh500", "t850_c", "z500", "lapse_850_500",
                "k_index", "total_totals", "vort850_e5", "div850_e5", "mfc850_e7", "upslope10", "pgrad_hpa100km",
                "tskin_c", "q2m_gkg", "soilw", "dswrf", "lcc", "mcc", "hcc", "avg_t2m_c", "spr_t2m_c", "avg_rh2m",
                "spr_rh2m", "avg_pwat", "spr_pwat", "avg_tcc", "avg_wind10", "t2m_nb9_std", "wdir10_sin", "wdir10_cos"]
IST_EXT_MAX = ["wind850", "k_index", "total_totals", "mfc850_e7", "upslope10", "pgrad_hpa100km", "cape255", "cape90",
               "hlcy", "prate_mmh", "tskin_c", "avg_cape", "spr_cape", "avg_cape_ml", "avg_crain", "avg_gust", "spr_gust",
               "cape_nb5_max", "refc_nb5_max", "gust_nb5_max"]
IST_EXT_MIN = ["w850", "w700", "w500", "rh700", "cin180", "lftx4"]


def ist_day_aggregates(p1: pd.DataFrame) -> pd.DataFrame:
    """Forecast steps -> one row per (station, run, IST day), with the extension predictors when present.

    Ensemble Tmax / Tmin use the GEFS 6-hour window max / min fields (windows ending 00, 06, 12, 18 UTC, which tile an
    IST day to within 30 minutes): daily max of the window maxima, and the mean window spread.
    """
    p1 = p1.copy()
    valid = pd.to_datetime(p1["run_date"]) + pd.to_timedelta(p1["lead_h"].astype(int), unit="h")
    p1["ist_day"] = (valid + IST).dt.floor("D")
    p1["valid_hour_utc"] = valid.dt.hour
    keys = ["point_id", "run_date", "ist_day"]
    grouped = p1.groupby(keys, sort=False)
    agg = grouped.agg(
        n_steps=("lead_h", "size"),
        t2m_max=("t2m_c", "max"), t2m_min=("t2m_c", "min"), t2m_mean=("t2m_c", "mean"),
        dpt_mean=("dpt2m_c", "mean"), rh_min=("rh2m", "min"), rh_mean=("rh2m", "mean"),
        wind_mean=("wind10", "mean"), gust_max=("gust", "max"), cape_max=("cape", "max"),
        tcc_mean=("tcc", "mean"), pwat_mean=("pwat", "mean"), hpbl_max=("hpbl", "max"),
        mslp_mean=("mslp_hpa", "mean"), lead_min=("lead_h", "min"), lead_max=("lead_h", "max"),
    ).reset_index()
    ext_spec = {}
    for names, fn, suffix in ((IST_EXT_MEAN, "mean", "mean"), (IST_EXT_MAX, "max", "max"), (IST_EXT_MIN, "min", "min")):
        for c in names:
            if c in p1.columns:
                ext_spec[f"{c}_{suffix}"] = (c, fn)
    if "rh700" in p1.columns:
        ext_spec["rh700_min"] = ("rh700", "min")
    if ext_spec:
        agg = agg.merge(grouped.agg(**ext_spec).reset_index(), on=keys, how="left")
    for col, label, fn in (("avg_tmax2m_c", "tmax", "max"), ("avg_tmin2m_c", "tmin", "min")):
        width_col = f"avg_{label}2m_width_h"
        if col in p1.columns and width_col in p1.columns:
            win = p1[(p1[width_col] == 6) & (p1["lead_h"] % 6 == 0)]
            parts = {f"gefs_{label}_mean_c": (col, fn)}
            if f"spr_{label}2m_c" in p1.columns:
                parts[f"gefs_{label}_spread_c"] = (f"spr_{label}2m_c", "mean")
            agg = agg.merge(win.groupby(keys, sort=False).agg(**parts).reset_index(), on=keys, how="left")
    agg["day_k"] = (agg["ist_day"] - pd.to_datetime(agg["run_date"])).dt.days
    return agg


def filter_complete_ist_days(table: pd.DataFrame) -> pd.DataFrame:
    """Forecast days 1..9 with enough steps to define an IST-day extreme (8 steps/day to 72 h, 4 steps beyond)."""
    k = table["day_k"]
    return table[(k >= 1) & (k <= 9) & (table["n_steps"] >= np.where(k <= 2, 7, 3))].copy()


def station_day_table(p1: pd.DataFrame, daily_truth: pd.DataFrame) -> pd.DataFrame:
    agg = ist_day_aggregates(p1)
    truth = daily_truth.copy()
    truth["ist_day"] = pd.to_datetime(truth["ist_day"])
    return agg.merge(truth, on=["point_id", "ist_day"], how="inner")


def load_metar_daily(truth_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((truth_dir / "metar_daily").glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["point_id"] = "S:" + path.stem
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- CHIRPS truth
def chirps_cell_truth(chirps_dir: Path, points: pd.DataFrame, half_cell_deg: float = 0.25) -> pd.DataFrame:
    """Areal rain truth for each point's cell (+-half_cell_deg box = 10x10 CHIRPS pixels).

    Per (point, UTC day): cell mean, cell max, and the FRACTION of the cell's pixels at or above
    each threshold.  That fraction is the probability that a random location in the cell sees
    that much rain -- the quantity a "will it rain where I am" model should be calibrated to.
    """
    grid = json.loads((chirps_dir / "grid.json").read_text())
    step, top, left = grid["step"], grid["lat_north"], grid["lon_west"]
    half = int(round(half_cell_deg / step))
    rows = np.round((top - points["lat"].to_numpy()) / step).astype(int)
    cols = np.round((points["lon"].to_numpy() - left) / step).astype(int)
    out = []
    for path in sorted(chirps_dir.glob("chirps_india_*.npz")):
        blob = np.load(path)
        data, dates = blob["data"].astype("float32"), pd.to_datetime(blob["dates"])
        n_days, n_rows, n_cols = data.shape
        for i, point_id in enumerate(points["point_id"].to_numpy()):
            r0, r1 = max(rows[i] - half, 0), min(rows[i] + half, n_rows)
            c0, c1 = max(cols[i] - half, 0), min(cols[i] + half, n_cols)
            if r1 <= r0 or c1 <= c0:
                continue
            cell = data[:, r0:r1, c0:c1].reshape(n_days, -1)
            valid = np.isfinite(cell)
            n_valid = valid.sum(axis=1)
            if n_valid.max() < 0.5 * cell.shape[1]:   # mostly ocean / outside CHIRPS coverage
                continue
            filled = np.where(valid, cell, 0.0)
            denom = np.maximum(n_valid, 1)
            row = {"point_id": point_id, "valid_date": dates, "cell_valid_px": n_valid,
                   "chirps_mean_mm": np.where(n_valid > 0, filled.sum(axis=1) / denom, np.nan),
                   "chirps_max_mm": np.where(n_valid > 0, np.where(valid, cell, -1).max(axis=1), np.nan)}
            for t in RAIN_THRESHOLDS:
                row[thr_col(t)] = np.where(n_valid > 0, ((cell >= t) & valid).sum(axis=1) / denom, np.nan)
            # Multi-day windows at pixel level.  Windows that would run past the end of this year's array (or of the
            # data) are left NaN rather than guessed.  The per-pixel validity mask is constant across days.
            px_valid = valid[0]
            for w, spec in WINDOW_SPECS.items():
                if n_days < w:
                    continue
                m = n_days - w + 1
                denom_w = max(int(px_valid.sum()), 1)
                for t in spec["union"]:
                    hit = (cell >= t) & valid
                    union = np.logical_or.reduce([hit[i:i + m] for i in range(w)])
                    col = np.full(n_days, np.nan)
                    col[:m] = union.sum(axis=1) / denom_w if px_valid.any() else np.nan
                    row[thr_col(t, f"frac_any{w}_ge_")] = col
                csum = np.concatenate([np.zeros((1, cell.shape[1])), np.cumsum(filled, axis=0)], axis=0)
                totals = csum[w:w + m] - csum[:m]                         # (m, pixels): window totals
                for t in spec["sum"]:
                    col = np.full(n_days, np.nan)
                    col[:m] = ((totals >= t) & px_valid).sum(axis=1) / denom_w if px_valid.any() else np.nan
                    row[thr_col(t, f"frac_sum{w}_ge_")] = col
            frame = pd.DataFrame(row)
            wide = frame.select_dtypes("float64").columns          # 10M+ rows x ~50 label columns: float32 is ample for fractions / mm
            frame[wide] = frame[wide].astype("float32")
            out.append(frame)
    return pd.concat(out, ignore_index=True)


def point_day_rain_table(p2: pd.DataFrame, truth: pd.DataFrame, min_valid_frac: float = 0.5, inplace: bool = False) -> pd.DataFrame:
    """`inplace=True` lets a caller that no longer needs `p2` skip the copy (the table is millions of rows wide)."""
    p2 = p2 if inplace else p2.copy()
    p2["valid_date"] = pd.to_datetime(p2["run_date"]) + pd.to_timedelta(p2["day_k"].astype(int), unit="D")
    merged = p2.merge(truth, on=["point_id", "valid_date"], how="inner")
    merged = merged[merged["cell_valid_px"] >= min_valid_frac * 100].copy()
    return merged


# ------------------------------------------------------- multi-day window table
WINDOW_DAY_FEATURES = ["rain_cell_mean_mm", "rain_cell_max_mm", "rain_mm", "pwat_mean", "cape_max",
                       "refc_max", "tcc_mean", "wind_mean", "t2m_max"]
# when the extension tables are present these per-day columns are added too (ensemble rain, neighbourhood PoP, dynamics)
WINDOW_DAY_EXT = ["gefs_rain_mean_mm", "gefs_rain_spread_mm", "rain_nb9_mean_mm", "rain_frac1_nb9", "mfc850_e7_max",
                  "w700_min", "avg_pwat_mean", "avg_crain_max", "k_index_max"]


def window_day_columns(available) -> list:
    return WINDOW_DAY_FEATURES + [c for c in WINDOW_DAY_EXT if c in available]


def window_feature_names(window: int = UNION_DAYS, day_cols: list | None = None) -> list:
    day_cols = day_cols or WINDOW_DAY_FEATURES
    names = [f"d{j}_{c}" for j in range(window) for c in day_cols]
    return names + [f"sum{window}_rain_cell_mean_mm", f"max{window}_rain_cell_max_mm", f"min{window}_rain_buckets"]


def window_feature_table(p2: pd.DataFrame, window: int = UNION_DAYS, day_cols: list | None = None) -> pd.DataFrame:
    """One row per (point, run, start day k0): features of days k0..k0+window-1 side by side.  NO truth here, so
    training and serving build window features through exactly the same code.

    "Will it rain this week?" is not the product of seven daily probabilities: consecutive days share a weather
    system, so the union (and the total) has to be learned.  Windows must fit inside the forecast days available;
    windows with an incomplete day (fewer than 4 six-hour rain buckets) are dropped.
    """
    day_cols = day_cols or WINDOW_DAY_FEATURES
    keys = ["point_id", "run_date"]
    complete = p2[p2["rain_buckets"] >= 4]
    max_k = int(complete["day_k"].max())
    if max_k < window - 1:
        raise ValueError(f"{window}-day windows need forecast days up to {window - 1}; the data only reaches day {max_k} "
                         f"(collect the long-lead set: GFS_LEADSET=long/all)")
    frames = []
    for k0 in range(0, max_k - window + 2):
        base = None
        for j in range(window):
            day = complete[complete["day_k"] == k0 + j][keys + day_cols + ["rain_buckets"]].copy()
            day = day.rename(columns={c: f"d{j}_{c}" for c in day_cols + ["rain_buckets"]})
            base = day if base is None else base.merge(day, on=keys, how="inner")
        base["day_k"] = k0
        frames.append(base)
    table = pd.concat(frames, ignore_index=True)
    table[f"sum{window}_rain_cell_mean_mm"] = sum(table[f"d{j}_rain_cell_mean_mm"] for j in range(window))
    table[f"max{window}_rain_cell_max_mm"] = table[[f"d{j}_rain_cell_max_mm" for j in range(window)]].max(axis=1)
    table[f"min{window}_rain_buckets"] = table[[f"d{j}_rain_buckets" for j in range(window)]].min(axis=1)
    table["valid_date"] = pd.to_datetime(table["run_date"]) + pd.to_timedelta(table["day_k"].astype(int), unit="D")
    return table


def rain_window_table(p2: pd.DataFrame, truth: pd.DataFrame, window: int = UNION_DAYS, day_cols: list | None = None,
                      min_valid_frac: float = 0.5) -> pd.DataFrame:
    """Window features (shared with serving) joined to the pixel-level union / total truth."""
    merged = window_feature_table(p2, window, day_cols).merge(truth, on=["point_id", "valid_date"], how="inner")
    return merged[merged["cell_valid_px"] >= min_valid_frac * 100].copy()


# ------------------------------------------------------------- extension tables
def merge_ext(base_table: pd.DataFrame, ext_table: pd.DataFrame, keys: list, how: str = "left") -> pd.DataFrame:
    """Join the extension predictors onto the base forecast table, refusing silent column collisions.
    how="inner" keeps only rows that HAVE extension features (no row trains with half its inputs missing)."""
    clash = (set(base_table.columns) & set(ext_table.columns)) - set(keys)
    if clash:
        raise ValueError(f"base/extension column collision: {sorted(clash)[:10]}")
    return base_table.merge(ext_table, on=keys, how=how)


def ext_columns(ext_table: pd.DataFrame, keys: list) -> list:
    return [c for c in ext_table.columns if c not in keys]


# ------------------------------------------------- IMD heat-wave / cold-wave labels
NORMAL_YEARS = (2016, 2023)      # normals use METAR strictly BEFORE the validation (2024) and test (2025+) periods
HILL_STATION_M = 1000.0


def station_normals(daily_truth: pd.DataFrame, years=NORMAL_YEARS, half_window: int = 7, min_years: int = 5) -> pd.DataFrame:
    """Per station and day-of-year Tmax / Tmin normals from the early METAR years, smoothed +-`half_window` days.

    A day-of-year with fewer than `min_years` years of data behind it is left NaN, and the label built on it is
    then NaN too (a heat wave cannot be declared without a normal).  A station that only began reporting in 2021
    therefore gets no normal rather than a noisy one.
    """
    d = daily_truth.copy()
    d["ist_day"] = pd.to_datetime(d["ist_day"])
    d = d[(d["ist_day"].dt.year >= years[0]) & (d["ist_day"].dt.year <= years[1])]
    d["doy"] = d["ist_day"].dt.dayofyear.clip(upper=365)
    out = []
    for point_id, g in d.groupby("point_id", sort=False):
        raw = g.groupby("doy")[["tmax_c", "tmin_c"]].agg(["mean", "count"])
        mean = pd.DataFrame({"tmax": raw[("tmax_c", "mean")], "tmin": raw[("tmin_c", "mean")]}).reindex(range(1, 366))
        count = raw[("tmax_c", "count")].reindex(range(1, 366)).fillna(0)
        ext = pd.concat([mean.iloc[-half_window:], mean, mean.iloc[:half_window]])        # circular smoothing
        cnt = pd.concat([count.iloc[-half_window:], count, count.iloc[:half_window]])
        smooth = ext.rolling(2 * half_window + 1, center=True, min_periods=3).mean().iloc[half_window:-half_window]
        smooth["n_years_window"] = cnt.rolling(2 * half_window + 1, center=True, min_periods=1).sum().iloc[half_window:-half_window]
        smooth = smooth.rename(columns={"tmax": "tmax_norm", "tmin": "tmin_norm"})
        needed = min_years * (2 * half_window + 1) * 0.7        # ~70% coverage of the window in each of min_years years
        smooth.loc[smooth["n_years_window"] < needed, ["tmax_norm", "tmin_norm"]] = np.nan
        smooth["point_id"], smooth["doy"] = point_id, smooth.index
        out.append(smooth.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def imd_wave_labels(daily: pd.DataFrame, normals: pd.DataFrame, station_elev: pd.Series) -> pd.DataFrame:
    """Add tmax_norm / tmin_norm, departures and IMD-style heat-wave / cold-wave flags to station-day rows.

    Heat wave  plains: Tmax >= 40 C and departure >= +4.5 C, or Tmax >= 45 C regardless;
               hill stations (>= 1000 m): Tmax >= 30 C and departure >= +4.5 C.   Severe: departure >= +6.5 C.
    Cold wave  plains: Tmin <= 10 C and departure <= -4.5 C, or Tmin <= 4 C regardless;
               hill stations: Tmin <= 0 C and departure <= -4.5 C.                Severe: departure <= -6.5 C.
    Rows with no normal get NaN labels (not False).
    """
    d = daily.copy()
    d["ist_day"] = pd.to_datetime(d["ist_day"])
    d["doy"] = d["ist_day"].dt.dayofyear.clip(upper=365)
    d = d.merge(normals[["point_id", "doy", "tmax_norm", "tmin_norm"]], on=["point_id", "doy"], how="left")
    hill = d["point_id"].map(station_elev).fillna(0).to_numpy() >= HILL_STATION_M
    d["tmax_dep"], d["tmin_dep"] = d["tmax_c"] - d["tmax_norm"], d["tmin_c"] - d["tmin_norm"]
    heat = np.where(hill, (d["tmax_c"] >= 30) & (d["tmax_dep"] >= 4.5),
                    ((d["tmax_c"] >= 40) & (d["tmax_dep"] >= 4.5)) | (d["tmax_c"] >= 45))
    cold = np.where(hill, (d["tmin_c"] <= 0) & (d["tmin_dep"] <= -4.5),
                    ((d["tmin_c"] <= 10) & (d["tmin_dep"] <= -4.5)) | (d["tmin_c"] <= 4))
    severe_heat = np.where(hill, (d["tmax_c"] >= 30) & (d["tmax_dep"] >= 6.5),
                           ((d["tmax_c"] >= 40) & (d["tmax_dep"] >= 6.5)) | (d["tmax_c"] >= 47))
    has_norm_hot, has_norm_cold = d["tmax_norm"].notna(), d["tmin_norm"].notna()
    # an absolute-threshold hit does not need a normal; everything else does
    d["heatwave"] = np.where(has_norm_hot | (d["tmax_c"] >= 45), heat, np.nan).astype(float)
    d["severe_heatwave"] = np.where(has_norm_hot | (d["tmax_c"] >= 47), severe_heat, np.nan).astype(float)
    d["coldwave"] = np.where(has_norm_cold | (d["tmin_c"] <= 4), cold, np.nan).astype(float)
    return d
