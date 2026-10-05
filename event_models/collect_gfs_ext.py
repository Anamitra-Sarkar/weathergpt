"""Extended predictors for the event models: upper-air dynamics, surface energy/moisture, multi-scale
neighbourhoods, terrain, and the GEFS ensemble mean/spread.  Same runs, same points, same keys as
`collect_gfs` -- written to separate files and joined on (point_id, run_date, lead_h | day_k).

Why these (each is a physical reason, not a kitchen sink):
  * 850/700/500/200 hPa winds, omega, vorticity, moisture-flux convergence, K-index, total totals,
    lapse rate, shear, upslope flow      -> monsoon / low-pressure / orographic rain, convective potential
  * skin temperature, soil moisture, radiation, fluxes, cloud layers, helicity, mixed-layer CAPE
                                          -> heat, fog, storm organisation
  * neighbourhood statistics (5x5, 9x9 nodes = 0.5-1 deg) and neighbourhood exceedance fractions
                                          -> the standard remedy for NWP placement/timing error
  * terrain complexity, slopes            -> valley / ridge / windward effects the 0.25 deg orography hides
  * GEFS ensemble mean and spread (APCP, CRAIN = share of members raining, CAPE, PWAT, T2m, 6 h Tmax/Tmin,
    gust, helicity ...)                   -> the single best predictor of forecast uncertainty

Output under /kaggle/working/gfs_ext/ :
  points_ext.parquet                 static terrain features per point
  p1x_station_steps_{YYYYMM}.parquet (station, run, lead) extra columns
  p2x_point_days_{YYYYMM}.parquet    (point, run, day_k) extra columns
  ext_report.json                    including which requested fields were absent from the archive
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

from event_models import collect_gfs as base

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices are expected and handled

OUT = Path("/kaggle/working/gfs_ext") if Path("/kaggle").exists() else Path("gfs_ext_out")
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"
KIND_PREFIX = {"geavg": "avg_", "gespr": "spr_"}

# ---------------------------------------------------------------- what to fetch
GFS_EXT = {
    # pressure levels
    "u850": ("UGRD", "850 mb"), "v850": ("VGRD", "850 mb"), "u700": ("UGRD", "700 mb"), "v700": ("VGRD", "700 mb"),
    "u500": ("UGRD", "500 mb"), "v500": ("VGRD", "500 mb"), "u200": ("UGRD", "200 mb"), "v200": ("VGRD", "200 mb"),
    "rh850": ("RH", "850 mb"), "rh700": ("RH", "700 mb"), "rh500": ("RH", "500 mb"),
    "t850": ("TMP", "850 mb"), "t700": ("TMP", "700 mb"), "t500": ("TMP", "500 mb"),
    "w850": ("VVEL", "850 mb"), "w700": ("VVEL", "700 mb"), "w500": ("VVEL", "500 mb"),
    "z850": ("HGT", "850 mb"), "z500": ("HGT", "500 mb"), "absv850": ("ABSV", "850 mb"), "absv500": ("ABSV", "500 mb"),
    "q850": ("SPFH", "850 mb"),
    # surface energy / moisture / cloud / storm structure
    "tskin": ("TMP", "surface"), "q2m": ("SPFH", "2 m above ground"), "soilw": ("SOILW", "0-0.1 m below ground"),
    "dswrf": ("DSWRF", "surface"), "lhtfl": ("LHTFL", "surface"), "shtfl": ("SHTFL", "surface"),
    "prate": ("PRATE", "surface"), "lcc": ("LCDC", "low cloud layer"), "mcc": ("MCDC", "middle cloud layer"),
    "hcc": ("HCDC", "high cloud layer"), "cape255": ("CAPE", "255-0 mb above ground"),
    "cape90": ("CAPE", "90-0 mb above ground"), "cin180": ("CIN", "180-0 mb above ground"),
    "hlcy": ("HLCY", "3000-0 m above ground"), "lftx4": ("4LFTX", "surface"),
    # re-fetched base fields needed for neighbourhoods / gradients / direction
    "apcp": ("APCP", "surface"), "cape": ("CAPE", "surface"), "refc": ("REFC", "entire atmosphere"),
    "gust": ("GUST", "surface"), "pwat": ("PWAT", "entire atmosphere (considered as a single layer)"),
    "t2m": ("TMP", "2 m above ground"), "u10": ("UGRD", "10 m above ground"), "v10": ("VGRD", "10 m above ground"),
    "mslp": ("PRMSL", "mean sea level"),
}
GEFS_AVG = {
    "apcp": ("APCP", "surface"), "crain": ("CRAIN", "surface"), "cape": ("CAPE", "surface"),
    "cape_ml": ("CAPE", "180-0 mb above ground"), "cin": ("CIN", "surface"),
    "pwat": ("PWAT", "entire atmosphere (considered as a single layer)"), "t2m": ("TMP", "2 m above ground"),
    "tmax2m": ("TMAX", "2 m above ground"), "tmin2m": ("TMIN", "2 m above ground"),
    "rh2m": ("RH", "2 m above ground"), "gust": ("GUST", "surface"), "u10": ("UGRD", "10 m above ground"),
    "v10": ("VGRD", "10 m above ground"), "tcc": ("TCDC", "entire atmosphere"),
    "hlcy": ("HLCY", "3000-0 m above ground"), "prmsl": ("PRMSL", "mean sea level"), "dswrf": ("DSWRF", "surface"),
    "soilw": ("SOILW", "0-0.1 m below ground"),
}
GEFS_SPR = {k: GEFS_AVG[k] for k in ("apcp", "cape", "pwat", "t2m", "tmax2m", "tmin2m", "rh2m", "gust", "prmsl",
                                     "tcc", "hlcy")}

NR, NC, STEP = base.NR, base.NC, base.STEP
LATS = base.LAT_TOP - STEP * np.arange(NR)
DY_M = STEP * 111_320.0
DX_M = DY_M * np.cos(np.deg2rad(LATS))          # per-row east-west spacing in metres


# ------------------------------------------------------------ numerics (tested)
def ddx(a):
    """d/dx (per metre), x = east."""
    return np.gradient(a, axis=1) / DX_M[:, None]


def ddy(a):
    """d/dy (per metre), y = north; rows run north -> south so the sign flips."""
    return -np.gradient(a, axis=0) / DY_M


def dewpoint_c(t_c, rh):
    a, b = 17.625, 243.04
    gamma = np.log(np.clip(rh, 1.0, 100.0) / 100.0) + a * t_c / (b + t_c)
    return b * gamma / (a - gamma)


def nb_filters(a, size):
    """neighbourhood mean / max / std over size x size nodes (edge-replicated)."""
    from scipy.ndimage import maximum_filter, uniform_filter
    a = np.nan_to_num(a)
    mean = uniform_filter(a, size=size, mode="nearest")
    var = np.maximum(uniform_filter(a * a, size=size, mode="nearest") - mean * mean, 0.0)
    return mean, maximum_filter(a, size=size, mode="nearest"), np.sqrt(var)


def terrain_features(hgt: np.ndarray) -> dict:
    """Static terrain descriptors on the window grid (metres)."""
    from scipy.ndimage import maximum_filter, minimum_filter
    hgt = np.nan_to_num(hgt)
    mean9, mx9, std9 = nb_filters(hgt, 9)
    _, _, std5 = nb_filters(hgt, 5)
    dzdx, dzdy = ddx(hgt), ddy(hgt)
    return {"hgt": hgt, "dzdx": dzdx, "dzdy": dzdy, "slope": np.hypot(dzdx, dzdy),
            "elev_std_5": std5, "elev_std_9": std9, "relief_9": mx9 - minimum_filter(hgt, size=9, mode="nearest"),
            "elev_minus_nb9": hgt - mean9}


# Physical validity ranges.  Out-of-range values become NaN (never clipped): the archive uses sentinels such as
# 9999 for undefined soil moisture, and interpolating one into a real value would fabricate numbers.
VALID = {"soilw": (0.0, 1.0), "avg_soilw": (0.0, 1.0), "lcc": (0, 100), "mcc": (0, 100), "hcc": (0, 100),
         "rh850": (0, 100), "rh700": (0, 100), "rh500": (0, 100), "avg_rh2m": (0, 100), "spr_rh2m": (0, 60),
         "tskin_c": (-90, 95), "k_index": (-80, 100), "q2m_gkg": (0, 60), "q850_gkg": (0, 40),
         "prate_mmh": (0, 1000), "hlcy": (-3000, 5000), "cape255": (0, 12000), "cape90": (0, 12000),
         "cin180": (-5000, 1), "lftx4": (-30, 30), "avg_tcc": (0, 100), "spr_tcc": (0, 100)}


def mask_invalid(arrays: dict) -> dict:
    out = {}
    for name, arr in arrays.items():
        bounds = VALID.get(name)
        if bounds is not None:
            arr = np.where((arr >= bounds[0]) & (arr <= bounds[1]), arr, np.nan)
        out[name] = arr
    return out


def derive_ext(f: dict, terrain: dict) -> dict:
    """Window arrays (raw GRIB units) -> model-ready derived arrays.  Missing inputs just skip their outputs."""
    out = {}
    have = f.keys()

    def need(*names):
        return all(n in have for n in names)

    for lvl in ("850", "700", "500", "200"):
        if need(f"u{lvl}", f"v{lvl}"):
            out[f"wind{lvl}"] = np.hypot(f[f"u{lvl}"], f[f"v{lvl}"])
    if need("u850", "v850", "u200", "v200"):
        out["shear_850_200"] = np.hypot(f["u200"] - f["u850"], f["v200"] - f["v850"])
    if need("u850", "v850", "u500", "v500"):
        out["shear_850_500"] = np.hypot(f["u500"] - f["u850"], f["v500"] - f["v850"])
    if need("u850", "v850"):
        out["vort850_e5"] = (ddx(f["v850"]) - ddy(f["u850"])) * 1e5
        out["div850_e5"] = (ddx(f["u850"]) + ddy(f["v850"])) * 1e5
        out["wdir850_sin"] = np.sin(np.arctan2(-f["u850"], -f["v850"]))
        out["wdir850_cos"] = np.cos(np.arctan2(-f["u850"], -f["v850"]))
    if need("q850", "u850", "v850"):                     # moisture-flux convergence at 850 hPa
        out["mfc850_e7"] = -(ddx(f["q850"] * f["u850"]) + ddy(f["q850"] * f["v850"])) * 1e7
        out["q850_gkg"] = f["q850"] * 1e3
    if need("t850", "t500", "rh850", "t700", "rh700"):
        t850, t700, t500 = f["t850"] - 273.15, f["t700"] - 273.15, f["t500"] - 273.15
        td850, td700 = dewpoint_c(t850, f["rh850"]), dewpoint_c(t700, f["rh700"])
        out["k_index"] = (t850 - t500) + td850 - (t700 - td700)
        out["total_totals"] = t850 + td850 - 2 * t500
        out["t850_c"], out["t700_c"], out["t500_c"] = t850, t700, t500
        out["dpt_depr_700"] = t700 - td700
    if need("t850", "t500", "z850", "z500"):
        out["lapse_850_500"] = (f["t850"] - f["t500"]) / np.maximum((f["z500"] - f["z850"]) / 1000.0, 1e-3)
        out["z500"], out["thickness_850_500"] = f["z500"], f["z500"] - f["z850"]
    for name in ("rh850", "rh700", "rh500", "w850", "w700", "w500", "absv850", "absv500"):
        if name in have:
            out[name] = f[name] * (1e5 if name.startswith("absv") else 1.0)
    if need("u10", "v10"):
        out["wdir10_sin"] = np.sin(np.arctan2(-f["u10"], -f["v10"]))
        out["wdir10_cos"] = np.cos(np.arctan2(-f["u10"], -f["v10"]))
        out["upslope10"] = f["u10"] * terrain["dzdx"] + f["v10"] * terrain["dzdy"]    # wind component up the terrain
    if need("mslp"):
        out["pgrad_hpa100km"] = np.hypot(ddx(f["mslp"]), ddy(f["mslp"])) * 1e3
    if need("t2m"):
        out["t2m_nb9_std"] = nb_filters(f["t2m"], 9)[2]
    if need("apcp"):
        m5, x5, _ = nb_filters(f["apcp"], 5)
        m9, x9, s9 = nb_filters(f["apcp"], 9)
        out.update({"apcp_nb5_mean": m5, "apcp_nb5_max": x5, "apcp_nb9_mean": m9, "apcp_nb9_max": x9, "apcp_nb9_std": s9})
    for name, size, stat, col in (("cape", 5, 1, "cape_nb5_max"), ("cape", 9, 1, "cape_nb9_max"),
                                  ("refc", 5, 1, "refc_nb5_max"), ("gust", 5, 1, "gust_nb5_max"),
                                  ("pwat", 9, 0, "pwat_nb9_mean")):
        if name in have:
            out[col] = nb_filters(f[name], size)[stat]
    if "tskin" in have:
        out["tskin_c"] = f["tskin"] - 273.15
    if "q2m" in have:
        out["q2m_gkg"] = f["q2m"] * 1e3
    if "prate" in have:
        out["prate_mmh"] = f["prate"] * 3600.0
    for name in ("soilw", "dswrf", "lhtfl", "shtfl", "lcc", "mcc", "hcc", "cape255", "cape90", "cin180", "hlcy", "lftx4"):
        if name in have:
            out[name] = f[name]
    # --- ensemble (GEFS) mean / spread
    for pre in ("avg_", "spr_"):
        for name in ("apcp", "cape", "pwat", "rh2m", "gust", "tcc", "hlcy", "dswrf", "soilw", "cin", "cape_ml", "crain"):
            if f"{pre}{name}" in have:
                out[f"{pre}{name}"] = f[f"{pre}{name}"]
        for name, off in (("t2m", 273.15), ("tmax2m", 273.15), ("tmin2m", 273.15)):
            if f"{pre}{name}" in have:
                # mean is a temperature (K -> C); spread is a difference, so it needs no offset
                out[f"{pre}{name}_c"] = f[f"{pre}{name}"] - (off if pre == "avg_" else 0.0)
        if f"{pre}prmsl" in have:
            out[f"{pre}prmsl_hpa"] = f[f"{pre}prmsl"] / 100.0
    if need("avg_u10", "avg_v10"):
        out["avg_wind10"] = np.hypot(f["avg_u10"], f["avg_v10"])
    return mask_invalid({k: np.asarray(v, "float32") for k, v in out.items()})


# --------------------------------------------------------------- daily reduction
MEAN_F = ["wind850", "wind200", "shear_850_200", "shear_850_500", "rh850", "rh700", "rh500", "t850_c", "t500_c",
          "z500", "lapse_850_500", "thickness_850_500", "k_index", "total_totals", "vort850_e5", "div850_e5",
          "mfc850_e7", "q850_gkg", "upslope10", "pgrad_hpa100km", "t2m_nb9_std", "tskin_c", "q2m_gkg", "soilw",
          "dswrf", "lcc", "mcc", "hcc", "absv500", "wdir10_sin", "wdir10_cos", "wdir850_sin", "wdir850_cos",
          "dpt_depr_700", "avg_t2m_c", "spr_t2m_c", "avg_rh2m", "spr_rh2m", "avg_pwat", "spr_pwat", "avg_tcc",
          "spr_tcc", "avg_prmsl_hpa", "spr_prmsl_hpa", "avg_wind10", "avg_dswrf", "avg_soilw", "lhtfl", "shtfl"]
MAX_F = ["wind850", "k_index", "total_totals", "mfc850_e7", "vort850_e5", "upslope10", "pgrad_hpa100km", "cape255",
         "cape90", "hlcy", "prate_mmh", "tskin_c", "absv500", "avg_cape", "spr_cape", "avg_cape_ml", "avg_crain",
         "avg_gust", "spr_gust", "avg_hlcy", "spr_hlcy", "apcp_nb5_max", "apcp_nb9_max", "cape_nb5_max",
         "cape_nb9_max", "refc_nb5_max", "gust_nb5_max", "spr_apcp", "lftx4"]
MIN_F = ["w850", "w700", "w500", "rh700", "cin180", "lftx4", "avg_cin", "div850_e5", "lapse_850_500"]
NB_RAIN = ("rain_nb5_mean_mm", "rain_nb5_max_mm", "rain_nb9_mean_mm", "rain_nb9_max_mm", "rain_nb9_std_mm",
           "rain_frac1_nb9", "rain_frac2p5_nb5", "rain_frac10_nb9")


def derive_all(arrays: dict, terrain: dict) -> dict:
    """derive_ext for every lead ONCE; both the station rows and the daily rows reuse it."""
    return {lead: derive_ext(fields, terrain) for lead, fields in arrays.items()}


def station_step_ext(run_date, arrays, terrain, stn_ids, stn_sampler, derived_all=None):
    derived_all = derived_all or derive_all(arrays, terrain)
    frames = []
    for lead in sorted(arrays):
        derived = derived_all[lead]
        data = {"point_id": stn_ids, "run_date": np.datetime64(run_date), "lead_h": np.int16(lead)}
        for name, arr in derived.items():
            data[name] = stn_sampler(arr).astype("float32")
        for wname in ("apcp", "tmax2m", "tmin2m"):
            for pre in ("avg_", "spr_"):          # GFS apcp width is already in the base table (apcp_width_h)
                if f"{pre}{wname}" in arrays[lead] and f"width:{pre}{wname}" in arrays[lead]:
                    data[f"{pre}{wname}_width_h"] = np.int8(arrays[lead][f"width:{pre}{wname}"])
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def point_day_ext(run_date, arrays, terrain, point_ids, sampler, derived_all=None):
    derived_all = derived_all or derive_all(arrays, terrain)
    sampled, grids = {}, {}
    for lead, fields in arrays.items():
        derived = derived_all[lead]
        sampled[lead] = {k: sampler(v).astype("float32") for k, v in derived.items()}
        sampled[lead]["_w"] = {k: fields[f"width:{k}"] for k in ("apcp", "avg_apcp", "avg_tmax2m", "avg_tmin2m", "spr_tmax2m", "spr_tmin2m")
                               if f"width:{k}" in fields}
        grids[lead] = {"apcp": fields.get("apcp"), "avg_apcp": fields.get("avg_apcp")}
    n = len(point_ids)
    nan = np.full(n, np.nan, "float32")
    frames = []
    for k, steps in base.DAY_STEPS.items():
        have = [h for h in steps if h in sampled]
        if not have:
            continue
        row = {"point_id": point_ids, "run_date": np.datetime64(run_date), "day_k": np.int8(k)}
        with np.errstate(all="ignore"):
            for names, reducer, suffix in ((MEAN_F, np.nanmean, "mean"), (MAX_F, np.nanmax, "max"), (MIN_F, np.nanmin, "min")):
                for name in names:
                    stack = [sampled[h][name] for h in have if name in sampled[h]]
                    row[f"{name}_{suffix}"] = reducer(np.stack(stack), axis=0).astype("float32") if stack else nan
            # rain over the UTC day from the 6-hour buckets, with neighbourhood statistics on the DAILY grid
            buckets = [h for h in have if h % 6 == 0 and sampled[h]["_w"].get("apcp") == 6 and grids[h]["apcp"] is not None]
            if buckets:
                day = np.sum([np.nan_to_num(grids[h]["apcp"]) for h in buckets], axis=0)
                m5, x5, _ = nb_filters(day, 5)
                m9, x9, s9 = nb_filters(day, 9)
                from scipy.ndimage import uniform_filter
                grids_day = {"rain_nb5_mean_mm": m5, "rain_nb5_max_mm": x5, "rain_nb9_mean_mm": m9, "rain_nb9_max_mm": x9,
                             "rain_nb9_std_mm": s9,
                             "rain_frac1_nb9": uniform_filter((day >= 1.0).astype("float32"), size=9, mode="nearest"),
                             "rain_frac2p5_nb5": uniform_filter((day >= 2.5).astype("float32"), size=5, mode="nearest"),
                             "rain_frac10_nb9": uniform_filter((day >= 10.0).astype("float32"), size=9, mode="nearest")}
                for name, arr in grids_day.items():
                    row[name] = sampler(arr.astype("float32")).astype("float32")
            else:
                for name in NB_RAIN:
                    row[name] = nan
            # ensemble rain: sum of the ensemble-mean 6 h buckets; spread combined in quadrature
            eb = [h for h in have if h % 6 == 0 and sampled[h]["_w"].get("avg_apcp") == 6 and "avg_apcp" in sampled[h]]
            row["gefs_rain_mean_mm"] = np.sum([sampled[h]["avg_apcp"] for h in eb], axis=0).astype("float32") if eb else nan
            sb = [h for h in eb if "spr_apcp" in sampled[h]]
            row["gefs_rain_spread_mm"] = np.sqrt(np.sum([sampled[h]["spr_apcp"] ** 2 for h in sb], axis=0)).astype("float32") if sb else nan
            # ensemble 6 h Tmax / Tmin: the UTC day's max of window-maxima / min of window-minima
            for kind, red, label in (("tmax2m", np.nanmax, "tmax"), ("tmin2m", np.nanmin, "tmin")):
                ws = [h for h in have if h % 6 == 0 and sampled[h]["_w"].get(f"avg_{kind}") == 6 and f"avg_{kind}_c" in sampled[h]]
                row[f"gefs_{label}_mean_c"] = red(np.stack([sampled[h][f"avg_{kind}_c"] for h in ws]), axis=0).astype("float32") if ws else nan
                sp = [h for h in ws if f"spr_{kind}_c" in sampled[h]]
                row[f"gefs_{label}_spread_c"] = np.mean([sampled[h][f"spr_{kind}_c"] for h in sp], axis=0).astype("float32") if sp else nan
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------- fetching
def gefs_idx_task(args):
    kind, run_date, lead = args
    url = (f"{GEFS}/gefs.{run_date:%Y%m%d}/{base.CYCLE}/atmos/pgrb2sp25/"
           f"{kind}.t{base.CYCLE}z.pgrb2s.0p25.f{lead:03d}")
    response = base._request(url + ".idx")
    if response is None:
        return (kind, lead), None
    rows = []
    for line in response.text.strip().splitlines():
        p = line.split(":")
        rows.append((int(p[1]), p[3], p[4], p[5]))
    return (kind, lead), (url, rows)


def fetch_ext_run(pool, run_date, static=False):
    gfs_idx = dict(pool.map(base.idx_task, [(run_date, lead) for lead in base.LEADS]))
    gefs_idx = dict(pool.map(gefs_idx_task, [(k, run_date, lead) for k in KIND_PREFIX for lead in base.LEADS]))
    if all(v is None for v in gfs_idx.values()):
        return None, {"absent": True}
    tasks, wanted_total, not_in_idx = [], Counter(), Counter()

    def add(lead, found, want, prefix):
        url, rows = found
        picked = base.pick(rows, want)
        for name in want:
            wanted_total[prefix + name] += 1
            if name not in picked:
                not_in_idx[prefix + name] += 1
        for name, (s, e, d) in picked.items():
            tasks.append((lead, prefix + name, url, s, e, d))

    for lead, found in gfs_idx.items():
        if found is not None:
            want = dict(GFS_EXT)
            if static and lead == base.LEADS[0]:
                want.update(base.STATIC)
            add(lead, found, want, "")
    for (kind, lead), found in gefs_idx.items():
        if found is not None:
            add(lead, found, GEFS_AVG if kind == "geavg" else GEFS_SPR, KIND_PREFIX[kind])
    arrays, bad = {}, []
    for lead, name, window, d in pool.map(base.msg_task, tasks, chunksize=4):
        if window is None:
            if d.startswith("bad:"):
                bad.append(f"f{lead:03d}:{name}")
            continue
        arrays.setdefault(lead, {})[name] = window
        width = base.window_hours(d)
        if width is not None:
            arrays[lead][f"width:{name}"] = width
    return arrays, {"bad": bad, "not_in_idx": dict(not_in_idx), "gfs_missing_steps": sum(v is None for v in gfs_idx.values()),
                    "gefs_missing_files": sum(v is None for v in gefs_idx.values())}


# ----------------------------------------------------------------------- driver
def main():
    base.ensure_eccodes()
    OUT.mkdir(parents=True, exist_ok=True)
    dates = base.run_dates()
    started = time.time()
    report = {"dates": len(dates), "absent_runs": [], "bad_messages": {}, "field_absence": Counter(), "n_runs_checked": 0}
    ctx = mp.get_context("fork")
    with cf.ProcessPoolExecutor(max_workers=12, mp_context=ctx, initializer=base._init_worker) as pool:
        points = static_arrays = None
        for d in dates:
            arrays, stats = fetch_ext_run(pool, d, static=True)
            if arrays is not None and base.LEADS[0] in arrays and "hgt" in arrays[base.LEADS[0]] and "land" in arrays[base.LEADS[0]]:
                points = base.build_points(arrays[base.LEADS[0]]["land"], arrays[base.LEADS[0]]["hgt"])
                static_arrays, first_date, first = arrays, d, (arrays, stats)
                break
        if points is None:
            raise RuntimeError("no date yielded orography + land mask")
        terrain = terrain_features(static_arrays[base.LEADS[0]]["hgt"])
        sampler = base.Sampler(points["lat"].to_numpy(), points["lon"].to_numpy())
        ext_points = points[["point_id"]].copy()
        for name in ("hgt", "slope", "elev_std_5", "elev_std_9", "relief_9", "elev_minus_nb9"):
            ext_points[f"terrain_{name}"] = sampler(terrain[name]).astype("float32")
        ext_points.to_parquet(OUT / "points_ext.parquet", index=False)
        point_ids = points["point_id"].to_numpy()
        stn = points[points["kind"] == "station"]
        stn_ids, stn_sampler = stn["point_id"].to_numpy(), base.Sampler(stn["lat"].to_numpy(), stn["lon"].to_numpy())
        print("[ext] points", len(points), "stations", len(stn), flush=True)

        p1_parts, p2_parts, month_key, written = [], [], None, []

        def flush(key):
            if not p2_parts:
                return
            pd.concat(p1_parts, ignore_index=True).to_parquet(OUT / f"p1x_station_steps_{key}.parquet", index=False)
            pd.concat(p2_parts, ignore_index=True).to_parquet(OUT / f"p2x_point_days_{key}.parquet", index=False)
            written.append(key)
            p1_parts.clear()
            p2_parts.clear()

        for n, d in enumerate(dates):
            key = f"{d:%Y%m}"
            if month_key is not None and key != month_key:
                flush(month_key)
            month_key = key
            arrays, stats = first if d == first_date else fetch_ext_run(pool, d)
            if arrays is None:
                report["absent_runs"].append(d.isoformat())
                continue
            report["n_runs_checked"] += 1
            report["field_absence"].update(stats.get("not_in_idx", {}))
            if stats.get("bad"):
                report["bad_messages"][d.isoformat()] = stats["bad"]
                print(f"[ext] {d}: {len(stats['bad'])} unreadable message(s): {stats['bad'][:4]}", flush=True)
            derived_all = derive_all(arrays, terrain)
            p1_parts.append(station_step_ext(d, arrays, terrain, stn_ids, stn_sampler, derived_all))
            p2_parts.append(point_day_ext(d, arrays, terrain, point_ids, sampler, derived_all))
            del derived_all
            if (n + 1) % 5 == 0 or n == len(dates) - 1:
                print(f"[ext] {n + 1}/{len(dates)} dates  {(time.time() - started) / (n + 1):.1f}s/date  "
                      f"absent={len(report['absent_runs'])} bad_runs={len(report['bad_messages'])}", flush=True)
        if month_key is not None:
            flush(month_key)
    report["months_written"] = written
    report["seconds"] = round(time.time() - started)
    report["field_absence"] = dict(report["field_absence"])
    (OUT / "ext_report.json").write_text(json.dumps(report, indent=1))
    print("EXT_REPORT_BEGIN")
    print(json.dumps(report, indent=1))
    print("EXT_REPORT_END")


if __name__ == "__main__":
    main()
