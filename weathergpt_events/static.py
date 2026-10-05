"""Static geography at any point, defined exactly as in training.

Training computed elevation, land fraction, distance to coast and the terrain descriptors by sampling grids built
from GFS orography and the GFS land-sea mask (`collect_gfs.build_points`, `collect_gfs_ext.terrain_features`).
A query at an arbitrary latitude/longitude must see the SAME definitions, so this module stores those grids once and
samples them with the same bilinear `Sampler`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from event_models import collect_gfs as base
from event_models import collect_gfs_ext as ext
from event_models import features_events as fe

TERRAIN_NAMES = ("hgt", "slope", "elev_std_5", "elev_std_9", "relief_9", "elev_minus_nb9")


def build_static_grids(land: np.ndarray, hgt: np.ndarray) -> dict:
    """Grids on the 129 x 125 India window.  `land` is the GFS LAND mask (0/1), `hgt` the surface height in metres."""
    from scipy.ndimage import distance_transform_edt

    terrain = ext.terrain_features(hgt)
    return {"land": np.asarray(land, "float32"), "hgt": np.nan_to_num(np.asarray(hgt, "float32")),
            "dzdx": np.asarray(terrain["dzdx"], "float32"), "dzdy": np.asarray(terrain["dzdy"], "float32"),   # for upslope wind
            "coast_px": distance_transform_edt(land >= 0.5).astype("float32"),     # land cell -> nearest sea, in 0.25 deg steps
            "sea_px": distance_transform_edt(land < 0.5).astype("float32"),
            **{f"terrain_{n}": np.asarray(terrain[n], "float32") for n in TERRAIN_NAMES}}


def save_static_grids(grids: dict, path: str | Path) -> None:
    np.savez_compressed(path, **grids)


def load_static_grids(path: str | Path) -> dict:
    blob = np.load(path)
    return {k: blob[k] for k in blob.files}


def point_static(grids: dict, lat, lon) -> pd.DataFrame:
    """One row per (lat, lon) with the same columns training's `points` table carried."""
    lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
    sampler = base.Sampler(lat, lon)
    land_here = sampler(grids["land"])
    # same rule as training: distance to the sea for land points, distance to land for sea points
    px = np.where(land_here >= 0.5, sampler(grids["coast_px"]), sampler(grids["sea_px"]))
    out = pd.DataFrame({"lat": lat, "lon": lon, "elevation_m": sampler(grids["hgt"]).astype("float32"),
                        "land_frac": land_here.astype("float32"),
                        "dist_coast_km": (px * base.STEP * 111.0).astype("float32")})
    for name in TERRAIN_NAMES:
        out[f"terrain_{name}"] = sampler(grids[f"terrain_{name}"]).astype("float32")
    out["zone"] = fe.region_of(out["lat"], out["lon"], out["elevation_m"], out["dist_coast_km"])
    out["zone_code"] = out["zone"].map({z: i for i, z in enumerate(fe.ZONES)}).astype("int8")
    return out


def in_domain(lat, lon, grids: dict | None = None, min_land: float = 0.5) -> tuple:
    """(ok, reason).  The models are land-only and valid inside the collected box."""
    lat_min, lat_max = base.LAT_TOP - (base.NR - 1) * base.STEP, base.LAT_TOP
    lon_min, lon_max = base.LON_LEFT, base.LON_LEFT + (base.NC - 1) * base.STEP
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return False, f"outside the modelled domain ({lat_min}-{lat_max}N, {lon_min}-{lon_max}E)"
    if grids is not None:
        land = float(base.Sampler([lat], [lon])(grids["land"])[0])
        if land < min_land:
            return False, f"not over land (land fraction {land:.2f}); these models are land-only"
    return True, "ok"
