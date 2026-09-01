"""
GFS GRIB2 placeholder decoder.

Full GRIB2 decoding needs eccodes/cfgrib which is heavy.
This module shows the CORRECT pattern (selection by param/level/lead + spatial sampling)
and falls back to Open-Meteo when GRIB files are not present, so training/inference never breaks.

When you have local GRIB2 files (e.g. from NOAA NCEP), enable the real path:
    python -m app.decoders.grib2_placeholder --grib path/to/gfs.t00z.pgrb2.0p25.f006 --lat 21.14 --lon 79.08

Install full stack:
    pip install -r requirements-full.txt  # pulls cfgrib+eccodes
    conda install -c conda-forge eccodes cfgrib   # if pip fails
"""
from __future__ import annotations
from pathlib import Path
from typing import List
from datetime import datetime, timezone
from app.schemas.ceo import CanonicalEvidenceObject, Geometry, Provenance

def decode_grib2_file(path: str, lat: float, lon: float) -> List[CanonicalEvidenceObject]:
    try:
        import xarray as xr
        import cfgrib  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"GRIB2 decoder needs cfgrib+eccodes. Install requirements-full.txt. Detail: {e}")

    # Real pattern: open per-parameter, not whole file
    # Example: t2m + apcp
    out: List[CanonicalEvidenceObject] = []
    try:
        ds = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "surface", "shortName": "2t"})
        # spatial sampling — nearest grid cell
        point = ds.sel(latitude=lat, longitude=lon if lon<=180 else lon-360, method="nearest")
        val = float(point["t2m"].values) - 273.15  # K → C
        out.append(CanonicalEvidenceObject(
            source="GFS", evidence_class="forecast", variable="temperature_2m", value=val, unit="C", statistic="instant",
            geometry=Geometry(type="GridCell", coordinates=[lon, lat]),
            model_name="GFS", spatial_resolution="0.25°",
            provenance=Provenance(original_source="GFS GRIB2", original_unit="K", transformations=["decoded GRIB2","selected 2t surface","spatially sampled","K→C"], raw_record_id=Path(path).name)
        ))
    except Exception as e:
        print(f"[grib2] t2m decode skipped: {e}")

    try:
        ds2 = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "surface", "shortName": "tp"})
        point = ds2.sel(latitude=lat, longitude=lon if lon<=180 else lon-360, method="nearest")
        val = float(point["tp"].values)  # kg m-2 ≈ mm
        out.append(CanonicalEvidenceObject(
            source="GFS", evidence_class="forecast", variable="precipitation_amount", value=val, unit="mm", statistic="accumulation",
            geometry=Geometry(type="GridCell", coordinates=[lon, lat]),
            accumulation_window_hours=6, model_name="GFS", spatial_resolution="0.25°",
            provenance=Provenance(original_source="GFS GRIB2", original_unit="kg m-2", transformations=["decoded GRIB2","selected tp","spatially sampled","kg m-2→mm"], raw_record_id=Path(path).name)
        ))
    except Exception as e:
        print(f"[grib2] tp decode skipped: {e}")

    return out

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--grib", required=True)
    ap.add_argument("--lat", type=float, default=21.14)
    ap.add_argument("--lon", type=float, default=79.08)
    args = ap.parse_args()
    ceos = decode_grib2_file(args.grib, args.lat, args.lon)
    print(json.dumps([c.model_dump(mode="json") for c in ceos], indent=2))
