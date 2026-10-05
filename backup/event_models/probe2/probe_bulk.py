"""Probe 2: can we bulk-collect without the free API's per-call weighting?

  1. Open-Meteo's public S3 bucket: what is in it, how is it laid out.
  2. CHIRPS daily 0.05 deg rainfall: download one day, read it, window it to
     India, check units / nodata / timestamp convention.
  3. IEM METAR archive depth: station start years, and a multi-year pull for
     four stations to size rows/year and seconds/station.
Logs only; nothing is written anywhere that matters.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import subprocess
import sys
import time

UA = {"User-Agent": "WeatherGPT-research/1.0 (SIH 2026; academic evaluation)"}


def s3_list(client, prefix: str, delimiter: str = "/", max_keys: int = 60) -> dict:
    response = client.get("https://openmeteo.s3.amazonaws.com/", params={
        "list-type": 2, "prefix": prefix, "delimiter": delimiter, "max-keys": max_keys},
        headers=UA, timeout=60)
    text = response.text
    return {"status": response.status_code,
            "prefixes": re.findall(r"<Prefix>([^<]+)</Prefix>", text)[1:] if prefix else re.findall(r"<Prefix>([^<]+)</Prefix>", text),
            "keys": [(k, int(s)) for k, s in zip(re.findall(r"<Key>([^<]+)</Key>", text),
                                                  re.findall(r"<Size>(\d+)</Size>", text))],
            "truncated": "<IsTruncated>true" in text}


def probe_s3(client) -> dict:
    out = {}
    for prefix in ("", "data/", "data_spatial/", "data_run/", "data_previous_run/"):
        try:
            out[prefix or "(root)"] = s3_list(client, prefix)
        except Exception as exc:
            out[prefix or "(root)"] = {"error": repr(exc)[:120]}
    # dig one level into the first few model folders under data/
    try:
        models = [p for p in out["data/"]["prefixes"]]
        out["data_models"] = models
        for model in [m for m in models if any(t in m for t in ("gfs", "ecmwf", "icon", "era5", "chirps"))][:8]:
            listing = s3_list(client, model, max_keys=25)
            out[model] = {"prefixes": listing["prefixes"][:14], "keys": listing["keys"][:6]}
    except Exception as exc:
        out["dig_error"] = repr(exc)[:120]
    return out


def probe_chirps(client) -> dict:
    try:
        import rasterio  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=False)
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile

    out = {}
    base = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"
    for day in ("2024.06.01", "2026.08.01", "2026.09.15"):
        url = f"{base}/{day[:4]}/chirps-v2.0.{day}.tif.gz"
        started = time.time()
        try:
            response = client.get(url, headers=UA, timeout=180, follow_redirects=True)
        except Exception as exc:
            out[day] = {"error": repr(exc)[:100]}
            continue
        if response.status_code != 200:
            out[day] = {"status": response.status_code}
            continue
        raw = gzip.decompress(response.content)
        with MemoryFile(raw) as memfile, memfile.open() as dataset:
            window = rasterio.windows.from_bounds(68, 6, 98, 37.5, dataset.transform)
            data = dataset.read(1, window=window).astype("float32")
            nodata = dataset.nodata
            data[data == nodata] = np.nan
            out[day] = {"seconds": round(time.time() - started, 1), "gz_bytes": len(response.content),
                        "global_shape": [dataset.height, dataset.width], "india_shape": list(data.shape),
                        "nodata": nodata, "res": dataset.res,
                        "valid_frac": round(float(np.isfinite(data).mean()), 3),
                        "mean_mm": round(float(np.nanmean(data)), 2),
                        "frac_gt1mm": round(float((data[np.isfinite(data)] > 1).mean()), 3),
                        "max_mm": round(float(np.nanmax(data)), 1)}
    return out


def probe_iem_history(client) -> dict:
    import pandas as pd

    out = {}
    response = client.get("https://mesonet.agron.iastate.edu/geojson/network/IN__ASOS.geojson", headers=UA, timeout=60)
    stations = [f["properties"] for f in response.json()["features"]]
    begins = pd.Series([(s.get("archive_begin") or "9999")[:4] for s in stations])
    ends = pd.Series([(s.get("archive_end") or "open")[:4] for s in stations])
    out["n_stations"] = len(stations)
    out["begin_year_counts"] = begins.value_counts().sort_index().to_dict()
    out["still_open"] = int((ends == "open").sum())
    out["timing"] = {}
    for sid in ("VIDP", "VABB", "VOMM", "VIAR"):
        started = time.time()
        params = {"station": sid, "data": ["wxcodes", "vsby", "gust", "tmpf", "dwpf", "sknt", "p01i", "alti", "skyc1"],
                  "year1": 2018, "month1": 1, "day1": 1, "year2": 2026, "month2": 9, "day2": 30,
                  "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "missing": "M", "trace": "T",
                  "report_type": [3, 4]}
        try:
            r = client.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params=params, headers=UA, timeout=300)
            frame = pd.read_csv(io.StringIO(r.text), na_values=["M"], low_memory=False)
            frame["year"] = frame["valid"].str[:4]
            wx = frame["wxcodes"].fillna("").astype(str)
            out["timing"][sid] = {
                "seconds": round(time.time() - started, 1), "rows": int(len(frame)),
                "rows_per_year": frame["year"].value_counts().sort_index().to_dict(),
                "ts_rows": int(wx.str.contains("TS").sum()),
                "temp_valid_frac": round(float(frame["tmpf"].notna().mean()), 3),
                "p01i_valid_frac": round(float(frame["p01i"].notna().mean()), 3),
                "wx_top": wx[wx != ""].value_counts().head(8).to_dict()}
        except Exception as exc:
            out["timing"][sid] = {"error": repr(exc)[:120]}
        time.sleep(1)
    return out


def main():
    import httpx

    with httpx.Client() as client:
        report = {"s3_openmeteo": probe_s3(client),
                  "chirps": probe_chirps(client),
                  "iem_history": probe_iem_history(client)}
    print("PROBE2_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("PROBE2_END")


if __name__ == "__main__":
    main()
