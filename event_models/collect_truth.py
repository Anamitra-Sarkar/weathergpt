"""Collect ground truth for the event models (runs as a Kaggle kernel).

  METAR   all Indian IEM stations, 2016-01-01 -> today.  Raw typed observations
          AND derived hourly / daily labels are saved per station.  Nothing is
          filtered here: cleaning is a separate, later pass so that a bad rule
          can be changed without re-collecting.
  CHIRPS  daily 0.05 deg rainfall over the whole India window, as float16 arrays
          per year.  CHIRPS declares no nodata value and uses -9999 over the
          ocean (found by the probe), so anything < 0 is masked explicitly.

Output layout under /kaggle/working/truth/ :
  stations.parquet                      one row per station + coverage stats
  metar_obs/{sid}.parquet               typed observations (+ raw wxcodes)
  metar_hourly/{sid}.parquet            hourly event labels + continuous truth
  metar_daily/{sid}.parquet             IST-day Tmax/Tmin + hot/cold flags
  chirps/chirps_india_{year}.npz        data (days, 640, 620) float16, dates
  chirps/grid.json                      lat/lon origin and step
  collect_report.json
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from event_models import labels

OUT = Path("/kaggle/working/truth") if Path("/kaggle").exists() else Path("truth_out")
UA = {"User-Agent": "WeatherGPT-research/1.0 (SIH 2026; academic evaluation)"}
START = date(2016, 1, 1)
IEM = "https://mesonet.agron.iastate.edu"
CHIRPS = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"

# India window on the CHIRPS p05 global grid (lon -180..180 / lat 50..-50, 0.05 deg)
LAT_N, LAT_S, LON_W, LON_E, STEP = 38.0, 6.0, 67.0, 98.0, 0.05
ROW0, ROW1 = int(round((50.0 - LAT_N) / STEP)), int(round((50.0 - LAT_S) / STEP))
COL0, COL1 = int(round((LON_W + 180.0) / STEP)), int(round((LON_E + 180.0) / STEP))


def _get(client, url, params=None, *, attempts=5, timeout=300):
    last = None
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params, headers=UA, timeout=timeout, follow_redirects=True)
            if response.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {response.status_code}"
                time.sleep(5 * (attempt + 1))
                continue
            return response
        except Exception as exc:  # network errors are retried, then surfaced
            last = repr(exc)[:120]
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url} failed: {last}")


# ----------------------------------------------------------------------- METAR
def collect_metar(client) -> pd.DataFrame:
    for sub in ("metar_obs", "metar_hourly", "metar_daily"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    response = _get(client, f"{IEM}/geojson/network/IN__ASOS.geojson")
    stations = []
    for feature in response.json()["features"]:
        props = feature["properties"]
        stations.append({"sid": props["sid"], "name": props.get("sname"), "state": props.get("state"),
                         "lat": feature["geometry"]["coordinates"][1],
                         "lon": feature["geometry"]["coordinates"][0],
                         "elevation_m": props.get("elevation"),
                         "archive_begin": props.get("archive_begin"), "archive_end": props.get("archive_end")})
    today = date.today()
    rows = []
    for number, station in enumerate(stations):
        sid = station["sid"]
        begin = (station["archive_begin"] or "9999-01-01")[:10]
        end = (station["archive_end"] or today.isoformat())[:10]
        record = dict(station, n_obs=0, n_hours=0, first_obs=None, last_obs=None, status="skipped")
        if begin[:4] == "9999" or end < START.isoformat():
            rows.append(record)
            continue
        start = max(date.fromisoformat(begin), START)
        stop = min(date.fromisoformat(end), today)
        obs_path = OUT / "metar_obs" / f"{sid}.parquet"
        if obs_path.exists():
            record["status"] = "cached"
            rows.append(record)
            continue
        params = {"station": sid, "data": ["wxcodes", "vsby", "gust", "tmpf", "dwpf", "sknt", "skyc1"],
                  "year1": start.year, "month1": start.month, "day1": start.day,
                  "year2": stop.year, "month2": stop.month, "day2": stop.day,
                  "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "missing": "M",
                  "trace": "T", "report_type": [3, 4]}
        try:
            response = _get(client, f"{IEM}/cgi-bin/request/asos.py", params)
            raw = pd.read_csv(io.StringIO(response.text), na_values=["M"], low_memory=False)
            if raw.empty:
                record["status"] = "empty"
                rows.append(record)
                continue
            obs = labels.clean_observations(raw)
            hourly = labels.hourly_labels(obs)
            daily = labels.daily_temperature_labels(obs)
            obs.to_parquet(obs_path, index=False)
            hourly.to_parquet(OUT / "metar_hourly" / f"{sid}.parquet", index=False)
            daily.to_parquet(OUT / "metar_daily" / f"{sid}.parquet", index=False)
            record.update(n_obs=int(len(obs)), n_hours=int(len(hourly)), status="ok",
                          first_obs=str(obs["valid"].min()), last_obs=str(obs["valid"].max()),
                          ts_hours=int(hourly["ts_any"].sum()), fog_hours=int(hourly["fog"].sum()),
                          strong_wind_hours=int(hourly["strong_wind"].sum()),
                          rain_hours=int(hourly["rain_any"].sum()), dust_hours=int(hourly["dust"].sum()),
                          hot_days=int(daily["hot_day"].sum()), n_days=int(len(daily)))
        except Exception as exc:
            record["status"] = f"error: {repr(exc)[:100]}"
        rows.append(record)
        if (number + 1) % 10 == 0:
            done = sum(1 for r in rows if r["status"] == "ok")
            print(f"[metar] {number + 1}/{len(stations)} stations, {done} ok", flush=True)
        time.sleep(1.0)
    table = pd.DataFrame(rows)
    table.to_parquet(OUT / "stations.parquet", index=False)
    return table


# ---------------------------------------------------------------------- CHIRPS
def _chirps_day(client, day: date):
    from rasterio.io import MemoryFile

    url = f"{CHIRPS}/{day.year}/chirps-v2.0.{day:%Y.%m.%d}.tif.gz"
    response = _get(client, url, attempts=4, timeout=180)
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(f"CHIRPS {day} HTTP {response.status_code}")
    with MemoryFile(gzip.decompress(response.content)) as memfile, memfile.open() as dataset:
        if not (abs(dataset.transform.c + 180.0) < 1e-6 and abs(dataset.transform.f - 50.0) < 1e-6):
            raise RuntimeError(f"unexpected CHIRPS grid origin {dataset.transform}")
        full = dataset.read(1)
    window = full[ROW0:ROW1, COL0:COL1].astype("float32")
    window[window < 0] = np.nan  # -9999 over ocean / no-coverage
    return window.astype("float16")


def collect_chirps(client) -> dict:
    try:
        import rasterio  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=False)
    folder = OUT / "chirps"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "grid.json").write_text(json.dumps({
        "lat_north": LAT_N, "lat_south": LAT_S, "lon_west": LON_W, "lon_east": LON_E, "step": STEP,
        "rows": ROW1 - ROW0, "cols": COL1 - COL0, "row_order": "north_to_south",
        "units": "mm/day", "dtype": "float16", "nan": "ocean / no coverage"}))
    summary = {}
    for year in range(START.year, date.today().year + 1):
        path = folder / f"chirps_india_{year}.npz"
        if path.exists():
            summary[year] = "cached"
            continue
        days = [date(year, 1, 1) + timedelta(days=i)
                for i in range((date(year, 12, 31) - date(year, 1, 1)).days + 1)
                if date(year, 1, 1) + timedelta(days=i) <= date.today()]
        with cf.ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda d: _chirps_day(client, d), days))
        kept = [(d, r) for d, r in zip(days, results) if r is not None]
        if not kept:
            summary[year] = "no data"
            continue
        stack = np.stack([r for _, r in kept])
        np.savez_compressed(path, data=stack, dates=np.array([d.isoformat() for d, _ in kept]))
        land = np.isfinite(stack[0])
        wet = float((stack[:, land] > 1.0).mean()) if land.any() else float("nan")
        summary[year] = {"days": len(kept), "missing": len(days) - len(kept),
                         "land_frac": round(float(land.mean()), 3), "wet_day_frac_land": round(wet, 3),
                         "max_mm": round(float(np.nanmax(stack.astype("float32"))), 1),
                         "last": kept[-1][0].isoformat(), "mb": round(path.stat().st_size / 1e6, 1)}
        print(f"[chirps] {year}: {summary[year]}", flush=True)
    return summary


def main():
    import httpx

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with httpx.Client(limits=httpx.Limits(max_connections=32)) as client:
        stations = collect_metar(client)
        chirps = collect_chirps(client)
    ok = stations[stations["status"] == "ok"]
    report = {
        "seconds": round(time.time() - started),
        "stations_total": int(len(stations)), "stations_ok": int(len(ok)),
        "status_counts": stations["status"].str.slice(0, 20).value_counts().to_dict(),
        "obs_total": int(ok["n_obs"].sum()), "hours_total": int(ok["n_hours"].sum()),
        "ts_hours": int(ok["ts_hours"].sum()), "fog_hours": int(ok["fog_hours"].sum()),
        "strong_wind_hours": int(ok["strong_wind_hours"].sum()),
        "rain_hours": int(ok["rain_hours"].sum()), "dust_hours": int(ok["dust_hours"].sum()),
        "hot_days": int(ok["hot_days"].sum()), "station_days": int(ok["n_days"].sum()),
        "chirps": chirps}
    (OUT / "collect_report.json").write_text(json.dumps(report, indent=1, default=str))
    print("COLLECT_REPORT_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("COLLECT_REPORT_END")


if __name__ == "__main__":
    main()
