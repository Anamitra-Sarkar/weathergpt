"""Feasibility probe for the event-model data foundry (Kaggle kernel, logs only).

Answers, with real requests and no assumptions:
  1. Are there real *observed* thunderstorm / fog / gust records for India
     (IEM's METAR archive), and how many stations / how complete?
  2. How far back does Open-Meteo's historical-forecast archive go per NWP
     model, and which event-relevant variables (CAPE, visibility, gusts...)
     actually carry data?
  3. Which independent rainfall truth sources are reachable from Modal?

Nothing here writes to a volume; the printed JSON is the deliverable.
"""
from __future__ import annotations

import io
import json
import time

UA = {"User-Agent": "WeatherGPT-research/1.0 (SIH 2026; academic evaluation)"}
HIST_FC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"]
EVENT_VARS = ["cape", "lifted_index", "convective_inhibition", "visibility",
              "wind_gusts_10m", "cloud_cover", "dew_point_2m", "surface_pressure",
              "boundary_layer_height", "weather_code", "precipitation_probability",
              "freezing_level_height", "shortwave_radiation"]


def _get(client, url, params, retries=3):
    last = None
    for attempt in range(retries):
        try:
            response = client.get(url, params=params, headers=UA, timeout=90)
            if response.status_code == 429:
                last = "429"
                time.sleep(5 * (attempt + 1))
                continue
            return response
        except Exception as exc:  # network errors are data here
            last = repr(exc)[:100]
            time.sleep(2)
    return last


def probe_iem(client) -> dict:
    import pandas as pd

    out: dict = {}
    response = _get(client, "https://mesonet.agron.iastate.edu/geojson/network/IN__ASOS.geojson", {})
    if isinstance(response, str) or response.status_code != 200:
        return {"error": f"station list failed: {response if isinstance(response, str) else response.status_code}"}
    stations = [f["properties"] | {"lon": f["geometry"]["coordinates"][0],
                                   "lat": f["geometry"]["coordinates"][1]}
                for f in response.json()["features"]]
    out["n_stations"] = len(stations)
    out["station_keys"] = sorted(stations[0].keys()) if stations else []
    out["sample"] = [{k: s.get(k) for k in ("sid", "sname", "lat", "lon", "elevation",
                                             "archive_begin", "archive_end")}
                     for s in stations[:5]]

    per_station = []
    totals = {"rows": 0, "ts": 0, "fg": 0, "vis_lt1km": 0, "gust_ge25kt": 0, "hz": 0, "du": 0, "ra": 0}
    for station in stations:
        sid = station["sid"]
        params = {"station": sid, "data": ["wxcodes", "vsby", "gust", "tmpf", "p01i", "skyc1"],
                  "year1": 2024, "month1": 1, "day1": 1, "year2": 2025, "month2": 1, "day2": 1,
                  "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "missing": "M",
                  "trace": "T", "report_type": [3, 4]}
        response = _get(client, "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params)
        time.sleep(0.6)
        if isinstance(response, str) or response.status_code != 200:
            per_station.append({"sid": sid, "error": str(response)[:80]})
            continue
        frame = pd.read_csv(io.StringIO(response.text), na_values=["M"], low_memory=False)
        if frame.empty:
            per_station.append({"sid": sid, "rows_2024": 0})
            continue
        wx = frame["wxcodes"].fillna("").astype(str)
        vis_km = pd.to_numeric(frame["vsby"], errors="coerce") * 1.609
        gust = pd.to_numeric(frame["gust"], errors="coerce")
        row = {"sid": sid, "rows_2024": int(len(frame)),
               "ts": int(wx.str.contains(r"\bTS|\+?TS|VCTS", regex=True).sum()),
               "fg": int(wx.str.contains(r"\bFG\b|\bFG ", regex=True).sum()),
               "vis_lt1km": int((vis_km < 1.0).sum()),
               "gust_ge25kt": int((gust >= 25).sum()),
               "hz": int(wx.str.contains("HZ").sum()),
               "du": int(wx.str.contains(r"\bDU|\bSS|\bDS", regex=True).sum()),
               "ra": int(wx.str.contains("RA").sum())}
        per_station.append(row)
        totals["rows"] += row["rows_2024"]
        for key, name in (("ts", "ts"), ("fg", "fg"), ("vis_lt1km", "vis_lt1km"),
                          ("gust_ge25kt", "gust_ge25kt"), ("hz", "hz"), ("du", "du"), ("ra", "ra")):
            totals[name] += row[key]
    out["totals_2024"] = totals
    out["stations_with_data_2024"] = sum(1 for s in per_station if s.get("rows_2024", 0) > 1000)
    out["per_station_2024"] = per_station
    # one raw wxcodes sample so the parsing assumption is visible in the log
    try:
        sample = [s for s in per_station if s.get("ts", 0) > 0][:1]
        if sample:
            out["ts_example_station"] = sample[0]["sid"]
    except Exception:
        pass
    return out


def probe_openmeteo(client) -> dict:
    import pandas as pd

    out: dict = {"forecast_archive": {}, "truth_archive": {}}
    dates = ["2021-06-01", "2022-06-01", "2023-06-01", "2024-06-01", "2025-06-01"]
    for model in MODELS:
        out["forecast_archive"][model] = {}
        for day in dates:
            end = day[:-2] + "03"
            response = _get(client, HIST_FC, {
                "latitude": 28.61, "longitude": 77.21, "start_date": day, "end_date": end,
                "hourly": ",".join(["temperature_2m", "precipitation"] + EVENT_VARS),
                "models": model, "timezone": "UTC"})
            if isinstance(response, str) or response.status_code != 200:
                status = response if isinstance(response, str) else f"{response.status_code} {response.text[:90]}"
                out["forecast_archive"][model][day] = {"error": status}
                time.sleep(1)
                continue
            frame = pd.DataFrame(response.json()["hourly"])
            nonnull = {c: round(float(frame[c].notna().mean()), 2) for c in frame.columns if c != "time"}
            out["forecast_archive"][model][day] = {
                "temp": nonnull.get("temperature_2m"),
                "have": sorted(c for c, v in nonnull.items() if v > 0.5 and c not in ("temperature_2m", "precipitation")),
            }
            time.sleep(1)
    for model in ("era5", "era5_seamless", "era5_land", "ecmwf_ifs"):
        response = _get(client, ARCHIVE, {
            "latitude": 28.61, "longitude": 77.21, "start_date": "2024-06-01", "end_date": "2024-06-03",
            "hourly": ",".join(["temperature_2m", "precipitation", "wind_gusts_10m", "cape",
                                "visibility", "cloud_cover", "dew_point_2m", "weather_code"]),
            "models": model, "timezone": "UTC"})
        if isinstance(response, str) or response.status_code != 200:
            out["truth_archive"][model] = {"error": response if isinstance(response, str) else f"{response.status_code} {response.text[:90]}"}
        else:
            frame = pd.DataFrame(response.json()["hourly"])
            out["truth_archive"][model] = {c: round(float(frame[c].notna().mean()), 2)
                                           for c in frame.columns if c != "time"}
        time.sleep(1)
    return out


def probe_rain_truth(client) -> dict:
    out: dict = {}
    targets = {
        "nasa_power_daily": ("https://power.larc.nasa.gov/api/temporal/daily/point", {
            "parameters": "PRECTOTCORR", "community": "AG", "longitude": 77.21, "latitude": 28.61,
            "start": "20240601", "end": "20240630", "format": "JSON"}),
        "chirps_daily_p05": ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/2024/chirps-v2.0.2024.06.01.tif.gz", None),
        "imd_gridded_page": ("https://imdpune.gov.in/lrfindex.php", None),
        "ncei_isd_delhi": ("https://www.ncei.noaa.gov/data/global-hourly/access/2024/42182099999.csv", None),
    }
    for name, (url, params) in targets.items():
        try:
            if name == "chirps_daily_p05":
                response = client.head(url, headers=UA, timeout=60, follow_redirects=True)
            else:
                response = client.get(url, params=params, headers=UA, timeout=60, follow_redirects=True)
            info = {"status": response.status_code,
                    "bytes": int(response.headers.get("content-length", len(response.content) if name != "chirps_daily_p05" else 0))}
            if name == "nasa_power_daily" and response.status_code == 200:
                values = response.json()["properties"]["parameter"]["PRECTOTCORR"]
                info["days"] = len(values)
                info["wet_days_>1mm"] = sum(1 for v in values.values() if v > 1)
            out[name] = info
        except Exception as exc:
            out[name] = {"error": repr(exc)[:120]}
    return out


def main():
    import httpx

    with httpx.Client() as client:
        report = {"iem_metar_india": probe_iem(client),
                  "open_meteo": probe_openmeteo(client),
                  "rain_truth_sources": probe_rain_truth(client)}
    # keep the log readable: per-station detail separately, headline first
    per_station = report["iem_metar_india"].pop("per_station_2024", [])
    print("PROBE_REPORT_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("--- per-station (2024) ---")
    for row in sorted(per_station, key=lambda r: -r.get("rows_2024", 0))[:60]:
        print(row)
    print("PROBE_REPORT_END")


if __name__ == "__main__":
    main()
