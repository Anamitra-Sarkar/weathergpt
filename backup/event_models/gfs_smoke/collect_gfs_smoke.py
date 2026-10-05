from __future__ import annotations
import os as _os
_os.environ['SHARD_DATES'] = '2021-06-02,2025-06-10'
_os.environ['GFS_LEADSET'] = 'long'
import sys as _sys, types as _types
_pkg = _types.ModuleType('event_models'); _pkg.__path__ = []; _sys.modules['event_models'] = _pkg
def _load(name, src):
    m = _types.ModuleType('event_models.' + name); _sys.modules['event_models.' + name] = m
    setattr(_pkg, name, m); exec(compile(src, 'event_models/' + name + '.py', 'exec'), m.__dict__)
"""Collect GFS 0.25 deg forecasts for the whole India window from NOAA's AWS archive.

No API weighting or rate limit: each GRIB2 message we need is fetched by HTTP
byte range (via the `.idx` sidecar), decoded with eccodes, and cut to the India
window (lat 6-38, lon 67-98 = 129 x 125 nodes).  From one decoded run we write

  p1_station_steps   (station, run, lead) rows: every forecast step at every METAR
                     station, bilinear-interpolated -> the hourly-event models
  p2_point_days      (point, run, day_k) rows: per-forecast-day aggregates at every
                     station AND every 0.5 deg land node of the India box -> the
                     daily / range models and the whole-country rain model
  points             static table: id, lat, lon, GFS orography, distance to coast

Domain = the South Asia box clipped to land, not India's legal outline: it spares
us drawing a border and gives border regions neighbouring context.  Region
hold-outs are assigned by lat/lon zone later.

Config comes from environment variables (the bundler bakes them per shard):
  SHARD_START / SHARD_END   inclusive run dates, default one smoke day
  SHARD_DATES               comma list overriding the range (smoke tests)
  GFS_CYCLE                 default "00"
"""

import concurrent.futures as cf
import io
import json
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working/gfs") if Path("/kaggle").exists() else Path("gfs_out")
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
IEM = "https://mesonet.agron.iastate.edu"
UA = {"User-Agent": "WeatherGPT-research/1.0 (SIH 2026; academic evaluation)"}
CYCLE = os.environ.get("GFS_CYCLE", "00")

# Lead sets (hours after the 00Z run).  short = what the first collection used; long = days 5-9; all = both.
#   3-hourly to 72 h, then 6-hourly to 240 h (GFS and GEFS both publish to 240 h at these steps)
_SHORT = list(range(3, 73, 3)) + list(range(78, 121, 6))          # 32 steps -> forecast days 0..4
_LONG = list(range(126, 241, 6))                                   # 20 steps -> forecast days 5..9
LEADSET = os.environ.get("GFS_LEADSET", "short")
LEADS = {"short": _SHORT, "long": _LONG, "all": _SHORT + _LONG}[LEADSET]
DAY_STEPS = {k: [h for h in LEADS if 24 * k < h <= 24 * k + 24] for k in range(10)}
DAY_STEPS = {k: v for k, v in DAY_STEPS.items() if v}              # only days that have steps in this lead set

FIELDS = {
    "t2m": ("TMP", "2 m above ground"), "rh2m": ("RH", "2 m above ground"),
    "dpt2m": ("DPT", "2 m above ground"), "u10": ("UGRD", "10 m above ground"),
    "v10": ("VGRD", "10 m above ground"), "gust": ("GUST", "surface"),
    "cape": ("CAPE", "surface"), "cape_ml": ("CAPE", "180-0 mb above ground"),
    "cin": ("CIN", "surface"), "lftx": ("LFTX", "surface"), "vis": ("VIS", "surface"),
    "tcc": ("TCDC", "entire atmosphere"),
    "pwat": ("PWAT", "entire atmosphere (considered as a single layer)"),
    "cwat": ("CWAT", "entire atmosphere (considered as a single layer)"),
    "mslp": ("PRMSL", "mean sea level"), "hpbl": ("HPBL", "surface"),
    "refc": ("REFC", "entire atmosphere"), "apcp": ("APCP", "surface"),
}
STATIC = {"hgt": ("HGT", "surface"), "land": ("LAND", "surface")}

# India window on the 0.25 deg global grid (lat 90 -> -90, lon 0 -> 359.75)
R0, R1, C0, C1 = 208, 337, 268, 393            # rows 38N..6N, cols 67E..98E
LAT_TOP, LON_LEFT, STEP = 38.0, 67.0, 0.25
NR, NC = R1 - R0, C1 - C0                     # 129 x 125


# ------------------------------------------------------------------ worker side
_client = None


def _init_worker():
    global _client
    import httpx
    _client = httpx.Client(timeout=90, limits=httpx.Limits(max_connections=4), headers=UA)


def _request(url, headers=None, attempts=5):
    last = None
    for attempt in range(attempts):
        try:
            response = _client.get(url, headers=headers)
            if response.status_code in (200, 206):
                return response
            if response.status_code in (403, 404):
                return None  # absent file: not a transient error
            last = f"HTTP {response.status_code}"
        except Exception as exc:
            last = repr(exc)[:80]
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def idx_task(args):
    run_date, lead = args
    ymd = f"{run_date:%Y%m%d}"
    url = f"{GFS}/gfs.{ymd}/{CYCLE}/atmos/gfs.t{CYCLE}z.pgrb2.0p25.f{lead:03d}"
    response = _request(url + ".idx")
    if response is None:
        return lead, None
    rows = []
    for line in response.text.strip().splitlines():
        p = line.split(":")
        rows.append((int(p[1]), p[3], p[4], p[5]))
    return lead, (url, rows)


WINDOWED = ("apcp", "tmax2m", "tmin2m")        # fields published as accumulations / window max / window min


def window_hours(desc: str):
    m = re.match(r"(\d+)-(\d+) hour (acc|max|min)", desc)
    return (int(m.group(2)) - int(m.group(1))) if m else None


def pick(rows, want):
    """name -> (offset, end|None, fcst_desc); instantaneous preferred, narrowest window for WINDOWED names."""
    picked = {}
    for name, (var, level) in want.items():
        cand = [i for i, r in enumerate(rows) if r[1] == var and r[2] == level]
        if not cand:
            continue
        if name in WINDOWED:
            cand.sort(key=lambda i: window_hours(rows[i][3]) or 10_000)
        else:
            inst = [i for i in cand if re.match(r"^\d+ hour fcst$", rows[i][3])]
            cand = inst or cand
        i = cand[0]
        end = rows[i + 1][0] - 1 if i + 1 < len(rows) else None
        picked[name] = (rows[i][0], end, rows[i][3])
    return picked


def msg_task(args):
    """Fetch + decode one GRIB message by byte range.

    A ranged download can come back short or misaligned (seen once in ~600k messages: "No final 7777").
    The message is only trusted if it is framed GRIB...7777 and, when the end offset is known, exactly
    the requested length; otherwise retry, and after that report it missing -- one bad message must
    not kill a shard that has hours of good data.
    """
    lead, name, url, start, end, desc = args
    import eccodes
    headers = {"Range": f"bytes={start}-{end}" if end is not None else f"bytes={start}-"}
    want = None if end is None else end - start + 1
    last = "no attempt"
    for attempt in range(4):
        response = _request(url, headers)
        if response is None:
            return lead, name, None, "absent"
        blob = response.content
        framed = blob[:4] == b"GRIB" and blob[-4:] == b"7777" and (want is None or len(blob) == want)
        if framed:
            try:
                gid = eccodes.codes_new_from_message(blob)
                try:
                    ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
                    if (ni, nj) != (1440, 721):
                        raise RuntimeError(f"unexpected grid {ni}x{nj}")
                    values = eccodes.codes_get_values(gid).reshape(nj, ni)
                finally:
                    eccodes.codes_release(gid)
                window = values[R0:R1, C0:C1].astype("float32")
                window[window > 1e19] = np.nan  # GRIB missing value
                return lead, name, window, desc
            except Exception as exc:
                last = f"decode {exc!r}"[:80]
        else:
            last = f"framing len={len(blob)} want={want}"
        time.sleep(1.0 * (attempt + 1))
    return lead, name, None, f"bad:{last}"


# ------------------------------------------------------------------ point sets
def station_points() -> pd.DataFrame:
    import httpx
    with httpx.Client(headers=UA, timeout=60) as client:
        feats = client.get(f"{IEM}/geojson/network/IN__ASOS.geojson").json()["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        begin = (p.get("archive_begin") or "9999")[:4]
        end = (p.get("archive_end") or "9999")[:4]
        if begin == "9999" or end < "2021":
            continue
        if LON_LEFT <= lon <= LON_LEFT + (NC - 1) * STEP and LAT_TOP - (NR - 1) * STEP <= lat <= LAT_TOP:
            rows.append({"point_id": f"S:{p['sid']}", "kind": "station", "lat": lat, "lon": lon,
                         "name": p.get("sname"), "state": p.get("state"),
                         "station_elev_m": p.get("elevation")})
    return pd.DataFrame(rows)


def bilinear_weights(lat, lon):
    r = (LAT_TOP - np.asarray(lat, float)) / STEP
    c = (np.asarray(lon, float) - LON_LEFT) / STEP
    r0 = np.clip(np.floor(r).astype(int), 0, NR - 2)
    c0 = np.clip(np.floor(c).astype(int), 0, NC - 2)
    fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
    return r0, c0, (1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc


class Sampler:
    """Bilinear sampling at fixed points.  NaN neighbours (masked sentinels, undefined cells) are skipped and the
    remaining weights renormalised, so one undefined node cannot poison a point; all-NaN gives NaN."""

    def __init__(self, lat, lon):
        self.r0, self.c0, self.w00, self.w01, self.w10, self.w11 = bilinear_weights(lat, lon)

    def __call__(self, arr):
        r0, c0 = self.r0, self.c0
        num, den = 0.0, 0.0
        for v, w in ((arr[r0, c0], self.w00), (arr[r0, c0 + 1], self.w01),
                     (arr[r0 + 1, c0], self.w10), (arr[r0 + 1, c0 + 1], self.w11)):
            ok = np.isfinite(v)
            num = num + np.where(ok, np.nan_to_num(v) * w, 0.0)
            den = den + np.where(ok, w, 0.0)
        return np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)


def build_points(land: np.ndarray, hgt: np.ndarray) -> pd.DataFrame:
    from scipy.ndimage import distance_transform_edt

    stations = station_points()
    lats = LAT_TOP - STEP * np.arange(NR)
    lons = LON_LEFT + STEP * np.arange(NC)
    # 0.5 deg nodes = every 2nd GFS node, which sit exactly on the 0.25 deg lattice
    node_rows = np.arange(0, NR, 2)
    node_cols = np.arange(0, NC, 2)
    rr, cc = np.meshgrid(node_rows, node_cols, indexing="ij")
    is_land = land[rr, cc] >= 0.5
    nodes = pd.DataFrame({"point_id": [f"G:{lats[r]:.2f}:{lons[c]:.2f}" for r, c in zip(rr[is_land], cc[is_land])],
                          "kind": "node", "lat": lats[rr[is_land]], "lon": lons[cc[is_land]]})
    pts = pd.concat([nodes, stations], ignore_index=True)
    coast_px = distance_transform_edt(land >= 0.5)           # land pixel -> nearest sea, in 0.25 deg steps
    sea_px = distance_transform_edt(land < 0.5)
    sampler = Sampler(pts["lat"].to_numpy(), pts["lon"].to_numpy())
    pts["elevation_m"] = sampler(np.nan_to_num(hgt)).astype("float32")
    pts["land_frac"] = sampler(land.astype("float32")).astype("float32")
    pts["dist_coast_km"] = (np.where(sampler(land) >= 0.5, sampler(coast_px), sampler(sea_px))
                            * STEP * 111.0).astype("float32")
    return pts


# ------------------------------------------------------------------ per-run work
def fetch_run(pool, run_date, with_static=False):
    """-> {lead: {field: window}} plus {'static': {...}} ; None when the run is absent."""
    idx_results = dict(pool.map(idx_task, [(run_date, lead) for lead in LEADS]))
    if all(v is None for v in idx_results.values()):
        return None, {"missing_steps": len(LEADS)}
    tasks, desc = [], {}
    for lead, found in idx_results.items():
        if found is None:
            continue
        url, rows = found
        want = dict(FIELDS)
        if with_static and lead == LEADS[0]:
            want.update(STATIC)
        for name, (start, end, d) in pick(rows, want).items():
            tasks.append((lead, name, url, start, end, d))
    arrays: dict = {}
    bad = []
    for lead, name, window, d in pool.map(msg_task, tasks, chunksize=4):
        if window is None:
            if d.startswith("bad:"):
                bad.append(f"f{lead:03d}:{name}")
            continue
        arrays.setdefault(lead, {})[name] = window
        if name == "apcp":
            m = re.match(r"(\d+)-(\d+) hour acc", d)
            arrays[lead]["apcp_width"] = (int(m.group(2)) - int(m.group(1))) if m else 0
    stats = {"missing_steps": sum(1 for v in idx_results.values() if v is None),
             "messages": len(tasks), "bad": bad}
    return arrays, stats


def derive(fields: dict) -> dict:
    """Raw GFS units -> the units the models use; adds wind speed and areal-mean rain."""
    from scipy.ndimage import maximum_filter, uniform_filter

    out = {}
    if "t2m" in fields:
        out["t2m_c"] = fields["t2m"] - 273.15
    if "dpt2m" in fields:
        out["dpt2m_c"] = fields["dpt2m"] - 273.15
    for name in ("rh2m", "gust", "cape", "cape_ml", "cin", "lftx", "tcc", "pwat", "cwat", "hpbl", "refc"):
        if name in fields:
            out[name] = fields[name]
    if "vis" in fields:
        out["vis_km"] = fields["vis"] / 1000.0
    if "mslp" in fields:
        out["mslp_hpa"] = fields["mslp"] / 100.0
    if "u10" in fields and "v10" in fields:
        out["wind10"] = np.hypot(fields["u10"], fields["v10"])
    if "apcp" in fields:
        out["apcp_mm"] = fields["apcp"]
        out["apcp_cell_mean"] = uniform_filter(np.nan_to_num(fields["apcp"]), size=3, mode="nearest")
        out["apcp_cell_max"] = maximum_filter(np.nan_to_num(fields["apcp"]), size=3, mode="nearest")
    return out


def station_step_rows(run_date, arrays, stn_idx, stn_ids, stn_sampler):
    frames = []
    for lead in sorted(arrays):
        derived = derive(arrays[lead])
        data = {"point_id": stn_ids, "run_date": np.datetime64(run_date), "lead_h": np.int16(lead)}
        for name, arr in derived.items():
            data[name] = stn_sampler(arr).astype("float32")
        data["apcp_width_h"] = np.int8(arrays[lead].get("apcp_width", 0))
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


DAILY = {  # name: (source column, reducer)
    "t2m_max": ("t2m_c", np.nanmax), "t2m_min": ("t2m_c", np.nanmin), "t2m_mean": ("t2m_c", np.nanmean),
    "rh_mean": ("rh2m", np.nanmean), "rh_min": ("rh2m", np.nanmin), "dpt_mean": ("dpt2m_c", np.nanmean),
    "wind_mean": ("wind10", np.nanmean), "gust_max": ("gust", np.nanmax),
    "cape_max": ("cape", np.nanmax), "cape_ml_max": ("cape_ml", np.nanmax), "cin_mean": ("cin", np.nanmean),
    "lftx_min": ("lftx", np.nanmin), "vis_min_km": ("vis_km", np.nanmin), "tcc_mean": ("tcc", np.nanmean),
    "pwat_mean": ("pwat", np.nanmean), "pwat_max": ("pwat", np.nanmax), "mslp_mean": ("mslp_hpa", np.nanmean),
    "hpbl_max": ("hpbl", np.nanmax), "refc_max": ("refc", np.nanmax), "cwat_max": ("cwat", np.nanmax),
}


def point_day_rows(run_date, arrays, point_ids, sampler):
    frames = []
    sampled = {}  # lead -> {col: values at all points}
    for lead in arrays:
        sampled[lead] = {k: sampler(v).astype("float32") for k, v in derive(arrays[lead]).items()}
        sampled[lead]["_width"] = arrays[lead].get("apcp_width", 0)
    for k, steps in DAY_STEPS.items():
        have = [h for h in steps if h in sampled]
        if not have:
            continue
        row = {"point_id": point_ids, "run_date": np.datetime64(run_date), "day_k": np.int8(k),
               "n_steps": np.int8(len(have))}
        with np.errstate(all="ignore"):
            for name, (col, reducer) in DAILY.items():
                stack = [sampled[h][col] for h in have if col in sampled[h]]
                row[name] = reducer(np.stack(stack), axis=0).astype("float32") if stack else np.full(len(point_ids), np.nan, "float32")
            # rain: sum of the 6-hour buckets only (lead % 6 == 0, window width 6)
            buckets = [h for h in have if h % 6 == 0 and sampled[h].get("_width") == 6 and "apcp_mm" in sampled[h]]
            row["rain_buckets"] = np.int8(len(buckets))
            for out_name, col in (("rain_mm", "apcp_mm"), ("rain_cell_mean_mm", "apcp_cell_mean"),
                                  ("rain_cell_max_mm", "apcp_cell_max")):
                row[out_name] = (np.sum([sampled[h][col] for h in buckets], axis=0).astype("float32")
                                 if buckets else np.full(len(point_ids), np.nan, "float32"))
            row["rain_max6_mm"] = (np.max([sampled[h]["apcp_mm"] for h in buckets], axis=0).astype("float32")
                                   if buckets else np.full(len(point_ids), np.nan, "float32"))
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------ driver
def run_dates():
    """Run dates for this shard.  SHARD_STRIDE=2 keeps every second day (by ordinal parity, so two shards with
    SHARD_OFFSET 0 and 1 partition the archive exactly); consecutive daily runs share ~90% of the atmosphere."""
    explicit = os.environ.get("SHARD_DATES")
    if explicit:
        return [date.fromisoformat(d) for d in explicit.split(",")]
    start = date.fromisoformat(os.environ.get("SHARD_START", "2021-01-01"))
    end = date.fromisoformat(os.environ.get("SHARD_END", "2021-01-03"))
    stride, offset = int(os.environ.get("SHARD_STRIDE", "1")), int(os.environ.get("SHARD_OFFSET", "0"))
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return [d for d in days if d.toordinal() % stride == offset]


def ensure_eccodes():
    """Kaggle's image has no eccodes; install the wheel (it bundles libeccodes) before forking workers."""
    try:
        import eccodes  # noqa: F401
    except ImportError:
        import importlib
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "eccodes"], check=True)
        importlib.invalidate_caches()
        import eccodes  # noqa: F401


def main():
    ensure_eccodes()
    OUT.mkdir(parents=True, exist_ok=True)
    dates = run_dates()
    ctx = mp.get_context("fork")
    started = time.time()
    report = {"dates": len(dates), "cycle": CYCLE, "leads": len(LEADS), "absent_runs": [], "partial_runs": {},
              "bad_messages": {}}
    with cf.ProcessPoolExecutor(max_workers=12, mp_context=ctx, initializer=_init_worker) as pool:
        # static fields (orography, land-sea mask) from the first date that exists
        points = None
        for d in dates:
            arrays, stats = fetch_run(pool, d, with_static=True)
            if arrays is not None and LEADS[0] in arrays and "hgt" in arrays[LEADS[0]] and "land" in arrays[LEADS[0]]:
                points = build_points(arrays[LEADS[0]]["land"], arrays[LEADS[0]]["hgt"])
                first_arrays, first_date = arrays, d
                break
        if points is None:
            raise RuntimeError("could not obtain orography/land mask from any requested date")
        points.to_parquet(OUT / "points.parquet", index=False)
        report["points"] = {"total": int(len(points)), "nodes": int((points["kind"] == "node").sum()),
                            "stations": int((points["kind"] == "station").sum())}
        print("[gfs] points:", report["points"], flush=True)

        point_ids = points["point_id"].to_numpy()
        sampler = Sampler(points["lat"].to_numpy(), points["lon"].to_numpy())
        stn = points[points["kind"] == "station"]
        stn_ids = stn["point_id"].to_numpy()
        stn_sampler = Sampler(stn["lat"].to_numpy(), stn["lon"].to_numpy())

        p1_parts, p2_parts, month_key, written = [], [], None, []

        def flush(key):
            if not p2_parts:
                return
            name1, name2 = OUT / f"p1_station_steps_{key}.parquet", OUT / f"p2_point_days_{key}.parquet"
            pd.concat(p1_parts, ignore_index=True).to_parquet(name1, index=False)
            pd.concat(p2_parts, ignore_index=True).to_parquet(name2, index=False)
            written.append(key)
            p1_parts.clear()
            p2_parts.clear()

        for n, d in enumerate(dates):
            key = f"{d:%Y%m}"
            if month_key is not None and key != month_key:
                flush(month_key)
            month_key = key
            if d == first_date:
                arrays, stats = first_arrays, {"missing_steps": 0}
            else:
                arrays, stats = fetch_run(pool, d)
            if arrays is None:
                report["absent_runs"].append(d.isoformat())
                continue
            if stats["missing_steps"]:
                report["partial_runs"][d.isoformat()] = stats["missing_steps"]
            if stats.get("bad"):
                report["bad_messages"][d.isoformat()] = stats["bad"]
                print(f"[gfs] {d}: {len(stats['bad'])} message(s) unreadable after retries: {stats['bad'][:5]}", flush=True)
            p1_parts.append(station_step_rows(d, arrays, None, stn_ids, stn_sampler))
            p2_parts.append(point_day_rows(d, arrays, point_ids, sampler))
            if (n + 1) % 5 == 0 or n == len(dates) - 1:
                elapsed = time.time() - started
                print(f"[gfs] {n + 1}/{len(dates)} dates  {elapsed / (n + 1):.1f}s/date  "
                      f"absent={len(report['absent_runs'])} partial={len(report['partial_runs'])}", flush=True)
        if month_key is not None:
            flush(month_key)
    report["months_written"] = written
    report["seconds"] = round(time.time() - started)
    (OUT / "gfs_report.json").write_text(json.dumps(report, indent=1))
    print("GFS_REPORT_BEGIN")
    print(json.dumps(report, indent=1))
    print("GFS_REPORT_END")


if __name__ == "__main__":
    main()
