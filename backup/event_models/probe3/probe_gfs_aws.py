"""Probe 3: can we pull GFS 0.25 deg forecasts for whole India straight from NOAA's
AWS archive (no API weighting), byte-range-subset the GRIB messages we need,
decode them, and at what throughput?

Logs only.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import subprocess
import sys
import time

UA = {"User-Agent": "WeatherGPT-research/1.0 (SIH 2026; academic evaluation)"}
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
WANT = {
    "t2m": ("TMP", "2 m above ground"), "rh2m": ("RH", "2 m above ground"),
    "dpt2m": ("DPT", "2 m above ground"), "u10": ("UGRD", "10 m above ground"),
    "v10": ("VGRD", "10 m above ground"), "gust": ("GUST", "surface"),
    "cape": ("CAPE", "surface"), "cin": ("CIN", "surface"), "vis": ("VIS", "surface"),
    "tcc": ("TCDC", "entire atmosphere"), "pwat": ("PWAT", "entire atmosphere"),
    "mslp": ("PRMSL", "mean sea level"), "hpbl": ("HPBL", "surface"),
    "lftx": ("LFTX", "surface"), "apcp": ("APCP", "surface"),
}


def s3_prefixes(client, base, prefix, max_keys=1000, token=None):
    params = {"list-type": 2, "prefix": prefix, "delimiter": "/", "max-keys": max_keys}
    if token:
        params["continuation-token"] = token
    response = client.get(base + "/", params=params, headers=UA, timeout=60)
    text = response.text
    found = re.findall(r"<Prefix>([^<]+)</Prefix>", text)
    nxt = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
    return [p for p in found if p != prefix], (nxt.group(1) if nxt else None), response.status_code


def probe_archive_range(client) -> dict:
    out = {}
    first, token, status = s3_prefixes(client, GFS, "gfs.", max_keys=1000)
    out["status"] = status
    out["first_prefixes"] = first[:3]
    count = len(first)
    last = first[-1] if first else None
    pages = 0
    while token and pages < 12:
        page, token, _ = s3_prefixes(client, GFS, "gfs.", max_keys=1000, token=token)
        count += len(page)
        last = page[-1] if page else last
        pages += 1
    out["n_date_prefixes_seen"] = count
    out["last_prefix"] = last
    out["more_pages_unread"] = bool(token)
    return out


def idx_lines(client, date, cycle, fhour):
    url = f"{GFS}/gfs.{date}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{fhour:03d}.idx"
    response = client.get(url, headers=UA, timeout=60)
    if response.status_code != 200:
        return None, response.status_code
    rows = []
    for line in response.text.strip().splitlines():
        parts = line.split(":")
        rows.append({"n": int(parts[0]), "offset": int(parts[1]), "var": parts[3],
                     "level": parts[4], "fcst": parts[5]})
    return rows, 200


def pick_messages(rows):
    """Return {name: (start, end_or_None, fcst_desc)} for WANT, shortest accumulation for APCP."""
    picked = {}
    for name, (var, level) in WANT.items():
        candidates = [i for i, r in enumerate(rows) if r["var"] == var and r["level"] == level]
        if not candidates:
            continue
        if name == "apcp":
            def width(i):
                m = re.match(r"(\d+)-(\d+) hour acc", rows[i]["fcst"])
                return (int(m.group(2)) - int(m.group(1))) if m else 10_000
            candidates.sort(key=width)
        i = candidates[0]
        end = rows[i + 1]["offset"] - 1 if i + 1 < len(rows) else None
        picked[name] = (rows[i]["offset"], end, rows[i]["fcst"])
    return picked


def fetch_range(client, url, start, end):
    headers = dict(UA)
    headers["Range"] = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
    response = client.get(url, headers=headers, timeout=120)
    return response.content


def decode(blob):
    import eccodes
    gid = eccodes.codes_new_from_message(blob)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        values = eccodes.codes_get_values(gid).reshape(nj, ni)
        meta = {"lat0": eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"),
                "lon0": eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"),
                "di": eccodes.codes_get(gid, "iDirectionIncrementInDegrees"),
                "dj": eccodes.codes_get(gid, "jDirectionIncrementInDegrees"),
                "short": eccodes.codes_get(gid, "shortName"), "units": eccodes.codes_get(gid, "units")}
        return values, meta
    finally:
        eccodes.codes_release(gid)


def main():
    import httpx
    import numpy as np

    try:
        import eccodes  # noqa: F401
        eccodes_ok = True
    except Exception as exc:
        print("eccodes import failed, installing:", repr(exc)[:100])
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "eccodes"], check=False)
        try:
            import eccodes  # noqa: F401
            eccodes_ok = True
        except Exception as exc2:
            eccodes_ok = False
            print("eccodes still unavailable:", repr(exc2)[:200])

    report = {"eccodes_ok": eccodes_ok}
    with httpx.Client(limits=httpx.Limits(max_connections=64)) as client:
        report["archive_range"] = probe_archive_range(client)

        report["dates"] = {}
        for date in ("20210601", "20220601", "20230601", "20250601"):
            rows, status = idx_lines(client, date, "00", 24)
            if rows is None:
                report["dates"][date] = {"status": status}
                continue
            picked = pick_messages(rows)
            report["dates"][date] = {
                "n_messages": len(rows), "have": sorted(picked), "missing": sorted(set(WANT) - set(picked)),
                "apcp_f24": picked.get("apcp", (0, 0, None))[2]}
        rows3, _ = idx_lines(client, "20250601", "00", 3)
        rows6, _ = idx_lines(client, "20250601", "00", 6)
        report["apcp_lines"] = {
            "f003": [r["fcst"] for r in rows3 if r["var"] == "APCP"] if rows3 else None,
            "f006": [r["fcst"] for r in rows6 if r["var"] == "APCP"] if rows6 else None}
        if rows6:
            report["f006_var_level_sample"] = sorted({f"{r['var']}:{r['level']}" for r in rows6
                                                      if r["var"] in ("CAPE", "CIN", "LFTX", "4LFTX", "VIS", "HPBL",
                                                                      "PWAT", "TCDC", "GUST", "REFC", "CWAT")})

        # throughput + decode test: one date, 12 lead hours x all wanted fields
        if eccodes_ok:
            date, cycle = "20250601", "00"
            jobs = []
            decode_seconds = []
            started = time.time()
            total_bytes = 0
            sample_points = None
            leads = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
            idx_cache = {}
            with cf.ThreadPoolExecutor(16) as pool:
                futs = {pool.submit(idx_lines, client, date, cycle, f): f for f in leads}
                for fut in cf.as_completed(futs):
                    idx_cache[futs[fut]] = fut.result()[0]
            for f in leads:
                picked = pick_messages(idx_cache[f])
                url = f"{GFS}/gfs.{date}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{f:03d}"
                for name, (start, end, desc) in picked.items():
                    jobs.append((f, name, url, start, end))
            with cf.ThreadPoolExecutor(32) as pool:
                blobs = list(pool.map(lambda j: (j, fetch_range(client, j[2], j[3], j[4])), jobs))
            fetch_seconds = time.time() - started
            total_bytes = sum(len(b) for _, b in blobs)
            report["throughput"] = {"messages": len(blobs), "mb": round(total_bytes / 1e6, 1),
                                    "seconds": round(fetch_seconds, 1),
                                    "mb_per_s": round(total_bytes / 1e6 / max(fetch_seconds, 1e-9), 1)}
            sample = {}
            for (f, name, *_), blob in blobs:
                t0 = time.time()
                try:
                    values, meta = decode(blob)
                except Exception as exc:
                    sample[f"{name}@{f}"] = f"decode error {repr(exc)[:80]}"
                    continue
                decode_seconds.append(time.time() - t0)
                if f == 24 or (name in ("cape", "vis") and f in (24, 48)):
                    lat = meta["lat0"] - np.arange(values.shape[0]) * meta["dj"]
                    lon = meta["lon0"] + np.arange(values.shape[1]) * meta["di"]
                    i = int(np.argmin(np.abs(lat - 28.61)))
                    j = int(np.argmin(np.abs(lon - 77.21)))
                    india = values[(lat <= 38) & (lat >= 6)][:, (lon >= 67) & (lon <= 98)]
                    sample[f"{name}@f{f}"] = {"delhi": round(float(values[i, j]), 3), "units": meta["units"],
                                              "shape": list(values.shape), "india_window": list(india.shape),
                                              "lat0": meta["lat0"], "lon0": meta["lon0"], "d": meta["di"]}
            report["decode"] = {"n": len(decode_seconds),
                                "mean_ms": round(1000 * float(np.mean(decode_seconds)), 1) if decode_seconds else None,
                                "samples": sample}
    print("PROBE3_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("PROBE3_END")


if __name__ == "__main__":
    main()
