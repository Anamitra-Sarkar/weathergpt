"""Probe 4: GEFS ensemble mean/spread on AWS (rain-probability predictor) + GFS pressure-level fields.

Questions answered with real requests:
  1. Does `noaa-gefs-pds` hold ensemble MEAN (geavg) and SPREAD (gespr) back to 2021-04, and in which layout?
  2. Which fields do those files carry (APCP, CAPE, PWAT, TMP, RH, wind ...)?
  3. Do the numbers look physical (decode one APCP mean/spread at a monsoon point)?
  4. Which pressure-level fields does the GFS 0.25 file carry (850 hPa wind/RH, 700 hPa omega ...)?
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time

UA = {"User-Agent": "WeatherGPT-research/1.0 (SIH 2026; academic evaluation)"}
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def lst(client, base, prefix, delimiter="/", max_keys=200):
    r = client.get(base + "/", params={"list-type": 2, "prefix": prefix, "delimiter": delimiter, "max-keys": max_keys},
                   headers=UA, timeout=60)
    return {"status": r.status_code,
            "prefixes": [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", r.text) if p != prefix],
            "keys": re.findall(r"<Key>([^<]+)</Key>", r.text)}


def idx(client, url):
    r = client.get(url + ".idx", headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    out = []
    for line in r.text.strip().splitlines():
        p = line.split(":")
        out.append((int(p[1]), p[3], p[4], p[5]))
    return out


def main():
    import httpx
    try:
        import eccodes  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "eccodes"], check=False)
    import eccodes
    import numpy as np

    report = {}
    with httpx.Client() as c:
        # 1 archive layout and range
        top = lst(c, GEFS, "gefs.", max_keys=1000)
        report["gefs_first_prefixes"] = top["prefixes"][:3]
        report["gefs_n_prefixes_page1"] = len(top["prefixes"])
        report["gefs_layout"] = {}
        for date in ("20210401", "20220601", "20250601"):
            entry = {}
            for cyc in ("00",):
                sub = lst(c, GEFS, f"gefs.{date}/{cyc}/", max_keys=50)
                entry["cycle_subdirs"] = sub["prefixes"][:12]
                atmos = lst(c, GEFS, f"gefs.{date}/{cyc}/atmos/", max_keys=50)
                entry["atmos_subdirs"] = atmos["prefixes"][:12]
                sp = lst(c, GEFS, f"gefs.{date}/{cyc}/atmos/pgrb2sp25/", delimiter="", max_keys=12)
                entry["pgrb2sp25_sample_keys"] = sp["keys"][:8]
                ap = lst(c, GEFS, f"gefs.{date}/{cyc}/atmos/pgrb2ap5/", delimiter="", max_keys=12)
                entry["pgrb2ap5_sample_keys"] = ap["keys"][:6]
            report["gefs_layout"][date] = entry

        # 2 fields in mean / spread files
        report["gefs_fields"] = {}
        for date in ("20210601", "20250601"):
            for kind in ("geavg", "gespr"):
                url = f"{GEFS}/gefs.{date}/00/atmos/pgrb2sp25/{kind}.t00z.pgrb2s.0p25.f024"
                rows = idx(c, url)
                key = f"{date}/{kind}/sp25/f024"
                if rows is None:
                    report["gefs_fields"][key] = "no idx"
                    continue
                report["gefs_fields"][key] = {"n": len(rows), "fields": sorted({f"{v}:{l}" for _, v, l, _ in rows})[:60]}
        url = f"{GEFS}/gefs.20250601/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f024"
        rows = idx(c, url)
        report["gefs_fields"]["20250601/geavg/ap5/f024"] = (
            {"n": len(rows), "levels_850_700": sorted({f"{v}:{l}" for _, v, l, _ in rows if l in ("850 mb", "700 mb", "500 mb")})[:40]}
            if rows else "no idx")

        # 3 decode APCP mean and spread, compare with deterministic GFS at a monsoon point
        def fetch_field(url, var, level, want_width=True):
            rows = idx(c, url)
            if rows is None:
                return None
            cand = [i for i, r in enumerate(rows) if r[1] == var and r[2] == level]
            if not cand:
                return None
            if var == "APCP":
                def width(i):
                    m = re.match(r"(\d+)-(\d+) hour acc", rows[i][3])
                    return int(m.group(2)) - int(m.group(1)) if m else 10_000
                cand.sort(key=width)
            i = cand[0]
            start = rows[i][0]
            end = rows[i + 1][0] - 1 if i + 1 < len(rows) else None
            h = dict(UA, Range=f"bytes={start}-{end}" if end else f"bytes={start}-")
            blob = c.get(url, headers=h, timeout=60).content
            gid = eccodes.codes_new_from_message(blob)
            ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
            vals = eccodes.codes_get_values(gid).reshape(nj, ni)
            meta = (eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(gid, "iDirectionIncrementInDegrees"), rows[i][3])
            eccodes.codes_release(gid)
            return vals, meta

        sample = {}
        date, lat, lon = "20250715", 19.0, 73.0   # Mumbai side, mid-monsoon
        for name, url in (("gefs_mean", f"{GEFS}/gefs.{date}/00/atmos/pgrb2sp25/geavg.t00z.pgrb2s.0p25.f024"),
                          ("gefs_spread", f"{GEFS}/gefs.{date}/00/atmos/pgrb2sp25/gespr.t00z.pgrb2s.0p25.f024"),
                          ("gfs_det", f"{GFS}/gfs.{date}/00/atmos/gfs.t00z.pgrb2.0p25.f024")):
            try:
                t0 = time.time()
                got = fetch_field(url, "APCP", "surface")
                if got is None:
                    sample[name] = "missing"
                    continue
                vals, (lat0, d, desc) = got
                r = int(round((lat0 - lat) / d)); col = int(round(lon / d))
                sample[name] = {"apcp_mm_at_point": round(float(vals[r, col]), 2), "desc": desc,
                                "shape": list(vals.shape), "seconds": round(time.time() - t0, 2)}
            except Exception as exc:
                sample[name] = f"error {exc!r}"[:160]
        report["monsoon_point_20250715_f024"] = sample

        # 4 GFS pressure-level fields
        try:
            rows = idx(c, f"{GFS}/gfs.20250715/00/atmos/gfs.t00z.pgrb2.0p25.f024")
            wanted = [("UGRD", "850 mb"), ("VGRD", "850 mb"), ("RH", "850 mb"), ("RH", "700 mb"), ("VVEL", "700 mb"),
                      ("TMP", "850 mb"), ("HGT", "500 mb"), ("ABSV", "500 mb"), ("UGRD", "200 mb"), ("VGRD", "200 mb"),
                      ("SPFH", "850 mb"), ("VVEL", "850 mb"), ("VVEL", "500 mb")]
            have = {f"{v}:{l}" for _, v, l, _ in rows}
            report["gfs_pressure_levels"] = {f"{v}:{l}": f"{v}:{l}" in have for v, l in wanted}
        except Exception as exc:
            report["gfs_pressure_levels"] = f"error {exc!r}"[:160]
    print("PROBE4_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("PROBE4_END")


if __name__ == "__main__":
    main()
