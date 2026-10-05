from __future__ import annotations
import os as _os
_os.environ['SHARD_START'] = '2022-08-17'
_os.environ['SHARD_END'] = '2024-01-01'
_os.environ['SHARD_STRIDE'] = '2'
_os.environ['SHARD_OFFSET'] = '0'
_os.environ['GFS_LEADSET'] = 'all'
import sys as _sys, types as _types
_pkg = _types.ModuleType('event_models'); _pkg.__path__ = []; _sys.modules['event_models'] = _pkg
def _load(name, src):
    m = _types.ModuleType('event_models.' + name); _sys.modules['event_models.' + name] = m
    setattr(_pkg, name, m); exec(compile(src, 'event_models/' + name + '.py', 'exec'), m.__dict__)
_load('collect_gfs', '"""Collect GFS 0.25 deg forecasts for the whole India window from NOAA\'s AWS archive.\n\nNo API weighting or rate limit: each GRIB2 message we need is fetched by HTTP\nbyte range (via the `.idx` sidecar), decoded with eccodes, and cut to the India\nwindow (lat 6-38, lon 67-98 = 129 x 125 nodes).  From one decoded run we write\n\n  p1_station_steps   (station, run, lead) rows: every forecast step at every METAR\n                     station, bilinear-interpolated -> the hourly-event models\n  p2_point_days      (point, run, day_k) rows: per-forecast-day aggregates at every\n                     station AND every 0.5 deg land node of the India box -> the\n                     daily / range models and the whole-country rain model\n  points             static table: id, lat, lon, GFS orography, distance to coast\n\nDomain = the South Asia box clipped to land, not India\'s legal outline: it spares\nus drawing a border and gives border regions neighbouring context.  Region\nhold-outs are assigned by lat/lon zone later.\n\nConfig comes from environment variables (the bundler bakes them per shard):\n  SHARD_START / SHARD_END   inclusive run dates, default one smoke day\n  SHARD_DATES               comma list overriding the range (smoke tests)\n  GFS_CYCLE                 default "00"\n"""\nfrom __future__ import annotations\n\nimport concurrent.futures as cf\nimport io\nimport json\nimport multiprocessing as mp\nimport os\nimport re\nimport sys\nimport time\nfrom datetime import date, datetime, timedelta\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nOUT = Path("/kaggle/working/gfs") if Path("/kaggle").exists() else Path("gfs_out")\nGFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"\nIEM = "https://mesonet.agron.iastate.edu"\nUA = {"User-Agent": "WeatherGPT-research/1.0 (SIH 2026; academic evaluation)"}\nCYCLE = os.environ.get("GFS_CYCLE", "00")\n\n# Lead sets (hours after the 00Z run).  short = what the first collection used; long = days 5-9; all = both.\n#   3-hourly to 72 h, then 6-hourly to 240 h (GFS and GEFS both publish to 240 h at these steps)\n_SHORT = list(range(3, 73, 3)) + list(range(78, 121, 6))          # 32 steps -> forecast days 0..4\n_LONG = list(range(126, 241, 6))                                   # 20 steps -> forecast days 5..9\nLEADSET = os.environ.get("GFS_LEADSET", "short")\nLEADS = {"short": _SHORT, "long": _LONG, "all": _SHORT + _LONG}[LEADSET]\nDAY_STEPS = {k: [h for h in LEADS if 24 * k < h <= 24 * k + 24] for k in range(10)}\nDAY_STEPS = {k: v for k, v in DAY_STEPS.items() if v}              # only days that have steps in this lead set\n\nFIELDS = {\n    "t2m": ("TMP", "2 m above ground"), "rh2m": ("RH", "2 m above ground"),\n    "dpt2m": ("DPT", "2 m above ground"), "u10": ("UGRD", "10 m above ground"),\n    "v10": ("VGRD", "10 m above ground"), "gust": ("GUST", "surface"),\n    "cape": ("CAPE", "surface"), "cape_ml": ("CAPE", "180-0 mb above ground"),\n    "cin": ("CIN", "surface"), "lftx": ("LFTX", "surface"), "vis": ("VIS", "surface"),\n    "tcc": ("TCDC", "entire atmosphere"),\n    "pwat": ("PWAT", "entire atmosphere (considered as a single layer)"),\n    "cwat": ("CWAT", "entire atmosphere (considered as a single layer)"),\n    "mslp": ("PRMSL", "mean sea level"), "hpbl": ("HPBL", "surface"),\n    "refc": ("REFC", "entire atmosphere"), "apcp": ("APCP", "surface"),\n}\nSTATIC = {"hgt": ("HGT", "surface"), "land": ("LAND", "surface")}\n\n# India window on the 0.25 deg global grid (lat 90 -> -90, lon 0 -> 359.75)\nR0, R1, C0, C1 = 208, 337, 268, 393            # rows 38N..6N, cols 67E..98E\nLAT_TOP, LON_LEFT, STEP = 38.0, 67.0, 0.25\nNR, NC = R1 - R0, C1 - C0                     # 129 x 125\n\n\n# ------------------------------------------------------------------ worker side\n_client = None\n\n\ndef _init_worker():\n    global _client\n    import httpx\n    _client = httpx.Client(timeout=90, limits=httpx.Limits(max_connections=4), headers=UA)\n\n\ndef _request(url, headers=None, attempts=5):\n    last = None\n    for attempt in range(attempts):\n        try:\n            response = _client.get(url, headers=headers)\n            if response.status_code in (200, 206):\n                return response\n            if response.status_code in (403, 404):\n                return None  # absent file: not a transient error\n            last = f"HTTP {response.status_code}"\n        except Exception as exc:\n            last = repr(exc)[:80]\n        time.sleep(1.5 * (attempt + 1))\n    raise RuntimeError(f"{url}: {last}")\n\n\ndef idx_task(args):\n    run_date, lead = args\n    ymd = f"{run_date:%Y%m%d}"\n    url = f"{GFS}/gfs.{ymd}/{CYCLE}/atmos/gfs.t{CYCLE}z.pgrb2.0p25.f{lead:03d}"\n    response = _request(url + ".idx")\n    if response is None:\n        return lead, None\n    rows = []\n    for line in response.text.strip().splitlines():\n        p = line.split(":")\n        rows.append((int(p[1]), p[3], p[4], p[5]))\n    return lead, (url, rows)\n\n\nWINDOWED = ("apcp", "tmax2m", "tmin2m")        # fields published as accumulations / window max / window min\n\n\ndef window_hours(desc: str):\n    m = re.match(r"(\\d+)-(\\d+) hour (acc|max|min)", desc)\n    return (int(m.group(2)) - int(m.group(1))) if m else None\n\n\ndef pick(rows, want):\n    """name -> (offset, end|None, fcst_desc); instantaneous preferred, narrowest window for WINDOWED names."""\n    picked = {}\n    for name, (var, level) in want.items():\n        cand = [i for i, r in enumerate(rows) if r[1] == var and r[2] == level]\n        if not cand:\n            continue\n        if name in WINDOWED:\n            cand.sort(key=lambda i: window_hours(rows[i][3]) or 10_000)\n        else:\n            inst = [i for i in cand if re.match(r"^\\d+ hour fcst$", rows[i][3])]\n            cand = inst or cand\n        i = cand[0]\n        end = rows[i + 1][0] - 1 if i + 1 < len(rows) else None\n        picked[name] = (rows[i][0], end, rows[i][3])\n    return picked\n\n\ndef msg_task(args):\n    """Fetch + decode one GRIB message by byte range.\n\n    A ranged download can come back short or misaligned (seen once in ~600k messages: "No final 7777").\n    The message is only trusted if it is framed GRIB...7777 and, when the end offset is known, exactly\n    the requested length; otherwise retry, and after that report it missing -- one bad message must\n    not kill a shard that has hours of good data.\n    """\n    lead, name, url, start, end, desc = args\n    import eccodes\n    headers = {"Range": f"bytes={start}-{end}" if end is not None else f"bytes={start}-"}\n    want = None if end is None else end - start + 1\n    last = "no attempt"\n    for attempt in range(4):\n        response = _request(url, headers)\n        if response is None:\n            return lead, name, None, "absent"\n        blob = response.content\n        framed = blob[:4] == b"GRIB" and blob[-4:] == b"7777" and (want is None or len(blob) == want)\n        if framed:\n            try:\n                gid = eccodes.codes_new_from_message(blob)\n                try:\n                    ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")\n                    if (ni, nj) != (1440, 721):\n                        raise RuntimeError(f"unexpected grid {ni}x{nj}")\n                    values = eccodes.codes_get_values(gid).reshape(nj, ni)\n                finally:\n                    eccodes.codes_release(gid)\n                window = values[R0:R1, C0:C1].astype("float32")\n                window[window > 1e19] = np.nan  # GRIB missing value\n                return lead, name, window, desc\n            except Exception as exc:\n                last = f"decode {exc!r}"[:80]\n        else:\n            last = f"framing len={len(blob)} want={want}"\n        time.sleep(1.0 * (attempt + 1))\n    return lead, name, None, f"bad:{last}"\n\n\n# ------------------------------------------------------------------ point sets\ndef station_points() -> pd.DataFrame:\n    import httpx\n    with httpx.Client(headers=UA, timeout=60) as client:\n        feats = client.get(f"{IEM}/geojson/network/IN__ASOS.geojson").json()["features"]\n    rows = []\n    for f in feats:\n        p = f["properties"]\n        lon, lat = f["geometry"]["coordinates"]\n        begin = (p.get("archive_begin") or "9999")[:4]\n        end = (p.get("archive_end") or "9999")[:4]\n        if begin == "9999" or end < "2021":\n            continue\n        if LON_LEFT <= lon <= LON_LEFT + (NC - 1) * STEP and LAT_TOP - (NR - 1) * STEP <= lat <= LAT_TOP:\n            rows.append({"point_id": f"S:{p[\'sid\']}", "kind": "station", "lat": lat, "lon": lon,\n                         "name": p.get("sname"), "state": p.get("state"),\n                         "station_elev_m": p.get("elevation")})\n    return pd.DataFrame(rows)\n\n\ndef bilinear_weights(lat, lon):\n    r = (LAT_TOP - np.asarray(lat, float)) / STEP\n    c = (np.asarray(lon, float) - LON_LEFT) / STEP\n    r0 = np.clip(np.floor(r).astype(int), 0, NR - 2)\n    c0 = np.clip(np.floor(c).astype(int), 0, NC - 2)\n    fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)\n    return r0, c0, (1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc\n\n\nclass Sampler:\n    """Bilinear sampling at fixed points.  NaN neighbours (masked sentinels, undefined cells) are skipped and the\n    remaining weights renormalised, so one undefined node cannot poison a point; all-NaN gives NaN."""\n\n    def __init__(self, lat, lon):\n        self.r0, self.c0, self.w00, self.w01, self.w10, self.w11 = bilinear_weights(lat, lon)\n\n    def __call__(self, arr):\n        r0, c0 = self.r0, self.c0\n        num, den = 0.0, 0.0\n        for v, w in ((arr[r0, c0], self.w00), (arr[r0, c0 + 1], self.w01),\n                     (arr[r0 + 1, c0], self.w10), (arr[r0 + 1, c0 + 1], self.w11)):\n            ok = np.isfinite(v)\n            num = num + np.where(ok, np.nan_to_num(v) * w, 0.0)\n            den = den + np.where(ok, w, 0.0)\n        return np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)\n\n\ndef build_points(land: np.ndarray, hgt: np.ndarray) -> pd.DataFrame:\n    from scipy.ndimage import distance_transform_edt\n\n    stations = station_points()\n    lats = LAT_TOP - STEP * np.arange(NR)\n    lons = LON_LEFT + STEP * np.arange(NC)\n    # 0.5 deg nodes = every 2nd GFS node, which sit exactly on the 0.25 deg lattice\n    node_rows = np.arange(0, NR, 2)\n    node_cols = np.arange(0, NC, 2)\n    rr, cc = np.meshgrid(node_rows, node_cols, indexing="ij")\n    is_land = land[rr, cc] >= 0.5\n    nodes = pd.DataFrame({"point_id": [f"G:{lats[r]:.2f}:{lons[c]:.2f}" for r, c in zip(rr[is_land], cc[is_land])],\n                          "kind": "node", "lat": lats[rr[is_land]], "lon": lons[cc[is_land]]})\n    pts = pd.concat([nodes, stations], ignore_index=True)\n    coast_px = distance_transform_edt(land >= 0.5)           # land pixel -> nearest sea, in 0.25 deg steps\n    sea_px = distance_transform_edt(land < 0.5)\n    sampler = Sampler(pts["lat"].to_numpy(), pts["lon"].to_numpy())\n    pts["elevation_m"] = sampler(np.nan_to_num(hgt)).astype("float32")\n    pts["land_frac"] = sampler(land.astype("float32")).astype("float32")\n    pts["dist_coast_km"] = (np.where(sampler(land) >= 0.5, sampler(coast_px), sampler(sea_px))\n                            * STEP * 111.0).astype("float32")\n    return pts\n\n\n# ------------------------------------------------------------------ per-run work\ndef fetch_run(pool, run_date, with_static=False):\n    """-> {lead: {field: window}} plus {\'static\': {...}} ; None when the run is absent."""\n    idx_results = dict(pool.map(idx_task, [(run_date, lead) for lead in LEADS]))\n    if all(v is None for v in idx_results.values()):\n        return None, {"missing_steps": len(LEADS)}\n    tasks, desc = [], {}\n    for lead, found in idx_results.items():\n        if found is None:\n            continue\n        url, rows = found\n        want = dict(FIELDS)\n        if with_static and lead == LEADS[0]:\n            want.update(STATIC)\n        for name, (start, end, d) in pick(rows, want).items():\n            tasks.append((lead, name, url, start, end, d))\n    arrays: dict = {}\n    bad = []\n    for lead, name, window, d in pool.map(msg_task, tasks, chunksize=4):\n        if window is None:\n            if d.startswith("bad:"):\n                bad.append(f"f{lead:03d}:{name}")\n            continue\n        arrays.setdefault(lead, {})[name] = window\n        if name == "apcp":\n            m = re.match(r"(\\d+)-(\\d+) hour acc", d)\n            arrays[lead]["apcp_width"] = (int(m.group(2)) - int(m.group(1))) if m else 0\n    stats = {"missing_steps": sum(1 for v in idx_results.values() if v is None),\n             "messages": len(tasks), "bad": bad}\n    return arrays, stats\n\n\ndef derive(fields: dict) -> dict:\n    """Raw GFS units -> the units the models use; adds wind speed and areal-mean rain."""\n    from scipy.ndimage import maximum_filter, uniform_filter\n\n    out = {}\n    if "t2m" in fields:\n        out["t2m_c"] = fields["t2m"] - 273.15\n    if "dpt2m" in fields:\n        out["dpt2m_c"] = fields["dpt2m"] - 273.15\n    for name in ("rh2m", "gust", "cape", "cape_ml", "cin", "lftx", "tcc", "pwat", "cwat", "hpbl", "refc"):\n        if name in fields:\n            out[name] = fields[name]\n    if "vis" in fields:\n        out["vis_km"] = fields["vis"] / 1000.0\n    if "mslp" in fields:\n        out["mslp_hpa"] = fields["mslp"] / 100.0\n    if "u10" in fields and "v10" in fields:\n        out["wind10"] = np.hypot(fields["u10"], fields["v10"])\n    if "apcp" in fields:\n        out["apcp_mm"] = fields["apcp"]\n        out["apcp_cell_mean"] = uniform_filter(np.nan_to_num(fields["apcp"]), size=3, mode="nearest")\n        out["apcp_cell_max"] = maximum_filter(np.nan_to_num(fields["apcp"]), size=3, mode="nearest")\n    return out\n\n\ndef station_step_rows(run_date, arrays, stn_idx, stn_ids, stn_sampler):\n    frames = []\n    for lead in sorted(arrays):\n        derived = derive(arrays[lead])\n        data = {"point_id": stn_ids, "run_date": np.datetime64(run_date), "lead_h": np.int16(lead)}\n        for name, arr in derived.items():\n            data[name] = stn_sampler(arr).astype("float32")\n        data["apcp_width_h"] = np.int8(arrays[lead].get("apcp_width", 0))\n        frames.append(pd.DataFrame(data))\n    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n\n\nDAILY = {  # name: (source column, reducer)\n    "t2m_max": ("t2m_c", np.nanmax), "t2m_min": ("t2m_c", np.nanmin), "t2m_mean": ("t2m_c", np.nanmean),\n    "rh_mean": ("rh2m", np.nanmean), "rh_min": ("rh2m", np.nanmin), "dpt_mean": ("dpt2m_c", np.nanmean),\n    "wind_mean": ("wind10", np.nanmean), "gust_max": ("gust", np.nanmax),\n    "cape_max": ("cape", np.nanmax), "cape_ml_max": ("cape_ml", np.nanmax), "cin_mean": ("cin", np.nanmean),\n    "lftx_min": ("lftx", np.nanmin), "vis_min_km": ("vis_km", np.nanmin), "tcc_mean": ("tcc", np.nanmean),\n    "pwat_mean": ("pwat", np.nanmean), "pwat_max": ("pwat", np.nanmax), "mslp_mean": ("mslp_hpa", np.nanmean),\n    "hpbl_max": ("hpbl", np.nanmax), "refc_max": ("refc", np.nanmax), "cwat_max": ("cwat", np.nanmax),\n}\n\n\ndef point_day_rows(run_date, arrays, point_ids, sampler):\n    frames = []\n    sampled = {}  # lead -> {col: values at all points}\n    for lead in arrays:\n        sampled[lead] = {k: sampler(v).astype("float32") for k, v in derive(arrays[lead]).items()}\n        sampled[lead]["_width"] = arrays[lead].get("apcp_width", 0)\n    for k, steps in DAY_STEPS.items():\n        have = [h for h in steps if h in sampled]\n        if not have:\n            continue\n        row = {"point_id": point_ids, "run_date": np.datetime64(run_date), "day_k": np.int8(k),\n               "n_steps": np.int8(len(have))}\n        with np.errstate(all="ignore"):\n            for name, (col, reducer) in DAILY.items():\n                stack = [sampled[h][col] for h in have if col in sampled[h]]\n                row[name] = reducer(np.stack(stack), axis=0).astype("float32") if stack else np.full(len(point_ids), np.nan, "float32")\n            # rain: sum of the 6-hour buckets only (lead % 6 == 0, window width 6)\n            buckets = [h for h in have if h % 6 == 0 and sampled[h].get("_width") == 6 and "apcp_mm" in sampled[h]]\n            row["rain_buckets"] = np.int8(len(buckets))\n            for out_name, col in (("rain_mm", "apcp_mm"), ("rain_cell_mean_mm", "apcp_cell_mean"),\n                                  ("rain_cell_max_mm", "apcp_cell_max")):\n                row[out_name] = (np.sum([sampled[h][col] for h in buckets], axis=0).astype("float32")\n                                 if buckets else np.full(len(point_ids), np.nan, "float32"))\n            row["rain_max6_mm"] = (np.max([sampled[h]["apcp_mm"] for h in buckets], axis=0).astype("float32")\n                                   if buckets else np.full(len(point_ids), np.nan, "float32"))\n        frames.append(pd.DataFrame(row))\n    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n\n\n# ------------------------------------------------------------------ driver\ndef run_dates():\n    """Run dates for this shard.  SHARD_STRIDE=2 keeps every second day (by ordinal parity, so two shards with\n    SHARD_OFFSET 0 and 1 partition the archive exactly); consecutive daily runs share ~90% of the atmosphere."""\n    explicit = os.environ.get("SHARD_DATES")\n    if explicit:\n        return [date.fromisoformat(d) for d in explicit.split(",")]\n    start = date.fromisoformat(os.environ.get("SHARD_START", "2021-01-01"))\n    end = date.fromisoformat(os.environ.get("SHARD_END", "2021-01-03"))\n    stride, offset = int(os.environ.get("SHARD_STRIDE", "1")), int(os.environ.get("SHARD_OFFSET", "0"))\n    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]\n    return [d for d in days if d.toordinal() % stride == offset]\n\n\ndef ensure_eccodes():\n    """Kaggle\'s image has no eccodes; install the wheel (it bundles libeccodes) before forking workers."""\n    try:\n        import eccodes  # noqa: F401\n    except ImportError:\n        import importlib\n        import subprocess\n        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "eccodes"], check=True)\n        importlib.invalidate_caches()\n        import eccodes  # noqa: F401\n\n\ndef main():\n    ensure_eccodes()\n    OUT.mkdir(parents=True, exist_ok=True)\n    dates = run_dates()\n    ctx = mp.get_context("fork")\n    started = time.time()\n    report = {"dates": len(dates), "cycle": CYCLE, "leads": len(LEADS), "absent_runs": [], "partial_runs": {},\n              "bad_messages": {}}\n    with cf.ProcessPoolExecutor(max_workers=12, mp_context=ctx, initializer=_init_worker) as pool:\n        # static fields (orography, land-sea mask) from the first date that exists\n        points = None\n        for d in dates:\n            arrays, stats = fetch_run(pool, d, with_static=True)\n            if arrays is not None and LEADS[0] in arrays and "hgt" in arrays[LEADS[0]] and "land" in arrays[LEADS[0]]:\n                points = build_points(arrays[LEADS[0]]["land"], arrays[LEADS[0]]["hgt"])\n                first_arrays, first_date = arrays, d\n                break\n        if points is None:\n            raise RuntimeError("could not obtain orography/land mask from any requested date")\n        points.to_parquet(OUT / "points.parquet", index=False)\n        report["points"] = {"total": int(len(points)), "nodes": int((points["kind"] == "node").sum()),\n                            "stations": int((points["kind"] == "station").sum())}\n        print("[gfs] points:", report["points"], flush=True)\n\n        point_ids = points["point_id"].to_numpy()\n        sampler = Sampler(points["lat"].to_numpy(), points["lon"].to_numpy())\n        stn = points[points["kind"] == "station"]\n        stn_ids = stn["point_id"].to_numpy()\n        stn_sampler = Sampler(stn["lat"].to_numpy(), stn["lon"].to_numpy())\n\n        p1_parts, p2_parts, month_key, written = [], [], None, []\n\n        def flush(key):\n            if not p2_parts:\n                return\n            name1, name2 = OUT / f"p1_station_steps_{key}.parquet", OUT / f"p2_point_days_{key}.parquet"\n            pd.concat(p1_parts, ignore_index=True).to_parquet(name1, index=False)\n            pd.concat(p2_parts, ignore_index=True).to_parquet(name2, index=False)\n            written.append(key)\n            p1_parts.clear()\n            p2_parts.clear()\n\n        for n, d in enumerate(dates):\n            key = f"{d:%Y%m}"\n            if month_key is not None and key != month_key:\n                flush(month_key)\n            month_key = key\n            if d == first_date:\n                arrays, stats = first_arrays, {"missing_steps": 0}\n            else:\n                arrays, stats = fetch_run(pool, d)\n            if arrays is None:\n                report["absent_runs"].append(d.isoformat())\n                continue\n            if stats["missing_steps"]:\n                report["partial_runs"][d.isoformat()] = stats["missing_steps"]\n            if stats.get("bad"):\n                report["bad_messages"][d.isoformat()] = stats["bad"]\n                print(f"[gfs] {d}: {len(stats[\'bad\'])} message(s) unreadable after retries: {stats[\'bad\'][:5]}", flush=True)\n            p1_parts.append(station_step_rows(d, arrays, None, stn_ids, stn_sampler))\n            p2_parts.append(point_day_rows(d, arrays, point_ids, sampler))\n            if (n + 1) % 5 == 0 or n == len(dates) - 1:\n                elapsed = time.time() - started\n                print(f"[gfs] {n + 1}/{len(dates)} dates  {elapsed / (n + 1):.1f}s/date  "\n                      f"absent={len(report[\'absent_runs\'])} partial={len(report[\'partial_runs\'])}", flush=True)\n        if month_key is not None:\n            flush(month_key)\n    report["months_written"] = written\n    report["seconds"] = round(time.time() - started)\n    (OUT / "gfs_report.json").write_text(json.dumps(report, indent=1))\n    print("GFS_REPORT_BEGIN")\n    print(json.dumps(report, indent=1))\n    print("GFS_REPORT_END")\n\n\nif __name__ == "__main__":\n    main()\n')
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


def station_step_ext(run_date, arrays, terrain, stn_ids, stn_sampler):
    frames = []
    for lead in sorted(arrays):
        derived = derive_ext(arrays[lead], terrain)
        data = {"point_id": stn_ids, "run_date": np.datetime64(run_date), "lead_h": np.int16(lead)}
        for name, arr in derived.items():
            data[name] = stn_sampler(arr).astype("float32")
        for wname in ("apcp", "tmax2m", "tmin2m"):
            for pre in ("avg_", "spr_"):          # GFS apcp width is already in the base table (apcp_width_h)
                if f"{pre}{wname}" in arrays[lead] and f"width:{pre}{wname}" in arrays[lead]:
                    data[f"{pre}{wname}_width_h"] = np.int8(arrays[lead][f"width:{pre}{wname}"])
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def point_day_ext(run_date, arrays, terrain, point_ids, sampler):
    sampled, grids = {}, {}
    for lead, fields in arrays.items():
        derived = derive_ext(fields, terrain)
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
            p1_parts.append(station_step_ext(d, arrays, terrain, stn_ids, stn_sampler))
            p2_parts.append(point_day_ext(d, arrays, terrain, point_ids, sampler))
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
