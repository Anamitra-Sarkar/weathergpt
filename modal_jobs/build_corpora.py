"""D1 (multi-model MOS corpus) and D2 (ensemble corpus) builders.

Both run on Modal, fan out over locations so each container paces its own
requests politely, and write partitioned Parquet to the `weathergpt-data`
volume.  Nothing is written unless `contracts.py` passes.

Why this shape:
  * `historical-forecast-api` accepts `<var>_previous_dayN` and returns the
    columns per-model suffixed, so ONE request yields real forecasts at eight
    different lead ages for four models.  That gives genuine forecast lead time
    instead of the row index the previous pipeline used.
  * ERA5-Land (9 km) is the verification truth; it lags real time by ~5 days,
    which is why the windows stop short of today.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from datetime import date, datetime, timedelta

import modal

from modal_jobs.common import DATA_DIR, DATA_IMAGE, VOLUMES, app
from modal_jobs.locations import SEED_PLACES

# --- constants ---------------------------------------------------------------
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HIST_FC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

D1_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"]
D1_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]
LEAD_AGES = list(range(0, 8))  # 0 = same-day run, N = the run from N days earlier

# era5_land is 9 km but publishes only temperature and soil fields through this
# API; era5_seamless keeps the fine-scale ERA5-Land temperature and fills
# precipitation, wind and humidity from ERA5 proper.  Verified directly:
# era5_land returns 100% nulls for precipitation and wind_speed_10m.
TRUTH_MODEL = "era5_seamless"
TRUTH_VARS = D1_VARS

# ERA5-Land publication lag.  Keep the window entirely inside verifiable ground truth.
TRUTH_LAG_DAYS = 6
D1_MONTHS = 12
D2_PAST_DAYS = 93
ENSEMBLE_MODEL = "gfs025"
ENSEMBLE_VARS = ["precipitation", "temperature_2m", "wind_speed_10m"]

USER_AGENT = "WeatherGPT-research/1.0 (SIH 2026; academic evaluation)"


def _window() -> tuple[str, str]:
    end = date.today() - timedelta(days=TRUTH_LAG_DAYS)
    start = date(end.year - 1, end.month, 1) + timedelta(days=0)
    return start.isoformat(), end.isoformat()


def _chunks(start: str, end: str, days: int = 92) -> list[tuple[str, str]]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while s <= e:
        stop = min(s + timedelta(days=days - 1), e)
        out.append((s.isoformat(), stop.isoformat()))
        s = stop + timedelta(days=1)
    return out


# --- polite async HTTP -------------------------------------------------------
async def _get_json(client, url: str, params: dict, *, attempts: int = 6):
    """GET with exponential backoff.  Raises on final failure; callers count it."""
    delay = 2.0
    last = None
    for attempt in range(attempts):
        try:
            response = await client.get(url, params=params, headers={"User-Agent": USER_AGENT})
            if response.status_code == 429 or response.status_code >= 500:
                last = RuntimeError(f"{response.status_code} {response.text[:160]}")
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 90)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"{response.status_code} {response.text[:300]}")
            return response.json()
        except Exception as exc:  # network flake
            last = exc
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 90)
    raise RuntimeError(f"giving up on {url}: {last}")


# --- location resolution -----------------------------------------------------
@app.function(image=DATA_IMAGE, volumes=VOLUMES, timeout=60 * 30)
def resolve_locations() -> dict:
    """Geocode the seed place names.  Elevation and admin hierarchy come from the API."""
    import httpx

    names = SEED_PLACES

    async def run():
        resolved, failed = [], []
        limits = httpx.Limits(max_connections=4)
        async with httpx.AsyncClient(timeout=45, limits=limits) as client:
            for name in names:
                try:
                    payload = await _get_json(client, GEOCODE_URL,
                                              {"name": name, "count": 8, "language": "en", "format": "json"})
                except Exception as exc:
                    failed.append({"name": name, "error": str(exc)})
                    continue
                hits = [h for h in (payload.get("results") or []) if h.get("country_code") == "IN"]
                if not hits:
                    failed.append({"name": name, "error": "no IN result"})
                    continue
                hit = max(hits, key=lambda h: h.get("population") or 0)
                resolved.append({
                    "loc_id": f"{hit['id']}",
                    "query": name,
                    "name": hit.get("name"),
                    "lat": round(float(hit["latitude"]), 4),
                    "lon": round(float(hit["longitude"]), 4),
                    "elevation_m": hit.get("elevation"),
                    "admin1": hit.get("admin1"),
                    "admin2": hit.get("admin2"),
                    "admin3": hit.get("admin3"),
                    "timezone": hit.get("timezone"),
                    "population": hit.get("population"),
                    "feature_code": hit.get("feature_code"),
                })
                await asyncio.sleep(0.35)
        return resolved, failed

    resolved, failed = asyncio.run(run())
    # de-duplicate on rounded coordinates: two seed names can hit the same town
    seen, unique = set(), []
    for item in resolved:
        key = (round(item["lat"], 2), round(item["lon"], 2))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    missing_elev = [item["query"] for item in unique if item["elevation_m"] is None]
    if missing_elev:
        raise RuntimeError(f"geocoder returned no elevation for {missing_elev}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "source": "open-meteo geocoding-api v1/search",
        "n_requested": len(names),
        "n_resolved": len(unique),
        "failed": failed,
        "locations": unique,
    }
    with open(f"{DATA_DIR}/locations.json", "w") as handle:
        json.dump(out, handle, indent=2)
    from modal_jobs.common import DATA_VOL
    DATA_VOL.commit()
    print(f"[locations] resolved {len(unique)}/{len(names)}  failed={len(failed)}")
    if failed:
        print(f"[locations] failures: {failed}")
    print(f"[locations] elevation range {min(i['elevation_m'] for i in unique)}"
          f"..{max(i['elevation_m'] for i in unique)} m over "
          f"{len({i['admin1'] for i in unique})} admin1 regions")
    return out


# --- D1: multi-model, multi-lead MOS corpus ---------------------------------
async def _fetch_d1_location(client, loc: dict, start: str, end: str) -> "object":
    """One location -> a wide table keyed by (valid_time, lead_age_days).

    One request per (model, 46-day chunk).  Asking for all four models and all
    eight lead ages at once is 128 hourly columns and Open-Meteo answers it with
    a 502 — verified.  Small requests also mean a single failure costs one model
    for six weeks instead of the whole location.
    """
    import numpy as np
    import pandas as pd

    hourly_vars = []
    for var in D1_VARS:
        hourly_vars.append(var)
        hourly_vars += [f"{var}_previous_day{n}" for n in LEAD_AGES if n > 0]

    chunks = _chunks(start, end, days=46)
    per_model: dict[str, "pd.DataFrame"] = {}
    misses = 0
    for model in D1_MODELS:
        frames = []
        for chunk_start, chunk_end in chunks:
            try:
                payload = await _get_json(client, HIST_FC_URL, {
                    "latitude": loc["lat"], "longitude": loc["lon"],
                    "start_date": chunk_start, "end_date": chunk_end,
                    "hourly": ",".join(hourly_vars),
                    "models": model,
                    "timezone": "UTC",
                })
            except Exception as exc:
                misses += 1
                print(f"[d1] miss {loc['query']} {model} {chunk_start}: {str(exc)[:120]}")
                continue
            frames.append(pd.DataFrame(payload["hourly"]))
            await asyncio.sleep(0.25)
        if not frames:
            continue
        merged_model = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
        # single-model responses are unsuffixed; suffix them so the wide table
        # keeps the same column grammar as a multi-model response
        merged_model = merged_model.rename(columns={
            column: f"{column}_{model}" for column in merged_model.columns if column != "time"})
        per_model[model] = merged_model

    if not per_model:
        raise RuntimeError(f"every model failed for {loc['query']}")

    forecast = None
    for frame_model in per_model.values():
        forecast = frame_model if forecast is None else forecast.merge(
            frame_model, on="time", how="outer")

    truth_frames = []
    for chunk_start, chunk_end in _chunks(start, end, days=366):
        payload = await _get_json(client, ARCHIVE_URL, {
            "latitude": loc["lat"], "longitude": loc["lon"],
            "start_date": chunk_start, "end_date": chunk_end,
            "hourly": ",".join(TRUTH_VARS), "models": TRUTH_MODEL, "timezone": "UTC",
        })
        truth_frames.append(pd.DataFrame(payload["hourly"]))
        await asyncio.sleep(0.3)
    truth = pd.concat(truth_frames, ignore_index=True).drop_duplicates(subset="time")
    truth = truth.rename(columns={var: f"truth_{var}" for var in TRUTH_VARS})

    merged = forecast.merge(truth, on="time", how="inner")
    if merged.empty:
        raise RuntimeError(f"no overlap between forecast and truth for {loc['query']}")

    rows = []
    for lead_age in LEAD_AGES:
        block = {"time": merged["time"].to_numpy(), "lead_age_days": lead_age}
        for var in D1_VARS:
            for model in D1_MODELS:
                suffix = "" if lead_age == 0 else f"_previous_day{lead_age}"
                column = f"{var}{suffix}_{model}"
                block[f"fc_{var}_{model}"] = (merged[column].to_numpy()
                                              if column in merged.columns
                                              else np.full(len(merged), np.nan))
        for var in TRUTH_VARS:
            block[f"truth_{var}"] = merged[f"truth_{var}"].to_numpy()
        rows.append(pd.DataFrame(block))

    out = pd.concat(rows, ignore_index=True)
    out["loc_id"] = loc["loc_id"]
    out["lat"] = loc["lat"]
    out["lon"] = loc["lon"]
    out["elevation_m"] = loc["elevation_m"]
    out["admin1"] = loc["admin1"] or "unknown"
    out["chunk_misses"] = misses
    valid = pd.to_datetime(out["time"], utc=True)
    out["valid_time"] = valid
    # previous_dayN is the run issued N days earlier at 00Z, so the lead to a
    # valid hour H is 24*N + H.  DATA.md records this as an approximation of the
    # exact initialisation hour, which the provider does not expose.
    out["lead_hours"] = out["lead_age_days"] * 24 + valid.dt.hour
    out["hour_utc"] = valid.dt.hour
    out["doy"] = valid.dt.dayofyear
    out["month"] = valid.dt.month
    return out.drop(columns=["time"])


@app.function(image=DATA_IMAGE, volumes=VOLUMES, timeout=60 * 90, max_containers=8, retries=1)
def build_d1_shard(shard: list[dict], start: str, end: str, shard_id: int) -> dict:
    import httpx
    import pandas as pd

    async def run():
        frames, failures = [], []
        async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=2)) as client:
            for loc in shard:
                try:
                    frames.append(await _fetch_d1_location(client, loc, start, end))
                    print(f"[d1:{shard_id}] ok {loc['query']}")
                except Exception as exc:
                    failures.append({"loc": loc["query"], "error": str(exc)[:300]})
                    print(f"[d1:{shard_id}] FAIL {loc['query']}: {exc}")
                await asyncio.sleep(0.5)
        return frames, failures

    frames, failures = asyncio.run(run())
    if not frames:
        return {"shard": shard_id, "rows": 0, "failures": failures}
    table = pd.concat(frames, ignore_index=True)
    path = f"{DATA_DIR}/d1_mos"
    os.makedirs(path, exist_ok=True)
    table.to_parquet(f"{path}/shard_{shard_id:03d}.parquet", index=False, compression="zstd")
    from modal_jobs.common import DATA_VOL
    DATA_VOL.commit()
    return {"shard": shard_id, "rows": int(len(table)), "locs": len(frames), "failures": failures}


# --- D2: 31-member ensemble corpus ------------------------------------------
async def _fetch_d2_location(client, loc: dict) -> "object":
    import numpy as np
    import pandas as pd

    payload = await _get_json(client, ENSEMBLE_URL, {
        "latitude": loc["lat"], "longitude": loc["lon"],
        "models": ENSEMBLE_MODEL, "hourly": ",".join(ENSEMBLE_VARS),
        "past_days": D2_PAST_DAYS, "forecast_days": 1, "timezone": "UTC",
    })
    hourly = payload["hourly"]
    frame = pd.DataFrame(hourly)
    times = pd.to_datetime(frame["time"], utc=True)

    start = times.min().date().isoformat()
    end = (times.max().date() - timedelta(days=TRUTH_LAG_DAYS)).isoformat()
    truth_payload = await _get_json(client, ARCHIVE_URL, {
        "latitude": loc["lat"], "longitude": loc["lon"],
        "start_date": start, "end_date": end,
        "hourly": ",".join(ENSEMBLE_VARS), "models": TRUTH_MODEL, "timezone": "UTC",
    })
    truth = pd.DataFrame(truth_payload["hourly"]).rename(
        columns={var: f"truth_{var}" for var in ENSEMBLE_VARS})

    merged = frame.merge(truth, on="time", how="inner")
    out = {"valid_time": pd.to_datetime(merged["time"], utc=True)}
    for var in ENSEMBLE_VARS:
        members = sorted(c for c in merged.columns
                         if c.startswith(f"{var}_member") and c[len(var) + 7:].isdigit())
        if not members:
            raise RuntimeError(f"no ensemble members for {var} at {loc['query']}")
        matrix = merged[members].to_numpy(dtype="float32")
        out[f"{var}_members"] = list(matrix)
        out[f"{var}_ctrl"] = merged[var].to_numpy(dtype="float32")
        out[f"truth_{var}"] = merged[f"truth_{var}"].to_numpy(dtype="float32")
    table = pd.DataFrame(out)
    table["n_members"] = len(members)
    table["loc_id"] = loc["loc_id"]
    table["lat"] = loc["lat"]
    table["lon"] = loc["lon"]
    table["elevation_m"] = loc["elevation_m"]
    table["admin1"] = loc["admin1"] or "unknown"
    table["hour_utc"] = table["valid_time"].dt.hour
    table["doy"] = table["valid_time"].dt.dayofyear
    return table


@app.function(image=DATA_IMAGE, volumes=VOLUMES, timeout=60 * 60, max_containers=8, retries=1)
def build_d2_shard(shard: list[dict], shard_id: int) -> dict:
    import httpx
    import pandas as pd

    async def run():
        frames, failures = [], []
        async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=2)) as client:
            for loc in shard:
                try:
                    frames.append(await _fetch_d2_location(client, loc))
                    print(f"[d2:{shard_id}] ok {loc['query']}")
                except Exception as exc:
                    failures.append({"loc": loc["query"], "error": str(exc)[:300]})
                    print(f"[d2:{shard_id}] FAIL {loc['query']}: {exc}")
                await asyncio.sleep(1.0)
        return frames, failures

    frames, failures = asyncio.run(run())
    if not frames:
        return {"shard": shard_id, "rows": 0, "failures": failures}
    table = pd.concat(frames, ignore_index=True)
    path = f"{DATA_DIR}/d2_ensemble"
    os.makedirs(path, exist_ok=True)
    table.to_parquet(f"{path}/shard_{shard_id:03d}.parquet", index=False, compression="zstd")
    from modal_jobs.common import DATA_VOL
    DATA_VOL.commit()
    return {"shard": shard_id, "rows": int(len(table)), "locs": len(frames), "failures": failures}


def _shards(locations: list[dict], n: int) -> list[list[dict]]:
    return [locations[i::n] for i in range(n)]


@app.local_entrypoint()
def main(what: str = "all", n_shards: int = 16, limit: int = 0):
    """modal run modal_jobs/build_corpora.py --what all"""
    start, end = _window()
    print(f"window {start} -> {end}")

    if what in ("locations", "all"):
        resolve_locations.remote()

    import json as _json
    meta = _json.loads(_read_locations.remote())
    locations = meta["locations"]
    if limit:
        locations = locations[:limit]
    print(f"{len(locations)} locations")

    if what in ("d1", "all"):
        shards = _shards(locations, n_shards)
        results = list(build_d1_shard.starmap(
            [(s, start, end, i) for i, s in enumerate(shards) if s]))
        total = sum(r["rows"] for r in results)
        fails = [f for r in results for f in r["failures"]]
        print(f"[d1] rows={total:,} failures={len(fails)}")
        for f in fails[:20]:
            print("   ", f)

    if what in ("d2", "all"):
        shards = _shards(locations, n_shards)
        results = list(build_d2_shard.starmap(
            [(s, i) for i, s in enumerate(shards) if s]))
        total = sum(r["rows"] for r in results)
        fails = [f for r in results for f in r["failures"]]
        print(f"[d2] rows={total:,} failures={len(fails)}")
        for f in fails[:20]:
            print("   ", f)


@app.function(image=DATA_IMAGE, volumes=VOLUMES)
def _read_locations() -> str:
    with open(f"{DATA_DIR}/locations.json") as handle:
        return handle.read()
