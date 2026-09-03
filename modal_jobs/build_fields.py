"""D3 — the M1 corpus, harvested from authoritative parameter tables.

No string augmentation.  Every row is a field name that really exists in a
meteorological data standard, paired with that standard's own declared unit,
level and time-range metadata, and labelled by `app/services/field_taxonomy.py`
from *that metadata* rather than from the abbreviation.  Consequences:

  * `TMAX (mm)` is impossible — the labeller abstains on a unit contradiction.
  * accumulation windows are read out of real time-range strings
    (`0-3 hour acc fcst`), not sampled from `{1,3,6,24}`.
  * the split is **by source table**: train on CF + GRIB2, test zero-shot on
    WRF / NCEP / BUFR / Open-Meteo / IMD names the model has never seen.  That
    measures the thing the product actually needs — generalising to a schema
    nobody has mapped yet.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta

from modal_jobs.common import DATA_DIR, DATA_IMAGE, VOLUMES, app

ECCODES_BASE = "https://raw.githubusercontent.com/ecmwf/eccodes/develop/definitions"
CF_TABLE_URL = "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
WRF_REGISTRY_URLS = [
    "https://raw.githubusercontent.com/wrf-model/WRF/master/Registry/Registry.EM_COMMON",
    "https://raw.githubusercontent.com/wrf-model/WRF/master/Registry/registry.diags",
]
GFS_IDX_TEMPLATE = ("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{cycle}/atmos/"
                    "gfs.t{cycle}z.pgrb2.0p25.f{fhr:03d}.idx")
NCEP_TABLE_TEMPLATE = ("https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/"
                       "grib2_table4-2-{discipline}-{category}.shtml")
# discipline/category pairs that carry the surface weather parameters GFS ships
NCEP_TABLES = [(0, c) for c in (0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20)] + \
              [(1, c) for c in (0, 1)] + [(2, c) for c in (0, 3, 4)] + \
              [(10, c) for c in (0, 1, 2, 3, 4)]
SACHET_RSS = "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml"
SACHET_CAP = "https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier={identifier}"

TRAIN_SOURCES = ("cf_standard_names", "grib2_eccodes", "cap_sachet")
# Held-out schemas are split in two.  The abstention threshold has to be tuned
# somewhere, and tuning it on in-domain validation and applying it zero-shot is
# a mismatch: the similarity distribution shifts when the schema changes, so the
# operating point lands in the wrong place.  `dev_zeroshot` is a *different*
# unseen schema used only to pick that threshold; `test_zeroshot` is never
# touched until the final measurement.
DEV_ZEROSHOT_SOURCES = ("bufr_wmo", "open_meteo", "imd_products")
ZEROSHOT_SOURCES = ("wrf_registry", "ncep_gfs_idx")

# GRIB2 code table 4.5 — enough of it to place a level.
GRIB_SURFACE = {
    1: "surface", 2: "cloud_base", 3: "cloud_top", 4: "level_of_0c_isotherm",
    6: "max_wind_level", 7: "tropopause", 8: "nominal_top", 9: "sea_bottom",
    10: "column", 20: "isothermal_level", 100: "pressure_level", 101: "mean_sea_level",
    102: "mean_sea_level", 103: "height_above_ground", 104: "sigma_level",
    105: "hybrid_level", 106: "soil", 107: "potential_vorticity_surface",
    108: "pressure_above_ground", 109: "potential_vorticity_surface",
    160: "depth_below_sea", 200: "column", 220: "boundary_layer",
}

# Real Open-Meteo hourly variable names (the fields our own adapters consume).
OPEN_METEO_FIELDS: list[tuple[str, str, str]] = [
    ("temperature_2m", "Air temperature at 2 metres above ground", "°C"),
    ("relative_humidity_2m", "Relative humidity at 2 metres above ground", "%"),
    ("dew_point_2m", "Dew point temperature at 2 metres above ground", "°C"),
    ("apparent_temperature", "Apparent temperature, perceived feels-like temperature", "°C"),
    ("precipitation", "Total precipitation, sum of rain, showers and snow of the preceding hour", "mm"),
    ("rain", "Rain from large scale weather systems of the preceding hour", "mm"),
    ("showers", "Showers from convective precipitation of the preceding hour", "mm"),
    ("snowfall", "Snowfall amount of the preceding hour", "cm"),
    ("precipitation_probability", "Probability of precipitation with more than 0.1 mm of the preceding hour", "%"),
    ("pressure_msl", "Atmospheric air pressure reduced to mean sea level", "hPa"),
    ("surface_pressure", "Atmospheric air pressure at surface", "hPa"),
    ("cloud_cover", "Total cloud cover as an area fraction", "%"),
    ("visibility", "Viewing distance in metres", "m"),
    ("evapotranspiration", "Evapotranspiration of the preceding hour from land surface and plants", "mm"),
    ("et0_fao_evapotranspiration", "Reference evapotranspiration of a well watered grass field", "mm"),
    ("wind_speed_10m", "Wind speed at 10 metres above ground", "km/h"),
    ("wind_direction_10m", "Wind direction at 10 metres above ground", "°"),
    ("wind_gusts_10m", "Gusts at 10 metres above ground as a maximum of the preceding hour", "km/h"),
    ("soil_temperature_0cm", "Soil temperature at 0 cm depth", "°C"),
    ("soil_moisture_0_to_1cm", "Average soil water content as volumetric mixing ratio", "m3 m-3"),
    ("shortwave_radiation", "Shortwave solar radiation as average of the preceding hour", "W m-2"),
    ("cape", "Convective available potential energy", "J kg-1"),
    ("sunshine_duration", "Number of seconds of sunshine of the preceding hour", "s"),
    ("temperature_2m_max", "Maximum daily air temperature at 2 metres above ground", "°C"),
    ("temperature_2m_min", "Minimum daily air temperature at 2 metres above ground", "°C"),
    ("precipitation_sum", "Sum of daily precipitation including rain, showers and snowfall", "mm"),
    ("wave_height", "Significant height of combined wind waves and swell", "m"),
    ("wave_period", "Mean wave period of combined wind waves and swell", "s"),
    ("sea_surface_temperature", "Sea surface temperature", "°C"),
]

# Real IMD product field names, as documented for their public API family and in
# the project's own problem brief.  Warnings stay categorical.
IMD_FIELDS: list[tuple[str, str, str, str]] = [
    ("Temperature", "Current air temperature reported by an AWS station", "°C", "observation"),
    ("Temp", "Station air temperature", "°C", "observation"),
    ("Max Temp", "Maximum temperature of the day for the district", "°C", "forecast"),
    ("Min Temp", "Minimum temperature of the day for the district", "°C", "forecast"),
    ("Humidity", "Relative humidity reported by an AWS station", "%", "observation"),
    ("RH", "Relative humidity", "%", "observation"),
    ("Wind Speed", "Wind speed reported by an AWS station", "km/h", "observation"),
    ("Wind Direction", "Wind direction reported by an AWS station", "°", "observation"),
    ("RF", "Rainfall accumulated over the past 24 hours at the station", "mm", "observation"),
    ("Rainfall", "Rainfall accumulated over the past 24 hours", "mm", "observation"),
    ("Actual Rainfall", "Actual district rainfall accumulated over 24 hours", "mm", "observation"),
    ("Normal Rainfall", "Long period average district rainfall for the same 24 hours", "mm", "climatology"),
    ("Departure", "Percentage departure of actual district rainfall from normal", "%", "climatology"),
    ("Cumulative Rainfall", "Season cumulative district rainfall", "mm", "observation"),
    ("QPF", "Quantitative precipitation forecast accumulated over 24 hours for a river basin", "mm", "forecast"),
    ("Rain Prob", "Probability of precipitation for the forecast day", "%", "forecast"),
    ("Heavy Rainfall", "Heavy rainfall warning issued for the district", None, "warning"),
    ("Very Heavy Rainfall", "Very heavy rainfall warning issued for the district", None, "warning"),
    ("Extremely Heavy Rainfall", "Extremely heavy rainfall warning issued for the district", None, "warning"),
    ("Thunderstorm & Lightning", "Thunderstorm with lightning warning issued for the district", None, "warning"),
    ("Squall", "Squall warning issued for the district", None, "warning"),
    ("Hailstorm", "Hail warning issued for the district", None, "warning"),
    ("Dust Storm", "Dust storm warning issued for the district", None, "warning"),
    ("Heat Wave", "Heat wave warning issued for the district", None, "warning"),
    ("Cold Wave", "Cold wave warning issued for the district", None, "warning"),
    ("Ground Frost", "Ground frost warning issued for the district", None, "warning"),
    ("Dense Fog", "Dense fog warning issued for the district", None, "warning"),
    ("Cyclone", "Cyclonic storm warning issued for the coastal district", None, "warning"),
    ("Gale Warning", "Gale warning issued for the sea area", None, "warning"),
    ("Rough Sea", "Sea condition warning, sea very rough to high", None, "warning"),
    ("Widespread Rain", "District rainfall distribution category, widespread", None, "forecast"),
    ("Scattered Rain", "District rainfall distribution category, scattered", None, "forecast"),
    ("Isolated Rain", "District rainfall distribution category, isolated", None, "forecast"),
    ("Nowcast TS", "Nowcast thunderstorm category code valid for the next three hours", None, "nowcast"),
]


# --- parsers -----------------------------------------------------------------
_BLOCK = re.compile(r"^#(?P<comment>.*)\n'(?P<key>(?:[^'\\]|\\.)*)'\s*=\s*\{(?P<body>[^}]*)\}",
                    re.M)


def _parse_def(text: str) -> list[dict]:
    """Parse one eccodes `*.def` file into ordered blocks."""
    out = []
    for match in _BLOCK.finditer(text):
        body = {}
        for line in match.group("body").split(";"):
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            body[key.strip()] = value.strip()
        out.append({"comment": match.group("comment").strip(),
                    "key": match.group("key").strip(),
                    "body": body})
    return out


def _parse_cf(xml_text: str) -> list[dict]:
    from lxml import etree

    root = etree.fromstring(xml_text.encode("utf-8"))
    rows = []
    for entry in root.iter("entry"):
        name = entry.get("id")
        if not name:
            continue
        units = (entry.findtext("canonical_units") or "").strip()
        description = (entry.findtext("description") or "").strip()
        rows.append({"raw_field": name, "unit": units or None,
                     "description": description or name.replace("_", " ")})
    return rows


_WRF_STATE = re.compile(
    r'^\s*state\s+\S+\s+(?P<var>\w+)\s+\S+.*?"(?P<dname>[^"]*)"\s+"(?P<desc>[^"]*)"\s+"(?P<unit>[^"]*)"',
    re.M)


def _parse_wrf(text: str) -> list[dict]:
    rows = []
    for match in _WRF_STATE.finditer(text):
        var, desc, unit = match.group("var"), match.group("desc").strip(), match.group("unit").strip()
        if not desc:
            continue
        rows.append({"raw_field": var, "unit": unit or None, "description": desc})
    return rows


_IDX_LINE = re.compile(r"^\d+:\d+:d=(?P<init>\d+):(?P<abbrev>[^:]+):(?P<level>[^:]*):(?P<trange>[^:]*):")


def _parse_gfs_idx(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        match = _IDX_LINE.match(line)
        if not match:
            continue
        rows.append({"raw_field": match.group("abbrev").strip(),
                     "level_text": match.group("level").strip(),
                     "time_range_text": match.group("trange").strip()})
    return rows


_HTML_ROW = re.compile(r"<tr>(.*?)</tr>", re.S | re.I)
_HTML_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)


def _parse_ncep_table(html: str) -> dict[str, tuple[str, str | None]]:
    """NOAA GRIB2 code table 4.2-d-c -> {NCEP abbreviation: (description, unit)}.

    The `.idx` inventories name parameters with NCEP abbreviations (`APCP`,
    `TMP`, `UGRD`), which are not the eccodes shortNames (`tp`, `2t`, `10u`).
    Without this join the inventory rows carry no description and the labeller
    correctly abstains on all of them, which is how the real accumulation
    windows in those files were being thrown away.
    """
    out: dict[str, tuple[str, str | None]] = {}
    for row in _HTML_ROW.findall(html):
        cells = [re.sub(r"<[^>]+>", "", cell).replace("&nbsp;", " ").strip()
                 for cell in _HTML_CELL.findall(row)]
        if len(cells) < 4:
            continue
        _number, description, unit, abbrev = cells[0], cells[1], cells[2], cells[3]
        if not abbrev or not description or " " in abbrev or abbrev.lower() in ("abbrev", "-"):
            continue
        if abbrev in out:
            continue
        unit_clean = None if unit in ("-", "", "see", "Code table") else unit
        out[abbrev] = (description, unit_clean)
    return out


def _parse_sachet_rss(xml_text: str) -> list[dict]:
    """Real live Indian alerts -> identifier + headline + issuing authority."""
    from lxml import etree

    root = etree.fromstring(xml_text.encode("utf-8"))
    items = []
    for item in root.iter("item"):
        guid = (item.findtext("guid") or "").strip()
        title = (item.findtext("title") or "").strip()
        author = (item.findtext("author") or "").strip()
        category = (item.findtext("category") or "").strip()
        if guid and title:
            items.append({"identifier": guid, "headline": title,
                          "author": author, "category": category})
    return items


_CAP_NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}


def _parse_cap_event(xml_text: str) -> dict | None:
    from lxml import etree

    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except Exception:
        return None
    info = root.find("cap:info", _CAP_NS)
    if info is None:
        return None
    return {
        "event": (info.findtext("cap:event", namespaces=_CAP_NS) or "").strip(),
        "headline": (info.findtext("cap:headline", namespaces=_CAP_NS) or "").strip(),
        "severity": (info.findtext("cap:severity", namespaces=_CAP_NS) or "").strip(),
        "certainty": (info.findtext("cap:certainty", namespaces=_CAP_NS) or "").strip(),
        "sender": (root.findtext("cap:sender", namespaces=_CAP_NS) or "").strip(),
    }


def _parse_bufr(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        abbrev, name, unit = parts[1].strip(), parts[3].strip(), parts[4].strip()
        if not abbrev or not name:
            continue
        if unit.lower() in ("ccitt ia5", "numeric table", "flag table", "code table"):
            unit_out = None if "table" in unit.lower() else unit
        else:
            unit_out = unit
        rows.append({"raw_field": abbrev, "unit": unit_out, "description": name.title()})
    return rows


# --- the build ---------------------------------------------------------------
@app.function(image=DATA_IMAGE, volumes=VOLUMES, timeout=60 * 40)
def build_d3() -> dict:
    import httpx
    import pandas as pd

    from app.services.field_taxonomy import classify_native_field

    async def fetch_all() -> dict[str, str]:
        urls = {
            "shortName": f"{ECCODES_BASE}/grib2/shortName.def",
            "name": f"{ECCODES_BASE}/grib2/name.def",
            "units": f"{ECCODES_BASE}/grib2/units.def",
            "paramId": f"{ECCODES_BASE}/grib2/paramId.def",
            "cf": CF_TABLE_URL,
            "bufr": f"{ECCODES_BASE}/bufr/tables/0/wmo/43/element.table",
        }
        for index, url in enumerate(WRF_REGISTRY_URLS):
            urls[f"wrf{index}"] = url
        ymd = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
        for fhr in (3, 6, 12, 24, 48):
            urls[f"idx{fhr}"] = GFS_IDX_TEMPLATE.format(ymd=ymd, cycle="00", fhr=fhr)
        for discipline, category in NCEP_TABLES:
            urls[f"ncep_{discipline}_{category}"] = NCEP_TABLE_TEMPLATE.format(
                discipline=discipline, category=category)
        urls["sachet_rss"] = SACHET_RSS

        out = {}
        async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
            async def one(key, url):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    out[key] = response.text
                except Exception as exc:
                    print(f"[d3] fetch FAIL {key}: {exc}")
            await asyncio.gather(*(one(key, url) for key, url in urls.items()))
        return out

    blobs = asyncio.run(fetch_all())
    required = ["shortName", "name", "units", "cf"]
    missing = [key for key in required if key not in blobs]
    if missing:
        raise RuntimeError(f"cannot build D3 without {missing}")

    raw_rows: list[dict] = []

    # 1. GRIB2 via eccodes: four files, identical block order.
    short_names = _parse_def(blobs["shortName"])
    long_names = _parse_def(blobs["name"])
    units = _parse_def(blobs["units"])
    param_ids = _parse_def(blobs.get("paramId", "")) if "paramId" in blobs else []
    if not (len(short_names) == len(long_names) == len(units)):
        raise RuntimeError(f"eccodes block parity broken: "
                           f"{len(short_names)}/{len(long_names)}/{len(units)}")
    print(f"[d3] grib2 blocks: {len(short_names)}")
    for index, (sn, nm, un) in enumerate(zip(short_names, long_names, units)):
        body = sn["body"]
        surface = body.get("typeOfFirstFixedSurface")
        try:
            level_code = int(surface) if surface is not None and surface.isdigit() else None
        except ValueError:
            level_code = None
        level_text = GRIB_SURFACE.get(level_code, "")
        height = body.get("scaledValueOfFirstFixedSurface")
        if level_code == 103 and height and height.isdigit():
            level_text = f"{height} m above ground"
        statistical = body.get("typeOfStatisticalProcessing")
        raw_rows.append({
            "source_table": "grib2_eccodes",
            "raw_field": sn["key"],
            "description": nm["key"] or sn["comment"],
            "unit": un["key"] or None,
            "level_text": level_text,
            "time_range_text": "",
            "grib_statistical_processing": int(statistical) if statistical and statistical.isdigit() else None,
            "evidence_class_hint": None,
            "source_record_id": (param_ids[index]["key"] if index < len(param_ids) else None),
        })

    # 2. CF standard names.
    for row in _parse_cf(blobs["cf"]):
        raw_rows.append({
            "source_table": "cf_standard_names",
            "raw_field": row["raw_field"],
            "description": row["description"],
            "unit": row["unit"],
            "level_text": "", "time_range_text": "",
            "grib_statistical_processing": None, "evidence_class_hint": None,
            "source_record_id": row["raw_field"],
        })
    print(f"[d3] cf standard names: {sum(1 for r in raw_rows if r['source_table']=='cf_standard_names')}")

    # 3. WRF registry.
    wrf_text = "\n".join(blobs[key] for key in blobs if key.startswith("wrf"))
    for row in _parse_wrf(wrf_text):
        raw_rows.append({
            "source_table": "wrf_registry", "raw_field": row["raw_field"],
            "description": row["description"], "unit": row["unit"],
            "level_text": "", "time_range_text": "",
            "grib_statistical_processing": None, "evidence_class_hint": None,
            "source_record_id": row["raw_field"],
        })

    # 4. Real NCEP GFS inventories — the only place real accumulation windows live.
    ncep_params: dict[str, tuple[str, str | None]] = {}
    for key in sorted(k for k in blobs if k.startswith("ncep_")):
        ncep_params.update(_parse_ncep_table(blobs[key]))
    print(f"[d3] ncep abbreviation table: {len(ncep_params)} entries")

    idx_seen = set()
    for key in sorted(k for k in blobs if k.startswith("idx")):
        for row in _parse_gfs_idx(blobs[key]):
            signature = (row["raw_field"], row["level_text"], row["time_range_text"])
            if signature in idx_seen:
                continue
            idx_seen.add(signature)
            description, unit = ncep_params.get(row["raw_field"], ("", None))
            raw_rows.append({
                "source_table": "ncep_gfs_idx", "raw_field": row["raw_field"],
                "description": description, "unit": unit,
                "level_text": row["level_text"], "time_range_text": row["time_range_text"],
                "grib_statistical_processing": None, "evidence_class_hint": None,
                "source_record_id": "|".join(signature),
            })

    # 5. BUFR element table.
    if "bufr" in blobs:
        for row in _parse_bufr(blobs["bufr"]):
            raw_rows.append({
                "source_table": "bufr_wmo", "raw_field": row["raw_field"],
                "description": row["description"], "unit": row["unit"],
                "level_text": "", "time_range_text": "",
                "grib_statistical_processing": None, "evidence_class_hint": None,
                "source_record_id": row["raw_field"],
            })

    # 5b. Live SACHET/NDMA alerts: the authoritative vocabulary of Indian official
    # warning event names.  GRIB2 and CF contain no warning classes at all, so
    # without this the warning half of the label space has no training signal.
    async def fetch_cap_events(items: list[dict]) -> list[dict]:
        collected: list[dict] = []
        semaphore = asyncio.Semaphore(6)
        async with httpx.AsyncClient(timeout=60, follow_redirects=True, verify=False) as client:
            async def one(item):
                async with semaphore:
                    try:
                        response = await client.get(
                            SACHET_CAP.format(identifier=item["identifier"]))
                        response.raise_for_status()
                    except Exception:
                        return
                    parsed = _parse_cap_event(response.text)
                    if parsed and parsed["event"]:
                        parsed["author"] = item["author"]
                        collected.append(parsed)
            await asyncio.gather(*(one(item) for item in items))
        return collected

    if "sachet_rss" in blobs:
        try:
            alerts = _parse_sachet_rss(blobs["sachet_rss"])
            events = asyncio.run(fetch_cap_events(alerts))
            print(f"[d3] sachet: {len(alerts)} alerts, {len(events)} CAP events parsed")
            by_event: dict[str, dict] = {}
            for event in events:
                key = event["event"].strip()
                if not key:
                    continue
                existing = by_event.get(key)
                if existing is None or len(event["headline"]) > len(existing["headline"]):
                    by_event[key] = event
            for name, event in sorted(by_event.items()):
                raw_rows.append({
                    "source_table": "cap_sachet", "raw_field": name,
                    "description": (event["headline"] or name)[:400],
                    "unit": None, "level_text": "", "time_range_text": "",
                    "grib_statistical_processing": None, "evidence_class_hint": "warning",
                    "source_record_id": f"cap:{name}",
                })
        except Exception as exc:
            print(f"[d3] sachet harvest failed (non-fatal): {exc}")

    # 6/7. Provider field names we actually consume.
    for name, description, unit in OPEN_METEO_FIELDS:
        raw_rows.append({
            "source_table": "open_meteo", "raw_field": name, "description": description,
            "unit": unit, "level_text": "", "time_range_text": description,
            "grib_statistical_processing": None, "evidence_class_hint": None,
            "source_record_id": name,
        })
    for name, description, unit, hint in IMD_FIELDS:
        raw_rows.append({
            "source_table": "imd_products", "raw_field": name, "description": description,
            "unit": unit, "level_text": "", "time_range_text": description,
            "grib_statistical_processing": None, "evidence_class_hint": hint,
            "source_record_id": name,
        })

    print(f"[d3] harvested {len(raw_rows)} raw rows from "
          f"{len({r['source_table'] for r in raw_rows})} source tables")

    # --- label ---------------------------------------------------------------
    labelled, abstained = [], 0
    for row in raw_rows:
        label = classify_native_field(
            row["raw_field"], description=row["description"], unit=row["unit"],
            level_text=row["level_text"], time_range_text=row["time_range_text"],
            grib_statistical_processing=row["grib_statistical_processing"],
            evidence_class_hint=row["evidence_class_hint"],
        )
        record = dict(row)
        if label is None:
            abstained += 1
            record.update({"canonical_variable": "other", "statistic": "instant",
                           "accumulation_hours": None, "vertical_level": "surface",
                           "evidence_class": row["evidence_class_hint"] or "forecast",
                           "unit_family": None, "label_confidence": 1.0,
                           "matched_on": "", "is_abstain": True})
        else:
            record.update(label.as_row())
            record["is_abstain"] = False
        labelled.append(record)

    frame = pd.DataFrame(labelled)
    frame["raw_field"] = frame["raw_field"].astype(str).str.strip()
    frame = frame[frame["raw_field"].str.len() > 0]
    # de-duplicate on the exact (field, unit, level, window, label) tuple so the
    # same standard entry appearing twice cannot leak across the split
    frame["dedup_key"] = (frame["raw_field"].str.lower() + "|" +
                          frame["unit"].fillna("").astype(str).str.lower() + "|" +
                          frame["level_text"].fillna("").astype(str).str.lower() + "|" +
                          frame["time_range_text"].fillna("").astype(str).str.lower())
    before = len(frame)
    frame = frame.drop_duplicates(subset=["source_table", "dedup_key"])
    frame = frame.sort_values(["source_table", "raw_field"]).reset_index(drop=True)

    def split_of(table: str) -> str:
        if table in TRAIN_SOURCES:
            return "train"
        if table in DEV_ZEROSHOT_SOURCES:
            return "dev_zeroshot"
        return "test_zeroshot"

    frame["split"] = frame["source_table"].map(split_of)
    # carve an in-domain validation slice deterministically out of the train sources
    train_mask = frame["split"] == "train"
    digest = frame.loc[train_mask, "dedup_key"].map(
        lambda key: int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 100)
    frame.loc[train_mask & (digest < 12), "split"] = "val"

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = f"{DATA_DIR}/d3_fields.parquet"
    frame.drop(columns=["dedup_key"]).to_parquet(out_path, index=False, compression="zstd")

    stats = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "rows_before_dedup": int(before),
        "rows": int(len(frame)),
        "abstain_rows": int(frame["is_abstain"].sum()),
        "by_source": frame["source_table"].value_counts().to_dict(),
        "by_split": frame["split"].value_counts().to_dict(),
        "labelled_by_variable": frame.loc[~frame["is_abstain"], "canonical_variable"]
            .value_counts().to_dict(),
        "labelled_by_statistic": frame.loc[~frame["is_abstain"], "statistic"]
            .value_counts().to_dict(),
        "accumulation_windows": frame["accumulation_hours"].dropna().value_counts().to_dict(),
        "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
    }
    with open(f"{DATA_DIR}/d3_fields.stats.json", "w") as handle:
        json.dump(stats, handle, indent=2, default=str)
    from modal_jobs.common import DATA_VOL
    DATA_VOL.commit()

    print(json.dumps({k: v for k, v in stats.items() if k != "labelled_by_variable"},
                     indent=2, default=str))
    print("labelled_by_variable:", json.dumps(stats["labelled_by_variable"], default=str))
    return stats


@app.local_entrypoint()
def main():
    build_d3.remote()
