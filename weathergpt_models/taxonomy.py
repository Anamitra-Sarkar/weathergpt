"""Authoritative field taxonomy — one source of truth for native-field semantics.

This module is imported by three places and duplicated in none:
  * `modal_jobs/build_fields.py` labels the harvested CF / GRIB2 / WRF / NCEP /
    BUFR parameter tables with it, producing the M1 training corpus;
  * `app/ml/field_mapper.py` uses its vocabulary as the model's label space;
  * `app/services/variable_registry.py` falls back to it for exact lookups.

Two rules make the whole thing trustworthy:

1. **A label is derived from a source table's own metadata** (long name, canonical
   unit, level type, statistical processing, time-range indicator) — never from a
   guess about the abbreviation.  That is why a row labelled `temperature_max`
   can never carry `mm`: `classify_native_field` rejects the pairing outright.
2. **It abstains.**  When keywords and unit family do not agree, it returns
   `None` rather than the nearest-looking class.  Abstention is a first-class
   outcome that the trained mapper is taught to reproduce.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# --- unit families -----------------------------------------------------------
# Keys are normalised (lowercased, whitespace collapsed) unit strings as they
# actually appear in CF `canonical_units`, eccodes `units.def` and WRF Registry.
_UNIT_FAMILY: dict[str, str] = {}


def _register_units(family: str, units: Iterable[str]) -> None:
    for unit in units:
        _UNIT_FAMILY[unit] = family


_register_units("temperature", ["k", "kelvin", "c", "°c", "degc", "deg c", "degree c",
                                "celsius", "degrees c", "f", "°f", "degf", "deg k"])
# A depth unit cannot tell rain from visibility from snow pack, so every length
# lives in one family and the description does the disambiguating.
_register_units("length", ["m", "km", "cm", "mm", "ft", "inch", "in",
                           "kg m-2", "kg/m2", "kg m**-2", "kg m^-2",
                           "mm water equivalent", "m of water equivalent"])
_register_units("precip_rate", ["kg m-2 s-1", "kg m**-2 s**-1", "kg/m2/s", "mm/h",
                                "mm h-1", "mm/hr", "mm s-1", "cm/h", "m s-1 water"])
_register_units("speed", ["m s-1", "m/s", "m s**-1", "km/h", "km h-1", "kt", "knot",
                          "knots", "mph", "km s-1"])
_register_units("direction", ["degree", "degrees", "degree true", "deg", "rad", "radian",
                              "degrees true"])
_register_units("fraction", ["%", "percent", "1", "(0 - 1)", "0-1", "fraction",
                             "proportion", "dimensionless", "numeric"])
_register_units("pressure", ["pa", "hpa", "mb", "millibar", "kpa", "bar", "mbar"])
_register_units("energy_flux", ["w m-2", "w/m2", "w m**-2", "j m-2", "j m**-2", "j/m2",
                                "mj m-2", "mj/m2", "w m-2 sr-1"])
_register_units("time", ["s", "sec", "second", "seconds", "h", "hour", "hours", "min",
                         "minutes", "day", "days"])
_register_units("mixing_ratio", ["kg kg-1", "kg/kg", "kg kg**-1", "g kg-1", "g/kg"])
_register_units("volumetric", ["m3 m-3", "m**3 m**-3", "m3/m3", "m3 m**-3"])
_register_units("energy_mass", ["j kg-1", "j/kg", "j kg**-1"])
_register_units("count_rate", ["m-2 s-1", "km-2 h-1", "1/km2/h", "m**-2 s**-1"])
_register_units("categorical", ["", "none", "code table", "code", "categorical", "-"])


def unit_family(unit: Optional[str]) -> str:
    """Normalise a raw unit string to a family name, or `unknown`."""
    if unit is None:
        return "categorical"
    key = re.sub(r"\s+", " ", str(unit).strip().lower())
    key = key.replace("**", "").replace("^", "")
    if key in ("", "-", "1"):
        return _UNIT_FAMILY.get(key, "categorical" if key in ("", "-") else "fraction")
    if key in _UNIT_FAMILY:
        return _UNIT_FAMILY[key]
    # normalise a few spacing variants seen across the tables
    compact = key.replace(" ", "")
    for known, family in _UNIT_FAMILY.items():
        if known and known.replace(" ", "") == compact:
            return family
    return "unknown"


# --- statistics --------------------------------------------------------------
STATISTICS = ("instant", "accumulation", "mean", "max", "min", "probability", "categorical")

# GRIB2 code table 4.10 — typeOfStatisticalProcessing
GRIB_STAT_PROCESSING = {
    0: "mean", 1: "accumulation", 2: "max", 3: "min", 4: "instant",
    5: "instant", 6: "instant", 7: "instant", 8: "instant", 9: "instant", 10: "mean",
}


@dataclass(frozen=True)
class VariableSpec:
    canonical: str
    families: tuple[str, ...]
    default_statistic: str
    keywords: tuple[str, ...]
    anti_keywords: tuple[str, ...] = ()
    evidence_classes: tuple[str, ...] = ("forecast", "observation", "reanalysis", "nowcast")
    level: str = "surface"
    priority: int = 0


def _spec(canonical, families, statistic, keywords, anti=(), evidence=None, level="surface", priority=0):
    return VariableSpec(canonical, tuple(families), statistic, tuple(keywords), tuple(anti),
                        tuple(evidence) if evidence else ("forecast", "observation", "reanalysis", "nowcast"),
                        level, priority)


# Order matters only through `priority`; higher wins a tie.
VARIABLE_SPECS: tuple[VariableSpec, ...] = (
    # --- precipitation family: three mutually non-substitutable members --------
    _spec("precipitation_probability", ["fraction"], "probability",
          [r"\bprobability\b.*\b(precip|rain|rainfall)\b", r"\b(precip|rain)\w*\s+probab",
           r"\bpop\b", r"probability of (precipitation|rain)"], priority=3),
    _spec("precipitation_rate", ["precip_rate"], "instant",
          [r"precipitation rate", r"\brain(fall)? rate\b", r"\bprate\b",
           r"rate of (precipitation|rainfall)", r"instantaneous.*precipitation"], priority=2),
    _spec("precipitation_amount", ["length", "precip_depth"], "accumulation",
          [r"\bprecipitation\b", r"\brainfall\b", r"\btotal precipitation\b", r"\brain\b",
           r"\bapcp\b", r"large.?scale precipitation", r"convective precipitation"],
          anti=[r"\bsnow\b", r"\bprobability\b", r"\brate\b", r"\bfrequency\b", r"\btype\b"]),
    _spec("snowfall_amount", ["length", "precip_depth"], "accumulation",
          [r"\bsnowfall\b", r"\bsnow\b.*\b(amount|accumulation|fall|depth of fall)\b",
           r"water equivalent of.*snow"], anti=[r"\bdepth\b.*\bsnow ?pack\b"], priority=2),
    _spec("snow_depth", ["length"], "instant",
          [r"\bsnow depth\b", r"\bdepth of snow\b", r"snow ?pack"], priority=3),
    # --- temperature -----------------------------------------------------------
    _spec("temperature_max", ["temperature"], "max",
          [r"maximum.*temperature", r"\btmax\b", r"temperature.*maximum",
           r"\bmax\b.*\btemp"], priority=3),
    _spec("temperature_min", ["temperature"], "min",
          [r"minimum.*temperature", r"\btmin\b", r"temperature.*minimum",
           r"\bmin\b.*\btemp"], priority=3),
    _spec("dewpoint_2m", ["temperature"], "instant",
          [r"dew ?point"], priority=3),
    _spec("apparent_temperature", ["temperature"], "instant",
          [r"apparent temperature", r"\bheat index\b", r"wind chill", r"\bwet bulb globe\b",
           r"\bfeels like\b"], priority=3),
    _spec("soil_temperature", ["temperature"], "instant",
          [r"soil temperature", r"temperature of soil", r"\bsoil\b.*\btemp"],
          level="soil", priority=3),
    _spec("sea_surface_temperature", ["temperature"], "instant",
          [r"sea surface temperature", r"\bsst\b", r"surface temperature of the sea"],
          level="sea_surface", priority=3),
    _spec("temperature_2m", ["temperature"], "instant",
          [r"\btemperature\b", r"\btemp\b", r"\bt2m\b", r"\b2 metre temperature\b",
           r"air temperature", r"\btmp\b"],
          anti=[r"\bpotential\b", r"\bvirtual\b", r"\bsoil\b", r"\bsea surface\b",
                r"\bdew ?point\b", r"\bmaximum\b", r"\bminimum\b", r"\btendency\b",
                r"\banomaly\b", r"\bskin\b", r"\bbrightness\b"]),
    # --- wind ------------------------------------------------------------------
    _spec("wind_gust", ["speed"], "max",
          [r"\bgust\b", r"maximum wind speed", r"wind speed.*gust"], priority=3),
    _spec("wind_u", ["speed"], "instant",
          [r"\bu[- ]?component of wind\b", r"\bu10\b", r"\bzonal wind\b",
           r"eastward wind"], priority=3),
    _spec("wind_v", ["speed"], "instant",
          [r"\bv[- ]?component of wind\b", r"\bv10\b", r"\bmeridional wind\b",
           r"northward wind"], priority=3),
    _spec("wind_direction", ["direction"], "instant",
          [r"wind direction", r"direction of wind", r"\bwind from direction\b"], priority=3),
    _spec("wind_speed", ["speed"], "instant",
          [r"\bwind speed\b", r"speed of wind"],
          anti=[r"\bgust\b", r"\bfriction\b", r"\bshear\b", r"\bcomponent\b"]),
    # --- moisture / pressure ---------------------------------------------------
    _spec("humidity", ["fraction"], "instant",
          [r"relative humidity", r"\brh\b", r"\bhumidity\b"],
          anti=[r"\bspecific\b"], priority=2),
    _spec("specific_humidity", ["mixing_ratio"], "instant",
          [r"specific humidity", r"\bmixing ratio\b", r"\bq\b humidity"], priority=3),
    _spec("pressure_msl", ["pressure"], "instant",
          [r"mean sea level pressure", r"\bmslp?\b", r"\bprmsl\b",
           r"pressure reduced to msl", r"sea level pressure"], priority=3),
    _spec("pressure_surface", ["pressure"], "instant",
          [r"surface pressure", r"\bsp\b pressure", r"station pressure",
           r"\bpressure\b"], anti=[r"\bsea level\b", r"\btendency\b", r"\bvapour\b",
                                   r"\bvapor\b", r"\bpartial\b"]),
    # --- radiation / cloud / visibility ---------------------------------------
    _spec("solar_radiation", ["energy_flux"], "mean",
          [r"solar radiation", r"shortwave radiation", r"\bghi\b",
           r"downward.*short.?wave", r"global.*irradiance"], priority=2),
    _spec("sunshine_duration", ["time"], "accumulation",
          [r"sunshine duration", r"duration of sunshine"], priority=3),
    _spec("cloud_cover", ["fraction"], "instant",
          [r"cloud cover", r"cloud (area )?fraction", r"\btcdc\b", r"cloudiness"], priority=2),
    _spec("visibility", ["length"], "instant",
          [r"\bvisibility\b", r"horizontal visibility"], priority=3),
    # --- land surface ----------------------------------------------------------
    _spec("soil_moisture", ["volumetric", "length"], "instant",
          [r"soil moisture", r"soil water", r"volumetric.*soil"], level="soil", priority=3),
    _spec("evapotranspiration", ["length", "precip_depth"], "accumulation",
          [r"evapotranspiration", r"\bevaporation\b", r"\bet0\b",
           r"reference evapotranspiration"], priority=3),
    # --- convection ------------------------------------------------------------
    _spec("cape", ["energy_mass"], "instant",
          [r"convective available potential energy", r"\bcape\b"], priority=3),
    _spec("cin", ["energy_mass"], "instant",
          [r"convective inhibition", r"\bcin\b"], priority=3),
    _spec("lightning_density", ["count_rate"], "mean",
          [r"lightning", r"flash density", r"flash rate"], priority=3),
    _spec("thunderstorm_probability", ["fraction"], "probability",
          [r"probability.*thunderstorm", r"thunderstorm probability"], priority=4),
    # --- marine ----------------------------------------------------------------
    _spec("wave_height", ["length"], "instant",
          [r"wave height", r"significant height.*wave", r"\bswh\b"],
          level="sea_surface", priority=3),
    _spec("wave_period", ["time"], "instant",
          [r"wave period", r"period.*wave"], level="sea_surface", priority=3),
    # --- official warnings: categorical, never fused numerically ---------------
    _spec("heavy_rain_warning", ["categorical"], "categorical",
          [r"heavy rain", r"extremely heavy rain", r"very heavy rain"],
          evidence=["warning"], priority=4),
    _spec("thunderstorm_warning", ["categorical"], "categorical",
          [r"thunderstorm", r"lightning", r"squall"], evidence=["warning"], priority=3),
    _spec("cyclone_warning", ["categorical"], "categorical",
          [r"cyclone", r"depression", r"\bhurricane\b", r"\btyphoon\b"],
          evidence=["warning"], priority=4),
    _spec("heat_warning", ["categorical"], "categorical",
          [r"heat wave", r"heatwave", r"\bwarm wave\b"], evidence=["warning"], priority=4),
    _spec("cold_wave_warning", ["categorical"], "categorical",
          [r"cold wave", r"cold day", r"\bground frost\b", r"\bfrost\b"],
          evidence=["warning"], priority=4),
    _spec("fog_warning", ["categorical"], "categorical",
          [r"\bfog\b", r"dense fog", r"\bmist\b"], evidence=["warning"], priority=4),
    _spec("hail_warning", ["categorical"], "categorical",
          [r"\bhail\b", r"hailstorm"], evidence=["warning"], priority=4),
    _spec("flood_warning", ["categorical"], "categorical",
          [r"\bflood\b", r"inundation", r"river.*(warning|danger) level"],
          evidence=["warning"], priority=4),
    _spec("marine_warning", ["categorical"], "categorical",
          [r"\bsea\b.*(rough|high|very rough)", r"gale warning", r"port warning",
           r"fishermen.*advis"], evidence=["warning"], priority=4),
    _spec("dust_storm_warning", ["categorical"], "categorical",
          [r"dust storm", r"duststorm", r"\bsandstorm\b", r"blowing dust"],
          evidence=["warning"], priority=4),
    _spec("snow_warning", ["categorical"], "categorical",
          [r"heavy snow", r"snowfall warning", r"\bblizzard\b"],
          evidence=["warning"], priority=4),
    _spec("rainfall_distribution", ["categorical"], "categorical",
          [r"widespread", r"scattered", r"isolated", r"fairly widespread",
           r"rainfall distribution"], evidence=["forecast", "warning"], priority=4),
)

CANONICAL_VARIABLES: tuple[str, ...] = tuple(dict.fromkeys(
    [spec.canonical for spec in VARIABLE_SPECS] + ["other"]))

# canonical variable -> the unit families that are physically compatible with
# it.  Used as a hard veto: no statistical model, however confident, is allowed
# to map a field into a variable whose declared unit contradicts it.
ALLOWED_UNIT_FAMILIES: dict[str, tuple[str, ...]] = {}
for _spec in VARIABLE_SPECS:
    ALLOWED_UNIT_FAMILIES.setdefault(_spec.canonical, tuple(_spec.families))
del _spec

_COMPILED = [
    (spec,
     [re.compile(pattern, re.I) for pattern in spec.keywords],
     [re.compile(pattern, re.I) for pattern in spec.anti_keywords])
    for spec in VARIABLE_SPECS
]


@dataclass
class FieldLabel:
    canonical_variable: str
    statistic: str
    accumulation_hours: Optional[float]
    vertical_level: str
    evidence_class: str
    unit_family: str
    confidence: float
    matched_on: str = ""

    def as_row(self) -> dict:
        return {
            "canonical_variable": self.canonical_variable,
            "statistic": self.statistic,
            "accumulation_hours": self.accumulation_hours,
            "vertical_level": self.vertical_level,
            "evidence_class": self.evidence_class,
            "unit_family": self.unit_family,
            "label_confidence": self.confidence,
            "matched_on": self.matched_on,
        }


# --- accumulation / level parsing from real table metadata -------------------
_ACC_PATTERNS = (
    (re.compile(r"(\d+)\s*-\s*(\d+)\s*hour\s+acc", re.I), "range"),
    (re.compile(r"\bacc\w*\s+over\s+(\d+)\s*h", re.I), "single"),
    (re.compile(r"\b(\d+)\s*h(?:ou)?r(?:ly)?\s+(?:acc|total|sum|precip)", re.I), "single"),
    (re.compile(r"\bpast\s+(\d+)\s*h", re.I), "single"),
    (re.compile(r"/(\d+)\s*h\b", re.I), "single"),
    (re.compile(r"\((\d+)\s*h\)", re.I), "single"),
)


def parse_accumulation_hours(text: str) -> Optional[float]:
    """Pull a real accumulation window out of a source table's time-range string.

    Handles the NCEP inventory form (`0-3 hour acc fcst`) and the suffix forms
    that appear in provider field names (`precip (mm/24h)`).
    """
    if not text:
        return None
    for pattern, kind in _ACC_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if kind == "range":
            start, stop = float(match.group(1)), float(match.group(2))
            if stop > start:
                return stop - start
            continue
        value = float(match.group(1))
        if 0 < value <= 720:
            return value
    if re.search(r"\bdaily\b|\b24\s*hour", text, re.I):
        return 24.0
    return None


_LEVEL_PATTERNS = (
    (re.compile(r"\b2\s*m(?:etre|eter)?\b|\bat 2 m\b", re.I), "2m"),
    (re.compile(r"\b10\s*m(?:etre|eter)?\b|\bat 10 m\b", re.I), "10m"),
    (re.compile(r"mean sea level|\bmsl\b", re.I), "mean_sea_level"),
    (re.compile(r"\bsurface\b|\bground\b", re.I), "surface"),
    (re.compile(r"\b(\d+)\s*(?:hpa|mb)\b", re.I), "pressure_level"),
    (re.compile(r"\bsoil\b|\bunderground\b|\bdepth below land\b", re.I), "soil"),
    (re.compile(r"\bcloud (?:base|top)\b|\bentire atmosphere\b|\bcolumn\b", re.I), "column"),
    (re.compile(r"\btropopause\b", re.I), "tropopause"),
)


def parse_vertical_level(text: str, default: str = "surface") -> str:
    if not text:
        return default
    for pattern, level in _LEVEL_PATTERNS:
        if pattern.search(text):
            return level
    return default


def classify_native_field(
    name: str,
    *,
    description: str = "",
    unit: Optional[str] = None,
    level_text: str = "",
    time_range_text: str = "",
    grib_statistical_processing: Optional[int] = None,
    evidence_class_hint: Optional[str] = None,
) -> Optional[FieldLabel]:
    """Label one native field from its own source-table metadata, or abstain.

    Returns `None` when the description does not clearly identify a canonical
    variable, or when the declared unit family contradicts the best keyword
    match.  Abstention is deliberate: fabricating a mapping is the failure mode
    this whole project exists to prevent.
    """
    haystack = " ".join(part for part in (description or "", name or "", level_text or "",
                                          time_range_text or "") if part).strip()
    if not haystack:
        return None

    family = unit_family(unit)
    candidates: list[tuple[int, int, VariableSpec, str]] = []
    for spec, patterns, anti in _COMPILED:
        if any(pattern.search(haystack) for pattern in anti):
            continue
        hit = next((pattern.pattern for pattern in patterns if pattern.search(haystack)), None)
        if hit is None:
            continue
        family_ok = family in spec.families or family == "unknown"
        if evidence_class_hint == "warning" and "warning" not in spec.evidence_classes:
            continue
        if evidence_class_hint != "warning" and spec.evidence_classes == ("warning",):
            continue
        # A unit contradiction is disqualifying, not a penalty: this is exactly
        # the `TMAX (mm)` failure the previous corpus was full of.
        if not family_ok:
            continue
        candidates.append((spec.priority, len(hit), spec, hit))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, spec, matched = candidates[0]

    statistic = spec.default_statistic
    if grib_statistical_processing is not None:
        mapped = GRIB_STAT_PROCESSING.get(int(grib_statistical_processing))
        if mapped and spec.default_statistic not in ("probability", "categorical"):
            statistic = mapped

    accumulation = parse_accumulation_hours(f"{time_range_text} {name} {description}")
    if statistic == "accumulation" and accumulation is None and family in ("length", "precip_depth"):
        # A depth-unit accumulation with no declared window is genuinely ambiguous;
        # keep it unset so the semantic gate refuses to compare it.
        accumulation = None
    if statistic != "accumulation":
        accumulation = None

    level = parse_vertical_level(f"{level_text} {name} {description}", default=spec.level)
    evidence = evidence_class_hint or (
        "warning" if spec.evidence_classes == ("warning",) else "forecast")

    ambiguity = len([c for c in candidates if c[0] == candidates[0][0]])
    confidence = 1.0 if ambiguity == 1 else max(0.55, 1.0 - 0.15 * (ambiguity - 1))

    return FieldLabel(
        canonical_variable=spec.canonical,
        statistic=statistic,
        accumulation_hours=accumulation,
        vertical_level=level,
        evidence_class=evidence,
        unit_family=family,
        confidence=round(confidence, 3),
        matched_on=matched,
    )


# --- label texts -------------------------------------------------------------
# Used as the *label side* of the M1 bi-encoder: the model scores a native field
# against these sentences rather than against an opaque softmax row, which is
# what lets the rare official-warning classes work from a handful of examples.
LABEL_DESCRIPTIONS: dict[str, str] = {
    "precipitation_amount": "accumulated precipitation amount, total rain depth collected over a time window, in millimetres or kilograms per square metre",
    "precipitation_probability": "probability of precipitation, the chance that measurable rain occurs, expressed as a percentage",
    "precipitation_rate": "instantaneous precipitation rate, how fast rain is falling right now, in millimetres per hour",
    "snowfall_amount": "snowfall amount, depth or water equivalent of snow that fell over a time window",
    "snow_depth": "snow depth, the thickness of snow lying on the ground",
    "temperature_2m": "air temperature at two metres above the ground",
    "temperature_max": "maximum air temperature reached over a period",
    "temperature_min": "minimum air temperature reached over a period",
    "dewpoint_2m": "dew point temperature at two metres above the ground",
    "apparent_temperature": "apparent or feels-like temperature, heat index and wind chill",
    "soil_temperature": "soil temperature measured at a depth below the land surface",
    "sea_surface_temperature": "sea surface temperature of the ocean skin",
    "wind_speed": "scalar wind speed near the surface",
    "wind_gust": "maximum wind gust speed over a period",
    "wind_direction": "direction the wind is blowing from, in degrees",
    "wind_u": "eastward zonal u component of the wind vector",
    "wind_v": "northward meridional v component of the wind vector",
    "humidity": "relative humidity as a percentage of saturation",
    "specific_humidity": "specific humidity or water vapour mixing ratio, mass of vapour per mass of air",
    "pressure_msl": "atmospheric pressure reduced to mean sea level",
    "pressure_surface": "atmospheric pressure at the surface or station level",
    "solar_radiation": "downward shortwave solar radiation flux at the surface",
    "sunshine_duration": "duration of bright sunshine over a period",
    "cloud_cover": "total cloud cover as a fraction of the sky",
    "visibility": "horizontal visibility distance at the surface",
    "soil_moisture": "soil moisture, volumetric water content of the soil",
    "evapotranspiration": "evaporation and evapotranspiration water loss from land and plants",
    "cape": "convective available potential energy, instability fuel for thunderstorms",
    "cin": "convective inhibition, the energy barrier suppressing convection",
    "lightning_density": "lightning flash density or flash rate",
    "thunderstorm_probability": "probability that a thunderstorm occurs",
    "wave_height": "significant height of combined wind waves and swell at sea",
    "wave_period": "mean period of ocean surface waves",
    "heavy_rain_warning": "official heavy, very heavy or extremely heavy rainfall warning issued by a meteorological authority",
    "thunderstorm_warning": "official thunderstorm, lightning or squall warning issued by a meteorological authority",
    "cyclone_warning": "official cyclone, depression or tropical storm warning issued by a meteorological authority",
    "heat_warning": "official heat wave warning issued by a meteorological authority",
    "cold_wave_warning": "official cold wave, cold day or ground frost warning issued by a meteorological authority",
    "fog_warning": "official dense fog or low visibility warning issued by a meteorological authority",
    "hail_warning": "official hail or hailstorm warning issued by a meteorological authority",
    "flood_warning": "official flood or river inundation warning issued by an authority",
    "marine_warning": "official marine warning, gale warning or rough sea advisory for fishermen",
    "dust_storm_warning": "official dust storm, sandstorm or blowing dust warning",
    "snow_warning": "official heavy snow or blizzard warning",
    "rainfall_distribution": "categorical rainfall distribution over an area, widespread scattered or isolated",
    "other": "not a mappable surface weather variable, a model state variable, a coordinate, an identifier or an unrelated quantity",
}

assert set(LABEL_DESCRIPTIONS) >= set(CANONICAL_VARIABLES), (
    f"missing label text for {set(CANONICAL_VARIABLES) - set(LABEL_DESCRIPTIONS)}")

VERTICAL_LEVELS: tuple[str, ...] = (
    "surface", "2m", "10m", "mean_sea_level", "pressure_level", "soil", "column",
    "sea_surface", "tropopause", "height_above_ground", "cloud_base", "cloud_top",
    "boundary_layer", "other",
)
EVIDENCE_CLASSES: tuple[str, ...] = (
    "forecast", "observation", "reanalysis", "nowcast", "warning", "climatology",
    "advisory", "radar", "satellite",
)
