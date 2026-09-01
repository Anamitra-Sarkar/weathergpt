"""
Canonical Evidence Object (CEO) — interoperability envelope.
Preserves original source; never silently averages incompatible values.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid

class EvidenceSource(str, Enum):
    IMD = "IMD"
    GFS = "GFS"
    WRF = "WRF"
    ERA5 = "ERA5"
    INSAT = "INSAT"
    RADAR = "RADAR"
    CAP = "CAP"
    OPEN_METEO = "OPEN_METEO"
    GEFS = "GEFS"
    NASA_POWER = "NASA_POWER"
    OTHER = "OTHER"

class EvidenceClass(str, Enum):
    observation = "observation"
    forecast = "forecast"
    nowcast = "nowcast"
    warning = "warning"
    radar = "radar"
    satellite = "satellite"
    climate = "climate"
    advisory = "advisory"
    reanalysis = "reanalysis"
    climatology = "climatology"

class CanonicalVariable(str, Enum):
    precipitation_amount = "precipitation_amount"
    precipitation_probability = "precipitation_probability"
    precipitation_rate = "precipitation_rate"
    temperature_2m = "temperature_2m"
    temperature_max = "temperature_max"
    temperature_min = "temperature_min"
    wind_speed = "wind_speed"
    wind_gust = "wind_gust"
    wind_direction = "wind_direction"
    humidity = "humidity"
    pressure_msl = "pressure_msl"
    cloud_cover = "cloud_cover"
    thunderstorm_probability = "thunderstorm_probability"
    rainfall_distribution = "rainfall_distribution"
    heavy_rain_warning = "heavy_rain_warning"
    thunderstorm_warning = "thunderstorm_warning"
    cyclone_warning = "cyclone_warning"
    heat_warning = "heat_warning"
    flood_warning = "flood_warning"
    marine_warning = "marine_warning"
    visibility = "visibility"
    other = "other"

class Statistic(str, Enum):
    instant = "instant"
    accumulation = "accumulation"
    mean = "mean"
    max = "max"
    min = "min"
    probability = "probability"
    categorical = "categorical"

class GeometryType(str, Enum):
    Point = "Point"
    Polygon = "Polygon"
    GridCell = "GridCell"
    RasterCell = "RasterCell"
    Route = "Route"

class Geometry(BaseModel):
    type: GeometryType = Field(default=GeometryType.Point)
    coordinates: Any = Field(default=None, description="lon/lat, polygon ring, or reference id")
    reference: Optional[str] = None  # e.g. district code, basin id

class Provenance(BaseModel):
    original_source: str
    original_field: Optional[str] = None
    original_unit: Optional[str] = None
    transformations: List[str] = Field(default_factory=list)
    raw_record_id: Optional[str] = None

class CanonicalEvidenceObject(BaseModel):
    model_config = {"protected_namespaces": ()}
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: EvidenceSource
    source_type: Optional[str] = Field(default=None, description="e.g. api, gridded, reanalysis")
    source_record_id: Optional[str] = None
    evidence_class: EvidenceClass
    variable: CanonicalVariable
    value: Optional[float] = None
    raw_value: Optional[Any] = None  # original categorical / string value
    unit: Optional[str] = None
    statistic: Statistic = Field(default=Statistic.instant)
    geometry: Geometry = Field(default_factory=Geometry)
    spatial_resolution: Optional[str] = None
    observed_at: Optional[datetime] = None
    issued_at: Optional[datetime] = None
    model_initialization_time: Optional[datetime] = None
    forecast_reference_time: Optional[datetime] = Field(default=None, description="alias for model_initialization_time")
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    forecast_lead_hours: Optional[float] = None
    accumulation_window_hours: Optional[float] = None
    vertical_level: Optional[str] = Field(default="surface")
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    ensemble_member: Optional[int] = None
    probability: Optional[float] = Field(default=None, ge=0, le=1)
    quality_flag: Optional[str] = None
    quality: Optional[str] = Field(default=None, description="alias for quality_flag")
    confidence: Optional[str] = None
    warning_severity: Optional[str] = None  # green/yellow/orange/red
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    retrieval_timestamp: Optional[datetime] = Field(default=None, description="when retrieved from source")
    provenance: Provenance
    parent_ids: List[str] = Field(default_factory=list, description="parent evidence IDs if derived")
    transformation: Optional[str] = Field(default=None, description="transformation performed")
    transformation_timestamp: Optional[datetime] = None
    algorithm_version: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
