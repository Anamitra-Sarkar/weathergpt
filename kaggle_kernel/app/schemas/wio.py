"""Weather Intelligence Object — compact LLM-safe summary."""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class WIOQuery(BaseModel):
    raw_text: str
    resolved_location: Optional[Dict[str, Any]] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    intent: Optional[str] = None  # e.g. precipitation, pesticide_spraying, marine
    lang: str = "en"

class WIOWeather(BaseModel):
    summary: str = ""
    rain: Optional[Dict[str, Any]] = None
    wind: Optional[Dict[str, Any]] = None
    temperature: Optional[Dict[str, Any]] = None
    humidity: Optional[Dict[str, Any]] = None
    other: Dict[str, Any] = Field(default_factory=dict)

class WIOWarning(BaseModel):
    active: bool = False
    authority: Optional[str] = None
    severity: Optional[str] = None
    event: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    areas: List[str] = Field(default_factory=list)
    provenance: Optional[Dict[str, Any]] = None

class WIOAgreement(BaseModel):
    status: str = "unknown"  # full_agreement | partial_agreement | conflict | insufficient_evidence
    notes: str = ""

class EvidenceSummary(BaseModel):
    evidence_id: str
    source: str
    evidence_class: str
    variable: str
    value: Optional[float] = None
    unit: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    provenance: Optional[Dict[str, Any]] = None

class WeatherIntelligenceObject(BaseModel):
    query: WIOQuery
    weather: WIOWeather = Field(default_factory=WIOWeather)
    official_warning: WIOWarning = Field(default_factory=WIOWarning)
    agreement: WIOAgreement = Field(default_factory=WIOAgreement)
    evidence: List[EvidenceSummary] = Field(default_factory=list)
    disagreements: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    wio_version: str = "1.0"

class QueryRequest(BaseModel):
    question: str
    location: Optional[Dict[str, Any]] = None  # {raw, lat, lon, district, pincode}
    lang: str = "en"
    profile: Optional[Dict[str, Any]] = None  # optional context for RADE
    horizon_hint: Optional[str] = None  # nowcast|short|medium|climate

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    wio: WeatherIntelligenceObject
    evidence_count: int = 0
    warnings: List[WIOWarning] = Field(default_factory=list)
    lang: str = "en"
