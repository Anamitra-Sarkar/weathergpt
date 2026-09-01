"""WIO v2 — fused weather state with distributions."""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ForecastDistribution(BaseModel):
    variable: str
    probability: Optional[float] = None
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    mean: Optional[float] = None
    spread: Optional[float] = None
    unit: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)

class WIOv2Forecast(BaseModel):
    temperature: Optional[ForecastDistribution] = None
    precipitation: Optional[ForecastDistribution] = None
    wind: Optional[ForecastDistribution] = None
    other: Dict[str, ForecastDistribution] = Field(default_factory=dict)

class WIOv2Warning(BaseModel):
    active: bool = False
    warnings: List[Dict[str, Any]] = Field(default_factory=list)  # list of warning CEOs

class WIOv2(BaseModel):
    request_id: Optional[str] = None
    location: Dict[str, Any] = Field(default_factory=dict)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    forecast: WIOv2Forecast = Field(default_factory=WIOv2Forecast)
    warnings: WIOv2Warning = Field(default_factory=WIOv2Warning)
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    historical: List[Dict[str, Any]] = Field(default_factory=list)
    agreement: float = 0.0
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    quality: Optional[str] = None
    fusion_metadata: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    wio_version: str = "2.0"
