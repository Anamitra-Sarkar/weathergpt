"""Versioned public contracts.  Internal services do not accept untyped dictionaries."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class LocationInput(BaseModel):
    raw: str | None = Field(default=None, max_length=256)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    @field_validator("raw")
    @classmethod
    def strip_raw(cls, value: str | None) -> str | None:
        return value.strip() if value else value

    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None


class QueryRequestV1(BaseModel):
    question: str = Field(min_length=1, max_length=4096)
    location: LocationInput | None = None
    user_id: str | None = Field(default=None, max_length=128)
    language: str = Field(default="en", max_length=16)
    profile: dict[str, Any] = Field(default_factory=dict)
    timezone: str | None = Field(default=None, max_length=64)


class DecisionRequest(QueryRequestV1):
    decision_type: Literal["spray", "irrigate", "harvest", "travel", "marine"] | None = None


class ContextFactInput(BaseModel):
    fact: str = Field(min_length=1, max_length=64)
    value: Any
    confidence: float = Field(default=0.9, ge=0, le=1)
    source: Literal["user", "system", "import"] = "user"
    confirmed: bool = True
    expiry: datetime | None = None


class ContextRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    fact: ContextFactInput


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    decision_id: str | None = None
    actual_outcome: dict[str, Any]
    user_feedback: str | None = Field(default=None, max_length=2048)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
