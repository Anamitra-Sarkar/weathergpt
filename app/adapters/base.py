"""WeatherSourceAdapter — common interface for all weather sources."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.schemas.ceo import CanonicalEvidenceObject

class WeatherSourceAdapter(ABC):
    source_name: str
    supported_evidence_classes: List[str]
    supported_variables: List[str]

    @abstractmethod
    async def fetch(self, **kwargs) -> Any:
        """Fetch raw data from source. Must raise on unavailable/credential missing."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw: Any, **kwargs) -> List[CanonicalEvidenceObject]:
        """Convert raw source data into CEOs, preserving provenance."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return {available: bool, latency_ms, reason, last_success} without LLM."""
        raise NotImplementedError
