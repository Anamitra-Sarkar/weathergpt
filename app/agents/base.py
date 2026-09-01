"""AgentResult — structured output for all agents."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Claim(BaseModel):
    claim: str
    value: Any
    unit: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.8
    extra: Dict[str, Any] = Field(default_factory=dict)

class AgentResult(BaseModel):
    agent_name: str
    claims: List[Claim] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.8
    uncertainty: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    execution_time_ms: int = 0
    model: Optional[str] = None
    status: str = "success"  # success, partial, failed
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
