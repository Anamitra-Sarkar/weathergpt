"""API v1 — versioned, mobile-friendly, Pydantic validated."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from app.context.store import upsert_fact, get_context, add_feedback

router = APIRouter(prefix="/api/v1")

class ContextRequest(BaseModel):
    user_id: str
    fact: str
    value: Any
    confidence: float = 0.9
    source: str = "user"
    confirmed: bool = True

class FeedbackRequest(BaseModel):
    user_id: str
    decision: str
    forecast: str
    actual: str
    feedback: str

@router.post("/context")
async def post_context(req: ContextRequest):
    upsert_fact(req.user_id, req.fact, req.value, req.confidence, req.source, req.confirmed)
    return {"status": "ok", "user_id": req.user_id, "fact": req.fact}

@router.get("/context/{user_id}")
async def get_context_api(user_id: str):
    return {"user_id": user_id, "context": get_context(user_id)}

@router.post("/feedback")
async def post_feedback(req: FeedbackRequest):
    add_feedback(req.user_id, req.decision, req.forecast, req.actual, req.feedback)
    return {"status": "ok"}

@router.get("/health")
async def health_v1(request: Request):
    # Reuse main health but versioned
    from app.main import health
    h = await health()
    h["api_version"] = "v1"
    return h
