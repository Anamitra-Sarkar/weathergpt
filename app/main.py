"""WeatherGPT modular-monolith API.  Weather truth is assembled before language synthesis."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from app.adapters.registry import health_all
from app.agents.orchestrator import run_all_agents
from app.config import settings
from app.errors import WeatherGPTError
from app.orchestrator.retrieval_planner import build_retrieval_plan
from app.rade.v2 import decide
from app.schemas.api import ContextRequest, DecisionRequest, FeedbackRequest, LocationInput, QueryRequestV1
from app.schemas.location import ResolvedLocation
from app.schemas.wio import QueryRequest, QueryResponse
from app.services.cache import weather_cache
from app.services.disagreement import detect_disagreement
from app.services.evidence_store import evidence_store
from app.services.location_resolver import LocationAmbiguousError, LocationNotFoundError, extract_location, resolve_location
from app.services.retrieval import retrieve
from app.services.semantic_gate import validated_evidence
from app.services.temporal_align import filter_by_window
from app.services.time_parser import parse_time_window
from app.services.wio_builder import build_wio
from app.context.store import add_feedback, get_context, upsert_fact


app = FastAPI(title="WeatherGPT", version="2.0.0", description="Evidence-backed weather intelligence")
_started_at = time.monotonic()
_metrics: dict[str, float] = {"requests": 0, "errors": 0, "wio_latency_ms_total": 0, "rade_latency_ms_total": 0}


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-length") and int(request.headers["content-length"]) > settings.request_max_bytes:
            return JSONResponse(status_code=413, content={"error": {"code": "REQUEST_TOO_LARGE", "message": "Request exceeds configured size limit", "details": {}, "request_id": "unavailable"}})
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(round((time.monotonic() - started) * 1000, 1))
        return response


app.add_middleware(RequestIDMiddleware)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def _error(request: Request, error: WeatherGPTError) -> JSONResponse:
    _metrics["errors"] += 1
    return JSONResponse(status_code=error.status_code, content={"error": {"code": error.code, "message": error.message, "details": error.details, "request_id": request.state.request_id}})


@app.exception_handler(WeatherGPTError)
async def weathergpt_error(request: Request, exc: WeatherGPTError):
    return _error(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return _error(request, WeatherGPTError("VALIDATION_ERROR", "Invalid request", {"errors": exc.errors()}, 422))


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    return _error(request, WeatherGPTError("INTERNAL_ERROR", "Internal server error", {}, 500))


def _resolve_location(location: LocationInput | None, question: str) -> ResolvedLocation:
    try:
        if location and location.has_coordinates():
            return ResolvedLocation(raw=location.raw or "coordinates", lat=location.latitude, lon=location.longitude,
                                    confidence=1.0, source="request", normalized_name=location.raw, resolution_method="coordinates")
        if location and location.raw:
            return resolve_location(location.raw)
        extracted = extract_location(question)
        if extracted:
            return extracted
    except LocationAmbiguousError as exc:
        raise WeatherGPTError("LOCATION_AMBIGUOUS", "More than one location matches the request", {"candidates": exc.candidates}, 409) from exc
    except LocationNotFoundError as exc:
        raise WeatherGPTError("LOCATION_NOT_FOUND", str(exc), {"raw": exc.raw}, 404) from exc
    raise WeatherGPTError("LOCATION_REQUIRED", "Provide a city, pincode, or latitude/longitude; no location is stored or inferred.", {}, 422)


async def _weather_request(req: QueryRequestV1, request_id: str) -> tuple[Any, list, dict[str, Any], list, Any]:
    started = time.monotonic()
    location = _resolve_location(req.location, req.question)
    valid_from, valid_to, horizon, time_confidence = parse_time_window(req.question)
    plan = build_retrieval_plan(req.question, horizon)
    evidence, retrieval_status = await retrieve(plan, lat=location.lat, lon=location.lon, valid_from=valid_from, valid_to=valid_to)
    evidence = filter_by_window(evidence, valid_from.astimezone(timezone.utc), valid_to.astimezone(timezone.utc))
    evidence, semantic_rejections = validated_evidence(evidence)
    evidence_store.add_many(evidence)
    wio = build_wio(req.question, location.model_dump(), valid_from, valid_to, horizon, evidence, lang=req.language)
    wio.query.intent = plan.decision_context or horizon
    wio.query.resolved_location["time_resolution_confidence"] = time_confidence
    wio.query.resolved_location["retrieval_plan"] = plan.model_dump()
    wio.query.resolved_location["retrieval_status"] = retrieval_status
    if semantic_rejections:
        wio.agreement.notes = (wio.agreement.notes + " ").strip() + "Some incompatible evidence was rejected."
    profile = dict(req.profile)
    if req.user_id:
        profile.update({key: value["value"] for key, value in get_context(req.user_id).items() if key not in profile})
    agents = await run_all_agents(evidence, wio, profile, req.language)
    reviewer = next(result for result in agents if result.agent_name == "reviewer")
    if reviewer.status != "success":
        raise WeatherGPTError("REVIEW_FAILED", "Evidence-grounding review failed", {"errors": reviewer.errors}, 503)
    _metrics["wio_latency_ms_total"] += (time.monotonic() - started) * 1000
    return wio, evidence, retrieval_status, agents, profile


def _synthesize(wio, decision=None) -> str:
    parts: list[str] = []
    if wio.weather.summary:
        parts.append(wio.weather.summary)
    else:
        parts.append("No compatible weather evidence was available for the requested time window.")
    if wio.official_warning.active:
        parts.append(f"Official {wio.official_warning.severity} warning: {wio.official_warning.event}.")
    if decision:
        parts.append(f"Recommendation: {decision.recommended_action}. {decision.rationale}")
    parts.append(f"Confidence context: {wio.agreement.status}. Evidence IDs: {', '.join(e.evidence_id for e in wio.evidence) or 'none'}.")
    return " ".join(parts)


@app.get("/")
async def root():
    return {"service": "WeatherGPT", "version": app.version, "openapi": "/openapi.json"}


@app.get("/health")
@app.get("/api/v1/health")
async def health():
    sources = await health_all()
    return {"status": "ok", "liveness": True, "readiness": any(s.get("available") for s in sources.values()),
            "uptime_s": int(time.monotonic() - _started_at), "sources": sources,
            "database": {"available": True, "driver": "sqlite"}, "cache": weather_cache.status(),
            "llm": {"configured": bool(__import__("os").environ.get("GROQ_API_KEY")), "checked": False},
            "models": {"runtime_loading": "rule-based fallback only; artifacts are not trusted until registry validation"}}


@app.post("/wio/query")
@app.post("/api/v1/wio/query")
async def wio_query_v1(req: QueryRequestV1, request: Request):
    _metrics["requests"] += 1
    wio, evidence, retrieval_status, agents, _ = await _weather_request(req, request.state.request_id)
    return {"wio": wio, "evidence_count": len(evidence), "retrieval": retrieval_status, "agents": agents, "request_id": request.state.request_id}


@app.post("/query")
@app.post("/api/v1/query")
async def query_v1(req: QueryRequestV1, request: Request):
    _metrics["requests"] += 1
    wio, evidence, retrieval_status, agents, profile = await _weather_request(req, request.state.request_id)
    decision = None
    plan = build_retrieval_plan(req.question, wio.query.intent or "short")
    if plan.decision_context:
        decision = decide(wio, profile, req.question)
    return {"answer": _synthesize(wio, decision), "wio": wio, "decision": decision, "agents": agents,
            "retrieval": retrieval_status, "request_id": request.state.request_id}


@app.post("/decision")
@app.post("/rade/advise")
@app.post("/api/v1/decision")
async def decision_endpoint(req: DecisionRequest, request: Request):
    _metrics["requests"] += 1
    started = time.monotonic()
    wio, evidence, retrieval_status, agents, profile = await _weather_request(req, request.state.request_id)
    result = decide(wio, profile, req.decision_type or req.question)
    result.evidence_ids = [item.evidence_id for item in evidence]
    _metrics["rade_latency_ms_total"] += (time.monotonic() - started) * 1000
    return {"decision": result, "wio": wio, "agents": agents, "retrieval": retrieval_status, "request_id": request.state.request_id}


@app.get("/evidence/{evidence_id}")
async def get_evidence(evidence_id: str):
    evidence = evidence_store.get(evidence_id)
    if evidence is None:
        raise HTTPException(404, detail={"code": "EVIDENCE_NOT_FOUND", "message": "Evidence is absent or expired from this process"})
    return evidence


@app.post("/context")
@app.post("/api/v1/context")
async def post_context(req: ContextRequest):
    upsert_fact(req.user_id, req.fact.fact, req.fact.value, req.fact.confidence, req.fact.source, req.fact.confirmed,
                req.fact.expiry.isoformat() if req.fact.expiry else None)
    return {"status": "ok", "user_id": req.user_id, "fact": req.fact.fact}


@app.post("/feedback")
@app.post("/api/v1/feedback")
async def post_feedback(req: FeedbackRequest):
    add_feedback(req.user_id, req.decision_id or "unspecified", "stored with decision", str(req.actual_outcome), req.user_feedback or "")
    return {"status": "recorded"}


@app.get("/warnings/active")
async def active_warnings(location: str, question: str = "warnings today"):
    response = await _weather_request(QueryRequestV1(question=question, location=LocationInput(raw=location)), "warnings")
    wio, _, retrieval, _, _ = response
    return {"warnings": [wio.official_warning] if wio.official_warning.active else [], "retrieval": retrieval}


@app.get("/forecast")
async def forecast(location: str, question: str = "weather today"):
    wio, evidence, retrieval, _, _ = await _weather_request(QueryRequestV1(question=question, location=LocationInput(raw=location)), "forecast")
    return {"wio": wio, "retrieval": retrieval, "evidence_count": len(evidence)}


@app.get("/metrics")
async def metrics():
    requests = _metrics["requests"] or 1
    return {**_metrics, "wio_latency_ms_mean": _metrics["wio_latency_ms_total"] / requests,
            "rade_latency_ms_mean": _metrics["rade_latency_ms_total"] / requests, "cache": weather_cache.status()}
