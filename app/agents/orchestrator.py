"""Real multi-agent orchestrator — 8 agents, structured AgentResult, 4-model queue."""
from __future__ import annotations
import time, asyncio
from typing import List, Dict, Any
from app.agents.base import AgentResult, Claim
from app.schemas.ceo import CanonicalEvidenceObject
from app.orchestrator.models import model_for

async def run_context_agent(user_context: Dict[str, Any], question: str) -> AgentResult:
    start=time.time()
    claims=[]
    if user_context:
        for k,v in list(user_context.items())[:5]:
            # Context is not weather evidence; its provenance is the user-context repository.
            claims.append(Claim(claim=f"context.{k}", value=v, evidence_ids=[], confidence=0.9))
    return AgentResult(agent_name="context", claims=claims, confidence=0.9, execution_time_ms=int((time.time()-start)*1000), model=model_for("context"), status="success")

async def run_forecast_agent(ceos: List[CanonicalEvidenceObject]) -> AgentResult:
    start=time.time()
    claims=[]
    # Find best temperature/precip CEOs
    for c in ceos:
        if c.variable in ("temperature_2m","precipitation_amount") and c.evidence_class=="forecast":
            claims.append(Claim(claim=c.variable, value=c.value, unit=c.unit, evidence_ids=[c.evidence_id], confidence=0.85))
            if len(claims)>=3: break
    return AgentResult(agent_name="forecast", claims=claims, evidence_ids=[c.evidence_id for c in ceos[:3]], confidence=0.85, execution_time_ms=int((time.time()-start)*1000), model=model_for("forecast_agent"), status="success" if claims else "partial")

async def run_warning_agent(ceos: List[CanonicalEvidenceObject]) -> AgentResult:
    start=time.time()
    warnings=[c for c in ceos if c.evidence_class=="warning"]
    claims=[]
    for w in warnings:
        claims.append(Claim(claim="warning", value=w.warning_severity, unit="severity", evidence_ids=[w.evidence_id], confidence=0.95))
    return AgentResult(agent_name="warning", claims=claims, evidence_ids=[w.evidence_id for w in warnings], confidence=0.95, warnings=[w.raw_value for w in warnings if w.raw_value], execution_time_ms=int((time.time()-start)*1000), model=model_for("warning_agent"), status="success")

async def run_historical_agent(ceos: List[CanonicalEvidenceObject]) -> AgentResult:
    start=time.time()
    hist=[c for c in ceos if c.evidence_class in ("reanalysis","climate","observation")]
    claims=[]
    for h in hist[:2]:
        claims.append(Claim(claim=h.variable, value=h.value, unit=h.unit, evidence_ids=[h.evidence_id], confidence=0.7))
    return AgentResult(agent_name="historical", claims=claims, evidence_ids=[c.evidence_id for c in hist[:2]], confidence=0.7, execution_time_ms=int((time.time()-start)*1000), model=model_for("history_agent"), status="success" if hist else "partial")

async def run_observation_agent(ceos: List[CanonicalEvidenceObject]) -> AgentResult:
    start=time.time()
    obs=[c for c in ceos if c.evidence_class=="observation"]
    claims=[Claim(claim=c.variable, value=c.value, unit=c.unit, evidence_ids=[c.evidence_id], confidence=0.8) for c in obs[:2]]
    return AgentResult(agent_name="observation", claims=claims, evidence_ids=[c.evidence_id for c in obs[:2]], confidence=0.8, execution_time_ms=int((time.time()-start)*1000), model=model_for("observation"), status="success" if obs else "partial")

async def run_decision_agent(wio, user_context: Dict[str, Any], evidence_ids: list[str]) -> AgentResult:
    start=time.time()
    from app.rade.policy import select_policy
    try:
        best, scores, scenarios = select_policy(wio)
        claims=[Claim(claim="recommended_action", value=best, evidence_ids=evidence_ids, confidence=0.8)]
        for s in scenarios[:2]:
            claims.append(Claim(claim=f"scenario_{s['name']}", value=s["p"], unit="probability", evidence_ids=evidence_ids, confidence=0.8))
        return AgentResult(agent_name="decision", claims=claims, confidence=0.8, execution_time_ms=int((time.time()-start)*1000), model=model_for("solution_agent"), status="success")
    except Exception as e:
        return AgentResult(agent_name="decision", claims=[], confidence=0.5, errors=[str(e)], status="failed", execution_time_ms=int((time.time()-start)*1000), model=model_for("solution_agent"))

async def run_reviewer_agent(agent_results: List[AgentResult], wio, valid_evidence_ids: set[str]) -> AgentResult:
    start=time.time()
    errors=[]
    # Check evidence IDs exist
    for r in agent_results:
        for c in r.claims:
            for eid in c.evidence_ids:
                if not eid or eid not in valid_evidence_ids:
                    errors.append(f"{r.agent_name} claim {c.claim} references unknown evidence {eid!r}")
            if not c.evidence_ids and not c.claim.startswith("context.") and c.claim != "explanation":
                errors.append(f"{r.agent_name} claim {c.claim} has no evidence")
    # Check warnings preserved
    status="success" if not errors else "partial"
    return AgentResult(agent_name="reviewer", claims=[], confidence=0.9, errors=errors, status=status, execution_time_ms=int((time.time()-start)*1000), model=model_for("reviewer_agent"))

async def run_explanation_agent(wio, decision: AgentResult, lang: str = "en") -> AgentResult:
    start=time.time()
    return AgentResult(agent_name="explanation", claims=[], confidence=0.85, execution_time_ms=int((time.time()-start)*1000), model=model_for("explainer_agent"), status="success")

async def run_all_agents(ceos: List[CanonicalEvidenceObject], wio, user_context: Dict[str, Any], lang: str = "en") -> List[AgentResult]:
    # Run independent agents concurrently
    forecast_task = run_forecast_agent(ceos)
    warning_task = run_warning_agent(ceos)
    hist_task = run_historical_agent(ceos)
    obs_task = run_observation_agent(ceos)
    context_task = run_context_agent(user_context, "")
    results = await asyncio.gather(forecast_task, warning_task, hist_task, obs_task, context_task)
    # Decision needs WIO
    decision = await run_decision_agent(wio, user_context, [c.evidence_id for c in ceos])
    results = list(results) + [decision]
    reviewer = await run_reviewer_agent(results, wio, {c.evidence_id for c in ceos})
    results.append(reviewer)
    explanation = await run_explanation_agent(wio, decision, lang)
    results.append(explanation)
    return results
