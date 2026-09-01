"""
WeatherGPT FastAPI — interoperability + agentic retrieval layer.
 boots in mock mode without any API keys (WIO + RADE still work).
"""
from __future__ import annotations
import os
import time
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.schemas.wio import QueryRequest, QueryResponse
from app.schemas.ceo import CanonicalEvidenceObject
from app.services.location_resolver import resolve_location
from app.services.time_parser import parse_time_window
from app.services.wio_builder import build_wio
from app.orchestrator.retrieval_planner import plan_retrieval
from app.decoders.open_meteo import fetch_open_meteo, decode_open_meteo
from app.decoders.imd_json import decode as decode_imd
from app.rade.policy import select_policy, explain_policy

app = FastAPI(title="WeatherGPT", version="1.0.0", description="Meteorological interoperability — CEO → WIO → RADE → LLM explain")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_start = time.time()

@app.get("/")
def root():
    return {"message": "WeatherGPT API — see /docs", "mock_mode": not bool(os.getenv("GROQ_API_KEY"))}

@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": int(time.time()-_start), "version": "1.0.0", "mock_mode": not bool(os.getenv("GROQ_API_KEY")), "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/plan")
def plan(q: str = "Will it rain in Nagpur tomorrow afternoon?", location: str = "Nagpur"):
    from app.services.time_parser import parse_time_window
    from app.services.location_resolver import resolve_location
    loc = resolve_location(location)
    vf, vt, hor = parse_time_window(q)
    classes = plan_retrieval(hor, q)
    return {"location": loc.model_dump(), "valid_from": vf.isoformat(), "valid_to": vt.isoformat(), "horizon": hor, "evidence_classes": classes}

async def _collect_ceos(question: str, loc, valid_from, valid_to, horizon: str) -> List[CanonicalEvidenceObject]:
    ceos: List[CanonicalEvidenceObject] = []

    # 1) Open-Meteo (always available, no key) — primary NWP source in demo
    try:
        payload = await fetch_open_meteo(loc.lat, loc.lon, forecast_days=5)
        ceos.extend(decode_open_meteo(payload, loc.lat, loc.lon))
    except Exception as e:
        print(f"[weathergpt] open-meteo fetch failed: {e}")

    # 2) IMD fixture (if you later wire real IMD, replace this block with real fetch)
    # For now we inject a synthetic warning when horizon=short to demo warning preservation
    # Remove once real IMD warnings are wired.
    try:
        # demo warning CEOS — only to show that warnings stay separate
        # comment out if you don't want synthetic warnings
        pass
    except Exception:
        pass

    # 3) Optional: decode any IMD fixtures in training/datasets/imd_samples.jsonl if present
    try:
        from pathlib import Path
        p = Path("training/datasets/imd_samples.jsonl")
        if p.exists():
            import json
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ceos.extend(decode_imd(rec, product=rec.get("_product","forecast")))
    except Exception as e:
        print(f"[weathergpt] imd fixtures skipped: {e}")

    return ceos

def _filter_ceos_to_window(ceos, valid_from, valid_to):
    from app.services.temporal_align import filter_by_window, to_utc
    q_from = valid_from.astimezone(timezone.utc)
    q_to = valid_to.astimezone(timezone.utc)
    return filter_by_window(ceos, q_from, q_to)

@app.post("/wio/query", response_model=QueryResponse)
async def wio_query(req: QueryRequest):
    q = req.question
    raw_loc = (req.location or {}).get("raw") or (req.location or {}).get("district") or "Nagpur"
    if "lat" in (req.location or {}) and "lon" in (req.location or {}):
        from app.schemas.location import ResolvedLocation
        loc = ResolvedLocation(raw=raw_loc, lat=float(req.location["lat"]), lon=float(req.location["lon"]))
    else:
        loc = resolve_location(raw_loc)
    valid_from, valid_to, horizon = parse_time_window(q)
    if req.horizon_hint:
        horizon = req.horizon_hint

    ceos = await _collect_ceos(q, loc, valid_from, valid_to, horizon)
    ceos_window = _filter_ceos_to_window(ceos, valid_from, valid_to)
    # if filtering left nothing, fall back to all (so demo always returns something)
    if not ceos_window and ceos:
        ceos_window = ceos[:20]

    # optional bias-correction hook: if model artifact exists, apply
    try:
        from pathlib import Path
        if (Path("training/models/bias_correction/best.pt")).exists():
            # lightweight: no heavy load in request path, just demo hook
            pass
    except Exception:
        pass

    wio = build_wio(q, loc.model_dump(), valid_from, valid_to, horizon, ceos_window, lang=req.lang)
    return QueryResponse(answer=None, wio=wio, evidence_count=len(ceos_window), warnings=[wio.official_warning] if wio.official_warning.active else [], lang=req.lang)

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # first build WIO
    resp = await wio_query(req)
    wio = resp.wio

    # RADE advice
    try:
        best, scores, scenarios = select_policy(wio)
        rade_note = explain_policy(best, scores, scenarios, wio)
    except Exception as e:
        best, rade_note = "wait", f"RADE unavailable: {e}"

    # LLM explainer — only gets WIO + RADE, never raw CEOS/GRIB
    # Uses 4 user-approved Groq models, queued round-robin; orchestrator = qwen/qwen3.8-27b, no llama
    lang = req.lang or "en"
    # try to load Groq key from env or file
    groq_key = os.getenv("GROQ_API_KEY") or ""
    if not groq_key:
        for _p in ["/home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt", "/home/anamitra/groq_api.txt"]:
            try:
                groq_key = open(_p).read().strip()
                if groq_key:
                    os.environ["GROQ_API_KEY"] = groq_key
                    break
            except: pass
    if groq_key or os.getenv("GROQ_API_KEY"):
        try:
            from app.orchestrator.groq_client import generate as groq_generate
            from app.orchestrator.models import ORCHESTRATOR_MODEL
            # orchestrator builds the final explanation prompt from WIO + RADE
            wio_json = wio.model_dump(mode="json")
            prompt = (
                f"You are WeatherGPT orchestrator ({ORCHESTRATOR_MODEL}). Explain the WIO for a user in {lang}.\n"
                f"Rules: use ONLY WIO numbers, preserve warnings separately, never average incompatible values, state uncertainty.\n"
                f"WIO: {str(wio_json)[:8000]}\nRADE: {rade_note} best={best}\n"
                f"Respond in {lang}, concise, with provenance line."
            )
            answer = await groq_generate(
                [{"role":"system","content": f"You are WeatherGPT, orchestrator model {ORCHESTRATOR_MODEL}. Be concise, provenance-aware, no llama."},
                 {"role":"user","content": prompt}],
                role="explainer_agent", temperature=0.35, max_tokens=800
            )
            answer = answer.strip() or f"[WeatherGPT] {wio.weather.summary} RADE: {rade_note}"
        except Exception as e:
            answer = f"[WeatherGPT] {wio.weather.summary or 'Forecast ready.'} RADE: {rade_note} (Groq fallback: {e})"
    else:
        # mock explainer — deterministic, shows WIO was the source
        parts = [wio.weather.summary or "Weather intelligence ready."]
        if wio.official_warning.active:
            parts.append(f"⚠️ Official {wio.official_warning.authority} warning {wio.official_warning.severity} until {wio.official_warning.valid_until}.")
        parts.append(f"Advice: {best} — {rade_note}")
        if wio.agreement.notes:
            parts.append(f"Agreement: {wio.agreement.status} — {wio.agreement.notes}")
        parts.append(f"Evidence: {len(wio.evidence)} sources; horizon={wio.query.intent}. No LLM key — mock explanation (set GROQ_API_KEY for LLM).")
        answer = " ".join(parts)

    resp.answer = answer
    return resp

@app.post("/rade/advise")
async def rade_advise(req: QueryRequest):
    resp = await wio_query(req)
    wio = resp.wio
    best, scores, scenarios = select_policy(wio)
    note = explain_policy(best, scores, scenarios, wio)
    return {"wio": wio, "best_action": best, "scores": scores, "scenarios": scenarios, "explanation": note}
