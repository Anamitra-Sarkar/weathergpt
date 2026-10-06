"""Optional FastAPI router for the events service (fastapi is imported only when you call `create_router` / `create_app`).

    from weathergpt_events.api import create_router
    app.include_router(create_router(service), prefix="/events")

Routes: GET /status, GET /catalogue, GET /forecast?lat=&lon=&targets=a,b&horizon_days=3, POST /tool {"name": ..., "arguments": {...}}.
A refused or unavailable forecast is HTTP 200 with `available: false` and the reason (it is an answer, not a server error);
a malformed tool call is HTTP 422.  The run fetch is blocking, so routes are plain `def` (FastAPI runs them in a thread pool).
"""
# NB: no `from __future__ import annotations` here: FastAPI must resolve the locally defined request model from real annotations.
from weathergpt_events.service import ForecastService, ToolCallError


def create_router(service: ForecastService):
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field

    class ToolCall(BaseModel):
        name: str
        arguments: dict = Field(default_factory=dict)

    router = APIRouter(tags=["weathergpt-events"])

    @router.get("/status")
    def status():
        return service.status()

    @router.get("/catalogue")
    def catalogue():
        return {"tools": service.catalogue()}

    @router.get("/forecast")
    def forecast(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180),
                 targets: str = Query("", description="comma-separated model names; empty = all served"),
                 horizon_days: int = Query(10, ge=1, le=10)):
        names = [t for t in targets.split(",") if t] or None
        return service.forecast(lat, lon, targets=names, horizon_days=horizon_days)

    @router.post("/tool")
    def tool(call: ToolCall):
        try:
            return service.call_tool(call.name, call.arguments)
        except ToolCallError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return router


def create_app(service: ForecastService):
    from fastapi import FastAPI
    app = FastAPI(title="WeatherGPT events", version="1.0")
    app.include_router(create_router(service))
    return app
