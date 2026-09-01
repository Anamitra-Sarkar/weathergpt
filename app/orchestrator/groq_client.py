"""Groq client that respects user's 4-model preference. No llama."""
from __future__ import annotations
import os
import asyncio
import random
import httpx
from .models import model_for, ORCHESTRATOR_MODEL

GROQ_API = "https://api.groq.com/openai/v1/chat/completions"

def _key() -> str:
    return (os.getenv("GROQ_API_KEY") or "").strip()

async def generate(messages, *, model: str | None = None, temperature: float = 0.35, max_tokens: int = 1024, role: str = "explainer_agent") -> str:
    key = _key()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (env or groq_api.txt)")
    use_model = model or model_for(role)
    # safety: never allow llama
    if "llama" in use_model.lower():
        raise ValueError(f"llama models deprecated per user — got {use_model}")
    async with httpx.AsyncClient(timeout=40) as client:
        for attempt in range(3):
            try:
                r = await client.post(GROQ_API, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                      json={"model": use_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
                if r.status_code not in {429, 500, 502, 503, 504}:
                    r.raise_for_status()
                    j = r.json()
                    return j["choices"][0]["message"]["content"] or ""
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise
            if attempt == 2:
                r.raise_for_status()
            await asyncio.sleep((0.4 * (2 ** attempt)) + random.uniform(0, 0.2))
    raise RuntimeError("Groq retries exhausted")

async def orchestrator_generate(messages, **kw):
    return await generate(messages, model=ORCHESTRATOR_MODEL, role="orchestrator", **kw)

# quick self-test
if __name__ == "__main__":
    import asyncio
    async def _t():
        for role in ["intent_parser","forecast_agent","solution_agent","reviewer_agent","explainer_agent","history_agent"]:
            txt = await generate([{"role":"user","content":f"Say you are {role} in 2 words"}], role=role, max_tokens=10)
            print(f"{role} via {model_for(role)} -> {txt.strip()[:60]!r}")
    asyncio.run(_t())
