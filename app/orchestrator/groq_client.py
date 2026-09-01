"""Groq client that respects user's 4-model preference. No llama."""
from __future__ import annotations
import os
import httpx
from .models import model_for, ORCHESTRATOR_MODEL

GROQ_API = "https://api.groq.com/openai/v1/chat/completions"

def _key() -> str:
    k = os.getenv("GROQ_API_KEY") or ""
    # also try reading from file if not in env (user gave /home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt)
    if not k:
        for p in ["/home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt", "/home/anamitra/groq_api.txt"]:
            try:
                k = open(p).read().strip()
                if k:
                    break
            except: pass
    return k.strip()

async def generate(messages, *, model: str | None = None, temperature: float = 0.35, max_tokens: int = 1024, role: str = "explainer_agent") -> str:
    key = _key()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (env or groq_api.txt)")
    use_model = model or model_for(role)
    # safety: never allow llama
    if "llama" in use_model.lower():
        raise ValueError(f"llama models deprecated per user — got {use_model}")
    async with httpx.AsyncClient(timeout=40) as client:
        r = await client.post(GROQ_API, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": use_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        r.raise_for_status()
        j = r.json()
        try:
            return j["choices"][0]["message"]["content"] or ""
        except Exception as e:
            raise RuntimeError(f"Groq bad response {j}") from e

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
