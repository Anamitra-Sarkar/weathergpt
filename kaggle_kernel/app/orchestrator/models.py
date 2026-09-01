"""
Groq model routing for WeatherGPT orchestrator.
- Orchestrator uses qwen/qwen3.8-27b (user-specified)
- Sub-agents round-robin across the 4 user-preferred Groq models
- Llama models are NOT used (deprecated per user)
"""
from __future__ import annotations

# User's exact preferences (in priority order)
PREFERRED_MODELS = [
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

ORCHESTRATOR_MODEL = "qwen/qwen3.8-27b"

# Sub-agent → model mapping (round-robin, queue if >4 agents)
AGENT_ROLES = [
    "intent_parser",
    "location_resolver",
    "forecast_agent",
    "history_agent",
    "warning_agent",
    "solution_agent",
    "reviewer_agent",
    "explainer_agent",
]

def model_for(role: str | int) -> str:
    """Return Groq model for a given agent role or index."""
    if isinstance(role, int):
        return PREFERRED_MODELS[role % len(PREFERRED_MODELS)]
    # role is string name
    if role == "orchestrator":
        return ORCHESTRATOR_MODEL
    try:
        idx = AGENT_ROLES.index(role)
    except ValueError:
        idx = hash(role) % len(PREFERRED_MODELS)
    return PREFERRED_MODELS[idx % len(PREFERRED_MODELS)]

def plan_for_num_agents(n: int):
    """For n agents, return list of models in order (queues through 4)."""
    return [PREFERRED_MODELS[i % len(PREFERRED_MODELS)] for i in range(n)]

def describe_routing():
    lines = [f"orchestrator → {ORCHESTRATOR_MODEL}"]
    for role in AGENT_ROLES:
        lines.append(f"{role:20s} → {model_for(role)}")
    return "\n".join(lines)
