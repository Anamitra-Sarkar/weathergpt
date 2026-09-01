"""Disagreement detection — compare structured AgentResult claims."""
from __future__ import annotations
from typing import List, Dict, Any
from app.agents.base import AgentResult

def detect_disagreement(agent_results: List[AgentResult]) -> List[Dict[str, Any]]:
    disagreements=[]
    # Collect all claims by variable
    claims_by_var: Dict[str, List] = {}
    for r in agent_results:
        for c in r.claims:
            claims_by_var.setdefault(c.claim, []).append((r.agent_name, c))
    for var, claims in claims_by_var.items():
        if len(claims) <2:
            continue
        # Numeric: check spread
        try:
            vals=[float(c[1].value) for c in claims if isinstance(c[1].value,(int,float))]
            if len(vals)>=2:
                spread=max(vals)-min(vals)
                if spread>5:  # threshold for temp 5C, for precip 10mm etc. — simplified
                    disagreements.append({"variable":var, "type":"numeric", "spread":spread, "claims":[ (a,c.value) for a,c in claims]})
        except: pass
        # Categorical: check conflict
        vals_set=set(str(c[1].value) for c in claims)
        if len(vals_set)>1 and var=="warning":
            disagreements.append({"variable":var, "type":"categorical", "conflict": list(vals_set)})
    return disagreements
