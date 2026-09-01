from __future__ import annotations
from typing import List, Dict, Tuple
from .enumerator import enumerate_scenarios
from .utility import expected_utility, all_actions

def select_policy(wio) -> Tuple[str, Dict[str, float], List[Dict]]:
    scenarios = enumerate_scenarios(wio)
    scores = {a: expected_utility(a, scenarios) for a in all_actions()}
    # filter to non-dummy: for general weather not agriculture, reduce actions
    # keep all but rank
    best = max(scores, key=scores.get)
    return best, scores, scenarios

def explain_policy(best: str, scores: Dict[str, float], scenarios: List[Dict], wio) -> str:
    p_rain = next((s["p"] for s in scenarios if s["name"]=="rain"), 0)
    if best == "spray" and p_rain > 0.5:
        return f"Not recommended to spray now: rain probability {p_rain:.0%} — high wash-off risk."
    if best == "wait" and p_rain > 0.5:
        return f"Wait: rain likely ({p_rain:.0%}), spraying would be wasted."
    if best == "irrigate":
        return "Irrigation recommended: no significant rain expected."
    return f"Recommended action: {best} (expected utility {scores[best]:.1f})."
