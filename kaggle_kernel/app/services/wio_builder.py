from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional
from app.schemas.ceo import CanonicalEvidenceObject
from app.schemas.wio import WeatherIntelligenceObject, WIOQuery, WIOWeather, WIOWarning, WIOAgreement, EvidenceSummary
from app.services.ranker import rank, detect_disagreements

def build_wio(query_text: str, resolved_location: dict, valid_from, valid_to, horizon: str,
              ceos: List[CanonicalEvidenceObject], lang: str = "en") -> WeatherIntelligenceObject:
    now = datetime.now(timezone.utc)
    q_lat = resolved_location.get("lat", 21.14)
    q_lon = resolved_location.get("lon", 79.08)

    scored = rank(ceos, q_lat, q_lon, now)
    # weather panel — pick best evidence per variable
    weather = WIOWeather()
    # group by variable
    best_by_var = {}
    for score, ev in scored:
        if ev.variable not in best_by_var and ev.evidence_class != "warning":
            best_by_var[ev.variable] = (score, ev)

    # rain
    if "precipitation_amount" in best_by_var:
        _, ev = best_by_var["precipitation_amount"]
        # probability from matching prob CEO if available
        prob = None
        for s2, e2 in scored:
            if e2.variable == "precipitation_probability":
                prob = e2.probability if e2.probability is not None else (e2.value/100 if e2.value else None)
                break
        weather.rain = {
            "value_mm": ev.value,
            "unit": ev.unit or "mm",
            "accumulation_hours": ev.accumulation_window_hours,
            "probability": prob,
            "source": ev.source,
            "valid_from": ev.valid_from.isoformat() if ev.valid_from else None,
            "valid_to": ev.valid_to.isoformat() if ev.valid_to else None,
        }
        if prob is not None:
            if prob >= 0.6:
                weather.summary = f"Rain likely ({prob:.0%} probability, {ev.value:.1f} mm expected)."
            elif prob >= 0.3:
                weather.summary = f"Rain possible ({prob:.0%}, {ev.value:.1f} mm)."
            else:
                weather.summary = f"Rain unlikely ({prob:.0%})."
        else:
            weather.summary = f"Forecast precipitation {ev.value:.1f} mm ({ev.source})."

    if "temperature_2m" in best_by_var or "temperature_max" in best_by_var:
        key = "temperature_2m" if "temperature_2m" in best_by_var else "temperature_max"
        _, ev = best_by_var[key]
        weather.temperature = {"value": ev.value, "unit": ev.unit or "C", "source": ev.source}
    if "wind_speed" in best_by_var:
        _, ev = best_by_var["wind_speed"]
        weather.wind = {"value_kmh": round(ev.value*3.6,1) if ev.unit=="m/s" else ev.value, "source": ev.source}

    # warnings — preserve all warning CEOS separately, never averaged
    warnings = [e for e in ceos if e.evidence_class == "warning"]
    wio_warning = WIOWarning(active=len(warnings)>0)
    if warnings:
        # highest severity
        w = sorted(warnings, key=lambda x: {"green":0,"yellow":1,"orange":2,"red":3}.get((x.warning_severity or "yellow").lower(),1), reverse=True)[0]
        wio_warning = WIOWarning(
            active=True, authority=w.source, severity=w.warning_severity or "yellow",
            event=w.raw_value or w.variable, valid_from=w.valid_from, valid_until=w.valid_to,
            areas=[w.geometry.reference] if w.geometry and w.geometry.reference else [],
            provenance={"source_record_id": w.source_record_id, "transformations": w.provenance.transformations}
        )

    # agreement
    disagreements = detect_disagreements(scored)
    if not ceos:
        agreement = WIOAgreement(status="insufficient_evidence", notes="No evidence matched the query window.")
    elif disagreements:
        agreement = WIOAgreement(status="partial_agreement", notes="; ".join(disagreements))
    elif len(scored) >= 2:
        agreement = WIOAgreement(status="full_agreement", notes="Sources agree on occurrence and magnitude within tolerance.")
    else:
        agreement = WIOAgreement(status="single_source", notes="Only one source available for this window.")

    evidence_summaries = [
        EvidenceSummary(evidence_id=e.evidence_id, source=e.source, evidence_class=e.evidence_class,
                        variable=e.variable, value=e.value, unit=e.unit,
                        valid_from=e.valid_from, valid_to=e.valid_to,
                        provenance={"transformations": e.provenance.transformations, "original_unit": e.provenance.original_unit})
        for _, e in scored
    ]

    query = WIOQuery(raw_text=query_text, resolved_location=resolved_location,
                     valid_from=valid_from, valid_to=valid_to, intent=horizon, lang=lang)

    return WeatherIntelligenceObject(
        query=query, weather=weather, official_warning=wio_warning,
        agreement=agreement, evidence=evidence_summaries, disagreements=disagreements
    )
