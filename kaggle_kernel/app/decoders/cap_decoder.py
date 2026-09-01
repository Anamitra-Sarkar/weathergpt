"""CAP XML → CEOs. Preserves alert lifecycle (update/cancel)."""
from __future__ import annotations
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Any
from app.schemas.ceo import CanonicalEvidenceObject, Geometry, Provenance

NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}

def _text(el, tag):
    n = el.find(f"cap:{tag}", NS)
    if n is None:
        n = el.find(tag)
    return n.text.strip() if n is not None and n.text else None

def decode_cap_xml(xml_bytes: bytes) -> List[CanonicalEvidenceObject]:
    out: List[CanonicalEvidenceObject] = []
    root = ET.fromstring(xml_bytes)
    identifier = _text(root, "identifier") or "cap-unknown"
    sender = _text(root, "sender") or "CAP"
    sent = _text(root, "sent")
    msg_type = _text(root, "msgType") or "Alert"
    status = _text(root, "status") or "Actual"
    # handle lifecycle: if Cancel, mark severity cancelled
    is_cancel = msg_type.lower() == "cancel" or status.lower() == "cancel"
    try:
        issued = datetime.fromisoformat(sent.replace("Z","+00:00")) if sent else datetime.now(timezone.utc)
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=timezone.utc)
    except Exception:
        issued = datetime.now(timezone.utc)

    for info in root.findall("cap:info", NS) + root.findall("info"):
        event = _text(info, "event") or "weather"
        severity = (_text(info, "severity") or "Moderate").lower()
        # map CAP severity to colour
        colour_map = {"minor":"green","moderate":"yellow","severe":"orange","extreme":"red"}
        colour = colour_map.get(severity, "yellow")
        if is_cancel:
            colour = "cancelled"
        expires = _text(info, "expires")
        effective = _text(info, "effective")
        try:
            valid_from = datetime.fromisoformat(effective.replace("Z","+00:00")) if effective else issued
            valid_to = datetime.fromisoformat(expires.replace("Z","+00:00")) if expires else None
            for dt in (valid_from, valid_to):
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            valid_from = valid_from.astimezone(timezone.utc) if valid_from else issued
            valid_to = valid_to.astimezone(timezone.utc) if valid_to else None
        except Exception:
            valid_from, valid_to = issued, None
        # areas
        areas = []
        for area in info.findall("cap:area", NS) + info.findall("area"):
            desc = _text(area, "areaDesc") or "unknown"
            areas.append(desc)
        area_ref = ", ".join(areas) if areas else "unknown"
        geom = Geometry(type="Polygon", coordinates=None, reference=area_ref)
        out.append(CanonicalEvidenceObject(
            source="CAP", source_record_id=identifier, evidence_class="warning",
            variable="heavy_rain_warning", value=None, raw_value=event, unit=None, statistic="categorical",
            geometry=geom, issued_at=issued, valid_from=valid_from, valid_to=valid_to,
            warning_severity=colour,
            provenance=Provenance(original_source=sender, original_field="CAP info", transformations=["parsed CAP XML", f"msgType={msg_type} status={status}"]),
            extra={"cap_severity": severity, "cap_status": status, "cap_msgType": msg_type, "areas": areas}
        ))
    return out
