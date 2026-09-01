from __future__ import annotations

from app.schemas.ceo import CanonicalEvidenceObject


class EvidenceStore:
    """Process-local evidence index, deliberately keyed by server-generated IDs only."""
    def __init__(self) -> None:
        self._items: dict[str, CanonicalEvidenceObject] = {}

    def add_many(self, items: list[CanonicalEvidenceObject]) -> None:
        self._items.update({item.evidence_id: item for item in items})

    def get(self, evidence_id: str) -> CanonicalEvidenceObject | None:
        return self._items.get(evidence_id)


evidence_store = EvidenceStore()
