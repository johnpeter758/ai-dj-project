from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ChildSectionRecipe:
    backbone_owner: str
    donor_support_required: bool
    motif_anchor_parent: str
    motif_anchor_label: str | None
    motif_recurrence_strength: float
    tension_target: str
    rhythmic_constraint: str
    harmonic_constraint: str
    timbral_anchor: str
    support_parent: str | None = None
    support_mode: str | None = None
    support_gain_db: float | None = None
    integration_strength: float = 0.0
    policy_id: str = "section_recipe_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
