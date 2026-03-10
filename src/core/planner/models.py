from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CompatibilityFactors:
    tempo: float
    harmony: float
    structure: float
    energy: float
    stem_conflict: float
    notes: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        return (self.tempo + self.harmony + self.structure + self.energy + self.stem_conflict) / 5.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["overall"] = self.overall
        return payload


@dataclass(slots=True)
class ParentReference:
    source_path: str
    tempo_bpm: float
    key_tonic: str
    key_mode: str
    duration_seconds: float


@dataclass(slots=True)
class CompatibilityReport:
    parent_a: ParentReference
    parent_b: ParentReference
    factors: CompatibilityFactors
    analysis_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["factors"]["overall"] = self.factors.overall
        return payload


@dataclass(slots=True)
class PlannedSection:
    label: str
    start_bar: int
    bar_count: int
    source_parent: str
    source_section_label: str | None = None
    target_energy: float | None = None
    transition_in: str | None = None
    transition_out: str | None = None


@dataclass(slots=True)
class ChildArrangementPlan:
    parents: list[ParentReference]
    compatibility: CompatibilityFactors
    sections: list[PlannedSection]
    planning_notes: list[str] = field(default_factory=list)
    analysis_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compatibility"]["overall"] = self.compatibility.overall
        return payload
