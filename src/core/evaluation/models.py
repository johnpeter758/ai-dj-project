from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ListenSubscore:
    score: float
    summary: str
    evidence: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ListenReport:
    source_path: str
    duration_seconds: float
    overall_score: float
    structure: ListenSubscore
    groove: ListenSubscore
    energy_arc: ListenSubscore
    transition: ListenSubscore
    coherence: ListenSubscore
    mix_sanity: ListenSubscore
    song_likeness: ListenSubscore
    verdict: str
    top_reasons: list[str] = field(default_factory=list)
    top_fixes: list[str] = field(default_factory=list)
    gating: dict[str, Any] = field(default_factory=dict)
    analysis_version: str = "0.2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "duration_seconds": self.duration_seconds,
            "overall_score": self.overall_score,
            "structure": self.structure.to_dict(),
            "groove": self.groove.to_dict(),
            "energy_arc": self.energy_arc.to_dict(),
            "transition": self.transition.to_dict(),
            "coherence": self.coherence.to_dict(),
            "mix_sanity": self.mix_sanity.to_dict(),
            "song_likeness": self.song_likeness.to_dict(),
            "verdict": self.verdict,
            "top_reasons": self.top_reasons,
            "top_fixes": self.top_fixes,
            "gating": self.gating,
            "analysis_version": self.analysis_version,
        }
