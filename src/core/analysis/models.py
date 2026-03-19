from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SongDNA:
    source_path: str
    sample_rate: int
    duration_seconds: float
    tempo_bpm: float
    key: dict[str, Any]
    structure: dict[str, Any] = field(default_factory=dict)
    energy: dict[str, Any] = field(default_factory=dict)
    stems: dict[str, Any] = field(default_factory=lambda: {"enabled": False, "files": {}})
    musical_intelligence: dict[str, Any] = field(default_factory=dict)
    analysis_version: str = "0.1.0"
    metadata: dict[str, Any] = field(default_factory=lambda: {"schema_version": "0.1.0"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
