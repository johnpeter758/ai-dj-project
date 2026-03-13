from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ParentId = Literal["A", "B"]
TransitionType = Literal["cut", "blend", "swap", "lift", "drop"]


@dataclass(slots=True)
class ResolverConfig:
    sample_rate: int = 44100
    beats_per_bar: int = 4
    bars_per_phrase: int = 4
    direct_snap_tolerance_sec: float = 0.120
    conditional_snap_tolerance_sec: float = 0.250
    blend_beats_default: float = 4.0
    blend_beats_max: float = 8.0
    low_end_crossover_hz: float = 120.0
    min_clip_beats: int = 8


@dataclass(slots=True)
class ParentGrid:
    parent_id: ParentId
    source_path: str
    tempo_bpm: float
    beat_times: list[float]
    phrase_boundaries_seconds: list[float]
    duration_seconds: float


@dataclass(slots=True)
class SourceSectionRef:
    parent_id: ParentId
    source_path: str
    source_section_label: str | None
    raw_start_sec: float
    raw_end_sec: float
    snapped_start_sec: float
    snapped_end_sec: float


@dataclass(slots=True)
class TargetSectionTiming:
    start_bar: int
    bar_count: int
    start_sec: float
    end_sec: float
    duration_sec: float
    anchor_bpm: float


@dataclass(slots=True)
class ResolvedSection:
    index: int
    label: str
    source_parent: ParentId
    source: SourceSectionRef
    target: TargetSectionTiming
    foreground_owner: ParentId
    background_owner: ParentId | None
    low_end_owner: ParentId
    vocal_policy: str
    allowed_overlap: bool
    overlap_beats_max: float
    collapse_if_conflict: bool
    transition_in: TransitionType | None = None
    transition_out: TransitionType | None = None
    transition_mode: str | None = None
    stretch_ratio: float = 1.0
    semitone_shift: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AudioWorkOrder:
    order_id: str
    section_index: int
    order_type: str
    parent_id: ParentId
    role: str
    source_path: str
    source_start_sec: float
    source_end_sec: float
    target_start_sec: float
    target_duration_sec: float
    stretch_ratio: float
    semitone_shift: float
    gain_db: float
    fade_in_sec: float
    fade_out_sec: float
    transition_type: TransitionType | None
    transition_mode: str | None
    foreground_state: str
    low_end_state: str
    vocal_state: str
    conflict_policy: str


@dataclass(slots=True)
class ResolvedRenderPlan:
    schema_version: str
    sample_rate: int
    target_bpm: float
    sections: list[ResolvedSection]
    work_orders: list[AudioWorkOrder]
    warnings: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
