from __future__ import annotations

from ..analysis.models import SongDNA
from .compatibility import build_compatibility_report
from .models import ChildArrangementPlan, ParentReference, PlannedSection


def build_stub_arrangement_plan(song_a: SongDNA, song_b: SongDNA) -> ChildArrangementPlan:
    report = build_compatibility_report(song_a, song_b)

    sections = [
        PlannedSection(label="intro", start_bar=0, bar_count=8, source_parent="A", source_section_label="section_0", target_energy=0.25, transition_out="lift"),
        PlannedSection(label="build", start_bar=8, bar_count=8, source_parent="B", source_section_label="section_0", target_energy=0.55, transition_in="blend", transition_out="swap"),
        PlannedSection(label="payoff", start_bar=16, bar_count=16, source_parent="A", source_section_label="section_1", target_energy=0.85, transition_in="drop"),
    ]

    return ChildArrangementPlan(
        parents=[
            ParentReference(song_a.source_path, song_a.tempo_bpm, str(song_a.key.get("tonic", "unknown")), str(song_a.key.get("mode", "unknown")), song_a.duration_seconds),
            ParentReference(song_b.source_path, song_b.tempo_bpm, str(song_b.key.get("tonic", "unknown")), str(song_b.key.get("mode", "unknown")), song_b.duration_seconds),
        ],
        compatibility=report.factors,
        sections=sections,
        planning_notes=[
            "Stub planner output: replace with phrase-safe search over section and bar artifacts.",
            "Planner should eventually choose sections from compatibility graph rather than fixed template placeholders.",
        ],
    )
