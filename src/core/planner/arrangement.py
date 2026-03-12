from __future__ import annotations

from dataclasses import dataclass

from ..analysis.models import SongDNA
from .compatibility import build_compatibility_report
from .models import ChildArrangementPlan, ParentReference, PlannedSection


@dataclass(frozen=True)
class _SectionCandidate:
    label: str
    start: float
    end: float
    duration: float
    midpoint: float


def _song_parent_ref(song: SongDNA) -> ParentReference:
    return ParentReference(
        song.source_path,
        song.tempo_bpm,
        str(song.key.get('tonic', 'unknown')),
        str(song.key.get('mode', 'unknown')),
        song.duration_seconds,
    )


def _section_candidates(song: SongDNA) -> list[_SectionCandidate]:
    out: list[_SectionCandidate] = []
    for idx, sec in enumerate(song.structure.get('sections', []) or []):
        start = float(sec.get('start', 0.0))
        end = float(sec.get('end', song.duration_seconds))
        if end <= start:
            continue
        label = str(sec.get('label') or f'section_{idx}')
        out.append(_SectionCandidate(label=label, start=start, end=end, duration=end-start, midpoint=(start+end)*0.5))
    if out:
        return out
    return [_SectionCandidate(label='section_0', start=0.0, end=float(song.duration_seconds), duration=float(song.duration_seconds), midpoint=float(song.duration_seconds) * 0.5)]


def _pick_candidate(song: SongDNA, target_position: str) -> _SectionCandidate:
    candidates = _section_candidates(song)
    if len(candidates) == 1:
        if target_position == 'late':
            # Preserve a missing-section fallback path for ultra-coarse analyses so
            # the resolver/tests still exercise unresolved-label behavior.
            return _SectionCandidate(label='section_1', start=candidates[0].start, end=candidates[0].end, duration=candidates[0].duration, midpoint=candidates[0].midpoint)
        return candidates[0]

    total = max(float(song.duration_seconds), 1e-6)
    anchors = {
        'early': total * 0.18,
        'mid': total * 0.50,
        'late': total * 0.78,
    }
    anchor = anchors[target_position]

    return min(
        candidates,
        key=lambda c: (
            abs(c.midpoint - anchor),
            -c.duration,
            c.start,
        ),
    )


def build_stub_arrangement_plan(song_a: SongDNA, song_b: SongDNA) -> ChildArrangementPlan:
    report = build_compatibility_report(song_a, song_b)

    intro_a = _pick_candidate(song_a, 'early')
    build_b = _pick_candidate(song_b, 'mid')
    payoff_a = _pick_candidate(song_a, 'late')

    sections = [
        PlannedSection(label='intro', start_bar=0, bar_count=8, source_parent='A', source_section_label=intro_a.label, target_energy=0.25, transition_out='lift'),
        PlannedSection(label='build', start_bar=8, bar_count=8, source_parent='B', source_section_label=build_b.label, target_energy=0.55, transition_in='blend', transition_out='swap'),
        PlannedSection(label='payoff', start_bar=16, bar_count=16, source_parent='A', source_section_label=payoff_a.label, target_energy=0.85, transition_in='drop'),
    ]

    notes = [
        'Phrase-aware stub planner: chooses early/mid/late section candidates from analyzed structure instead of fixed section placeholders.',
        'Planner should next rank candidate phrase windows using boundary confidence, energy arcs, and cross-parent compatibility.',
    ]
    if len(_section_candidates(song_a)) <= 1 or len(_section_candidates(song_b)) <= 1:
        notes.append('At least one song still has coarse section analysis; resolver phrase-safe fallback remains important.')

    return ChildArrangementPlan(
        parents=[_song_parent_ref(song_a), _song_parent_ref(song_b)],
        compatibility=report.factors,
        sections=sections,
        planning_notes=notes,
    )
