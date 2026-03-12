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
    energy: float
    origin: str


@dataclass(frozen=True)
class _SectionSpec:
    label: str
    start_bar: int
    bar_count: int
    target_energy: float
    source_parent_preference: str | None
    transition_in: str | None = None
    transition_out: str | None = None


def _song_parent_ref(song: SongDNA) -> ParentReference:
    return ParentReference(
        song.source_path,
        song.tempo_bpm,
        str(song.key.get('tonic', 'unknown')),
        str(song.key.get('mode', 'unknown')),
        song.duration_seconds,
    )


def _safe_float_list(values) -> list[float]:
    out: list[float] = []
    for value in values or []:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _window_energy(song: SongDNA, start: float, end: float) -> float:
    beat_times = _safe_float_list(song.energy.get('beat_times', []))
    beat_rms = _safe_float_list(song.energy.get('beat_rms', []))
    pairs = [(t, e) for t, e in zip(beat_times, beat_rms) if start <= t < end]
    if pairs:
        return sum(e for _, e in pairs) / len(pairs)
    return float(song.energy.get('summary', {}).get('mean_rms', 0.0))


def _section_candidates(song: SongDNA) -> list[_SectionCandidate]:
    out: list[_SectionCandidate] = []
    for idx, sec in enumerate(song.structure.get('sections', []) or []):
        start = float(sec.get('start', 0.0))
        end = float(sec.get('end', song.duration_seconds))
        if end <= start:
            continue
        label = str(sec.get('label') or f'section_{idx}')
        out.append(
            _SectionCandidate(
                label=label,
                start=start,
                end=end,
                duration=end - start,
                midpoint=(start + end) * 0.5,
                energy=_window_energy(song, start, end),
                origin='section',
            )
        )
    if out:
        return out
    return [
        _SectionCandidate(
            label='section_0',
            start=0.0,
            end=float(song.duration_seconds),
            duration=float(song.duration_seconds),
            midpoint=float(song.duration_seconds) * 0.5,
            energy=_window_energy(song, 0.0, float(song.duration_seconds)),
            origin='section',
        )
    ]


def _phrase_window_candidates(song: SongDNA, bar_count: int) -> list[_SectionCandidate]:
    phrase_boundaries = sorted(set(_safe_float_list(song.structure.get('phrase_boundaries_seconds', []))))
    if not phrase_boundaries:
        return []
    if phrase_boundaries[0] > 0.0:
        phrase_boundaries = [0.0, *phrase_boundaries]
    duration = float(song.duration_seconds)
    if phrase_boundaries[-1] < duration:
        phrase_boundaries.append(duration)

    available_phrase_spans = max(1, len(phrase_boundaries) - 1)
    phrases_needed = min(available_phrase_spans, max(1, round(bar_count / 4)))
    candidates: list[_SectionCandidate] = []
    max_start = max(0, len(phrase_boundaries) - phrases_needed - 1)
    for start_idx in range(max_start + 1):
        end_idx = start_idx + phrases_needed
        start = float(phrase_boundaries[start_idx])
        end = float(phrase_boundaries[end_idx])
        if end <= start:
            continue
        candidates.append(
            _SectionCandidate(
                label=f'phrase_{start_idx}_{end_idx}',
                start=start,
                end=end,
                duration=end - start,
                midpoint=(start + end) * 0.5,
                energy=_window_energy(song, start, end),
                origin='phrase_window',
            )
        )
    return candidates


def _pick_candidate(song: SongDNA, target_position: str, bar_count: int, target_energy: float) -> _SectionCandidate:
    phrase_candidates = _phrase_window_candidates(song, bar_count)
    section_candidates = _section_candidates(song)
    candidates = phrase_candidates or section_candidates

    if len(section_candidates) == 1 and not phrase_candidates:
        if target_position == 'late':
            return _SectionCandidate(
                label='section_1',
                start=section_candidates[0].start,
                end=section_candidates[0].end,
                duration=section_candidates[0].duration,
                midpoint=section_candidates[0].midpoint,
                energy=section_candidates[0].energy,
                origin='synthetic_missing_section',
            )
        return section_candidates[0]

    total = max(float(song.duration_seconds), 1e-6)
    anchors = {
        'early': total * 0.16,
        'mid': total * 0.50,
        'late': total * 0.80,
    }
    anchor = anchors[target_position]

    energies = [c.energy for c in candidates]
    min_energy = min(energies) if energies else 0.0
    max_energy = max(energies) if energies else 0.0
    energy_span = max(max_energy - min_energy, 1e-6)

    weights = {
        'early': {'position': 1.25, 'energy': 0.85, 'shape': 0.45},
        'mid': {'position': 1.0, 'energy': 1.15, 'shape': 0.8},
        'late': {'position': 0.75, 'energy': 1.35, 'shape': 1.1},
    }[target_position]

    def score(candidate: _SectionCandidate) -> tuple[float, int, float, float, float]:
        position_error = abs(candidate.midpoint - anchor) / total
        energy_error = abs(candidate.energy - target_energy) / energy_span
        normalized_energy = (candidate.energy - min_energy) / energy_span
        if target_position == 'early':
            shape_error = normalized_energy
        elif target_position == 'late':
            shape_error = 1.0 - normalized_energy
        else:
            shape_error = abs(normalized_energy - 0.6)
        blended_error = (
            (weights['position'] * position_error)
            + (weights['energy'] * energy_error)
            + (weights['shape'] * shape_error)
        )
        return (
            blended_error,
            0 if candidate.origin == 'phrase_window' else 1,
            position_error,
            energy_error,
            candidate.start,
        )

    return min(candidates, key=score)


def _choose_parent(preferred: str | None, song_a: SongDNA, song_b: SongDNA, target_energy: float) -> str:
    if preferred in {'A', 'B'}:
        return preferred
    a_delta = abs(_window_energy(song_a, 0.0, float(song_a.duration_seconds)) - target_energy)
    b_delta = abs(_window_energy(song_b, 0.0, float(song_b.duration_seconds)) - target_energy)
    return 'A' if a_delta <= b_delta else 'B'


def build_stub_arrangement_plan(song_a: SongDNA, song_b: SongDNA) -> ChildArrangementPlan:
    report = build_compatibility_report(song_a, song_b)

    section_specs = [
        _SectionSpec(label='intro', start_bar=0, bar_count=8, target_energy=0.25, source_parent_preference='A', transition_out='lift'),
        _SectionSpec(label='build', start_bar=8, bar_count=8, target_energy=0.55, source_parent_preference='B', transition_in='blend', transition_out='swap'),
        _SectionSpec(label='payoff', start_bar=16, bar_count=16, target_energy=0.85, source_parent_preference=None, transition_in='drop'),
    ]

    sections: list[PlannedSection] = []
    selection_notes: list[str] = []
    song_map = {'A': song_a, 'B': song_b}
    for spec in section_specs:
        parent_id = _choose_parent(spec.source_parent_preference, song_a, song_b, spec.target_energy)
        candidate = _pick_candidate(song_map[parent_id], spec.label if spec.label in {'early', 'mid', 'late'} else ('early' if spec.label == 'intro' else 'mid' if spec.label == 'build' else 'late'), spec.bar_count, spec.target_energy)
        sections.append(
            PlannedSection(
                label=spec.label,
                start_bar=spec.start_bar,
                bar_count=spec.bar_count,
                source_parent=parent_id,
                source_section_label=candidate.label,
                target_energy=spec.target_energy,
                transition_in=spec.transition_in,
                transition_out=spec.transition_out,
            )
        )
        selection_notes.append(f"{spec.label}: parent {parent_id} -> {candidate.label} ({candidate.origin}, {candidate.start:.1f}-{candidate.end:.1f}s, energy {candidate.energy:.3f})")

    notes = [
        'Phrase-window planner: selects phrase-aligned source windows when phrase boundaries are available, instead of relying only on coarse section placeholders.',
        'Resolver understands phrase_<start>_<end> labels and snaps them directly to analyzed phrase boundaries.',
        *selection_notes,
    ]
    if len(_section_candidates(song_a)) <= 1 or len(_section_candidates(song_b)) <= 1:
        notes.append('At least one song still has coarse section analysis; phrase-window labels now reduce fallback dependence when phrase boundaries exist.')

    return ChildArrangementPlan(
        parents=[_song_parent_ref(song_a), _song_parent_ref(song_b)],
        compatibility=report.factors,
        sections=sections,
        planning_notes=notes,
    )
