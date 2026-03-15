from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class _RoleFeatures:
    start_idx: int
    end_idx: int
    position: float
    length_phrases: float
    energy: float
    normalized_energy: float
    energy_slope: float
    repetition: float
    novelty: float
    novelty_density: float
    section_progress: float
    tail_energy: float
    end_focus: float
    lift_strength: float
    plateau_stability: float
    headroom: float
    ramp_consistency: float
    groove_drive: float
    groove_stability: float
    hook_strength: float
    payoff_strength: float
    energy_confidence: float


@dataclass(frozen=True)
class _WindowSelection:
    parent_id: str
    song: SongDNA
    candidate: _SectionCandidate
    blended_error: float
    score_breakdown: dict[str, float]
    section_label: str | None = None


@dataclass(frozen=True)
class _PlannerListenFeedback:
    groove_confidence: float
    energy_arc_strength: float
    transition_readiness: float
    coherence_confidence: float
    payoff_readiness: float


@dataclass(frozen=True)
class _BackbonePlan:
    backbone_parent: str
    donor_parent: str
    backbone_score: float
    donor_score: float
    backbone_reasons: list[str]


ROLE_ALIAS = {
    'build': 'pre',
    'payoff': 'chorus_payoff',
}


ROLE_PRIOR_WEIGHTS = {
    'intro': {
        'position_low': 1.55,
        'position_high': 0.05,
        'energy_low': 1.20,
        'energy_high': 0.05,
        'slope_up': 0.30,
        'slope_down': 0.45,
        'repetition': 0.20,
        'novelty': 0.40,
        'section_early': 0.60,
        'section_late': 0.05,
    },
    'verse': {
        'position_low': 0.55,
        'position_mid': 0.65,
        'energy_mid': 1.00,
        'slope_flat': 0.55,
        'repetition': 0.95,
        'novelty': 0.35,
        'section_mid': 0.70,
    },
    'pre': {
        'position_mid': 0.85,
        'energy_mid_high': 1.15,
        'slope_up': 1.40,
        'repetition': 0.55,
        'novelty': 0.60,
        'section_mid': 0.80,
    },
    'chorus_payoff': {
        'position_high': 1.05,
        'energy_high': 1.50,
        'slope_up': 0.70,
        'slope_down': 0.10,
        'repetition': 1.20,
        'novelty': 0.70,
        'section_late': 1.00,
    },
    'bridge': {
        'position_high': 0.60,
        'energy_mid': 0.95,
        'energy_low': 0.35,
        'slope_down': 0.95,
        'repetition': 0.10,
        'novelty': 1.45,
        'section_late': 0.75,
    },
    'outro': {
        'position_high': 1.65,
        'energy_low': 1.10,
        'energy_high': 0.05,
        'slope_down': 1.35,
        'repetition': 0.45,
        'novelty': 0.20,
        'section_late': 1.55,
    },
}


SECTION_TARGET_POSITION = {
    'intro': 'early',
    'verse': 'mid',
    'build': 'mid',
    'payoff': 'late',
    'bridge': 'late',
    'outro': 'late',
}

_CONSERVATIVE_STRETCH_MIN = 0.75
_CONSERVATIVE_STRETCH_MAX = 1.25
_HARD_STRETCH_MIN = 0.60
_HARD_STRETCH_MAX = 1.40


def _song_parent_ref(song: SongDNA) -> ParentReference:
    return ParentReference(
        song.source_path,
        song.tempo_bpm,
        str(song.key.get('tonic', 'unknown')),
        str(song.key.get('mode', 'unknown')),
        song.duration_seconds,
    )


def _song_phrase_capacity(song: SongDNA) -> int:
    phrase_boundaries = sorted(set(_safe_float_list(song.structure.get('phrase_boundaries_seconds', []))))
    if phrase_boundaries:
        if phrase_boundaries[0] > 0.0:
            phrase_boundaries = [0.0, *phrase_boundaries]
        duration = float(song.duration_seconds)
        if phrase_boundaries[-1] < duration:
            phrase_boundaries.append(duration)
        return max(1, len(phrase_boundaries) - 1)

    sections = song.structure.get('sections', []) or []
    return max(1, len(sections))


def _song_phrase_energy_profile(song: SongDNA) -> list[float]:
    phrase_boundaries = sorted(set(_safe_float_list(song.structure.get('phrase_boundaries_seconds', []))))
    duration = float(song.duration_seconds)
    if phrase_boundaries:
        if phrase_boundaries[0] > 0.0:
            phrase_boundaries = [0.0, *phrase_boundaries]
        if phrase_boundaries[-1] < duration:
            phrase_boundaries.append(duration)
        profile = [
            _window_energy(song, start, end)
            for start, end in zip(phrase_boundaries[:-1], phrase_boundaries[1:])
            if end > start
        ]
        if profile:
            return profile

    return [_window_energy(song, 0.0, duration)]


def _song_extended_program_support(song: SongDNA) -> float:
    profile = _song_phrase_energy_profile(song)
    if len(profile) < 7:
        return 0.0

    total = len(profile)
    bridge_start = max(1, int(total * 0.56))
    bridge_end = max(bridge_start + 1, int(total * 0.78))
    final_start = max(bridge_end, int(total * 0.72))

    first_payoff = max(profile[max(1, int(total * 0.38)):max(2, int(total * 0.62))] or profile)
    bridge_floor = min(profile[bridge_start:bridge_end] or profile[-3:-1] or profile)
    final_payoff = max(profile[final_start:] or profile[-2:] or profile)

    energy_low = min(profile)
    energy_high = max(profile)
    energy_span = max(energy_high - energy_low, 1e-6)

    reset_depth = _clamp01((first_payoff - bridge_floor) / energy_span)
    relaunch_strength = _clamp01((final_payoff - bridge_floor) / energy_span)
    final_advantage = _clamp01((final_payoff - first_payoff) / energy_span + 0.5)

    feedback = _planner_listen_feedback(song)
    return _clamp01(
        0.35 * reset_depth
        + 0.35 * relaunch_strength
        + 0.10 * final_advantage
        + 0.10 * feedback.payoff_readiness
        + 0.10 * feedback.energy_arc_strength
    )


def _build_section_program(song_a: SongDNA, song_b: SongDNA) -> list[_SectionSpec]:
    capacity = max(_song_phrase_capacity(song_a), _song_phrase_capacity(song_b))

    compact = [
        _SectionSpec(label='intro', start_bar=0, bar_count=8, target_energy=0.25, source_parent_preference='A', transition_out='lift'),
        _SectionSpec(label='build', start_bar=8, bar_count=8, target_energy=0.55, source_parent_preference='B', transition_in='blend', transition_out='swap'),
        _SectionSpec(label='payoff', start_bar=16, bar_count=16, target_energy=0.85, source_parent_preference=None, transition_in='drop'),
    ]
    standard = [
        _SectionSpec(label='intro', start_bar=0, bar_count=8, target_energy=0.24, source_parent_preference='A', transition_out='lift'),
        _SectionSpec(label='verse', start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference='A', transition_in='blend', transition_out='lift'),
        _SectionSpec(label='build', start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference='B', transition_in='blend', transition_out='swap'),
        _SectionSpec(label='payoff', start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in='drop', transition_out='blend'),
        _SectionSpec(label='outro', start_bar=40, bar_count=8, target_energy=0.34, source_parent_preference='A', transition_in='blend'),
    ]
    extended = [
        _SectionSpec(label='intro', start_bar=0, bar_count=8, target_energy=0.22, source_parent_preference='A', transition_out='lift'),
        _SectionSpec(label='verse', start_bar=8, bar_count=8, target_energy=0.40, source_parent_preference='A', transition_in='blend', transition_out='lift'),
        _SectionSpec(label='build', start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference='B', transition_in='blend', transition_out='swap'),
        _SectionSpec(label='payoff', start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in='drop', transition_out='blend'),
        _SectionSpec(label='bridge', start_bar=40, bar_count=8, target_energy=0.52, source_parent_preference='A', transition_in='swap', transition_out='lift'),
        _SectionSpec(label='payoff', start_bar=48, bar_count=16, target_energy=0.90, source_parent_preference=None, transition_in='drop', transition_out='blend'),
        _SectionSpec(label='outro', start_bar=64, bar_count=8, target_energy=0.30, source_parent_preference='A', transition_in='blend'),
    ]

    if capacity >= 7:
        support_a = _song_extended_program_support(song_a)
        support_b = _song_extended_program_support(song_b)
        extended_support = max(support_a, support_b)
        shared_support = 0.5 * (support_a + support_b)
        if extended_support >= 0.42 and shared_support >= 0.36:
            return extended
        return standard
    if capacity >= 5:
        return standard
    return compact


def _choose_backbone_parent(song_a: SongDNA, song_b: SongDNA) -> _BackbonePlan:
    feedback_a = _planner_listen_feedback(song_a)
    feedback_b = _planner_listen_feedback(song_b)
    support_a = _song_extended_program_support(song_a)
    support_b = _song_extended_program_support(song_b)
    capacity_a = _song_phrase_capacity(song_a)
    capacity_b = _song_phrase_capacity(song_b)

    mean_rms_a = float(song_a.energy.get('summary', {}).get('mean_bar_rms', song_a.energy.get('summary', {}).get('mean_rms', 0.0)))
    mean_rms_b = float(song_b.energy.get('summary', {}).get('mean_bar_rms', song_b.energy.get('summary', {}).get('mean_rms', 0.0)))
    score_a = (
        0.32 * feedback_a.groove_confidence
        + 0.25 * feedback_a.coherence_confidence
        + 0.17 * feedback_a.transition_readiness
        + 0.14 * support_a
        + 0.07 * min(1.0, capacity_a / 8.0)
        + 0.05 * mean_rms_a
    )
    score_b = (
        0.32 * feedback_b.groove_confidence
        + 0.25 * feedback_b.coherence_confidence
        + 0.17 * feedback_b.transition_readiness
        + 0.14 * support_b
        + 0.07 * min(1.0, capacity_b / 8.0)
        + 0.05 * mean_rms_b
    )

    if abs(score_a - score_b) <= 0.01:
        backbone_parent = 'A' if mean_rms_a >= mean_rms_b else 'B'
    else:
        backbone_parent = 'A' if score_a >= score_b else 'B'
    donor_parent = 'B' if backbone_parent == 'A' else 'A'
    chosen_feedback = feedback_a if backbone_parent == 'A' else feedback_b
    chosen_support = support_a if backbone_parent == 'A' else support_b
    chosen_capacity = capacity_a if backbone_parent == 'A' else capacity_b
    chosen_score = score_a if backbone_parent == 'A' else score_b
    other_score = score_b if backbone_parent == 'A' else score_a

    reasons = [
        f"higher backbone score ({chosen_score:.3f} vs {other_score:.3f})",
        f"groove={chosen_feedback.groove_confidence:.3f}",
        f"coherence={chosen_feedback.coherence_confidence:.3f}",
        f"transition_readiness={chosen_feedback.transition_readiness:.3f}",
        f"extended_support={chosen_support:.3f}",
        f"phrase_capacity={chosen_capacity}",
    ]
    return _BackbonePlan(
        backbone_parent=backbone_parent,
        donor_parent=donor_parent,
        backbone_score=chosen_score,
        donor_score=other_score,
        backbone_reasons=reasons,
    )


def _program_with_backbone(section_specs: list[_SectionSpec], backbone_parent: str, donor_parent: str) -> list[_SectionSpec]:
    backbone_labels = {'intro', 'verse', 'bridge', 'outro'}
    donor_labels = {'build'}
    rewritten: list[_SectionSpec] = []
    for spec in section_specs:
        preference = spec.source_parent_preference
        if spec.label in backbone_labels:
            preference = backbone_parent
        elif spec.label in donor_labels:
            preference = donor_parent
        rewritten.append(
            _SectionSpec(
                label=spec.label,
                start_bar=spec.start_bar,
                bar_count=spec.bar_count,
                target_energy=spec.target_energy,
                source_parent_preference=preference,
                transition_in=spec.transition_in,
                transition_out=spec.transition_out,
            )
        )
    return rewritten


def _safe_float_list(values) -> list[float]:
    out: list[float] = []
    for value in values or []:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _window_energy(song: SongDNA, start: float, end: float) -> float:
    bar_times = _safe_float_list(song.energy.get('bar_times', []))
    bar_rms = _safe_float_list(song.energy.get('bar_rms', []))
    pairs = [(t, e) for t, e in zip(bar_times, bar_rms) if start <= t < end]
    if pairs:
        return sum(e for _, e in pairs) / len(pairs)

    beat_times = _safe_float_list(song.energy.get('beat_times', []))
    beat_rms = _safe_float_list(song.energy.get('beat_rms', []))
    pairs = [(t, e) for t, e in zip(beat_times, beat_rms) if start <= t < end]
    if pairs:
        return sum(e for _, e in pairs) / len(pairs)
    return float(song.energy.get('summary', {}).get('mean_bar_rms', song.energy.get('summary', {}).get('mean_rms', 0.0)))


def _window_energy_slope(song: SongDNA, start: float, end: float) -> float:
    bar_times = _safe_float_list(song.energy.get('bar_times', []))
    bar_rms = _safe_float_list(song.energy.get('bar_rms', []))
    window = [e for t, e in zip(bar_times, bar_rms) if start <= t < end]
    if len(window) >= 2:
        return float(window[-1] - window[0])

    beat_times = _safe_float_list(song.energy.get('beat_times', []))
    beat_rms = _safe_float_list(song.energy.get('beat_rms', []))
    window = [e for t, e in zip(beat_times, beat_rms) if start <= t < end]
    if len(window) >= 2:
        return float(window[-1] - window[0])
    return 0.0


def _candidate_phrase_indices(label: str) -> tuple[int, int] | None:
    parts = label.split('_')
    if len(parts) >= 3 and parts[0] == 'phrase':
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            return None
    return None


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


def _target_section_duration_seconds(song: SongDNA, bar_count: int) -> float:
    tempo = max(float(song.tempo_bpm or 0.0), 1e-6)
    return (60.0 / tempo) * 4.0 * bar_count



def _snap_time_to_available_grid(song: SongDNA, value: float, lower: float, upper: float) -> float:
    grid = sorted({
        *(_safe_float_list(song.energy.get('bar_times', [])) or []),
        *(_safe_float_list(song.energy.get('beat_times', [])) or []),
        float(lower),
        float(upper),
    })
    if not grid:
        return min(max(value, lower), upper)
    best = min(grid, key=lambda item: abs(item - value))
    return min(max(float(best), lower), upper)



def _append_candidate(candidates: list[_SectionCandidate], song: SongDNA, label: str, start: float, end: float, origin: str) -> None:
    if end <= start:
        return
    rounded = (round(start, 6), round(end, 6))
    if any((round(item.start, 6), round(item.end, 6)) == rounded for item in candidates):
        return
    candidates.append(
        _SectionCandidate(
            label=label,
            start=start,
            end=end,
            duration=end - start,
            midpoint=(start + end) * 0.5,
            energy=_window_energy(song, start, end),
            origin=origin,
        )
    )



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
    target_duration = _target_section_duration_seconds(song, bar_count)
    trim_floor = target_duration * 0.82
    trim_ceiling = target_duration * 1.18

    candidates: list[_SectionCandidate] = []
    max_start = max(0, len(phrase_boundaries) - phrases_needed - 1)
    for start_idx in range(max_start + 1):
        end_idx = start_idx + phrases_needed
        start = float(phrase_boundaries[start_idx])
        end = float(phrase_boundaries[end_idx])
        if end <= start:
            continue
        base_label = f'phrase_{start_idx}_{end_idx}'
        _append_candidate(candidates, song, base_label, start, end, 'phrase_window')

        raw_duration = end - start
        if raw_duration <= trim_ceiling:
            continue

        anchor_specs = [
            ('trim_tail', start, start + target_duration),
            ('trim_head', end - target_duration, end),
            ('trim_center', ((start + end) * 0.5) - (target_duration * 0.5), ((start + end) * 0.5) + (target_duration * 0.5)),
        ]
        for suffix, raw_start, raw_end in anchor_specs:
            snapped_start = _snap_time_to_available_grid(song, raw_start, start, end)
            snapped_end = _snap_time_to_available_grid(song, raw_end, snapped_start, end)
            trimmed_duration = snapped_end - snapped_start
            if trimmed_duration < trim_floor or trimmed_duration > trim_ceiling:
                continue
            _append_candidate(candidates, song, f'{base_label}_{suffix}', snapped_start, snapped_end, 'phrase_trim')
    return candidates


def _score_position_low(position: float) -> float:
    return max(0.0, 1.0 - position)


def _score_position_high(position: float) -> float:
    return max(0.0, position)


def _score_position_mid(position: float, center: float = 0.45, width: float = 0.35) -> float:
    return max(0.0, 1.0 - (abs(position - center) / max(width, 1e-6)))


def _score_energy_low(value: float) -> float:
    return max(0.0, 1.0 - value)


def _score_energy_high(value: float) -> float:
    return max(0.0, value)


def _score_energy_mid(value: float, center: float = 0.45, width: float = 0.35) -> float:
    return max(0.0, 1.0 - (abs(value - center) / max(width, 1e-6)))


def _score_energy_mid_high(value: float, center: float = 0.68, width: float = 0.28) -> float:
    return max(0.0, 1.0 - (abs(value - center) / max(width, 1e-6)))


def _score_slope_up(value: float) -> float:
    return max(0.0, value)


def _score_slope_down(value: float) -> float:
    return max(0.0, -value)


def _score_slope_flat(value: float) -> float:
    return max(0.0, 1.0 - abs(value))


def _window_profile(song: SongDNA, start: float, end: float, bins: int = 4, *, value_keys: tuple[str, ...] | None = None) -> list[float]:
    if value_keys is None:
        value_keys = ('bar_rms', 'beat_rms')
    times = _safe_float_list(song.energy.get('bar_times', [])) or _safe_float_list(song.energy.get('beat_times', []))
    values = []
    for key in value_keys:
        values = _safe_float_list(song.energy.get(key, []))
        if values:
            break
    window = [(t, e) for t, e in zip(times, values) if start <= t < end]
    if not window:
        mean_energy = _window_energy(song, start, end)
        return [mean_energy] * bins

    duration = max(end - start, 1e-6)
    out: list[float] = []
    for idx in range(bins):
        bin_start = start + (duration * idx / bins)
        bin_end = start + (duration * (idx + 1) / bins)
        chunk = [e for t, e in window if bin_start <= t < bin_end]
        if chunk:
            out.append(sum(chunk) / len(chunk))
        else:
            out.append(out[-1] if out else _window_energy(song, start, end))
    return out


def _signal_overlap_strength(song: SongDNA, candidate: _SectionCandidate, signal_key: str) -> float:
    derived = song.energy.get('derived', {}) or {}
    windows = derived.get(signal_key, []) or []
    best = 0.0
    for window in windows:
        start = float(window.get('start', 0.0))
        end = float(window.get('end', 0.0))
        overlap = min(candidate.end, end) - max(candidate.start, start)
        if overlap <= 0.0:
            continue
        coverage = overlap / max(candidate.duration, 1e-6)
        score = float(window.get('score', 0.0))
        best = max(best, min(1.0, coverage * score))
    return best


def _candidate_role_features(song: SongDNA, candidates: list[_SectionCandidate], candidate: _SectionCandidate) -> _RoleFeatures:
    phrase_lengths = []
    profiles = {}
    onset_profiles = {}
    energies = [c.energy for c in candidates]
    min_energy = min(energies) if energies else 0.0
    max_energy = max(energies) if energies else 0.0
    energy_span = max(max_energy - min_energy, 1e-6)

    slopes = []
    for item in candidates:
        indices = _candidate_phrase_indices(item.label)
        start_idx, end_idx = indices if indices is not None else (0, 1)
        phrase_lengths.append(max(1, end_idx - start_idx))
        slopes.append(_window_energy_slope(song, item.start, item.end))
        profiles[item.label] = _window_profile(song, item.start, item.end)
        onset_profiles[item.label] = _window_profile(song, item.start, item.end, value_keys=('onset_density', 'onset_strength'))

    slope_span = max(max(slopes) - min(slopes), 1e-6) if slopes else 1e-6
    candidate_indices = _candidate_phrase_indices(candidate.label)
    start_idx, end_idx = candidate_indices if candidate_indices is not None else (0, 1)
    candidate_profile = profiles[candidate.label]
    candidate_onset_profile = onset_profiles[candidate.label]

    similarity_scores: list[float] = []
    for other in candidates:
        if other.label == candidate.label:
            continue
        other_profile = profiles[other.label]
        diff = sum(abs(a - b) for a, b in zip(candidate_profile, other_profile)) / max(len(candidate_profile), 1)
        similarity = 1.0 - min(1.0, diff / energy_span)
        similarity_scores.append(similarity)
    repetition = max(similarity_scores) if similarity_scores else 0.0

    novelty_boundaries = _safe_float_list(song.structure.get('novelty_boundaries_seconds', []))
    novelty_hits = sum(1 for value in novelty_boundaries if candidate.start < value < candidate.end)
    novelty_density = min(1.0, novelty_hits / max(end_idx - start_idx, 1))
    novelty = min(1.0, 0.65 * (1.0 - repetition) + 0.35 * novelty_density)

    section_boundaries = _safe_float_list(song.structure.get('section_boundaries_seconds', []))
    section_hits = sum(1 for value in section_boundaries if candidate.start <= value < candidate.end)
    section_progress = min(1.0, section_hits / max(end_idx - start_idx, 1))

    slope = _window_energy_slope(song, candidate.start, candidate.end)
    normalized_slope = ((slope - min(slopes)) / slope_span) if slopes else 0.5
    normalized_slope = (normalized_slope * 2.0) - 1.0

    position = candidate.midpoint / max(float(song.duration_seconds), 1e-6)
    length_phrases = (end_idx - start_idx) / max(max(phrase_lengths), 1)
    head = candidate_profile[: max(1, len(candidate_profile) // 2)]
    tail = candidate_profile[len(candidate_profile) // 2 :] or candidate_profile[-1:]
    head_energy = sum(head) / max(len(head), 1)
    tail_energy = sum(tail) / max(len(tail), 1)
    profile_peak = max(candidate_profile) if candidate_profile else max(candidate.energy, 1e-6)
    end_focus = _clamp01(tail_energy / max(profile_peak, 1e-6))
    lift_strength = _clamp01((tail_energy - head_energy) / max(energy_span, 1e-6))
    tail_range = (max(tail) - min(tail)) if tail else energy_span
    plateau_stability = _clamp01(1.0 - (tail_range / max(energy_span, 1e-6)))
    headroom = _clamp01((profile_peak - head_energy) / max(energy_span, 1e-6))
    profile_steps = [candidate_profile[idx + 1] - candidate_profile[idx] for idx in range(len(candidate_profile) - 1)]
    if profile_steps:
        positive_ratio = sum(1 for step in profile_steps if step >= -1e-6) / len(profile_steps)
        largest_drop = max((max(0.0, -step) for step in profile_steps), default=0.0)
        ramp_consistency = _clamp01((0.65 * positive_ratio) + (0.35 * (1.0 - min(1.0, largest_drop / max(energy_span, 1e-6)))))
    else:
        ramp_consistency = 0.5

    onset_series_present = bool(_safe_float_list(song.energy.get('onset_density', [])) or _safe_float_list(song.energy.get('onset_strength', [])))
    if onset_series_present:
        onset_values = [value for profile in onset_profiles.values() for value in profile]
        onset_min = min(onset_values) if onset_values else 0.0
        onset_max = max(onset_values) if onset_values else 1.0
        onset_span = max(onset_max - onset_min, 1e-6)
        onset_mean = sum(candidate_onset_profile) / max(len(candidate_onset_profile), 1)
        groove_drive = _clamp01((onset_mean - onset_min) / onset_span)
        onset_steps = [abs(candidate_onset_profile[idx + 1] - candidate_onset_profile[idx]) for idx in range(len(candidate_onset_profile) - 1)]
        groove_stability = _clamp01(1.0 - ((sum(onset_steps) / max(len(onset_steps), 1)) / onset_span)) if onset_steps else 0.5
    else:
        groove_drive = 0.5
        groove_stability = 0.5

    return _RoleFeatures(
        start_idx=start_idx,
        end_idx=end_idx,
        position=position,
        length_phrases=length_phrases,
        energy=candidate.energy,
        normalized_energy=(candidate.energy - min_energy) / energy_span,
        energy_slope=normalized_slope,
        repetition=repetition,
        novelty=novelty,
        novelty_density=novelty_density,
        section_progress=section_progress,
        tail_energy=tail_energy,
        end_focus=end_focus,
        lift_strength=lift_strength,
        plateau_stability=plateau_stability,
        headroom=headroom,
        ramp_consistency=ramp_consistency,
        groove_drive=groove_drive,
        groove_stability=groove_stability,
        hook_strength=_signal_overlap_strength(song, candidate, 'hook_windows'),
        payoff_strength=_signal_overlap_strength(song, candidate, 'payoff_windows'),
        energy_confidence=float((song.energy.get('derived', {}) or {}).get('energy_confidence', 0.0)),
    )


def _role_prior_score(role: str, features: _RoleFeatures) -> float:
    canonical_role = ROLE_ALIAS.get(role, role)
    weights = ROLE_PRIOR_WEIGHTS.get(canonical_role)
    if not weights:
        return 0.0

    score = 0.0
    score += weights.get('position_low', 0.0) * _score_position_low(features.position)
    score += weights.get('position_high', 0.0) * _score_position_high(features.position)
    score += weights.get('position_mid', 0.0) * _score_position_mid(features.position)
    score += weights.get('energy_low', 0.0) * _score_energy_low(features.normalized_energy)
    score += weights.get('energy_high', 0.0) * _score_energy_high(features.normalized_energy)
    score += weights.get('energy_mid', 0.0) * _score_energy_mid(features.normalized_energy)
    score += weights.get('energy_mid_high', 0.0) * _score_energy_mid_high(features.normalized_energy)
    score += weights.get('slope_up', 0.0) * _score_slope_up(features.energy_slope)
    score += weights.get('slope_down', 0.0) * _score_slope_down(features.energy_slope)
    score += weights.get('slope_flat', 0.0) * _score_slope_flat(features.energy_slope)
    score += weights.get('repetition', 0.0) * features.repetition
    score += weights.get('novelty', 0.0) * features.novelty
    score += weights.get('section_early', 0.0) * _score_position_low(features.position)
    score += weights.get('section_mid', 0.0) * _score_position_mid(features.position, center=0.55, width=0.30)
    score += weights.get('section_late', 0.0) * _score_position_high(features.position)

    signal_confidence = 0.55 + (0.45 * features.energy_confidence)
    if canonical_role == 'chorus_payoff':
        score += 1.15 * features.payoff_strength * signal_confidence
        score += 0.55 * features.hook_strength * signal_confidence
        score += 0.85 * features.end_focus
        score += 0.55 * features.lift_strength
        score += 0.75 * features.plateau_stability
        score += 0.22 * features.groove_drive
        score += 0.16 * features.groove_stability
    elif canonical_role == 'pre':
        score += 0.35 * features.payoff_strength * signal_confidence
        score += 0.90 * features.lift_strength
        score += 0.45 * features.headroom
        score += 0.55 * features.ramp_consistency
        score += 0.30 * features.groove_drive
        score += 0.18 * features.groove_stability
        score += 0.10 * features.end_focus
        score -= 0.55 * (features.plateau_stability * features.end_focus)
        score -= 0.25 * max(0.0, features.end_focus - features.ramp_consistency)
    elif canonical_role == 'verse':
        score += 0.45 * features.hook_strength * signal_confidence
        score += 0.28 * features.groove_stability
        score += 0.18 * features.groove_drive
    elif canonical_role == 'bridge':
        score += 0.20 * features.hook_strength * signal_confidence
        score += 0.15 * features.groove_stability
    elif canonical_role == 'intro':
        score -= 0.15 * features.payoff_strength * signal_confidence

    return score


def _boundary_confidence(song: SongDNA, candidate: _SectionCandidate) -> float:
    phrase_boundaries = sorted(set(_safe_float_list(song.structure.get('phrase_boundaries_seconds', []))))
    if candidate.origin not in {'phrase_window', 'phrase_trim'} or not phrase_boundaries:
        return 0.35 if candidate.origin == 'section' else 0.20

    boundaries = list(phrase_boundaries)
    if boundaries[0] > 0.0:
        boundaries = [0.0, *boundaries]
    duration = float(song.duration_seconds)
    if boundaries[-1] < duration:
        boundaries.append(duration)

    def nearest_delta(value: float) -> float:
        return min(abs(value - boundary) for boundary in boundaries)

    tolerance = max(duration * 0.02, 0.25)
    start_score = max(0.0, 1.0 - (nearest_delta(candidate.start) / tolerance))
    end_score = max(0.0, 1.0 - (nearest_delta(candidate.end) / tolerance))

    section_boundaries = _safe_float_list(song.structure.get('section_boundaries_seconds', []))
    section_hits = sum(1 for boundary in section_boundaries if candidate.start <= boundary <= candidate.end)
    boundary_density = min(1.0, section_hits / max((candidate.end - candidate.start) / 8.0, 1.0))

    return min(1.0, 0.55 + (0.30 * ((start_score + end_score) * 0.5)) + (0.15 * boundary_density))


def _normalize_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    span = max(high - low, 1e-6)
    return {key: (value - low) / span for key, value in values.items()}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _planner_listen_feedback(song: SongDNA) -> _PlannerListenFeedback:
    derived = song.energy.get('derived', {}) or {}
    beat_times = _safe_float_list(song.metadata.get('tempo', {}).get('beat_times', []))
    beat_intervals = [max(1e-6, beat_times[idx + 1] - beat_times[idx]) for idx in range(len(beat_times) - 1)]
    if len(beat_intervals) >= 2:
        mean_interval = sum(beat_intervals) / len(beat_intervals)
        interval_variance = sum(abs(interval - mean_interval) for interval in beat_intervals) / len(beat_intervals)
        groove_confidence = _clamp01(1.0 - (interval_variance / max(mean_interval * 0.12, 1e-6)))
    else:
        groove_confidence = 0.45

    sections = song.structure.get('sections', []) or []
    phrase_boundaries = _safe_float_list(song.structure.get('phrase_boundaries_seconds', []))
    coherence_confidence = _clamp01(
        0.45 * min(1.0, len(sections) / 6.0)
        + 0.55 * min(1.0, max(0, len(phrase_boundaries) - 1) / 8.0)
    )

    energy_arc_strength = _clamp01(
        0.55 * float(derived.get('energy_confidence', 0.0))
        + 0.30 * float(derived.get('payoff_strength', 0.0))
        + 0.15 * float(derived.get('hook_repetition', 0.0))
    )
    transition_readiness = _clamp01(
        0.40 * groove_confidence
        + 0.30 * coherence_confidence
        + 0.20 * float(derived.get('energy_confidence', 0.0))
        + 0.10 * float(derived.get('hook_strength', 0.0))
    )
    payoff_readiness = _clamp01(
        0.50 * float(derived.get('payoff_strength', 0.0))
        + 0.30 * float(derived.get('hook_strength', 0.0))
        + 0.20 * float(derived.get('hook_repetition', 0.0))
    )
    return _PlannerListenFeedback(
        groove_confidence=groove_confidence,
        energy_arc_strength=energy_arc_strength,
        transition_readiness=transition_readiness,
        coherence_confidence=coherence_confidence,
        payoff_readiness=payoff_readiness,
    )


def _series_values(song: SongDNA, *keys: str) -> list[float]:
    for key in keys:
        values = _safe_float_list(song.energy.get(key, []))
        if values:
            return values
    return []


def _series_window_mean(song: SongDNA, keys: tuple[str, ...], start: float, end: float) -> float:
    values = _series_values(song, *keys)
    if not values:
        if keys == ('rms', 'beat_rms'):
            return _window_energy(song, start, end)
        return 0.0

    times = _safe_float_list(song.energy.get('beat_times', []))
    if len(times) == len(values):
        window = [value for t, value in zip(times, values) if start <= t < end]
        if window:
            return sum(window) / len(window)

    duration = max(float(song.duration_seconds), 1e-6)
    step = duration / max(len(values), 1)
    indexed = []
    for idx, value in enumerate(values):
        center = (idx + 0.5) * step
        if start <= center < end:
            indexed.append(value)
    if indexed:
        return sum(indexed) / len(indexed)

    center = 0.5 * (start + end)
    index = min(max(int(center / step), 0), len(values) - 1)
    return values[index]


def _normalized_delta(pre: float, post: float, floor: float = 1e-6) -> float:
    scale = max(abs(pre), abs(post), floor)
    return abs(post - pre) / scale


def _planner_seam_risk(previous: _WindowSelection | None, current_song: SongDNA, current_candidate: _SectionCandidate) -> tuple[float, dict[str, float]]:
    if previous is None:
        return 0.10, {
            'energy_jump': 0.0,
            'spectral_jump': 0.0,
            'onset_jump': 0.0,
            'low_end_crowding_risk': 0.0,
            'texture_shift': 0.0,
            'foreground_collision_risk': 0.0,
            'vocal_competition_risk': 0.0,
            'seam_risk': 0.10,
        }

    prev_song = previous.song
    prev_candidate = previous.candidate
    left_start = max(prev_candidate.start, prev_candidate.end - min(prev_candidate.duration, 8.0))
    right_end = min(current_candidate.end, current_candidate.start + min(current_candidate.duration, 8.0))

    pre_energy = _series_window_mean(prev_song, ('rms', 'beat_rms'), left_start, prev_candidate.end)
    post_energy = _series_window_mean(current_song, ('rms', 'beat_rms'), current_candidate.start, right_end)
    pre_centroid = _series_window_mean(prev_song, ('spectral_centroid',), left_start, prev_candidate.end)
    post_centroid = _series_window_mean(current_song, ('spectral_centroid',), current_candidate.start, right_end)
    pre_rolloff = _series_window_mean(prev_song, ('spectral_rolloff',), left_start, prev_candidate.end)
    post_rolloff = _series_window_mean(current_song, ('spectral_rolloff',), current_candidate.start, right_end)
    pre_onset = _series_window_mean(prev_song, ('onset_density', 'onset_strength'), left_start, prev_candidate.end)
    post_onset = _series_window_mean(current_song, ('onset_density', 'onset_strength'), current_candidate.start, right_end)
    pre_low = _series_window_mean(prev_song, ('low_band_ratio', 'bass_ratio', 'low_band_energy'), left_start, prev_candidate.end)
    post_low = _series_window_mean(current_song, ('low_band_ratio', 'bass_ratio', 'low_band_energy'), current_candidate.start, right_end)
    pre_flat = _series_window_mean(prev_song, ('spectral_flatness',), left_start, prev_candidate.end)
    post_flat = _series_window_mean(current_song, ('spectral_flatness',), current_candidate.start, right_end)

    energy_jump = _normalized_delta(pre_energy, post_energy, floor=0.01)
    spectral_jump = max(
        _normalized_delta(pre_centroid, post_centroid, floor=100.0),
        _normalized_delta(pre_rolloff, post_rolloff, floor=200.0),
    )
    onset_jump = _normalized_delta(pre_onset, post_onset, floor=0.05)
    low_end_crowding_risk = 0.0
    if pre_low > 0.0 or post_low > 0.0:
        overlap_low = min(max(pre_low, 0.0), max(post_low, 0.0))
        low_end_crowding_risk = overlap_low / max(max(pre_low, post_low, 0.0), 1e-6)
    texture_shift = max(
        _normalized_delta(pre_flat, post_flat, floor=0.01),
        _normalized_delta(pre_centroid + pre_onset, post_centroid + post_onset, floor=100.0),
    )
    foreground_collision_risk = _clamp01(
        0.45 * min(pre_energy, post_energy) / max(max(pre_energy, post_energy), 0.01)
        + 0.35 * min(pre_onset, post_onset) / max(max(pre_onset, post_onset), 0.05)
        + 0.20 * min(pre_centroid, post_centroid) / max(max(pre_centroid, post_centroid), 100.0)
    )
    vocal_competition_risk = _clamp01(
        0.50 * min(max(pre_centroid, 0.0), max(post_centroid, 0.0)) / max(max(pre_centroid, post_centroid, 0.0), 100.0)
        + 0.30 * min(max(pre_onset, 0.0), max(post_onset, 0.0)) / max(max(pre_onset, post_onset, 0.0), 0.05)
        + 0.20 * (1.0 - min(1.0, abs(pre_centroid - post_centroid) / max(max(pre_centroid, post_centroid), 100.0)))
    )
    seam_risk = _clamp01(
        0.22 * min(energy_jump, 1.5)
        + 0.18 * min(spectral_jump, 1.5)
        + 0.15 * min(onset_jump, 1.5)
        + 0.14 * min(low_end_crowding_risk, 1.5)
        + 0.11 * min(texture_shift, 1.5)
        + 0.10 * min(foreground_collision_risk, 1.5)
        + 0.10 * min(vocal_competition_risk, 1.5)
    )
    return seam_risk, {
        'energy_jump': energy_jump,
        'spectral_jump': spectral_jump,
        'onset_jump': onset_jump,
        'low_end_crowding_risk': low_end_crowding_risk,
        'texture_shift': texture_shift,
        'foreground_collision_risk': foreground_collision_risk,
        'vocal_competition_risk': vocal_competition_risk,
        'seam_risk': seam_risk,
    }


def _song_pair_compatibility(song_a: SongDNA, song_b: SongDNA) -> float:
    report = build_compatibility_report(song_a, song_b)
    return report.factors.overall


def _cross_parent_window_compatibility(song_a: SongDNA, candidate_a: _SectionCandidate, song_b: SongDNA, candidate_b: _SectionCandidate) -> float:
    pair_compat = _song_pair_compatibility(song_a, song_b)
    energy_delta = abs(candidate_a.energy - candidate_b.energy)
    energy_fit = max(0.0, 1.0 - (energy_delta / 0.35))
    slope_delta = abs(_window_energy_slope(song_a, candidate_a.start, candidate_a.end) - _window_energy_slope(song_b, candidate_b.start, candidate_b.end))
    slope_fit = max(0.0, 1.0 - (slope_delta / 0.45))
    return max(0.0, min(1.0, (0.60 * pair_compat) + (0.25 * energy_fit) + (0.15 * slope_fit)))


def _transition_viability(previous: _WindowSelection | None, current_parent: str, current_candidate: _SectionCandidate, transition_in: str | None) -> float:
    if previous is None:
        return 0.85

    previous_energy = previous.candidate.energy
    current_energy = current_candidate.energy
    delta = current_energy - previous_energy
    abs_delta = abs(delta)

    if transition_in == 'drop':
        return max(0.0, min(1.0, 0.45 + (0.55 * max(0.0, delta))))
    if transition_in == 'blend':
        same_parent_bonus = 0.10 if current_parent == previous.parent_id else 0.0
        if delta >= 0.0:
            # A build-friendly blend can tolerate a measured lift in energy;
            # punish steep upward jumps less aggressively than drops so the
            # planner does not flatten the arc just to keep adjacent windows similar.
            score = 0.88 - (abs_delta / 0.55)
        else:
            score = 0.95 - (abs_delta / 0.30)
        return max(0.0, min(1.0, score + same_parent_bonus))
    if transition_in == 'swap':
        parent_bonus = 0.10 if current_parent != previous.parent_id else 0.0
        return max(0.0, min(1.0, 0.90 - (abs_delta / 0.40) + parent_bonus))
    if transition_in == 'lift':
        return max(0.0, min(1.0, 0.55 + (0.45 * max(0.0, delta))))
    return max(0.0, min(1.0, 0.95 - (abs_delta / 0.35)))


def _transition_impact_fit(
    previous: _WindowSelection | None,
    candidate: _SectionCandidate,
    spec: _SectionSpec,
    seam_risk: float,
    seam_metrics: dict[str, float],
    stretch_ratio: float,
) -> float:
    if previous is None:
        return 0.5

    delta = candidate.energy - previous.candidate.energy
    transition_in = spec.transition_in
    if transition_in not in {'lift', 'drop'}:
        return 0.5

    if delta <= 0.0:
        return 0.0

    if transition_in == 'drop':
        target_delta = 0.24 if spec.label == 'payoff' else 0.18
        delta_width = 0.22
    else:
        target_delta = 0.14 if spec.label == 'build' else 0.10
        delta_width = 0.16

    delta_fit = max(0.0, 1.0 - (abs(delta - target_delta) / delta_width))
    safety_fit = max(
        0.0,
        min(
            1.0,
            1.0
            - (0.55 * seam_risk)
            - (0.20 * min(seam_metrics.get('onset_jump', 0.0), 1.5) / 1.5)
            - (0.15 * min(seam_metrics.get('spectral_jump', 0.0), 1.5) / 1.5)
            - (0.10 * _clamp01((stretch_ratio - 1.0) / 0.12)),
        ),
    )
    return _clamp01((0.62 * delta_fit) + (0.38 * safety_fit))


def _energy_arc_viability(previous: _WindowSelection | None, candidate: _SectionCandidate, spec: _SectionSpec) -> float:
    target_energy = max(0.0, min(1.0, spec.target_energy))
    current_energy = max(0.0, min(1.0, candidate.energy))

    if previous is None:
        return max(0.0, 1.0 - abs(current_energy - target_energy) / 0.35)

    previous_energy = max(0.0, min(1.0, previous.candidate.energy))
    desired_delta = target_energy - previous_energy
    actual_delta = current_energy - previous_energy

    delta_fit = max(0.0, 1.0 - (abs(actual_delta - desired_delta) / 0.35))
    target_fit = max(0.0, 1.0 - (abs(current_energy - target_energy) / 0.30))

    floor_fit = 1.0
    if spec.label in {'build', 'payoff'}:
        if current_energy < previous_energy:
            floor_fit = max(0.0, 1.0 - ((previous_energy - current_energy) / 0.20))
        elif spec.label == 'payoff' and current_energy < max(previous_energy, 0.72):
            floor_fit = max(0.0, 1.0 - ((max(previous_energy, 0.72) - current_energy) / 0.25))

    reset_fit = 1.0
    if spec.label in {'bridge', 'outro', 'verse'} and previous_energy >= 0.68:
        target_release = max(0.0, previous_energy - target_energy)
        if target_release > 0.08:
            actual_release = max(0.0, previous_energy - current_energy)
            release_fit = min(1.0, actual_release / target_release)
            headroom_fit = max(0.0, 1.0 - (max(0.0, current_energy - (target_energy + 0.08)) / 0.22))
            reset_fit = max(0.0, min(release_fit, headroom_fit))

    return max(0.0, min(1.0, (0.38 * delta_fit) + (0.30 * target_fit) + (0.16 * floor_fit) + (0.16 * reset_fit)))


def _target_duration_seconds(song: SongDNA, bar_count: int, beats_per_bar: int = 4, *, reference_tempo_bpm: float | None = None) -> float:
    tempo_bpm = float(reference_tempo_bpm if reference_tempo_bpm is not None else song.tempo_bpm)
    return float(bar_count) * float(beats_per_bar) * 60.0 / max(tempo_bpm, 1e-6)


def _stretch_profile(song: SongDNA, candidate: _SectionCandidate, bar_count: int, *, reference_tempo_bpm: float | None = None) -> tuple[float, float, float]:
    target_duration = _target_duration_seconds(song, bar_count, reference_tempo_bpm=reference_tempo_bpm)
    source_duration = max(candidate.duration, 1e-6)
    stretch_ratio = source_duration / max(target_duration, 1e-6)
    absolute_mismatch = abs(1.0 - stretch_ratio)
    early_mismatch = max(0.0, absolute_mismatch - 0.08)
    conservative_overflow = max(0.0, stretch_ratio - _CONSERVATIVE_STRETCH_MAX)
    hard_overflow = max(0.0, stretch_ratio - _HARD_STRETCH_MAX)
    undershoot_overflow = max(0.0, _CONSERVATIVE_STRETCH_MIN - stretch_ratio)
    penalty = (
        0.45 * absolute_mismatch
        + 1.10 * (early_mismatch / 0.25)
        + 1.75 * (conservative_overflow / max(1e-6, _HARD_STRETCH_MAX - _CONSERVATIVE_STRETCH_MAX))
        + 2.60 * (hard_overflow / 0.20)
        + 0.85 * (undershoot_overflow / max(1e-6, _CONSERVATIVE_STRETCH_MIN - _HARD_STRETCH_MIN))
    )
    return stretch_ratio, target_duration, max(0.0, penalty)


def _collect_parent_candidates(song: SongDNA, target_position: str, bar_count: int, target_energy: float, role: str | None) -> tuple[list[_SectionCandidate], dict[str, _RoleFeatures], dict[str, float]]:
    phrase_candidates = _phrase_window_candidates(song, bar_count)
    section_candidates = _section_candidates(song)
    candidates = phrase_candidates or section_candidates

    if len(section_candidates) == 1 and not phrase_candidates:
        synthetic = section_candidates[0]
        if target_position == 'late':
            synthetic = _SectionCandidate(
                label='section_1',
                start=synthetic.start,
                end=synthetic.end,
                duration=synthetic.duration,
                midpoint=synthetic.midpoint,
                energy=synthetic.energy,
                origin='synthetic_missing_section',
            )
        return [synthetic], {synthetic.label: _candidate_role_features(song, [synthetic], synthetic)}, {synthetic.label: 1.0}

    role_name = role or target_position
    features_map = {candidate.label: _candidate_role_features(song, candidates, candidate) for candidate in candidates}
    raw_scores = {candidate.label: _role_prior_score(role_name, features_map[candidate.label]) for candidate in candidates}
    return candidates, features_map, _normalize_scores(raw_scores)


def _pick_candidate(song: SongDNA, target_position: str, bar_count: int, target_energy: float, role: str | None = None) -> _SectionCandidate:
    candidates, features_map, role_scores = _collect_parent_candidates(song, target_position, bar_count, target_energy, role)

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
        'early': {'position': 1.10, 'energy': 0.80, 'shape': 0.25, 'role': 1.05, 'boundary': 0.80},
        'mid': {'position': 0.80, 'energy': 1.00, 'shape': 0.35, 'role': 1.25, 'boundary': 0.85},
        'late': {'position': 0.45, 'energy': 1.20, 'shape': 0.45, 'role': 1.50, 'boundary': 0.90},
    }[target_position]

    def score(candidate: _SectionCandidate) -> tuple[float, int, float, float, float, float]:
        position_error = abs(candidate.midpoint - anchor) / total
        energy_error = abs(candidate.energy - target_energy) / energy_span
        normalized_energy = (candidate.energy - min_energy) / energy_span
        if target_position == 'early':
            shape_error = normalized_energy
        elif target_position == 'late':
            shape_error = 1.0 - normalized_energy
        else:
            shape_error = abs(normalized_energy - 0.6)
        role_error = 1.0 - role_scores.get(candidate.label, 0.0)
        boundary_error = 1.0 - _boundary_confidence(song, candidate)
        blended_error = (
            (weights['position'] * position_error)
            + (weights['energy'] * energy_error)
            + (weights['shape'] * shape_error)
            + (weights['role'] * role_error)
            + (weights['boundary'] * boundary_error)
        )
        return (
            blended_error,
            0 if candidate.origin == 'phrase_window' else 1,
            role_error,
            position_error,
            energy_error,
            candidate.start,
        )

    return min(candidates, key=score)


def _selection_reuse_penalty(
    prior_selections: list[_WindowSelection],
    parent_id: str,
    candidate: _SectionCandidate,
) -> tuple[float, dict[str, float]]:
    if not prior_selections:
        return 0.0, {
            'exact_window_reuse': 0.0,
            'window_overlap_reuse': 0.0,
            'parent_streak': 0.0,
            'source_rewind': 0.0,
            'source_containment': 0.0,
        }

    exact_window_reuse = 0.0
    overlap_reuse = 0.0
    parent_streak = 0.0
    source_rewind = 0.0
    source_containment = 0.0

    streak = 0
    for selection in reversed(prior_selections):
        if selection.parent_id != parent_id:
            break
        streak += 1
    if streak >= 2:
        parent_streak = min(1.0, 0.35 + (0.20 * (streak - 2)))

    last_same_parent = next((selection for selection in reversed(prior_selections) if selection.parent_id == parent_id), None)
    if last_same_parent is not None:
        prior = last_same_parent.candidate
        if candidate.start < prior.start:
            rewind_span = prior.start - candidate.start
            source_rewind = min(1.0, rewind_span / max(prior.duration, candidate.duration, 1e-6))
        if candidate.start <= prior.start and candidate.end >= prior.end:
            contained = prior.end - prior.start
            source_containment = min(1.0, contained / max(candidate.duration, 1e-6))

    for selection in prior_selections:
        if selection.parent_id != parent_id:
            continue
        prior = selection.candidate
        if prior.label == candidate.label and abs(prior.start - candidate.start) < 1e-6 and abs(prior.end - candidate.end) < 1e-6:
            exact_window_reuse = 1.0
            overlap_reuse = max(overlap_reuse, 1.0)
            continue

        overlap_start = max(prior.start, candidate.start)
        overlap_end = min(prior.end, candidate.end)
        if overlap_end <= overlap_start:
            continue
        overlap = overlap_end - overlap_start
        candidate_ratio = overlap / max(candidate.duration, 1e-6)
        prior_ratio = overlap / max(prior.duration, 1e-6)
        overlap_reuse = max(overlap_reuse, min(1.0, max(candidate_ratio, prior_ratio)))

    penalty = min(
        1.0,
        (0.70 * exact_window_reuse)
        + (0.45 * overlap_reuse)
        + (0.20 * parent_streak)
        + (0.55 * source_rewind)
        + (0.35 * source_containment),
    )
    return penalty, {
        'exact_window_reuse': exact_window_reuse,
        'window_overlap_reuse': overlap_reuse,
        'parent_streak': parent_streak,
        'source_rewind': source_rewind,
        'source_containment': source_containment,
    }


def _selection_fusion_balance_penalty(
    prior_selections: list[_WindowSelection],
    parent_id: str,
    source_parent_preference: str | None,
    current_section_label: str | None = None,
) -> tuple[float, dict[str, float]]:
    if not prior_selections:
        return 0.0, {
            'parent_share_imbalance': 0.0,
            'same_parent_run_bias': 0.0,
            'preferred_parent_miss': 0.0,
            'major_section_lockout': 0.0,
            'major_identity_gap': 0.0,
            'second_parent_presence_gap': 0.0,
            'weighted_identity_presence_gap': 0.0,
            'late_major_handoff_gap': 0.0,
            'single_cameo_rebound_gap': 0.0,
        }

    same_parent_count = sum(1 for selection in prior_selections if selection.parent_id == parent_id)
    other_parent_count = len(prior_selections) - same_parent_count
    share_imbalance = max(0.0, (same_parent_count - other_parent_count) / max(len(prior_selections), 1))

    same_parent_run = 0
    for selection in reversed(prior_selections):
        if selection.parent_id != parent_id:
            break
        same_parent_run += 1
    same_parent_run_bias = 0.0
    if same_parent_run >= 2:
        same_parent_run_bias = min(1.0, 0.45 + (0.20 * (same_parent_run - 2)))

    preferred_parent_miss = 0.0
    if source_parent_preference is not None and parent_id != source_parent_preference:
        preferred_parent_count = sum(1 for selection in prior_selections if selection.parent_id == source_parent_preference)
        if preferred_parent_count == 0:
            preferred_parent_miss = 1.0
        elif preferred_parent_count < same_parent_count:
            preferred_parent_miss = min(1.0, (same_parent_count - preferred_parent_count) / max(len(prior_selections), 1))

    major_labels = {'verse', 'build', 'payoff', 'bridge'}
    late_major_labels = {'payoff', 'bridge'}
    section_identity_weight = {
        'intro': 0.35,
        'outro': 0.35,
        'verse': 1.0,
        'build': 1.0,
        'payoff': 1.0,
        'bridge': 1.0,
    }
    current_is_major = current_section_label in major_labels
    current_is_late_major = current_section_label in late_major_labels
    prior_major = [selection for selection in prior_selections if selection.section_label in major_labels]
    same_parent_major_count = sum(1 for selection in prior_major if selection.parent_id == parent_id)
    other_parent_major_count = len(prior_major) - same_parent_major_count
    major_section_lockout = 0.0
    if current_is_major and prior_major and same_parent_major_count == len(prior_major) and other_parent_major_count == 0:
        major_section_lockout = min(1.0, 0.55 + (0.20 * max(0, len(prior_major) - 1)))
        if source_parent_preference is not None and parent_id != source_parent_preference:
            major_section_lockout = min(1.0, major_section_lockout + 0.25)

    major_identity_gap = 0.0
    total_major_after_selection = len(prior_major) + (1 if current_is_major else 0)
    resulting_same_parent_major_count = same_parent_major_count + (1 if current_is_major else 0)
    resulting_other_parent_major_count = other_parent_major_count
    resulting_major_minority_count = min(resulting_same_parent_major_count, resulting_other_parent_major_count)
    if current_is_major and total_major_after_selection >= 3:
        if resulting_major_minority_count <= 0:
            major_identity_gap = 1.0
        elif resulting_major_minority_count == 1 and total_major_after_selection >= 5:
            major_identity_gap = 0.55
        elif resulting_major_minority_count == 1 and total_major_after_selection >= 4:
            major_identity_gap = 0.35

    resulting_same_parent_count = same_parent_count + 1
    resulting_other_parent_count = other_parent_count
    resulting_minority_count = min(resulting_same_parent_count, resulting_other_parent_count)
    total_after_selection = len(prior_selections) + 1
    second_parent_presence_gap = 0.0
    if total_after_selection >= 4:
        if resulting_minority_count <= 0:
            second_parent_presence_gap = 1.0 if current_is_major else 0.75
        elif resulting_minority_count == 1:
            if total_after_selection >= 6:
                second_parent_presence_gap = 0.95 if current_is_major else 0.70
            elif total_after_selection >= 5:
                second_parent_presence_gap = 0.80 if current_is_major else 0.55
            else:
                second_parent_presence_gap = 0.55 if current_is_major else 0.35

    weighted_identity_counts = {'A': 0.0, 'B': 0.0}
    for selection in prior_selections:
        weighted_identity_counts[selection.parent_id] += section_identity_weight.get(selection.section_label or '', 0.60)
    weighted_identity_counts[parent_id] += section_identity_weight.get(current_section_label or '', 0.60)
    weighted_identity_minority = min(weighted_identity_counts.values())
    weighted_identity_majority = max(weighted_identity_counts.values())
    weighted_identity_presence_gap = 0.0
    if total_after_selection >= 4 and weighted_identity_majority > 0.0:
        if weighted_identity_minority <= 0.0:
            weighted_identity_presence_gap = 1.0 if current_is_major else 0.80
        elif weighted_identity_minority < 1.0:
            scarcity = (1.0 - weighted_identity_minority)
            imbalance = (weighted_identity_majority - weighted_identity_minority) / weighted_identity_majority
            base_gap = (0.62 + (0.32 * scarcity) + (0.22 * imbalance)) if current_is_major else (0.35 + (0.20 * scarcity) + (0.15 * imbalance))
            if current_is_late_major:
                base_gap += 0.12
            weighted_identity_presence_gap = min(1.0, base_gap)

    prior_late_major = [selection for selection in prior_selections if selection.section_label in late_major_labels]
    same_parent_late_major_count = sum(1 for selection in prior_late_major if selection.parent_id == parent_id)
    other_parent_late_major_count = len(prior_late_major) - same_parent_late_major_count
    late_major_handoff_gap = 0.0
    if current_is_late_major and total_after_selection >= 4 and same_parent_count >= other_parent_count:
        if len(prior_late_major) == 0:
            late_major_handoff_gap = 0.85
            if source_parent_preference is not None and parent_id != source_parent_preference:
                late_major_handoff_gap = min(1.0, late_major_handoff_gap + 0.10)
        elif same_parent_late_major_count == len(prior_late_major) and other_parent_late_major_count == 0:
            late_major_handoff_gap = min(1.0, 0.60 + (0.18 * len(prior_late_major)))
            if source_parent_preference is not None and parent_id != source_parent_preference:
                late_major_handoff_gap = min(1.0, late_major_handoff_gap + 0.10)

    single_cameo_rebound_gap = 0.0
    if current_is_major and len(prior_selections) >= 4:
        prior_parent_counts = Counter(selection.parent_id for selection in prior_selections)
        majority_parent, majority_count = max(prior_parent_counts.items(), key=lambda item: item[1])
        minority_parent, minority_count = min(prior_parent_counts.items(), key=lambda item: item[1])
        previous_selection = prior_selections[-1]
        if (
            len(prior_parent_counts) == 2
            and majority_count >= 3
            and minority_count == 1
            and previous_selection.parent_id == minority_parent
            and parent_id == majority_parent
        ):
            single_cameo_rebound_gap = 0.70
            if previous_selection.section_label in major_labels:
                single_cameo_rebound_gap += 0.12
            if current_section_label in late_major_labels:
                single_cameo_rebound_gap += 0.08
            if source_parent_preference is not None and parent_id != source_parent_preference:
                single_cameo_rebound_gap += 0.10
            single_cameo_rebound_gap = min(1.0, single_cameo_rebound_gap)

    penalty = min(
        1.0,
        (0.45 * share_imbalance)
        + (0.30 * same_parent_run_bias)
        + (0.25 * preferred_parent_miss)
        + (0.70 * major_section_lockout)
        + (0.55 * major_identity_gap)
        + (0.60 * second_parent_presence_gap)
        + (0.95 * weighted_identity_presence_gap)
        + (0.75 * late_major_handoff_gap)
        + (0.80 * single_cameo_rebound_gap),
    )
    return penalty, {
        'parent_share_imbalance': share_imbalance,
        'same_parent_run_bias': same_parent_run_bias,
        'preferred_parent_miss': preferred_parent_miss,
        'major_section_lockout': major_section_lockout,
        'major_identity_gap': major_identity_gap,
        'second_parent_presence_gap': second_parent_presence_gap,
        'weighted_identity_presence_gap': weighted_identity_presence_gap,
        'late_major_handoff_gap': late_major_handoff_gap,
        'single_cameo_rebound_gap': single_cameo_rebound_gap,
    }


def _selection_section_shape_penalty(
    spec: _SectionSpec,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
    previous: _WindowSelection | None,
) -> tuple[float, dict[str, float]]:
    if spec.label not in {'intro', 'payoff'}:
        return 0.0, {
            'intro_hotspot': 0.0,
            'payoff_underhit': 0.0,
            'late_drop_gap': 0.0,
        }

    intro_hotspot = 0.0
    if spec.label == 'intro':
        intro_hotspot = _clamp01(
            (0.55 * features.payoff_strength)
            + (0.30 * features.hook_strength)
            + (0.25 * features.end_focus)
            + (0.15 * max(0.0, candidate.energy - 0.34) / 0.26)
        )

    payoff_underhit = 0.0
    late_drop_gap = 0.0
    if spec.label == 'payoff':
        payoff_hit = (
            (0.34 * features.end_focus)
            + (0.24 * features.plateau_stability)
            + (0.20 * features.payoff_strength)
            + (0.14 * features.hook_strength)
            + (0.08 * features.lift_strength)
        )
        payoff_underhit = max(0.0, 0.68 - payoff_hit)
        if previous is not None:
            desired_floor = 0.18 if previous.section_label == 'bridge' else 0.10
            late_drop_gap = max(0.0, desired_floor - (candidate.energy - previous.candidate.energy))

    penalty = min(1.0, (0.95 * intro_hotspot) + (1.05 * payoff_underhit) + (0.85 * late_drop_gap))
    return penalty, {
        'intro_hotspot': intro_hotspot,
        'payoff_underhit': payoff_underhit,
        'late_drop_gap': late_drop_gap,
    }


def _selection_build_to_payoff_contrast_penalty(
    spec: _SectionSpec,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
    previous: _WindowSelection | None,
) -> tuple[float, dict[str, float]]:
    if spec.label != 'payoff' or previous is None or previous.section_label != 'build':
        return 0.0, {
            'energy_lift_gap': 0.0,
            'tail_dominance_gap': 0.0,
            'payoff_conviction_gap': 0.0,
        }

    previous_energy = max(0.0, min(1.0, previous.candidate.energy))
    current_energy = max(0.0, min(1.0, candidate.energy))
    energy_lift = current_energy - previous_energy
    energy_lift_gap = max(0.0, 0.12 - energy_lift)

    previous_tail_energy = max(0.0, min(1.0, _window_energy(previous.song, max(previous.candidate.start, previous.candidate.end - min(previous.candidate.duration * 0.5, 8.0)), previous.candidate.end)))
    tail_dominance = features.tail_energy - previous_tail_energy
    tail_dominance_gap = max(0.0, 0.10 - tail_dominance)

    payoff_conviction = _clamp01(
        (0.34 * features.end_focus)
        + (0.28 * features.plateau_stability)
        + (0.18 * features.payoff_strength)
        + (0.12 * features.hook_strength)
        + (0.08 * features.energy_confidence)
    )
    payoff_conviction_gap = max(0.0, 0.66 - payoff_conviction)

    penalty = min(
        1.0,
        (1.05 * energy_lift_gap)
        + (0.95 * tail_dominance_gap)
        + (0.55 * payoff_conviction_gap),
    )
    return penalty, {
        'energy_lift_gap': energy_lift_gap,
        'tail_dominance_gap': tail_dominance_gap,
        'payoff_conviction_gap': payoff_conviction_gap,
    }


def _selection_groove_continuity_penalty(
    prior_selections: list[_WindowSelection],
    parent_id: str,
    feedback: _PlannerListenFeedback,
    alternate_feedback: _PlannerListenFeedback,
    transition_in: str | None,
) -> tuple[float, dict[str, float]]:
    if not prior_selections:
        return 0.0, {
            'same_parent_streak': 0.0,
            'alternate_groove_edge': 0.0,
            'alternate_transition_edge': 0.0,
            'coherence_gap': 0.0,
            'transition_weight': 0.0,
        }

    same_parent_streak = 0
    for selection in reversed(prior_selections):
        if selection.parent_id != parent_id:
            break
        same_parent_streak += 1
    if same_parent_streak <= 0:
        return 0.0, {
            'same_parent_streak': 0.0,
            'alternate_groove_edge': 0.0,
            'alternate_transition_edge': 0.0,
            'coherence_gap': 0.0,
            'transition_weight': 0.0,
        }

    alternate_groove_edge = max(0.0, alternate_feedback.groove_confidence - feedback.groove_confidence)
    alternate_transition_edge = max(0.0, alternate_feedback.transition_readiness - feedback.transition_readiness)
    coherence_gap = max(0.0, alternate_feedback.coherence_confidence - feedback.coherence_confidence)
    transition_weight = 1.0 if transition_in in {'blend', 'lift', 'swap'} else 0.55
    streak_weight = min(1.0, 0.30 + (0.25 * same_parent_streak))

    penalty = min(
        1.0,
        streak_weight
        * transition_weight
        * (
            (0.60 * alternate_groove_edge)
            + (0.30 * alternate_transition_edge)
            + (0.10 * coherence_gap)
        ),
    )
    return penalty, {
        'same_parent_streak': float(same_parent_streak),
        'alternate_groove_edge': alternate_groove_edge,
        'alternate_transition_edge': alternate_transition_edge,
        'coherence_gap': coherence_gap,
        'transition_weight': transition_weight,
    }


def _selection_phrase_groove_penalty(spec: _SectionSpec, features: _RoleFeatures) -> tuple[float, dict[str, float]]:
    if spec.label not in {'verse', 'build', 'payoff', 'bridge'}:
        return 0.0, {
            'groove_drive_gap': 0.0,
            'groove_stability_gap': 0.0,
        }

    drive_floor = {
        'verse': 0.42,
        'build': 0.52,
        'payoff': 0.50,
        'bridge': 0.36,
    }[spec.label]
    stability_floor = {
        'verse': 0.52,
        'build': 0.44,
        'payoff': 0.42,
        'bridge': 0.46,
    }[spec.label]
    groove_drive_gap = max(0.0, drive_floor - features.groove_drive)
    groove_stability_gap = max(0.0, stability_floor - features.groove_stability)
    penalty = min(1.0, (0.75 * groove_drive_gap) + (0.55 * groove_stability_gap))
    return penalty, {
        'groove_drive_gap': groove_drive_gap,
        'groove_stability_gap': groove_stability_gap,
    }


def _backbone_selection_guard_reason(
    spec: _SectionSpec,
    parent_id: str,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
    donor_parent: str | None,
) -> str | None:
    if backbone_parent is None or donor_parent is None:
        return None

    structural_labels = {'verse', 'bridge', 'outro'}
    if spec.label in structural_labels and parent_id != backbone_parent:
        return 'structural_backbone_only'
    if spec.label == 'build' and parent_id != donor_parent:
        return 'build_donor_only'

    if parent_id != donor_parent:
        return None

    prior_parents = [selection.parent_id for selection in prior_selections]
    if donor_parent not in prior_parents:
        return None

    last_donor_idx = max(idx for idx, prior_parent in enumerate(prior_parents) if prior_parent == donor_parent)
    if backbone_parent in prior_parents[last_donor_idx + 1:]:
        return 'donor_reentry_after_backbone'
    return None



def _selection_backbone_continuity_penalty(
    spec: _SectionSpec,
    parent_id: str,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
    donor_parent: str | None,
) -> tuple[float, dict[str, float]]:
    if backbone_parent is None or donor_parent is None:
        return 0.0, {
            'off_program_structural_handoff': 0.0,
            'donor_overreach': 0.0,
            'donor_reentry_after_backbone': 0.0,
        }

    structural_labels = {'verse', 'bridge', 'outro'}
    donor_feature_labels = {'build', 'payoff'}
    off_program_structural_handoff = 0.0
    donor_overreach = 0.0
    donor_reentry_after_backbone = 0.0

    if spec.label in structural_labels and parent_id != backbone_parent:
        off_program_structural_handoff = 1.0
    elif spec.label == 'build' and parent_id != donor_parent:
        off_program_structural_handoff = 0.85

    if parent_id == donor_parent and spec.label in donor_feature_labels:
        donor_count = sum(1 for selection in prior_selections if selection.parent_id == donor_parent)
        donor_major_count = sum(1 for selection in prior_selections if selection.parent_id == donor_parent and selection.section_label in donor_feature_labels)
        if donor_count >= 2:
            donor_overreach += min(1.0, 0.30 + (0.18 * (donor_count - 2)))
        if donor_major_count >= 2 and spec.label == 'payoff':
            donor_overreach += min(1.0, 0.35 + (0.20 * (donor_major_count - 2)))

        prior_parents = [selection.parent_id for selection in prior_selections]
        if donor_parent in prior_parents:
            last_donor_idx = max(idx for idx, prior_parent in enumerate(prior_parents) if prior_parent == donor_parent)
            backbone_reclaimed_after_donor = backbone_parent in prior_parents[last_donor_idx + 1:]
            if backbone_reclaimed_after_donor:
                donor_reentry_after_backbone = 1.0 if spec.label == 'payoff' else 0.85

    penalty = min(
        1.0,
        (0.90 * off_program_structural_handoff)
        + (0.70 * donor_overreach)
        + (0.95 * donor_reentry_after_backbone),
    )
    return penalty, {
        'off_program_structural_handoff': off_program_structural_handoff,
        'donor_overreach': min(1.0, donor_overreach),
        'donor_reentry_after_backbone': donor_reentry_after_backbone,
    }


def _enumerate_section_choices(
    spec: _SectionSpec,
    song_a: SongDNA,
    song_b: SongDNA,
    previous: _WindowSelection | None,
    prior_selections: list[_WindowSelection] | None = None,
    backbone_parent: str | None = None,
    donor_parent: str | None = None,
) -> list[_WindowSelection]:
    target_position = SECTION_TARGET_POSITION.get(spec.label, spec.label)
    prior_selections = list(prior_selections or [])
    by_parent = {
        'A': (song_a, song_b),
        'B': (song_b, song_a),
    }
    song_map = {'A': song_a, 'B': song_b}

    selections: list[_WindowSelection] = []
    for parent_id, (song, other_song) in by_parent.items():
        feedback = _planner_listen_feedback(song)
        alternate_feedback = _planner_listen_feedback(other_song)
        candidates, features_map, role_scores = _collect_parent_candidates(song, target_position, spec.bar_count, spec.target_energy, spec.label)
        energies = [candidate.energy for candidate in candidates]
        min_energy = min(energies) if energies else 0.0
        max_energy = max(energies) if energies else 0.0
        energy_span = max(max_energy - min_energy, 1e-6)
        total = max(float(song.duration_seconds), 1e-6)
        anchor_ratio = {'early': 0.16, 'mid': 0.50, 'late': 0.80}[target_position]
        pair_compat = _song_pair_compatibility(song_a, song_b)

        other_candidates, _, _ = _collect_parent_candidates(other_song, target_position, spec.bar_count, spec.target_energy, spec.label)
        best_cross_map = {
            candidate.label: max(
                (_cross_parent_window_compatibility(song, candidate, other_song, other_candidate) for other_candidate in other_candidates),
                default=pair_compat,
            )
            for candidate in candidates
        }

        guard_reason = _backbone_selection_guard_reason(
            spec,
            parent_id,
            prior_selections,
            backbone_parent,
            donor_parent,
        )
        if guard_reason is not None and any(
            _backbone_selection_guard_reason(spec, alternate_parent_id, prior_selections, backbone_parent, donor_parent) is None
            for alternate_parent_id in by_parent
            if alternate_parent_id != parent_id
        ):
            continue

        for candidate in candidates:
            position_error = abs((candidate.midpoint / total) - anchor_ratio)
            energy_error = abs(candidate.energy - spec.target_energy) / energy_span
            role_error = 1.0 - role_scores.get(candidate.label, 0.0)
            boundary_error = 1.0 - _boundary_confidence(song, candidate)
            compatibility_error = 1.0 - best_cross_map[candidate.label]
            transition_error = 1.0 - _transition_viability(previous, parent_id, candidate, spec.transition_in)
            arc_error = 1.0 - _energy_arc_viability(previous, candidate, spec)
            seam_risk, seam_metrics = _planner_seam_risk(previous, song, candidate)
            reuse_penalty, reuse_metrics = _selection_reuse_penalty(prior_selections, parent_id, candidate)
            fusion_balance_penalty, fusion_balance_metrics = _selection_fusion_balance_penalty(
                prior_selections,
                parent_id,
                spec.source_parent_preference,
                current_section_label=spec.label,
            )
            groove_continuity_penalty, groove_continuity_metrics = _selection_groove_continuity_penalty(
                prior_selections,
                parent_id,
                feedback,
                alternate_feedback,
                spec.transition_in,
            )
            phrase_groove_penalty, phrase_groove_metrics = _selection_phrase_groove_penalty(
                spec,
                features_map[candidate.label],
            )
            backbone_continuity_penalty, backbone_continuity_metrics = _selection_backbone_continuity_penalty(
                spec,
                parent_id,
                prior_selections,
                backbone_parent,
                donor_parent,
            )
            stretch_ratio, target_duration_seconds, stretch_penalty = _stretch_profile(
                song,
                candidate,
                spec.bar_count,
                reference_tempo_bpm=song_map[backbone_parent].tempo_bpm if backbone_parent in song_map else None,
            )
            stretch_gate = 0.0
            if stretch_ratio > _CONSERVATIVE_STRETCH_MAX:
                stretch_gate = _clamp01((stretch_ratio - _CONSERVATIVE_STRETCH_MAX) / max(1e-6, _HARD_STRETCH_MAX - _CONSERVATIVE_STRETCH_MAX))
            transition_impact_error = 1.0 - _transition_impact_fit(previous, candidate, spec, seam_risk, seam_metrics, stretch_ratio)
            section_shape_penalty, section_shape_metrics = _selection_section_shape_penalty(
                spec,
                candidate,
                features_map[candidate.label],
                previous,
            )
            build_to_payoff_contrast_penalty, build_to_payoff_contrast_metrics = _selection_build_to_payoff_contrast_penalty(
                spec,
                candidate,
                features_map[candidate.label],
                previous,
            )
            preference_error = 0.0 if spec.source_parent_preference in {None, parent_id} else 1.0
            final_payoff_delivery_penalty = 0.0
            if spec.label == 'payoff' and previous is not None and previous.section_label in {'build', 'bridge'}:
                features = features_map[candidate.label]
                payoff_delivery_floor = 0.62 if previous.section_label == 'bridge' else 0.58
                final_payoff_delivery_penalty = max(
                    0.0,
                    payoff_delivery_floor
                    - (
                        (0.34 * features.end_focus)
                        + (0.26 * features.plateau_stability)
                        + (0.20 * features.payoff_strength)
                        + (0.12 * features.hook_strength)
                        + (0.08 * features.lift_strength)
                    ),
                )

                late_arrival_floor = 0.66 if previous.section_label == 'bridge' else 0.58
                final_payoff_delivery_penalty += max(0.0, late_arrival_floor - features.position)

                candidate_start_position = candidate.start / max(float(song.duration_seconds), 1e-6)
                late_window_start_floor = 0.72 if previous.section_label == 'bridge' else 0.60
                late_window_freshness_gap = max(0.0, late_window_start_floor - candidate_start_position)
                final_payoff_delivery_penalty += 0.30 * late_window_freshness_gap

                sustained_payoff_conviction = _clamp01(
                    (0.30 * features.tail_energy)
                    + (0.24 * features.plateau_stability)
                    + (0.18 * features.payoff_strength)
                    + (0.16 * features.end_focus)
                    + (0.12 * features.energy_confidence)
                )
                sustained_conviction_floor = 0.70 if previous.section_label == 'bridge' else 0.62
                final_payoff_delivery_penalty += max(0.0, sustained_conviction_floor - sustained_payoff_conviction)

                if parent_id == previous.parent_id:
                    carryover_overlap = max(0.0, previous.candidate.end - candidate.start) / max(candidate.duration, 1e-6)
                    if carryover_overlap > 0.0:
                        final_payoff_delivery_penalty += min(0.35, carryover_overlap)

                    payoff_lift = candidate.energy - previous.candidate.energy
                    desired_lift = 0.16 if previous.section_label == 'bridge' else 0.12
                    if payoff_lift < desired_lift:
                        final_payoff_delivery_penalty += min(0.28, (desired_lift - payoff_lift) / 0.18)
            listen_feedback_penalty = 0.0
            if spec.label in {'build', 'payoff'}:
                listen_feedback_penalty += max(0.0, 0.65 - feedback.energy_arc_strength)
            if spec.label == 'payoff':
                listen_feedback_penalty += 1.2 * max(0.0, 0.60 - feedback.payoff_readiness)
            if previous is not None and parent_id != previous.parent_id:
                listen_feedback_penalty += 0.75 * max(0.0, 0.58 - feedback.transition_readiness)
                listen_feedback_penalty += 0.45 * max(0.0, 0.52 - feedback.groove_confidence)
            listen_feedback_penalty += 0.25 * max(0.0, 0.45 - feedback.coherence_confidence)

            if target_position == 'early':
                shape_error = max(0.0, (candidate.energy - min_energy) / energy_span)
            elif target_position == 'late':
                shape_error = max(0.0, 1.0 - ((candidate.energy - min_energy) / energy_span))
            else:
                normalized_energy = (candidate.energy - min_energy) / energy_span
                shape_error = abs(normalized_energy - 0.6)

            seam_gate_error = max(0.0, seam_risk - 0.60) / 0.40
            blended_error = (
                (0.75 * position_error)
                + (1.15 * energy_error)
                + (0.45 * shape_error)
                + (1.45 * role_error)
                + (0.85 * boundary_error)
                + (1.10 * compatibility_error)
                + (1.20 * transition_error)
                + (0.85 * transition_impact_error)
                + (1.05 * arc_error)
                + (1.25 * stretch_penalty)
                + (1.75 * stretch_gate)
                + (1.75 * seam_risk)
                + (1.35 * seam_gate_error)
                + (0.95 * listen_feedback_penalty)
                + (0.85 * final_payoff_delivery_penalty)
                + (0.95 * build_to_payoff_contrast_penalty)
                + (1.05 * section_shape_penalty)
                + (1.10 * reuse_penalty)
                + (0.85 * fusion_balance_penalty)
                + (0.90 * groove_continuity_penalty)
                + (0.85 * phrase_groove_penalty)
                + (1.10 * backbone_continuity_penalty)
                + (0.35 * preference_error)
            )
            selections.append(
                _WindowSelection(
                    parent_id=parent_id,
                    song=song,
                    candidate=candidate,
                    blended_error=blended_error,
                    section_label=spec.label,
                score_breakdown={
                        'position': position_error,
                        'energy_target': energy_error,
                        'role_prior': role_error,
                        'boundary_confidence': boundary_error,
                        'compatibility': compatibility_error,
                        'transition_viability': transition_error,
                        'transition_impact': transition_impact_error,
                        'energy_arc': arc_error,
                        'target_duration_seconds': target_duration_seconds,
                        'stretch_ratio': stretch_ratio,
                        'stretch_penalty': stretch_penalty,
                        'stretch_gate': stretch_gate,
                        'seam_risk': seam_risk,
                        'seam_gate': seam_gate_error,
                        'listen_feedback': listen_feedback_penalty,
                        'final_payoff_delivery': final_payoff_delivery_penalty,
                        'build_to_payoff_contrast': build_to_payoff_contrast_penalty,
                        'contrast_energy_lift_gap': build_to_payoff_contrast_metrics['energy_lift_gap'],
                        'contrast_tail_dominance_gap': build_to_payoff_contrast_metrics['tail_dominance_gap'],
                        'contrast_payoff_conviction_gap': build_to_payoff_contrast_metrics['payoff_conviction_gap'],
                        'section_shape': section_shape_penalty,
                        'shape_intro_hotspot': section_shape_metrics['intro_hotspot'],
                        'shape_payoff_underhit': section_shape_metrics['payoff_underhit'],
                        'shape_late_drop_gap': section_shape_metrics['late_drop_gap'],
                        'listen_groove_confidence': feedback.groove_confidence,
                        'listen_energy_arc_strength': feedback.energy_arc_strength,
                        'listen_transition_readiness': feedback.transition_readiness,
                        'listen_coherence_confidence': feedback.coherence_confidence,
                        'listen_payoff_readiness': feedback.payoff_readiness,
                        'selection_reuse': reuse_penalty,
                        'reuse_exact_window': reuse_metrics['exact_window_reuse'],
                        'reuse_window_overlap': reuse_metrics['window_overlap_reuse'],
                        'reuse_parent_streak': reuse_metrics['parent_streak'],
                        'reuse_source_rewind': reuse_metrics['source_rewind'],
                        'reuse_source_containment': reuse_metrics['source_containment'],
                        'fusion_balance': fusion_balance_penalty,
                        'fusion_parent_share_imbalance': fusion_balance_metrics['parent_share_imbalance'],
                        'fusion_same_parent_run_bias': fusion_balance_metrics['same_parent_run_bias'],
                        'fusion_preferred_parent_miss': fusion_balance_metrics['preferred_parent_miss'],
                        'fusion_major_section_lockout': fusion_balance_metrics['major_section_lockout'],
                        'fusion_major_identity_gap': fusion_balance_metrics['major_identity_gap'],
                        'fusion_second_parent_presence_gap': fusion_balance_metrics['second_parent_presence_gap'],
                        'fusion_weighted_identity_presence_gap': fusion_balance_metrics['weighted_identity_presence_gap'],
                        'fusion_late_major_handoff_gap': fusion_balance_metrics['late_major_handoff_gap'],
                        'fusion_single_cameo_rebound_gap': fusion_balance_metrics['single_cameo_rebound_gap'],
                        'groove_continuity': groove_continuity_penalty,
                        'groove_same_parent_streak': groove_continuity_metrics['same_parent_streak'],
                        'groove_alternate_groove_edge': groove_continuity_metrics['alternate_groove_edge'],
                        'groove_alternate_transition_edge': groove_continuity_metrics['alternate_transition_edge'],
                        'groove_coherence_gap': groove_continuity_metrics['coherence_gap'],
                        'groove_transition_weight': groove_continuity_metrics['transition_weight'],
                        'phrase_groove': phrase_groove_penalty,
                        'phrase_groove_drive_gap': phrase_groove_metrics['groove_drive_gap'],
                        'phrase_groove_stability_gap': phrase_groove_metrics['groove_stability_gap'],
                        'backbone_continuity': backbone_continuity_penalty,
                        'backbone_off_program_structural_handoff': backbone_continuity_metrics['off_program_structural_handoff'],
                        'backbone_donor_overreach': backbone_continuity_metrics['donor_overreach'],
                        'backbone_donor_reentry_after_backbone': backbone_continuity_metrics['donor_reentry_after_backbone'],
                        'parent_preference': preference_error,
                        'seam_energy_jump': seam_metrics['energy_jump'],
                        'seam_spectral_jump': seam_metrics['spectral_jump'],
                        'seam_onset_jump': seam_metrics['onset_jump'],
                        'seam_low_end_crowding': seam_metrics['low_end_crowding_risk'],
                        'seam_foreground_collision': seam_metrics['foreground_collision_risk'],
                        'seam_vocal_competition': seam_metrics['vocal_competition_risk'],
                    },
                )
            )

    return sorted(
        selections,
        key=lambda item: (
            item.blended_error,
            0 if item.candidate.origin == 'phrase_window' else 1,
            item.score_breakdown['transition_viability'],
            item.score_breakdown['role_prior'],
            item.candidate.start,
        ),
    )


def _choose_with_major_section_balance_guard(
    spec: _SectionSpec,
    ranked: list[_WindowSelection],
    prior_selections: list[_WindowSelection],
) -> tuple[_WindowSelection, str | None]:
    if not ranked:
        raise ValueError('ranked selections must not be empty')

    major_labels = {'verse', 'build', 'payoff', 'bridge'}
    if spec.label not in major_labels:
        return ranked[0], None

    prior_major = [selection for selection in prior_selections if selection.section_label in major_labels]
    if len(prior_major) < 2:
        return ranked[0], None

    chosen = ranked[0]
    chosen_parent = chosen.parent_id
    chosen_major_count = sum(1 for selection in prior_major if selection.parent_id == chosen_parent)
    other_parent = 'B' if chosen_parent == 'A' else 'A'
    other_major_count = len(prior_major) - chosen_major_count
    if chosen_major_count != len(prior_major) or other_major_count != 0:
        return chosen, None

    alternate = next((item for item in ranked if item.parent_id == other_parent), None)
    if alternate is None:
        return chosen, None

    error_delta = alternate.blended_error - chosen.blended_error
    max_delta = 0.42
    max_stretch_gate = 0.0
    max_stretch_ratio = 1.12
    if spec.label in {'payoff', 'bridge'}:
        max_delta = 1.05
        max_stretch_gate = 0.58
        max_stretch_ratio = 1.18

    alternate_stretch_gate = alternate.score_breakdown.get('stretch_gate', 0.0)
    alternate_stretch_ratio = alternate.score_breakdown.get('stretch_ratio', 1.0)
    alternate_seam_risk = alternate.score_breakdown.get('seam_risk', 1.0)
    chosen_seam_risk = chosen.score_breakdown.get('seam_risk', 1.0)
    alternate_transition_error = alternate.score_breakdown.get('transition_viability', 1.0)
    chosen_transition_error = chosen.score_breakdown.get('transition_viability', 1.0)
    alternate_role_error = alternate.score_breakdown.get('role_prior', 1.0)
    alternate_groove_confidence = alternate.score_breakdown.get('listen_groove_confidence', 1.0)
    chosen_groove_confidence = chosen.score_breakdown.get('listen_groove_confidence', 1.0)
    alternate_groove_penalty = alternate.score_breakdown.get('groove_continuity', 0.0)

    if error_delta > max_delta:
        return chosen, None
    if alternate_stretch_gate > max_stretch_gate:
        return chosen, None
    if alternate_stretch_ratio > max_stretch_ratio:
        return chosen, None
    if alternate_groove_confidence < max(0.58, chosen_groove_confidence - 0.08):
        return chosen, None
    if alternate_groove_penalty > 0.22:
        return chosen, None
    if alternate_seam_risk > min(0.78, chosen_seam_risk + 0.14):
        return chosen, None
    if alternate_transition_error > min(0.82, chosen_transition_error + 0.18):
        return chosen, None
    if alternate_role_error > 0.72:
        return chosen, None

    note = (
        f"major-section balance guard: {spec.label} switched to {alternate.parent_id}:{alternate.candidate.label} "
        f"to avoid a full one-parent major-section monopoly; alt delta {error_delta:.2f}; "
        f"guarded safe by stretch {alternate_stretch_ratio:.2f} and groove {alternate_groove_confidence:.2f}"
    )
    return alternate, note


def _infer_transition_mode(
    spec: _SectionSpec,
    chosen: _WindowSelection,
    previous: _WindowSelection | None,
    previous_label: str | None,
) -> str | None:
    if spec.transition_in is None:
        return None
    if previous is None or previous.parent_id == chosen.parent_id:
        return "same_parent_flow"
    if previous_label == "payoff" and spec.label in {"bridge", "outro"}:
        return "arrival_handoff"
    if spec.transition_in in {"swap", "drop"} or spec.label in {"build", "payoff"}:
        return "single_owner_handoff"
    return "crossfade_support"


def _apply_section_level_authenticity_guard(
    section_specs: list[_SectionSpec],
    chosen_selections: list[_WindowSelection],
    ranked_choices: list[list[_WindowSelection]],
) -> tuple[list[_WindowSelection], list[str]]:
    if not chosen_selections:
        return chosen_selections, []

    major_labels = {'verse', 'build', 'payoff', 'bridge'}
    chosen_parents = {selection.parent_id for selection in chosen_selections}
    dominant_parent = chosen_selections[0].parent_id
    alternate_parent = 'B' if dominant_parent == 'A' else 'A'

    full_section_monopoly = len(chosen_parents) == 1
    chosen_major = [selection for selection in chosen_selections if selection.section_label in major_labels]
    major_section_monopoly = bool(chosen_major) and len({selection.parent_id for selection in chosen_major}) == 1

    if not full_section_monopoly and not major_section_monopoly:
        return chosen_selections, []
    if major_section_monopoly:
        dominant_parent = chosen_major[0].parent_id
        alternate_parent = 'B' if dominant_parent == 'A' else 'A'

    def _is_safe_authenticity_alternate(current: _WindowSelection, alternate: _WindowSelection, spec: _SectionSpec) -> bool:
        error_delta = alternate.blended_error - current.blended_error
        max_delta = 0.48
        max_stretch_ratio = 1.12
        max_stretch_gate = 0.0
        max_seam_risk = min(0.76, current.score_breakdown.get('seam_risk', 1.0) + 0.12)
        max_transition_error = min(0.82, current.score_breakdown.get('transition_viability', 1.0) + 0.18)
        max_role_error = 0.74
        max_groove_penalty = 0.24
        min_groove_confidence = max(0.56, current.score_breakdown.get('listen_groove_confidence', 1.0) - 0.10)

        if spec.label in {'payoff', 'bridge'}:
            max_delta = 1.10
            max_stretch_ratio = 1.18
            max_stretch_gate = 0.58
            max_role_error = 0.78
            max_groove_penalty = 0.28

        return not (
            error_delta > max_delta
            or alternate.score_breakdown.get('stretch_gate', 0.0) > max_stretch_gate
            or alternate.score_breakdown.get('stretch_ratio', 1.0) > max_stretch_ratio
            or alternate.score_breakdown.get('seam_risk', 1.0) > max_seam_risk
            or alternate.score_breakdown.get('transition_viability', 1.0) > max_transition_error
            or alternate.score_breakdown.get('role_prior', 1.0) > max_role_error
            or alternate.score_breakdown.get('groove_continuity', 0.0) > max_groove_penalty
            or alternate.score_breakdown.get('listen_groove_confidence', 1.0) < min_groove_confidence
        )

    priority_labels = {'payoff': 0, 'bridge': 1, 'build': 2, 'verse': 3, 'outro': 4, 'intro': 5}
    candidate_swaps: list[tuple[int, _WindowSelection, _WindowSelection, _SectionSpec]] = []
    for idx, (spec, current, ranked) in enumerate(zip(section_specs, chosen_selections, ranked_choices)):
        if major_section_monopoly and spec.label not in major_labels:
            continue
        alternate = next((item for item in ranked if item.parent_id == alternate_parent), None)
        if alternate is None:
            continue
        if not _is_safe_authenticity_alternate(current, alternate, spec):
            continue
        candidate_swaps.append((idx, current, alternate, spec))

    if not candidate_swaps:
        return chosen_selections, []

    idx, current, alternate, spec = min(
        candidate_swaps,
        key=lambda item: (
            priority_labels.get(item[3].label, 99),
            item[2].blended_error - item[1].blended_error,
            item[0],
        ),
    )
    updated = list(chosen_selections)
    updated[idx] = alternate
    guard_reason = 'full one-parent major-section collapse' if major_section_monopoly and not full_section_monopoly else 'full one-parent section collapse'
    note = (
        f"section-level authenticity guard: {spec.label} switched to {alternate.parent_id}:{alternate.candidate.label} "
        f"to avoid a {guard_reason}; alt delta {alternate.blended_error - current.blended_error:.2f}; "
        f"guarded safe by stretch {alternate.score_breakdown.get('stretch_ratio', 1.0):.2f} "
        f"and groove {alternate.score_breakdown.get('listen_groove_confidence', 1.0):.2f}"
    )
    return updated, [note]


def build_stub_arrangement_plan(song_a: SongDNA, song_b: SongDNA) -> ChildArrangementPlan:
    report = build_compatibility_report(song_a, song_b)

    backbone_plan = _choose_backbone_parent(song_a, song_b)
    section_specs = _program_with_backbone(
        _build_section_program(song_a, song_b),
        backbone_plan.backbone_parent,
        backbone_plan.donor_parent,
    )
    parent_feedback = {
        'A': _planner_listen_feedback(song_a),
        'B': _planner_listen_feedback(song_b),
    }

    selection_notes: list[str] = []
    previous: _WindowSelection | None = None
    selection_history: list[_WindowSelection] = []
    chosen_selections: list[_WindowSelection] = []
    ranked_choices: list[list[_WindowSelection]] = []
    song_map = {'A': song_a, 'B': song_b}
    for spec in section_specs:
        ranked = _enumerate_section_choices(
            spec,
            song_a,
            song_b,
            previous,
            prior_selections=selection_history,
            backbone_parent=backbone_plan.backbone_parent,
            donor_parent=backbone_plan.donor_parent,
        )
        chosen, balance_guard_note = _choose_with_major_section_balance_guard(spec, ranked, selection_history)
        ranked_choices.append(ranked)
        chosen_selections.append(chosen)
        if balance_guard_note is not None:
            selection_notes.append(f"{spec.label}: {balance_guard_note}")
        selection_history.append(chosen)
        previous = chosen

    chosen_selections, authenticity_guard_notes = _apply_section_level_authenticity_guard(
        section_specs,
        chosen_selections,
        ranked_choices,
    )
    selection_notes.extend(authenticity_guard_notes)

    sections: list[PlannedSection] = []
    selection_diagnostics: list[dict[str, Any]] = []
    previous = None
    for spec, chosen in zip(section_specs, chosen_selections):
        candidate = chosen.candidate
        transition_mode = _infer_transition_mode(spec, chosen, previous, previous.section_label if previous else None)
        continuity_treatment = None
        if transition_mode == 'same_parent_flow' and chosen.parent_id == backbone_plan.backbone_parent and spec.label in {'verse', 'bridge', 'outro'}:
            continuity_treatment = 'backbone_flow'
        sections.append(
            PlannedSection(
                label=spec.label,
                start_bar=spec.start_bar,
                bar_count=spec.bar_count,
                source_parent=chosen.parent_id,
                source_section_label=candidate.label,
                target_energy=spec.target_energy,
                transition_in=spec.transition_in,
                transition_out=spec.transition_out,
                transition_mode=transition_mode,
            )
        )
        breakdown = ', '.join(f"{name}={value:.2f}" for name, value in chosen.score_breakdown.items())
        selection_notes.append(
            f"{spec.label}: ranked full-window candidates across both parents; chose {chosen.parent_id}:{candidate.label} ({candidate.origin}, {candidate.start:.1f}-{candidate.end:.1f}s, energy {candidate.energy:.3f}, error {chosen.blended_error:.2f}; {breakdown})"
        )
        feedback = parent_feedback[chosen.parent_id]
        selection_diagnostics.append(
            {
                'label': spec.label,
                'target_energy': round(spec.target_energy, 3),
                'selected_parent': chosen.parent_id,
                'selected_role': 'backbone' if chosen.parent_id == backbone_plan.backbone_parent else 'donor',
                'selected_window_label': candidate.label,
                'selected_window_origin': candidate.origin,
                'selected_window_seconds': {'start': round(candidate.start, 3), 'end': round(candidate.end, 3)},
                'backbone_tempo_bpm': round(song_map[backbone_plan.backbone_parent].tempo_bpm, 3),
                'candidate_tempo_bpm': round(chosen.song.tempo_bpm, 3),
                'target_section_seconds': round(chosen.score_breakdown['target_duration_seconds'], 3),
                'transition_in': spec.transition_in,
                'transition_out': spec.transition_out,
                'transition_mode': transition_mode,
                'continuity_treatment': continuity_treatment,
                'planner_error': round(chosen.blended_error, 3),
                'evaluator_alignment': {
                    'listen_feedback_penalty': round(chosen.score_breakdown['listen_feedback'], 3),
                    'seam_risk': round(chosen.score_breakdown['seam_risk'], 3),
                    'transition_viability': round(1.0 - chosen.score_breakdown['transition_viability'], 3),
                    'energy_arc_fit': round(1.0 - chosen.score_breakdown['energy_arc'], 3),
                    'groove_confidence': round(feedback.groove_confidence, 3),
                    'transition_readiness': round(feedback.transition_readiness, 3),
                    'coherence_confidence': round(feedback.coherence_confidence, 3),
                    'payoff_readiness': round(feedback.payoff_readiness, 3),
                },
            }
        )
        previous = chosen

    program_signature = ' -> '.join(f"{spec.label}({spec.bar_count})" for spec in section_specs)
    backbone_usage_counts = Counter(selection.parent_id for selection in chosen_selections)
    diagnostics = {
        'planner_evaluator_bridge': 'listen-aligned planner diagnostics',
        'backbone_plan': {
            'backbone_parent': backbone_plan.backbone_parent,
            'donor_parent': backbone_plan.donor_parent,
            'backbone_score': round(backbone_plan.backbone_score, 3),
            'donor_score': round(backbone_plan.donor_score, 3),
            'selection_reasons': backbone_plan.backbone_reasons,
            'section_usage': {
                backbone_plan.backbone_parent: backbone_usage_counts.get(backbone_plan.backbone_parent, 0),
                backbone_plan.donor_parent: backbone_usage_counts.get(backbone_plan.donor_parent, 0),
            },
        },
        'parent_listen_feedback': {
            parent_id: {
                'source_path': song_map[parent_id].source_path,
                'groove_confidence': round(feedback.groove_confidence, 3),
                'energy_arc_strength': round(feedback.energy_arc_strength, 3),
                'transition_readiness': round(feedback.transition_readiness, 3),
                'coherence_confidence': round(feedback.coherence_confidence, 3),
                'payoff_readiness': round(feedback.payoff_readiness, 3),
            }
            for parent_id, feedback in parent_feedback.items()
        },
        'selected_sections': selection_diagnostics,
    }
    notes = [
        'Planner now uses an explicit backbone-first child-song architecture: one parent carries macro continuity while the other is inserted selectively as donor material.',
        'Extended bridge/re-payoff forms are now gated by shared reset/relaunch evidence across the pair so one parent’s local late-song shape does not force a fake second climax onto the child program.',
        f"Backbone parent: {backbone_plan.backbone_parent}; donor parent: {backbone_plan.donor_parent}; reasons: {', '.join(backbone_plan.backbone_reasons)}.",
        'Planner now ranks explicit phrase windows section-by-section across both parents instead of relying on coarse early/mid/late anchor picking.',
        f'Section program is now capacity-aware instead of fixed: {program_signature}.',
        'Ranking factors are boundary confidence, role prior, target-energy fit, cross-parent compatibility, transition viability, explicit build-to-payoff contrast scoring, evaluator-style seam-risk priors, planner-facing listen feedback (groove/arc/transition/payoff readiness), backbone-continuity pressure, history-aware source-window reuse penalties, and derived hook/payoff confidence signals from canonical bar features.',
        'Sequential selection now discourages replaying the exact same source window or heavily overlapping window later in the child timeline unless the musical fit is clearly stronger.',
        'Seam-risk priors reuse listen-style handoff heuristics (energy/spectral/onset jumps plus low-end, foreground, and vocal-collision risk) to reject obviously awkward boundaries before render.',
        'Arrangement artifacts now expose listen-aligned planning_diagnostics so evaluator-facing groove/arc/transition signals are inspectable without parsing note strings.',
        'Stretch and bar-grid fit are now evaluated against the backbone parent tempo so donor phrases are judged on the child-song grid instead of each source parent silently keeping its own clock.',
        'For same-parent backbone continuity, diagnostics now flag backbone_flow candidates even though render still uses same_parent_flow until a dedicated low-overlap backbone treatment is wired end-to-end.',
        'Resolver understands phrase_<start>_<end> labels and snaps them directly to analyzed phrase boundaries.',
        *selection_notes,
    ]
    if len(_section_candidates(song_a)) <= 1 or len(_section_candidates(song_b)) <= 1:
        notes.append('At least one song still has coarse section analysis; phrase-window labels plus factorized ranking reduce fallback dependence when phrase boundaries exist.')

    return ChildArrangementPlan(
        parents=[_song_parent_ref(song_a), _song_parent_ref(song_b)],
        compatibility=report.factors,
        sections=sections,
        planning_notes=notes,
        planning_diagnostics=diagnostics,
    )
