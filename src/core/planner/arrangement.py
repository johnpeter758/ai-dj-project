from __future__ import annotations

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
        extended_support = max(_song_extended_program_support(song_a), _song_extended_program_support(song_b))
        if extended_support >= 0.30:
            return extended
        return standard
    if capacity >= 5:
        return standard
    return compact


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
    if len(parts) == 3 and parts[0] == 'phrase':
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


def _window_profile(song: SongDNA, start: float, end: float, bins: int = 4) -> list[float]:
    times = _safe_float_list(song.energy.get('bar_times', [])) or _safe_float_list(song.energy.get('beat_times', []))
    values = _safe_float_list(song.energy.get('bar_rms', [])) or _safe_float_list(song.energy.get('beat_rms', []))
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

    slope_span = max(max(slopes) - min(slopes), 1e-6) if slopes else 1e-6
    candidate_indices = _candidate_phrase_indices(candidate.label)
    start_idx, end_idx = candidate_indices if candidate_indices is not None else (0, 1)
    candidate_profile = profiles[candidate.label]

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
    elif canonical_role == 'pre':
        score += 0.35 * features.payoff_strength * signal_confidence
        score += 0.90 * features.lift_strength
        score += 0.45 * features.headroom
        score += 0.55 * features.ramp_consistency
        score += 0.10 * features.end_focus
        score -= 0.55 * (features.plateau_stability * features.end_focus)
        score -= 0.25 * max(0.0, features.end_focus - features.ramp_consistency)
    elif canonical_role == 'verse':
        score += 0.45 * features.hook_strength * signal_confidence
    elif canonical_role == 'bridge':
        score += 0.20 * features.hook_strength * signal_confidence
    elif canonical_role == 'intro':
        score -= 0.15 * features.payoff_strength * signal_confidence

    return score


def _boundary_confidence(song: SongDNA, candidate: _SectionCandidate) -> float:
    phrase_boundaries = sorted(set(_safe_float_list(song.structure.get('phrase_boundaries_seconds', []))))
    if candidate.origin != 'phrase_window' or not phrase_boundaries:
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


def _target_duration_seconds(song: SongDNA, bar_count: int, beats_per_bar: int = 4) -> float:
    return float(bar_count) * float(beats_per_bar) * 60.0 / max(float(song.tempo_bpm), 1e-6)


def _stretch_profile(song: SongDNA, candidate: _SectionCandidate, bar_count: int) -> tuple[float, float, float]:
    target_duration = _target_duration_seconds(song, bar_count)
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
            'second_parent_presence_gap': 0.0,
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
    current_is_major = current_section_label in major_labels
    prior_major = [selection for selection in prior_selections if selection.section_label in major_labels]
    same_parent_major_count = sum(1 for selection in prior_major if selection.parent_id == parent_id)
    other_parent_major_count = len(prior_major) - same_parent_major_count
    major_section_lockout = 0.0
    if current_is_major and prior_major and same_parent_major_count == len(prior_major) and other_parent_major_count == 0:
        major_section_lockout = min(1.0, 0.55 + (0.20 * max(0, len(prior_major) - 1)))
        if source_parent_preference is not None and parent_id != source_parent_preference:
            major_section_lockout = min(1.0, major_section_lockout + 0.25)

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

    penalty = min(
        1.0,
        (0.45 * share_imbalance)
        + (0.30 * same_parent_run_bias)
        + (0.25 * preferred_parent_miss)
        + (0.70 * major_section_lockout)
        + (0.60 * second_parent_presence_gap),
    )
    return penalty, {
        'parent_share_imbalance': share_imbalance,
        'same_parent_run_bias': same_parent_run_bias,
        'preferred_parent_miss': preferred_parent_miss,
        'major_section_lockout': major_section_lockout,
        'second_parent_presence_gap': second_parent_presence_gap,
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


def _enumerate_section_choices(
    spec: _SectionSpec,
    song_a: SongDNA,
    song_b: SongDNA,
    previous: _WindowSelection | None,
    prior_selections: list[_WindowSelection] | None = None,
) -> list[_WindowSelection]:
    target_position = SECTION_TARGET_POSITION.get(spec.label, spec.label)
    prior_selections = list(prior_selections or [])
    by_parent = {
        'A': (song_a, song_b),
        'B': (song_b, song_a),
    }

    selections: list[_WindowSelection] = []
    for parent_id, (song, other_song) in by_parent.items():
        feedback = _planner_listen_feedback(song)
        alternate_feedback = _planner_listen_feedback(other_song)
        candidates, _, role_scores = _collect_parent_candidates(song, target_position, spec.bar_count, spec.target_energy, spec.label)
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
            stretch_ratio, _, stretch_penalty = _stretch_profile(song, candidate, spec.bar_count)
            stretch_gate = 0.0
            if stretch_ratio > _CONSERVATIVE_STRETCH_MAX:
                stretch_gate = _clamp01((stretch_ratio - _CONSERVATIVE_STRETCH_MAX) / max(1e-6, _HARD_STRETCH_MAX - _CONSERVATIVE_STRETCH_MAX))
            preference_error = 0.0 if spec.source_parent_preference in {None, parent_id} else 1.0
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
                + (1.05 * arc_error)
                + (1.25 * stretch_penalty)
                + (1.75 * stretch_gate)
                + (1.75 * seam_risk)
                + (1.35 * seam_gate_error)
                + (0.95 * listen_feedback_penalty)
                + (1.10 * reuse_penalty)
                + (0.85 * fusion_balance_penalty)
                + (0.90 * groove_continuity_penalty)
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
                        'energy_arc': arc_error,
                        'stretch_ratio': stretch_ratio,
                        'stretch_penalty': stretch_penalty,
                        'stretch_gate': stretch_gate,
                        'seam_risk': seam_risk,
                        'seam_gate': seam_gate_error,
                        'listen_feedback': listen_feedback_penalty,
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
                        'fusion_second_parent_presence_gap': fusion_balance_metrics['second_parent_presence_gap'],
                        'groove_continuity': groove_continuity_penalty,
                        'groove_same_parent_streak': groove_continuity_metrics['same_parent_streak'],
                        'groove_alternate_groove_edge': groove_continuity_metrics['alternate_groove_edge'],
                        'groove_alternate_transition_edge': groove_continuity_metrics['alternate_transition_edge'],
                        'groove_coherence_gap': groove_continuity_metrics['coherence_gap'],
                        'groove_transition_weight': groove_continuity_metrics['transition_weight'],
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


def build_stub_arrangement_plan(song_a: SongDNA, song_b: SongDNA) -> ChildArrangementPlan:
    report = build_compatibility_report(song_a, song_b)

    section_specs = _build_section_program(song_a, song_b)
    parent_feedback = {
        'A': _planner_listen_feedback(song_a),
        'B': _planner_listen_feedback(song_b),
    }

    sections: list[PlannedSection] = []
    selection_notes: list[str] = []
    selection_diagnostics: list[dict[str, Any]] = []
    previous: _WindowSelection | None = None
    selection_history: list[_WindowSelection] = []
    song_map = {'A': song_a, 'B': song_b}
    for spec in section_specs:
        ranked = _enumerate_section_choices(spec, song_a, song_b, previous, prior_selections=selection_history)
        chosen = ranked[0]
        candidate = chosen.candidate
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
                'selected_window_label': candidate.label,
                'selected_window_origin': candidate.origin,
                'selected_window_seconds': {'start': round(candidate.start, 3), 'end': round(candidate.end, 3)},
                'transition_in': spec.transition_in,
                'transition_out': spec.transition_out,
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
        selection_history.append(chosen)
        previous = chosen

    program_signature = ' -> '.join(f"{spec.label}({spec.bar_count})" for spec in section_specs)
    diagnostics = {
        'planner_evaluator_bridge': 'listen-aligned planner diagnostics',
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
        'Planner now ranks explicit phrase windows section-by-section across both parents instead of relying on coarse early/mid/late anchor picking.',
        f'Section program is now capacity-aware instead of fixed: {program_signature}.',
        'Ranking factors are boundary confidence, role prior, target-energy fit, cross-parent compatibility, transition viability, evaluator-style seam-risk priors, planner-facing listen feedback (groove/arc/transition/payoff readiness), history-aware source-window reuse penalties, and derived hook/payoff confidence signals from canonical bar features.',
        'Sequential selection now discourages replaying the exact same source window or heavily overlapping window later in the child timeline unless the musical fit is clearly stronger.',
        'Seam-risk priors reuse listen-style handoff heuristics (energy/spectral/onset jumps plus low-end, foreground, and vocal-collision risk) to reject obviously awkward boundaries before render.',
        'Arrangement artifacts now expose listen-aligned planning_diagnostics so evaluator-facing groove/arc/transition signals are inspectable without parsing note strings.',
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
