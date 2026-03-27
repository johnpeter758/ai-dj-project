from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

from ..analysis.models import SongDNA
from ..intelligence import build_child_section_recipe
from .compatibility import baseline_pair_admissibility, build_compatibility_report
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
    source_section_prior: float


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
        'position_low': 1.85,
        'position_high': 0.02,
        'energy_low': 1.30,
        'energy_high': 0.03,
        'slope_up': 0.22,
        'slope_down': 0.55,
        'repetition': 0.34,
        'novelty': 0.12,
        'section_early': 0.90,
        'section_late': 0.02,
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


def _song_payoff_family_metrics(song: SongDNA) -> dict[str, float]:
    profile = _song_phrase_energy_profile(song)
    if len(profile) < 7:
        return {
            'reset_depth': 0.0,
            'relaunch_strength': 0.0,
            'final_advantage': 0.0,
            'first_payoff': 0.0,
            'final_payoff': 0.0,
            'bridge_floor': 0.0,
        }

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

    return {
        'reset_depth': _clamp01((first_payoff - bridge_floor) / energy_span),
        'relaunch_strength': _clamp01((final_payoff - bridge_floor) / energy_span),
        'final_advantage': _clamp01((final_payoff - first_payoff) / energy_span + 0.5),
        'first_payoff': first_payoff,
        'final_payoff': final_payoff,
        'bridge_floor': bridge_floor,
    }


def _extended_payoff_family_support(metrics: dict[str, float]) -> float:
    relaunch_advantage = _clamp01((metrics['final_advantage'] - 0.5) * 2.0)
    return _clamp01(
        0.38 * metrics['reset_depth']
        + 0.34 * metrics['relaunch_strength']
        + 0.28 * relaunch_advantage
    )


def _song_extended_program_support(song: SongDNA) -> float:
    metrics = _song_payoff_family_metrics(song)
    feedback = _planner_listen_feedback(song)
    return _clamp01(
        0.30 * metrics['reset_depth']
        + 0.28 * metrics['relaunch_strength']
        + 0.08 * metrics['final_advantage']
        + 0.14 * feedback.payoff_readiness
        + 0.10 * feedback.energy_arc_strength
        + 0.10 * _extended_payoff_family_support(metrics)
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
        family_a = _song_payoff_family_metrics(song_a)
        family_b = _song_payoff_family_metrics(song_b)
        extended_support = max(support_a, support_b)
        shared_support = 0.5 * (support_a + support_b)
        family_support_a = _extended_payoff_family_support(family_a)
        family_support_b = _extended_payoff_family_support(family_b)
        strongest_family_support = max(family_support_a, family_support_b)
        shared_family_support = 0.5 * (family_support_a + family_support_b)
        payoff_gain_a = family_a['final_payoff'] - family_a['first_payoff']
        payoff_gain_b = family_b['final_payoff'] - family_b['first_payoff']
        strongest_payoff_gain = max(payoff_gain_a, payoff_gain_b)
        weakest_payoff_gain = min(payoff_gain_a, payoff_gain_b)
        shared_payoff_gain = 0.5 * (payoff_gain_a + payoff_gain_b)
        shared_extended_support = (
            extended_support >= 0.42
            and shared_support >= 0.36
            and strongest_family_support >= 0.44
            and shared_family_support >= 0.36
            and strongest_payoff_gain >= 0.03
            and shared_payoff_gain >= 0.015
        )
        asymmetric_delayed_climax_support = (
            extended_support >= 0.58
            and min(support_a, support_b) >= 0.28
            and strongest_family_support >= 0.60
            and min(family_support_a, family_support_b) >= 0.24
            and strongest_payoff_gain >= 0.10
            and weakest_payoff_gain >= 0.0
        )
        if shared_extended_support or asymmetric_delayed_climax_support:
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


def _baseline_pair_admissibility(song_a: SongDNA, song_b: SongDNA) -> dict[str, Any]:
    return baseline_pair_admissibility(song_a, song_b)


def _build_baseline_section_program(song_a: SongDNA, song_b: SongDNA, backbone_parent: str, donor_parent: str) -> list[_SectionSpec]:
    capacity = min(_song_phrase_capacity(song_a), _song_phrase_capacity(song_b))
    if capacity >= 5:
        return [
            _SectionSpec(label='intro', start_bar=0, bar_count=8, target_energy=0.24, source_parent_preference=backbone_parent, transition_out='lift'),
            _SectionSpec(label='verse', start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference=backbone_parent, transition_in='blend', transition_out='swap'),
            _SectionSpec(label='build', start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference=donor_parent, transition_in='swap', transition_out='lift'),
            _SectionSpec(label='payoff', start_bar=24, bar_count=8, target_energy=0.78, source_parent_preference=donor_parent, transition_in='lift', transition_out='swap'),
            _SectionSpec(label='outro', start_bar=32, bar_count=8, target_energy=0.34, source_parent_preference=backbone_parent, transition_in='swap'),
        ]
    return [
        _SectionSpec(label='intro', start_bar=0, bar_count=8, target_energy=0.24, source_parent_preference=backbone_parent, transition_out='swap'),
        _SectionSpec(label='build', start_bar=8, bar_count=8, target_energy=0.56, source_parent_preference=donor_parent, transition_in='swap', transition_out='swap'),
        _SectionSpec(label='outro', start_bar=16, bar_count=8, target_energy=0.32, source_parent_preference=backbone_parent, transition_in='swap'),
    ]


def _baseline_mini_arc_metrics(build_choice: _WindowSelection, payoff_choice: _WindowSelection) -> dict[str, float]:
    build_candidate = build_choice.candidate
    payoff_candidate = payoff_choice.candidate
    build_indices = _candidate_phrase_indices(build_candidate.label)
    payoff_indices = _candidate_phrase_indices(payoff_candidate.label)

    overlap_seconds = max(0.0, build_candidate.end - payoff_candidate.start)
    overlap_ratio = overlap_seconds / max(min(build_candidate.duration, payoff_candidate.duration), 1e-6)
    backward_gap_seconds = max(0.0, build_candidate.start - payoff_candidate.start)
    handoff_gap_seconds = max(0.0, payoff_candidate.start - build_candidate.end)
    energy_lift = payoff_candidate.energy - build_candidate.energy
    seam_risk = float(payoff_choice.score_breakdown.get('seam_risk', 0.0))
    transition_penalty = float(payoff_choice.score_breakdown.get('transition_viability', 0.0))
    payoff_delivery_penalty = float(payoff_choice.score_breakdown.get('final_payoff_delivery', 0.0))
    payoff_hit_gap = float(payoff_choice.score_breakdown.get('payoff_hit', 0.0))
    sustained_gap = float(payoff_choice.score_breakdown.get('payoff_sustained_conviction', 0.0))

    phrase_jump = 0.0
    if build_indices is not None and payoff_indices is not None:
        phrase_jump = max(0.0, float(payoff_indices[0] - build_indices[1]))

    contiguity_penalty = 0.0
    if payoff_candidate.start < build_candidate.start:
        contiguity_penalty += min(1.0, backward_gap_seconds / max(build_candidate.duration, 1e-6))
    if handoff_gap_seconds > 0.0:
        contiguity_penalty += min(1.0, handoff_gap_seconds / max(build_candidate.duration * 0.75, 1e-6))
    if overlap_ratio > 0.55:
        contiguity_penalty += min(1.0, (overlap_ratio - 0.55) / 0.35)
    if phrase_jump > 1.25:
        contiguity_penalty += min(1.0, (phrase_jump - 1.25) / 1.75)

    arc_penalty = max(0.0, 0.04 - energy_lift) / 0.20
    arc_penalty += 0.85 * max(0.0, seam_risk - 0.42)
    arc_penalty += 0.65 * max(0.0, transition_penalty - 0.36)
    arc_penalty += 0.75 * max(0.0, payoff_delivery_penalty - 0.22)
    arc_penalty += 0.55 * max(0.0, payoff_hit_gap - 0.22)
    arc_penalty += 0.45 * max(0.0, sustained_gap - 0.26)

    legitimacy_penalty = contiguity_penalty + arc_penalty
    return {
        'build_start': round(build_candidate.start, 3),
        'build_end': round(build_candidate.end, 3),
        'payoff_start': round(payoff_candidate.start, 3),
        'payoff_end': round(payoff_candidate.end, 3),
        'overlap_ratio': round(overlap_ratio, 3),
        'handoff_gap_seconds': round(handoff_gap_seconds, 3),
        'backward_gap_seconds': round(backward_gap_seconds, 3),
        'phrase_jump': round(phrase_jump, 3),
        'energy_lift': round(energy_lift, 3),
        'seam_risk': round(seam_risk, 3),
        'transition_penalty': round(transition_penalty, 3),
        'payoff_delivery_penalty': round(payoff_delivery_penalty, 3),
        'contiguity_penalty': round(contiguity_penalty, 3),
        'arc_penalty': round(arc_penalty, 3),
        'legitimacy_penalty': round(legitimacy_penalty, 3),
        'legitimate': legitimacy_penalty <= 0.95,
    }


def _resolve_baseline_donor_mini_arc(
    section_specs: list[_SectionSpec],
    chosen_selections: list[_WindowSelection],
    ranked_choices: list[list[_WindowSelection]],
    backbone_parent: str,
    donor_parent: str,
) -> tuple[list[_WindowSelection], list[str], dict[str, Any]]:
    notes: list[str] = []
    diagnostics: dict[str, Any] = {'status': 'not_applicable'}
    build_idx = next((idx for idx, spec in enumerate(section_specs) if spec.label == 'build'), None)
    payoff_idx = next((idx for idx, spec in enumerate(section_specs) if spec.label == 'payoff'), None)
    if build_idx is None or payoff_idx is None:
        return chosen_selections, notes, diagnostics

    diagnostics = {
        'status': 'evaluated',
        'build_index': build_idx,
        'payoff_index': payoff_idx,
    }
    current_build = chosen_selections[build_idx]
    current_payoff = chosen_selections[payoff_idx]
    current_metrics = _baseline_mini_arc_metrics(current_build, current_payoff)
    diagnostics['initial_pair'] = {
        'build_parent': current_build.parent_id,
        'build_label': current_build.candidate.label,
        'payoff_parent': current_payoff.parent_id,
        'payoff_label': current_payoff.candidate.label,
        'metrics': current_metrics,
    }

    donor_build_options = [item for item in ranked_choices[build_idx] if item.parent_id == donor_parent]
    donor_payoff_options = [item for item in ranked_choices[payoff_idx] if item.parent_id == donor_parent]
    best_pair: tuple[float, _WindowSelection, _WindowSelection, dict[str, float]] | None = None
    for build_option in donor_build_options:
        for payoff_option in donor_payoff_options:
            metrics = _baseline_mini_arc_metrics(build_option, payoff_option)
            if not metrics['legitimate']:
                continue
            pair_score = (
                build_option.blended_error
                + payoff_option.blended_error
                + metrics['legitimacy_penalty']
                + (0.25 * max(0.0, -metrics['energy_lift']))
            )
            if best_pair is None or pair_score < best_pair[0]:
                best_pair = (pair_score, build_option, payoff_option, metrics)

    if current_build.parent_id == donor_parent and current_payoff.parent_id == donor_parent and current_metrics['legitimate']:
        diagnostics['status'] = 'accepted_initial_pair'
        return chosen_selections, notes, diagnostics

    updated = list(chosen_selections)
    if best_pair is not None:
        _, best_build, best_payoff, best_metrics = best_pair
        updated[build_idx] = best_build
        updated[payoff_idx] = best_payoff
        diagnostics['status'] = 'replaced_with_safer_donor_pair'
        diagnostics['resolved_pair'] = {
            'build_parent': best_build.parent_id,
            'build_label': best_build.candidate.label,
            'payoff_parent': best_payoff.parent_id,
            'payoff_label': best_payoff.candidate.label,
            'metrics': best_metrics,
        }
        notes.append(
            'baseline donor mini-arc guard: replaced the initial donor feature block with a safer contiguous donor build/payoff pair.'
        )
        return updated, notes, diagnostics

    updated[build_idx] = next(
        (item for item in ranked_choices[build_idx] if item.parent_id == backbone_parent),
        chosen_selections[build_idx],
    )
    updated[payoff_idx] = next(
        (item for item in ranked_choices[payoff_idx] if item.parent_id == backbone_parent),
        chosen_selections[payoff_idx],
    )
    diagnostics['status'] = 'fallback_backbone_only'
    diagnostics['resolved_pair'] = {
        'build_parent': updated[build_idx].parent_id,
        'build_label': updated[build_idx].candidate.label,
        'payoff_parent': updated[payoff_idx].parent_id,
        'payoff_label': updated[payoff_idx].candidate.label,
    }
    notes.append(
        'baseline donor mini-arc guard: donor build/payoff did not form a convincing contiguous mini-arc, so the planner fell back to backbone-owned feature sections.'
    )
    return updated, notes, diagnostics


def _safe_float_list(values) -> list[float]:
    out: list[float] = []
    for value in values or []:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        out.append(parsed)
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


def _opening_lane_candidates(song: SongDNA, bar_count: int, role: str | None) -> list[_SectionCandidate]:
    canonical_role = ROLE_ALIAS.get(role or '', role or '')
    if canonical_role not in {'intro', 'verse'}:
        return []

    target_duration = _target_section_duration_seconds(song, bar_count)
    min_duration = target_duration * (0.72 if canonical_role == 'intro' else 0.78)
    max_duration = target_duration * 1.12
    duration = float(song.duration_seconds)

    raw_boundaries = sorted({
        0.0,
        duration,
        *(_safe_float_list(song.structure.get('section_boundaries_seconds', [])) or []),
        *(_safe_float_list(song.structure.get('phrase_boundaries_seconds', [])) or []),
    })
    boundaries = [value for value in raw_boundaries if 0.0 <= value <= duration]
    if len(boundaries) < 2:
        return []

    opening_horizon = duration * (0.48 if canonical_role == 'intro' else 0.60)
    max_start_idx = 1 if canonical_role == 'intro' else 3
    max_end_idx = len(boundaries) - 1

    candidates: list[_SectionCandidate] = []
    for start_idx in range(min(max_start_idx + 1, len(boundaries) - 1)):
        start = float(boundaries[start_idx])
        if start >= opening_horizon:
            continue
        for end_idx in range(start_idx + 1, max_end_idx + 1):
            end = float(boundaries[end_idx])
            window_duration = end - start
            if window_duration <= 0.0:
                continue
            if window_duration < min_duration:
                continue
            if window_duration > max_duration:
                break
            if end > opening_horizon and canonical_role == 'intro':
                break
            label = f'opening_{canonical_role}_{start_idx}_{end_idx}'
            _append_candidate(candidates, song, label, start, end, 'opening_lane')
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


def _is_generic_source_section_label(label: str | None) -> bool:
    if not label:
        return True
    normalized = label.strip().lower()
    return normalized.startswith(('section_', 'part_', 'segment_'))



def _source_section_role_prior(song: SongDNA, candidate: _SectionCandidate, role: str | None) -> float:
    if not role:
        return 0.0

    canonical_role = ROLE_ALIAS.get(role, role)
    role_aliases = {
        'intro': {'intro', 'opening'},
        'verse': {'verse'},
        'pre': {'pre', 'build', 'prechorus', 'pre_chorus', 'rise'},
        'chorus_payoff': {'chorus', 'hook', 'drop', 'payoff', 'refrain'},
        'bridge': {'bridge', 'break', 'breakdown', 'middle8', 'middle_8'},
        'outro': {'outro', 'ending', 'end'},
    }
    hint_map = {
        'intro_like': {'intro': 1.0, 'verse': 0.18},
        'verse_like': {'verse': 1.0, 'pre': 0.22},
        'chorus_like': {'chorus_payoff': 1.0, 'pre': 0.18},
        'outro_like': {'outro': 1.0, 'bridge': 0.16},
        'section_like': {},
    }

    priors: list[float] = []
    for idx, section in enumerate(song.structure.get('sections', []) or []):
        if not isinstance(section, dict):
            continue
        try:
            start = float(section.get('start', 0.0))
            end = float(section.get('end', song.duration_seconds))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        overlap = max(0.0, min(candidate.end, end) - max(candidate.start, start))
        if overlap <= 0.0:
            continue

        overlap_ratio = overlap / max(min(candidate.duration, end - start), 1e-6)
        confidence = float(section.get('boundary_confidence', 1.0) or 1.0)
        section_prior = 0.0

        role_hint = str(section.get('role_hint') or '').strip().lower()
        if role_hint:
            section_prior = max(section_prior, hint_map.get(role_hint, {}).get(canonical_role, 0.0))

        raw_label = str(section.get('label') or f'section_{idx}').strip().lower()
        if not _is_generic_source_section_label(raw_label):
            compact_label = raw_label.replace('-', '_').replace(' ', '_')
            tokens = {token for token in compact_label.split('_') if token}
            aliases = role_aliases.get(canonical_role, set())
            if aliases & tokens or any(alias in compact_label for alias in aliases):
                section_prior = max(section_prior, 1.0)

        if section_prior > 0.0:
            priors.append(section_prior * overlap_ratio * max(0.4, min(confidence, 1.0)))

    return max(priors, default=0.0)



def _candidate_role_features(song: SongDNA, candidates: list[_SectionCandidate], candidate: _SectionCandidate, role: str | None = None) -> _RoleFeatures:
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
        source_section_prior=_source_section_role_prior(song, candidate, role),
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
        score += 0.32 * features.headroom
        score += 0.18 * max(0.0, 1.0 - features.end_focus)
        score += 0.12 * max(0.0, 1.0 - features.hook_strength)
        score -= 0.22 * features.payoff_strength * signal_confidence
        score -= 0.12 * features.end_focus

    score += 0.95 * features.source_section_prior
    return score


def _structure_boundary_evidence(song: SongDNA) -> list[dict[str, float]]:
    evidence: list[dict[str, float]] = []
    for item in song.structure.get('boundary_confidences_seconds', []) or []:
        if not isinstance(item, dict):
            continue
        try:
            evidence.append({
                'time': float(item.get('time', 0.0)),
                'confidence': _clamp01(float(item.get('confidence', 0.0))),
            })
        except (TypeError, ValueError):
            continue
    evidence.sort(key=lambda item: item['time'])
    return evidence


def _boundary_support(evidence: list[dict[str, float]], target_time: float, tolerance: float) -> float:
    best = 0.0
    for item in evidence:
        delta = abs(float(item['time']) - float(target_time))
        if delta > tolerance:
            continue
        proximity = 1.0 - (delta / max(tolerance, 1e-6))
        best = max(best, float(item['confidence']) * (0.60 + (0.40 * proximity)))
    return _clamp01(best)


def _section_candidate_boundary_confidence(song: SongDNA, candidate: _SectionCandidate, tolerance: float) -> float:
    sections = song.structure.get('sections', []) or []
    for idx, section in enumerate(sections):
        label = str(section.get('label') or f'section_{idx}')
        start = float(section.get('start', 0.0))
        end = float(section.get('end', song.duration_seconds))
        if label == candidate.label or (abs(start - candidate.start) <= tolerance and abs(end - candidate.end) <= tolerance):
            return _clamp01(float(section.get('boundary_confidence', 0.0)))
    return 0.0


def _boundary_confidence(song: SongDNA, candidate: _SectionCandidate) -> float:
    duration = float(song.duration_seconds)
    tolerance = max(duration * 0.02, 0.25)
    evidence = _structure_boundary_evidence(song)
    start_support = _boundary_support(evidence, candidate.start, tolerance)
    end_support = _boundary_support(evidence, candidate.end, tolerance)
    beat_grid_confidence = _clamp01(
        float(song.structure.get('beat_grid_confidence', song.metadata.get('tempo', {}).get('confidence', 0.0)))
    )
    phrase_boundary_method = str(song.structure.get('phrase_boundary_method', 'beat_phrase_grid'))

    phrase_boundaries = sorted(set(_safe_float_list(song.structure.get('phrase_boundaries_seconds', []))))
    if candidate.origin not in {'phrase_window', 'phrase_trim'} or not phrase_boundaries:
        section_confidence = _section_candidate_boundary_confidence(song, candidate, tolerance)
        return min(
            1.0,
            0.18
            + (0.42 * section_confidence)
            + (0.22 * ((start_support + end_support) * 0.5))
            + (0.18 * beat_grid_confidence),
        )

    boundaries = list(phrase_boundaries)
    if boundaries[0] > 0.0:
        boundaries = [0.0, *boundaries]
    if boundaries[-1] < duration:
        boundaries.append(duration)

    def nearest_delta(value: float) -> float:
        return min(abs(value - boundary) for boundary in boundaries)

    start_score = max(0.0, 1.0 - (nearest_delta(candidate.start) / tolerance))
    end_score = max(0.0, 1.0 - (nearest_delta(candidate.end) / tolerance))

    section_boundaries = _safe_float_list(song.structure.get('section_boundaries_seconds', []))
    section_hits = sum(1 for boundary in section_boundaries if candidate.start <= boundary <= candidate.end)
    boundary_density = min(1.0, section_hits / max((candidate.end - candidate.start) / 8.0, 1.0))

    confidence = min(
        1.0,
        0.14
        + (0.20 * beat_grid_confidence)
        + (0.24 * ((start_score + end_score) * 0.5))
        + (0.26 * ((start_support + end_support) * 0.5))
        + (0.16 * boundary_density),
    )
    if phrase_boundary_method != 'beat_phrase_grid':
        confidence *= 0.82
    return _clamp01(confidence)


def _normalize_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    finite_values: dict[str, float] = {}
    for key, value in values.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            finite_values[key] = parsed
    if not finite_values:
        return {key: 0.0 for key in values}
    low = min(finite_values.values())
    high = max(finite_values.values())
    span = max(high - low, 1e-6)
    normalized = {key: (value - low) / span for key, value in finite_values.items()}
    for key in values:
        normalized.setdefault(key, 0.0)
    return normalized


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
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


def _vocal_presence_proxy(energy: float, centroid: float, onset: float, flatness: float) -> float:
    centroid_presence = _clamp01((max(centroid, 0.0) - 1400.0) / 1800.0)
    tonal_presence = _clamp01((0.24 - max(flatness, 0.0)) / 0.16)
    energy_presence = _clamp01((max(energy, 0.0) - 0.10) / 0.14)
    onset_presence = _clamp01((max(onset, 0.0) - 0.16) / 0.22)
    return _clamp01(
        0.35 * centroid_presence
        + 0.30 * tonal_presence
        + 0.20 * energy_presence
        + 0.15 * onset_presence
    )


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
    pre_vocal_presence = _vocal_presence_proxy(pre_energy, pre_centroid, pre_onset, pre_flat)
    post_vocal_presence = _vocal_presence_proxy(post_energy, post_centroid, post_onset, post_flat)
    shared_vocal_presence = min(pre_vocal_presence, post_vocal_presence)
    centroid_similarity = 1.0 - min(1.0, abs(pre_centroid - post_centroid) / max(max(pre_centroid, post_centroid), 100.0))
    onset_similarity = min(max(pre_onset, 0.0), max(post_onset, 0.0)) / max(max(pre_onset, post_onset, 0.0), 0.05)
    energy_similarity = min(max(pre_energy, 0.0), max(post_energy, 0.0)) / max(max(pre_energy, post_energy, 0.0), 0.01)
    vocal_competition_risk = _clamp01(
        0.45 * shared_vocal_presence
        + 0.30 * centroid_similarity
        + 0.15 * onset_similarity
        + 0.10 * energy_similarity
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
    opening_candidates = _opening_lane_candidates(song, bar_count, role or target_position)
    section_candidates = _section_candidates(song)
    candidates = [*phrase_candidates]
    for candidate in opening_candidates:
        _append_candidate(candidates, song, candidate.label, candidate.start, candidate.end, candidate.origin)

    allow_generic_section_candidates = not (phrase_candidates or opening_candidates)
    for candidate in section_candidates:
        if not allow_generic_section_candidates and _is_generic_source_section_label(candidate.label):
            continue
        _append_candidate(candidates, song, candidate.label, candidate.start, candidate.end, candidate.origin)
    if not candidates:
        candidates = section_candidates

    if len(section_candidates) == 1 and not phrase_candidates and not opening_candidates:
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
        return [synthetic], {synthetic.label: _candidate_role_features(song, [synthetic], synthetic, role)}, {synthetic.label: 1.0}

    role_name = role or target_position
    features_map = {candidate.label: _candidate_role_features(song, candidates, candidate, role_name) for candidate in candidates}
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


def _selection_build_ramp_penalty(
    spec: _SectionSpec,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
    previous: _WindowSelection | None,
) -> tuple[float, dict[str, float]]:
    if spec.label != 'build':
        return 0.0, {
            'build_rise_gap': 0.0,
            'build_headroom_gap': 0.0,
            'build_plateau_risk': 0.0,
            'build_entry_lift_gap': 0.0,
        }

    build_rise = _clamp01(
        (0.42 * features.lift_strength)
        + (0.34 * features.ramp_consistency)
        + (0.24 * features.groove_drive)
    )
    build_rise_gap = max(0.0, 0.58 - build_rise)
    build_headroom_gap = max(0.0, 0.46 - features.headroom)
    build_plateau_risk = _clamp01(
        (0.44 * features.plateau_stability)
        + (0.30 * features.end_focus)
        + (0.16 * features.payoff_strength)
        + (0.10 * max(0.0, candidate.energy - 0.72) / 0.20)
        - 0.56
    )

    build_entry_lift_gap = 0.0
    if previous is not None and previous.section_label in {'intro', 'verse'}:
        build_entry_lift_gap = max(0.0, 0.05 - (candidate.energy - previous.candidate.energy))

    penalty = min(
        1.0,
        (1.00 * build_rise_gap)
        + (0.75 * build_headroom_gap)
        + (0.95 * build_plateau_risk)
        + (0.80 * build_entry_lift_gap),
    )
    return penalty, {
        'build_rise_gap': build_rise_gap,
        'build_headroom_gap': build_headroom_gap,
        'build_plateau_risk': build_plateau_risk,
        'build_entry_lift_gap': build_entry_lift_gap,
    }



def _hard_build_candidate_pool(
    spec: _SectionSpec,
    parent_id: str,
    candidates: list[_SectionCandidate],
    build_metrics_map: dict[str, dict[str, float]],
    previous: _WindowSelection | None,
) -> set[str]:
    if spec.label != 'build' or (spec.source_parent_preference is not None and parent_id != spec.source_parent_preference):
        return set()

    viable_labels = [
        candidate.label
        for candidate in candidates
        if build_metrics_map[candidate.label]['build_rise_gap'] <= 0.12
        and build_metrics_map[candidate.label]['build_headroom_gap'] <= 0.12
        and build_metrics_map[candidate.label]['build_plateau_risk'] <= 0.12
        and build_metrics_map[candidate.label]['build_entry_lift_gap'] <= 0.08
    ]
    if not viable_labels:
        return set()

    best_rise_gap = min(build_metrics_map[label]['build_rise_gap'] for label in viable_labels)
    best_headroom_gap = min(build_metrics_map[label]['build_headroom_gap'] for label in viable_labels)
    best_plateau_risk = min(build_metrics_map[label]['build_plateau_risk'] for label in viable_labels)
    best_entry_lift_gap = min(build_metrics_map[label]['build_entry_lift_gap'] for label in viable_labels)

    if best_rise_gap > 0.04 or best_headroom_gap > 0.06 or best_plateau_risk > 0.08:
        return set()

    return {
        label
        for label in viable_labels
        if build_metrics_map[label]['build_rise_gap'] <= (best_rise_gap + 0.04)
        and build_metrics_map[label]['build_headroom_gap'] <= (best_headroom_gap + 0.04)
        and build_metrics_map[label]['build_plateau_risk'] <= (best_plateau_risk + 0.06)
        and build_metrics_map[label]['build_entry_lift_gap'] <= (best_entry_lift_gap + 0.04)
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



def _hard_build_to_payoff_candidate_pool(
    spec: _SectionSpec,
    candidates: list[_SectionCandidate],
    build_to_payoff_metrics_map: dict[str, dict[str, float]],
    previous: _WindowSelection | None,
) -> set[str]:
    if spec.label != 'payoff' or previous is None or previous.section_label != 'build':
        return set()

    viable_labels = [
        candidate.label
        for candidate in candidates
        if build_to_payoff_metrics_map[candidate.label]['energy_lift_gap'] <= 0.06
        and build_to_payoff_metrics_map[candidate.label]['tail_dominance_gap'] <= 0.06
        and build_to_payoff_metrics_map[candidate.label]['payoff_conviction_gap'] <= 0.12
    ]
    if not viable_labels:
        return set()

    best_energy_lift_gap = min(
        build_to_payoff_metrics_map[label]['energy_lift_gap']
        for label in viable_labels
    )
    best_tail_dominance_gap = min(
        build_to_payoff_metrics_map[label]['tail_dominance_gap']
        for label in viable_labels
    )
    best_payoff_conviction_gap = min(
        build_to_payoff_metrics_map[label]['payoff_conviction_gap']
        for label in viable_labels
    )

    if best_energy_lift_gap > 0.03 and best_tail_dominance_gap > 0.03:
        return set()

    return {
        label
        for label in viable_labels
        if build_to_payoff_metrics_map[label]['energy_lift_gap'] <= (best_energy_lift_gap + 0.04)
        and build_to_payoff_metrics_map[label]['tail_dominance_gap'] <= (best_tail_dominance_gap + 0.04)
        and build_to_payoff_metrics_map[label]['payoff_conviction_gap'] <= (best_payoff_conviction_gap + 0.10)
    }


def _selection_source_role_position_penalty(
    spec: _SectionSpec,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
) -> tuple[float, dict[str, float]]:
    position = max(0.0, min(1.0, features.position))
    too_early_gap = 0.0
    too_late_gap = 0.0

    if spec.label == 'intro':
        too_late_gap = max(0.0, (position - 0.34) / 0.24)
    elif spec.label == 'verse':
        too_early_gap = max(0.0, (0.10 - position) / 0.10)
        too_late_gap = max(0.0, (position - 0.56) / 0.20)
    elif spec.label == 'build':
        too_early_gap = max(0.0, (0.22 - position) / 0.16)
        too_late_gap = max(0.0, (position - 0.78) / 0.16)
    elif spec.label == 'payoff':
        too_early_gap = max(0.0, (0.52 - position) / 0.18)
    elif spec.label == 'bridge':
        too_early_gap = max(0.0, (0.44 - position) / 0.16)
        too_late_gap = max(0.0, (position - 0.88) / 0.12)
    elif spec.label == 'outro':
        too_early_gap = max(0.0, (0.74 - position) / 0.18)

    phrase_trim_bonus = 0.0
    if candidate.origin == 'phrase_trim':
        phrase_trim_bonus = min(0.12, 0.06 + (0.06 * max(0.0, abs(position - 0.5) - 0.15)))

    penalty = min(1.0, (0.90 * too_early_gap) + (1.00 * too_late_gap) - phrase_trim_bonus)
    return max(0.0, penalty), {
        'source_role_too_early_gap': too_early_gap,
        'source_role_too_late_gap': too_late_gap,
        'source_role_trim_bonus': phrase_trim_bonus,
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


def _payoff_candidate_block_metrics(
    song: SongDNA,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
    previous: _WindowSelection | None,
) -> dict[str, float]:
    payoff_hit = _clamp01(
        (0.34 * features.end_focus)
        + (0.26 * features.plateau_stability)
        + (0.20 * features.payoff_strength)
        + (0.12 * features.hook_strength)
        + (0.08 * features.lift_strength)
    )
    sustained_conviction = _clamp01(
        (0.30 * features.tail_energy)
        + (0.24 * features.plateau_stability)
        + (0.18 * features.payoff_strength)
        + (0.16 * features.end_focus)
        + (0.12 * features.energy_confidence)
    )
    song_duration = max(float(song.duration_seconds), 1e-6)
    start_position = candidate.start / song_duration
    candidate_position = max(0.0, min(1.0, features.position))
    early_position_gap = max(0.0, 0.56 - candidate_position)
    early_start_gap = max(0.0, 0.60 - start_position)
    weak_conviction_gap = max(0.0, 0.60 - payoff_hit)
    sustained_gap = max(0.0, 0.62 - sustained_conviction)
    weak_tail_gap = max(0.0, 0.58 - features.tail_energy)
    build_lift_gap = 0.0
    if previous is not None and previous.section_label in {'build', 'bridge'}:
        desired_lift = 0.16 if previous.section_label == 'bridge' else 0.10
        build_lift_gap = max(0.0, desired_lift - (candidate.energy - previous.candidate.energy))

    hard_block = (
        (start_position < 0.30 and weak_conviction_gap > 0.05)
        or (weak_conviction_gap > 0.22 and sustained_gap > 0.18)
        or (weak_conviction_gap > 0.10 and weak_tail_gap > 0.12 and build_lift_gap > 0.10)
    )
    return {
        'payoff_hit': payoff_hit,
        'sustained_conviction': sustained_conviction,
        'early_position_gap': early_position_gap,
        'early_start_gap': early_start_gap,
        'weak_conviction_gap': weak_conviction_gap,
        'sustained_gap': sustained_gap,
        'weak_tail_gap': weak_tail_gap,
        'build_lift_gap': build_lift_gap,
        'hard_block': 1.0 if hard_block else 0.0,
    }


def _section_identity_metrics(spec: _SectionSpec, features: _RoleFeatures) -> dict[str, float]:
    if spec.label == 'intro':
        intro_identity = _clamp01(
            (0.32 * _score_position_low(features.position))
            + (0.22 * _score_energy_low(features.normalized_energy))
            + (0.16 * features.headroom)
            + (0.12 * max(0.0, 1.0 - features.end_focus))
            + (0.10 * max(0.0, 1.0 - features.hook_strength))
            + (0.08 * _score_slope_down(features.energy_slope))
            + (0.06 * features.source_section_prior)
        )
        fake_intro_risk = _clamp01(
            (0.32 * _score_position_mid(features.position, center=0.45, width=0.32))
            + (0.18 * _score_energy_mid_high(features.normalized_energy))
            + (0.18 * features.groove_drive)
            + (0.12 * features.groove_stability)
            + (0.10 * features.hook_strength)
            + (0.10 * features.end_focus)
            + (0.08 * features.payoff_strength)
            - (0.06 * features.source_section_prior)
        )
        return {
            'intro_identity': intro_identity,
            'verse_identity': 0.0,
            'fake_intro_risk': fake_intro_risk,
        }
    if spec.label == 'verse':
        verse_identity = _clamp01(
            (0.22 * _score_position_mid(features.position, center=0.34, width=0.30))
            + (0.18 * _score_energy_mid(features.normalized_energy))
            + (0.16 * _score_slope_flat(features.energy_slope))
            + (0.16 * features.groove_stability)
            + (0.12 * features.groove_drive)
            + (0.08 * features.repetition)
            + (0.06 * features.hook_strength)
            + (0.10 * features.source_section_prior)
        )
        intro_like_risk = _clamp01(
            (0.30 * _score_position_low(features.position))
            + (0.20 * _score_energy_low(features.normalized_energy))
            + (0.18 * features.headroom)
            + (0.16 * max(0.0, 1.0 - features.groove_drive))
            + (0.16 * max(0.0, 1.0 - features.groove_stability))
            - (0.12 * features.source_section_prior)
        )
        return {
            'intro_identity': 0.0,
            'verse_identity': verse_identity,
            'fake_intro_risk': intro_like_risk,
        }
    return {
        'intro_identity': 0.0,
        'verse_identity': 0.0,
        'fake_intro_risk': 0.0,
    }


def _is_hard_fake_intro_candidate(spec: _SectionSpec, metrics: dict[str, float]) -> bool:
    return (
        spec.label == 'intro'
        and metrics['intro_identity'] < 0.52
        and metrics['fake_intro_risk'] > 0.58
    )


def _hard_backbone_intro_source_pool(
    spec: _SectionSpec,
    parent_id: str,
    candidates: list[_SectionCandidate],
    identity_map: dict[str, dict[str, float]],
    intro_followthrough_metrics_map: dict[str, dict[str, float]],
    features_map: dict[str, _RoleFeatures],
    backbone_parent: str | None,
) -> set[str]:
    if spec.label != 'intro' or parent_id != backbone_parent:
        return set()

    viable_labels = [
        candidate.label
        for candidate in candidates
        if identity_map[candidate.label]['intro_identity'] >= 0.52
        and identity_map[candidate.label]['fake_intro_risk'] <= 0.60
        and (
            (
                intro_followthrough_metrics_map[candidate.label]['opening_followthrough_gap'] <= 0.42
                and intro_followthrough_metrics_map[candidate.label]['opening_followthrough_identity_gap'] <= 0.34
                and intro_followthrough_metrics_map[candidate.label]['opening_followthrough_lane_gap'] <= 0.30
            )
            or (
                candidate.origin == 'opening_lane'
                and features_map[candidate.label].position <= 0.26
            )
        )
    ]
    if len(viable_labels) < 2:
        return set()

    best_position = min(features_map[label].position for label in viable_labels)
    best_start = min(candidate.start for candidate in candidates if candidate.label in viable_labels)
    start_tolerance = max(8.0, 0.16 * max(candidate.end for candidate in candidates))
    position_tolerance = 0.14

    return {
        candidate.label
        for candidate in candidates
        if (
            candidate.label in viable_labels
            or (
                candidate.origin == 'opening_lane'
                and identity_map[candidate.label]['intro_identity'] >= 0.46
                and identity_map[candidate.label]['fake_intro_risk'] <= 0.66
            )
        )
        and features_map[candidate.label].position <= (best_position + position_tolerance)
        and candidate.start <= (best_start + start_tolerance)
    }



def _intro_followthrough_metrics(
    spec: _SectionSpec,
    parent_id: str,
    song: SongDNA,
    candidate: _SectionCandidate,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
) -> dict[str, float]:
    if spec.label != 'intro' or parent_id != backbone_parent:
        return {
            'opening_followthrough_gap': 0.0,
            'opening_followthrough_identity_gap': 0.0,
            'opening_followthrough_lane_gap': 0.0,
        }

    synthetic_intro = _WindowSelection(
        parent_id=parent_id,
        song=song,
        candidate=candidate,
        blended_error=0.0,
        score_breakdown={},
        section_label='intro',
    )
    verse_spec = _SectionSpec(
        label='verse',
        start_bar=spec.start_bar + spec.bar_count,
        bar_count=spec.bar_count,
        target_energy=max(spec.target_energy + 0.16, 0.38),
        source_parent_preference=parent_id,
        transition_in='blend',
        transition_out='lift',
    )
    verse_candidates, features_map, _ = _collect_parent_candidates(
        song,
        SECTION_TARGET_POSITION['verse'],
        verse_spec.bar_count,
        verse_spec.target_energy,
        verse_spec.label,
    )
    if not verse_candidates:
        return {
            'opening_followthrough_gap': 1.0,
            'opening_followthrough_identity_gap': 1.0,
            'opening_followthrough_lane_gap': 1.0,
        }

    viable_pairs: list[tuple[float, float]] = []
    synthetic_prior = [*prior_selections, synthetic_intro]
    for verse_candidate in verse_candidates:
        continuity = _opening_continuity_metrics(
            verse_spec,
            parent_id,
            verse_candidate,
            synthetic_prior,
            backbone_parent,
        )
        lane_gap = _clamp01(
            (0.55 * continuity['opening_phrase_jump_gap'])
            + (0.30 * continuity['opening_time_jump_gap'])
            + (0.15 * continuity['opening_rewind_gap'])
        )
        if (
            continuity['opening_rewind_gap'] > 0.0
            or continuity['opening_phrase_jump_gap'] > 0.65
            or continuity['opening_time_jump_gap'] > 0.75
        ):
            continue
        verse_features = features_map[verse_candidate.label]
        pair_metrics = _opening_lane_pair_metrics(
            verse_spec,
            parent_id,
            verse_candidate,
            verse_features,
            synthetic_prior,
            backbone_parent,
        )
        viable_pairs.append((pair_metrics['opening_joint_identity_gap'], lane_gap))

    if not viable_pairs:
        return {
            'opening_followthrough_gap': 1.0,
            'opening_followthrough_identity_gap': 1.0,
            'opening_followthrough_lane_gap': 1.0,
        }

    best_identity_gap, best_lane_gap = min(viable_pairs, key=lambda item: (item[0] + (1.15 * item[1]), item[1], item[0]))
    return {
        'opening_followthrough_gap': _clamp01((0.95 * best_identity_gap) + (1.10 * best_lane_gap)),
        'opening_followthrough_identity_gap': best_identity_gap,
        'opening_followthrough_lane_gap': best_lane_gap,
    }


def _early_verse_readability_metrics(
    spec: _SectionSpec,
    parent_id: str,
    features: _RoleFeatures,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
) -> dict[str, float]:
    if spec.label != 'verse' or parent_id != backbone_parent or not prior_selections:
        return {
            'early_verse_readability_gap': 0.0,
            'early_verse_position_gap': 0.0,
            'early_verse_clarity_gap': 0.0,
            'early_verse_payoff_risk': 0.0,
        }

    prior_intro = next(
        (
            selection for selection in reversed(prior_selections)
            if selection.parent_id == parent_id and selection.section_label == 'intro'
        ),
        None,
    )
    if prior_intro is None or any(
        selection.parent_id == parent_id and selection.section_label == 'verse'
        for selection in prior_selections
    ):
        return {
            'early_verse_readability_gap': 0.0,
            'early_verse_position_gap': 0.0,
            'early_verse_clarity_gap': 0.0,
            'early_verse_payoff_risk': 0.0,
        }

    position_gap = max(0.0, features.position - 0.42) / 0.28
    clarity_score = _clamp01(
        (0.34 * features.source_section_prior)
        + (0.22 * features.groove_stability)
        + (0.16 * features.groove_drive)
        + (0.14 * _score_energy_mid(features.normalized_energy))
        + (0.14 * _score_slope_flat(features.energy_slope))
    )
    clarity_gap = max(0.0, 0.58 - clarity_score)
    payoff_risk = _clamp01(
        (0.48 * features.payoff_strength)
        + (0.32 * features.end_focus)
        + (0.20 * features.hook_strength)
    )
    readability_gap = _clamp01(
        (0.48 * position_gap)
        + (0.74 * clarity_gap)
        + (0.34 * payoff_risk)
    )
    return {
        'early_verse_readability_gap': readability_gap,
        'early_verse_position_gap': _clamp01(position_gap),
        'early_verse_clarity_gap': _clamp01(clarity_gap),
        'early_verse_payoff_risk': payoff_risk,
    }



def _opening_lane_pair_metrics(
    spec: _SectionSpec,
    parent_id: str,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
) -> dict[str, float]:
    if spec.label != 'verse' or parent_id != backbone_parent or not prior_selections:
        return {
            'opening_joint_identity_gap': 0.0,
            'opening_joint_lane_gap': 0.0,
            'early_verse_readability_gap': 0.0,
            'early_verse_position_gap': 0.0,
            'early_verse_clarity_gap': 0.0,
            'early_verse_payoff_risk': 0.0,
        }

    prior_intro = next(
        (
            selection for selection in reversed(prior_selections)
            if selection.parent_id == parent_id and selection.section_label == 'intro'
        ),
        None,
    )
    if prior_intro is None:
        return {
            'opening_joint_identity_gap': 0.0,
            'opening_joint_lane_gap': 0.0,
            'early_verse_readability_gap': 0.0,
            'early_verse_position_gap': 0.0,
            'early_verse_clarity_gap': 0.0,
            'early_verse_payoff_risk': 0.0,
        }

    intro_candidates, _, _ = _collect_parent_candidates(prior_intro.song, 'early', spec.bar_count, 0.0, 'intro')
    if all(item.label != prior_intro.candidate.label for item in intro_candidates):
        intro_candidates = [*intro_candidates, prior_intro.candidate]
    intro_features = _candidate_role_features(prior_intro.song, intro_candidates, prior_intro.candidate)
    intro_identity = _section_identity_metrics(_SectionSpec(label='intro', start_bar=0, bar_count=spec.bar_count, target_energy=0.0, source_parent_preference=parent_id), intro_features)['intro_identity']
    verse_identity = _section_identity_metrics(spec, features)['verse_identity']
    continuity = _opening_continuity_metrics(spec, parent_id, candidate, prior_selections, backbone_parent)
    readability_metrics = _early_verse_readability_metrics(
        spec,
        parent_id,
        features,
        prior_selections,
        backbone_parent,
    )
    joint_identity_gap = _clamp01(
        max(0.0, 0.65 - ((0.48 * intro_identity) + (0.52 * verse_identity)))
        + (0.85 * readability_metrics['early_verse_readability_gap'])
    )
    joint_lane_gap = _clamp01(
        (0.55 * continuity['opening_phrase_jump_gap'])
        + (0.30 * continuity['opening_time_jump_gap'])
        + (0.15 * continuity['opening_rewind_gap'])
    )
    return {
        'opening_joint_identity_gap': joint_identity_gap,
        'opening_joint_lane_gap': joint_lane_gap,
        **readability_metrics,
    }


def _opening_continuity_metrics(
    spec: _SectionSpec,
    parent_id: str,
    candidate: _SectionCandidate,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
) -> dict[str, float]:
    if spec.label != 'verse' or parent_id != backbone_parent or not prior_selections:
        return {
            'opening_phrase_jump_gap': 0.0,
            'opening_time_jump_gap': 0.0,
            'opening_rewind_gap': 0.0,
        }

    prior_intro = next(
        (
            selection for selection in reversed(prior_selections)
            if selection.parent_id == parent_id and selection.section_label == 'intro'
        ),
        None,
    )
    if prior_intro is None:
        return {
            'opening_phrase_jump_gap': 0.0,
            'opening_time_jump_gap': 0.0,
            'opening_rewind_gap': 0.0,
        }

    intro_candidate = prior_intro.candidate
    intro_indices = _candidate_phrase_indices(intro_candidate.label)
    verse_indices = _candidate_phrase_indices(candidate.label)

    opening_phrase_jump_gap = 0.0
    if intro_indices is not None and verse_indices is not None:
        phrase_jump = verse_indices[0] - intro_indices[1]
        if phrase_jump < 0:
            opening_phrase_jump_gap = min(1.0, abs(phrase_jump) / 1.5)
        elif phrase_jump > 1:
            opening_phrase_jump_gap = min(1.0, (phrase_jump - 1) / 2.0)

    intro_span = max(intro_candidate.duration, candidate.duration, 1e-6)
    raw_time_jump = candidate.start - intro_candidate.end
    opening_time_jump_gap = 0.0
    if raw_time_jump > 0.0:
        opening_time_jump_gap = min(1.0, max(0.0, raw_time_jump - (0.35 * intro_span)) / max(1e-6, 0.90 * intro_span))

    opening_rewind_gap = 0.0
    if candidate.start < intro_candidate.start:
        opening_rewind_gap = min(1.0, (intro_candidate.start - candidate.start) / intro_span)

    return {
        'opening_phrase_jump_gap': opening_phrase_jump_gap,
        'opening_time_jump_gap': opening_time_jump_gap,
        'opening_rewind_gap': opening_rewind_gap,
    }


def _is_hard_opening_continuity_candidate(
    spec: _SectionSpec,
    parent_id: str,
    candidate: _SectionCandidate,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
) -> bool:
    metrics = _opening_continuity_metrics(spec, parent_id, candidate, prior_selections, backbone_parent)
    return (
        metrics['opening_rewind_gap'] <= 0.0
        and metrics['opening_phrase_jump_gap'] <= 0.50
        and metrics['opening_time_jump_gap'] <= 0.60
    )


def _is_hard_opening_identity_candidate(
    spec: _SectionSpec,
    parent_id: str,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
) -> bool:
    metrics = _opening_lane_pair_metrics(
        spec,
        parent_id,
        candidate,
        features,
        prior_selections,
        backbone_parent,
    )
    return (
        metrics['opening_joint_identity_gap'] <= 0.28
        and metrics['opening_joint_lane_gap'] <= 0.32
    )


def _hard_opening_readability_candidate_pool(
    spec: _SectionSpec,
    parent_id: str,
    candidates: list[_SectionCandidate],
    opening_identity_metrics_map: dict[str, dict[str, float]],
    backbone_parent: str | None,
) -> set[str]:
    if spec.label != 'verse' or parent_id != backbone_parent:
        return set()

    viable_labels = [
        candidate.label
        for candidate in candidates
        if opening_identity_metrics_map[candidate.label]['early_verse_readability_gap'] <= 0.22
        and opening_identity_metrics_map[candidate.label]['early_verse_payoff_risk'] <= 0.45
        and opening_identity_metrics_map[candidate.label]['early_verse_position_gap'] <= 0.20
        and opening_identity_metrics_map[candidate.label]['opening_joint_identity_gap'] <= 0.34
        and opening_identity_metrics_map[candidate.label]['opening_joint_lane_gap'] <= 0.34
    ]
    if not viable_labels:
        return set()

    best_readability_gap = min(
        opening_identity_metrics_map[label]['early_verse_readability_gap']
        for label in viable_labels
    )
    best_payoff_risk = min(
        opening_identity_metrics_map[label]['early_verse_payoff_risk']
        for label in viable_labels
    )
    best_position_gap = min(
        opening_identity_metrics_map[label]['early_verse_position_gap']
        for label in viable_labels
    )
    best_joint_identity_gap = min(
        opening_identity_metrics_map[label]['opening_joint_identity_gap']
        for label in viable_labels
    )
    best_joint_lane_gap = min(
        opening_identity_metrics_map[label]['opening_joint_lane_gap']
        for label in viable_labels
    )
    if best_readability_gap > 0.18:
        return set()
    return {
        label
        for label in viable_labels
        if opening_identity_metrics_map[label]['early_verse_readability_gap'] <= (best_readability_gap + 0.10)
        and opening_identity_metrics_map[label]['early_verse_payoff_risk'] <= (best_payoff_risk + 0.14)
        and opening_identity_metrics_map[label]['early_verse_position_gap'] <= (best_position_gap + 0.12)
        and opening_identity_metrics_map[label]['opening_joint_identity_gap'] <= (best_joint_identity_gap + 0.08)
        and opening_identity_metrics_map[label]['opening_joint_lane_gap'] <= (best_joint_lane_gap + 0.10)
    }


def _hard_groove_candidate_pool(
    spec: _SectionSpec,
    song: SongDNA,
    candidates: list[_SectionCandidate],
    features_map: dict[str, _RoleFeatures],
    *,
    reference_tempo_bpm: float | None = None,
) -> set[str]:
    if spec.label not in {'verse', 'payoff'} or not candidates:
        return set()

    onset_series_present = bool(_safe_float_list(song.energy.get('onset_density', [])) or _safe_float_list(song.energy.get('onset_strength', [])))
    if not onset_series_present:
        return set()

    feedback = _planner_listen_feedback(song)
    if feedback.groove_confidence < 0.42:
        return set()

    drive_gap_ceiling = {
        'verse': 0.10,
        'build': 0.14,
        'payoff': 0.14,
        'bridge': 0.16,
    }[spec.label]
    stability_gap_ceiling = {
        'verse': 0.12,
        'build': 0.16,
        'payoff': 0.16,
        'bridge': 0.18,
    }[spec.label]

    groove_metrics_map: dict[str, dict[str, float]] = {}
    viable_labels: list[str] = []
    for candidate in candidates:
        stretch_ratio, _, stretch_penalty = _stretch_profile(
            song,
            candidate,
            spec.bar_count,
            reference_tempo_bpm=reference_tempo_bpm,
        )
        stretch_gate = 0.0
        if stretch_ratio > _CONSERVATIVE_STRETCH_MAX:
            stretch_gate = _clamp01((stretch_ratio - _CONSERVATIVE_STRETCH_MAX) / max(1e-6, _HARD_STRETCH_MAX - _CONSERVATIVE_STRETCH_MAX))
        elif stretch_ratio < _CONSERVATIVE_STRETCH_MIN:
            stretch_gate = _clamp01((_CONSERVATIVE_STRETCH_MIN - stretch_ratio) / max(1e-6, _CONSERVATIVE_STRETCH_MIN - _HARD_STRETCH_MIN))

        phrase_penalty, phrase_metrics = _selection_phrase_groove_penalty(spec, features_map[candidate.label])
        groove_metrics_map[candidate.label] = {
            'stretch_ratio': stretch_ratio,
            'stretch_penalty': stretch_penalty,
            'stretch_gate': stretch_gate,
            'phrase_groove': phrase_penalty,
            'groove_drive_gap': phrase_metrics['groove_drive_gap'],
            'groove_stability_gap': phrase_metrics['groove_stability_gap'],
            'groove_drive': features_map[candidate.label].groove_drive,
            'groove_stability': features_map[candidate.label].groove_stability,
        }
        if (
            stretch_gate <= 0.0
            and stretch_penalty <= 0.16
            and phrase_penalty <= 0.16
            and phrase_metrics['groove_drive_gap'] <= drive_gap_ceiling
            and phrase_metrics['groove_stability_gap'] <= stability_gap_ceiling
        ):
            viable_labels.append(candidate.label)

    if not viable_labels:
        return set()

    best_phrase_penalty = min(groove_metrics_map[label]['phrase_groove'] for label in viable_labels)
    best_stretch_penalty = min(groove_metrics_map[label]['stretch_penalty'] for label in viable_labels)
    best_drive_gap = min(groove_metrics_map[label]['groove_drive_gap'] for label in viable_labels)
    best_stability_gap = min(groove_metrics_map[label]['groove_stability_gap'] for label in viable_labels)

    return {
        label
        for label in viable_labels
        if groove_metrics_map[label]['phrase_groove'] <= (best_phrase_penalty + 0.08)
        and groove_metrics_map[label]['stretch_penalty'] <= (best_stretch_penalty + 0.10)
        and groove_metrics_map[label]['groove_drive_gap'] <= (best_drive_gap + 0.10)
        and groove_metrics_map[label]['groove_stability_gap'] <= (best_stability_gap + 0.10)
    }


def _hard_payoff_candidate_pool(
    spec: _SectionSpec,
    candidates: list[_SectionCandidate],
    payoff_metrics_map: dict[str, dict[str, float]],
) -> set[str]:
    if spec.label != 'payoff' or spec.start_bar < 24:
        return set()

    viable_labels = [
        candidate.label
        for candidate in candidates
        if payoff_metrics_map[candidate.label]['hard_block'] <= 0.0
        and payoff_metrics_map[candidate.label]['payoff_hit'] >= 0.62
        and payoff_metrics_map[candidate.label]['sustained_conviction'] >= 0.64
        and payoff_metrics_map[candidate.label]['early_position_gap'] <= 0.12
        and payoff_metrics_map[candidate.label]['early_start_gap'] <= 0.10
    ]
    if not viable_labels:
        return set()

    best_payoff_hit = max(payoff_metrics_map[label]['payoff_hit'] for label in viable_labels)
    best_sustained_conviction = max(
        payoff_metrics_map[label]['sustained_conviction']
        for label in viable_labels
    )
    best_position_gap = min(
        payoff_metrics_map[label]['early_position_gap']
        for label in viable_labels
    )
    best_start_gap = min(
        payoff_metrics_map[label]['early_start_gap']
        for label in viable_labels
    )

    return {
        label
        for label in viable_labels
        if payoff_metrics_map[label]['payoff_hit'] >= (best_payoff_hit - 0.08)
        and payoff_metrics_map[label]['sustained_conviction'] >= (best_sustained_conviction - 0.08)
        and payoff_metrics_map[label]['early_position_gap'] <= (best_position_gap + 0.08)
        and payoff_metrics_map[label]['early_start_gap'] <= (best_start_gap + 0.08)
    }


def _bridge_reset_candidate_metrics(
    spec: _SectionSpec,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
    previous: _WindowSelection | None,
) -> dict[str, float]:
    if spec.label != 'bridge' or previous is None or previous.section_label != 'payoff':
        return {
            'energy_reset_gap': 0.0,
            'bridge_identity_gap': 0.0,
            'bridge_release_gap': 0.0,
            'late_position_gap': 0.0,
            'hard_block': 0.0,
        }

    prior_energy = max(0.0, min(1.0, previous.candidate.energy))
    current_energy = max(0.0, min(1.0, candidate.energy))
    energy_reset = prior_energy - current_energy
    energy_reset_gap = max(0.0, 0.18 - energy_reset)

    bridge_identity = _clamp01(
        (0.34 * features.novelty)
        + (0.20 * max(0.0, 1.0 - features.repetition))
        + (0.16 * _score_slope_down(features.energy_slope))
        + (0.12 * max(0.0, 1.0 - features.hook_strength))
        + (0.10 * max(0.0, 1.0 - features.payoff_strength))
        + (0.08 * max(0.0, (features.position - 0.50) / 0.35))
    )
    bridge_identity_gap = max(0.0, 0.54 - bridge_identity)
    bridge_release = _clamp01(
        (0.32 * max(0.0, 1.0 - features.normalized_energy))
        + (0.24 * max(0.0, 1.0 - features.end_focus))
        + (0.18 * features.headroom)
        + (0.16 * max(0.0, 1.0 - features.plateau_stability))
        + (0.10 * _score_slope_down(features.energy_slope))
    )
    bridge_release_gap = max(0.0, 0.52 - bridge_release)
    late_position_gap = max(0.0, 0.84 - features.position)

    hard_block = (
        (energy_reset_gap > 0.16 and bridge_identity_gap > 0.14)
        or (features.position < 0.44 and bridge_identity_gap > 0.18)
        or (energy_reset_gap > 0.10 and bridge_release_gap > 0.22 and bridge_identity_gap > 0.10)
    )
    return {
        'energy_reset_gap': energy_reset_gap,
        'bridge_identity_gap': bridge_identity_gap,
        'bridge_release_gap': bridge_release_gap,
        'late_position_gap': late_position_gap,
        'hard_block': 1.0 if hard_block else 0.0,
    }



def _hard_bridge_reset_candidate_pool(
    spec: _SectionSpec,
    candidates: list[_SectionCandidate],
    bridge_reset_metrics_map: dict[str, dict[str, float]],
) -> set[str]:
    if spec.label != 'bridge' or not bridge_reset_metrics_map:
        return set()

    viable_labels = [
        candidate.label
        for candidate in candidates
        if bridge_reset_metrics_map[candidate.label]['hard_block'] <= 0.0
        and bridge_reset_metrics_map[candidate.label]['energy_reset_gap'] <= 0.14
        and bridge_reset_metrics_map[candidate.label]['bridge_identity_gap'] <= 0.18
        and bridge_reset_metrics_map[candidate.label]['bridge_release_gap'] <= 0.16
        and bridge_reset_metrics_map[candidate.label]['late_position_gap'] <= 0.16
    ]
    if not viable_labels:
        return set()

    best_reset_gap = min(
        bridge_reset_metrics_map[label]['energy_reset_gap']
        for label in viable_labels
    )
    best_identity_gap = min(
        bridge_reset_metrics_map[label]['bridge_identity_gap']
        for label in viable_labels
    )
    best_release_gap = min(
        bridge_reset_metrics_map[label]['bridge_release_gap']
        for label in viable_labels
    )
    best_position_gap = min(
        bridge_reset_metrics_map[label]['late_position_gap']
        for label in viable_labels
    )

    if best_reset_gap > 0.10:
        return set()

    return {
        label
        for label in viable_labels
        if bridge_reset_metrics_map[label]['energy_reset_gap'] <= (best_reset_gap + 0.08)
        and bridge_reset_metrics_map[label]['bridge_identity_gap'] <= (best_identity_gap + 0.12)
        and bridge_reset_metrics_map[label]['bridge_release_gap'] <= (best_release_gap + 0.10)
        and bridge_reset_metrics_map[label]['late_position_gap'] <= (best_position_gap + 0.10)
    }


def _outro_release_candidate_metrics(
    spec: _SectionSpec,
    candidate: _SectionCandidate,
    features: _RoleFeatures,
    previous: _WindowSelection | None,
) -> dict[str, float]:
    if spec.label != 'outro' or previous is None or previous.section_label not in {'payoff', 'bridge'}:
        return {
            'energy_release_gap': 0.0,
            'outro_identity_gap': 0.0,
            'closing_lane_gap': 0.0,
            'still_climaxing_risk': 0.0,
            'late_position_gap': 0.0,
            'hard_block': 0.0,
        }

    prior_energy = max(0.0, min(1.0, previous.candidate.energy))
    current_energy = max(0.0, min(1.0, candidate.energy))
    desired_release = 0.22 if previous.section_label == 'payoff' else 0.14
    energy_release_gap = max(0.0, desired_release - (prior_energy - current_energy))

    outro_identity = _clamp01(
        (0.30 * _score_position_high(features.position))
        + (0.22 * max(0.0, 1.0 - features.normalized_energy))
        + (0.18 * max(0.0, 1.0 - features.end_focus))
        + (0.14 * features.headroom)
        + (0.10 * _score_slope_down(features.energy_slope))
        + (0.06 * features.source_section_prior)
    )
    outro_identity_gap = max(0.0, 0.56 - outro_identity)

    closing_lane = _clamp01(
        (0.30 * max(0.0, 1.0 - features.tail_energy))
        + (0.22 * max(0.0, 1.0 - features.plateau_stability))
        + (0.18 * max(0.0, 1.0 - features.payoff_strength))
        + (0.12 * max(0.0, 1.0 - features.hook_strength))
        + (0.10 * features.headroom)
        + (0.08 * _score_slope_down(features.energy_slope))
    )
    closing_lane_gap = max(0.0, 0.54 - closing_lane)

    still_climaxing_risk = _clamp01(
        (0.32 * features.tail_energy)
        + (0.24 * features.plateau_stability)
        + (0.18 * features.end_focus)
        + (0.14 * features.payoff_strength)
        + (0.12 * features.hook_strength)
        - 0.46
    )
    late_position_gap = max(0.0, 0.80 - features.position)

    hard_block = (
        (still_climaxing_risk > 0.22 and closing_lane_gap > 0.10)
        or (energy_release_gap > 0.12 and still_climaxing_risk > 0.16)
        or (late_position_gap > 0.18 and outro_identity_gap > 0.16)
    )
    return {
        'energy_release_gap': energy_release_gap,
        'outro_identity_gap': outro_identity_gap,
        'closing_lane_gap': closing_lane_gap,
        'still_climaxing_risk': still_climaxing_risk,
        'late_position_gap': late_position_gap,
        'hard_block': 1.0 if hard_block else 0.0,
    }


def _hard_outro_release_candidate_pool(
    spec: _SectionSpec,
    candidates: list[_SectionCandidate],
    outro_release_metrics_map: dict[str, dict[str, float]],
) -> set[str]:
    if spec.label != 'outro' or not outro_release_metrics_map:
        return set()

    viable_labels = [
        candidate.label
        for candidate in candidates
        if outro_release_metrics_map[candidate.label]['hard_block'] <= 0.0
        and outro_release_metrics_map[candidate.label]['energy_release_gap'] <= 0.14
        and outro_release_metrics_map[candidate.label]['outro_identity_gap'] <= 0.20
        and outro_release_metrics_map[candidate.label]['closing_lane_gap'] <= 0.18
        and outro_release_metrics_map[candidate.label]['still_climaxing_risk'] <= 0.22
        and outro_release_metrics_map[candidate.label]['late_position_gap'] <= 0.20
    ]
    if not viable_labels:
        return set()

    best_release_gap = min(outro_release_metrics_map[label]['energy_release_gap'] for label in viable_labels)
    best_identity_gap = min(outro_release_metrics_map[label]['outro_identity_gap'] for label in viable_labels)
    best_closing_gap = min(outro_release_metrics_map[label]['closing_lane_gap'] for label in viable_labels)
    best_still_climaxing = min(outro_release_metrics_map[label]['still_climaxing_risk'] for label in viable_labels)
    best_position_gap = min(outro_release_metrics_map[label]['late_position_gap'] for label in viable_labels)

    if best_release_gap > 0.10 and best_closing_gap > 0.10:
        return set()

    return {
        label
        for label in viable_labels
        if outro_release_metrics_map[label]['energy_release_gap'] <= (best_release_gap + 0.08)
        and outro_release_metrics_map[label]['outro_identity_gap'] <= (best_identity_gap + 0.12)
        and outro_release_metrics_map[label]['closing_lane_gap'] <= (best_closing_gap + 0.10)
        and outro_release_metrics_map[label]['still_climaxing_risk'] <= (best_still_climaxing + 0.12)
        and outro_release_metrics_map[label]['late_position_gap'] <= (best_position_gap + 0.10)
    }


def _selection_opening_continuity_penalty(
    spec: _SectionSpec,
    parent_id: str,
    candidate: _SectionCandidate,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
) -> tuple[float, dict[str, float]]:
    metrics = _opening_continuity_metrics(spec, parent_id, candidate, prior_selections, backbone_parent)
    penalty = min(
        1.0,
        (1.10 * metrics['opening_phrase_jump_gap'])
        + (0.95 * metrics['opening_time_jump_gap'])
        + (0.90 * metrics['opening_rewind_gap']),
    )
    return penalty, metrics


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
    donor_feature_labels = {'build', 'payoff'}
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

    if spec.label in donor_feature_labels:
        trailing_donor_feature_cluster = 0
        for selection in reversed(prior_selections):
            if selection.parent_id != donor_parent or selection.section_label not in donor_feature_labels:
                break
            trailing_donor_feature_cluster += 1
        if trailing_donor_feature_cluster >= 2:
            return 'donor_feature_cluster_limit'
    return None



def _is_hard_backbone_continuity_candidate(
    spec: _SectionSpec,
    parent_id: str,
    candidate: _SectionCandidate,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
    donor_parent: str | None,
) -> bool:
    structural_labels = {'verse', 'bridge', 'outro'}
    if spec.label not in structural_labels or backbone_parent is None or donor_parent is None:
        return True
    if parent_id != backbone_parent:
        return True

    _, metrics = _selection_backbone_continuity_penalty(
        spec,
        parent_id,
        prior_selections,
        backbone_parent,
        donor_parent,
        candidate=candidate,
    )
    return (
        metrics['off_program_structural_handoff'] <= 0.0
        and metrics['donor_reentry_after_backbone'] <= 0.0
        and metrics['forward_backbone_rewind'] <= 0.0
        and metrics['forward_backbone_jump'] <= 0.60
    )



def _hard_backbone_contiguous_lane_pool(
    spec: _SectionSpec,
    parent_id: str,
    candidates: list[_SectionCandidate],
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
    donor_parent: str | None,
) -> set[str]:
    structural_labels = {'verse', 'bridge', 'outro'}
    if spec.label not in structural_labels or parent_id != backbone_parent:
        return set()

    continuity_metrics_map = {
        candidate.label: _selection_backbone_continuity_penalty(
            spec,
            parent_id,
            prior_selections,
            backbone_parent,
            donor_parent,
            candidate=candidate,
        )[1]
        for candidate in candidates
    }
    viable_labels = [
        candidate.label
        for candidate in candidates
        if continuity_metrics_map[candidate.label]['off_program_structural_handoff'] <= 0.0
        and continuity_metrics_map[candidate.label]['donor_reentry_after_backbone'] <= 0.0
        and continuity_metrics_map[candidate.label]['forward_backbone_rewind'] <= 0.0
        and continuity_metrics_map[candidate.label]['forward_backbone_jump'] <= 0.35
    ]
    if not viable_labels:
        return set()

    best_jump_gap = min(continuity_metrics_map[label]['forward_backbone_jump'] for label in viable_labels)
    return {
        label
        for label in viable_labels
        if continuity_metrics_map[label]['forward_backbone_jump'] <= (best_jump_gap + 0.08)
    }



def _donor_support_budget_metrics(
    spec: _SectionSpec,
    parent_id: str,
    prior_selections: list[_WindowSelection],
    donor_parent: str | None,
) -> tuple[float, float]:
    if donor_parent is None or parent_id != donor_parent:
        return 0.0, 0.0

    support_cost_by_label = {
        'intro': 0.35,
        'verse': 0.60,
        'build': 0.75,
        'payoff': 1.00,
        'bridge': 0.80,
        'outro': 0.50,
    }
    current_cost = support_cost_by_label.get(spec.label, 0.60)
    spent = sum(
        support_cost_by_label.get(selection.section_label or '', 0.60)
        for selection in prior_selections
        if selection.parent_id == donor_parent
    )

    base_budget = 1.15
    if any(selection.parent_id == donor_parent and selection.section_label == 'build' for selection in prior_selections):
        base_budget += 0.25
    if any(selection.parent_id == donor_parent and selection.section_label == 'payoff' for selection in prior_selections):
        base_budget -= 0.20

    protected_late_support_slot = spec.label in {'bridge', 'outro'} or (spec.label == 'payoff' and spec.start_bar >= 40)
    if not protected_late_support_slot:
        return 0.0, spent

    if spec.label == 'payoff':
        base_budget -= 0.20
    elif spec.label in {'bridge', 'outro'}:
        base_budget -= 0.10

    budget = max(0.85, base_budget)
    overflow = max(0.0, (spent + current_cost) - budget)
    normalized_overflow = min(1.0, overflow / max(0.35, current_cost))
    return normalized_overflow, spent



def _selection_backbone_continuity_penalty(
    spec: _SectionSpec,
    parent_id: str,
    prior_selections: list[_WindowSelection],
    backbone_parent: str | None,
    donor_parent: str | None,
    candidate: _SectionCandidate | None = None,
) -> tuple[float, dict[str, float]]:
    if backbone_parent is None or donor_parent is None:
        return 0.0, {
            'off_program_structural_handoff': 0.0,
            'donor_overreach': 0.0,
            'donor_reentry_after_backbone': 0.0,
            'donor_feature_cluster_overflow': 0.0,
            'donor_support_budget_overflow': 0.0,
            'donor_support_budget_spent': 0.0,
            'forward_backbone_rewind': 0.0,
            'forward_backbone_jump': 0.0,
        }

    structural_labels = {'verse', 'bridge', 'outro'}
    donor_feature_labels = {'build', 'payoff'}
    off_program_structural_handoff = 0.0
    donor_overreach = 0.0
    donor_reentry_after_backbone = 0.0
    donor_feature_cluster_overflow = 0.0
    donor_support_budget_overflow, donor_support_budget_spent = _donor_support_budget_metrics(
        spec,
        parent_id,
        prior_selections,
        donor_parent,
    )
    forward_backbone_rewind = 0.0
    forward_backbone_jump = 0.0

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

        trailing_donor_feature_cluster = 0
        for selection in reversed(prior_selections):
            if selection.parent_id != donor_parent or selection.section_label not in donor_feature_labels:
                break
            trailing_donor_feature_cluster += 1
        if trailing_donor_feature_cluster >= 2:
            donor_feature_cluster_overflow = min(1.0, 0.85 + (0.10 * (trailing_donor_feature_cluster - 2)))

        prior_parents = [selection.parent_id for selection in prior_selections]
        if donor_parent in prior_parents:
            last_donor_idx = max(idx for idx, prior_parent in enumerate(prior_parents) if prior_parent == donor_parent)
            backbone_reclaimed_after_donor = backbone_parent in prior_parents[last_donor_idx + 1:]
            if backbone_reclaimed_after_donor:
                donor_reentry_after_backbone = 1.0 if spec.label == 'payoff' else 0.85

    if candidate is not None and parent_id == backbone_parent and spec.label in structural_labels:
        last_backbone = next((selection for selection in reversed(prior_selections) if selection.parent_id == backbone_parent), None)
        if last_backbone is not None and last_backbone.candidate is not None:
            prior = last_backbone.candidate
            reference_span = max(prior.duration, candidate.duration, 1e-6)
            if candidate.start < prior.start:
                forward_backbone_rewind = min(1.0, (prior.start - candidate.start) / reference_span)
            elif candidate.start > prior.end:
                allowed_gap = 0.30 * reference_span
                forward_backbone_jump = min(1.0, max(0.0, candidate.start - prior.end - allowed_gap) / max(1e-6, 0.85 * reference_span))

    penalty = min(
        1.0,
        (0.90 * off_program_structural_handoff)
        + (0.70 * donor_overreach)
        + (1.10 * donor_feature_cluster_overflow)
        + (1.75 * donor_support_budget_overflow)
        + (0.95 * donor_reentry_after_backbone)
        + (1.05 * forward_backbone_rewind)
        + (0.80 * forward_backbone_jump),
    )
    return penalty, {
        'off_program_structural_handoff': off_program_structural_handoff,
        'donor_overreach': min(1.0, donor_overreach),
        'donor_reentry_after_backbone': donor_reentry_after_backbone,
        'donor_feature_cluster_overflow': donor_feature_cluster_overflow,
        'donor_support_budget_overflow': donor_support_budget_overflow,
        'donor_support_budget_spent': donor_support_budget_spent,
        'forward_backbone_rewind': forward_backbone_rewind,
        'forward_backbone_jump': forward_backbone_jump,
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

        hard_opening_candidates = {
            candidate.label
            for candidate in candidates
            if _is_hard_opening_continuity_candidate(
                spec,
                parent_id,
                candidate,
                prior_selections,
                backbone_parent,
            )
        }
        opening_identity_metrics_map = {
            candidate.label: _opening_lane_pair_metrics(
                spec,
                parent_id,
                candidate,
                features_map[candidate.label],
                prior_selections,
                backbone_parent,
            )
            for candidate in candidates
        }
        intro_followthrough_metrics_map = {
            candidate.label: _intro_followthrough_metrics(
                spec,
                parent_id,
                song,
                candidate,
                prior_selections,
                backbone_parent,
            )
            for candidate in candidates
        }
        hard_intro_followthrough_candidates = {
            candidate.label
            for candidate in candidates
            if intro_followthrough_metrics_map[candidate.label]['opening_followthrough_gap'] <= 0.32
            and intro_followthrough_metrics_map[candidate.label]['opening_followthrough_identity_gap'] <= 0.28
            and intro_followthrough_metrics_map[candidate.label]['opening_followthrough_lane_gap'] <= 0.24
        }
        if hard_intro_followthrough_candidates:
            best_followthrough_gap = min(
                intro_followthrough_metrics_map[label]['opening_followthrough_gap']
                for label in hard_intro_followthrough_candidates
            )
            best_followthrough_identity_gap = min(
                intro_followthrough_metrics_map[label]['opening_followthrough_identity_gap']
                for label in hard_intro_followthrough_candidates
            )
            best_followthrough_lane_gap = min(
                intro_followthrough_metrics_map[label]['opening_followthrough_lane_gap']
                for label in hard_intro_followthrough_candidates
            )
            hard_intro_followthrough_candidates = {
                label
                for label in hard_intro_followthrough_candidates
                if intro_followthrough_metrics_map[label]['opening_followthrough_gap'] <= (best_followthrough_gap + 0.12)
                and intro_followthrough_metrics_map[label]['opening_followthrough_identity_gap'] <= (best_followthrough_identity_gap + 0.10)
                and intro_followthrough_metrics_map[label]['opening_followthrough_lane_gap'] <= (best_followthrough_lane_gap + 0.12)
            }
        hard_opening_identity_candidates = {
            candidate.label
            for candidate in candidates
            if _is_hard_opening_identity_candidate(
                spec,
                parent_id,
                candidate,
                features_map[candidate.label],
                prior_selections,
                backbone_parent,
            )
        }
        if hard_opening_identity_candidates:
            best_identity_gap = min(
                opening_identity_metrics_map[label]['opening_joint_identity_gap']
                for label in hard_opening_identity_candidates
            )
            best_lane_gap = min(
                opening_identity_metrics_map[label]['opening_joint_lane_gap']
                for label in hard_opening_identity_candidates
            )
            hard_opening_identity_candidates = {
                label
                for label in hard_opening_identity_candidates
                if opening_identity_metrics_map[label]['opening_joint_identity_gap'] <= (best_identity_gap + 0.08)
                and opening_identity_metrics_map[label]['opening_joint_lane_gap'] <= (best_lane_gap + 0.18)
            }
        hard_opening_readability_candidates = _hard_opening_readability_candidate_pool(
            spec,
            parent_id,
            candidates,
            opening_identity_metrics_map,
            backbone_parent,
        )
        hard_backbone_candidates = {
            candidate.label
            for candidate in candidates
            if _is_hard_backbone_continuity_candidate(
                spec,
                parent_id,
                candidate,
                prior_selections,
                backbone_parent,
                donor_parent,
            )
        }
        hard_backbone_contiguous_lane_pool = _hard_backbone_contiguous_lane_pool(
            spec,
            parent_id,
            candidates,
            prior_selections,
            backbone_parent,
            donor_parent,
        )
        identity_map = {
            candidate.label: _section_identity_metrics(spec, features_map[candidate.label])
            for candidate in candidates
        }
        hard_backbone_intro_source_pool = _hard_backbone_intro_source_pool(
            spec,
            parent_id,
            candidates,
            identity_map,
            intro_followthrough_metrics_map,
            features_map,
            backbone_parent,
        )
        if hard_backbone_intro_source_pool:
            best_intro_identity = max(identity_map[label]['intro_identity'] for label in hard_backbone_intro_source_pool)
            best_followthrough_gap = min(
                intro_followthrough_metrics_map[label]['opening_followthrough_gap']
                for label in hard_backbone_intro_source_pool
            )
            hard_backbone_intro_source_pool = {
                label
                for label in hard_backbone_intro_source_pool
                if identity_map[label]['intro_identity'] >= (best_intro_identity - 0.12)
                and intro_followthrough_metrics_map[label]['opening_followthrough_gap'] <= (best_followthrough_gap + 0.14)
            }
        viable_intro_labels = [
            candidate.label
            for candidate in candidates
            if not _is_hard_fake_intro_candidate(spec, identity_map[candidate.label])
        ]
        hard_groove_candidates = _hard_groove_candidate_pool(
            spec,
            song,
            candidates,
            features_map,
            reference_tempo_bpm=song_map[backbone_parent].tempo_bpm if backbone_parent in song_map else None,
        )
        payoff_block_metrics_map = {
            candidate.label: _payoff_candidate_block_metrics(
                song,
                candidate,
                features_map[candidate.label],
                previous,
            )
            for candidate in candidates
        } if spec.label == 'payoff' and spec.start_bar >= 24 else {}
        viable_payoff_labels = [
            candidate.label
            for candidate in candidates
            if payoff_block_metrics_map.get(candidate.label, {}).get('hard_block', 0.0) <= 0.0
        ]
        hard_payoff_candidates = _hard_payoff_candidate_pool(
            spec,
            candidates,
            payoff_block_metrics_map,
        )
        build_metrics_map = {
            candidate.label: _selection_build_ramp_penalty(
                spec,
                candidate,
                features_map[candidate.label],
                previous,
            )[1]
            for candidate in candidates
        } if spec.label == 'build' else {}
        hard_build_candidates = _hard_build_candidate_pool(
            spec,
            parent_id,
            candidates,
            build_metrics_map,
            previous,
        )
        build_to_payoff_metrics_map = {
            candidate.label: _selection_build_to_payoff_contrast_penalty(
                spec,
                candidate,
                features_map[candidate.label],
                previous,
            )[1]
            for candidate in candidates
        } if spec.label == 'payoff' and previous is not None and previous.section_label == 'build' else {}
        hard_build_to_payoff_candidates = _hard_build_to_payoff_candidate_pool(
            spec,
            candidates,
            build_to_payoff_metrics_map,
            previous,
        )
        bridge_reset_metrics_map = {
            candidate.label: _bridge_reset_candidate_metrics(
                spec,
                candidate,
                features_map[candidate.label],
                previous,
            )
            for candidate in candidates
        } if spec.label == 'bridge' and previous is not None and previous.section_label == 'payoff' else {}
        hard_bridge_reset_candidates = _hard_bridge_reset_candidate_pool(
            spec,
            candidates,
            bridge_reset_metrics_map,
        )
        outro_release_metrics_map = {
            candidate.label: _outro_release_candidate_metrics(
                spec,
                candidate,
                features_map[candidate.label],
                previous,
            )
            for candidate in candidates
        } if spec.label == 'outro' and previous is not None and previous.section_label in {'payoff', 'bridge'} else {}
        hard_outro_release_candidates = _hard_outro_release_candidate_pool(
            spec,
            candidates,
            outro_release_metrics_map,
        )

        for candidate in candidates:
            if hard_opening_candidates and candidate.label not in hard_opening_candidates:
                continue
            if hard_opening_identity_candidates and candidate.label not in hard_opening_identity_candidates:
                continue
            if hard_opening_readability_candidates and candidate.label not in hard_opening_readability_candidates:
                continue
            if hard_backbone_candidates and candidate.label not in hard_backbone_candidates:
                continue
            if hard_backbone_contiguous_lane_pool and candidate.label not in hard_backbone_contiguous_lane_pool:
                continue
            if spec.label == 'intro' and hard_intro_followthrough_candidates and candidate.label not in hard_intro_followthrough_candidates:
                continue
            if spec.label == 'intro' and hard_backbone_intro_source_pool and candidate.label not in hard_backbone_intro_source_pool:
                continue
            if spec.label == 'intro' and viable_intro_labels and candidate.label not in viable_intro_labels:
                continue
            if hard_groove_candidates and candidate.label not in hard_groove_candidates:
                continue
            if spec.label == 'build' and hard_build_candidates and candidate.label not in hard_build_candidates:
                continue
            if spec.label == 'payoff' and viable_payoff_labels and candidate.label not in viable_payoff_labels:
                continue
            if spec.label == 'payoff' and hard_payoff_candidates and candidate.label not in hard_payoff_candidates:
                continue
            if spec.label == 'payoff' and hard_build_to_payoff_candidates and candidate.label not in hard_build_to_payoff_candidates:
                continue
            if spec.label == 'bridge' and hard_bridge_reset_candidates and candidate.label not in hard_bridge_reset_candidates:
                continue
            if spec.label == 'outro' and hard_outro_release_candidates and candidate.label not in hard_outro_release_candidates:
                continue

            identity_metrics = identity_map[candidate.label]
            payoff_block_metrics = payoff_block_metrics_map.get(candidate.label, {
                'payoff_hit': 0.0,
                'sustained_conviction': 0.0,
                'early_position_gap': 0.0,
                'early_start_gap': 0.0,
                'weak_conviction_gap': 0.0,
                'sustained_gap': 0.0,
                'weak_tail_gap': 0.0,
                'build_lift_gap': 0.0,
                'hard_block': 0.0,
            })
            bridge_reset_metrics = bridge_reset_metrics_map.get(candidate.label, {
                'energy_reset_gap': 0.0,
                'bridge_identity_gap': 0.0,
                'bridge_release_gap': 0.0,
                'late_position_gap': 0.0,
                'hard_block': 0.0,
            })
            build_metrics = build_metrics_map.get(candidate.label, {
                'build_rise_gap': 0.0,
                'build_headroom_gap': 0.0,
                'build_plateau_risk': 0.0,
                'build_entry_lift_gap': 0.0,
            })
            outro_release_metrics = outro_release_metrics_map.get(candidate.label, {
                'energy_release_gap': 0.0,
                'outro_identity_gap': 0.0,
                'closing_lane_gap': 0.0,
                'still_climaxing_risk': 0.0,
                'late_position_gap': 0.0,
                'hard_block': 0.0,
            })
            opening_lane_metrics = opening_identity_metrics_map[candidate.label]
            intro_followthrough_metrics = intro_followthrough_metrics_map[candidate.label]
            if spec.label == 'intro':
                section_identity_penalty = min(
                    1.0,
                    max(0.0, 0.62 - identity_metrics['intro_identity'])
                    + (0.90 * identity_metrics['fake_intro_risk']),
                )
            elif spec.label == 'verse':
                section_identity_penalty = min(
                    1.0,
                    max(0.0, 0.58 - identity_metrics['verse_identity'])
                    + (0.55 * identity_metrics['fake_intro_risk']),
                )
            else:
                section_identity_penalty = 0.0

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
            opening_continuity_penalty, opening_continuity_metrics = _selection_opening_continuity_penalty(
                spec,
                parent_id,
                candidate,
                prior_selections,
                backbone_parent,
            )
            backbone_continuity_penalty, backbone_continuity_metrics = _selection_backbone_continuity_penalty(
                spec,
                parent_id,
                prior_selections,
                backbone_parent,
                donor_parent,
                candidate,
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
            build_ramp_penalty, build_ramp_metrics = _selection_build_ramp_penalty(
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
            source_role_position_penalty, source_role_position_metrics = _selection_source_role_position_penalty(
                spec,
                candidate,
                features_map[candidate.label],
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
                + (0.90 * build_ramp_penalty)
                + (0.85 * final_payoff_delivery_penalty)
                + (0.95 * build_to_payoff_contrast_penalty)
                + (1.05 * section_shape_penalty)
                + (0.95 * outro_release_metrics['energy_release_gap'])
                + (0.85 * outro_release_metrics['outro_identity_gap'])
                + (0.90 * outro_release_metrics['closing_lane_gap'])
                + (1.10 * outro_release_metrics['still_climaxing_risk'])
                + (0.70 * outro_release_metrics['late_position_gap'])
                + (1.05 * source_role_position_penalty)
                + (1.10 * reuse_penalty)
                + (0.85 * fusion_balance_penalty)
                + (0.90 * groove_continuity_penalty)
                + (0.85 * phrase_groove_penalty)
                + (1.05 * section_identity_penalty)
                + (1.05 * intro_followthrough_metrics['opening_followthrough_gap'])
                + (1.15 * opening_continuity_penalty)
                + (0.95 * opening_lane_metrics['opening_joint_identity_gap'])
                + (1.10 * opening_lane_metrics['opening_joint_lane_gap'])
                + (1.20 * backbone_continuity_penalty)
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
                        'build_ramp': build_ramp_penalty,
                        'build_rise_gap': build_ramp_metrics['build_rise_gap'],
                        'build_headroom_gap': build_ramp_metrics['build_headroom_gap'],
                        'build_plateau_risk': build_ramp_metrics['build_plateau_risk'],
                        'build_entry_lift_gap': build_ramp_metrics['build_entry_lift_gap'],
                        'final_payoff_delivery': final_payoff_delivery_penalty,
                        'build_to_payoff_contrast': build_to_payoff_contrast_penalty,
                        'contrast_energy_lift_gap': build_to_payoff_contrast_metrics['energy_lift_gap'],
                        'contrast_tail_dominance_gap': build_to_payoff_contrast_metrics['tail_dominance_gap'],
                        'contrast_payoff_conviction_gap': build_to_payoff_contrast_metrics['payoff_conviction_gap'],
                        'section_shape': section_shape_penalty,
                        'shape_intro_hotspot': section_shape_metrics['intro_hotspot'],
                        'shape_payoff_underhit': section_shape_metrics['payoff_underhit'],
                        'shape_late_drop_gap': section_shape_metrics['late_drop_gap'],
                        'payoff_hard_block': payoff_block_metrics['hard_block'],
                        'payoff_hit': payoff_block_metrics['payoff_hit'],
                        'payoff_sustained_conviction': payoff_block_metrics['sustained_conviction'],
                        'payoff_early_position_gap': payoff_block_metrics['early_position_gap'],
                        'payoff_early_start_gap': payoff_block_metrics['early_start_gap'],
                        'payoff_weak_conviction_gap': payoff_block_metrics['weak_conviction_gap'],
                        'payoff_sustained_gap': payoff_block_metrics['sustained_gap'],
                        'payoff_weak_tail_gap': payoff_block_metrics['weak_tail_gap'],
                        'payoff_build_lift_gap': payoff_block_metrics['build_lift_gap'],
                        'bridge_reset_gap': bridge_reset_metrics['energy_reset_gap'],
                        'bridge_identity_gap': bridge_reset_metrics['bridge_identity_gap'],
                        'bridge_release_gap': bridge_reset_metrics['bridge_release_gap'],
                        'bridge_late_position_gap': bridge_reset_metrics['late_position_gap'],
                        'bridge_hard_block': bridge_reset_metrics['hard_block'],
                        'outro_release_gap': outro_release_metrics['energy_release_gap'],
                        'outro_identity_gap': outro_release_metrics['outro_identity_gap'],
                        'outro_closing_lane_gap': outro_release_metrics['closing_lane_gap'],
                        'outro_still_climaxing_risk': outro_release_metrics['still_climaxing_risk'],
                        'outro_late_position_gap': outro_release_metrics['late_position_gap'],
                        'outro_hard_block': outro_release_metrics['hard_block'],
                        'source_role_position': source_role_position_penalty,
                        'source_role_too_early_gap': source_role_position_metrics['source_role_too_early_gap'],
                        'source_role_too_late_gap': source_role_position_metrics['source_role_too_late_gap'],
                        'source_role_trim_bonus': source_role_position_metrics['source_role_trim_bonus'],
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
                        'section_identity': section_identity_penalty,
                        'intro_identity': identity_metrics['intro_identity'],
                        'verse_identity': identity_metrics['verse_identity'],
                        'fake_intro_risk': identity_metrics['fake_intro_risk'],
                        'opening_followthrough': intro_followthrough_metrics['opening_followthrough_gap'],
                        'opening_followthrough_identity_gap': intro_followthrough_metrics['opening_followthrough_identity_gap'],
                        'opening_followthrough_lane_gap': intro_followthrough_metrics['opening_followthrough_lane_gap'],
                        'opening_continuity': opening_continuity_penalty,
                        'opening_phrase_jump_gap': opening_continuity_metrics['opening_phrase_jump_gap'],
                        'opening_time_jump_gap': opening_continuity_metrics['opening_time_jump_gap'],
                        'opening_rewind_gap': opening_continuity_metrics['opening_rewind_gap'],
                        'opening_joint_identity_gap': opening_lane_metrics['opening_joint_identity_gap'],
                        'opening_joint_lane_gap': opening_lane_metrics['opening_joint_lane_gap'],
                        'early_verse_readability_gap': opening_lane_metrics['early_verse_readability_gap'],
                        'early_verse_position_gap': opening_lane_metrics['early_verse_position_gap'],
                        'early_verse_clarity_gap': opening_lane_metrics['early_verse_clarity_gap'],
                        'early_verse_payoff_risk': opening_lane_metrics['early_verse_payoff_risk'],
                        'backbone_continuity': backbone_continuity_penalty,
                        'backbone_off_program_structural_handoff': backbone_continuity_metrics['off_program_structural_handoff'],
                        'backbone_donor_overreach': backbone_continuity_metrics['donor_overreach'],
                        'backbone_donor_reentry_after_backbone': backbone_continuity_metrics['donor_reentry_after_backbone'],
                        'backbone_donor_feature_cluster_overflow': backbone_continuity_metrics['donor_feature_cluster_overflow'],
                        'backbone_donor_support_budget_overflow': backbone_continuity_metrics['donor_support_budget_overflow'],
                        'backbone_donor_support_budget_spent': backbone_continuity_metrics['donor_support_budget_spent'],
                        'backbone_forward_rewind': backbone_continuity_metrics['forward_backbone_rewind'],
                        'backbone_forward_jump': backbone_continuity_metrics['forward_backbone_jump'],
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

    if spec.label == 'intro':
        chosen = ranked[0]
        if chosen.score_breakdown.get('parent_preference', 0.0) > 0.0:
            preferred_alternate = next(
                (item for item in ranked if item.score_breakdown.get('parent_preference', 1.0) <= 0.0),
                None,
            )
            if preferred_alternate is not None:
                error_delta = preferred_alternate.blended_error - chosen.blended_error
                alternate_intro_identity = preferred_alternate.score_breakdown.get('intro_identity', 0.0)
                alternate_fake_intro_risk = preferred_alternate.score_breakdown.get('fake_intro_risk', 1.0)
                alternate_followthrough_gap = preferred_alternate.score_breakdown.get('opening_followthrough', 1.0)
                alternate_followthrough_identity_gap = preferred_alternate.score_breakdown.get('opening_followthrough_identity_gap', 1.0)
                alternate_followthrough_lane_gap = preferred_alternate.score_breakdown.get('opening_followthrough_lane_gap', 1.0)
                alternate_joint_identity_gap = preferred_alternate.score_breakdown.get('opening_joint_identity_gap', 1.0)
                alternate_joint_lane_gap = preferred_alternate.score_breakdown.get('opening_joint_lane_gap', 1.0)
                alternate_stretch_ratio = preferred_alternate.score_breakdown.get('stretch_ratio', 1.0)
                alternate_stretch_gate = preferred_alternate.score_breakdown.get('stretch_gate', 0.0)
                alternate_section_identity = preferred_alternate.score_breakdown.get('section_identity', 1.0)
                alternate_shape_penalty = preferred_alternate.score_breakdown.get('shape_intro_hotspot', 1.0)
                alternate_groove_confidence = preferred_alternate.score_breakdown.get('listen_groove_confidence', 0.0)
                chosen_intro_identity = chosen.score_breakdown.get('intro_identity', 0.0)
                chosen_fake_intro_risk = chosen.score_breakdown.get('fake_intro_risk', 1.0)
                chosen_followthrough_gap = chosen.score_breakdown.get('opening_followthrough', 1.0)
                chosen_followthrough_identity_gap = chosen.score_breakdown.get('opening_followthrough_identity_gap', 1.0)
                chosen_followthrough_lane_gap = chosen.score_breakdown.get('opening_followthrough_lane_gap', 1.0)
                chosen_section_identity = chosen.score_breakdown.get('section_identity', 1.0)
                chosen_shape_penalty = chosen.score_breakdown.get('shape_intro_hotspot', 1.0)

                preferred_parent_lane_is_safe = not (
                    error_delta > 0.85
                    or alternate_intro_identity < 0.60
                    or alternate_fake_intro_risk > 0.54
                    or alternate_followthrough_gap > 0.26
                    or alternate_followthrough_identity_gap > 0.24
                    or alternate_followthrough_lane_gap > 0.20
                    or alternate_joint_identity_gap > 0.16
                    or alternate_joint_lane_gap > 0.18
                    or alternate_stretch_gate > 0.0
                    or alternate_stretch_ratio > 1.12
                    or alternate_section_identity > 0.32
                    or alternate_shape_penalty > 0.26
                    or alternate_groove_confidence < 0.54
                    or alternate_intro_identity < (chosen_intro_identity - 0.08)
                    or alternate_followthrough_gap > min(0.30, chosen_followthrough_gap + 0.10)
                )
                chosen_reads_like_pseudo_opening = (
                    chosen_fake_intro_risk > 0.34
                    or chosen_followthrough_gap > 0.24
                    or chosen_followthrough_identity_gap > 0.20
                    or chosen_followthrough_lane_gap > 0.16
                    or chosen_section_identity > 0.24
                    or chosen_shape_penalty > 0.22
                )
                preferred_parent_lane_is_materially_safer = (
                    alternate_intro_identity >= max(0.62, chosen_intro_identity - 0.04)
                    and alternate_fake_intro_risk <= min(0.28, chosen_fake_intro_risk - 0.08)
                    and alternate_followthrough_gap <= min(0.20, chosen_followthrough_gap - 0.10)
                    and alternate_followthrough_identity_gap <= min(0.16, chosen_followthrough_identity_gap - 0.04)
                    and alternate_followthrough_lane_gap <= min(0.12, chosen_followthrough_lane_gap - 0.04)
                    and alternate_section_identity <= min(0.22, chosen_section_identity - 0.04)
                    and alternate_shape_penalty <= min(0.20, chosen_shape_penalty - 0.02)
                )

                if preferred_parent_lane_is_safe and (
                    not chosen_reads_like_pseudo_opening
                    or preferred_parent_lane_is_materially_safer
                ):
                    reason = 'the preferred-parent opening lane stayed musically readable'
                    if chosen_reads_like_pseudo_opening and preferred_parent_lane_is_materially_safer:
                        reason = 'the top opening read like a pseudo-opening while the preferred-parent lane preserved a safer intro→verse backbone'
                    note = (
                        f"intro backbone-preference guard: switched to {preferred_alternate.parent_id}:{preferred_alternate.candidate.label} "
                        f"because {reason}; alt delta {error_delta:.2f}; "
                        f"intro identity {alternate_intro_identity:.2f}; followthrough gap {alternate_followthrough_gap:.2f}"
                    )
                    return preferred_alternate, note
        return chosen, None

    if spec.label == 'verse':
        chosen = ranked[0]
        intro_selection = next((selection for selection in prior_selections if selection.section_label == 'intro'), None)
        backbone_parent = spec.source_parent_preference
        if (
            intro_selection is not None
            and backbone_parent in {'A', 'B'}
            and intro_selection.parent_id != backbone_parent
            and chosen.parent_id == backbone_parent
        ):
            chosen_readability_gap = chosen.score_breakdown.get('early_verse_readability_gap', 0.0)
            chosen_position_gap = chosen.score_breakdown.get('early_verse_position_gap', 0.0)
            chosen_payoff_risk = chosen.score_breakdown.get('early_verse_payoff_risk', 0.0)
            chosen_joint_identity_gap = chosen.score_breakdown.get('opening_joint_identity_gap', 0.0)
            chosen_joint_lane_gap = chosen.score_breakdown.get('opening_joint_lane_gap', 0.0)
            if (
                chosen_readability_gap > 0.26
                or chosen_position_gap > 0.26
                or chosen_payoff_risk > 0.46
                or chosen_joint_identity_gap > 0.36
                or chosen_joint_lane_gap > 0.36
            ):
                alternate = next(
                    (
                        item for item in ranked
                        if item.parent_id == backbone_parent
                        and item.score_breakdown.get('early_verse_readability_gap', 1.0) <= 0.22
                        and item.score_breakdown.get('early_verse_position_gap', 1.0) <= 0.20
                        and item.score_breakdown.get('early_verse_payoff_risk', 1.0) <= 0.45
                        and item.score_breakdown.get('opening_joint_identity_gap', 1.0) <= 0.34
                        and item.score_breakdown.get('opening_joint_lane_gap', 1.0) <= 0.34
                        and item.score_breakdown.get('stretch_gate', 1.0) <= 0.0
                        and item.score_breakdown.get('stretch_ratio', 1.0) <= 1.12
                        and item.score_breakdown.get('listen_groove_confidence', 0.0) >= max(0.54, chosen.score_breakdown.get('listen_groove_confidence', 0.0) - 0.08)
                    ),
                    None,
                )
                if alternate is not None:
                    error_delta = alternate.blended_error - chosen.blended_error
                    if error_delta <= 0.95:
                        note = (
                            f"donor-intro opening-lane verse guard: switched to {alternate.parent_id}:{alternate.candidate.label} "
                            f"because the donor intro would otherwise hand off into a weak late backbone verse; "
                            f"alt delta {error_delta:.2f}; readability gap {alternate.score_breakdown.get('early_verse_readability_gap', 1.0):.2f}; "
                            f"position gap {alternate.score_breakdown.get('early_verse_position_gap', 1.0):.2f}"
                        )
                        return alternate, note

    if spec.label == 'payoff':
        chosen = ranked[0]
        chosen_delivery = chosen.score_breakdown.get('final_payoff_delivery', 0.0)
        chosen_hit = chosen.score_breakdown.get('payoff_hit', 0.0)
        chosen_sustained = chosen.score_breakdown.get('payoff_sustained_conviction', 0.0)
        chosen_position_gap = chosen.score_breakdown.get('payoff_early_position_gap', 0.0)
        chosen_start_gap = chosen.score_breakdown.get('payoff_early_start_gap', 0.0)
        chosen_transition_error = chosen.score_breakdown.get('transition_viability', 1.0)
        chosen_seam_risk = chosen.score_breakdown.get('seam_risk', 1.0)
        chosen_groove_confidence = chosen.score_breakdown.get('listen_groove_confidence', 0.0)

        chosen_reads_like_pseudo_payoff = (
            chosen_delivery > 0.18
            or chosen_hit < 0.74
            or chosen_sustained < 0.76
            or chosen_position_gap > 0.06
            or chosen_start_gap > 0.06
        )
        if chosen_reads_like_pseudo_payoff:
            alternate = next(
                (
                    item for item in ranked[1:]
                    if item.score_breakdown.get('final_payoff_delivery', 1.0) <= min(0.10, chosen_delivery - 0.12)
                    and item.score_breakdown.get('payoff_hit', 0.0) >= max(0.78, chosen_hit + 0.06)
                    and item.score_breakdown.get('payoff_sustained_conviction', 0.0) >= max(0.80, chosen_sustained + 0.06)
                    and item.score_breakdown.get('payoff_early_position_gap', 1.0) <= min(0.04, chosen_position_gap)
                    and item.score_breakdown.get('payoff_early_start_gap', 1.0) <= min(0.04, chosen_start_gap)
                    and item.score_breakdown.get('stretch_gate', 1.0) <= 0.0
                    and item.score_breakdown.get('stretch_ratio', 1.0) <= 1.12
                    and item.score_breakdown.get('listen_groove_confidence', 0.0) >= max(0.56, chosen_groove_confidence - 0.08)
                    and item.score_breakdown.get('seam_risk', 1.0) <= min(0.76, chosen_seam_risk + 0.12)
                    and item.score_breakdown.get('transition_viability', 1.0) <= min(0.80, chosen_transition_error + 0.14)
                ),
                None,
            )
            if alternate is not None:
                error_delta = alternate.blended_error - chosen.blended_error
                if error_delta <= 0.90:
                    note = (
                        f"late-payoff legitimacy guard: switched to {alternate.parent_id}:{alternate.candidate.label} "
                        f"because a safer late sustained payoff existed behind an earlier/weaker pseudo-payoff; "
                        f"alt delta {error_delta:.2f}; delivery {alternate.score_breakdown.get('final_payoff_delivery', 1.0):.2f}; "
                        f"payoff hit {alternate.score_breakdown.get('payoff_hit', 0.0):.2f}; sustained {alternate.score_breakdown.get('payoff_sustained_conviction', 0.0):.2f}"
                    )
                    return alternate, note

    if spec.label == 'outro':
        chosen = ranked[0]
        chosen_release_gap = chosen.score_breakdown.get('outro_release_gap', 0.0)
        chosen_closing_gap = chosen.score_breakdown.get('outro_closing_lane_gap', 0.0)
        chosen_still_climaxing = chosen.score_breakdown.get('outro_still_climaxing_risk', 0.0)
        chosen_identity_gap = chosen.score_breakdown.get('outro_identity_gap', 0.0)
        chosen_transition_error = chosen.score_breakdown.get('transition_viability', 1.0)
        chosen_seam_risk = chosen.score_breakdown.get('seam_risk', 1.0)
        chosen_groove_confidence = chosen.score_breakdown.get('listen_groove_confidence', 0.0)

        chosen_reads_like_fake_release = (
            chosen_still_climaxing > 0.18
            or chosen_release_gap > 0.08
            or chosen_closing_gap > 0.10
            or chosen_identity_gap > 0.12
        )
        if chosen_reads_like_fake_release:
            alternate = next(
                (
                    item for item in ranked[1:]
                    if item.score_breakdown.get('outro_still_climaxing_risk', 1.0) <= min(0.10, chosen_still_climaxing - 0.10)
                    and item.score_breakdown.get('outro_release_gap', 1.0) <= min(0.06, chosen_release_gap - 0.04)
                    and item.score_breakdown.get('outro_closing_lane_gap', 1.0) <= min(0.08, chosen_closing_gap - 0.04)
                    and item.score_breakdown.get('outro_identity_gap', 1.0) <= min(0.10, chosen_identity_gap + 0.02)
                    and item.score_breakdown.get('stretch_gate', 1.0) <= 0.0
                    and item.score_breakdown.get('stretch_ratio', 1.0) <= 1.12
                    and item.score_breakdown.get('listen_groove_confidence', 0.0) >= max(0.54, chosen_groove_confidence - 0.10)
                    and item.score_breakdown.get('seam_risk', 1.0) <= min(0.74, chosen_seam_risk + 0.14)
                    and item.score_breakdown.get('transition_viability', 1.0) <= min(0.82, chosen_transition_error + 0.16)
                ),
                None,
            )
            if alternate is not None:
                error_delta = alternate.blended_error - chosen.blended_error
                if error_delta <= 0.95:
                    note = (
                        f"late-outro release guard: switched to {alternate.parent_id}:{alternate.candidate.label} "
                        f"because the top outro still read like post-payoff climax while a safer closing lane existed; "
                        f"alt delta {error_delta:.2f}; release gap {alternate.score_breakdown.get('outro_release_gap', 1.0):.2f}; "
                        f"closing gap {alternate.score_breakdown.get('outro_closing_lane_gap', 1.0):.2f}; still-climaxing {alternate.score_breakdown.get('outro_still_climaxing_risk', 1.0):.2f}"
                    )
                    return alternate, note

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


def _normalize_section_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    return token or None



def _infer_transition_mode(
    spec: _SectionSpec,
    chosen: _WindowSelection,
    previous: _WindowSelection | None,
    previous_label: str | None,
) -> str | None:
    transition_in = _normalize_section_token(spec.transition_in)
    if transition_in is None:
        return None
    if previous is None:
        return None

    section_label = _normalize_section_token(spec.label)
    previous_section_label = _normalize_section_token(previous_label)

    if previous.parent_id == chosen.parent_id:
        return "same_parent_flow"
    if previous_section_label == "payoff" and section_label in {"bridge", "outro"}:
        return "arrival_handoff"
    if transition_in in {"swap", "drop"} or section_label in {"build", "payoff"}:
        return "single_owner_handoff"
    return "crossfade_support"


def _selection_identity(selection: _WindowSelection) -> tuple[str, str, float, float, str]:
    candidate = selection.candidate
    return (
        selection.parent_id,
        candidate.label,
        round(candidate.start, 6),
        round(candidate.end, 6),
        candidate.origin,
    )


def _selection_shortlist_entry(
    selection: _WindowSelection,
    *,
    rank: int,
    selected: bool,
    selected_error: float,
    top_error: float,
    backbone_parent: str,
) -> dict[str, Any]:
    candidate = selection.candidate
    return {
        'rank': rank,
        'parent_id': selection.parent_id,
        'role': 'backbone' if selection.parent_id == backbone_parent else 'donor',
        'window_label': candidate.label,
        'window_origin': candidate.origin,
        'window_seconds': {'start': round(candidate.start, 3), 'end': round(candidate.end, 3)},
        'planner_error': round(selection.blended_error, 3),
        'error_delta_vs_rank_1': round(selection.blended_error - top_error, 3),
        'error_delta_vs_selected': round(selection.blended_error - selected_error, 3),
        'selected': selected,
        'score_breakdown': {name: round(value, 3) for name, value in selection.score_breakdown.items()},
    }


def _build_selection_shortlist_diagnostics(
    selected: _WindowSelection,
    ranked: list[_WindowSelection],
    *,
    backbone_parent: str,
    limit: int = 3,
) -> dict[str, Any]:
    if not ranked:
        raise ValueError('ranked selections must not be empty')

    ranked_by_identity = {_selection_identity(item): idx + 1 for idx, item in enumerate(ranked)}
    selected_rank = ranked_by_identity.get(_selection_identity(selected))
    top_ranked = ranked[0]
    shortlist = list(ranked[: max(limit, 0)])
    shortlist_identities = {_selection_identity(item) for item in shortlist}
    selected_identity = _selection_identity(selected)
    if selected_identity not in shortlist_identities:
        shortlist.append(selected)
        shortlist_identities.add(selected_identity)

    other_parent = 'B' if selected.parent_id == 'A' else 'A'
    cross_parent_best_alternate = next((item for item in ranked if item.parent_id == other_parent), None)
    top_error = top_ranked.blended_error
    selected_error = selected.blended_error

    return {
        'selected_rank': selected_rank,
        'selected_by_guard': bool(selected_rank and selected_rank > 1),
        'rank_1_parent': top_ranked.parent_id,
        'rank_1_window_label': top_ranked.candidate.label,
        'selected_error_delta_vs_rank_1': round(selected_error - top_error, 3),
        'candidate_shortlist': [
            _selection_shortlist_entry(
                item,
                rank=ranked_by_identity[_selection_identity(item)],
                selected=_selection_identity(item) == selected_identity,
                selected_error=selected_error,
                top_error=top_error,
                backbone_parent=backbone_parent,
            )
            for item in shortlist
        ],
        'cross_parent_best_alternate': (
            _selection_shortlist_entry(
                cross_parent_best_alternate,
                rank=ranked_by_identity[_selection_identity(cross_parent_best_alternate)],
                selected=False,
                selected_error=selected_error,
                top_error=top_error,
                backbone_parent=backbone_parent,
            )
            if cross_parent_best_alternate is not None
            else None
        ),
    }


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

    cameo_rebound_idx: int | None = None
    cameo_parent: str | None = None
    if len(chosen_major) >= 4:
        major_with_indices = [
            (idx, selection)
            for idx, selection in enumerate(chosen_selections)
            if selection.section_label in major_labels
        ]
        major_parents = [selection.parent_id for _, selection in major_with_indices]
        for local_idx in range(2, len(major_with_indices) - 1):
            prev_run = major_parents[:local_idx]
            cameo = major_parents[local_idx]
            rebound = major_parents[local_idx + 1]
            if len(prev_run) < 2 or len(set(prev_run)) != 1:
                continue
            if cameo != prev_run[0] and rebound == prev_run[0]:
                cameo_rebound_idx = major_with_indices[local_idx + 1][0]
                cameo_parent = cameo
                break

    if not full_section_monopoly and not major_section_monopoly and cameo_rebound_idx is None:
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
    candidate_swaps: list[tuple[int, _WindowSelection, _WindowSelection, _SectionSpec, str]] = []
    for idx, (spec, current, ranked) in enumerate(zip(section_specs, chosen_selections, ranked_choices)):
        if major_section_monopoly and spec.label not in major_labels:
            continue
        target_parent = alternate_parent
        guard_reason = 'full one-parent major-section collapse' if major_section_monopoly and not full_section_monopoly else 'full one-parent section collapse'
        if cameo_rebound_idx is not None:
            if idx != cameo_rebound_idx or spec.label not in major_labels or cameo_parent not in {'A', 'B'}:
                continue
            target_parent = cameo_parent
            guard_reason = 'single-major cameo rebound that breaks contiguous donor identity'

        alternate = next((item for item in ranked if item.parent_id == target_parent), None)
        if alternate is None:
            continue
        if not _is_safe_authenticity_alternate(current, alternate, spec):
            continue
        candidate_swaps.append((idx, current, alternate, spec, guard_reason))

    if not candidate_swaps:
        return chosen_selections, []

    idx, current, alternate, spec, guard_reason = min(
        candidate_swaps,
        key=lambda item: (
            priority_labels.get(item[3].label, 99),
            item[2].blended_error - item[1].blended_error,
            item[0],
        ),
    )
    updated = list(chosen_selections)
    updated[idx] = alternate
    note = (
        f"section-level authenticity guard: {spec.label} switched to {alternate.parent_id}:{alternate.candidate.label} "
        f"to avoid a {guard_reason}; alt delta {alternate.blended_error - current.blended_error:.2f}; "
        f"guarded safe by stretch {alternate.score_breakdown.get('stretch_ratio', 1.0):.2f} "
        f"and groove {alternate.score_breakdown.get('listen_groove_confidence', 1.0):.2f}"
    )
    return updated, [note]


_SUPPORT_GAIN_BY_LABEL = {
    'verse': -10.5,
    'build': -9.0,
    'bridge': -9.5,
    'payoff': -8.0,
}


def _support_gain_db_for_recipe(
    spec: _SectionSpec,
    *,
    support_mode: str,
    error_delta: float,
    seam_risk: float,
    foreground_collision: float,
) -> float:
    base_gain = float(_SUPPORT_GAIN_BY_LABEL.get(spec.label, -10.0))

    closeness_bonus = 0.0
    if error_delta <= 0.14:
        closeness_bonus += 1.1
    elif error_delta <= 0.24:
        closeness_bonus += 0.7
    elif error_delta <= 0.36:
        closeness_bonus += 0.35

    seam_bonus = 0.0
    if seam_risk <= 0.40:
        seam_bonus += 0.45
    elif seam_risk <= 0.50:
        seam_bonus += 0.20

    collision_penalty = 0.0
    if foreground_collision >= 0.40:
        collision_penalty += 0.75
    elif foreground_collision >= 0.30:
        collision_penalty += 0.35

    mode_bonus = 0.0
    min_gain = -12.5
    max_gain = -7.0
    if support_mode == 'foreground_counterlayer':
        mode_bonus += 0.85
        max_gain = -6.0 if spec.label == 'payoff' else -6.5
    elif spec.label in {'build', 'bridge'}:
        max_gain = -7.5

    if spec.label == 'verse':
        max_gain = min(max_gain, -9.0)
        min_gain = -12.5
    elif spec.label == 'payoff':
        min_gain = -10.0

    gain_db = base_gain + closeness_bonus + seam_bonus + mode_bonus - collision_penalty
    return round(min(max_gain, max(min_gain, gain_db)), 2)


def _choose_support_recipe(
    spec: _SectionSpec,
    chosen: _WindowSelection,
    shortlist_diagnostics: dict[str, Any],
    *,
    backbone_parent: str,
) -> dict[str, Any] | None:
    if spec.label not in {'verse', 'build', 'bridge', 'payoff'}:
        return None

    shortlist = list(shortlist_diagnostics.get('candidate_shortlist') or [])
    alternate = shortlist_diagnostics.get('cross_parent_best_alternate') or None
    ranked_pool: list[dict[str, Any]] = []
    seen_identities: set[tuple[str, str]] = set()
    for item in ([alternate] if isinstance(alternate, dict) else []) + shortlist:
        if not isinstance(item, dict):
            continue
        alt_parent = str(item.get('parent_id') or '')
        alt_label = str(item.get('window_label') or '')
        identity = (alt_parent, alt_label)
        if not alt_parent or not alt_label or identity in seen_identities:
            continue
        seen_identities.add(identity)
        ranked_pool.append(item)

    if not ranked_pool:
        return None

    max_error_delta = 0.42
    max_stretch_gate = 0.35
    max_stretch_ratio = 1.12
    max_seam_risk = 0.58
    max_transition_viability = 0.68
    max_foreground_collision = 0.42
    if spec.label in {'build', 'bridge', 'payoff'}:
        max_error_delta = 0.68
        max_seam_risk = 0.66
        max_transition_viability = 0.76
        max_foreground_collision = 0.48

    for candidate in ranked_pool:
        alt_parent = str(candidate.get('parent_id') or '')
        alt_label = str(candidate.get('window_label') or '')
        if not alt_parent or not alt_label or alt_parent == chosen.parent_id:
            continue

        score_breakdown = dict(candidate.get('score_breakdown') or {})
        stretch_ratio = float(score_breakdown.get('stretch_ratio', 1.0) or 1.0)
        stretch_gate = float(score_breakdown.get('stretch_gate', 0.0) or 0.0)
        seam_risk = float(score_breakdown.get('seam_risk', 1.0) or 1.0)
        transition_viability = float(score_breakdown.get('transition_viability', 1.0) or 1.0)
        foreground_collision = float(
            score_breakdown.get(
                'seam_foreground_collision',
                score_breakdown.get('foreground_collision', 0.0) or 0.0,
            ) or 0.0
        )
        error_delta = float(candidate.get('error_delta_vs_selected', 99.0) or 99.0)
        if (
            error_delta > max_error_delta
            or stretch_gate > max_stretch_gate
            or stretch_ratio > max_stretch_ratio
            or seam_risk > max_seam_risk
            or transition_viability > max_transition_viability
            or foreground_collision > max_foreground_collision
        ):
            continue

        support_mode = 'filtered_counterlayer'
        if spec.label == 'payoff' and alt_parent != backbone_parent:
            support_mode = 'foreground_counterlayer'
        gain_db = _support_gain_db_for_recipe(
            spec,
            support_mode=support_mode,
            error_delta=error_delta,
            seam_risk=seam_risk,
            foreground_collision=foreground_collision,
        )
        return {
            'parent_id': alt_parent,
            'window_label': alt_label,
            'gain_db': gain_db,
            'mode': support_mode,
            'planner_error': round(float(candidate.get('planner_error', chosen.blended_error) or chosen.blended_error), 3),
            'error_delta_vs_selected': round(error_delta, 3),
            'score_breakdown': score_breakdown,
        }
    return None


_ALLOWED_ARRANGEMENT_MODES = {'adaptive', 'baseline'}


def _normalize_arrangement_mode(arrangement_mode: str | None) -> tuple[str, str | None]:
    requested = str(arrangement_mode or '').strip().lower()
    if requested in _ALLOWED_ARRANGEMENT_MODES:
        return requested, None
    return 'adaptive', (
        f"Unknown arrangement_mode={arrangement_mode!r}; falling back to 'adaptive'. "
        "Allowed values: 'adaptive', 'baseline'."
    )


def build_stub_arrangement_plan(song_a: SongDNA, song_b: SongDNA, arrangement_mode: str = 'adaptive') -> ChildArrangementPlan:
    arrangement_mode, arrangement_mode_warning = _normalize_arrangement_mode(arrangement_mode)
    report = build_compatibility_report(song_a, song_b)

    backbone_plan = _choose_backbone_parent(song_a, song_b)
    baseline_admissibility = _baseline_pair_admissibility(song_a, song_b)
    if arrangement_mode == 'baseline':
        if baseline_admissibility['admissible']:
            section_specs = _build_baseline_section_program(
                song_a,
                song_b,
                backbone_plan.backbone_parent,
                backbone_plan.donor_parent,
            )
        else:
            section_specs = [
                _SectionSpec(label='intro', start_bar=0, bar_count=8, target_energy=0.24, source_parent_preference=backbone_plan.backbone_parent, transition_out='lift'),
                _SectionSpec(label='verse', start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference=backbone_plan.backbone_parent, transition_in='blend', transition_out='lift'),
                _SectionSpec(label='payoff', start_bar=16, bar_count=8, target_energy=0.74, source_parent_preference=backbone_plan.backbone_parent, transition_in='lift', transition_out='blend'),
                _SectionSpec(label='outro', start_bar=24, bar_count=8, target_energy=0.32, source_parent_preference=backbone_plan.backbone_parent, transition_in='blend'),
            ]
    else:
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
    if arrangement_mode_warning is not None:
        selection_notes.append(arrangement_mode_warning)
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
        if not ranked:
            fallback_parent = spec.source_parent_preference if spec.source_parent_preference in {'A', 'B'} else backbone_plan.backbone_parent
            fallback_song = song_map[fallback_parent]
            fallback_candidate = (_section_candidates(fallback_song) or [
                _SectionCandidate(
                    label='section_0',
                    start=0.0,
                    end=float(fallback_song.duration_seconds),
                    duration=float(fallback_song.duration_seconds),
                    midpoint=float(fallback_song.duration_seconds) * 0.5,
                    energy=_window_energy(fallback_song, 0.0, float(fallback_song.duration_seconds)),
                    origin='section',
                )
            ])[0]
            chosen = _WindowSelection(
                parent_id=fallback_parent,
                song=fallback_song,
                candidate=fallback_candidate,
                blended_error=999.0,
                score_breakdown={
                    'fallback_no_ranked_candidates': 1.0,
                },
                section_label=spec.label,
            )
            balance_guard_note = (
                f"planner fallback: no ranked candidates available; using {fallback_parent}:{fallback_candidate.label}"
            )
        else:
            try:
                chosen, balance_guard_note = _choose_with_major_section_balance_guard(spec, ranked, selection_history)
            except Exception as exc:
                chosen = ranked[0]
                balance_guard_note = (
                    f"major-section balance guard fallback: {type(exc).__name__}: {exc}; using top-ranked candidate"
                )
        if arrangement_mode == 'baseline' and spec.source_parent_preference in {'A', 'B'}:
            preferred_choice = next((item for item in ranked if item.parent_id == spec.source_parent_preference), None)
            if preferred_choice is not None:
                chosen = preferred_choice
                if chosen is not ranked[0]:
                    balance_guard_note = (
                        f"baseline preferred-parent guard: locked {spec.label} to {chosen.parent_id}:{chosen.candidate.label} "
                        f"to preserve single-backbone / contiguous-donor chronology; alt delta {chosen.blended_error - ranked[0].blended_error:.2f}"
                    )
        if not ranked:
            ranked = [chosen]
        ranked_choices.append(ranked)
        chosen_selections.append(chosen)
        if balance_guard_note is not None:
            selection_notes.append(f"{spec.label}: {balance_guard_note}")
        selection_history.append(chosen)
        previous = chosen

    baseline_mini_arc_diagnostics = {'status': 'not_applicable'}
    if arrangement_mode == 'baseline' and baseline_admissibility['admissible']:
        chosen_selections, baseline_mini_arc_notes, baseline_mini_arc_diagnostics = _resolve_baseline_donor_mini_arc(
            section_specs,
            chosen_selections,
            ranked_choices,
            backbone_plan.backbone_parent,
            backbone_plan.donor_parent,
        )
        selection_notes.extend(baseline_mini_arc_notes)

    chosen_selections, authenticity_guard_notes = _apply_section_level_authenticity_guard(
        section_specs,
        chosen_selections,
        ranked_choices,
    )
    selection_notes.extend(authenticity_guard_notes)

    sections: list[PlannedSection] = []
    selection_diagnostics: list[dict[str, Any]] = []
    previous = None
    for spec, chosen, ranked in zip(section_specs, chosen_selections, ranked_choices):
        candidate = chosen.candidate
        transition_mode = _infer_transition_mode(spec, chosen, previous, previous.section_label if previous else None)
        continuity_treatment = None
        if transition_mode == 'same_parent_flow' and chosen.parent_id == backbone_plan.backbone_parent and spec.label in {'verse', 'bridge', 'outro'}:
            transition_mode = 'backbone_flow'
            continuity_treatment = 'backbone_flow'
        breakdown = ', '.join(f"{name}={value:.2f}" for name, value in chosen.score_breakdown.items())
        feedback = parent_feedback[chosen.parent_id]
        shortlist_diagnostics = _build_selection_shortlist_diagnostics(
            chosen,
            ranked,
            backbone_parent=backbone_plan.backbone_parent,
        )
        support_recipe = None if arrangement_mode == 'baseline' else _choose_support_recipe(
            spec,
            chosen,
            shortlist_diagnostics,
            backbone_parent=backbone_plan.backbone_parent,
        )
        primary_mi = ((song_map[chosen.parent_id].musical_intelligence or {}).get('summary') or {})
        support_parent_id = support_recipe['parent_id'] if support_recipe is not None else None
        support_mi = ((song_map[support_parent_id].musical_intelligence or {}).get('summary') or {}) if support_parent_id else {}
        recipe = build_child_section_recipe(
            section_label=spec.label,
            backbone_parent=backbone_plan.backbone_parent,
            chosen_parent=chosen.parent_id,
            chosen_label=candidate.label,
            support_recipe=support_recipe,
            primary_mi_summary=primary_mi,
            support_mi_summary=support_mi,
            arrangement_mode=arrangement_mode,
        )
        sections.append(
            PlannedSection(
                label=spec.label,
                start_bar=spec.start_bar,
                bar_count=spec.bar_count,
                source_parent=chosen.parent_id,
                source_section_label=candidate.label,
                support_parent=support_parent_id,
                support_section_label=support_recipe['window_label'] if support_recipe is not None else None,
                support_gain_db=support_recipe['gain_db'] if support_recipe is not None else None,
                support_mode=support_recipe['mode'] if support_recipe is not None else None,
                support_transition_risk=(
                    float((support_recipe.get('score_breakdown') or {}).get('seam_risk', 0.0))
                    if support_recipe is not None
                    else None
                ),
                support_foreground_collision_risk=(
                    float((support_recipe.get('score_breakdown') or {}).get('seam_foreground_collision', 0.0))
                    if support_recipe is not None
                    else None
                ),
                support_transition_viability=(
                    float((support_recipe.get('score_breakdown') or {}).get('transition_viability', 0.0))
                    if support_recipe is not None
                    else None
                ),
                backbone_owner=recipe.backbone_owner,
                donor_support_required=recipe.donor_support_required,
                motif_anchor_parent=recipe.motif_anchor_parent,
                motif_anchor_label=recipe.motif_anchor_label,
                motif_recurrence_strength=recipe.motif_recurrence_strength,
                tension_target=recipe.tension_target,
                rhythmic_constraint=recipe.rhythmic_constraint,
                harmonic_constraint=recipe.harmonic_constraint,
                timbral_anchor=recipe.timbral_anchor,
                target_energy=spec.target_energy,
                transition_in=spec.transition_in,
                transition_out=spec.transition_out,
                transition_mode=transition_mode,
            )
        )
        if support_recipe is not None:
            selection_notes.append(
                f"{spec.label}: ranked full-window candidates across both parents; chose {chosen.parent_id}:{candidate.label} ({candidate.origin}, {candidate.start:.1f}-{candidate.end:.1f}s, energy {candidate.energy:.3f}, error {chosen.blended_error:.2f}; {breakdown}) and scheduled filtered donor support from {support_recipe['parent_id']}:{support_recipe['window_label']} at {support_recipe['gain_db']:.1f} dB."
            )
        else:
            selection_notes.append(
                f"{spec.label}: ranked full-window candidates across both parents; chose {chosen.parent_id}:{candidate.label} ({candidate.origin}, {candidate.start:.1f}-{candidate.end:.1f}s, energy {candidate.energy:.3f}, error {chosen.blended_error:.2f}; {breakdown})"
            )
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
                'target_section_seconds': round(float(chosen.score_breakdown.get('target_duration_seconds', candidate.duration) or candidate.duration), 3),
                'transition_in': spec.transition_in,
                'transition_out': spec.transition_out,
                'transition_mode': transition_mode,
                'continuity_treatment': continuity_treatment,
                'planner_error': round(chosen.blended_error, 3),
                'selection_rank': shortlist_diagnostics['selected_rank'],
                'selected_by_guard': shortlist_diagnostics['selected_by_guard'],
                'selected_error_delta_vs_rank_1': shortlist_diagnostics['selected_error_delta_vs_rank_1'],
                'candidate_shortlist': shortlist_diagnostics['candidate_shortlist'],
                'cross_parent_best_alternate': shortlist_diagnostics['cross_parent_best_alternate'],
                'support_recipe': support_recipe,
                'child_section_recipe': recipe.to_dict(),
                'evaluator_alignment': {
                    'listen_feedback_penalty': round(float(chosen.score_breakdown.get('listen_feedback', 1.0) or 1.0), 3),
                    'seam_risk': round(float(chosen.score_breakdown.get('seam_risk', 1.0) or 1.0), 3),
                    'transition_viability': round(1.0 - float(chosen.score_breakdown.get('transition_viability', 1.0) or 1.0), 3),
                    'energy_arc_fit': round(1.0 - float(chosen.score_breakdown.get('energy_arc', 1.0) or 1.0), 3),
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
        'arrangement_mode': arrangement_mode,
        'arrangement_mode_warning': arrangement_mode_warning,
        'baseline_admissibility': baseline_admissibility,
        'baseline_donor_mini_arc': baseline_mini_arc_diagnostics,
        'selection_shortlist_policy': {
            'per_section_limit': 3,
            'always_include_selected_if_guarded': True,
            'include_best_cross_parent_alternate': True,
        },
        'musical_intelligence_summary': {
            parent_id: dict((song_map[parent_id].musical_intelligence or {}).get('summary') or {})
            for parent_id in song_map
        },
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
        'donor_integration_summary': {
            'sections_with_support': sum(1 for item in selection_diagnostics if item.get('support_recipe')),
            'support_parent_counts': dict(Counter(item['support_recipe']['parent_id'] for item in selection_diagnostics if item.get('support_recipe'))),
            'motif_anchor_parents': dict(Counter(item['child_section_recipe']['motif_anchor_parent'] for item in selection_diagnostics if item.get('child_section_recipe'))),
            'mean_motif_recurrence_strength': round(
                sum(float(item['child_section_recipe'].get('motif_recurrence_strength', 0.0) or 0.0) for item in selection_diagnostics if item.get('child_section_recipe')) / max(len(selection_diagnostics), 1),
                3,
            ),
            'tension_targets': [item['child_section_recipe']['tension_target'] for item in selection_diagnostics if item.get('child_section_recipe')],
        },
        'selected_sections': selection_diagnostics,
    }
    notes = [
        'Planner now uses an explicit backbone-first child-song architecture: one parent carries macro continuity while the other is inserted selectively as donor material.',
        f'Arrangement mode: {arrangement_mode}.',
        'Extended bridge/re-payoff forms are now gated by shared reset/relaunch evidence across the pair so one parent’s local late-song shape does not force a fake second climax onto the child program.',
        f"Backbone parent: {backbone_plan.backbone_parent}; donor parent: {backbone_plan.donor_parent}; reasons: {', '.join(backbone_plan.backbone_reasons)}.",
        'Planner now ranks explicit phrase windows section-by-section across both parents instead of relying on coarse early/mid/late anchor picking.',
        f'Section program is now capacity-aware instead of fixed: {program_signature}.',
        'Ranking factors are boundary confidence, role prior, target-energy fit, cross-parent compatibility, transition viability, explicit build-to-payoff contrast scoring, evaluator-style seam-risk priors, planner-facing listen feedback (groove/arc/transition/payoff readiness), backbone-continuity pressure, history-aware source-window reuse penalties, and derived hook/payoff confidence signals from canonical bar features.',
        'Sequential selection now discourages replaying the exact same source window or heavily overlapping window later in the child timeline unless the musical fit is clearly stronger.',
        'Seam-risk priors reuse listen-style handoff heuristics (energy/spectral/onset jumps plus low-end, foreground, and vocal-collision risk) to reject obviously awkward boundaries before render.',
        'Arrangement artifacts now expose listen-aligned planning_diagnostics so evaluator-facing groove/arc/transition signals are inspectable without parsing note strings.',
        'Per-section diagnostics now include a structured multi-candidate shortlist plus the best cross-parent alternate so guard-driven picks and near-miss alternates stay machine-readable downstream.',
        'Major child sections can now schedule filtered donor support from the best safe cross-parent alternate so the render can build integrated two-parent sections instead of only section-to-section handoffs.',
        'Stretch and bar-grid fit are now evaluated against the backbone parent tempo so donor phrases are judged on the child-song grid instead of each source parent silently keeping its own clock.',
        'For same-parent backbone continuity, diagnostics can emit backbone_flow and the renderer now applies a dedicated low-overlap/low-end-protect treatment so backbone handoffs stay cleaner than generic same_parent_flow.',
        'Resolver understands phrase_<start>_<end> labels and snaps them directly to analyzed phrase boundaries.',
        *selection_notes,
    ]
    if arrangement_mode == 'baseline':
        if baseline_admissibility['admissible']:
            notes.append(
                f"Baseline admissibility passed: tempo_ratio={baseline_admissibility['tempo_ratio']:.3f}, "
                f"key_distance_semitones={baseline_admissibility['key_distance_semitones']}, "
                'using a conservative single-backbone / single-donor-block plan with support layers disabled.'
            )
        else:
            notes.append(
                'Baseline admissibility failed; planner fell back to backbone-only chronology: '
                + '; '.join(baseline_admissibility['reasons'])
            )
    if len(_section_candidates(song_a)) <= 1 or len(_section_candidates(song_b)) <= 1:
        notes.append('At least one song still has coarse section analysis; phrase-window labels plus factorized ranking reduce fallback dependence when phrase boundaries exist.')

    return ChildArrangementPlan(
        parents=[_song_parent_ref(song_a), _song_parent_ref(song_b)],
        compatibility=report.factors,
        sections=sections,
        planning_notes=notes,
        planning_diagnostics=diagnostics,
    )
