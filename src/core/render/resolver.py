from __future__ import annotations

import math
import re
from typing import Any

from ..analysis.models import SongDNA
from ..planner.models import ChildArrangementPlan, PlannedSection
from .manifest import (
    AudioWorkOrder,
    ParentGrid,
    ResolverConfig,
    ResolvedRenderPlan,
    ResolvedSection,
    SourceSectionRef,
    TargetSectionTiming,
)
from .transitions import incoming_gain_db, transition_overlap_beats, transition_overlap_seconds

_CONSERVATIVE_STRETCH_MIN = 0.75
_CONSERVATIVE_STRETCH_MAX = 1.25
_HARD_STRETCH_MIN = 0.5
_HARD_STRETCH_MAX = 2.0
_GENERIC_SECTION_PREFIXES = ("section_", "part_", "segment_")
_WEAK_SECTION_SPAN_RATIO = 0.8
_PHRASE_LABEL_RE = re.compile(r"^phrase_(\d+)_(\d+)$")


def _beat_times(song: SongDNA) -> list[float]:
    return [float(x) for x in song.metadata.get("tempo", {}).get("beat_times", [])]


def _phrase_boundaries(song: SongDNA) -> list[float]:
    return [float(x) for x in song.structure.get("phrase_boundaries_seconds", [])]


def build_parent_grid(song: SongDNA, parent_id: str, config: ResolverConfig) -> ParentGrid:
    beats = sorted(float(x) for x in _beat_times(song) if float(x) >= 0.0)
    if not beats:
        beat_interval = 60.0 / max(song.tempo_bpm, 1e-6)
        total_beats = max(config.min_clip_beats, int(song.duration_seconds / beat_interval))
        beats = [i * beat_interval for i in range(total_beats + 1)]
    elif beats[0] > 0.0:
        beats = [0.0, *beats]

    phrases = sorted(float(x) for x in _phrase_boundaries(song) if 0.0 <= float(x) <= float(song.duration_seconds))
    if not phrases:
        phrase_beats = config.beats_per_bar * config.bars_per_phrase
        phrases = [beats[i] for i in range(0, len(beats), phrase_beats)]
    if not phrases or phrases[0] > 0.0:
        phrases = [0.0, *phrases]
    if phrases[-1] < float(song.duration_seconds):
        phrases.append(float(song.duration_seconds))

    return ParentGrid(
        parent_id=parent_id,
        source_path=song.source_path,
        tempo_bpm=song.tempo_bpm,
        beat_times=beats,
        phrase_boundaries_seconds=phrases,
        duration_seconds=song.duration_seconds,
    )


def _section_map(song: SongDNA) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for i, section in enumerate(song.structure.get("sections", [])):
        label = str(section.get("label") or f"section_{i}")
        mapped[label] = section
    return mapped


def _snap_before(value: float, grid: list[float]) -> float:
    vals = [g for g in grid if g <= value]
    return vals[-1] if vals else (grid[0] if grid else value)


def _snap_after(value: float, grid: list[float]) -> float:
    vals = [g for g in grid if g >= value]
    return vals[0] if vals else (grid[-1] if grid else value)


def _validate_section_timing(section_info: dict[str, Any], song: SongDNA) -> tuple[float, float, list[str]]:
    warnings: list[str] = []
    try:
        raw_start = float(section_info.get("start", 0.0))
        raw_end = float(section_info.get("end", song.duration_seconds))
    except (TypeError, ValueError):
        return 0.0, float(song.duration_seconds), ["section timing was non-numeric or malformed"]

    if raw_start < 0.0 or raw_end < 0.0:
        return 0.0, float(song.duration_seconds), ["section timing contained negative bounds"]
    if raw_end <= raw_start:
        return 0.0, float(song.duration_seconds), ["section timing was empty or reversed"]
    if raw_start >= float(song.duration_seconds):
        return 0.0, float(song.duration_seconds), ["section start exceeded song duration"]

    if raw_end > float(song.duration_seconds):
        warnings.append("section end exceeded song duration; clamped to song end")
        raw_end = float(song.duration_seconds)
    return raw_start, raw_end, warnings


def _cap_late_payoff_handoff_overlap(
    previous_label: str | None,
    current_label: str | None,
    overlap_beats: float,
    transition_mode: str | None = None,
) -> tuple[float, str | None]:
    prev = (previous_label or "").strip().lower()
    curr = (current_label or "").strip().lower()
    explicit_single_owner = transition_mode in {"arrival_handoff", "single_owner_handoff"}
    if prev == "payoff" and curr in {"outro", "bridge"}:
        cap = 0.5 if explicit_single_owner else 1.0
        if overlap_beats > cap:
            beat_label = "beat" if math.isclose(cap, 1.0, rel_tol=0.0, abs_tol=1e-9) else "beats"
            return cap, (
                f"late payoff handoff overlap capped from {overlap_beats:.1f} to {cap:.1f} {beat_label} "
                f"to reduce seam crowding and force a cleaner arrival owner"
            )
    return overlap_beats, None


def _apply_transition_mode_constraints(
    transition_mode: str | None,
    overlap_beats: float,
    cross_parent_handoff: bool,
) -> tuple[float, bool, str | None]:
    if not cross_parent_handoff or transition_mode is None:
        return overlap_beats, False, None
    if transition_mode == "arrival_handoff" and overlap_beats > 1.0:
        return 1.0, True, f"transition_mode=arrival_handoff capped overlap from {overlap_beats:.1f} to 1.0 beat and disabled donor background ownership"
    if transition_mode == "single_owner_handoff" and overlap_beats > 2.0:
        return 2.0, True, f"transition_mode=single_owner_handoff capped overlap from {overlap_beats:.1f} to 2.0 beats and disabled donor background ownership"
    if transition_mode == "crossfade_support":
        return overlap_beats, False, None
    return overlap_beats, True, f"transition_mode={transition_mode} disabled donor background ownership for an explicit single-owner handoff"


def _resolve_incoming_gain_db(
    transition_in: str | None,
    transition_mode: str | None,
    overlap_beats: float,
    previous_label: str | None,
    current_label: str | None,
) -> float:
    gain_db = incoming_gain_db(transition_in, transition_mode)
    prev = (previous_label or "").strip().lower()
    curr = (current_label or "").strip().lower()

    if overlap_beats >= 4.0 and transition_mode == "same_parent_flow":
        gain_db -= 0.75
    elif overlap_beats >= 4.0 and transition_in in {"blend", "lift"}:
        gain_db -= 0.5

    if prev == "payoff" and curr in {"bridge", "outro"} and overlap_beats > 0.0:
        gain_db -= 0.75

    return gain_db


def _phrase_label_bounds(requested_label: str | None, song: SongDNA) -> tuple[float, float, list[str]] | None:
    if not requested_label:
        return None
    match = _PHRASE_LABEL_RE.match(requested_label.strip())
    if not match:
        return None

    start_idx = int(match.group(1))
    end_idx = int(match.group(2))
    phrase_boundaries = sorted(float(x) for x in song.structure.get("phrase_boundaries_seconds", []) if 0.0 <= float(x) <= float(song.duration_seconds))
    if not phrase_boundaries:
        return 0.0, float(song.duration_seconds), [f"phrase window label '{requested_label}' could not be resolved because phrase boundaries were missing"]
    if phrase_boundaries[0] > 0.0:
        phrase_boundaries = [0.0, *phrase_boundaries]
    if phrase_boundaries[-1] < float(song.duration_seconds):
        phrase_boundaries.append(float(song.duration_seconds))

    if start_idx < 0 or end_idx >= len(phrase_boundaries) or end_idx <= start_idx:
        return 0.0, float(song.duration_seconds), [f"phrase window label '{requested_label}' was out of range for available phrase boundaries"]
    return float(phrase_boundaries[start_idx]), float(phrase_boundaries[end_idx]), []


def _target_duration_seconds(section: PlannedSection, song: SongDNA, config: ResolverConfig) -> float:
    return section.bar_count * config.beats_per_bar * 60.0 / max(float(song.tempo_bpm), 1e-6)


def _is_generic_section_label(label: str | None) -> bool:
    if not label:
        return True
    normalized = label.strip().lower()
    return any(normalized.startswith(prefix) for prefix in _GENERIC_SECTION_PREFIXES)


def _is_weak_section(requested_label: str | None, raw_start: float, raw_end: float, song: SongDNA) -> bool:
    span = max(0.0, raw_end - raw_start)
    song_duration = max(float(song.duration_seconds), 1e-6)
    return _is_generic_section_label(requested_label) or (span / song_duration) >= _WEAK_SECTION_SPAN_RATIO


def _candidate_phrase_windows(
    *,
    grid: ParentGrid,
    raw_start: float,
    raw_end: float,
    target_duration_sec: float,
    weak_section: bool,
) -> list[tuple[float, float]]:
    phrase_grid = sorted({float(x) for x in (grid.phrase_boundaries_seconds or grid.beat_times) if 0.0 <= float(x) <= grid.duration_seconds})
    if not phrase_grid:
        return [(0.0, min(grid.duration_seconds, target_duration_sec))]
    if phrase_grid[0] > 0.0:
        phrase_grid = [0.0, *phrase_grid]
    if phrase_grid[-1] < grid.duration_seconds:
        phrase_grid.append(grid.duration_seconds)

    if weak_section:
        start_candidates = phrase_grid[:-1]
    else:
        start_candidates = [value for value in phrase_grid[:-1] if raw_start <= value < raw_end]
        if not start_candidates:
            start_candidates = [_snap_before(raw_start, phrase_grid)]

    windows: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for start in start_candidates:
        end = _snap_after(start + target_duration_sec, phrase_grid)
        if end <= start:
            continue
        if end - start < target_duration_sec * _HARD_STRETCH_MIN:
            continue
        if end > grid.duration_seconds:
            end = grid.duration_seconds
        window = (max(0.0, start), min(grid.duration_seconds, end))
        if window[1] <= window[0]:
            continue
        if window not in seen:
            windows.append(window)
            seen.add(window)
    return windows


def _score_phrase_window(
    *,
    start: float,
    end: float,
    raw_start: float,
    raw_end: float,
    target_duration_sec: float,
    weak_section: bool,
    song_duration: float,
) -> tuple[float, float, float, float]:
    duration = max(0.0, end - start)
    abs_error = abs(duration - target_duration_sec)
    if duration <= 0.0:
        stretch_penalty = float("inf")
    else:
        stretch_penalty = abs(1.0 - (duration / max(target_duration_sec, 1e-6)))

    center = (start + end) * 0.5
    if weak_section:
        anchor = min(song_duration * 0.35, max(0.0, song_duration - target_duration_sec) * 0.5)
    else:
        anchor = max(raw_start, min((raw_start + raw_end) * 0.5 - (target_duration_sec * 0.5), max(raw_end - target_duration_sec, raw_start)))
    anchor_distance = abs(start - anchor)
    tail_penalty = 1.0 if end > song_duration - max(target_duration_sec * 0.5, 1.0) else 0.0
    return (abs_error, stretch_penalty, anchor_distance + tail_penalty, center)


def _select_phrase_safe_window(
    section: PlannedSection,
    song: SongDNA,
    grid: ParentGrid,
    requested_label: str | None,
    raw_start: float,
    raw_end: float,
    config: ResolverConfig,
) -> tuple[SourceSectionRef, list[str]]:
    warnings: list[str] = []
    target_duration_sec = _target_duration_seconds(section, song, config)
    weak_section = _is_weak_section(requested_label, raw_start, raw_end, song)

    if weak_section:
        if requested_label is None:
            warnings.append("source section label missing or unresolved; using phrase-safe fallback search")
        else:
            warnings.append(
                f"source section '{requested_label}' was too coarse for direct use; using phrase-safe fallback (target-length selection)"
            )

    candidates = _candidate_phrase_windows(
        grid=grid,
        raw_start=raw_start,
        raw_end=raw_end,
        target_duration_sec=target_duration_sec,
        weak_section=weak_section,
    )
    if not candidates:
        snap_grid = grid.beat_times or grid.phrase_boundaries_seconds or [0.0, grid.duration_seconds]
        fallback_start = _snap_before(raw_start if not weak_section else 0.0, snap_grid)
        fallback_end = _snap_after(fallback_start + target_duration_sec, snap_grid)
        if fallback_end <= fallback_start:
            fallback_end = min(grid.duration_seconds, fallback_start + target_duration_sec)
        warnings.append("no phrase-safe candidate window fit target duration; using beat-aligned fallback window")
        return SourceSectionRef(
            parent_id=grid.parent_id,
            source_path=song.source_path,
            source_section_label=requested_label,
            raw_start_sec=raw_start,
            raw_end_sec=raw_end,
            snapped_start_sec=max(0.0, fallback_start),
            snapped_end_sec=min(grid.duration_seconds, fallback_end),
        ), warnings

    best_start, best_end = min(
        candidates,
        key=lambda item: _score_phrase_window(
            start=item[0],
            end=item[1],
            raw_start=raw_start,
            raw_end=raw_end,
            target_duration_sec=target_duration_sec,
            weak_section=weak_section,
            song_duration=grid.duration_seconds,
        ),
    )
    return SourceSectionRef(
        parent_id=grid.parent_id,
        source_path=song.source_path,
        source_section_label=requested_label,
        raw_start_sec=raw_start,
        raw_end_sec=raw_end,
        snapped_start_sec=max(0.0, best_start),
        snapped_end_sec=min(grid.duration_seconds, best_end),
    ), warnings


def _resolve_source_window(section: PlannedSection, song: SongDNA, grid: ParentGrid, config: ResolverConfig) -> tuple[SourceSectionRef, list[str]]:
    warnings: list[str] = []
    sections = _section_map(song)
    requested_label = (section.source_section_label or "").strip() or None

    phrase_bounds = _phrase_label_bounds(requested_label, song)
    if phrase_bounds is not None:
        raw_start, raw_end, phrase_warnings = phrase_bounds
        warnings.extend(phrase_warnings)
        if not phrase_warnings:
            return SourceSectionRef(
                parent_id=grid.parent_id,
                source_path=song.source_path,
                source_section_label=requested_label,
                raw_start_sec=raw_start,
                raw_end_sec=raw_end,
                snapped_start_sec=raw_start,
                snapped_end_sec=raw_end,
            ), warnings

    section_info = sections.get(requested_label or "")
    if section_info is None:
        raw_start, raw_end = 0.0, float(song.duration_seconds)
        warnings.append("source section label missing or unresolved")
    else:
        raw_start, raw_end, timing_warnings = _validate_section_timing(section_info, song)
        warnings.extend(timing_warnings)

    source_ref, selection_warnings = _select_phrase_safe_window(
        section,
        song,
        grid,
        requested_label,
        raw_start,
        raw_end,
        config,
    )
    warnings.extend(selection_warnings)
    return source_ref, warnings


def _validate_planned_section(idx: int, sec: PlannedSection, previous_start_bar: int | None) -> None:
    if sec.source_parent not in {"A", "B"}:
        raise ValueError(f"section {idx} ({sec.label}) has unsupported source_parent: {sec.source_parent}")
    if not isinstance(sec.start_bar, int) or sec.start_bar < 0:
        raise ValueError(f"section {idx} ({sec.label}) has invalid start_bar: {sec.start_bar}")
    if not isinstance(sec.bar_count, int) or sec.bar_count <= 0:
        raise ValueError(f"section {idx} ({sec.label}) has invalid bar_count: {sec.bar_count}")
    if previous_start_bar is not None and sec.start_bar < previous_start_bar:
        raise ValueError("arrangement sections must be sorted by non-decreasing start_bar")


def _clamp_stretch_ratio(stretch_ratio: float) -> tuple[float, list[str], list[str]]:
    warnings: list[str] = []
    fallbacks: list[str] = []
    if stretch_ratio <= 0:
        stretch_ratio = 1.0
        message = "invalid stretch ratio; defaulted to 1.0"
        warnings.append(message)
        fallbacks.append(message)
        return stretch_ratio, warnings, fallbacks

    if stretch_ratio < _CONSERVATIVE_STRETCH_MIN or stretch_ratio > _CONSERVATIVE_STRETCH_MAX:
        warnings.append(
            f"stretch ratio {stretch_ratio:.3f} is outside conservative bounds "
            f"[{_CONSERVATIVE_STRETCH_MIN:.2f}, {_CONSERVATIVE_STRETCH_MAX:.2f}]"
        )

    if stretch_ratio < _HARD_STRETCH_MIN:
        fallbacks.append(f"stretch ratio {stretch_ratio:.3f} was too small; clamped to {_HARD_STRETCH_MIN:.2f}")
        stretch_ratio = _HARD_STRETCH_MIN
    elif stretch_ratio > _HARD_STRETCH_MAX:
        fallbacks.append(f"stretch ratio {stretch_ratio:.3f} was too large; clamped to {_HARD_STRETCH_MAX:.2f}")
        stretch_ratio = _HARD_STRETCH_MAX
    elif stretch_ratio < _CONSERVATIVE_STRETCH_MIN:
        fallbacks.append(f"stretch ratio {stretch_ratio:.3f} was outside conservative bounds; clamped to {_HARD_STRETCH_MAX:.2f} for v1 safety")
        stretch_ratio = _HARD_STRETCH_MAX
    elif stretch_ratio > _CONSERVATIVE_STRETCH_MAX and math.isclose(stretch_ratio, _HARD_STRETCH_MAX, rel_tol=0.0, abs_tol=1e-9):
        fallbacks.append(f"stretch ratio {stretch_ratio:.3f} was outside conservative bounds; clamped to {_HARD_STRETCH_MAX:.2f}")

    return stretch_ratio, warnings, fallbacks


def resolve_render_plan(plan: ChildArrangementPlan, parent_a: SongDNA, parent_b: SongDNA, config: ResolverConfig | None = None) -> ResolvedRenderPlan:
    config = config or ResolverConfig()
    grid_a = build_parent_grid(parent_a, "A", config)
    grid_b = build_parent_grid(parent_b, "B", config)
    song_map = {"A": parent_a, "B": parent_b}
    grid_map = {"A": grid_a, "B": grid_b}

    resolved_sections: list[ResolvedSection] = []
    work_orders: list[AudioWorkOrder] = []
    warnings: list[str] = []
    fallbacks: list[str] = []

    previous_start_bar: int | None = None
    previous_end_bar = 0
    for idx, sec in enumerate(plan.sections):
        _validate_planned_section(idx, sec, previous_start_bar)
        parent_id = sec.source_parent
        assert parent_id in song_map

        if sec.start_bar > previous_end_bar:
            gap_message = f"section {idx} ({sec.label}) starts after a gap at bar {sec.start_bar}; render timeline will contain silence"
            warnings.append(gap_message)
            fallbacks.append(gap_message)
        elif sec.start_bar < previous_end_bar:
            raise ValueError(
                f"section {idx} ({sec.label}) overlaps previous section: start_bar={sec.start_bar}, previous_end_bar={previous_end_bar}"
            )
        previous_start_bar = sec.start_bar
        previous_end_bar = sec.start_bar + sec.bar_count

        song = song_map[parent_id]
        grid = grid_map[parent_id]
        source_ref, section_warnings = _resolve_source_window(sec, song, grid, config)
        anchor_bpm = float(song.tempo_bpm)
        target_start_sec = sec.start_bar * config.beats_per_bar * 60.0 / anchor_bpm
        target_duration_sec = sec.bar_count * config.beats_per_bar * 60.0 / anchor_bpm
        target_end_sec = target_start_sec + target_duration_sec
        source_duration = max(0.0, source_ref.snapped_end_sec - source_ref.snapped_start_sec)
        raw_stretch_ratio = source_duration / target_duration_sec if target_duration_sec > 0 and source_duration > 0 else 1.0
        stretch_ratio, stretch_warnings, stretch_fallbacks = _clamp_stretch_ratio(raw_stretch_ratio)
        section_warnings.extend(stretch_warnings)

        if section_warnings:
            warnings.extend([f"section {idx} ({sec.label}): {w}" for w in section_warnings])
        if stretch_fallbacks:
            fallbacks.extend([f"section {idx} ({sec.label}): {w}" for w in stretch_fallbacks])
        if section_warnings or stretch_fallbacks:
            fallbacks.extend([f"section {idx} ({sec.label}): {w}" for w in section_warnings])

        previous_section = resolved_sections[-1] if resolved_sections else None
        previous_label = previous_section.label if previous_section else None
        overlap_beats = transition_overlap_beats(sec.transition_in, config=config, stretch_ratio=stretch_ratio)
        overlap_beats, late_handoff_warning = _cap_late_payoff_handoff_overlap(
            previous_label,
            sec.label,
            overlap_beats,
            transition_mode=sec.transition_mode,
        )
        if late_handoff_warning:
            section_warnings.append(late_handoff_warning)
        cross_parent_handoff = previous_section is not None and previous_section.source_parent != parent_id
        overlap_beats, suppress_background_owner, transition_mode_warning = _apply_transition_mode_constraints(
            sec.transition_mode,
            overlap_beats,
            cross_parent_handoff,
        )
        if transition_mode_warning:
            section_warnings.append(transition_mode_warning)
        if overlap_beats > 0.0 and (stretch_ratio < _CONSERVATIVE_STRETCH_MIN or stretch_ratio > _CONSERVATIVE_STRETCH_MAX):
            section_warnings.append(
                f"transition overlap capped to {overlap_beats:.1f} beats because stretch ratio {stretch_ratio:.3f} is outside conservative bounds"
            )
        background_owner = None
        if overlap_beats > 0.0 and cross_parent_handoff and not suppress_background_owner:
            background_owner = previous_section.source_parent
        resolved = ResolvedSection(
            index=idx,
            label=sec.label,
            source_parent=parent_id,
            source=source_ref,
            target=TargetSectionTiming(
                start_bar=sec.start_bar,
                bar_count=sec.bar_count,
                start_sec=target_start_sec,
                end_sec=target_end_sec,
                duration_sec=target_duration_sec,
                anchor_bpm=anchor_bpm,
            ),
            foreground_owner=parent_id,
            background_owner=background_owner,
            low_end_owner=parent_id,
            vocal_policy=f"{parent_id}_only",
            allowed_overlap=overlap_beats > 0,
            overlap_beats_max=overlap_beats,
            collapse_if_conflict=True,
            transition_in=sec.transition_in,
            transition_out=sec.transition_out,
            transition_mode=sec.transition_mode,
            stretch_ratio=stretch_ratio,
            semitone_shift=0.0,
            warnings=section_warnings + stretch_fallbacks,
        )
        resolved_sections.append(resolved)

        fade_in_sec = overlap_beats * 60.0 / max(anchor_bpm, 1e-6)
        work_orders.append(AudioWorkOrder(
            order_id=f"section_{idx}_base",
            section_index=idx,
            order_type="section_base",
            parent_id=parent_id,
            role="full_mix",
            source_path=source_ref.source_path,
            source_start_sec=source_ref.snapped_start_sec,
            source_end_sec=source_ref.snapped_end_sec,
            target_start_sec=target_start_sec,
            target_duration_sec=target_duration_sec,
            stretch_ratio=stretch_ratio,
            semitone_shift=0.0,
            gain_db=_resolve_incoming_gain_db(
                sec.transition_in,
                sec.transition_mode,
                overlap_beats,
                previous_label,
                sec.label,
            ),
            fade_in_sec=fade_in_sec,
            fade_out_sec=transition_overlap_seconds(sec.transition_out, anchor_bpm, config=config, stretch_ratio=stretch_ratio),
            transition_type=sec.transition_in,
            transition_mode=sec.transition_mode,
            foreground_state="owner",
            low_end_state="owner",
            vocal_state="lead_only",
            conflict_policy="collapse_to_single_source",
        ))

    target_bpm = float(plan.parents[0].tempo_bpm if plan.parents else parent_a.tempo_bpm)
    return ResolvedRenderPlan(
        schema_version="0.1.0",
        sample_rate=config.sample_rate,
        target_bpm=target_bpm,
        sections=resolved_sections,
        work_orders=work_orders,
        warnings=warnings,
        fallbacks=fallbacks,
    )
