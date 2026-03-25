# LISTEN CODE SINGLE FILE
# Generated for read-through convenience.
# Source blocks: src/core/evaluation/listen.py + ai_dj.py listen command + server.py song-rating helpers/api

# ===== BEGIN src/core/evaluation/listen.py =====

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np

from ..analysis.models import SongDNA
from .models import ListenReport, ListenSubscore


TRANSITION_ANALYSIS_VERSION = "0.5.0"


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_neighbor_manifest(song: SongDNA) -> dict[str, Any] | None:
    track_path = Path(song.source_path)
    candidates = [track_path.with_name('render_manifest.json')]
    if track_path.parent != track_path.parent.parent:
        candidates.append(track_path.parent.parent / 'render_manifest.json')
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = _safe_json(candidate)
        if not payload:
            continue
        outputs = payload.get('outputs') or {}
        output_matches = {Path(v).resolve() for v in outputs.values() if isinstance(v, str)}
        try:
            resolved_track = track_path.resolve()
        except Exception:
            resolved_track = track_path
        if not output_matches or resolved_track in output_matches:
            return payload
    return None


def _manifest_primary_sequence(manifest: dict[str, Any] | None) -> list[str]:
    if not manifest:
        return []
    primary_sequence: list[str] = []
    for section in manifest.get('sections') or []:
        owner = section.get('source_parent') or section.get('foreground_owner')
        if owner in {'A', 'B'}:
            primary_sequence.append(str(owner))
    return primary_sequence


def _manifest_overlap_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    sections = manifest.get('sections') or []
    work_orders = manifest.get('work_orders') or []
    warnings = [str(x) for x in (manifest.get('warnings') or [])]
    fallbacks = [str(x) for x in (manifest.get('fallbacks') or [])]

    overlap_sections = 0
    explicit_overlap_beats = 0.0
    long_overlap_sections = 0
    multi_owner_conflicts = 0
    crowding_sections = 0
    lead_conflicts = 0
    conservative_collapses = 0
    seam_risk_sections = 0
    stretch_warning_sections = 0
    low_end_switches = 0
    low_end_overlap_switches = 0
    low_end_owner_sequence: list[str] = []
    transition_risk_rows: list[dict[str, Any]] = []
    section_primary_counts = {'A': 0, 'B': 0}
    foreground_owner_counts = {'A': 0, 'B': 0}
    background_only_presence_counts = {'A': 0, 'B': 0}
    major_section_primary_counts = {'A': 0, 'B': 0}
    integrated_two_parent_indices: set[int] = set()
    foreground_counter_indices: set[int] = set()
    support_layer_indices: set[int] = set()
    major_labels = {'verse', 'build', 'payoff', 'bridge'}

    for section in sections:
        allowed_overlap = bool(section.get('allowed_overlap', False))
        overlap_beats = float(section.get('overlap_beats_max', 0.0) or 0.0)
        label = str(section.get('label') or '')
        primary_owner = section.get('source_parent') or section.get('foreground_owner')
        fg = section.get('foreground_owner')
        bg = section.get('background_owner')
        low = section.get('low_end_owner')
        vocal_policy = str(section.get('vocal_policy') or '')
        transition_in = section.get('transition_in')
        transition_out = section.get('transition_out')
        stretch_ratio = abs(float(section.get('stretch_ratio', 1.0) or 1.0) - 1.0)
        collapse_if_conflict = bool(section.get('collapse_if_conflict', False))

        if primary_owner in section_primary_counts:
            section_primary_counts[primary_owner] += 1
            if label in major_labels:
                major_section_primary_counts[primary_owner] += 1
        if fg in foreground_owner_counts:
            foreground_owner_counts[fg] += 1
        raw_section_index = section.get('index')
        section_index = int(raw_section_index) if isinstance(raw_section_index, int) else len(transition_risk_rows)
        if bg in background_only_presence_counts and bg not in {primary_owner, fg, low}:
            background_only_presence_counts[bg] += 1
        if allowed_overlap and primary_owner in {'A', 'B'} and bg in {'A', 'B'} and bg != primary_owner:
            integrated_two_parent_indices.add(section_index)
            support_layer_indices.add(section_index)
        if allowed_overlap and primary_owner in {'A', 'B'} and fg in {'A', 'B'} and fg != primary_owner:
            foreground_counter_indices.add(section_index)
        if low in {'A', 'B'}:
            low_end_owner_sequence.append(str(low))

        overlap_risk = 0.0
        if allowed_overlap:
            overlap_sections += 1
            explicit_overlap_beats += overlap_beats
            overlap_risk += min(overlap_beats / 8.0, 1.0)
            if overlap_beats >= 4.0:
                long_overlap_sections += 1

        if fg and bg and fg == bg:
            overlap_risk += 0.2
        if fg and low and fg != low and allowed_overlap:
            overlap_risk += 0.15

        if vocal_policy == 'both' or ('A_only' not in vocal_policy and 'B_only' not in vocal_policy and 'none' not in vocal_policy and allowed_overlap):
            lead_conflicts += 1
            overlap_risk += 0.45

        if allowed_overlap and bg is not None and overlap_beats >= 2.0:
            crowding_sections += 1
            overlap_risk += 0.25

        if transition_in in {'blend', 'swap', 'lift', 'drop'} or transition_out in {'blend', 'swap', 'lift', 'drop'}:
            overlap_risk += 0.15
        if stretch_ratio > 0.20:
            seam_risk_sections += 1
            stretch_warning_sections += 1
            overlap_risk += min(stretch_ratio / 0.35, 0.6)
        elif stretch_ratio > 0.08:
            seam_risk_sections += 1
            overlap_risk += 0.15
        if collapse_if_conflict:
            conservative_collapses += 1

        if overlap_risk > 0.0:
            transition_risk_rows.append({
                'section_index': int(section.get('index', len(transition_risk_rows))),
                'label': str(section.get('label') or f'section_{len(transition_risk_rows)}'),
                'allowed_overlap': allowed_overlap,
                'overlap_beats_max': round(overlap_beats, 3),
                'stretch_delta': round(stretch_ratio, 3),
                'risk': round(min(overlap_risk, 1.5), 3),
            })

    for left, right in zip(sections, sections[1:]):
        left_low = left.get('low_end_owner')
        right_low = right.get('low_end_owner')
        if left_low in {'A', 'B'} and right_low in {'A', 'B'} and left_low != right_low:
            low_end_switches += 1
            if bool(left.get('allowed_overlap', False)) or bool(right.get('allowed_overlap', False)):
                low_end_overlap_switches += 1

    owner_orders = {}
    for order in work_orders:
        idx = int(order.get('section_index', -1))
        owner_orders.setdefault(idx, []).append(order)

    for idx, orders in owner_orders.items():
        low_end_owners = {o.get('parent_id') for o in orders if o.get('low_end_state') == 'owner'}
        foreground_owners = {o.get('parent_id') for o in orders if o.get('foreground_state') == 'owner'}
        lead_vocal_owners = {o.get('parent_id') for o in orders if o.get('vocal_state') in {'lead_only', 'lead'} }
        parent_ids = {o.get('parent_id') for o in orders if o.get('parent_id') in {'A', 'B'}}
        roles = {str(o.get('role') or '') for o in orders}
        if len(parent_ids) > 1 and any(role != 'full_mix' for role in roles):
            integrated_two_parent_indices.add(idx)
        if 'foreground_counterlayer' in roles:
            foreground_counter_indices.add(idx)
        if any(role in {'filtered_counterlayer', 'filtered_support', 'foreground_counterlayer'} for role in roles):
            support_layer_indices.add(idx)
        if len(low_end_owners) > 1 or len(foreground_owners) > 1:
            multi_owner_conflicts += 1
        if len(lead_vocal_owners) > 1:
            lead_conflicts += 1
        dense_active_orders = sum(1 for o in orders if o.get('foreground_state') == 'owner' or o.get('low_end_state') == 'owner')
        if dense_active_orders > 1:
            crowding_sections += 1

    warning_text = ' '.join(warnings + fallbacks).lower()
    if 'stretch ratio' in warning_text:
        stretch_warning_sections = max(stretch_warning_sections, 1)
    if 'collapse' in warning_text:
        conservative_collapses = max(conservative_collapses, 1)

    primary_sequence = _manifest_primary_sequence(manifest)
    owner_switches = sum(1 for left, right in zip(primary_sequence, primary_sequence[1:]) if left != right)
    owner_switch_ratio = owner_switches / max(len(primary_sequence) - 1, 1) if len(primary_sequence) >= 2 else 0.0

    section_count = max(len(sections), 1)
    low_end_owner_count = len(low_end_owner_sequence)
    low_end_switch_ratio = low_end_switches / max(low_end_owner_count - 1, 1) if low_end_owner_count >= 2 else 0.0
    low_end_overlap_switch_ratio = low_end_overlap_switches / max(low_end_owner_count - 1, 1) if low_end_owner_count >= 2 else 0.0
    low_end_ping_pong_count = sum(
        1
        for left, mid, right in zip(low_end_owner_sequence, low_end_owner_sequence[1:], low_end_owner_sequence[2:])
        if left == right and left != mid
    )
    low_end_ping_pong_ratio = low_end_ping_pong_count / max(low_end_owner_count - 2, 1) if low_end_owner_count >= 3 else 0.0
    longest_low_end_run = 0
    current_low_end_run = 0
    previous_low_end_owner: str | None = None
    for owner in low_end_owner_sequence:
        if owner == previous_low_end_owner:
            current_low_end_run += 1
        else:
            current_low_end_run = 1
            previous_low_end_owner = owner
        longest_low_end_run = max(longest_low_end_run, current_low_end_run)
    low_end_longest_run_ratio = longest_low_end_run / max(low_end_owner_count, 1) if low_end_owner_count else 0.0
    low_end_owner_majority_ratio = max((low_end_owner_sequence.count('A'), low_end_owner_sequence.count('B')), default=0) / max(low_end_owner_count, 1) if low_end_owner_count else 0.0
    low_end_owner_stability_risk = _clamp01(
        0.45 * _clamp01((low_end_switch_ratio - 0.22) / 0.48)
        + 0.25 * _clamp01((low_end_overlap_switch_ratio - 0.10) / 0.40)
        + 0.20 * _clamp01((low_end_ping_pong_ratio - 0.10) / 0.45)
        + 0.10 * _clamp01((0.45 - low_end_longest_run_ratio) / 0.25)
    )
    true_two_parent_section_ratio = round(min(section_primary_counts.values()) / section_count, 3)
    true_two_parent_major_section_ratio = round(min(major_section_primary_counts.values()) / section_count, 3)
    background_only_presence_ratio = round(sum(background_only_presence_counts.values()) / section_count, 3)
    minority_parent = min(section_primary_counts, key=section_primary_counts.get)
    background_only_identity_gap = round(
        max(0.0, (background_only_presence_counts[minority_parent] / section_count) - true_two_parent_section_ratio),
        3,
    )

    integrated_two_parent_section_ratio = round(len(integrated_two_parent_indices) / section_count, 3)
    foreground_counter_section_ratio = round(len(foreground_counter_indices) / section_count, 3)
    support_layer_section_ratio = round(len(support_layer_indices) / section_count, 3)
    full_mix_medley_risk = round(
        _clamp01(
            0.38 * _clamp01((owner_switch_ratio - 0.32) / 0.40)
            + 0.26 * _clamp01((0.34 - integrated_two_parent_section_ratio) / 0.34)
            + 0.18 * _clamp01((0.24 - support_layer_section_ratio) / 0.24)
            + 0.18 * float(conservative_collapses / section_count)
        ),
        3,
    )
    aggregate = {
        'section_count': len(sections),
        'overlap_section_ratio': round(overlap_sections / section_count, 3),
        'avg_overlap_beats': round(explicit_overlap_beats / section_count, 3),
        'long_overlap_ratio': round(long_overlap_sections / section_count, 3),
        'multi_owner_conflict_ratio': round(multi_owner_conflicts / section_count, 3),
        'lead_conflict_ratio': round(lead_conflicts / section_count, 3),
        'crowding_ratio': round(crowding_sections / section_count, 3),
        'seam_risk_ratio': round(seam_risk_sections / section_count, 3),
        'stretch_warning_ratio': round(stretch_warning_sections / section_count, 3),
        'collapse_ratio': round(conservative_collapses / section_count, 3),
        'integrated_two_parent_section_ratio': integrated_two_parent_section_ratio,
        'foreground_counter_section_ratio': foreground_counter_section_ratio,
        'support_layer_section_ratio': support_layer_section_ratio,
        'full_mix_medley_risk': full_mix_medley_risk,
        'low_end_owner_switch_count': low_end_switches,
        'low_end_owner_switch_ratio': round(low_end_switch_ratio, 3),
        'low_end_overlap_switch_ratio': round(low_end_overlap_switch_ratio, 3),
        'low_end_ping_pong_count': low_end_ping_pong_count,
        'low_end_ping_pong_ratio': round(low_end_ping_pong_ratio, 3),
        'low_end_longest_run_ratio': round(low_end_longest_run_ratio, 3),
        'low_end_owner_majority_ratio': round(low_end_owner_majority_ratio, 3),
        'low_end_owner_stability_risk': round(low_end_owner_stability_risk, 3),
        'true_two_parent_section_ratio': true_two_parent_section_ratio,
        'true_two_parent_major_section_ratio': true_two_parent_major_section_ratio,
        'background_only_presence_ratio': background_only_presence_ratio,
        'background_only_identity_gap': background_only_identity_gap,
        'warning_count': len(warnings),
        'fallback_count': len(fallbacks),
    }
    return {
        'aggregate_metrics': aggregate,
        'fusion_identity': {
            'section_primary_counts': section_primary_counts,
            'major_section_primary_counts': major_section_primary_counts,
            'foreground_owner_counts': foreground_owner_counts,
            'background_only_presence_counts': background_only_presence_counts,
            'minority_parent': minority_parent,
        },
        'risky_sections': sorted(transition_risk_rows, key=lambda row: row['risk'], reverse=True)[:5],
        'warnings': warnings[:5],
        'fallbacks': fallbacks[:5],
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_band(value: float, low: float, high: float, soft: float) -> float:
    if low <= value <= high:
        return 1.0
    dist = min(abs(value - low), abs(value - high))
    return _clamp01(1.0 - (dist / max(soft, 1e-6)))


def _median_diff(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    if vals.size < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(vals))))


def _safe_array(values: object) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr[np.isfinite(arr)]


def _series_times(values: np.ndarray, duration_seconds: float) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=float)
    if values.size == 1:
        return np.asarray([0.0], dtype=float)
    return np.linspace(0.0, max(duration_seconds, 1e-6), num=values.size, endpoint=False, dtype=float)


def _window_mean(values: np.ndarray, times: np.ndarray, start: float, end: float) -> float:
    if values.size == 0:
        return 0.0
    if times.size != values.size:
        return float(np.mean(values))
    start = max(0.0, float(start))
    end = max(start, float(end))
    mask = (times >= start) & (times < end)
    if np.any(mask):
        return float(np.mean(values[mask]))
    center = 0.5 * (start + end)
    idx = int(np.argmin(np.abs(times - center)))
    return float(values[idx])


def _series_resolution(times: np.ndarray, duration_seconds: float) -> float:
    if times.size >= 2:
        diffs = np.diff(times)
        finite = diffs[np.isfinite(diffs) & (diffs > 0)]
        if finite.size:
            return float(np.median(finite))
    return max(float(duration_seconds), 1.0)


def _boundary_side_mean(values: np.ndarray, times: np.ndarray, boundary: float, window: float, before: bool) -> float:
    if values.size == 0:
        return 0.0
    if times.size != values.size:
        return float(np.mean(values))
    boundary = float(boundary)
    window = max(float(window), 1e-6)
    if before:
        mask = (times < boundary) & (times >= boundary - window)
        if np.any(mask):
            return float(np.mean(values[mask]))
        candidates = np.where(times < boundary)[0]
        if candidates.size:
            return float(values[int(candidates[-1])])
    else:
        mask = (times >= boundary) & (times < boundary + window)
        if np.any(mask):
            return float(np.mean(values[mask]))
        candidates = np.where(times >= boundary)[0]
        if candidates.size:
            return float(values[int(candidates[0])])
    idx = int(np.argmin(np.abs(times - boundary)))
    return float(values[idx])


def _boundary_edge_value(values: np.ndarray, times: np.ndarray, boundary: float, before: bool) -> float:
    if values.size == 0:
        return 0.0
    if times.size != values.size:
        return float(values[-1] if before else values[0])
    boundary = float(boundary)
    if before:
        candidates = np.where(times < boundary)[0]
        if candidates.size:
            return float(values[int(candidates[-1])])
    else:
        candidates = np.where(times >= boundary)[0]
        if candidates.size:
            return float(values[int(candidates[0])])
    idx = int(np.argmin(np.abs(times - boundary)))
    return float(values[idx])


def _normalized_delta(pre: float, post: float, floor: float = 1e-6) -> float:
    scale = max(abs(pre), abs(post), floor)
    return abs(post - pre) / scale


def _smooth_series(values: np.ndarray, window: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or window <= 1:
        return values
    window = min(int(window), int(values.size))
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def _median_beat_interval(song: SongDNA) -> float:
    beat_times = sorted(set(_safe_float_list(song.metadata.get("tempo", {}).get("beat_times", []))))
    if len(beat_times) < 2:
        return 0.5
    intervals = np.diff(np.asarray(beat_times, dtype=float))
    finite = intervals[np.isfinite(intervals) & (intervals > 0)]
    if finite.size == 0:
        return 0.5
    return float(np.median(finite))


def _seam_window_seconds(song: SongDNA) -> float:
    beat_interval = _median_beat_interval(song)
    # Prefer a short seam-local view (about two bars) rather than averaging whole sections.
    return float(np.clip(beat_interval * 8.0, 2.0, 8.0))


def _transition_intent(left: dict[str, Any], right: dict[str, Any]) -> str:
    explicit = str(right.get("transition_in") or left.get("transition_out") or "").strip().lower()
    if explicit in {"blend", "swap", "lift", "drop", "cut"}:
        return explicit

    left_label = str(left.get("label") or "").lower()
    right_label = str(right.get("label") or "").lower()
    joined = f"{left_label} {right_label}"
    if any(token in joined for token in {"payoff", "chorus", "drop", "hook", "climax"}):
        return "drop"
    if any(token in joined for token in {"build", "pre", "lift", "rise", "buildup"}):
        return "lift"
    if any(token in joined for token in {"swap", "verse", "outro", "break", "breakdown"}):
        return "swap"
    return "cut"


def _intent_mismatch(intent: str, signed_delta: float, tolerance: float) -> float:
    tolerance = max(float(tolerance), 1e-6)
    if intent in {"drop", "lift"}:
        if signed_delta >= 0.0:
            return 0.0
        return min(abs(signed_delta) / tolerance, 1.5)
    if intent == "blend":
        return max(0.0, (abs(signed_delta) - tolerance) / tolerance)
    if intent == "swap":
        return max(0.0, (abs(signed_delta) - tolerance * 1.15) / (tolerance * 1.15))
    return max(0.0, (abs(signed_delta) - tolerance * 0.85) / (tolerance * 0.85))


def _energy_series(song: SongDNA, stem: str) -> np.ndarray:
    energy = song.energy or {}
    preferred = _safe_array(energy.get(f"bar_{stem}"))
    if preferred.size:
        return preferred
    beat = _safe_array(energy.get(f"beat_{stem}"))
    if beat.size:
        return beat
    return _safe_array(energy.get(stem))


def _safe_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _safe_bar_boundaries(song: SongDNA, beats_per_bar: int = 4) -> list[float]:
    beat_times = sorted(set(_safe_float_list(song.metadata.get("tempo", {}).get("beat_times", []))))
    duration = float(song.duration_seconds)
    if not beat_times:
        return [0.0, duration] if duration > 0 else [0.0]

    bar_boundaries = [0.0]
    for idx in range(0, len(beat_times), beats_per_bar):
        bar_boundaries.append(float(beat_times[idx]))
    if bar_boundaries[-1] < duration:
        bar_boundaries.append(duration)
    return sorted(set(max(0.0, min(duration, x)) for x in bar_boundaries))


def _aggregate_bar_features(song: SongDNA, beats_per_bar: int = 4) -> tuple[np.ndarray | None, np.ndarray | None, list[float], list[str]]:
    path = Path(song.source_path)
    if not path.exists():
        return None, None, [], ["source audio is unavailable on disk, so coherence fell back to metadata-only scoring"]

    try:
        audio, sample_rate = librosa.load(path.as_posix(), sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - depends on local codec support
        return None, None, [], [f"source audio could not be reloaded for coherence analysis: {exc}"]

    if audio.size == 0:
        return None, None, [], ["source audio was empty during coherence analysis"]

    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=hop_length, n_mfcc=13)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sample_rate, hop_length=hop_length)
    bar_boundaries = _safe_bar_boundaries(song, beats_per_bar=beats_per_bar)

    if len(bar_boundaries) < 3:
        return None, None, bar_boundaries, ["too few bar boundaries were available for bar-synchronous coherence analysis"]

    features: list[np.ndarray] = []
    centers: list[float] = []
    for start, end in zip(bar_boundaries[:-1], bar_boundaries[1:]):
        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            continue
        bar_chroma = np.mean(chroma[:, mask], axis=1)
        bar_mfcc = np.mean(mfcc[:, mask], axis=1)
        bar_rms = np.mean(rms[:, mask], axis=1)
        vector = np.concatenate([bar_chroma, 0.5 * bar_mfcc, 2.0 * bar_rms], axis=0)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        features.append(vector.astype(float))
        centers.append(float((start + end) * 0.5))

    if len(features) < 4:
        return None, None, bar_boundaries, ["bar feature extraction produced too few populated bars for coherence analysis"]
    return np.vstack(features), np.asarray(centers, dtype=float), bar_boundaries, []


def _structure_score(song: SongDNA) -> ListenSubscore:
    sections = song.structure.get("sections", []) or []
    phrases = song.structure.get("phrase_boundaries_seconds", []) or []
    novelty = song.structure.get("novelty_boundaries_seconds", []) or []

    evidence: list[str] = []
    fixes: list[str] = []
    section_count = len(sections)
    phrase_count = len(phrases)
    novelty_count = len(novelty)

    section_score = _score_band(float(section_count), 6.0, 32.0, 8.0)
    phrase_score = _score_band(float(phrase_count), 8.0, 80.0, 12.0)
    novelty_score = _score_band(float(novelty_count), 4.0, 40.0, 8.0)

    duration_seconds = max(float(song.duration_seconds or 0.0), 1e-6)
    section_durations: list[float] = []
    covered_seconds = 0.0
    for section in sections:
        start = float(section.get("start", 0.0) or 0.0)
        end = float(section.get("end", start) or start)
        span = max(0.0, end - start)
        if span > 0.0:
            section_durations.append(span)
            covered_seconds += span

    if section_durations:
        median_section = float(np.median(section_durations))
        useful_ratio = float(np.mean(np.asarray(section_durations) >= 8.0))
        largest_ratio = max(section_durations) / duration_seconds
        coverage_ratio = min(covered_seconds / duration_seconds, 1.0)
        span_band_score = _score_band(median_section, 12.0, 48.0, 16.0)
        usefulness_score = _clamp01((useful_ratio - 0.45) / 0.45)
        coverage_score = _clamp01((coverage_ratio - 0.55) / 0.35)
        dominance_score = 1.0 - _clamp01((largest_ratio - 0.45) / 0.25)
        section_span_quality = (
            0.35 * span_band_score
            + 0.25 * usefulness_score
            + 0.25 * coverage_score
            + 0.15 * dominance_score
        )
    else:
        median_section = 0.0
        useful_ratio = 0.0
        largest_ratio = 1.0
        coverage_ratio = 0.0
        section_span_quality = 0.0

    score = round(
        100.0 * (
            0.34 * section_score
            + 0.28 * phrase_score
            + 0.16 * novelty_score
            + 0.22 * section_span_quality
        ),
        1,
    )

    evidence.append(f"detected {section_count} coarse sections, {phrase_count} phrase boundaries, {novelty_count} novelty boundaries")
    evidence.append(
        f"section span quality {section_span_quality:.3f}; median section {median_section:.1f}s; useful-span ratio {useful_ratio:.3f}; coverage {coverage_ratio:.3f}; largest-section ratio {largest_ratio:.3f}"
    )
    if section_count <= 2:
        fixes.append("Increase structural certainty so the planner is not forced into coarse whole-song windows.")
    if phrase_count < 6:
        fixes.append("Improve beat/downbeat and phrase extraction to create more musically legal planning windows.")
    if coverage_ratio < 0.75:
        fixes.append("Expand section coverage so more of the song is mapped into usable non-trivial structural spans.")
    if useful_ratio < 0.60 or median_section < 8.0:
        fixes.append("Reduce tiny section fragments; the current map is too chopped to support confident phrase-level planning.")
    if largest_ratio > 0.50:
        fixes.append("Avoid one dominant mega-section; rebalance the section map so major turns land across the full song.")
    summary = "Structure is reasonably segmented for planning." if score >= 70 else "Structure segmentation is still too coarse or span-imbalanced for strong planning."
    return ListenSubscore(
        score=score,
        summary=summary,
        evidence=evidence,
        fixes=fixes,
        details={
            "aggregate_metrics": {
                "section_count": section_count,
                "phrase_count": phrase_count,
                "novelty_count": novelty_count,
                "median_section_seconds": round(median_section, 3),
                "useful_span_ratio": round(useful_ratio, 3),
                "coverage_ratio": round(coverage_ratio, 3),
                "largest_section_ratio": round(largest_ratio, 3),
                "section_span_quality": round(section_span_quality, 3),
            }
        },
    )


def _groove_score(song: SongDNA) -> ListenSubscore:
    beat_times = [float(x) for x in song.metadata.get("tempo", {}).get("beat_times", [])]
    evidence: list[str] = []
    fixes: list[str] = []
    if len(beat_times) < 8:
        return ListenSubscore(
            score=20.0,
            summary="Groove confidence is weak because beat tracking is sparse.",
            evidence=[f"only {len(beat_times)} beat markers available"],
            fixes=["Improve beat/downbeat tracking before trusting groove-related planning or evaluation."],
        )

    intervals = np.diff(np.asarray(beat_times, dtype=float))
    median_interval = float(np.median(intervals)) if intervals.size else 0.0
    cv = float(np.std(intervals) / max(np.mean(intervals), 1e-6)) if intervals.size else 1.0
    beat_stability = _clamp01(1.0 - min(cv / 0.12, 1.0))
    tempo_consistency = _score_band(median_interval, 0.35, 0.75, 0.25)

    def _pocket_metrics(onset_values: np.ndarray, low_values: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        target_len = max(onset_values.size, low_values.size)
        if target_len < 2:
            return 0.55, 0.55, 0.0, np.asarray([], dtype=float)

        def _resample(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return np.full(target_len, 0.0, dtype=float)
            if values.size == target_len:
                return values.astype(float)
            if target_len <= 1 or values.size <= 1:
                return np.full(target_len, float(np.mean(values)), dtype=float)
            source_x = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
            target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
            return np.interp(target_x, source_x, values.astype(float))

        def _normalize(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return values
            low_v = float(np.min(values))
            high_v = float(np.max(values))
            span = max(high_v - low_v, 1e-6)
            return (values - low_v) / span

        onset = _resample(onset_values)
        low = _resample(low_values)
        groove_series = 0.65 * _normalize(onset) + 0.35 * _normalize(low)
        if groove_series.size < 2:
            return 0.55, 0.55, 0.0, groove_series

        smooth_window = min(max(3, target_len // 12), 8)
        smoothed = _smooth_series(groove_series, window=smooth_window)
        groove_steps = np.diff(smoothed) if smoothed.size >= 2 else np.asarray([], dtype=float)
        if groove_steps.size == 0:
            return 0.55, 0.55, 0.0, groove_series

        abs_steps = np.abs(groove_steps)
        down_steps = np.maximum(-groove_steps, 0.0)
        collapse_floor = max(0.12, 0.35 * float(np.std(smoothed)))
        collapse_events = down_steps[down_steps >= collapse_floor]
        largest_drop = float(np.max(collapse_events)) if collapse_events.size else 0.0
        sustained_drop = float(np.mean(collapse_events)) if collapse_events.size else 0.0
        collapse_density = float(collapse_events.size / max(down_steps.size, 1))

        stability = _clamp01(
            1.0 - (0.80 * float(np.median(abs_steps)) + 1.35 * sustained_drop + 0.55 * collapse_density) / 0.22
        )
        consistency = _clamp01(
            1.0 - (0.75 * float(np.percentile(abs_steps, 90)) + 0.95 * largest_drop + 0.45 * collapse_density) / 0.47
        )
        collapse = _clamp01((0.72 * largest_drop + 0.28 * sustained_drop) / 0.42)
        return stability, consistency, collapse, groove_series

    energy = song.energy or {}
    bar_onset = _safe_array(energy.get("bar_onset_density") or energy.get("onset_density"))
    bar_low = _safe_array(energy.get("bar_low_band_ratio") or energy.get("low_band_ratio"))
    beat_onset = _safe_array(energy.get("beat_onset_density"))
    beat_low = _safe_array(energy.get("beat_low_band_ratio"))

    bar_pocket_stability = 0.55
    bar_pocket_consistency = 0.55
    bar_collapse_severity = 0.0
    bar_groove_series = np.asarray([], dtype=float)
    if bar_onset.size >= 8 or bar_low.size >= 8:
        bar_pocket_stability, bar_pocket_consistency, bar_collapse_severity, bar_groove_series = _pocket_metrics(bar_onset, bar_low)

    beat_pulse_stability = bar_pocket_stability
    beat_pulse_consistency = bar_pocket_consistency
    beat_collapse_severity = bar_collapse_severity
    beat_groove_series = np.asarray([], dtype=float)
    if beat_onset.size >= 16 or beat_low.size >= 16:
        beat_pulse_stability, beat_pulse_consistency, beat_collapse_severity, beat_groove_series = _pocket_metrics(beat_onset, beat_low)

    if beat_groove_series.size:
        pocket_stability = 0.45 * bar_pocket_stability + 0.55 * beat_pulse_stability
        pocket_consistency = 0.45 * bar_pocket_consistency + 0.55 * beat_pulse_consistency
        collapse_severity = max(bar_collapse_severity, 0.65 * beat_collapse_severity + 0.35 * bar_collapse_severity)
        groove_series = beat_groove_series
    else:
        pocket_stability = bar_pocket_stability
        pocket_consistency = bar_pocket_consistency
        collapse_severity = bar_collapse_severity
        groove_series = bar_groove_series

    collapse_penalty = 0.20 * collapse_severity + 0.16 * _clamp01((collapse_severity - 0.45) / 0.35)
    score = round(
        100.0
        * (
            0.50 * beat_stability
            + 0.20 * tempo_consistency
            + 0.18 * pocket_stability
            + 0.12 * pocket_consistency
            - collapse_penalty
        ),
        1,
    )

    evidence.append(
        f"{len(beat_times)} beats detected; median beat interval {median_interval:.3f}s; interval variation {cv:.3f}"
    )
    if bar_groove_series.size:
        evidence.append(
            "bar-pocket stability %.3f, consistency %.3f, max collapse %.3f"
            % (bar_pocket_stability, bar_pocket_consistency, bar_collapse_severity)
        )
    if beat_groove_series.size:
        evidence.append(
            "beat-pocket stability %.3f, consistency %.3f, max collapse %.3f"
            % (beat_pulse_stability, beat_pulse_consistency, beat_collapse_severity)
        )
    if cv > 0.10:
        fixes.append("Stabilize beat/downbeat tracking or use a stronger bar-grid estimator for planning and evaluation.")
    if collapse_severity > 0.45:
        fixes.append("Tighten bar-to-bar rhythmic continuity; the beat grid exists but the pocket collapses abruptly across adjacent windows.")
    elif pocket_stability < 0.45:
        fixes.append("Reduce abrupt onset/low-end churn between adjacent bars and beats so the groove pocket stays locked.")
    summary = "Groove grid looks reasonably stable." if score >= 70 else "Groove grid looks unstable, collapsed, or under-detected."
    return ListenSubscore(
        score=score,
        summary=summary,
        evidence=evidence,
        fixes=fixes,
        details={
            "beat_stability": round(beat_stability, 4),
            "tempo_consistency": round(tempo_consistency, 4),
            "pocket_stability": round(pocket_stability, 4),
            "pocket_consistency": round(pocket_consistency, 4),
            "collapse_severity": round(collapse_severity, 4),
            "bar_pocket_stability": round(bar_pocket_stability, 4),
            "bar_pocket_consistency": round(bar_pocket_consistency, 4),
            "bar_collapse_severity": round(bar_collapse_severity, 4),
            "beat_pulse_stability": round(beat_pulse_stability, 4),
            "beat_pulse_consistency": round(beat_pulse_consistency, 4),
            "beat_collapse_severity": round(beat_collapse_severity, 4),
        },
    )


def _energy_arc_score(song: SongDNA) -> ListenSubscore:
    energy = song.energy or {}
    bar_rms = _energy_series(song, "rms")
    bar_onset = _energy_series(song, "onset_density")
    bar_low = _energy_series(song, "low_band_ratio")
    bar_flat = _energy_series(song, "spectral_flatness")
    derived = energy.get("derived") or {}
    evidence: list[str] = []
    fixes: list[str] = []

    if bar_rms.size < 8:
        return ListenSubscore(
            score=35.0,
            summary="Energy contour is too sparse to judge confidently.",
            evidence=[f"only {bar_rms.size} energy windows available"],
            fixes=["Emit denser bar- or phrase-level energy summaries for stronger arc evaluation."],
        )

    def _norm(arr: np.ndarray, invert: bool = False) -> np.ndarray:
        if arr.size == 0:
            return np.zeros(bar_rms.size, dtype=float)
        low = float(np.min(arr))
        high = float(np.max(arr))
        span = max(high - low, 1e-6)
        normed = (arr - low) / span
        if invert:
            normed = 1.0 - normed
        return normed

    composite = (
        0.45 * _norm(bar_rms)
        + 0.25 * _norm(bar_onset)
        + 0.20 * _norm(bar_low)
        + 0.10 * _norm(bar_flat, invert=True)
    )
    smoothed = _smooth_series(composite, window=3)
    quartiles = np.array_split(smoothed, 4)
    quartile_means = [float(np.mean(chunk)) for chunk in quartiles if chunk.size]
    early_mean = quartile_means[0] if quartile_means else 0.0
    mid_mean = float(np.mean(quartile_means[1:3])) if len(quartile_means) >= 3 else early_mean
    late_mean = quartile_means[-1] if quartile_means else early_mean

    contrast = float(np.percentile(smoothed, 90) - np.percentile(smoothed, 10)) if smoothed.size else 0.0
    late_lift = late_mean - early_mean
    peak_idx = int(np.argmax(smoothed)) if smoothed.size else 0
    peak_position = peak_idx / max(len(smoothed) - 1, 1)
    step_deltas = np.abs(np.diff(smoothed)) if smoothed.size >= 2 else np.asarray([], dtype=float)
    step_median = float(np.median(step_deltas)) if step_deltas.size else 0.0
    step_p90 = float(np.percentile(step_deltas, 90)) if step_deltas.size else 0.0

    payoff_strength = float(derived.get("payoff_strength") or 0.0)
    hook_strength = float(derived.get("hook_strength") or 0.0)
    hook_repetition = float(derived.get("hook_repetition") or 0.0)
    hook_spend = float(derived.get("hook_spend") or 0.0)
    early_hook_strength = float(derived.get("early_hook_strength") or 0.0)
    late_hook_strength = float(derived.get("late_hook_strength") or 0.0)
    late_payoff_strength = float(derived.get("late_payoff_strength") or 0.0)
    energy_confidence = float(derived.get("energy_confidence") or 0.0)

    contrast_score = _clamp01((contrast - 0.10) / 0.35)
    late_lift_score = _clamp01((late_lift - 0.04) / 0.28)
    peak_late_bonus = _score_band(peak_position, 0.45, 0.95, 0.25)
    trajectory_score = 0.65 * late_lift_score + 0.35 * peak_late_bonus
    payoff_score = _clamp01(0.50 * payoff_strength + 0.30 * hook_strength + 0.20 * hook_repetition)
    anti_spend_score = 1.0 - hook_spend
    stability_score = (
        0.65 * _score_band(step_median, 0.015, 0.14, 0.08)
        + 0.35 * _score_band(step_p90, 0.04, 0.32, 0.12)
    )

    raw_score = (
        0.26 * contrast_score
        + 0.24 * trajectory_score
        + 0.24 * payoff_score
        + 0.11 * stability_score
        + 0.15 * anti_spend_score
    )
    confidence_gate = 0.70 + 0.30 * _clamp01(energy_confidence)
    final_norm = _clamp01(raw_score) * confidence_gate
    score = round(100.0 * final_norm, 1)

    evidence.append(
        f"bar-energy contrast {contrast:.3f}; quartile means early={early_mean:.3f}, mid={mid_mean:.3f}, late={late_mean:.3f}; late lift {late_lift:.3f}"
    )
    evidence.append(
        f"peak energy occurs at {peak_position * 100.0:.0f}% of song duration; payoff {payoff_strength:.3f}, late payoff {late_payoff_strength:.3f}, hook {hook_strength:.3f}, early hook {early_hook_strength:.3f}, late hook {late_hook_strength:.3f}, hook repetition {hook_repetition:.3f}, hook spend {hook_spend:.3f}, confidence {energy_confidence:.3f}"
    )
    evidence.append(f"bar-energy step stability median {step_median:.3f}, p90 {step_p90:.3f}")

    if contrast < 0.12:
        fixes.append("Increase macro-dynamic contrast so payoff sections separate clearly from setup sections.")
    if late_lift < 0.04 or peak_position < 0.35:
        fixes.append("Rework the energy journey so the strongest material arrives later instead of peaking too early.")
    if payoff_score < 0.35:
        fixes.append("Strengthen recurring payoff windows; current energy rises without a convincing chorus/drop identity.")
    if hook_spend > 0.35:
        fixes.append("The fusion appears to spend its hook too early; hold back the strongest repeated material until later payoff sections or relaunch it with a stronger late payoff.")
    if step_p90 > 0.38:
        fixes.append("Concentrate bigger energy moves at phrase/section turns instead of constant bar-to-bar churn.")
    if energy_confidence < 0.35:
        fixes.append("Energy extraction confidence is weak; improve bar-level analysis so arc judgments are more trustworthy.")

    if score >= 75:
        summary = "Energy arc shows clear build, identifiable payoff windows, and useful macro contrast."
    elif score >= 60:
        summary = "Energy arc is usable but still underpowered, early-peaking, or not sustained enough."
    else:
        summary = "Energy arc is weak, flat, front-loaded, or poorly shaped."

    details = {
        "aggregate_metrics": {
            "contrast": round(contrast, 3),
            "early_mean": round(early_mean, 3),
            "mid_mean": round(mid_mean, 3),
            "late_mean": round(late_mean, 3),
            "late_lift": round(late_lift, 3),
            "peak_position": round(peak_position, 3),
            "median_step_delta": round(step_median, 3),
            "p90_step_delta": round(step_p90, 3),
            "payoff_strength": round(payoff_strength, 3),
            "hook_strength": round(hook_strength, 3),
            "hook_repetition": round(hook_repetition, 3),
            "hook_spend": round(hook_spend, 3),
            "early_hook_strength": round(early_hook_strength, 3),
            "late_hook_strength": round(late_hook_strength, 3),
            "late_payoff_strength": round(late_payoff_strength, 3),
            "energy_confidence": round(energy_confidence, 3),
        },
        "top_payoff_windows": (derived.get("payoff_windows") or [])[:3],
        "top_hook_windows": (derived.get("hook_windows") or [])[:3],
    }
    return ListenSubscore(score=score, summary=summary, evidence=evidence, fixes=fixes, details=details)


def _ownership_clutter_metrics(song: SongDNA) -> dict[str, float]:
    energy = song.energy or {}
    rms = _safe_array(energy.get("rms"))
    centroid = _safe_array(energy.get("spectral_centroid"))
    onset = _safe_array(energy.get("onset_density") or energy.get("onset_strength"))
    low_band = _safe_array(energy.get("low_band_ratio") or energy.get("bass_ratio") or energy.get("low_band_energy"))
    flatness = _safe_array(energy.get("spectral_flatness"))

    mean_rms = float(np.mean(rms)) if rms.size else 0.0
    rms_var = float(np.std(rms)) if rms.size else 0.0
    mean_centroid = float(np.mean(centroid)) if centroid.size else 0.0
    centroid_var = float(np.std(centroid)) if centroid.size else 0.0
    mean_onset = float(np.mean(onset)) if onset.size else 0.0
    onset_var = float(np.std(onset)) if onset.size else 0.0
    mean_low = float(np.mean(low_band)) if low_band.size else 0.0
    low_var = float(np.std(low_band)) if low_band.size else 0.0
    mean_flatness = float(np.mean(flatness)) if flatness.size else 0.0
    flatness_var = float(np.std(flatness)) if flatness.size else 0.0

    low_end_conflict = _clamp01(
        0.55 * _clamp01((mean_low - 0.42) / 0.18)
        + 0.25 * _clamp01((mean_rms - 0.11) / 0.10)
        + 0.20 * _clamp01((0.05 - low_var) / 0.05)
    )
    foreground_overload = _clamp01(
        0.35 * _clamp01((mean_rms - 0.11) / 0.10)
        + 0.30 * _clamp01((mean_centroid - 2600.0) / 1800.0)
        + 0.20 * _clamp01((mean_onset - 0.35) / 0.25)
        + 0.15 * _clamp01((0.06 - rms_var) / 0.06)
    )
    overcompressed_flatness = _clamp01(
        0.40 * _clamp01((mean_flatness - 0.22) / 0.18)
        + 0.35 * _clamp01((0.035 - rms_var) / 0.035)
        + 0.15 * _clamp01((0.18 - centroid_var) / 1200.0)
        + 0.10 * _clamp01((mean_rms - 0.14) / 0.08)
    )
    vocal_competition = _clamp01(
        0.35 * _clamp01((mean_centroid - 1800.0) / 1800.0)
        + 0.25 * _clamp01((mean_onset - 0.28) / 0.22)
        + 0.20 * _clamp01((0.05 - flatness_var) / 0.05)
        + 0.20 * _clamp01((0.07 - centroid_var / max(mean_centroid, 1.0)) / 0.07)
    )
    overcrowded_overlap = _clamp01(
        0.35 * low_end_conflict
        + 0.30 * foreground_overload
        + 0.20 * vocal_competition
        + 0.15 * _clamp01((mean_rms - 0.12) / 0.10)
    )

    crowding_series_size = min(
        size for size in (rms.size, centroid.size, onset.size, low_band.size, flatness.size) if size > 0
    ) if all(arr.size > 0 for arr in (rms, centroid, onset, low_band, flatness)) else 0
    crowding_peak_density = 0.0
    crowding_sustained_ratio = 0.0
    crowding_burst_count = 0
    crowding_burst_risk = 0.0
    if crowding_series_size >= 4:
        rms_s = rms[:crowding_series_size]
        centroid_s = centroid[:crowding_series_size]
        onset_s = onset[:crowding_series_size]
        low_band_s = low_band[:crowding_series_size]
        flatness_s = flatness[:crowding_series_size]
        frame_risk = np.clip(
            0.30 * np.clip((rms_s - 0.13) / 0.08, 0.0, 1.0)
            + 0.22 * np.clip((low_band_s - 0.44) / 0.16, 0.0, 1.0)
            + 0.20 * np.clip((onset_s - 0.34) / 0.22, 0.0, 1.0)
            + 0.16 * np.clip((centroid_s - 2400.0) / 1800.0, 0.0, 1.0)
            + 0.12 * np.clip((flatness_s - 0.22) / 0.16, 0.0, 1.0),
            0.0,
            1.0,
        )
        crowding_peak_density = float(np.max(frame_risk))
        crowded_frames = frame_risk >= 0.58
        crowding_sustained_ratio = float(np.mean(crowded_frames)) if crowded_frames.size else 0.0
        transitions = np.diff(np.concatenate(([0], crowded_frames.astype(int), [0])))
        burst_starts = np.where(transitions == 1)[0]
        burst_ends = np.where(transitions == -1)[0]
        burst_lengths = burst_ends - burst_starts
        crowding_burst_count = int(np.sum(burst_lengths >= 2))
        longest_burst = float(np.max(burst_lengths)) / max(crowding_series_size, 1) if burst_lengths.size else 0.0
        crowding_burst_risk = _clamp01(
            0.45 * crowding_peak_density
            + 0.35 * crowding_sustained_ratio
            + 0.20 * _clamp01(longest_burst / 0.35)
        )

    return {
        "mean_rms": round(mean_rms, 4),
        "rms_variation": round(rms_var, 4),
        "mean_centroid_hz": round(mean_centroid, 1),
        "centroid_variation_hz": round(centroid_var, 1),
        "mean_onset_density": round(mean_onset, 4),
        "onset_variation": round(onset_var, 4),
        "mean_low_band_ratio": round(mean_low, 4),
        "low_band_variation": round(low_var, 4),
        "mean_spectral_flatness": round(mean_flatness, 4),
        "flatness_variation": round(flatness_var, 4),
        "low_end_conflict_risk": round(low_end_conflict, 3),
        "foreground_overload_risk": round(foreground_overload, 3),
        "overcrowded_overlap_risk": round(overcrowded_overlap, 3),
        "overcompressed_flatness_risk": round(overcompressed_flatness, 3),
        "vocal_competition_risk": round(vocal_competition, 3),
        "crowding_peak_density": round(crowding_peak_density, 3),
        "crowding_sustained_ratio": round(crowding_sustained_ratio, 3),
        "crowding_burst_count": crowding_burst_count,
        "crowding_burst_risk": round(crowding_burst_risk, 3),
    }


def _boundary_worst_moment_clips(
    rows: list[dict[str, float | int | str]],
    *,
    duration_seconds: float,
    max_items: int = 5,
) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    track_duration = max(float(duration_seconds or 0.0), 0.0)
    for row in sorted(rows, key=lambda item: float(item.get("severity", 0.0)), reverse=True)[: max_items]:
        boundary_time = float(row.get("boundary_time", 0.0) or 0.0)
        seam_window = max(float(row.get("seam_window_seconds", 0.0) or 0.0), 2.0)
        clip_half = max(1.5, min(seam_window * 0.5, 4.0))
        start_time = max(0.0, boundary_time - clip_half)
        end_time = boundary_time + clip_half
        if track_duration > 0.0:
            end_time = min(track_duration, end_time)
            start_time = max(0.0, min(start_time, end_time))
        hot_axes = [
            axis
            for axis, value in [
                ("energy", row.get("energy_jump", 0.0)),
                ("spectral", row.get("spectral_jump", 0.0)),
                ("onset", row.get("onset_jump", 0.0)),
                ("edge_cliff", row.get("edge_cliff_risk", 0.0)),
                ("low_end", row.get("low_end_crowding_risk", 0.0)),
                ("foreground", row.get("foreground_collision_risk", 0.0)),
                ("vocals", row.get("vocal_competition_risk", 0.0)),
                ("texture", row.get("texture_shift", 0.0)),
                ("intent", row.get("intent_mismatch", 0.0)),
            ]
            if float(value or 0.0) >= 0.4
        ]
        summary = f"{row.get('intent', 'cut')} seam near {boundary_time:.1f}s"
        if hot_axes:
            summary += f" is exposed on {', '.join(hot_axes[:3])}"
        clips.append(
            {
                "kind": "boundary_transition",
                "component": "transition",
                "boundary_index": int(row.get("boundary_index", len(clips))),
                "intent": str(row.get("intent", "cut")),
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "center_time": round(boundary_time, 3),
                "duration_seconds": round(max(0.0, end_time - start_time), 3),
                "severity": round(float(row.get("severity", 0.0) or 0.0), 3),
                "reason_codes": hot_axes,
                "summary": summary,
                "evidence": {
                    "energy_jump": row.get("energy_jump"),
                    "spectral_jump": row.get("spectral_jump"),
                    "onset_jump": row.get("onset_jump"),
                    "edge_cliff_risk": row.get("edge_cliff_risk"),
                    "low_end_crowding_risk": row.get("low_end_crowding_risk"),
                    "foreground_collision_risk": row.get("foreground_collision_risk"),
                    "vocal_competition_risk": row.get("vocal_competition_risk"),
                    "intent_mismatch": row.get("intent_mismatch"),
                },
            }
        )
    return clips


def _transition_score(song: SongDNA) -> ListenSubscore:
    sections = song.structure.get("sections", []) or []
    evidence: list[str] = []
    fixes: list[str] = []
    if len(sections) < 2:
        return ListenSubscore(
            score=30.0,
            summary="Transition scoring is weak because there are not enough detected section boundaries.",
            evidence=[f"only {len(sections)} section(s) detected"],
            fixes=["Improve section/phrase boundary detection so transition seams can be evaluated and planned better."],
        )

    duration = float(song.duration_seconds or 0.0)
    energy = song.energy or {}
    rms = _safe_array(energy.get("rms"))
    centroid = _safe_array(energy.get("spectral_centroid"))
    rolloff = _safe_array(energy.get("spectral_rolloff"))
    onset = _safe_array(energy.get("onset_density") or energy.get("onset_strength"))
    low_band = _safe_array(energy.get("low_band_ratio") or energy.get("bass_ratio") or energy.get("low_band_energy"))
    flatness = _safe_array(energy.get("spectral_flatness"))

    rms_t = _series_times(rms, duration)
    centroid_t = _series_times(centroid, duration)
    rolloff_t = _series_times(rolloff, duration)
    onset_t = _series_times(onset, duration)
    low_band_t = _series_times(low_band, duration)
    flatness_t = _series_times(flatness, duration)
    seam_window = _seam_window_seconds(song)
    seam_window = max(
        seam_window,
        0.75 * min(
            value
            for value in [
                _series_resolution(rms_t, duration),
                _series_resolution(centroid_t, duration),
                _series_resolution(rolloff_t, duration),
                _series_resolution(onset_t, duration),
                _series_resolution(low_band_t, duration),
                _series_resolution(flatness_t, duration),
            ]
            if value > 0
        ),
    )

    severity_rows: list[dict[str, float | int | str]] = []
    transition_types: list[str] = []
    for idx in range(len(sections) - 1):
        left = sections[idx]
        right = sections[idx + 1]
        boundary = float(left.get("end", 0.0))
        left_start = max(float(left.get("start", max(0.0, boundary - seam_window))), boundary - seam_window)
        right_end = min(float(right.get("end", boundary + seam_window)), boundary + seam_window)
        intent = _transition_intent(left, right)
        local_window = max(boundary - left_start, right_end - boundary)

        pre_energy = _boundary_side_mean(rms, rms_t, boundary, local_window, before=True)
        post_energy = _boundary_side_mean(rms, rms_t, boundary, local_window, before=False)
        pre_centroid = _boundary_side_mean(centroid, centroid_t, boundary, local_window, before=True)
        post_centroid = _boundary_side_mean(centroid, centroid_t, boundary, local_window, before=False)
        pre_rolloff = _boundary_side_mean(rolloff, rolloff_t, boundary, local_window, before=True)
        post_rolloff = _boundary_side_mean(rolloff, rolloff_t, boundary, local_window, before=False)
        pre_onset = _boundary_side_mean(onset, onset_t, boundary, local_window, before=True)
        post_onset = _boundary_side_mean(onset, onset_t, boundary, local_window, before=False)
        pre_low = _boundary_side_mean(low_band, low_band_t, boundary, local_window, before=True)
        post_low = _boundary_side_mean(low_band, low_band_t, boundary, local_window, before=False)
        pre_flat = _boundary_side_mean(flatness, flatness_t, boundary, local_window, before=True)
        post_flat = _boundary_side_mean(flatness, flatness_t, boundary, local_window, before=False)
        edge_pre_energy = _boundary_edge_value(rms, rms_t, boundary, before=True)
        edge_post_energy = _boundary_edge_value(rms, rms_t, boundary, before=False)
        edge_pre_onset = _boundary_edge_value(onset, onset_t, boundary, before=True)
        edge_post_onset = _boundary_edge_value(onset, onset_t, boundary, before=False)
        edge_pre_centroid = _boundary_edge_value(centroid, centroid_t, boundary, before=True)
        edge_post_centroid = _boundary_edge_value(centroid, centroid_t, boundary, before=False)
        edge_pre_rolloff = _boundary_edge_value(rolloff, rolloff_t, boundary, before=True)
        edge_post_rolloff = _boundary_edge_value(rolloff, rolloff_t, boundary, before=False)

        energy_signed = (post_energy - pre_energy) / max(max(abs(pre_energy), abs(post_energy)), 0.01)
        onset_signed = (post_onset - pre_onset) / max(max(abs(pre_onset), abs(post_onset)), 0.05)
        energy_jump = _normalized_delta(pre_energy, post_energy, floor=0.01)
        spectral_jump = max(
            _normalized_delta(pre_centroid, post_centroid, floor=100.0),
            _normalized_delta(pre_rolloff, post_rolloff, floor=200.0),
        )
        onset_jump = _normalized_delta(pre_onset, post_onset, floor=0.05)
        edge_energy_jump = _normalized_delta(edge_pre_energy, edge_post_energy, floor=0.01)
        edge_onset_jump = _normalized_delta(edge_pre_onset, edge_post_onset, floor=0.05)
        edge_spectral_jump = max(
            _normalized_delta(edge_pre_centroid, edge_post_centroid, floor=100.0),
            _normalized_delta(edge_pre_rolloff, edge_post_rolloff, floor=200.0),
        )
        edge_cliff_risk = _clamp01(max(edge_energy_jump, 0.8 * edge_onset_jump, 0.7 * edge_spectral_jump))
        low_end_crowding_risk = 0.0
        if low_band.size:
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
        flatness_crowding_risk = _clamp01(
            0.55 * min(max(pre_flat, 0.0), max(post_flat, 0.0)) / max(max(pre_flat, post_flat, 0.0), 1e-6)
            + 0.45 * min(max(pre_energy, 0.0), max(post_energy, 0.0)) / max(max(pre_energy, post_energy, 0.0), 1e-6)
        )
        vocal_competition_risk = _clamp01(
            0.50 * min(max(pre_centroid, 0.0), max(post_centroid, 0.0)) / max(max(pre_centroid, post_centroid, 0.0), 100.0)
            + 0.30 * min(max(pre_onset, 0.0), max(post_onset, 0.0)) / max(max(pre_onset, post_onset, 0.0), 0.05)
            + 0.20 * (1.0 - min(1.0, abs(pre_centroid - post_centroid) / max(max(pre_centroid, post_centroid), 100.0)))
        )

        intent_mismatch = max(
            _intent_mismatch(intent, energy_signed, tolerance=0.28),
            0.7 * _intent_mismatch(intent, onset_signed, tolerance=0.34),
        )
        intent_energy_weight = 0.10 if intent in {"drop", "lift"} and energy_signed >= 0.0 else 0.22
        intent_onset_weight = 0.08 if intent in {"drop", "lift"} and onset_signed >= 0.0 else 0.15
        severity = (
            intent_energy_weight * min(energy_jump, 1.5)
            + 0.18 * min(spectral_jump, 1.5)
            + intent_onset_weight * min(onset_jump, 1.5)
            + 0.12 * min(edge_cliff_risk, 1.5)
            + 0.14 * min(low_end_crowding_risk, 1.5)
            + 0.11 * min(texture_shift, 1.5)
            + 0.10 * min(foreground_collision_risk, 1.5)
            + 0.05 * min(flatness_crowding_risk, 1.5)
            + 0.05 * min(vocal_competition_risk, 1.5)
            + 0.09 * min(intent_mismatch, 1.5)
        )
        severity_rows.append(
            {
                "boundary_index": idx,
                "boundary_time": round(boundary, 3),
                "intent": intent,
                "seam_window_seconds": round(seam_window, 3),
                "energy_jump": round(energy_jump, 3),
                "energy_signed_delta": round(float(energy_signed), 3),
                "spectral_jump": round(spectral_jump, 3),
                "onset_jump": round(onset_jump, 3),
                "onset_signed_delta": round(float(onset_signed), 3),
                "edge_energy_jump": round(edge_energy_jump, 3),
                "edge_onset_jump": round(edge_onset_jump, 3),
                "edge_spectral_jump": round(edge_spectral_jump, 3),
                "edge_cliff_risk": round(edge_cliff_risk, 3),
                "low_end_crowding_risk": round(low_end_crowding_risk, 3),
                "texture_shift": round(texture_shift, 3),
                "foreground_collision_risk": round(foreground_collision_risk, 3),
                "flatness_crowding_risk": round(flatness_crowding_risk, 3),
                "vocal_competition_risk": round(vocal_competition_risk, 3),
                "intent_mismatch": round(float(intent_mismatch), 3),
                "severity": round(float(severity), 3),
            }
        )

        hot_axes = [
            name
            for name, value in [
                ("energy", energy_jump),
                ("spectral", spectral_jump),
                ("onset", onset_jump),
                ("edge_cliff", edge_cliff_risk),
                ("low_end", low_end_crowding_risk),
                ("foreground", foreground_collision_risk),
                ("flatness", flatness_crowding_risk),
                ("vocals", vocal_competition_risk),
                ("texture", texture_shift),
                ("intent", intent_mismatch),
            ]
            if value >= 0.4
        ]
        if hot_axes:
            transition_types.append(
                f"boundary {idx} @ {boundary:.1f}s ({intent}) is exposed on {', '.join(hot_axes[:4])}"
            )

    severities = np.asarray([float(row["severity"]) for row in severity_rows], dtype=float)
    avg_severity = float(np.mean(severities)) if severities.size else 1.0
    worst = sorted(severity_rows, key=lambda row: float(row["severity"]), reverse=True)[:3]

    coverage_bonus = _score_band(float(len(severity_rows)), 4.0, 24.0, 8.0)
    seam_score = _clamp01(1.0 - min(avg_severity / 0.85, 1.0))
    score = round(100.0 * (0.82 * seam_score + 0.18 * coverage_bonus), 1)

    aggregate_metrics = {
        "boundary_count": len(severity_rows),
        "seam_window_seconds": round(seam_window, 3),
        "avg_energy_jump": round(float(np.mean([row["energy_jump"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_energy_signed_delta": round(float(np.mean([row["energy_signed_delta"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_spectral_jump": round(float(np.mean([row["spectral_jump"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_onset_jump": round(float(np.mean([row["onset_jump"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_onset_signed_delta": round(float(np.mean([row["onset_signed_delta"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_edge_energy_jump": round(float(np.mean([row["edge_energy_jump"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_edge_onset_jump": round(float(np.mean([row["edge_onset_jump"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_edge_spectral_jump": round(float(np.mean([row["edge_spectral_jump"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_edge_cliff_risk": round(float(np.mean([row["edge_cliff_risk"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_low_end_crowding_risk": round(float(np.mean([row["low_end_crowding_risk"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_texture_shift": round(float(np.mean([row["texture_shift"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_foreground_collision_risk": round(float(np.mean([row["foreground_collision_risk"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_flatness_crowding_risk": round(float(np.mean([row["flatness_crowding_risk"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_vocal_competition_risk": round(float(np.mean([row["vocal_competition_risk"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_intent_mismatch": round(float(np.mean([row["intent_mismatch"] for row in severity_rows])) if severity_rows else 0.0, 3),
        "avg_boundary_severity": round(avg_severity, 3),
    }

    manifest_payload = _load_neighbor_manifest(song)
    manifest_details = _manifest_overlap_metrics(manifest_payload) if manifest_payload else None
    if manifest_details:
        manifest_metrics = manifest_details['aggregate_metrics']
        primary_sequence = _manifest_primary_sequence(manifest_payload)
        owner_switches = sum(1 for left, right in zip(primary_sequence, primary_sequence[1:]) if left != right)
        owner_switch_ratio = owner_switches / max(len(primary_sequence) - 1, 1) if len(primary_sequence) >= 2 else 0.0
        alternating_triplets = sum(1 for left, mid, right in zip(primary_sequence, primary_sequence[1:], primary_sequence[2:]) if left == right and left != mid)
        alternating_triplet_ratio = alternating_triplets / max(len(primary_sequence) - 2, 1) if len(primary_sequence) >= 3 else 0.0
        swap_sections = sum(
            1
            for section in (manifest_payload or {}).get('sections') or []
            if (section.get('transition_in') in {'swap'} or section.get('transition_out') in {'swap'})
        )
        swap_density = swap_sections / max(len((manifest_payload or {}).get('sections') or []), 1)
        switch_detector_risk = _clamp01(
            0.55 * _clamp01((owner_switch_ratio - 0.34) / 0.46)
            + 0.30 * _clamp01((alternating_triplet_ratio - 0.12) / 0.38)
            + 0.15 * _clamp01((swap_density - 0.25) / 0.45)
        )

        aggregate_metrics['manifest_overlap_section_ratio'] = manifest_metrics['overlap_section_ratio']
        aggregate_metrics['manifest_avg_overlap_beats'] = manifest_metrics['avg_overlap_beats']
        aggregate_metrics['manifest_multi_owner_conflict_ratio'] = manifest_metrics['multi_owner_conflict_ratio']
        aggregate_metrics['manifest_lead_conflict_ratio'] = manifest_metrics['lead_conflict_ratio']
        aggregate_metrics['manifest_crowding_ratio'] = manifest_metrics['crowding_ratio']
        aggregate_metrics['manifest_seam_risk_ratio'] = manifest_metrics['seam_risk_ratio']
        aggregate_metrics['manifest_collapse_ratio'] = manifest_metrics['collapse_ratio']
        aggregate_metrics['manifest_owner_switch_ratio'] = round(owner_switch_ratio, 3)
        aggregate_metrics['manifest_owner_switch_count'] = owner_switches
        aggregate_metrics['manifest_low_end_owner_switch_ratio'] = manifest_metrics['low_end_owner_switch_ratio']
        aggregate_metrics['manifest_low_end_overlap_switch_ratio'] = manifest_metrics['low_end_overlap_switch_ratio']
        aggregate_metrics['manifest_low_end_owner_stability_risk'] = manifest_metrics['low_end_owner_stability_risk']
        aggregate_metrics['manifest_alternating_triplet_ratio'] = round(alternating_triplet_ratio, 3)
        aggregate_metrics['manifest_swap_density'] = round(swap_density, 3)
        aggregate_metrics['manifest_switch_detector_risk'] = round(switch_detector_risk, 3)
        manifest_penalty = (
            0.24 * manifest_metrics['multi_owner_conflict_ratio']
            + 0.16 * manifest_metrics['lead_conflict_ratio']
            + 0.14 * manifest_metrics['crowding_ratio']
            + 0.16 * manifest_metrics['seam_risk_ratio']
            + 0.10 * manifest_metrics['avg_overlap_beats'] / 8.0
            + 0.06 * manifest_metrics['long_overlap_ratio']
            + 0.08 * manifest_metrics['low_end_owner_stability_risk']
            + 0.14 * switch_detector_risk
        )
        score = round(max(0.0, score - 22.0 * manifest_penalty), 1)
        evidence.append(
            "manifest seam risks — "
            f"overlap ratio {manifest_metrics['overlap_section_ratio']:.2f}, "
            f"avg overlap beats {manifest_metrics['avg_overlap_beats']:.2f}, "
            f"owner conflicts {manifest_metrics['multi_owner_conflict_ratio']:.2f}, "
            f"lead conflicts {manifest_metrics['lead_conflict_ratio']:.2f}, "
            f"crowding {manifest_metrics['crowding_ratio']:.2f}, "
            f"low-end owner stability risk {manifest_metrics['low_end_owner_stability_risk']:.2f}, "
            f"stretch/seam risk {manifest_metrics['seam_risk_ratio']:.2f}"
        )
        evidence.append(
            "manifest switch detector — "
            f"owner switch ratio {owner_switch_ratio:.2f}, "
            f"alternating-triplet ratio {alternating_triplet_ratio:.2f}, "
            f"swap density {swap_density:.2f}, "
            f"switch risk {switch_detector_risk:.2f}"
        )
        if manifest_metrics['multi_owner_conflict_ratio'] > 0.0:
            fixes.append("Resolved manifest still contains explicit multi-owner conflicts; collapse those seams to one low-end owner and one foreground owner.")
        if manifest_metrics['lead_conflict_ratio'] > 0.0:
            fixes.append("Resolved manifest still permits competing lead ownership in some sections; keep lead-vocal ownership singular through swaps.")
        if manifest_metrics['crowding_ratio'] > 0.34 or manifest_metrics['avg_overlap_beats'] > 3.0:
            fixes.append("Reduce manifest overlap windows and donor density; the explicit plan is already too crowded before audio rendering artifacts enter the picture.")
        if manifest_metrics['seam_risk_ratio'] > 0.34:
            fixes.append("High-stretch or transition-heavy sections in the manifest are likely seam risks; prefer phrase-safer windows or shorter swaps.")
        if switch_detector_risk > 0.45:
            fixes.append("Manifest ownership is flipping often enough to read like track switching; keep a steadier parent backbone and reserve swaps for major structural turns.")
        if manifest_metrics['low_end_owner_stability_risk'] > 0.45:
            fixes.append("Low-end ownership is not staying anchored through adjacent sections; keep one kick/sub owner across the seam unless the swap is structurally unavoidable.")

    evidence.append(
        "avg seam-local boundary deltas — "
        f"energy {aggregate_metrics['avg_energy_jump']:.3f} (signed {aggregate_metrics['avg_energy_signed_delta']:.3f}), "
        f"spectral {aggregate_metrics['avg_spectral_jump']:.3f}, "
        f"onset {aggregate_metrics['avg_onset_jump']:.3f} (signed {aggregate_metrics['avg_onset_signed_delta']:.3f}), "
        f"low-end crowding {aggregate_metrics['avg_low_end_crowding_risk']:.3f}, "
        f"foreground collision {aggregate_metrics['avg_foreground_collision_risk']:.3f}, "
        f"vocal competition {aggregate_metrics['avg_vocal_competition_risk']:.3f}, "
        f"texture {aggregate_metrics['avg_texture_shift']:.3f}, intent mismatch {aggregate_metrics['avg_intent_mismatch']:.3f}"
    )
    for row in worst:
        evidence.append(
            f"boundary {row['boundary_index']} @ {row['boundary_time']:.1f}s {row['intent']} severity {row['severity']:.2f} "
            f"(energy {row['energy_jump']:.2f} / signed {row['energy_signed_delta']:.2f}, spectral {row['spectral_jump']:.2f}, onset {row['onset_jump']:.2f}, "
            f"low-end {row['low_end_crowding_risk']:.2f}, foreground {row['foreground_collision_risk']:.2f}, vocals {row['vocal_competition_risk']:.2f}, "
            f"texture {row['texture_shift']:.2f}, intent mismatch {row['intent_mismatch']:.2f})"
        )

    if aggregate_metrics["avg_energy_jump"] > 0.35 and aggregate_metrics["avg_intent_mismatch"] > 0.15:
        fixes.append("Smooth energy handoffs at section seams so transitions do not feel pasted or cliff-like.")
    if aggregate_metrics["avg_spectral_jump"] > 0.30:
        fixes.append("Control brightness and timbre swaps at boundaries; filtered handoffs or better source-window matching should reduce spectral shock.")
    if aggregate_metrics["avg_onset_jump"] > 0.35 and aggregate_metrics["avg_intent_mismatch"] > 0.12:
        fixes.append("Reduce abrupt rhythmic-density changes at boundaries unless the transition is an intentional payoff/drop.")
    if aggregate_metrics["avg_low_end_crowding_risk"] > 0.45:
        fixes.append("Keep one clear low-end owner through transitions; avoid dual kick/sub overlap in the seam window.")
    if aggregate_metrics["avg_foreground_collision_risk"] > 0.45:
        fixes.append("Do not let both parents push the foreground at the same seam; stage a clear lead handoff instead of simultaneous spotlight overload.")
    if aggregate_metrics["avg_flatness_crowding_risk"] > 0.45:
        fixes.append("If the seam already feels dense, do not compress or layer both sources flat through the crossover bar.")
    if aggregate_metrics["avg_vocal_competition_risk"] > 0.45:
        fixes.append("Avoid unresolved vocal competition across section handoffs; one lead voice should own the listener focus.")
    if aggregate_metrics["avg_texture_shift"] > 0.35:
        fixes.append("Avoid switching groove, brightness, and texture all at once without a setup bar or transitional layer.")
    if aggregate_metrics["avg_intent_mismatch"] > 0.22:
        fixes.append("Match seam behavior to transition intent more closely; builds/lifts should rise with control, while swaps/cuts should avoid accidental cliff jumps.")

    if not fixes and avg_severity < 0.20:
        summary = "Transition seams look reasonably controlled across detected boundaries."
    elif score >= 70:
        summary = "Transition seams are mostly usable, with a few exposed boundaries to clean up."
    else:
        summary = "Transition seams are exposing abrupt jumps, crowding, or ownership collisions."

    details = {
        "aggregate_metrics": aggregate_metrics,
        "worst_boundaries": worst,
        "worst_moments": _boundary_worst_moment_clips(severity_rows, duration_seconds=duration),
        "transition_diagnostics": transition_types[:5],
    }
    if manifest_details:
        details['manifest_metrics'] = manifest_details
    return ListenSubscore(score=score, summary=summary, evidence=evidence, fixes=fixes, details=details)


def _coherence_score(song: SongDNA) -> ListenSubscore:
    evidence: list[str] = []
    fixes: list[str] = []
    features, bar_centers, bar_boundaries, warnings = _aggregate_bar_features(song)
    evidence.extend(warnings)
    if features is None or bar_centers is None or features.shape[0] < 4:
        return ListenSubscore(
            score=45.0,
            summary="Coherence scoring fell back because bar-synchronous features were unavailable.",
            evidence=evidence or ["coherence fallback triggered"],
            fixes=["Preserve readable source audio plus stable beat grids so bar-synchronous coherence can be evaluated."],
        )

    similarity = np.clip(features @ features.T, -1.0, 1.0)
    distance = 1.0 - similarity
    bar_count = features.shape[0]

    local_distances = np.diag(distance, k=1)
    within_bar_consistency = _clamp01(1.0 - float(np.median(local_distances)) / 0.45) if local_distances.size else 0.0

    nonlocal_maxima: list[float] = []
    for idx in range(bar_count):
        mask = np.ones(bar_count, dtype=bool)
        lo = max(0, idx - 1)
        hi = min(bar_count, idx + 2)
        mask[lo:hi] = False
        if np.any(mask):
            nonlocal_maxima.append(float(np.max(similarity[idx, mask])))
    repetition = _score_band(float(np.mean(nonlocal_maxima)) if nonlocal_maxima else 0.0, 0.35, 0.90, 0.25)

    novelty = np.linalg.norm(np.diff(features, axis=0), axis=1)
    phrase_boundaries = sorted(set(_safe_float_list(song.structure.get("phrase_boundaries_seconds", []))))
    section_starts = sorted({float(sec.get("start", 0.0)) for sec in (song.structure.get("sections", []) or []) if float(sec.get("start", 0.0)) > 0.0})
    target_boundaries = sorted({b for b in phrase_boundaries + section_starts if 0.0 < b < float(song.duration_seconds)})

    boundary_hits: list[float] = []
    if novelty.size and target_boundaries:
        transition_times = np.asarray(bar_boundaries[1:1 + novelty.size], dtype=float)
        for boundary in target_boundaries:
            nearest_idx = int(np.argmin(np.abs(transition_times - boundary)))
            boundary_hits.append(float(novelty[nearest_idx]))
    boundary_novelty = float(np.mean(boundary_hits)) if boundary_hits else 0.0
    baseline_novelty = float(np.mean(novelty)) if novelty.size else 0.0
    novelty_alignment = _clamp01(boundary_novelty / max(baseline_novelty * 1.25, 1e-6)) if target_boundaries else 0.4

    phrase_spans = []
    full_phrase_grid = [0.0, *target_boundaries, float(song.duration_seconds)]
    for start, end in zip(full_phrase_grid[:-1], full_phrase_grid[1:]):
        idx = np.where((bar_centers >= start) & (bar_centers < end))[0]
        if idx.size >= 2:
            span_similarity = similarity[np.ix_(idx, idx)]
            upper = span_similarity[np.triu_indices_from(span_similarity, k=1)]
            if upper.size:
                phrase_spans.append(float(np.mean(upper)))
    phrase_cohesion = _score_band(float(np.mean(phrase_spans)) if phrase_spans else within_bar_consistency, 0.45, 0.95, 0.30)

    score = round(100.0 * (0.30 * within_bar_consistency + 0.25 * repetition + 0.25 * novelty_alignment + 0.20 * phrase_cohesion), 1)

    evidence.append(f"{bar_count} bars evaluated from beat grid; median adjacent-bar distance {float(np.median(local_distances)) if local_distances.size else 0.0:.3f}")
    evidence.append(f"mean best nonlocal self-similarity {float(np.mean(nonlocal_maxima)) if nonlocal_maxima else 0.0:.3f}; boundary novelty lift {(boundary_novelty / max(baseline_novelty, 1e-6)) if baseline_novelty else 0.0:.3f}")
    if within_bar_consistency < 0.55:
        fixes.append("Adjacent bars are changing too erratically; tighten loop-level continuity and downbeat alignment.")
    if repetition < 0.50:
        fixes.append("Recurring motifs are weak; reinforce phrase-level repetition so sections feel intentional rather than random.")
    if novelty_alignment < 0.55:
        fixes.append("Large changes are not concentrating at phrase/section seams; align arrangement turns to boundary bars more clearly.")
    summary = "Bar-level coherence and recurrence look reasonably musical." if score >= 70 else "Bar-synchronous coherence is weak, erratic, or poorly aligned to phrase turns."
    return ListenSubscore(score=score, summary=summary, evidence=evidence, fixes=fixes)


def _mix_sanity_score(song: SongDNA) -> ListenSubscore:
    tonal = np.asarray(song.energy.get("spectral_centroid", []) or [], dtype=float)
    rms = np.asarray(song.energy.get("rms", []) or [], dtype=float)
    evidence: list[str] = []
    fixes: list[str] = []
    if tonal.size == 0 or rms.size == 0:
        return ListenSubscore(
            score=40.0,
            summary="Mix sanity metrics are too sparse to judge confidently.",
            evidence=["missing RMS or spectral centroid summaries"],
            fixes=["Emit stable spectral and dynamics summaries during analysis."],
        )

    centroid_mean = float(np.mean(tonal))
    centroid_var = float(np.std(tonal))
    rms_mean = float(np.mean(rms))
    rms_var = float(np.std(rms))
    centroid_score = _score_band(centroid_mean, 1200.0, 3200.0, 1800.0)
    rms_score = _score_band(rms_mean, 0.02, 0.18, 0.12)
    variance_score = _score_band(centroid_var + rms_var, 0.02, 0.25, 0.20)

    clutter = _ownership_clutter_metrics(song)
    manifest_details = _manifest_overlap_metrics(_load_neighbor_manifest(song)) if _load_neighbor_manifest(song) else None
    ownership_penalty = (
        0.20 * clutter["overcrowded_overlap_risk"]
        + 0.18 * clutter["low_end_conflict_risk"]
        + 0.16 * clutter["foreground_overload_risk"]
        + 0.15 * clutter["overcompressed_flatness_risk"]
        + 0.13 * clutter["vocal_competition_risk"]
        + 0.10 * clutter.get("crowding_burst_risk", 0.0)
        + 0.08 * clutter.get("crowding_sustained_ratio", 0.0)
    )
    if manifest_details:
        manifest_metrics = manifest_details['aggregate_metrics']
        ownership_penalty += (
            0.18 * manifest_metrics['multi_owner_conflict_ratio']
            + 0.16 * manifest_metrics['lead_conflict_ratio']
            + 0.14 * manifest_metrics['crowding_ratio']
            + 0.12 * manifest_metrics['avg_overlap_beats'] / 8.0
            + 0.10 * manifest_metrics['seam_risk_ratio']
            + 0.10 * manifest_metrics['low_end_owner_stability_risk']
        )
    base_score = 0.32 * centroid_score + 0.28 * rms_score + 0.18 * variance_score
    ownership_score = _clamp01(1.0 - ownership_penalty)
    score = round(100.0 * (base_score + 0.22 * ownership_score), 1)

    evidence.append(f"mean spectral centroid {centroid_mean:.1f} Hz, centroid variation {centroid_var:.1f}")
    evidence.append(f"mean RMS {rms_mean:.4f}, RMS variation {rms_var:.4f}")
    evidence.append(
        "ownership/clutter risks — "
        f"overlap {clutter['overcrowded_overlap_risk']:.2f}, "
        f"low-end {clutter['low_end_conflict_risk']:.2f}, "
        f"foreground {clutter['foreground_overload_risk']:.2f}, "
        f"flatness {clutter['overcompressed_flatness_risk']:.2f}, "
        f"vocals {clutter['vocal_competition_risk']:.2f}, "
        f"crowding burst {clutter.get('crowding_burst_risk', 0.0):.2f}, "
        f"sustained crowding {clutter.get('crowding_sustained_ratio', 0.0):.2f}"
    )
    if manifest_details:
        manifest_metrics = manifest_details['aggregate_metrics']
        evidence.append(
            "manifest ownership risks — "
            f"owner conflicts {manifest_metrics['multi_owner_conflict_ratio']:.2f}, "
            f"lead conflicts {manifest_metrics['lead_conflict_ratio']:.2f}, "
            f"crowding {manifest_metrics['crowding_ratio']:.2f}, "
            f"avg overlap beats {manifest_metrics['avg_overlap_beats']:.2f}, "
            f"low-end owner switch ratio {manifest_metrics['low_end_owner_switch_ratio']:.2f}, "
            f"low-end stability risk {manifest_metrics['low_end_owner_stability_risk']:.2f}, "
            f"collapse ratio {manifest_metrics['collapse_ratio']:.2f}"
        )
        if manifest_metrics['multi_owner_conflict_ratio'] > 0.0:
            fixes.append("The resolved manifest still assigns conflicting owners inside at least one section; make low-end and foreground ownership explicit and singular.")
        if manifest_metrics['lead_conflict_ratio'] > 0.0:
            fixes.append("Lead-vocal ownership is still ambiguous in the manifest; avoid explicit dual-lead sections.")
        if manifest_metrics['crowding_ratio'] > 0.34 or manifest_metrics['avg_overlap_beats'] > 3.0:
            fixes.append("The manifest itself is crowding the arrangement with long or dense overlaps; shorten the donor window before worrying about mastering.")
        if manifest_metrics['low_end_owner_stability_risk'] > 0.45:
            fixes.append("Low-end ownership is flipping too often across adjacent sections, especially through overlaps; keep the kick/sub anchor on one parent longer.")

    if rms_mean > 0.20:
        fixes.append("Watch overcompression or excessive density; leave more headroom and contrast.")
    if centroid_mean < 900.0:
        fixes.append("Low-end / low-mid heaviness may be obscuring clarity; check congestion and masking.")
    if centroid_mean > 4000.0:
        fixes.append("Top end may be too harsh or thin; rebalance spectral weight.")
    if clutter["overcrowded_overlap_risk"] > 0.50:
        fixes.append("Reduce full-spectrum overlap; too many simultaneous elements are making the render feel crowded instead of arranged.")
    if clutter["low_end_conflict_risk"] > 0.50:
        fixes.append("Enforce one low-end owner; stacked kick/bass weight is reading as conflict rather than impact.")
    if clutter["foreground_overload_risk"] > 0.50:
        fixes.append("Choose a single foreground owner in dense sections; both parents are fighting for listener attention.")
    if clutter["overcompressed_flatness_risk"] > 0.50:
        fixes.append("Back off flat, constant density; the render needs more dynamic contour and less wall-of-sound compression feel.")
    if clutter["vocal_competition_risk"] > 0.50:
        fixes.append("Resolve lead-vocal ownership more clearly; the current texture suggests competing lead material.")
    if clutter.get("crowding_burst_risk", 0.0) > 0.45:
        fixes.append("Dense mix crowding is spiking in short bursts; trim overlap entries/exits so sections stop bunching up at handoffs.")
    if clutter.get("crowding_sustained_ratio", 0.0) > 0.30:
        fixes.append("Crowding stays active for too much of the render; reduce sustained simultaneous full-spectrum occupancy instead of only EQing the master.")
    summary = "Mix sanity looks broadly usable." if score >= 70 else "Mix sanity indicators suggest congestion, ownership conflicts, or flatness."
    details = {"ownership_clutter_metrics": clutter}
    if manifest_details:
        details['manifest_metrics'] = manifest_details
    return ListenSubscore(score=score, summary=summary, evidence=evidence, fixes=fixes, details=details)


def _section_readability_metrics(song: SongDNA) -> dict[str, float]:
    sections = song.structure.get("sections", []) or []
    duration = max(float(song.duration_seconds or 0.0), 1e-6)
    if not sections:
        return {
            "readable_section_ratio": 0.0,
            "boundary_recovery": 0.0,
            "role_plausibility": 0.0,
            "climax_conviction": 0.0,
            "audio_boundary_clarity": 0.0,
            "section_contrast": 0.0,
            "narrative_flow": 0.0,
            "direction_flip_ratio": 0.0,
            "label_support_ratio": 0.0,
        }

    phrase_boundaries = sorted(set(_safe_float_list(song.structure.get("phrase_boundaries_seconds", []) or [])))
    novelty_boundaries = sorted(set(_safe_float_list(song.structure.get("novelty_boundaries_seconds", []) or [])))
    candidate_boundaries = sorted({b for b in phrase_boundaries + novelty_boundaries if 0.0 < b < duration})
    tolerance = max(_median_beat_interval(song) * 2.0, duration / max(len(sections) * 10.0, 8.0), 0.75)

    bar_rms = _energy_series(song, "rms")
    bar_onset = _energy_series(song, "onset_density")
    bar_low = _energy_series(song, "low_band_ratio")
    bar_flat = _energy_series(song, "spectral_flatness")
    bar_t = _series_times(bar_rms, duration)

    def _section_mask(start: float, end: float) -> np.ndarray:
        if bar_rms.size == 0:
            return np.asarray([], dtype=bool)
        if bar_t.size != bar_rms.size:
            return np.ones(bar_rms.size, dtype=bool)
        mask = (bar_t >= start) & (bar_t < end)
        if np.any(mask):
            return mask
        mid = (start + end) * 0.5
        idx = int(np.argmin(np.abs(bar_t - mid)))
        mask = np.zeros(bar_rms.size, dtype=bool)
        mask[idx] = True
        return mask

    def _masked_mean(values: np.ndarray, mask: np.ndarray, default: float = 0.0) -> float:
        if values.size == 0 or mask.size == 0:
            return default
        usable = values[mask[: values.size]] if mask.size != values.size else values[mask]
        if usable.size == 0:
            return default
        return float(np.mean(usable))

    def _section_energy(start: float, end: float) -> float:
        if bar_rms.size == 0:
            return 0.0
        mask = _section_mask(start, end)
        onset_mean = _masked_mean(bar_onset, mask)
        low_mean = _masked_mean(bar_low, mask)
        flat_mean = _masked_mean(bar_flat, mask)
        rms_mean = _masked_mean(bar_rms, mask)
        return float(
            0.55 * rms_mean
            + 0.20 * onset_mean
            + 0.15 * low_mean
            + 0.10 * max(0.0, 1.0 - flat_mean)
        )

    section_profiles: list[dict[str, float]] = []
    section_energies: list[float] = []
    for section in sections:
        start = float(section.get("start", 0.0) or 0.0)
        end = float(section.get("end", start) or start)
        mask = _section_mask(start, end)
        rms_mean = _masked_mean(bar_rms, mask)
        onset_mean = _masked_mean(bar_onset, mask)
        low_mean = _masked_mean(bar_low, mask)
        flat_mean = _masked_mean(bar_flat, mask)
        energy_value = float(
            0.55 * rms_mean
            + 0.20 * onset_mean
            + 0.15 * low_mean
            + 0.10 * max(0.0, 1.0 - flat_mean)
        )
        section_energies.append(energy_value)

        rms_vals = bar_rms[mask] if bar_rms.size and mask.size == bar_rms.size else np.asarray([rms_mean], dtype=float)
        if rms_vals.size == 0:
            rms_vals = np.asarray([rms_mean], dtype=float)
        thirds = np.array_split(rms_vals, min(3, max(1, int(rms_vals.size))))
        start_mean = float(np.mean(thirds[0])) if thirds else rms_mean
        end_mean = float(np.mean(thirds[-1])) if thirds else rms_mean
        peak_mean = float(np.max(rms_vals)) if rms_vals.size else rms_mean
        valley_mean = float(np.min(rms_vals)) if rms_vals.size else rms_mean
        rise = end_mean - start_mean
        headroom = max(peak_mean - rms_mean, 0.0)
        sustain = max(min(end_mean, peak_mean) - max(valley_mean, start_mean), 0.0)
        end_focus = end_mean / max(peak_mean, 1e-6)
        section_profiles.append({
            "energy": energy_value,
            "rms_mean": rms_mean,
            "onset_mean": onset_mean,
            "low_mean": low_mean,
            "flat_mean": flat_mean,
            "start_mean": start_mean,
            "end_mean": end_mean,
            "peak_mean": peak_mean,
            "valley_mean": valley_mean,
            "rise": rise,
            "headroom": headroom,
            "sustain": sustain,
            "end_focus": end_focus,
        })

    energy_low = min(section_energies) if section_energies else 0.0
    energy_high = max(section_energies) if section_energies else 0.0
    energy_span = max(energy_high - energy_low, 1e-6)

    rise_values = [profile["rise"] for profile in section_profiles]
    headroom_values = [profile["headroom"] for profile in section_profiles]
    sustain_values = [profile["sustain"] for profile in section_profiles]
    rise_span = max((max(rise_values) - min(rise_values)), 1e-6) if rise_values else 1e-6
    headroom_span = max((max(headroom_values) - min(headroom_values)), 1e-6) if headroom_values else 1e-6
    sustain_span = max((max(sustain_values) - min(sustain_values)), 1e-6) if sustain_values else 1e-6

    readable_sections = 0
    boundary_hits = 0
    role_scores: list[float] = []
    climax_scores: list[float] = []
    climax_relative_centers: list[float] = []
    boundary_clarity_scores: list[float] = []
    narrative_flow_scores: list[float] = []
    section_direction_changes: list[float] = []
    label_hits = 0
    known_tokens = {"intro", "verse", "build", "pre", "payoff", "chorus", "drop", "bridge", "outro"}

    for idx, section in enumerate(sections):
        start = float(section.get("start", 0.0) or 0.0)
        end = max(float(section.get("end", start) or start), start)
        if end <= start:
            continue
        span = end - start
        center = (start + end) * 0.5
        rel_center = center / duration
        profile = section_profiles[idx] if idx < len(section_profiles) else {
            "rise": 0.0,
            "headroom": 0.0,
            "sustain": 0.0,
            "end_focus": 0.0,
            "onset_mean": 0.0,
            "low_mean": 0.0,
            "flat_mean": 0.0,
        }
        energy_norm = _clamp01((section_energies[idx] - energy_low) / energy_span) if section_energies else 0.0
        rise_norm = _clamp01((profile["rise"] - min(rise_values)) / rise_span) if rise_values else 0.0
        headroom_norm = _clamp01((profile["headroom"] - min(headroom_values)) / headroom_span) if headroom_values else 0.0
        sustain_norm = _clamp01((profile["sustain"] - min(sustain_values)) / sustain_span) if sustain_values else 0.0

        start_gap = min((abs(start - boundary) for boundary in candidate_boundaries), default=duration)
        end_gap = min((abs(end - boundary) for boundary in candidate_boundaries), default=duration)
        start_support = _clamp01(1.0 - start_gap / max(tolerance, 1e-6)) if idx > 0 else 1.0
        end_support = _clamp01(1.0 - end_gap / max(tolerance, 1e-6)) if idx < len(sections) - 1 else 1.0
        boundary_support = 0.5 * (start_support + end_support)
        if idx > 0 and start_support >= 0.5:
            boundary_hits += 1
        if idx < len(sections) - 1 and end_support >= 0.5:
            boundary_hits += 1

        label = str(section.get("role") or section.get("label") or "").lower()
        has_known_label = any(token in label for token in known_tokens)
        if has_known_label:
            label_hits += 1

        expected_profile = []
        if rel_center < 0.18:
            expected_profile.append(1.0 - energy_norm)
        if 0.12 <= rel_center <= 0.55:
            expected_profile.append(1.0 - abs(energy_norm - 0.45) / 0.45)
        if 0.35 <= rel_center <= 0.78:
            expected_profile.append(1.0 - abs(energy_norm - 0.65) / 0.35)
        if rel_center >= 0.55:
            expected_profile.append(energy_norm)
        if rel_center >= 0.82:
            expected_profile.append(1.0 - abs(energy_norm - 0.35) / 0.35)
        position_energy_fit = float(np.mean([_clamp01(v) for v in expected_profile])) if expected_profile else 0.5

        transition_bonus = 0.0
        transition_in = str(section.get("transition_in") or "").lower()
        transition_out = str(section.get("transition_out") or "").lower()
        if transition_in in {"lift", "drop"}:
            transition_bonus += 0.15 * energy_norm
        if transition_out in {"lift", "drop"}:
            transition_bonus += 0.10 * energy_norm
        shape_fit = 0.5
        if rel_center < 0.20:
            shape_fit = _clamp01(
                0.50 * (1.0 - energy_norm)
                + 0.20 * (1.0 - rise_norm)
                + 0.15 * (1.0 - sustain_norm)
                + 0.15 * (1.0 - min(profile["onset_mean"] / 0.45, 1.0))
            )
        elif rel_center < 0.52:
            shape_fit = _clamp01(
                0.40 * (1.0 - abs(energy_norm - 0.42) / 0.42)
                + 0.25 * (1.0 - abs(rise_norm - 0.45) / 0.45)
                + 0.20 * (1.0 - sustain_norm)
                + 0.15 * (1.0 - min(profile["flat_mean"] / 0.35, 1.0))
            )
        elif rel_center < 0.80:
            shape_fit = _clamp01(
                0.32 * (1.0 - abs(energy_norm - 0.66) / 0.34)
                + 0.26 * rise_norm
                + 0.16 * headroom_norm
                + 0.16 * profile["end_focus"]
                + 0.10 * (1.0 - sustain_norm)
            )
        else:
            shape_fit = _clamp01(
                0.36 * energy_norm
                + 0.24 * sustain_norm
                + 0.20 * profile["end_focus"]
                + 0.12 * (1.0 - headroom_norm)
                + 0.08 * (1.0 - max(rise_norm - 0.55, 0.0))
            )

        if has_known_label:
            role_hint = 0.0
            if any(token in label for token in {"intro", "outro"}):
                role_hint = _clamp01(
                    0.45 * (1.0 - energy_norm)
                    + 0.20 * (1.0 - rise_norm)
                    + 0.20 * (1.0 - sustain_norm)
                    + 0.15 * (1.0 - min(profile["onset_mean"] / 0.45, 1.0))
                )
            elif any(token in label for token in {"verse", "bridge"}):
                role_hint = _clamp01(
                    0.42 * (1.0 - abs(energy_norm - 0.45) / 0.45)
                    + 0.23 * (1.0 - abs(rise_norm - 0.40) / 0.40)
                    + 0.20 * (1.0 - sustain_norm)
                    + 0.15 * (1.0 - min(profile["flat_mean"] / 0.35, 1.0))
                )
            elif any(token in label for token in {"build", "pre"}):
                rise_commit = _clamp01(profile["end_focus"] * rise_norm)
                role_hint = _clamp01(
                    0.22 * (1.0 - abs(energy_norm - 0.68) / 0.32)
                    + 0.28 * rise_commit
                    + 0.18 * headroom_norm
                    + 0.12 * profile["end_focus"]
                    + 0.10 * (1.0 - sustain_norm)
                )
                if profile["rise"] <= 0.0:
                    role_hint = _clamp01(role_hint - 0.22)
            elif any(token in label for token in {"payoff", "chorus", "drop"}):
                role_hint = _clamp01(
                    0.30 * energy_norm
                    + 0.26 * sustain_norm
                    + 0.22 * profile["end_focus"]
                    + 0.12 * (1.0 - headroom_norm)
                    + 0.10 * (1.0 - max(rise_norm - 0.55, 0.0))
                )
                if profile["end_focus"] < 0.72:
                    role_hint = _clamp01(role_hint - 0.18)
            position_energy_fit = 0.45 * position_energy_fit + 0.30 * shape_fit + 0.25 * _clamp01(role_hint)
        else:
            position_energy_fit = 0.62 * position_energy_fit + 0.38 * shape_fit

        prev_energy_norm = _clamp01((section_energies[idx - 1] - energy_low) / energy_span) if idx > 0 and section_energies else energy_norm
        next_energy_norm = _clamp01((section_energies[idx + 1] - energy_low) / energy_span) if idx + 1 < len(section_energies) else energy_norm
        local_contrast = max(abs(energy_norm - prev_energy_norm), abs(next_energy_norm - energy_norm)) if len(section_energies) > 1 else 0.0
        contrast_score = _clamp01((local_contrast - 0.08) / 0.28)

        role_score = _clamp01(0.50 * position_energy_fit + 0.25 * boundary_support + 0.15 * contrast_score + transition_bonus)
        if has_known_label:
            role_score = _clamp01(0.60 * role_score + 0.40 * _clamp01(role_hint))
        role_scores.append(role_score)

        readability = _clamp01(0.45 * boundary_support + 0.35 * position_energy_fit + 0.20 * contrast_score)
        if readability >= 0.55:
            readable_sections += 1

        if idx > 0:
            boundary_clarity_scores.append(_clamp01(0.55 * start_support + 0.45 * contrast_score))
        if idx < len(sections) - 1:
            boundary_clarity_scores.append(_clamp01(0.55 * end_support + 0.45 * contrast_score))

        local_climax = _clamp01(
            0.50 * energy_norm
            + 0.20 * boundary_support
            + 0.15 * contrast_score
            + 0.15 * _clamp01(1.0 - abs(rel_center - 0.72) / 0.28)
        )
        if transition_in == "drop":
            local_climax = _clamp01(local_climax + 0.15)
        climax_scores.append(local_climax)
        climax_relative_centers.append(rel_center)

        if idx > 0:
            expected_direction = 0.0
            if rel_center >= 0.55:
                expected_direction = 1.0
            elif rel_center <= 0.18:
                expected_direction = -0.4
            actual_direction = energy_norm - prev_energy_norm
            direction_fit = 1.0 - min(abs(actual_direction - expected_direction), 1.0)
            if idx > 1:
                previous_direction = prev_energy_norm - _clamp01((section_energies[idx - 2] - energy_low) / energy_span)
                if abs(actual_direction) >= 0.12 and abs(previous_direction) >= 0.12:
                    section_direction_changes.append(1.0 if np.sign(actual_direction) != np.sign(previous_direction) else 0.0)
            narrative_flow_scores.append(_clamp01(0.65 * direction_fit + 0.35 * contrast_score))

    direction_flip_ratio = float(np.mean(section_direction_changes)) if section_direction_changes else 0.0
    narrative_flow = _clamp01(
        (float(np.mean(narrative_flow_scores)) if narrative_flow_scores else 0.0)
        - 0.45 * direction_flip_ratio
    )
    climax_position = 0.0
    if climax_scores and climax_relative_centers and len(climax_scores) == len(climax_relative_centers):
        climax_position = float(climax_relative_centers[int(np.argmax(np.asarray(climax_scores, dtype=float)))])
    internal_boundary_count = max((len(sections) - 1) * 2, 1)
    return {
        "readable_section_ratio": round(readable_sections / max(len(sections), 1), 3),
        "boundary_recovery": round(boundary_hits / internal_boundary_count, 3),
        "role_plausibility": round(float(np.mean(role_scores)) if role_scores else 0.0, 3),
        "climax_conviction": round(max(climax_scores) if climax_scores else 0.0, 3),
        "climax_position": round(climax_position, 3),
        "audio_boundary_clarity": round(float(np.mean(boundary_clarity_scores)) if boundary_clarity_scores else 0.0, 3),
        "section_contrast": round(float(np.mean([abs(section_energies[idx] - section_energies[idx - 1]) for idx in range(1, len(section_energies))])) / max(energy_span, 1e-6) if len(section_energies) > 1 else 0.0, 3),
        "narrative_flow": round(narrative_flow, 3),
        "direction_flip_ratio": round(direction_flip_ratio, 3),
        "label_support_ratio": round(label_hits / max(len(sections), 1), 3),
    }



def _song_likeness_score(
    song: SongDNA,
    structure: ListenSubscore,
    groove: ListenSubscore,
    energy_arc: ListenSubscore,
    transition: ListenSubscore,
    coherence: ListenSubscore,
    mix_sanity: ListenSubscore,
) -> ListenSubscore:
    evidence: list[str] = []
    fixes: list[str] = []

    section_count = len(song.structure.get("sections", []) or [])
    readability = _section_readability_metrics(song)
    readable_section_ratio = float(readability["readable_section_ratio"])
    boundary_recovery = float(readability["boundary_recovery"])
    role_plausibility = float(readability["role_plausibility"])
    planner_climax_conviction = float(readability["climax_conviction"])
    audio_boundary_clarity = float(readability["audio_boundary_clarity"])
    section_contrast = float(readability["section_contrast"])
    narrative_flow = float(readability["narrative_flow"])
    direction_flip_ratio = float(readability["direction_flip_ratio"])
    label_support_ratio = float(readability["label_support_ratio"])
    musical_intelligence = getattr(song, 'musical_intelligence', {}) or {}
    mi_summary = dict(musical_intelligence.get('summary') or {})
    has_musical_intelligence = bool(mi_summary)
    melodic_identity_strength = float(mi_summary.get('melodic_identity_strength', 0.5) if has_musical_intelligence else 0.5)
    rhythmic_confidence = float(mi_summary.get('rhythmic_confidence', 0.5) if has_musical_intelligence else 0.5)
    harmonic_confidence = float(mi_summary.get('harmonic_confidence', 0.5) if has_musical_intelligence else 0.5)
    timbral_coherence_signal = float(mi_summary.get('timbral_coherence', 0.5) if has_musical_intelligence else 0.5)
    tension_release_confidence = float(mi_summary.get('tension_release_confidence', 0.5) if has_musical_intelligence else 0.5)
    motif_repeat_strength = float(((musical_intelligence.get('motif_reuse') or {}).get('motif_repeat_strength', 0.5)) if has_musical_intelligence else 0.5)
    cadence_strength = float(((musical_intelligence.get('harmonic_function') or {}).get('cadence_strength', 0.5)) if has_musical_intelligence else 0.5)
    anchor_stability = float(((musical_intelligence.get('timbral_anchors') or {}).get('anchor_stability', 0.5)) if has_musical_intelligence else 0.5)
    recognizable_ratio = _clamp01(
        0.30 * readable_section_ratio
        + 0.22 * boundary_recovery
        + 0.18 * role_plausibility
        + 0.12 * audio_boundary_clarity
        + 0.12 * narrative_flow
        + 0.06 * planner_climax_conviction
    )

    manifest_details = _manifest_overlap_metrics(_load_neighbor_manifest(song)) if _load_neighbor_manifest(song) else None
    manifest_metrics = (manifest_details or {}).get("aggregate_metrics", {})
    identity = (manifest_details or {}).get("fusion_identity", {})
    section_primary_counts = identity.get("section_primary_counts") or {"A": 0, "B": 0}

    primary_sequence: list[str] = []
    if manifest_details:
        manifest_payload = _load_neighbor_manifest(song) or {}
        for section in manifest_payload.get("sections") or []:
            owner = section.get("source_parent") or section.get("foreground_owner")
            if owner in {"A", "B"}:
                primary_sequence.append(str(owner))

    owner_switches = sum(1 for left, right in zip(primary_sequence, primary_sequence[1:]) if left != right)
    owner_switch_ratio = owner_switches / max(len(primary_sequence) - 1, 1) if len(primary_sequence) >= 2 else 0.0
    max_parent_share = max(section_primary_counts.values()) / max(sum(section_primary_counts.values()), 1)
    backbone_continuity = _clamp01(
        0.22 * _clamp01((structure.score - 55.0) / 30.0)
        + 0.18 * _clamp01((groove.score - 55.0) / 30.0)
        + 0.18 * _clamp01((coherence.score - 50.0) / 35.0)
        + 0.14 * recognizable_ratio
        + 0.10 * boundary_recovery
        + 0.10 * audio_boundary_clarity
        + 0.08 * narrative_flow
        + 0.06 * (1.0 - direction_flip_ratio)
        + 0.08 * (1.0 - _clamp01((owner_switch_ratio - 0.42) / 0.45))
        + 0.04 * rhythmic_confidence
        + 0.04 * timbral_coherence_signal
    )

    donor_clutter_rejection = _clamp01(
        0.30 * _clamp01((mix_sanity.score - 45.0) / 35.0)
        + 0.22 * _clamp01((transition.score - 45.0) / 35.0)
        + 0.12 * role_plausibility
        + 0.08 * audio_boundary_clarity
        + 0.08 * (1.0 - section_contrast)
        + 0.14 * (1.0 - min(float(manifest_metrics.get("avg_overlap_beats", 0.0)) / 4.0, 1.0))
        + 0.10 * (1.0 - float(manifest_metrics.get("crowding_ratio", 0.0)))
        + 0.06 * (1.0 - float(manifest_metrics.get("lead_conflict_ratio", 0.0)))
        + 0.06 * timbral_coherence_signal
        + 0.04 * harmonic_confidence
    )

    climax_conviction = _clamp01(
        0.28 * _clamp01((energy_arc.score - 50.0) / 35.0)
        + 0.16 * _clamp01((transition.score - 45.0) / 35.0)
        + 0.12 * _clamp01((coherence.score - 50.0) / 35.0)
        + 0.10 * _clamp01((groove.score - 55.0) / 30.0)
        + 0.14 * planner_climax_conviction
        + 0.10 * role_plausibility
        + 0.10 * audio_boundary_clarity
        + 0.10 * section_contrast
        + 0.08 * tension_release_confidence
        + 0.06 * cadence_strength
    )

    integrated_two_parent_section_ratio = float(manifest_metrics.get("integrated_two_parent_section_ratio", 0.0) or 0.0)
    support_layer_section_ratio = float(manifest_metrics.get("support_layer_section_ratio", 0.0) or 0.0)
    full_mix_medley_risk = float(manifest_metrics.get("full_mix_medley_risk", 0.0) or 0.0)

    identity_penalty = 0.0
    if manifest_details:
        identity_penalty = (
            0.24 * float(manifest_metrics.get("background_only_identity_gap", 0.0))
            + 0.14 * float(manifest_metrics.get("collapse_ratio", 0.0))
            + 0.16 * _clamp01(max_parent_share - 0.86)
            + 0.14 * float(manifest_metrics.get("crowding_ratio", 0.0))
            + 0.12 * float(manifest_metrics.get("seam_risk_ratio", 0.0))
            + 0.20 * full_mix_medley_risk
            + 0.08 * _clamp01((0.35 - melodic_identity_strength) / 0.35)
            + 0.08 * _clamp01((0.35 - harmonic_confidence) / 0.35)
            + 0.08 * _clamp01((0.35 - timbral_coherence_signal) / 0.35)
        )
        if not has_musical_intelligence:
            identity_penalty += 0.04 * full_mix_medley_risk + 0.03 * float(manifest_metrics.get("crowding_ratio", 0.0))

    composite_song_risk = _clamp01(
        0.22 * (1.0 - backbone_continuity)
        + 0.14 * (1.0 - recognizable_ratio)
        + 0.10 * (1.0 - boundary_recovery)
        + 0.08 * (1.0 - role_plausibility)
        + 0.08 * (1.0 - narrative_flow)
        + 0.07 * (1.0 - planner_climax_conviction)
        + 0.07 * (1.0 - audio_boundary_clarity)
        + 0.10 * _clamp01((owner_switch_ratio - 0.38) / 0.34)
        + 0.08 * float(manifest_metrics.get("background_only_identity_gap", 0.0))
        + 0.05 * float(manifest_metrics.get("crowding_ratio", 0.0))
        + 0.05 * float(manifest_metrics.get("seam_risk_ratio", 0.0))
        + 0.16 * full_mix_medley_risk
        + 0.08 * _clamp01((0.20 - support_layer_section_ratio) / 0.20)
        + 0.06 * _clamp01((0.30 - melodic_identity_strength) / 0.30)
        + 0.06 * _clamp01((0.32 - harmonic_confidence) / 0.32)
        + 0.05 * _clamp01((0.32 - timbral_coherence_signal) / 0.32)
        + 0.04 * _clamp01((0.32 - tension_release_confidence) / 0.32)
    )

    raw_norm = _clamp01(
        0.39 * backbone_continuity
        + 0.29 * donor_clutter_rejection
        + 0.20 * climax_conviction
        + 0.07 * integrated_two_parent_section_ratio
        + 0.05 * support_layer_section_ratio
        + 0.05 * melodic_identity_strength
        + 0.04 * harmonic_confidence
        + 0.04 * timbral_coherence_signal
        + 0.03 * tension_release_confidence
        - identity_penalty
    )
    score = round(100.0 * raw_norm, 1)

    evidence.append(
        f"backbone continuity {backbone_continuity:.3f}; donor-clutter rejection {donor_clutter_rejection:.3f}; climax conviction {climax_conviction:.3f}; readable section ratio {recognizable_ratio:.3f}"
    )
    evidence.append(
        f"section readability: readable sections {readable_section_ratio:.3f}, boundary recovery {boundary_recovery:.3f}, audio boundary clarity {audio_boundary_clarity:.3f}, role plausibility {role_plausibility:.3f}, narrative flow {narrative_flow:.3f}, direction flips {direction_flip_ratio:.3f}, section contrast {section_contrast:.3f}, planner/audio climax {planner_climax_conviction:.3f}, label support {label_support_ratio:.3f}, composite-song risk {composite_song_risk:.3f}"
    )
    evidence.append(
        f"musical intelligence: melodic identity {melodic_identity_strength:.3f}, motif repeat {motif_repeat_strength:.3f}, rhythmic confidence {rhythmic_confidence:.3f}, harmonic confidence {harmonic_confidence:.3f}, cadence strength {cadence_strength:.3f}, timbral coherence {timbral_coherence_signal:.3f}, anchor stability {anchor_stability:.3f}, tension/release confidence {tension_release_confidence:.3f}"
    )
    if manifest_details:
        evidence.append(
            f"section-owner backbone: counts {section_primary_counts}, switches {owner_switches}, switch ratio {owner_switch_ratio:.3f}, max parent share {max_parent_share:.3f}, background-only identity gap {float(manifest_metrics.get('background_only_identity_gap', 0.0)):.3f}"
        )
        evidence.append(
            f"manifest anti-clutter: avg overlap beats {float(manifest_metrics.get('avg_overlap_beats', 0.0)):.2f}, crowding {float(manifest_metrics.get('crowding_ratio', 0.0)):.2f}, lead conflicts {float(manifest_metrics.get('lead_conflict_ratio', 0.0)):.2f}, seam risk {float(manifest_metrics.get('seam_risk_ratio', 0.0)):.2f}, collapse {float(manifest_metrics.get('collapse_ratio', 0.0)):.2f}, integrated two-parent sections {integrated_two_parent_section_ratio:.2f}, support-layer sections {support_layer_section_ratio:.2f}, medley risk {full_mix_medley_risk:.2f}"
        )

    if backbone_continuity < 0.52:
        fixes.append("Strengthen whole-song backbone continuity; adjacent sections currently behave more like stitched donors than one child song.")
    if donor_clutter_rejection < 0.52:
        fixes.append("Reject cluttered donor carryover more aggressively; long overlaps and competing foreground material are making the child feel pasted together.")
    if climax_conviction < 0.52:
        fixes.append("Deliver a clearer musical backbone into the payoff; the current section program rises numerically but does not sell one convincing song arc.")
    if boundary_recovery < 0.45:
        fixes.append("Recover phrase/section seams more faithfully; weak boundary support is making section turns feel arbitrary instead of planned.")
    if role_plausibility < 0.48:
        fixes.append("Make section roles more plausible in timing and energy shape; the current program does not read like stable setup/build/payoff behavior.")
    if recognizable_ratio < 0.45:
        fixes.append("Use a more believable section program with readable section turns instead of generic undifferentiated blocks.")
    if direction_flip_ratio > 0.45:
        fixes.append("Reduce section-to-section energy ping-pong; repeated rise/fall reversals are making the render feel stitched instead of like one song.")
    if manifest_details and owner_switch_ratio > 0.65:
        fixes.append("Too many section-owner flips are weakening continuity; keep a steadier backbone and reserve parent swaps for real structural turns.")
    if manifest_details and float(manifest_metrics.get("background_only_identity_gap", 0.0)) > 0.20:
        fixes.append("Do not count background-only donor presence as fusion identity; promote real section ownership or reject the render as fake two-parent glue.")
    if manifest_details and integrated_two_parent_section_ratio < 0.20:
        fixes.append("Stop rendering whole-section full-mix swaps as the default; major sections need explicit donor support or counterlayers so the child sounds recomposed, not stitched.")
    if manifest_details and full_mix_medley_risk > 0.55:
        fixes.append("Hard reject medley-like section switching; the current manifest is still alternating between parent full mixes instead of constructing integrated child sections.")
    if melodic_identity_strength < 0.30:
        fixes.append("Strengthen motif continuity and melodic anchor reuse; the child currently lacks a memorable contour thread across sections.")
    if harmonic_confidence < 0.32:
        fixes.append("Reject tonal incoherence and weak cadence logic; child sections need a clearer local harmonic center and better phrase resolution.")
    if rhythmic_confidence < 0.38:
        fixes.append("Reject rhythm-language mismatch; drum grammar and pulse lock are too weak to sell one coherent groove system.")
    if timbral_coherence_signal < 0.32:
        fixes.append("Stabilize timbral anchors across the child; the palette is changing faster than the arrangement can justify.")
    if tension_release_confidence < 0.30:
        fixes.append("Clarify tension/release targets; builds and payoffs are not creating a convincing emotional rise and release pattern.")
    if manifest_details and float(manifest_metrics.get("avg_overlap_beats", 0.0)) > 3.0:
        fixes.append("Reduce donor linger across section seams; current overlap length is high enough to blur the child-song backbone.")
    if composite_song_risk > 0.50:
        fixes.append("Hard reject stitched-composite arrangements that still read like multiple pasted songs instead of one continuous child record.")

    if score >= 75:
        summary = "Song-likeness is strong enough that the render reads like one coherent child song more than a stitched mashup."
    elif score >= 60:
        summary = "Song-likeness is partial: some backbone is there, but continuity or donor clutter still weakens the illusion of one song."
    else:
        summary = "Song-likeness is weak: the render is still reading more like stitched donor material than one coherent song."

    return ListenSubscore(
        score=score,
        summary=summary,
        evidence=evidence,
        fixes=fixes,
        details={
            "aggregate_metrics": {
                "backbone_continuity": round(backbone_continuity, 3),
                "donor_clutter_rejection": round(donor_clutter_rejection, 3),
                "climax_conviction": round(climax_conviction, 3),
                "readable_section_ratio": round(readable_section_ratio, 3),
                "recognizable_section_ratio": round(recognizable_ratio, 3),
                "boundary_recovery": round(boundary_recovery, 3),
                "audio_boundary_clarity": round(audio_boundary_clarity, 3),
                "role_plausibility": round(role_plausibility, 3),
                "narrative_flow": round(narrative_flow, 3),
                "direction_flip_ratio": round(direction_flip_ratio, 3),
                "section_contrast": round(section_contrast, 3),
                "planner_audio_climax_conviction": round(planner_climax_conviction, 3),
                "climax_section_relative_center": round(float(readability.get("climax_position", 0.0) or 0.0), 3),
                "label_support_ratio": round(label_support_ratio, 3),
                "composite_song_risk": round(composite_song_risk, 3),
                "owner_switch_ratio": round(owner_switch_ratio, 3),
                "owner_switch_count": owner_switches,
                "max_parent_share": round(max_parent_share, 3),
                "background_only_identity_gap": round(float(manifest_metrics.get("background_only_identity_gap", 0.0)), 3),
                "integrated_two_parent_section_ratio": round(integrated_two_parent_section_ratio, 3),
                "support_layer_section_ratio": round(support_layer_section_ratio, 3),
                "full_mix_medley_risk": round(full_mix_medley_risk, 3),
                "melodic_identity_strength": round(melodic_identity_strength, 3),
                "motif_repeat_strength": round(motif_repeat_strength, 3),
                "rhythmic_confidence": round(rhythmic_confidence, 3),
                "harmonic_confidence": round(harmonic_confidence, 3),
                "cadence_strength": round(cadence_strength, 3),
                "timbral_coherence": round(timbral_coherence_signal, 3),
                "timbral_anchor_stability": round(anchor_stability, 3),
                "tension_release_confidence": round(tension_release_confidence, 3),
            },
            "manifest_metrics": manifest_details,
            "musical_intelligence": musical_intelligence,
        },
    )


def _build_gating(
    overall: float,
    song_likeness: ListenSubscore,
    groove: ListenSubscore,
    transition: ListenSubscore,
    coherence: ListenSubscore,
    mix_sanity: ListenSubscore,
) -> tuple[float, str, dict[str, Any]]:
    hard_fail_reasons: list[str] = []
    soft_fail_reasons: list[str] = []

    transition_metrics = transition.details.get("aggregate_metrics", {}) if isinstance(transition.details, dict) else {}
    song_metrics = song_likeness.details.get("aggregate_metrics", {}) if isinstance(song_likeness.details, dict) else {}
    manifest_switch_detector_risk = float(transition_metrics.get("manifest_switch_detector_risk", 0.0) or 0.0)
    manifest_owner_switch_ratio = float(transition_metrics.get("manifest_owner_switch_ratio", 0.0) or 0.0)
    manifest_alternating_triplet_ratio = float(transition_metrics.get("manifest_alternating_triplet_ratio", 0.0) or 0.0)
    manifest_swap_density = float(transition_metrics.get("manifest_swap_density", 0.0) or 0.0)
    composite_song_risk = float(song_metrics.get("composite_song_risk", 0.0) or 0.0)
    integrated_two_parent_section_ratio = float(song_metrics.get("integrated_two_parent_section_ratio", 0.0) or 0.0)
    support_layer_section_ratio = float(song_metrics.get("support_layer_section_ratio", 0.0) or 0.0)
    full_mix_medley_risk = float(song_metrics.get("full_mix_medley_risk", 0.0) or 0.0)
    melodic_identity_strength = float(song_metrics.get("melodic_identity_strength", 0.5) or 0.5)
    harmonic_confidence = float(song_metrics.get("harmonic_confidence", 0.5) or 0.5)
    rhythmic_confidence = float(song_metrics.get("rhythmic_confidence", 0.5) or 0.5)
    timbral_coherence = float(song_metrics.get("timbral_coherence", 0.5) or 0.5)
    groove_floor_triggered = groove.score < 55.0

    if song_likeness.score < 45.0:
        hard_fail_reasons.append("song-likeness is too weak")
    if coherence.score < 42.0:
        hard_fail_reasons.append("coherence is too weak")
    if (
        manifest_switch_detector_risk >= 0.72
        and manifest_owner_switch_ratio >= 0.72
        and (manifest_alternating_triplet_ratio >= 0.34 or manifest_swap_density >= 0.60)
    ):
        hard_fail_reasons.append("obvious track-switch seams detected")
    if composite_song_risk > 0.50:
        hard_fail_reasons.append("composite detector says the render still sounds like multiple pasted songs")
    if full_mix_medley_risk >= 0.68 and integrated_two_parent_section_ratio < 0.20:
        hard_fail_reasons.append("render is still mostly alternating full-mix parent sections instead of integrated child sections")
    if melodic_identity_strength < 0.24 and harmonic_confidence < 0.24 and timbral_coherence < 0.24:
        hard_fail_reasons.append("musical identity is too weak across melody, harmony, and timbre")
    if groove_floor_triggered:
        soft_fail_reasons.append("groove grid is not stable enough")
    if transition.score < 45.0:
        soft_fail_reasons.append("transition seams are still too exposed")
    if mix_sanity.score < 45.0:
        soft_fail_reasons.append("mix/ownership clutter is still too high")
    if float(song_metrics.get("background_only_identity_gap", 0.0) or 0.0) > 0.35:
        soft_fail_reasons.append("minority-parent presence is mostly background-only")
    if support_layer_section_ratio < 0.18 and manifest_owner_switch_ratio > 0.45 and full_mix_medley_risk > 0.55:
        soft_fail_reasons.append("sections are switching parents faster than they are being recomposed")
    if melodic_identity_strength < 0.34:
        soft_fail_reasons.append("motif continuity is too weak")
    if harmonic_confidence < 0.34:
        soft_fail_reasons.append("tonal continuity is too weak")
    if rhythmic_confidence < 0.34:
        soft_fail_reasons.append("rhythmic language is too unstable")
    if timbral_coherence < 0.34:
        soft_fail_reasons.append("timbral palette coherence is too weak")

    gated_overall = overall
    if hard_fail_reasons:
        gated_overall = min(gated_overall, 49.0)
        status = "reject"
    elif soft_fail_reasons and (song_likeness.score < 60.0 or coherence.score < 55.0):
        gated_overall = min(gated_overall, 59.0)
        status = "review"
    else:
        status = "pass"

    return round(gated_overall, 1), status, {
        "status": status,
        "hard_fail_reasons": hard_fail_reasons,
        "soft_fail_reasons": soft_fail_reasons,
        "song_likeness_floor_triggered": bool(song_likeness.score < 45.0),
        "coherence_floor_triggered": bool(coherence.score < 42.0),
        "groove_floor_triggered": bool(groove_floor_triggered),
        "manifest_switch_detector_risk": round(manifest_switch_detector_risk, 3),
        "manifest_owner_switch_ratio": round(manifest_owner_switch_ratio, 3),
        "manifest_alternating_triplet_ratio": round(manifest_alternating_triplet_ratio, 3),
        "manifest_swap_density": round(manifest_swap_density, 3),
        "composite_song_risk": round(composite_song_risk, 3),
        "integrated_two_parent_section_ratio": round(integrated_two_parent_section_ratio, 3),
        "support_layer_section_ratio": round(support_layer_section_ratio, 3),
        "full_mix_medley_risk": round(full_mix_medley_risk, 3),
        "composite_detector_triggered": bool(composite_song_risk > 0.50),
        "track_switch_seam_hard_reject_triggered": bool(
            manifest_switch_detector_risk >= 0.72
            and manifest_owner_switch_ratio >= 0.72
            and (manifest_alternating_triplet_ratio >= 0.34 or manifest_swap_density >= 0.60)
        ),
    }


def _verdict(score: float) -> str:
    if score >= 85:
        return "strong"
    if score >= 70:
        return "promising"
    if score >= 55:
        return "mixed"
    if score >= 40:
        return "weak"
    return "poor"


def evaluate_song(song: SongDNA) -> ListenReport:
    structure = _structure_score(song)
    groove = _groove_score(song)
    energy_arc = _energy_arc_score(song)
    transition = _transition_score(song)
    coherence = _coherence_score(song)
    mix_sanity = _mix_sanity_score(song)
    song_likeness = _song_likeness_score(song, structure, groove, energy_arc, transition, coherence, mix_sanity)

    raw_overall = round(
        0.16 * structure.score
        + 0.14 * groove.score
        + 0.15 * energy_arc.score
        + 0.12 * transition.score
        + 0.14 * coherence.score
        + 0.13 * mix_sanity.score
        + 0.16 * song_likeness.score,
        1,
    )
    overall, gate_status, gating = _build_gating(raw_overall, song_likeness, groove, transition, coherence, mix_sanity)

    reasons = [
        f"Song-likeness: {song_likeness.summary}",
        f"Structure: {structure.summary}",
        f"Groove: {groove.summary}",
        f"Energy arc: {energy_arc.summary}",
        f"Transitions: {transition.summary}",
        f"Coherence: {coherence.summary}",
        f"Mix sanity: {mix_sanity.summary}",
    ]
    if gate_status != "pass":
        reasons.insert(0, f"Gate: {gate_status} — {'; '.join(gating['hard_fail_reasons'] + gating['soft_fail_reasons'])}")

    prioritized_parts = [song_likeness, transition, mix_sanity, coherence, groove, energy_arc, structure]
    fixes = []
    for part in prioritized_parts:
        fixes.extend(part.fixes)
    deduped_fixes = []
    for item in fixes:
        if item not in deduped_fixes:
            deduped_fixes.append(item)

    report = ListenReport(
        source_path=song.source_path,
        duration_seconds=song.duration_seconds,
        overall_score=overall,
        structure=structure,
        groove=groove,
        energy_arc=energy_arc,
        transition=transition,
        coherence=coherence,
        mix_sanity=mix_sanity,
        song_likeness=song_likeness,
        verdict=_verdict(overall),
        top_reasons=reasons,
        top_fixes=deduped_fixes[:10],
        gating={
            **gating,
            "raw_overall_score": raw_overall,
        },
    )
    report.analysis_version = TRANSITION_ANALYSIS_VERSION
    return report


# ===== END src/core/evaluation/listen.py =====

# ===== BEGIN ai_dj.py (listen CLI section) =====

            "left_profile": {
                "case_id": left.get("case_id"),
                "display_label": left_label,
                "strengths": _report_strengths_and_weaknesses(left_report)[0],
                "weaknesses": _report_strengths_and_weaknesses(left_report)[1],
                "components": _report_component_snapshot(left_report),
            },
            "right_profile": {
                "case_id": right.get("case_id"),
                "display_label": right_label,
                "strengths": _report_strengths_and_weaknesses(right_report)[0],
                "weaknesses": _report_strengths_and_weaknesses(right_report)[1],
                "components": _report_component_snapshot(right_report),
            },
        },
        "summary": _summarize_comparison(
            left_report,
            right_report,
            {
                "overall_score_delta": overall_delta,
                "component_score_deltas": component_deltas,
            },
            left_label,
            right_label,
        ),
    }
    return comparison


def _build_listen_comparison(left_input: str, right_input: str) -> dict[str, Any]:
    left, right = _assign_display_labels([
        _resolve_compare_input(left_input),
        _resolve_compare_input(right_input),
    ])
    return _build_listen_comparison_from_resolved(left, right)


def listen(track: str, output: Optional[str], score_only: bool = False) -> int:
    analyze_audio = _get_analyze_audio_file()
    evaluate = _get_evaluate_song()
    track_path = _resolve_existing_audio_path(track, "track")
    song = analyze_audio(track_path)
    report = evaluate(song).to_dict()

    resolved_output = _resolve_output_path(output)
    if resolved_output:
        _write_json(resolved_output, report)
        print(f"Wrote listen report: {resolved_output}")
    elif not score_only:
        print(json.dumps(report, indent=2, sort_keys=True))

    if score_only:
        print(f"{float(report['overall_score']):.1f}")
        return 0

    print(f"Track: {Path(track_path).name}")
    print(f"Overall score: {report['overall_score']}")
    print(f"Verdict: {report['verdict']}")
    for key in LISTEN_COMPONENT_KEYS:
        part = report[key]
        print(f"- {key}: {part['score']} — {part['summary']}")
    return 0


def compare_listen(left: str, right: str, output: Optional[str]) -> int:
    comparison = _build_listen_comparison(left, right)
    resolved_output = _resolve_output_path(
        output,
        default_path=_stable_compare_output_path(left, right),
        default_filename="listen_compare.json",
    )
    if resolved_output:
        _write_json(resolved_output, comparison)
        print(f"Wrote listen comparison: {resolved_output}")

    if output is None:
        print(json.dumps(comparison, indent=2, sort_keys=True))

    left_label = comparison['left'].get('display_label') or comparison['left'].get('input_label', 'left')
    right_label = comparison['right'].get('display_label') or comparison['right'].get('input_label', 'right')
    print(f"Compare: {left_label} vs {right_label}")
    print(f"Overall winner: {comparison['winner']['overall']}")
    print(f"Overall score delta ({left_label} - {right_label}): {comparison['deltas']['overall_score_delta']:+.1f}")
    for key in LISTEN_COMPONENT_KEYS:
        delta = comparison['deltas']['component_score_deltas'][key]
        winner = comparison['winner']['components'][key]
        winner_label = left_label if winner == 'left' else right_label if winner == 'right' else 'tie'
        print(f"- {_metric_label(key)}: {winner_label} ({delta:+.1f})")
    decision = comparison.get('decision') or {}
    if decision.get('confidence'):
        print(f"Decision confidence: {decision['confidence']}")
    for line in decision.get('why') or []:
        print(f"  Why: {line}")
    for line in comparison['summary']:
        print(f"  {line}")
    return 0


# ===== END ai_dj.py (listen CLI section) =====

# ===== BEGIN ai_dj.py (argparse listen options) =====

    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    gen_parser = subparsers.add_parser("generate", help="Generate a new track")
    gen_parser.add_argument("--genre", "-g", help="Music genre")
    gen_parser.add_argument("--bpm", "-b", type=int, help="Beats per minute")
    gen_parser.add_argument("--key", "-k", help="Musical key (e.g., C minor)")
    gen_parser.add_argument("--output", "-o", help="Output file path")

    ana_parser = subparsers.add_parser("analyze", help="Analyze a track")
    ana_parser.add_argument("track", help="Path to track file")
    ana_parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed analysis")
    ana_parser.add_argument("--output", "-o", help="Path to output JSON")

    listen_parser = subparsers.add_parser("listen", help="Evaluate how musically strong/coherent a track appears")
    listen_parser.add_argument("track", help="Path to track file")
    listen_parser.add_argument("--output", "-o", help="Path to output JSON")
    listen_parser.add_argument("--score-only", action="store_true", help="Print only the overall score (0-100) for quick scripting")

    compare_parser = subparsers.add_parser("compare-listen", help="Compare two listen reports, audio files, or rendered outputs")
    compare_parser.add_argument("left", help="Left input: listen JSON, audio file, render manifest JSON, or render output directory")
    compare_parser.add_argument("right", help="Right input: listen JSON, audio file, render manifest JSON, or render output directory")
    compare_parser.add_argument("--output", "-o", help="Path to output comparison JSON")

    benchmark_parser = subparsers.add_parser("benchmark-listen", help="Round-robin benchmark multiple listen reports, audio files, or rendered outputs")
    benchmark_parser.add_argument("inputs", nargs="+", help="Two or more inputs: listen JSON, audio file, render manifest JSON, or render output directory")
    benchmark_parser.add_argument("--output", "-o", help="Path to output benchmark JSON")

    listener_agent_parser = subparsers.add_parser("listener-agent", help="Gate multiple outputs so only promising survivors reach human listening")
    listener_agent_parser.add_argument("inputs", nargs="+", help="One or more inputs: listen JSON, audio file, render manifest JSON, or render output directory")
    listener_agent_parser.add_argument("--shortlist", type=int, default=3, help="Maximum number of survivors to recommend for human review")
    listener_agent_parser.add_argument("--output", "-o", help="Path to output listener-agent JSON")

    auto_shortlist_parser = subparsers.add_parser("auto-shortlist-fusion", help="Render several candidate fusions, gate them automatically, and only keep survivors for human listening")
    auto_shortlist_parser.add_argument("track1", help="Path to first track")
    auto_shortlist_parser.add_argument("track2", help="Path to second track")
    auto_shortlist_parser.add_argument("--output", "-o", help="Output directory (or report path inside one) for shortlist artifacts")
    auto_shortlist_parser.add_argument("--batch-size", type=int, default=AUTO_SHORTLIST_DEFAULT_BATCH_SIZE, help="How many candidate variants to generate")
    auto_shortlist_parser.add_argument("--shortlist", type=int, default=AUTO_SHORTLIST_DEFAULT_SHORTLIST, help="Maximum number of survivors to surface for human review")
    auto_shortlist_parser.add_argument("--variant-mode", default="safe", help="Variant generation mode (currently: safe)")
    auto_shortlist_parser.add_argument("--arrangement-mode", default="baseline", choices=["baseline", "adaptive"], help="Arrangement planning mode")
    auto_shortlist_parser.add_argument("--keep-non-survivors", action="store_true", help="Do not delete rejected/non-shortlisted candidate run folders after gating")

    feedback_learning_parser = subparsers.add_parser("distill-feedback-learning", help="Distill stored human feedback into a stable learning snapshot used by shortlist ranking")
    feedback_learning_parser.add_argument("--output", "-o", help="Output JSON path for the distilled learning snapshot")

    closed_loop_parser = subparsers.add_parser("closed-loop", help="Run a bounded listener-driven improvement loop for one fusion pair")
    closed_loop_parser.add_argument("song_a", help="Path to parent song A")
    closed_loop_parser.add_argument("song_b", help="Path to parent song B")
    closed_loop_parser.add_argument("references", nargs="+", help="One or more good reference inputs")
    closed_loop_parser.add_argument("--output", "-o", help="Directory (or JSON path inside a directory) for closed-loop artifacts/report")
    closed_loop_parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of loop iterations")
    closed_loop_parser.add_argument("--quality-gate", type=float, default=85.0, help="Stop once the candidate clears this overall score")
    closed_loop_parser.add_argument("--plateau-limit", type=int, default=2, help="Stop after this many non-improving iterations")
    closed_loop_parser.add_argument("--min-improvement", type=float, default=0.5, help="Minimum score gain required to reset plateau detection")
    closed_loop_parser.add_argument("--target-score", type=float, default=99.0, help="Long-term aspirational target score for the feedback brief")
    closed_loop_parser.add_argument("--change-command", help="Optional direct command template used to change code between iterations")
    closed_loop_parser.add_argument("--test-command", help="Optional direct command template used to validate changes between iterations")
    closed_loop_parser.add_argument("--change-dispatch", help="Optional JSON dispatch spec for the change step")
    closed_loop_parser.add_argument("--test-dispatch", help="Optional JSON dispatch spec for the test step")

    fus_parser = subparsers.add_parser("fusion", help="Render a first-pass fused audio prototype")
    fus_parser.add_argument("track1", help="Path to first track")
    fus_parser.add_argument("track2", help="Path to second track")
    fus_parser.add_argument("--genre", "-g", help="Target genre for fusion (accepted but not yet applied)")
    fus_parser.add_argument("--bpm", "-b", type=int, help="Target BPM (accepted but not yet applied)")
    fus_parser.add_argument("--key", "-k", help="Target musical key")
    fus_parser.add_argument("--output", "-o", help="Output directory for render artifacts")
    fus_parser.add_argument("--arrangement-mode", default="baseline", choices=["baseline", "adaptive"], help="Arrangement planning mode")

    proto_parser = subparsers.add_parser("prototype", help="Generate first-pass two-song prototype artifacts")
    proto_parser.add_argument("song_a", help="Path to parent song A")
    proto_parser.add_argument("song_b", help="Path to parent song B")
    proto_parser.add_argument("--output-dir", "-o", required=True, help="Directory to write prototype artifacts")
    proto_parser.add_argument("--stems-dir", help="Optional directory for stem outputs")
    proto_parser.add_argument("--arrangement-mode", default="baseline", choices=["baseline", "adaptive"], help="Arrangement planning mode")

    doctor_parser = subparsers.add_parser("doctor", help="Check whether local dependencies are installed")
    doctor_parser.add_argument("--output", "-o", help="Optional path to write dependency report JSON")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "generate":
            return generate(args.genre, args.bpm, args.key, args.output)
        if args.command == "analyze":
            return analyze(args.track, args.detailed, args.output)
        if args.command == "fusion":
            return fusion(args.track1, args.track2, args.genre, args.bpm, args.key, args.output, args.arrangement_mode)
        if args.command == "prototype":
            return prototype(args.song_a, args.song_b, args.output_dir, args.stems_dir, args.arrangement_mode)
        if args.command == "listen":
            return listen(args.track, args.output, args.score_only)
        if args.command == "compare-listen":
            return compare_listen(args.left, args.right, args.output)
        if args.command == "benchmark-listen":
            return benchmark_listen(args.inputs, args.output)
        if args.command == "listener-agent":
            return listener_agent(args.inputs, args.output, args.shortlist)
        if args.command == "auto-shortlist-fusion":
            return auto_shortlist_fusion(
                args.track1,
                args.track2,
                args.output,
                batch_size=args.batch_size,
                shortlist=args.shortlist,


# ===== END ai_dj.py (argparse listen options) =====

# ===== BEGIN server.py (song-rater helpers + api route) =====

                    {
                        "id": song_id,
                        "title": full_path.stem,
                        "artist": rel_path.parts[0] if len(rel_path.parts) > 1 else "Unknown",
                        "file": str(rel_path),
                        "absolute_path": str(full_path),
                    }
                )
    songs.sort(key=lambda s: (s["artist"].lower(), s["title"].lower()))
    return songs


def resolve_song_path(song_id: str) -> Path | None:
    for song in load_songs():
        if song["id"] == song_id:
            return Path(song["absolute_path"])
    return None


def _component_score(report: dict, key: str) -> float:
    try:
        return float((report.get(key) or {}).get("score") or 0.0)
    except Exception:
        return 0.0


def _chart_calibrated_score(report: dict) -> float:
    """Heavier commercial calibration: reward chart-like traits, punish rough mixes hard."""
    overall = float(report.get("overall_score") or 0.0)
    song_likeness = _component_score(report, "song_likeness")
    mix_sanity = _component_score(report, "mix_sanity")
    structure = _component_score(report, "structure")
    coherence = _component_score(report, "coherence")
    groove = _component_score(report, "groove")
    energy_arc = _component_score(report, "energy_arc")
    transition = _component_score(report, "transition")

    score = overall

    # Upside: aggressively reward commercial-readability cues.
    score += max(0.0, song_likeness - 75.0) * 0.35
    score += max(0.0, mix_sanity - 72.0) * 0.20
    score += max(0.0, structure - 75.0) * 0.15
    score += max(0.0, coherence - 75.0) * 0.10
    score += max(0.0, energy_arc - 70.0) * 0.10
    score += max(0.0, transition - 65.0) * 0.05
    score += max(0.0, groove - 60.0) * 0.05

    # Downside: heavily punish low-quality amateur-ish output.
    score -= max(0.0, 45.0 - groove) * 0.55
    score -= max(0.0, 45.0 - mix_sanity) * 0.45
    score -= max(0.0, 42.0 - song_likeness) * 0.60
    score -= max(0.0, 40.0 - overall) * 0.40

    # Bonus / malus anchors for practical chart-like calibration.
    if song_likeness >= 85.0 and mix_sanity >= 80.0 and structure >= 85.0:
        score += 6.0
    if groove < 35.0 and mix_sanity < 40.0:
        score -= 12.0

    return round(max(0.0, min(100.0, score)), 1)


def _run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=10,
    )
            "fuse_ui": "/",
            "debug_ui": "/debug",
            "status_ui": "/status",
            "listener_agent_api": "/api/listener-agent",
            "auto_shortlist_api": "/api/auto-shortlist-fusion",
            "closed_loop_api": "/api/closed-loop",
            "benchmark_spec_api": "/api/benchmark-spec",
        },
    }


@app.route("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "simple_fuse.html")


@app.route("/song-rater")
def song_rater_page():
    return send_from_directory(TEMPLATES_DIR, "song_rater.html")


@app.route("/debug")
def debug_index():
    return send_from_directory(TEMPLATES_DIR, "prototype_debug.html")


@app.route("/status")
def status_page():
    return send_from_directory(TEMPLATES_DIR, "status.html")


@app.route("/updates")
def updates_page():
    return send_from_directory(TEMPLATES_DIR, "updates.html")


@app.route("/api/songs")
def list_songs():
    songs = load_songs()
    return jsonify({"status": "success", "songs": songs, "count": len(songs)})


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/api/rate-song", methods=["POST"])
def api_rate_song():
    song = request.files.get("song")
    if song is None:
        return jsonify({"status": "error", "error": "One audio file is required."}), 400

    allowed = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    ext = Path(song.filename or "song").suffix.lower()
    if ext not in allowed:
        return jsonify({"status": "error", "error": "Only MP3/WAV/FLAC/M4A/AAC files are supported."}), 400

    run_id = f"song_rating_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    upload_dir = UPLOADS_DIR / run_id
    report_dir = RUNS_DIR / "song_ratings" / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    song_path = upload_dir / f"song{ext}"
    song.save(song_path)

    try:
        dna = analyze_audio_file(str(song_path))
        report = evaluate_song(dna).to_dict()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"Song rating failed: {exc}"}), 500

    overall_score = round(float(report.get("overall_score") or 0.0), 1)
    chart_calibrated_score = _chart_calibrated_score(report)
    report["chart_calibrated_score"] = chart_calibrated_score

    report_path = report_dir / "listen_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return jsonify(
        {
            "status": "success",
            "song_name": Path(song.filename or "song").name,
            "overall_score": overall_score,
            "chart_calibrated_score": chart_calibrated_score,
            "verdict": report.get("verdict"),
            "components": {key: report.get(key, {}) for key in ("structure", "groove", "energy_arc", "transition", "coherence", "mix_sanity", "song_likeness")},
            "report_path": str(report_path),
            "run_id": run_id,
        }
    )


@app.route("/fuse", methods=["POST"])
def start_simple_fuse_job():
    song_a = request.files.get("song_a")
    song_b = request.files.get("song_b")

    if song_a is None or song_b is None:
        return jsonify({"status": "error", "error": "Two audio files are required."}), 400

    allowed = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    ext_a = Path(song_a.filename or "song_a").suffix.lower()
    ext_b = Path(song_b.filename or "song_b").suffix.lower()


# ===== END server.py (song-rater helpers + api route) =====
