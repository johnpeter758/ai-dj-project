from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np

from ..analysis.models import SongDNA
from .models import ListenReport, ListenSubscore


TRANSITION_ANALYSIS_VERSION = "0.4.0"


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
    transition_risk_rows: list[dict[str, Any]] = []

    for section in sections:
        allowed_overlap = bool(section.get('allowed_overlap', False))
        overlap_beats = float(section.get('overlap_beats_max', 0.0) or 0.0)
        fg = section.get('foreground_owner')
        bg = section.get('background_owner')
        low = section.get('low_end_owner')
        vocal_policy = str(section.get('vocal_policy') or '')
        transition_in = section.get('transition_in')
        transition_out = section.get('transition_out')
        stretch_ratio = abs(float(section.get('stretch_ratio', 1.0) or 1.0) - 1.0)
        collapse_if_conflict = bool(section.get('collapse_if_conflict', False))

        overlap_risk = 0.0
        if allowed_overlap:
            overlap_sections += 1
            explicit_overlap_beats += overlap_beats
            overlap_risk += min(overlap_beats / 8.0, 1.0)
            if overlap_beats >= 4.0:
                long_overlap_sections += 1

        if fg and bg and fg == bg:
            multi_owner_conflicts += 1
            overlap_risk += 0.7
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

    owner_orders = {}
    for order in work_orders:
        idx = int(order.get('section_index', -1))
        owner_orders.setdefault(idx, []).append(order)

    for idx, orders in owner_orders.items():
        low_end_owners = {o.get('parent_id') for o in orders if o.get('low_end_state') == 'owner'}
        foreground_owners = {o.get('parent_id') for o in orders if o.get('foreground_state') == 'owner'}
        lead_vocal_owners = {o.get('parent_id') for o in orders if o.get('vocal_state') in {'lead_only', 'lead'} }
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

    section_count = max(len(sections), 1)
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
        'warning_count': len(warnings),
        'fallback_count': len(fallbacks),
    }
    return {
        'aggregate_metrics': aggregate,
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
    score = round(100.0 * (0.45 * section_score + 0.35 * phrase_score + 0.20 * novelty_score), 1)

    evidence.append(f"detected {section_count} coarse sections, {phrase_count} phrase boundaries, {novelty_count} novelty boundaries")
    if section_count <= 2:
        fixes.append("Increase structural certainty so the planner is not forced into coarse whole-song windows.")
    if phrase_count < 6:
        fixes.append("Improve beat/downbeat and phrase extraction to create more musically legal planning windows.")
    summary = "Structure is reasonably segmented for planning." if score >= 70 else "Structure segmentation is still too coarse for strong planning."
    return ListenSubscore(score=score, summary=summary, evidence=evidence, fixes=fixes)


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
    stability = _clamp01(1.0 - min(cv / 0.12, 1.0))
    tempo_consistency = _score_band(median_interval, 0.35, 0.75, 0.25)
    score = round(100.0 * (0.7 * stability + 0.3 * tempo_consistency), 1)

    evidence.append(f"{len(beat_times)} beats detected; median beat interval {median_interval:.3f}s; interval variation {cv:.3f}")
    if cv > 0.10:
        fixes.append("Stabilize beat/downbeat tracking or use a stronger bar-grid estimator for planning and evaluation.")
    summary = "Groove grid looks reasonably stable." if score >= 70 else "Groove grid looks unstable or under-detected."
    return ListenSubscore(score=score, summary=summary, evidence=evidence, fixes=fixes)


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
    energy_confidence = float(derived.get("energy_confidence") or 0.0)

    contrast_score = _clamp01((contrast - 0.10) / 0.35)
    late_lift_score = _clamp01((late_lift - 0.04) / 0.28)
    peak_late_bonus = _score_band(peak_position, 0.45, 0.95, 0.25)
    trajectory_score = 0.65 * late_lift_score + 0.35 * peak_late_bonus
    payoff_score = _clamp01(0.50 * payoff_strength + 0.30 * hook_strength + 0.20 * hook_repetition)
    stability_score = (
        0.65 * _score_band(step_median, 0.015, 0.14, 0.08)
        + 0.35 * _score_band(step_p90, 0.04, 0.32, 0.12)
    )

    raw_score = (
        0.30 * contrast_score
        + 0.30 * trajectory_score
        + 0.25 * payoff_score
        + 0.15 * stability_score
    )
    confidence_gate = 0.70 + 0.30 * _clamp01(energy_confidence)
    final_norm = _clamp01(raw_score) * confidence_gate
    score = round(100.0 * final_norm, 1)

    evidence.append(
        f"bar-energy contrast {contrast:.3f}; quartile means early={early_mean:.3f}, mid={mid_mean:.3f}, late={late_mean:.3f}; late lift {late_lift:.3f}"
    )
    evidence.append(
        f"peak energy occurs at {peak_position * 100.0:.0f}% of song duration; payoff {payoff_strength:.3f}, hook {hook_strength:.3f}, hook repetition {hook_repetition:.3f}, confidence {energy_confidence:.3f}"
    )
    evidence.append(f"bar-energy step stability median {step_median:.3f}, p90 {step_p90:.3f}")

    if contrast < 0.12:
        fixes.append("Increase macro-dynamic contrast so payoff sections separate clearly from setup sections.")
    if late_lift < 0.04 or peak_position < 0.35:
        fixes.append("Rework the energy journey so the strongest material arrives later instead of peaking too early.")
    if payoff_score < 0.35:
        fixes.append("Strengthen recurring payoff windows; current energy rises without a convincing chorus/drop identity.")
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
    }


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
        aggregate_metrics['manifest_overlap_section_ratio'] = manifest_metrics['overlap_section_ratio']
        aggregate_metrics['manifest_avg_overlap_beats'] = manifest_metrics['avg_overlap_beats']
        aggregate_metrics['manifest_multi_owner_conflict_ratio'] = manifest_metrics['multi_owner_conflict_ratio']
        aggregate_metrics['manifest_lead_conflict_ratio'] = manifest_metrics['lead_conflict_ratio']
        aggregate_metrics['manifest_crowding_ratio'] = manifest_metrics['crowding_ratio']
        aggregate_metrics['manifest_seam_risk_ratio'] = manifest_metrics['seam_risk_ratio']
        aggregate_metrics['manifest_collapse_ratio'] = manifest_metrics['collapse_ratio']
        manifest_penalty = (
            0.28 * manifest_metrics['multi_owner_conflict_ratio']
            + 0.18 * manifest_metrics['lead_conflict_ratio']
            + 0.16 * manifest_metrics['crowding_ratio']
            + 0.18 * manifest_metrics['seam_risk_ratio']
            + 0.12 * manifest_metrics['avg_overlap_beats'] / 8.0
            + 0.08 * manifest_metrics['long_overlap_ratio']
        )
        score = round(max(0.0, score - 22.0 * manifest_penalty), 1)
        evidence.append(
            "manifest seam risks — "
            f"overlap ratio {manifest_metrics['overlap_section_ratio']:.2f}, "
            f"avg overlap beats {manifest_metrics['avg_overlap_beats']:.2f}, "
            f"owner conflicts {manifest_metrics['multi_owner_conflict_ratio']:.2f}, "
            f"lead conflicts {manifest_metrics['lead_conflict_ratio']:.2f}, "
            f"crowding {manifest_metrics['crowding_ratio']:.2f}, "
            f"stretch/seam risk {manifest_metrics['seam_risk_ratio']:.2f}"
        )
        if manifest_metrics['multi_owner_conflict_ratio'] > 0.0:
            fixes.append("Resolved manifest still contains explicit multi-owner conflicts; collapse those seams to one low-end owner and one foreground owner.")
        if manifest_metrics['lead_conflict_ratio'] > 0.0:
            fixes.append("Resolved manifest still permits competing lead ownership in some sections; keep lead-vocal ownership singular through swaps.")
        if manifest_metrics['crowding_ratio'] > 0.34 or manifest_metrics['avg_overlap_beats'] > 3.0:
            fixes.append("Reduce manifest overlap windows and donor density; the explicit plan is already too crowded before audio rendering artifacts enter the picture.")
        if manifest_metrics['seam_risk_ratio'] > 0.34:
            fixes.append("High-stretch or transition-heavy sections in the manifest are likely seam risks; prefer phrase-safer windows or shorter swaps.")

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
        0.24 * clutter["overcrowded_overlap_risk"]
        + 0.22 * clutter["low_end_conflict_risk"]
        + 0.20 * clutter["foreground_overload_risk"]
        + 0.18 * clutter["overcompressed_flatness_risk"]
        + 0.16 * clutter["vocal_competition_risk"]
    )
    if manifest_details:
        manifest_metrics = manifest_details['aggregate_metrics']
        ownership_penalty += (
            0.18 * manifest_metrics['multi_owner_conflict_ratio']
            + 0.16 * manifest_metrics['lead_conflict_ratio']
            + 0.14 * manifest_metrics['crowding_ratio']
            + 0.12 * manifest_metrics['avg_overlap_beats'] / 8.0
            + 0.10 * manifest_metrics['seam_risk_ratio']
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
        f"vocals {clutter['vocal_competition_risk']:.2f}"
    )
    if manifest_details:
        manifest_metrics = manifest_details['aggregate_metrics']
        evidence.append(
            "manifest ownership risks — "
            f"owner conflicts {manifest_metrics['multi_owner_conflict_ratio']:.2f}, "
            f"lead conflicts {manifest_metrics['lead_conflict_ratio']:.2f}, "
            f"crowding {manifest_metrics['crowding_ratio']:.2f}, "
            f"avg overlap beats {manifest_metrics['avg_overlap_beats']:.2f}, "
            f"collapse ratio {manifest_metrics['collapse_ratio']:.2f}"
        )
        if manifest_metrics['multi_owner_conflict_ratio'] > 0.0:
            fixes.append("The resolved manifest still assigns conflicting owners inside at least one section; make low-end and foreground ownership explicit and singular.")
        if manifest_metrics['lead_conflict_ratio'] > 0.0:
            fixes.append("Lead-vocal ownership is still ambiguous in the manifest; avoid explicit dual-lead sections.")
        if manifest_metrics['crowding_ratio'] > 0.34 or manifest_metrics['avg_overlap_beats'] > 3.0:
            fixes.append("The manifest itself is crowding the arrangement with long or dense overlaps; shorten the donor window before worrying about mastering.")

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
    summary = "Mix sanity looks broadly usable." if score >= 70 else "Mix sanity indicators suggest congestion, ownership conflicts, or flatness."
    details = {"ownership_clutter_metrics": clutter}
    if manifest_details:
        details['manifest_metrics'] = manifest_details
    return ListenSubscore(score=score, summary=summary, evidence=evidence, fixes=fixes, details=details)


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

    overall = round(
        0.20 * structure.score
        + 0.16 * groove.score
        + 0.18 * energy_arc.score
        + 0.14 * transition.score
        + 0.16 * coherence.score
        + 0.16 * mix_sanity.score,
        1,
    )

    reasons = [
        f"Structure: {structure.summary}",
        f"Groove: {groove.summary}",
        f"Energy arc: {energy_arc.summary}",
        f"Transitions: {transition.summary}",
        f"Coherence: {coherence.summary}",
        f"Mix sanity: {mix_sanity.summary}",
    ]
    fixes = []
    for part in [structure, groove, energy_arc, transition, coherence, mix_sanity]:
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
        verdict=_verdict(overall),
        top_reasons=reasons,
        top_fixes=deduped_fixes[:8],
    )
    report.analysis_version = TRANSITION_ANALYSIS_VERSION
    return report
