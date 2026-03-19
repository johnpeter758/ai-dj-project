from __future__ import annotations

from typing import Any

import librosa
import numpy as np


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    values = values.astype(float)
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 1e-6)
    return (values - low) / span


def _frame_times(frame_count: int, sr: int, hop_length: int) -> np.ndarray:
    if frame_count <= 0:
        return np.asarray([], dtype=float)
    return librosa.frames_to_time(np.arange(frame_count), sr=sr, hop_length=hop_length).astype(float)


def _melodic_contours(audio: np.ndarray, sr: int, *, hop_length: int = 512) -> dict[str, Any]:
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    contour = np.nan_to_num(f0.astype(float), nan=0.0)
    voiced = voiced_flag.astype(float) if voiced_flag is not None else np.zeros_like(contour)
    melodic_motion = np.abs(np.diff(np.log2(np.maximum(contour, 1e-6)))) if contour.size >= 2 else np.asarray([], dtype=float)
    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0
    contour_span_semitones = float(12.0 * (np.log2(max(np.max(contour), 1e-6)) - np.log2(max(np.min(contour[contour > 0]) if np.any(contour > 0) else 1e-6, 1e-6)))) if np.any(contour > 0) else 0.0
    contour_stability = 1.0 - _clamp01(float(np.median(melodic_motion)) / 0.5) if melodic_motion.size else 0.5
    return {
        "voiced_ratio": round(voiced_ratio, 3),
        "contour_span_semitones": round(contour_span_semitones, 3),
        "contour_stability": round(contour_stability, 3),
        "mean_voiced_confidence": round(float(np.mean(np.nan_to_num(voiced_prob, nan=0.0))) if voiced_prob is not None else 0.0, 3),
        "frame_times": _frame_times(contour.size, sr, hop_length).round(3).tolist(),
        "pitch_hz": contour.round(3).tolist(),
        "voiced_mask": voiced.astype(int).tolist(),
    }


def _motif_reuse(audio: np.ndarray, sr: int, *, hop_length: int = 512) -> dict[str, Any]:
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    if chroma.size == 0:
        return {"motif_repeat_strength": 0.0, "motif_cluster_count": 0, "motif_anchor_positions": []}
    similarity = librosa.segment.recurrence_matrix(chroma, mode="affinity", sym=True, width=3)
    repeated_frames = similarity.sum(axis=1)
    repeat_strength = _clamp01(float(np.mean(_safe_norm(repeated_frames))))
    anchors = np.where(_safe_norm(repeated_frames) > 0.7)[0]
    times = librosa.frames_to_time(anchors, sr=sr, hop_length=hop_length) if anchors.size else np.asarray([], dtype=float)
    cluster_count = int(max(0, np.sum(np.diff(anchors) > 6) + 1)) if anchors.size else 0
    return {
        "motif_repeat_strength": round(repeat_strength, 3),
        "motif_cluster_count": cluster_count,
        "motif_anchor_positions": times.round(3).tolist()[:64],
    }


def _drum_grammar(audio: np.ndarray, sr: int, *, hop_length: int = 512) -> dict[str, Any]:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length) if beat_frames.size else np.asarray([], dtype=float)
    intervals = np.diff(beat_times) if beat_times.size >= 2 else np.asarray([], dtype=float)
    pocket_stability = 1.0 - _clamp01(float(np.std(intervals) / max(np.mean(intervals), 1e-6) / 0.18)) if intervals.size else 0.5
    onset_pulse_lock = 1.0 - _clamp01(float(np.median(np.abs(np.diff(onset_env)))) / max(float(np.mean(onset_env) + 1e-6), 1e-6) / 2.5) if onset_env.size >= 2 else 0.5
    swing_hint = _clamp01(float(np.std(np.diff(beat_times[::2])) / max(np.mean(intervals), 1e-6) / 0.35)) if beat_times.size >= 4 and intervals.size else 0.0
    return {
        "estimated_tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 3) if np.size(tempo) else 0.0,
        "pocket_stability": round(pocket_stability, 3),
        "onset_pulse_lock": round(onset_pulse_lock, 3),
        "swing_hint": round(swing_hint, 3),
        "beat_times": beat_times.round(3).tolist()[:256],
    }


def _harmonic_function(audio: np.ndarray, sr: int, *, hop_length: int = 512) -> dict[str, Any]:
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    if chroma.size == 0:
        return {"tonal_clarity": 0.0, "harmonic_motion": 0.0, "cadence_strength": 0.0, "tonal_centers": []}
    chroma_norm = _safe_norm(chroma.mean(axis=1))
    tonal_clarity = float(np.max(chroma_norm))
    harmonic_motion = float(np.mean(np.abs(np.diff(chroma, axis=1)))) if chroma.shape[1] >= 2 else 0.0
    cadence_strength = _clamp01(float(np.percentile(chroma.max(axis=0), 90) - np.percentile(chroma.max(axis=0), 25)) / 0.35)
    tonal_centers = np.argsort(chroma_norm)[-3:][::-1].astype(int).tolist()
    return {
        "tonal_clarity": round(tonal_clarity, 3),
        "harmonic_motion": round(harmonic_motion, 3),
        "cadence_strength": round(cadence_strength, 3),
        "tonal_centers": tonal_centers,
    }


def _timbral_anchors(audio: np.ndarray, sr: int, *, hop_length: int = 512) -> dict[str, Any]:
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
    anchor_stability = 1.0 - _clamp01(float(np.std(centroid) / max(np.mean(centroid), 1e-6)) / 0.6) if centroid.size else 0.5
    palette_contrast = _clamp01(float(np.mean(contrast)) / 40.0) if contrast.size else 0.0
    flatness_mean = float(np.mean(flatness)) if flatness.size else 0.0
    return {
        "anchor_brightness_hz": round(float(np.mean(centroid)) if centroid.size else 0.0, 3),
        "anchor_bandwidth_hz": round(float(np.mean(bandwidth)) if bandwidth.size else 0.0, 3),
        "anchor_stability": round(anchor_stability, 3),
        "palette_contrast": round(palette_contrast, 3),
        "flatness_mean": round(flatness_mean, 3),
    }


def _tension_release(audio: np.ndarray, sr: int, *, hop_length: int = 512) -> dict[str, Any]:
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    if rms.size == 0:
        return {"tension_curve_strength": 0.0, "late_release_strength": 0.0, "surge_density": 0.0}
    composite = 0.45 * _safe_norm(rms) + 0.30 * _safe_norm(centroid) + 0.25 * _safe_norm(onset_env)
    split = max(1, composite.size // 2)
    early = composite[:split]
    late = composite[split:]
    late_release_strength = _clamp01(float(np.mean(late) - np.mean(early) + 0.5) / 1.0) if late.size else 0.0
    tension_curve_strength = _clamp01(float(np.percentile(composite, 90) - np.percentile(composite, 20)) / 0.65)
    surge_density = _clamp01(float(np.mean(np.diff(composite) > 0.08)) if composite.size >= 2 else 0.0)
    return {
        "tension_curve_strength": round(tension_curve_strength, 3),
        "late_release_strength": round(late_release_strength, 3),
        "surge_density": round(surge_density, 3),
    }


def analyze_musical_intelligence(audio: np.ndarray, sr: int) -> dict[str, Any]:
    melodic = _melodic_contours(audio, sr)
    motifs = _motif_reuse(audio, sr)
    drums = _drum_grammar(audio, sr)
    harmony = _harmonic_function(audio, sr)
    timbre = _timbral_anchors(audio, sr)
    tension = _tension_release(audio, sr)
    summary = {
        "melodic_identity_strength": round(
            0.55 * float(melodic.get("voiced_ratio", 0.0))
            + 0.45 * float(motifs.get("motif_repeat_strength", 0.0)),
            3,
        ),
        "rhythmic_confidence": round(
            0.6 * float(drums.get("pocket_stability", 0.0))
            + 0.4 * float(drums.get("onset_pulse_lock", 0.0)),
            3,
        ),
        "harmonic_confidence": round(
            0.6 * float(harmony.get("tonal_clarity", 0.0))
            + 0.4 * float(harmony.get("cadence_strength", 0.0)),
            3,
        ),
        "timbral_coherence": round(
            0.65 * float(timbre.get("anchor_stability", 0.0))
            + 0.35 * (1.0 - _clamp01(float(timbre.get("flatness_mean", 0.0)) / 0.5)),
            3,
        ),
        "tension_release_confidence": round(
            0.55 * float(tension.get("tension_curve_strength", 0.0))
            + 0.45 * float(tension.get("late_release_strength", 0.0)),
            3,
        ),
    }
    return {
        "summary": summary,
        "melodic_contours": melodic,
        "motif_reuse": motifs,
        "drum_grammar": drums,
        "harmonic_function": harmony,
        "timbral_anchors": timbre,
        "tension_release": tension,
    }
