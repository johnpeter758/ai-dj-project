from __future__ import annotations

import librosa
import numpy as np


def _tempo_scalar(tempo) -> float:
    values = np.asarray(tempo, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(values[0])


def beat_grid_metrics(beat_times: np.ndarray, duration: float | None = None) -> dict[str, float]:
    beat_times = np.asarray(beat_times, dtype=float).reshape(-1)
    if beat_times.size < 2:
        return {
            "count": float(beat_times.size),
            "median_interval": 0.0,
            "interval_cv": 1.0,
            "coverage": 0.0,
            "confidence": 0.0,
        }

    intervals = np.diff(beat_times)
    valid = intervals[(intervals > 0.2) & (intervals < 2.0)]
    if valid.size == 0:
        return {
            "count": float(beat_times.size),
            "median_interval": 0.0,
            "interval_cv": 1.0,
            "coverage": 0.0,
            "confidence": 0.0,
        }

    median_interval = float(np.median(valid))
    cv = float(np.std(valid) / (np.mean(valid) + 1e-8))
    regularity = max(0.0, min(1.0, 1.0 - cv))
    support = max(0.0, min(1.0, valid.size / 16.0))

    if duration and duration > 0.0:
        coverage = max(0.0, min(1.0, float(beat_times[-1] - beat_times[0]) / float(duration)))
    else:
        coverage = 1.0

    confidence = regularity * (0.35 + 0.65 * coverage) * (0.25 + 0.75 * support)

    return {
        "count": float(beat_times.size),
        "median_interval": median_interval,
        "interval_cv": cv,
        "coverage": coverage,
        "confidence": float(max(0.0, min(1.0, confidence))),
    }


# Backward-compatible private alias for internal callers/tests that may still import it.
_defunct_name = beat_grid_metrics


def _beat_grid_metrics(beat_times: np.ndarray, duration: float | None = None) -> dict[str, float]:
    return beat_grid_metrics(beat_times, duration=duration)


def detect_tempo(audio: np.ndarray, sample_rate: int, hop_length: int = 512) -> dict[str, float | list[float] | str]:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    beat_times = np.asarray(beat_times, dtype=float)

    duration = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0
    metrics = beat_grid_metrics(beat_times, duration=duration)
    bpm = _tempo_scalar(tempo)

    return {
        "bpm": bpm,
        "confidence": float(metrics["confidence"]),
        "beat_times": beat_times.tolist(),
        "method": "librosa",
    }
