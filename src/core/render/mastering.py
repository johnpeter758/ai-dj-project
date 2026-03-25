from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover - optional dependency fallback
    pyln = None


def bpm_synced_glue_compress(
    audio: np.ndarray,
    sr: int,
    bpm: float,
    threshold_db: float = -16.0,
    ratio: float = 1.8,
    makeup_db: float = 0.75,
) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    threshold_lin = 10 ** (threshold_db / 20.0)
    envelope = np.max(np.abs(audio), axis=0).astype(np.float32)
    over = np.maximum(envelope, threshold_lin) / threshold_lin
    gain = np.ones_like(envelope, dtype=np.float32)
    mask = envelope > threshold_lin
    if np.any(mask):
        compressed = np.power(over[mask], -(1.0 - 1.0 / max(ratio, 1.0)))
        gain[mask] = compressed.astype(np.float32)

    beat_interval_sec = 60.0 / max(float(bpm), 1e-6)
    release_sec = float(np.clip(beat_interval_sec * 0.65, 0.08, 0.24))
    window = max(64, int(round(release_sec * sr)))
    kernel = np.ones(window, dtype=np.float32) / np.float32(window)
    smoothed = np.convolve(gain, kernel, mode="same")
    out = audio * smoothed[np.newaxis, :]
    return apply_gain_db(out, makeup_db)


def lookahead_envelope_limit(
    audio: np.ndarray,
    sr: int,
    ceiling_db: float = -1.2,
    attack_ms: float = 2.0,
    release_ms: float = 60.0,
    lookahead_ms: float = 1.5,
) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)

    ceiling_lin = np.float32(10 ** (ceiling_db / 20.0))
    envelope = np.max(np.abs(audio), axis=0).astype(np.float32)
    if not np.any(envelope > ceiling_lin):
        return audio.astype(np.float32)

    lookahead_samples = max(1, int(round(max(lookahead_ms, 0.0) * sr / 1000.0)))
    predicted = ndimage.maximum_filter1d(envelope, size=lookahead_samples, mode="nearest").astype(np.float32)
    target_gain = np.minimum(1.0, ceiling_lin / np.maximum(predicted, 1e-6)).astype(np.float32)

    attack_samples = max(sr * max(attack_ms, 0.1) / 1000.0, 1.0)
    attack_coeff = np.float32(math.exp(-1.0 / attack_samples))
    release_coeff = np.float32(math.exp(-1.0 / max(sr * max(release_ms, attack_ms, 0.1) / 1000.0, 1.0)))
    smoothed_gain = np.ones_like(target_gain, dtype=np.float32)
    current_gain = np.float32(1.0)
    for idx, desired_gain in enumerate(target_gain):
        if desired_gain < current_gain:
            # Attack quickly for short transients while preventing gain
            # undershoot below the instantaneous target.
            current_gain = max(current_gain * attack_coeff, desired_gain)
        else:
            current_gain = desired_gain + release_coeff * (current_gain - desired_gain)
        smoothed_gain[idx] = current_gain

    return (audio * smoothed_gain[np.newaxis, :]).astype(np.float32)


def lufs_normalize(audio: np.ndarray, target_lufs: float = -12.0, sr: int = 44100) -> np.ndarray:
    if audio.size == 0 or pyln is None:
        return audio.astype(np.float32)
    mono = np.mean(audio, axis=0).astype(np.float32)
    try:
        meter = pyln.Meter(sr)
        loudness = float(meter.integrated_loudness(mono))
    except Exception:
        return audio.astype(np.float32)
    if not math.isfinite(loudness) or loudness < -70.0:
        return audio.astype(np.float32)
    gain_db = target_lufs - loudness
    return apply_gain_db(audio, gain_db)


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if not np.isfinite(gain_db) or gain_db == 0.0:
        return audio.astype(np.float32)
    return (audio * np.float32(10 ** (gain_db / 20.0))).astype(np.float32)
