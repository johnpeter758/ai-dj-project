from __future__ import annotations

import librosa
import numpy as np

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_TO_CAMELOT = {
    ("C", "major"): "8B", ("C", "minor"): "5A",
    ("C#", "major"): "3B", ("C#", "minor"): "12A",
    ("D", "major"): "10B", ("D", "minor"): "7A",
    ("D#", "major"): "5B", ("D#", "minor"): "2A",
    ("E", "major"): "12B", ("E", "minor"): "9A",
    ("F", "major"): "7B", ("F", "minor"): "4A",
    ("F#", "major"): "2B", ("F#", "minor"): "11A",
    ("G", "major"): "9B", ("G", "minor"): "6A",
    ("G#", "major"): "4B", ("G#", "minor"): "1A",
    ("A", "major"): "11B", ("A", "minor"): "8A",
    ("A#", "major"): "6B", ("A#", "minor"): "3A",
    ("B", "major"): "1B", ("B", "minor"): "10A",
}
KRUHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(audio: np.ndarray, sample_rate: int) -> dict[str, float | str | list[float]]:
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-8)

    best_score = -np.inf
    best_key = "C"
    best_mode = "major"

    for root_idx in range(12):
        rotated = np.roll(chroma_norm, -root_idx)
        major_corr = float(np.corrcoef(rotated, KRUHANSL_MAJOR)[0, 1])
        minor_corr = float(np.corrcoef(rotated, KRUHANSL_MINOR)[0, 1])
        if major_corr > best_score:
            best_score = major_corr
            best_key = NOTES[root_idx]
            best_mode = "major"
        if minor_corr > best_score:
            best_score = minor_corr
            best_key = NOTES[root_idx]
            best_mode = "minor"

    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
    camelot = KEY_TO_CAMELOT.get((best_key, best_mode), "8B")

    return {
        "tonic": best_key,
        "mode": best_mode,
        "camelot": camelot,
        "confidence": float(confidence),
        "method": "krumhansl_correlation",
        "chroma": chroma_mean.tolist(),
    }
