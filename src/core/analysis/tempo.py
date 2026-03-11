from __future__ import annotations

import librosa
import numpy as np


def detect_tempo(audio: np.ndarray, sample_rate: int, hop_length: int = 512) -> dict[str, float | list[float] | str]:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)

    confidence = 0.5
    if len(beat_times) >= 4:
        intervals = np.diff(beat_times)
        valid = intervals[(intervals > 0.2) & (intervals < 2.0)]
        if len(valid) >= 2:
            cv = float(np.std(valid) / (np.mean(valid) + 1e-8))
            confidence = max(0.0, min(1.0, 1.0 - cv))

    bpm = float(np.asarray(tempo).reshape(-1)[0])

    return {
        "bpm": bpm,
        "confidence": float(confidence),
        "beat_times": beat_times.tolist(),
        "method": "librosa",
    }
