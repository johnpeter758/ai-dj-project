from __future__ import annotations

import librosa
import numpy as np


def compute_energy_profile(audio: np.ndarray, sample_rate: int, hop_length: int = 512) -> dict:
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)

    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    beat_energy = np.interp(beat_times, times, rms) if len(beat_times) else np.array([])

    return {
        "frame_times": times.tolist(),
        "frame_rms": rms.tolist(),
        "beat_times": beat_times.tolist(),
        "beat_rms": beat_energy.tolist(),
        "summary": {
            "mean_rms": float(np.mean(rms)) if len(rms) else 0.0,
            "max_rms": float(np.max(rms)) if len(rms) else 0.0,
            "dynamic_range_rms": float(np.max(rms) - np.min(rms)) if len(rms) else 0.0,
        },
    }
