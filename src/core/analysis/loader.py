from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

DEFAULT_SAMPLE_RATE = 44100


def load_audio(path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE, mono: bool = True) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(path), sr=sample_rate, mono=mono)
    return audio, int(sr)


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return float(len(audio) / sample_rate)
