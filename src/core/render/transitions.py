from __future__ import annotations

import numpy as np

from .manifest import ResolverConfig, TransitionType


def equal_power_fade_in(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x = np.linspace(0.0, 1.0, n, endpoint=True)
    return np.sin(x * np.pi / 2.0).astype(np.float32)


def equal_power_fade_out(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x = np.linspace(0.0, 1.0, n, endpoint=True)
    return np.cos(x * np.pi / 2.0).astype(np.float32)


def transition_overlap_beats(kind: TransitionType | None, config: ResolverConfig | None = None, stretch_ratio: float = 1.0) -> float:
    if kind is None or kind == "cut":
        return 0.0
    config = config or ResolverConfig()
    beats = min(config.blend_beats_max, {"blend": 8.0, "swap": 4.0, "lift": 4.0, "drop": 2.0}.get(kind, 0.0))
    if stretch_ratio < 0.75 or stretch_ratio > 1.25:
        return min(beats, 2.0)
    return beats


def transition_overlap_seconds(kind: TransitionType | None, bpm: float, config: ResolverConfig | None = None, stretch_ratio: float = 1.0) -> float:
    if kind is None or kind == "cut":
        return 0.0
    safe_bpm = max(float(bpm), 1e-6)
    beats = transition_overlap_beats(kind, config=config, stretch_ratio=stretch_ratio)
    return beats * 60.0 / safe_bpm


def incoming_gain_db(kind: TransitionType | None) -> float:
    if kind in {"blend", "swap", "lift", "drop"}:
        return -1.5
    return 0.0
