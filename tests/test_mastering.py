from __future__ import annotations

import numpy as np

from src.core.render.mastering import bpm_synced_glue_compress, lookahead_envelope_limit, lufs_normalize


def test_bpm_synced_glue_compress_preserves_shape_and_is_finite():
    audio = np.vstack([
        np.linspace(-0.8, 0.8, 44100, dtype=np.float32),
        np.linspace(0.8, -0.8, 44100, dtype=np.float32),
    ])
    out = bpm_synced_glue_compress(audio, sr=44100, bpm=128.0)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()


def test_lufs_normalize_returns_same_shape():
    t = np.linspace(0.0, 1.0, 44100, endpoint=False, dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    audio = np.vstack([tone, tone]).astype(np.float32)
    out = lufs_normalize(audio, target_lufs=-12.0, sr=44100)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()


def test_lookahead_envelope_limit_catches_short_transient_without_crushing_body():
    sr = 44100
    audio = np.full((2, sr // 2), 0.25, dtype=np.float32)
    spike_start = 5000
    spike_len = 32
    audio[:, spike_start: spike_start + spike_len] = 1.35

    limited = lookahead_envelope_limit(audio, sr=sr, ceiling_db=-1.2, attack_ms=1.5, release_ms=50.0, lookahead_ms=1.5)
    ceiling_lin = 10 ** (-1.2 / 20.0)

    assert limited.shape == audio.shape
    assert np.max(np.abs(limited[:, spike_start: spike_start + spike_len])) <= ceiling_lin * 1.02
    body_region = limited[:, spike_start + 2000: spike_start + 6000]
    assert float(np.mean(np.abs(body_region))) > 0.20
    assert np.isfinite(limited).all()
