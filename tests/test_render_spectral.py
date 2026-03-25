from __future__ import annotations

import numpy as np

from src.core.render.spectral import apply_spectral_carve, compute_vocal_presence_mask


def test_compute_vocal_presence_mask_is_bounded_and_nonempty():
    sr = 44100
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    inst = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    vox = 0.2 * np.sin(2 * np.pi * 1000.0 * t)
    mask = compute_vocal_presence_mask(inst, vox, sr)
    assert mask.ndim == 2
    assert mask.size > 0
    assert np.isfinite(mask).all()
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_apply_spectral_carve_preserves_shape_and_reduces_masked_stft_energy():
    import librosa

    sr = 44100
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    inst_mono = 0.35 * np.sin(2 * np.pi * 1000.0 * t) + 0.2 * np.sin(2 * np.pi * 220.0 * t)
    vox = 0.3 * np.sin(2 * np.pi * 1200.0 * t)
    stereo = np.vstack([inst_mono, inst_mono]).astype(np.float32)
    mask = compute_vocal_presence_mask(inst_mono, vox, sr)
    carved = apply_spectral_carve(stereo, mask, sr, carve_db=6.0, carve_lo_hz=500.0, carve_hi_hz=2500.0)
    assert carved.shape == stereo.shape
    assert np.isfinite(carved).all()

    orig_spec = np.abs(librosa.stft(stereo[0], n_fft=2048, hop_length=512))
    carved_spec = np.abs(librosa.stft(carved[0], n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= 500.0) & (freqs <= 2500.0)
    weighted = mask[band]
    original_mid_energy = float(np.mean(orig_spec[band] * weighted))
    carved_mid_energy = float(np.mean(carved_spec[band] * weighted))
    assert carved_mid_energy < original_mid_energy


def test_apply_spectral_carve_handles_mask_shape_mismatch_without_nan():
    sr = 44100
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    inst_mono = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    stereo = np.vstack([inst_mono, inst_mono]).astype(np.float32)

    # Deliberately wrong shape to ensure renderer adapts mask to STFT dimensions.
    vocal_mask = np.full((8, 8), 0.9, dtype=np.float32)
    carved = apply_spectral_carve(stereo, vocal_mask, sr, carve_db=6.0)

    assert carved.shape == stereo.shape
    assert np.isfinite(carved).all()
