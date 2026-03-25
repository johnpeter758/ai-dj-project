from __future__ import annotations

import numpy as np

from src.core.render.spectral import apply_spectral_carve, compute_vocal_presence_mask, _fit_mask_to_shape, _smooth_mask_2d


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


def test_fit_mask_to_shape_resamples_instead_of_repeating_tiles():
    # A strict gradient should stay monotonic after resizing (tiling via np.resize breaks this).
    mask = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(3, 2)
    fit = _fit_mask_to_shape(mask, (9, 7))

    assert fit.shape == (9, 7)
    assert np.isfinite(fit).all()
    assert float(fit.min()) >= 0.0
    assert float(fit.max()) <= 1.0

    freq_trend = np.diff(fit.mean(axis=1))
    time_trend = np.diff(fit.mean(axis=0))
    assert np.all(freq_trend >= -1e-4)
    assert np.all(time_trend >= -1e-4)


def test_smooth_mask_2d_preserves_flat_mask_at_edges():
    # Edge-preserving smoothing should keep a flat mask flat (no boundary droop).
    flat = np.full((11, 13), 0.9, dtype=np.float32)
    smoothed = _smooth_mask_2d(flat, freq_bins=5, time_frames=7)

    assert smoothed.shape == flat.shape
    assert np.isfinite(smoothed).all()
    assert np.allclose(smoothed, flat, atol=1e-6)
