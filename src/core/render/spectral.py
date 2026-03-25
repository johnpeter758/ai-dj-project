from __future__ import annotations

import numpy as np
import librosa


def _resample_axis_linear(arr: np.ndarray, target_size: int, axis: int) -> np.ndarray:
    if arr.shape[axis] == target_size:
        return arr.astype(np.float32, copy=False)
    if target_size <= 1:
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, 1)
        return arr[tuple(slicer)].astype(np.float32, copy=False)

    src_size = arr.shape[axis]
    src_pos = np.linspace(0.0, 1.0, src_size, dtype=np.float32)
    dst_pos = np.linspace(0.0, 1.0, target_size, dtype=np.float32)

    moved = np.moveaxis(arr, axis, 0).astype(np.float32, copy=False)
    flat = moved.reshape(src_size, -1)
    out_flat = np.empty((target_size, flat.shape[1]), dtype=np.float32)
    for i in range(flat.shape[1]):
        out_flat[:, i] = np.interp(dst_pos, src_pos, flat[:, i]).astype(np.float32)
    out = out_flat.reshape((target_size, *moved.shape[1:]))
    return np.moveaxis(out, 0, axis)


def _fit_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    out = np.asarray(mask, dtype=np.float32)
    if out.ndim != 2:
        out = np.atleast_2d(out)
    out = _resample_axis_linear(out, target_shape[0], axis=0)
    out = _resample_axis_linear(out, target_shape[1], axis=1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _smooth_mask_2d(mask: np.ndarray, freq_bins: int = 3, time_frames: int = 5) -> np.ndarray:
    out = mask.astype(np.float32, copy=True)
    if out.size == 0:
        return out

    if freq_bins > 1:
        kernel_f = np.ones(freq_bins, dtype=np.float32) / np.float32(freq_bins)
        out = np.apply_along_axis(lambda col: np.convolve(col, kernel_f, mode='same'), 0, out)

    if time_frames > 1:
        kernel_t = np.ones(time_frames, dtype=np.float32) / np.float32(time_frames)
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel_t, mode='same'), 1, out)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_vocal_presence_mask(
    inst_mono: np.ndarray,
    vox_mono: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int = 512,
    smooth_frames: int = 5,
) -> np.ndarray:
    if inst_mono.size == 0 or vox_mono.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    length = min(inst_mono.shape[-1], vox_mono.shape[-1])
    inst = inst_mono[:length].astype(np.float32)
    vox = vox_mono[:length].astype(np.float32)
    inst_spec = np.abs(librosa.stft(inst, n_fft=n_fft, hop_length=hop))
    vox_spec = np.abs(librosa.stft(vox, n_fft=n_fft, hop_length=hop))
    raw = vox_spec / (vox_spec + inst_spec + 1e-6)
    if smooth_frames > 1:
        raw = _smooth_mask_2d(raw, freq_bins=3, time_frames=smooth_frames)
    return np.clip(raw.astype(np.float32), 0.0, 1.0)


def apply_spectral_carve(
    inst_stereo: np.ndarray,
    vocal_mask: np.ndarray,
    sr: int,
    carve_db: float = 3.0,
    carve_lo_hz: float = 150.0,
    carve_hi_hz: float = 6000.0,
    n_fft: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    if inst_stereo.size == 0:
        return inst_stereo.astype(np.float32)
    out_channels: list[np.ndarray] = []
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = ((freqs >= carve_lo_hz) & (freqs <= carve_hi_hz)).astype(np.float32)[:, np.newaxis]
    min_gain = np.float32(10 ** (-float(carve_db) / 20.0))
    for ch in range(inst_stereo.shape[0]):
        spec = librosa.stft(inst_stereo[ch].astype(np.float32), n_fft=n_fft, hop_length=hop)
        mask = _fit_mask_to_shape(vocal_mask, spec.shape)
        mask = _smooth_mask_2d(mask, freq_bins=3, time_frames=5)
        max_cut = np.float32(1.0 - min_gain)
        masked_cut = 1.0 - (max_cut * mask * band)
        masked_cut = np.clip(masked_cut, min_gain, 1.0).astype(np.float32)
        carved = spec * masked_cut
        out = librosa.istft(carved, hop_length=hop, length=inst_stereo.shape[1])
        out_channels.append(out.astype(np.float32))
    return np.vstack(out_channels).astype(np.float32)
