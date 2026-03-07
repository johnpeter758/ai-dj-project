#!/usr/bin/env python3
"""
AI DJ Project - Audio Utilities

Additional audio processing utilities complementing the core utilities in utils.py.
Focuses on audio analysis, format conversion, quality metrics, and transformations.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np


# =============================================================================
# Audio Analysis Utilities
# =============================================================================

def compute_rms(y: np.ndarray) -> float:
    """Compute RMS (root mean square) of audio signal."""
    return np.sqrt(np.mean(y ** 2))


def compute_rms_per_frame(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute RMS per frame using librosa."""
    try:
        import librosa
        return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    except ImportError:
        # Manual implementation
        n_frames = 1 + (len(y) - frame_length) // hop_length
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start + frame_length]
            rms[i] = np.sqrt(np.mean(frame ** 2))
        return rms


def zero_crossing_rate(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute zero crossing rate per frame."""
    try:
        import librosa
        return librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    except ImportError:
        n_frames = 1 + (len(y) - frame_length) // hop_length
        zcr = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start + frame_length]
            zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length)
        return zcr


def spectral_centroid(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Compute spectral centroid per frame."""
    try:
        import librosa
        return librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    except ImportError:
        return np.zeros(1)


def spectral_rolloff(y: np.ndarray, sr: int = 22050, roll_percent: float = 0.85) -> np.ndarray:
    """Compute spectral rolloff per frame."""
    try:
        import librosa
        return librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    except ImportError:
        return np.zeros(1)


def spectral_flux(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Compute spectral flux (rate of change of spectral magnitude)."""
    try:
        import librosa
        S = np.abs(librosa.stft(y))
        spectral_diff = np.diff(S, axis=1)
        flux = np.sqrt(np.mean(spectral_diff ** 2, axis=0))
        return flux
    except ImportError:
        return np.zeros(1)


def compute_spectral_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Compute multiple spectral features at once."""
    features = {}
    try:
        import librosa
        features['centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['contrast'] = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        features['flatness'] = librosa.feature.spectral_flatness(y=y)[0]
    except ImportError:
        pass
    return features


# =============================================================================
# Format Conversion Utilities
# =============================================================================

def convert_audio_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    format: Optional[str] = None,
    bitrate: str = "320k",
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None
) -> bool:
    """Convert audio to different format using ffmpeg."""
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    
    if format:
        cmd.extend(["-f", format])
    if bitrate:
        cmd.extend(["-b:a", bitrate])
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    if channels:
        cmd.extend(["-ac", str(channels)])
    
    cmd.append(str(output_path))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def get_audio_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Get audio file information using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
    except Exception:
        pass
    
    # Fallback to librosa
    try:
        import librosa
        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return {
            'format': {'duration': duration},
            'streams': [{
                'sample_rate': sr,
                'channels': 1 if y.ndim == 1 else y.shape[0],
            }]
        }
    except Exception:
        return {}


def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return y
    
    try:
        import librosa
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Manual linear interpolation (less accurate)
        length = int(len(y) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, len(y) - 1, length),
            np.arange(len(y)),
            y
        )


def convert_channels(y: np.ndarray, target_channels: int) -> np.ndarray:
    """Convert between mono and stereo."""
    if target_channels == 1:
        # Stereo to mono
        if y.ndim == 2:
            return np.mean(y, axis=1)
        return y
    elif target_channels == 2:
        # Mono to stereo
        if y.ndim == 1:
            return np.stack([y, y], axis=1)
        return y
    return y


# =============================================================================
# Audio Quality Metrics
# =============================================================================

def snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute signal-to-noise ratio in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def crest_factor(y: np.ndarray) -> float:
    """Compute crest factor (peak to RMS ratio)."""
    peak = np.max(np.abs(y))
    rms = compute_rms(y)
    if rms == 0:
        return 0
    return peak / rms


def dynamic_range(y: np.ndarray) -> float:
    """Compute dynamic range in dB."""
    y_db = 20 * np.log10(np.abs(y) + 1e-10)
    return np.max(y_db) - np.min(y_db)


def peak_level(y: np.ndarray) -> float:
    """Get peak level in dB."""
    peak = np.max(np.abs(y))
    return 20 * np.log10(peak + 1e-10)


def measure_loudness(y: np.ndarray, sr: int) -> float:
    """Measure integrated loudness in LUFS (approximation)."""
    try:
        import librosa
        # Simple approximation using RMS
        rms = compute_rms(y)
        return -0.691 + 10 * np.log10(rms ** 2 + 1e-10)
    except ImportError:
        return -23.0  # Default target


@dataclass
class AudioQualityMetrics:
    """Container for audio quality metrics."""
    rms: float
    peak_db: float
    crest_factor: float
    dynamic_range: float
    snr: Optional[float] = None
    loudness: Optional[float] = None


def analyze_audio_quality(y: np.ndarray, sr: int) -> AudioQualityMetrics:
    """Analyze and return comprehensive audio quality metrics."""
    return AudioQualityMetrics(
        rms=compute_rms(y),
        peak_db=peak_level(y),
        crest_factor=crest_factor(y),
        dynamic_range=dynamic_range(y),
        loudness=measure_loudness(y, sr)
    )


# =============================================================================
# Audio Transformation Utilities
# =============================================================================

def invert_phase(y: np.ndarray) -> np.ndarray:
    """Invert the phase of audio signal."""
    return -y


def reverse_audio(y: np.ndarray) -> np.ndarray:
    """Reverse audio signal."""
    return y[::-1]


def trim_silence(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    try:
        import librosa
        y_trimmed, _ = librosa.effects.trim(
            y,
            top_db=-threshold_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return y_trimmed
    except ImportError:
        # Manual implementation
        amplitude_threshold = 10 ** (threshold_db / 20)
        non_silent = np.where(np.abs(y) > amplitude_threshold)[0]
        if len(non_silent) == 0:
            return y
        return y[non_silent[0]:non_silent[-1] + 1]


def add_silence(y: np.ndarray, sr: int, duration: float, position: str = "end") -> np.ndarray:
    """Add silence to audio."""
    silence_samples = int(duration * sr)
    silence = np.zeros(silence_samples)
    
    if position == "start":
        return np.concatenate([silence, y])
    elif position == "end":
        return np.concatenate([y, silence])
    elif position == "both":
        half_silence = np.zeros(silence_samples // 2)
        return np.concatenate([half_silence, y, half_silence])
    return y


def normalize_peak(y: np.ndarray, target_db: float = -0.1) -> np.ndarray:
    """Normalize peak to target dB."""
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    target_amplitude = 10 ** (target_db / 20)
    return y * (target_amplitude / peak)


def normalize_loudness(y: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Normalize loudness to target LUFS."""
    current_loudness = measure_loudness(y, sr)
    gain_db = target_lufs - current_loudness
    gain = 10 ** (gain_db / 20)
    return y * gain


def apply_gain(y: np.ndarray, db: float) -> np.ndarray:
    """Apply gain in dB."""
    gain = 10 ** (db / 20)
    return y * gain


def soft_clip(y: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Apply soft clipping (tanh-based)."""
    return np.tanh(y / threshold) * threshold


def hard_clip(y: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Apply hard clipping."""
    return np.clip(y, -threshold, threshold)


# =============================================================================
# Audio Segmentation Utilities
# =============================================================================

def split_by_silence(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_duration: float = 0.5,
    frame_length: int = 2048,
    hop_length: int = 512
) -> List[Tuple[np.ndarray, float, float]]:
    """Split audio by silence segments."""
    try:
        import librosa
        intervals = librosa.effects.split(
            y,
            top_db=-threshold_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        segments = []
        for start, end in intervals:
            duration = (end - start) / sr
            if duration >= min_duration:
                segment = y[start:end]
                start_time = start / sr
                end_time = end / sr
                segments.append((segment, start_time, end_time))
        
        return segments
    except ImportError:
        return [(y, 0.0, len(y) / sr)]


def extract_segment(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Extract a segment from audio."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    start_sample = max(0, start_sample)
    end_sample = min(len(y), end_sample)
    return y[start_sample:end_sample]


def fade(
    y: np.ndarray,
    sr: int,
    fade_in: float = 0.0,
    fade_out: float = 0.0
) -> np.ndarray:
    """Apply fade in and/or fade out."""
    if fade_in > 0:
        fade_samples = int(fade_in * sr)
        fade_curve = np.linspace(0, 1, fade_samples)
        y[:fade_samples] *= fade_curve
    
    if fade_out > 0:
        fade_samples = int(fade_out * sr)
        fade_curve = np.linspace(1, 0, fade_samples)
        y[-fade_samples:] *= fade_curve
    
    return y


def crossfade(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    overlap: float = 1.0
) -> np.ndarray:
    """Crossfade two audio segments."""
    overlap_samples = int(overlap * sr)
    
    if overlap_samples >= len(y1) or overlap_samples >= len(y2):
        overlap_samples = min(len(y1), len(y2))
    
    # Create crossfade envelope
    fade_out = np.linspace(1, 0, overlap_samples)
    fade_in = np.linspace(0, 1, overlap_samples)
    
    # Apply crossfade
    y1_end = y1[-overlap_samples:] * fade_out
    y2_start = y2[:overlap_samples] * fade_in
    
    # Combine
    crossfaded = y1_end + y2_start
    
    # Build result
    if len(y1) > overlap_samples:
        result = np.concatenate([y1[:-overlap_samples], crossfaded])
    else:
        result = crossfaded
    
    if len(y2) > overlap_samples:
        result = np.concatenate([result, y2[overlap_samples:]])
    
    return result


# =============================================================================
# Buffer & Window Utilities
# =============================================================================

def generate_window(window_type: str, length: int) -> np.ndarray:
    """Generate a window function."""
    if window_type == "hann":
        return np.hanning(length)
    elif window_type == "hamming":
        return np.hamming(length)
    elif window_type == "blackman":
        return np.blackman(length)
    elif window_type == "bartlett":
        return np.bartlett(length)
    elif window_type == "rectangle":
        return np.ones(length)
    else:
        return np.hanning(length)


def pad_to_length(y: np.ndarray, target_length: int, mode: str = "constant") -> np.ndarray:
    """Pad audio to target length."""
    if len(y) >= target_length:
        return y[:target_length]
    
    if mode == "constant":
        padding = target_length - len(y)
        return np.pad(y, (0, padding), mode=mode)
    elif mode == "wrap":
        # Repeat audio to fill
        repeats = (target_length // len(y)) + 1
        return np.tile(y, repeats)[:target_length]
    elif mode == "mirror":
        return np.pad(y, (0, target_length - len(y)), mode="reflect")
    
    return y


def frames_to_samples(frames: np.ndarray, hop_length: int = 512) -> np.ndarray:
    """Convert frame indices to sample indices."""
    return frames * hop_length


def samples_to_frames(samples: np.ndarray, hop_length: int = 512) -> np.ndarray:
    """Convert sample indices to frame indices."""
    return samples // hop_length


# =============================================================================
# Utility Functions
# =============================================================================

def detect_leading_silence(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -40.0
) -> float:
    """Detect leading silence duration in seconds."""
    amplitude_threshold = 10 ** (threshold_db / 20)
    non_silent = np.where(np.abs(y) > amplitude_threshold)[0]
    
    if len(non_silent) == 0:
        return len(y) / sr
    
    return non_silent[0] / sr


def detect_trailing_silence(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -40.0
) -> float:
    """Detect trailing silence duration in seconds."""
    amplitude_threshold = 10 ** (threshold_db / 20)
    non_silent = np.where(np.abs(y) > amplitude_threshold)[0]
    
    if len(non_silent) == 0:
        return len(y) / sr
    
    return (len(y) - non_silent[-1] - 1) / sr


def find_peaks(
    y: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 100
) -> np.ndarray:
    """Find peaks in audio signal."""
    peaks = []
    above_threshold = np.where(y > threshold)[0]
    
    if len(above_threshold) == 0:
        return np.array([])
    
    current_peak = above_threshold[0]
    
    for i, idx in enumerate(above_threshold):
        if idx - current_peak >= min_distance:
            # Find max in range
            segment = y[current_peak:idx]
            if len(segment) > 0:
                peak_offset = np.argmax(segment)
                peaks.append(current_peak + peak_offset)
            current_peak = idx
    
    # Add last peak
    segment = y[current_peak:]
    if len(segment) > 0:
        peaks.append(current_peak + np.argmax(segment))
    
    return np.array(peaks)


def energy_attack_decay(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048
) -> Dict[str, float]:
    """Calculate attack and decay characteristics."""
    rms = compute_rms_per_frame(y, frame_length=frame_length)
    
    if len(rms) < 2:
        return {"attack": 0, "decay": 0, "sustain": 0}
    
    # Attack: time to reach 90% of peak
    peak_idx = np.argmax(rms)
    peak_value = rms[peak_idx]
    threshold = peak_value * 0.9
    
    attack_frames = 0
    for i in range(peak_idx):
        if rms[i] < threshold:
            attack_frames = i
            break
    
    # Decay: time to drop to 10% of peak after peak
    decay_frames = 0
    for i in range(peak_idx, len(rms)):
        if rms[i] < peak_value * 0.1:
            decay_frames = i - peak_idx
            break
    
    attack_time = (attack_frames * frame_length) / sr
    decay_time = (decay_frames * frame_length) / sr
    
    return {
        "attack": attack_time,
        "decay": decay_time,
        "sustain": rms[peak_idx] if peak_idx < len(rms) else 0
    }


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Analysis
    'compute_rms',
    'compute_rms_per_frame',
    'zero_crossing_rate',
    'spectral_centroid',
    'spectral_rolloff',
    'spectral_flux',
    'compute_spectral_features',
    
    # Format conversion
    'convert_audio_format',
    'get_audio_info',
    'resample_audio',
    'convert_channels',
    
    # Quality metrics
    'snr',
    'crest_factor',
    'dynamic_range',
    'peak_level',
    'measure_loudness',
    'AudioQualityMetrics',
    'analyze_audio_quality',
    
    # Transformations
    'invert_phase',
    'reverse_audio',
    'trim_silence',
    'add_silence',
    'normalize_peak',
    'normalize_loudness',
    'apply_gain',
    'soft_clip',
    'hard_clip',
    
    # Segmentation
    'split_by_silence',
    'extract_segment',
    'fade',
    'crossfade',
    
    # Buffer utilities
    'generate_window',
    'pad_to_length',
    'frames_to_samples',
    'samples_to_frames',
    
    # Utility functions
    'detect_leading_silence',
    'detect_trailing_silence',
    'find_peaks',
    'energy_attack_decay',
]
