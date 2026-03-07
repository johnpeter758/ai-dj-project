"""
Math utilities for AI DJ project.
Provides common mathematical functions for audio processing, signal analysis, and music theory.
"""

import math
from typing import List, Tuple, Optional, Union


# ============== Audio Signal Math ==============

def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    if linear <= 0:
        return -float('inf')
    return 20.0 * math.log10(linear)


def rms(samples: List[float]) -> float:
    """Calculate RMS (root mean square) of a signal."""
    if not samples:
        return 0.0
    sum_squares = sum(x * x for x in samples)
    return math.sqrt(sum_squares / len(samples))


def peak(samples: List[float]) -> float:
    """Get peak amplitude from a signal."""
    if not samples:
        return 0.0
    return max(abs(x) for x in samples)


def normalize(samples: List[float], target_peak: float = 1.0) -> List[float]:
    """Normalize signal to a target peak amplitude."""
    if not samples:
        return []
    current_peak = peak(samples)
    if current_peak == 0:
        return samples
    scale = target_peak / current_peak
    return [x * scale for x in samples]


def fade_in(samples: List[float], samples_fade: int) -> List[float]:
    """Apply linear fade-in to signal."""
    if not samples or samples_fade <= 0:
        return samples
    
    result = samples.copy()
    fade_len = min(samples_fade, len(samples))
    for i in range(fade_len):
        gain = i / fade_len
        result[i] *= gain
    return result


def fade_out(samples: List[float], samples_fade: int) -> List[float]:
    """Apply linear fade-out to signal."""
    if not samples or samples_fade <= 0:
        return samples
    
    result = samples.copy()
    fade_len = min(samples_fade, len(samples))
    for i in range(fade_len):
        gain = (fade_len - i) / fade_len
        result[len(result) - fade_len + i] *= gain
    return result


# ============== Frequency & Pitch Math ==============

def hz_to_midi(hz: float) -> float:
    """Convert frequency in Hz to MIDI note number."""
    if hz <= 0:
        return 0.0
    return 69 + 12 * math.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def hz_to_bark(hz: float) -> float:
    """Convert frequency in Hz to Bark scale."""
    if hz <= 0:
        return 0.0
    return 13 * math.atan(0.00076 * hz) + 3.5 * math.atan((hz / 7500) ** 2)


def hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to Mel scale."""
    if hz <= 0:
        return 0.0
    return 2595 * math.log10(1 + hz / 700)


def mel_to_hz(mel: float) -> float:
    """Convert Mel scale to frequency in Hz."""
    return 700 * (10 ** (mel / 2595) - 1)


def pitch_shift_ratio(semitones: float) -> float:
    """Calculate pitch shift ratio from semitones."""
    return 2 ** (semitones / 12.0)


# ============== Tempo & Time Math ==============

def bpm_to_ms(bpm: float) -> float:
    """Convert BPM to milliseconds per beat."""
    if bpm <= 0:
        return 0.0
    return 60000.0 / bpm


def ms_to_bpm(ms: float) -> float:
    """Convert milliseconds per beat to BPM."""
    if ms <= 0:
        return 0.0
    return 60000.0 / ms


def beats_to_samples(beats: float, bpm: float, sample_rate: int) -> int:
    """Convert beats to sample count."""
    ms_per_beat = bpm_to_ms(bpm)
    ms_per_beat_per_sample = 1000.0 / sample_rate
    return int(beats * ms_per_beat / ms_per_beat_per_sample)


def samples_to_beats(samples: int, bpm: float, sample_rate: int) -> float:
    """Convert sample count to beats."""
    ms_per_beat = bpm_to_ms(bpm)
    ms_per_sample = 1000.0 / sample_rate
    return (samples * ms_per_sample) / ms_per_beat


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to sample count."""
    return int(seconds * sample_rate)


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert sample count to seconds."""
    return samples / sample_rate


def time_stretch_ratio(source_bpm: float, target_bpm: float) -> float:
    """Calculate time stretch ratio for tempo change."""
    if source_bpm <= 0:
        return 1.0
    return source_bpm / target_bpm


# ============== Interpolation ==============

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def lerp_array(a: List[float], b: List[float], t: float) -> List[float]:
    """Linear interpolation between two arrays."""
    min_len = min(len(a), len(b))
    return [lerp(a[i], b[i], t) for i in range(min_len)]


def smooth_step(t: float) -> float:
    """Smooth step interpolation (ease in-out)."""
    return t * t * (3 - 2 * t)


def cosine_interpolate(a: float, b: float, t: float) -> float:
    """Cosine interpolation for smoother transitions."""
    mu2 = (1 - math.cos(t * math.pi)) / 2
    return a * (1 - mu2) + b * mu2


def cubic_interpolate(y0: float, y1: float, y2: float, y3: float, t: float) -> float:
    """Cubic interpolation using Catmull-Rom spline."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * y1) +
        (-y0 + y2) * t +
        (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2 +
        (-y0 + 3 * y1 - 3 * y2 + y3) * t3
    )


# ============== Window Functions ==============

def hann_window(length: int) -> List[float]:
    """Generate Hann window."""
    if length <= 0:
        return []
    return [0.5 * (1 - math.cos(2 * math.pi * i / (length - 1))) 
            for i in range(length)]


def hamming_window(length: int) -> List[float]:
    """Generate Hamming window."""
    if length <= 0:
        return []
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (length - 1)) 
            for i in range(length)]


def blackman_window(length: int) -> List[float]:
    """Generate Blackman window."""
    if length <= 0:
        return []
    return [0.42 - 0.5 * math.cos(2 * math.pi * i / (length - 1)) + 
            0.08 * math.cos(4 * math.pi * i / (length - 1)) 
            for i in range(length)]


def sine_window(length: int) -> List[float]:
    """Generate sine window."""
    if length <= 0:
        return []
    return [math.sin(math.pi * i / (length - 1)) 
            for i in range(length)]


# ============== FFT Helper Functions ==============

def fft_size_to_window_size(fft_size: int, overlap: float = 0.5) -> int:
    """Calculate window size from FFT size and overlap."""
    return int(fft_size * (1 - overlap))


def hop_size(fft_size: int, overlap: float = 0.5) -> int:
    """Calculate hop size from FFT size and overlap."""
    return int(fft_size * (1 - overlap))


def frequency_to_bin(freq: float, sample_rate: int, fft_size: int) -> int:
    """Convert frequency to FFT bin index."""
    return int(freq * fft_size / sample_rate)


def bin_to_frequency(bin_idx: int, sample_rate: int, fft_size: int) -> float:
    """Convert FFT bin index to frequency."""
    return bin_idx * sample_rate / fft_size


# ============== Statistical Functions ==============

def mean(samples: List[float]) -> float:
    """Calculate arithmetic mean."""
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def variance(samples: List[float]) -> float:
    """Calculate variance."""
    if len(samples) < 2:
        return 0.0
    m = mean(samples)
    return sum((x - m) ** 2 for x in samples) / (len(samples) - 1)


def std_dev(samples: List[float]) -> float:
    """Calculate standard deviation."""
    return math.sqrt(variance(samples))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def normalize_range(value: float, old_min: float, old_max: float, 
                    new_min: float = 0.0, new_max: float = 1.0) -> float:
    """Map value from old range to new range."""
    if old_max == old_min:
        return new_min
    normalized = (value - old_min) / (old_max - old_min)
    return new_min + normalized * (new_max - new_min)


# ============== Envelope Functions ==============

def attack_decay(t: float, attack: float, decay: float) -> float:
    """Simple AD envelope."""
    if t < attack:
        return t / attack if attack > 0 else 1.0
    decay_t = t - attack
    return math.exp(-decay_t / max(decay, 0.001))


def adsr(t: float, attack: float, decay: float, sustain: float, 
         release: float, duration: float) -> float:
    """ADSR envelope generator."""
    if t < attack:
        return t / attack if attack > 0 else 1.0
    elif t < attack + decay:
        return 1.0 - (1.0 - sustain) * (t - attack) / decay
    elif t < duration - release:
        return sustain
    elif t <= duration:
        return sustain * (duration - t) / release
    return 0.0


# ============== Curve Fitting ==============

def exponential_decay(start: float, end: float, t: float, total: float) -> float:
    """Exponential decay curve."""
    if total <= 0:
        return start
    return end + (start - end) * math.exp(-3 * t / total)


def logarithmic_curve(start: float, end: float, t: float, total: float) -> float:
    """Logarithmic curve."""
    if total <= 0:
        return start
    return start + (end - start) * math.log(1 + 9 * t / total) / math.log(10)


# ============== Geometry & Angles ==============

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


def angle_diff(a1: float, a2: float) -> float:
    """Calculate smallest difference between two angles in degrees."""
    diff = (a2 - a1 + 180) % 360 - 180
    return diff + 360 if diff < -180 else diff


def panning_law(pan: float) -> Tuple[float, float]:
    """
    Apply equal-power panning law.
    Pan: -1 (left) to 1 (right), 0 is center.
    Returns: (left_gain, right_gain)
    """
    pan = clamp(pan, -1.0, 1.0)
    angle = (pan + 1) * math.pi / 4
    left = math.cos(angle)
    right = math.sin(angle)
    return left, right
