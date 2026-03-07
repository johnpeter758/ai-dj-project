"""
DSP Utilities
Common Digital Signal Processing utilities for audio manipulation.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable, Union
from functools import wraps

# Try to import audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import scipy
    from scipy import signal
    from scipy.fft import fft, ifft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# WINDOW FUNCTIONS
# =============================================================================

def hann_window(length: int) -> np.ndarray:
    """Generate Hann (Hanning) window."""
    return np.hanning(length)


def hamming_window(length: int) -> np.ndarray:
    """Generate Hamming window."""
    return np.hamming(length)


def blackman_window(length: int) -> np.ndarray:
    """Generate Blackman window."""
    return np.blackman(length)


def get_window(window_type: str, length: int) -> np.ndarray:
    """
    Get a window function by name.
    
    Args:
        window_type: Type of window - 'hann', 'hamming', 'blackman', 'kaiser', 'boxcar'
        length: Window length in samples
        
    Returns:
        Window array
    """
    window_type = window_type.lower()
    
    if window_type == 'hann':
        return np.hanning(length)
    elif window_type == 'hamming':
        return np.hamming(length)
    elif window_type == 'blackman':
        return np.blackman(length)
    elif window_type == 'boxcar':
        return np.boxcar(length)
    elif window_type == 'kaiser':
        # Default beta=14 for good stopband attenuation
        return np.kaiser(length, beta=14)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


# =============================================================================
# FFT / FREQUENCY DOMAIN
# =============================================================================

def compute_fft(signal: np.ndarray, 
                n_fft: Optional[int] = None,
                hop_length: Optional[int] = None,
                window: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute STFT (Short-Time Fourier Transform).
    
    Args:
        signal: Input audio signal
        n_fft: FFT size (default: next power of 2 >= len(signal))
        hop_length: Number of samples between frames
        window: Window function (default: Hann)
        
    Returns:
        Tuple of (magnitude, phase) spectra
    """
    if n_fft is None:
        n_fft = next_power_of_2(len(signal))
    
    if hop_length is None:
        hop_length = n_fft // 4
    
    if window is None:
        window = np.hanning(n_fft)
    
    # Pad signal if needed
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')
    
    # Compute STFT using overlap-add
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    
    # Pad to accommodate all frames
    padded = np.pad(signal, (0, n_fft), mode='constant')
    
    stft_result = []
    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start:start + n_fft] * window
        fft_frame = np.fft.fft(frame, n=n_fft)
        stft_result.append(fft_frame)
    
    stft_matrix = np.array(stft_result).T
    
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    return magnitude, phase


def compute_ifft(magnitude: np.ndarray, 
                 phase: np.ndarray,
                 hop_length: Optional[int] = None,
                 window: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute inverse STFT to reconstruct signal.
    
    Args:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram
        hop_length: Hop length used in STFT
        window: Window function
        
    Returns:
        Reconstructed time-domain signal
    """
    n_fft = magnitude.shape[0]
    n_frames = magnitude.shape[1]
    
    if hop_length is None:
        hop_length = n_fft // 4
    
    if window is None:
        window = np.hanning(n_fft)
    
    # Reconstruct complex spectrum
    stft_matrix = magnitude * np.exp(1j * phase)
    
    # Overlap-add synthesis
    signal = np.zeros(n_fft + (n_frames - 1) * hop_length)
    window_sum = np.zeros_like(signal)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = np.fft.ifft(stft_matrix[:, i], n=n_fft).real
        signal[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window
    
    # Normalize by window sum
    window_sum = np.maximum(window_sum, 1e-8)
    signal = signal / window_sum
    
    return signal


def frequency_bins(n_fft: int, sample_rate: int) -> np.ndarray:
    """
    Get frequency bins for FFT.
    
    Args:
        n_fft: FFT size
        sample_rate: Sample rate in Hz
        
    Returns:
        Array of frequency values for each bin
    """
    return np.fft.fftfreq(n_fft, 1.0 / sample_rate)[:n_fft // 2]


def next_power_of_2(n: int) -> int:
    """Get next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()


# =============================================================================
# FILTER UTILITIES
# =============================================================================

def apply_highpass(signal: np.ndarray, 
                   cutoff: float, 
                   sample_rate: int,
                   order: int = 4) -> np.ndarray:
    """
    Apply highpass filter to remove low frequencies.
    
    Args:
        signal: Input signal
        cutoff: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for filtering")
    
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    sos = signal.butter(order, normalized_cutoff, btype='high', output='sos')
    return signal.sosfilt(sos, signal)


def apply_lowpass(signal: np.ndarray, 
                  cutoff: float, 
                  sample_rate: int,
                  order: int = 4) -> np.ndarray:
    """
    Apply lowpass filter to remove high frequencies.
    
    Args:
        signal: Input signal
        cutoff: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for filtering")
    
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    sos = signal.butter(order, normalized_cutoff, btype='low', output='sos')
    return signal.sosfilt(sos, signal)


def apply_bandpass(signal: np.ndarray, 
                   low_cutoff: float,
                   high_cutoff: float,
                   sample_rate: int,
                   order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter.
    
    Args:
        signal: Input signal
        low_cutoff: Low cutoff frequency in Hz
        high_cutoff: High cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for filtering")
    
    nyquist = sample_rate / 2
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return signal.sosfilt(sos, signal)


def design_notch_filter(frequency: float, 
                        sample_rate: int, 
                        quality: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a notch filter to remove a specific frequency (e.g., hum).
    
    Args:
        frequency: Frequency to remove in Hz
        sample_rate: Sample rate in Hz
        quality: Q factor (higher = narrower notch)
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for filtering")
    
    nyquist = sample_rate / 2
    normalized_freq = frequency / nyquist
    
    b, a = signal.iirnotch(normalized_freq, quality, sample_rate)
    return b, a


def apply_notch_filter(signal: np.ndarray, 
                       frequency: float,
                       sample_rate: int,
                       quality: float = 30.0) -> np.ndarray:
    """Apply notch filter to remove specific frequency."""
    b, a = design_notch_filter(frequency, sample_rate, quality)
    return signal.lfilter(b, a, signal)


# =============================================================================
# NORMALIZATION & AMPLITUDE
# =============================================================================

def normalize_peak(signal: np.ndarray, target_db: float = -0.1) -> np.ndarray:
    """
    Normalize signal to target peak level in dB.
    
    Args:
        signal: Input signal
        target_db: Target peak level in dB
        
    Returns:
        Normalized signal
    """
    current_peak = np.abs(signal).max()
    if current_peak == 0:
        return signal
    
    target_linear = 10 ** (target_db / 20)
    gain = target_linear / current_peak
    
    return signal * gain


def normalize_loudness(signal: np.ndarray, 
                       sample_rate: int,
                       target_lufs: float = -14.0) -> np.ndarray:
    """
    Normalize signal to target loudness in LUFS.
    
    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        target_lufs: Target loudness in LUFS
        
    Returns:
        Normalized signal
    """
    # Compute integrated loudness (simplified)
    loudness = compute_loudness(signal, sample_rate)
    
    if loudness <= -70:  # Silence
        return signal
    
    gain_db = target_lufs - loudness
    gain_linear = 10 ** (gain_db / 20)
    
    return signal * gain_linear


def compute_loudness(signal: np.ndarray, sample_rate: int) -> float:
    """
    Compute approximate loudness of signal in LUFS.
    Simplified implementation using RMS.
    
    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Approximate loudness in LUFS
    """
    # Compute RMS
    rms = np.sqrt(np.mean(signal ** 2))
    
    if rms < 1e-10:
        return -70.0
    
    # Convert to LUFS (simplified approximation)
    lufs = -0.691 + 10 * np.log10(rms ** 2)
    
    return max(-70.0, min(0.0, lufs))


def apply_gain(signal: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain in dB to signal.
    
    Args:
        signal: Input signal
        gain_db: Gain in dB
        
    Returns:
        Signal with gain applied
    """
    gain_linear = 10 ** (gain_db / 20)
    return signal * gain_linear


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """Convert linear gain to decibels."""
    if linear <= 0:
        return -np.inf
    return 20 * np.log10(linear)


# =============================================================================
# SIGNAL MODIFICATION
# =============================================================================

def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from signal.
    
    Args:
        signal: Input signal
        
    Returns:
        Signal with DC offset removed
    """
    return signal - np.mean(signal)


def invert_phase(signal: np.ndarray) -> np.ndarray:
    """
    Invert the phase of the signal.
    
    Args:
        signal: Input signal
        
    Returns:
        Phase-inverted signal
    """
    return -signal


def mono_to_stereo(signal: np.ndarray) -> np.ndarray:
    """
    Convert mono signal to stereo (duplicated).
    
    Args:
        signal: Mono signal
        
    Returns:
        Stereo signal (2D array)
    """
    return np.column_stack([signal, signal])


def stereo_to_mono(stereo: np.ndarray) -> np.ndarray:
    """
    Convert stereo signal to mono.
    
    Args:
        stereo: Stereo signal (2D array)
        
    Returns:
        Mono signal
    """
    return np.mean(stereo, axis=1)


def sum_to_mono(*signals: np.ndarray) -> np.ndarray:
    """
    Sum multiple mono signals to single mono output.
    
    Args:
        *signals: Variable number of mono signals
        
    Returns:
        Combined mono signal
    """
    max_len = max(len(s) for s in signals)
    
    # Pad shorter signals
    padded = []
    for s in signals:
        if len(s) < max_len:
            s = np.pad(s, (0, max_len - len(s)), mode='constant')
        padded.append(s)
    
    return np.sum(padded, axis=0)


def mix_signals(*signals: np.ndarray, 
                gains: Optional[List[float]] = None) -> np.ndarray:
    """
    Mix multiple signals with optional gain control.
    
    Args:
        *signals: Variable number of signals
        gains: List of gain values for each signal (linear)
        
    Returns:
        Mixed signal
    """
    if gains is None:
        gains = [1.0] * len(signals)
    
    if len(gains) != len(signals):
        raise ValueError("Number of gains must match number of signals")
    
    max_len = max(len(s) for s in signals)
    
    # Pad and apply gains
    mixed = np.zeros(max_len)
    for s, g in zip(signals, gains):
        if len(s) < max_len:
            s = np.pad(s, (0, max_len - len(s)), mode='constant')
        mixed += s * g
    
    # Normalize to prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed


# =============================================================================
# CROSSFADES & FADES
# =============================================================================

def crossfade(signal1: np.ndarray, 
             signal2: np.ndarray,
             crossfade_duration: float,
             sample_rate: int,
             fade_curve: str = 'equal_power') -> np.ndarray:
    """
    Create crossfade between two signals.
    
    Args:
        signal1: First signal
        signal2: Second signal
        crossfade_duration: Duration in seconds
        sample_rate: Sample rate in Hz
        fade_curve: 'linear' or 'equal_power'
        
    Returns:
        Crossfaded signal
    """
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    if fade_curve == 'equal_power':
        # Equal power crossfade (smoother)
        fade_in = np.linspace(0, 1, crossfade_samples) ** 0.5
        fade_out = np.linspace(1, 0, crossfade_samples) ** 0.5
    else:
        # Linear crossfade
        fade_in = np.linspace(0, 1, crossfade_samples)
        fade_out = np.linspace(1, 0, crossfade_samples)
    
    # Pad signals if needed
    len1 = len(signal1)
    len2 = len(signal2)
    
    # Determine output length
    # Overlap region at end of signal1 and start of signal2
    output_len = max(len1, len2 + crossfade_samples)
    
    result = np.zeros(output_len)
    
    # Copy signal1
    result[:len1] = signal1
    
    # Apply fade out to end of signal1
    if len1 >= crossfade_samples:
        result[len1 - crossfade_samples:len1] *= fade_out
    
    # Add signal2 with fade in
    if len2 > 0:
        # Extend result if needed
        if len2 + crossfade_samples > len(result):
            result = np.pad(result, (0, len2 + crossfade_samples - len(result)))
        
        result[:crossfade_samples] += signal2[:crossfade_samples] * fade_in
        result[crossfade_samples:min(len2, len(result))] += signal2[crossfade_samples:min(len2, len(result))]
    
    return result


def fade_in(signal: np.ndarray, 
            duration: float, 
            sample_rate: int,
            curve: str = 'exponential') -> np.ndarray:
    """
    Apply fade in to signal.
    
    Args:
        signal: Input signal
        duration: Fade duration in seconds
        sample_rate: Sample rate in Hz
        curve: 'linear', 'exponential', or 's_curve'
        
    Returns:
        Faded signal
    """
    fade_samples = int(duration * sample_rate)
    fade_samples = min(fade_samples, len(signal))
    
    if curve == 'linear':
        fade_curve = np.linspace(0, 1, fade_samples)
    elif curve == 'exponential':
        fade_curve = np.linspace(0, 1, fade_samples) ** 2
    elif curve == 's_curve':
        fade_curve = np.sin(np.linspace(0, np.pi / 2, fade_samples))
    else:
        raise ValueError(f"Unknown curve: {curve}")
    
    result = signal.copy()
    result[:fade_samples] *= fade_curve
    
    return result


def fade_out(signal: np.ndarray, 
             duration: float, 
             sample_rate: int,
             curve: str = 'exponential') -> np.ndarray:
    """
    Apply fade out to signal.
    
    Args:
        signal: Input signal
        duration: Fade duration in seconds
        sample_rate: Sample rate in Hz
        curve: 'linear', 'exponential', or 's_curve'
        
    Returns:
        Faded signal
    """
    fade_samples = int(duration * sample_rate)
    fade_samples = min(fade_samples, len(signal))
    
    if curve == 'linear':
        fade_curve = np.linspace(1, 0, fade_samples)
    elif curve == 'exponential':
        fade_curve = np.linspace(1, 0, fade_samples) ** 2
    elif curve == 's_curve':
        fade_curve = np.sin(np.linspace(np.pi / 2, 0, fade_samples))
    else:
        raise ValueError(f"Unknown curve: {curve}")
    
    result = signal.copy()
    result[-fade_samples:] *= fade_curve
    
    return result


# =============================================================================
# RESAMPLING
# =============================================================================

def resample(signal: np.ndarray, 
             orig_sr: int, 
             target_sr: int) -> np.ndarray:
    """
    Resample signal to new sample rate.
    
    Args:
        signal: Input signal
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled signal
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for resampling")
    
    if orig_sr == target_sr:
        return signal
    
    # Use polyphase resampling for quality
    num_samples = int(len(signal) * target_sr / orig_sr)
    return signal.resample_poly(signal, target_sr, orig_sr, num_samples)


def resample_linear(signal: np.ndarray,
                    orig_sr: int,
                    target_sr: int) -> np.ndarray:
    """
    Simple linear resampling (faster but lower quality).
    
    Args:
        signal: Input signal
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled signal
    """
    if orig_sr == target_sr:
        return signal
    
    ratio = target_sr / orig_sr
    new_length = int(len(signal) * ratio)
    
    # Use scipy's resample (FIR-based)
    return signal.resample(signal, new_length)


# =============================================================================
# TIME STRETCHING / PITCH SHIFTING (HELPER FUNCTIONS)
# =============================================================================

def get_stretch_factor(current_bpm: float, target_bpm: float) -> float:
    """
    Calculate time stretch factor for tempo matching.
    
    Args:
        current_bpm: Current tempo in BPM
        target_bpm: Target tempo in BPM
        
    Returns:
        Stretch factor (1.0 = no change, >1 = slower, <1 = faster)
    """
    return current_bpm / target_bpm


def estimate_pitch_shift(semitones: float) -> float:
    """
    Estimate pitch shift ratio from semitones.
    
    Args:
        semitones: Number of semitones to shift
        
    Returns:
        Pitch shift ratio
    """
    return 2 ** (semitones / 12)


def semitones_to_hz(semitones: float, reference_hz: float = 440.0) -> float:
    """
    Convert semitones to frequency in Hz.
    
    Args:
        semitones: Number of semitones from reference
        reference_hz: Reference frequency in Hz (default: A4 = 440)
        
    Returns:
        Frequency in Hz
    """
    return reference_hz * 2 ** (semitones / 12)


def hz_to_note(frequency: float) -> str:
    """
    Convert frequency to note name.
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        Note name (e.g., 'A4', 'C#5')
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    if frequency <= 0:
        return 'N/A'
    
    # Calculate MIDI note number (A4 = 69 = 440Hz)
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    midi_note = round(midi_note)
    
    note_index = midi_note % 12
    octave = (midi_note // 12) - 1
    
    return f"{note_names[note_index]}{octave}"


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_sine(frequency: float, 
                 duration: float, 
                 sample_rate: int,
                 amplitude: float = 0.5,
                 phase: float = 0.0) -> np.ndarray:
    """
    Generate sine wave.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0-1)
        phase: Initial phase in radians
        
    Returns:
        Sine wave signal
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def generate_square(frequency: float,
                    duration: float,
                    sample_rate: int,
                    amplitude: float = 0.5,
                    duty_cycle: float = 0.5) -> np.ndarray:
    """Generate square wave."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    return amplitude * signal.square(2 * np.pi * frequency * t, duty_cycle)


def generate_sawtooth(frequency: float,
                       duration: float,
                       sample_rate: int,
                       amplitude: float = 0.5) -> np.ndarray:
    """Generate sawtooth wave."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    return amplitude * signal.sawtooth(2 * np.pi * frequency * t)


def generate_noise(duration: float,
                   sample_rate: int,
                   color: str = 'white',
                   amplitude: float = 0.5) -> np.ndarray:
    """
    Generate noise signal.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        color: 'white', 'pink', or 'brown'
        amplitude: Output amplitude
        
    Returns:
        Noise signal
    """
    num_samples = int(duration * sample_rate)
    
    if color == 'white':
        noise = np.random.randn(num_samples)
    elif color == 'pink':
        # Pink noise using Voss algorithm approximation
        noise = pink_noise(num_samples)
    elif color == 'brown':
        # Brown noise (random walk)
        noise = np.cumsum(np.random.randn(num_samples))
        noise = noise / np.max(np.abs(noise))
    else:
        raise ValueError(f"Unknown noise color: {color}")
    
    # Normalize and apply amplitude
    noise = noise / np.max(np.abs(noise)) if np.max(np.abs(noise)) > 0 else noise
    return noise * amplitude


def pink_noise(num_samples: int) -> np.ndarray:
    """Generate pink noise (1/f noise)."""
    # Approximate pink noise using white noise + filtering
    white = np.random.randn(num_samples)
    
    # Simple pink noise approximation
    b = [0.99886, 0.0555179, -0.0750759, 0.1538520, 0.1134854, 0.0924498, 0.0469703]
    a = [1.0, -2.4949562, 2.636527, -1.616935, 0.666789, -0.135220, 0.093418]
    
    pink = signal.lfilter(b, a, white)
    return pink


# =============================================================================
# PHASE & CORRELATION
# =============================================================================

def compute_autocorrelation(signal: np.ndarray, 
                            max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation of signal.
    
    Args:
        signal: Input signal
        max_lag: Maximum lag to compute
        
    Returns:
        Autocorrelation array
    """
    signal = signal - np.mean(signal)
    
    if max_lag is None:
        max_lag = len(signal) - 1
    
    # Use FFT for efficient computation
    fft = np.fft.fft(signal, n=2 * len(signal))
    psd = fft * np.conj(fft)
    autocorr = np.fft.ifft(psd).real
    
    return autocorr[:max_lag + 1]


def compute_crosscorrelation(signal1: np.ndarray, 
                             signal2: np.ndarray) -> np.ndarray:
    """
    Compute cross-correlation between two signals.
    
    Args:
        signal1: First signal
        signal2: Second signal
        
    Returns:
        Cross-correlation array
    """
    # Pad shorter signal
    if len(signal1) != len(signal2):
        max_len = max(len(signal1), len(signal2))
        signal1 = np.pad(signal1, (0, max_len - len(signal1)))
        signal2 = np.pad(signal2, (0, max_len - len(signal2)))
    
    # FFT-based cross-correlation
    fft1 = np.fft.fft(signal1)
    fft2 = np.fft.fft(signal2)
    xcorr = np.fft.ifft(fft1 * np.conj(fft2)).real
    
    return xcorr


def find_lag(signal1: np.ndarray, signal2: np.ndarray) -> int:
    """
    Find time lag between two signals using cross-correlation.
    
    Args:
        signal1: First signal
        signal2: Second signal
        
    Returns:
        Lag in samples (positive = signal2 leads)
    """
    xcorr = compute_crosscorrelation(signal1, signal2)
    return np.argmax(xcorr) - len(signal1) + 1


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap phase to remove discontinuities.
    
    Args:
        phase: Phase array in radians
        
    Returns:
        Unwrapped phase
    """
    return np.unwrap(phase)


def minimum_phase(signal: np.ndarray) -> np.ndarray:
    """
    Compute minimum phase representation of signal.
    
    Args:
        signal: Input signal
        
    Returns:
        Minimum phase signal
    """
    # FFT
    fft_signal = np.fft.fft(signal, n=2 * len(signal))
    
    # Log magnitude
    log_magnitude = np.log(np.abs(fft_signal) + 1e-8)
    
    # Hilbert transform for cepstrum
    cepstrum = np.fft.ifft(log_magnitude).real
    
    # Make it odd
    n = len(cepstrum)
    cepstrum[1:n//2] *= 2
    cepstrum[n//2 + 1:] = 0
    
    # Reconstruct
    min_phase_cepstrum = np.fft.fft(cepstrum)
    min_phase_spectrum = np.exp(np.fft.ifft(min_phase_cepstrum).real)
    
    return np.fft.ifft(min_phase_spectrum * np.exp(1j * np.angle(fft_signal))).real[:len(signal)]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def split_signal(signal: np.ndarray, 
                 num_chunks: int) -> List[np.ndarray]:
    """
    Split signal into equal chunks.
    
    Args:
        signal: Input signal
        num_chunks: Number of chunks
        
    Returns:
        List of signal chunks
    """
    chunk_size = len(signal) // num_chunks
    return [signal[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


def combine_chunks(chunks: List[np.ndarray]) -> np.ndarray:
    """
    Combine signal chunks into single signal.
    
    Args:
        chunks: List of signal chunks
        
    Returns:
        Combined signal
    """
    return np.concatenate(chunks)


def trim_silence(signal: np.ndarray, 
                 sample_rate: int,
                 threshold_db: float = -40.0,
                 min_duration: float = 0.1) -> np.ndarray:
    """
    Trim silence from beginning and end of signal.
    
    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        threshold_db: Silence threshold in dB
        min_duration: Minimum silence duration to trim
        
    Returns:
        Trimmed signal
    """
    # Convert threshold to linear
    threshold = 10 ** (threshold_db / 20)
    
    # Find non-silent regions
    is_silent = np.abs(signal) < threshold
    
    # Find first and last non-silent sample
    nonzero_indices = np.where(~is_silent)[0]
    
    if len(nonzero_indices) == 0:
        return signal
    
    start = nonzero_indices[0]
    end = nonzero_indices[-1] + 1
    
    # Apply minimum duration
    min_samples = int(min_duration * sample_rate)
    start = max(0, start - min_samples)
    end = min(len(signal), end + min_samples)
    
    return signal[start:end]


def pad_to_length(signal: np.ndarray, 
                  target_length: int,
                  mode: str = 'constant') -> np.ndarray:
    """
    Pad signal to target length.
    
    Args:
        signal: Input signal
        target_length: Target length in samples
        mode: Padding mode - 'constant', 'wrap', 'reflect'
        
    Returns:
        Padded signal
    """
    if len(signal) >= target_length:
        return signal[:target_length]
    
    if mode == 'constant':
        return np.pad(signal, (0, target_length - len(signal)), mode='constant')
    elif mode == 'wrap':
        # Wrap around
        repeats = (target_length // len(signal)) + 1
        return np.tile(signal, repeats)[:target_length]
    elif mode == 'reflect':
        return np.pad(signal, (0, target_length - len(signal)), mode='reflect')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")


def ensure_mono(signal: np.ndarray) -> np.ndarray:
    """
    Ensure signal is mono (reduce stereo to mono).
    
    Args:
        signal: Input signal (can be 1D or 2D)
        
    Returns:
        Mono signal
    """
    if signal.ndim == 1:
        return signal
    elif signal.ndim == 2:
        return np.mean(signal, axis=1)
    else:
        raise ValueError(f"Invalid signal shape: {signal.shape}")


def ensure_stereo(signal: np.ndarray) -> np.ndarray:
    """
    Ensure signal is stereo (convert mono to stereo if needed).
    
    Args:
        signal: Input signal
        
    Returns:
        Stereo signal
    """
    if signal.ndim == 1:
        return np.column_stack([signal, signal])
    elif signal.ndim == 2 and signal.shape[1] == 2:
        return signal
    else:
        raise ValueError(f"Invalid signal shape: {signal.shape}")


# =============================================================================
# STEREO UTILITIES
# =============================================================================

def get_stereo_width(signal: np.ndarray) -> float:
    """
    Estimate stereo width of signal.
    
    Args:
        signal: Stereo signal (2D array)
        
    Returns:
        Width value (0 = mono, 1 = wide)
    """
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise ValueError("Stereo signal required")
    
    left = signal[:, 0]
    right = signal[:, 1]
    
    # Mid and side
    mid = (left + right) / 2
    side = (left - right) / 2
    
    mid_energy = np.mean(mid ** 2)
    side_energy = np.mean(side ** 2)
    
    if mid_energy == 0:
        return 0.0
    
    return np.sqrt(side_energy / mid_energy)


def stereo_to_mid_side(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert stereo to mid/side representation.
    
    Args:
        signal: Stereo signal (2D array)
        
    Returns:
        Tuple of (mid, side) signals
    """
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise ValueError("Stereo signal required")
    
    left = signal[:, 0]
    right = signal[:, 1]
    
    mid = (left + right) / 2
    side = (left - right) / 2
    
    return mid, side


def mid_side_to_stereo(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """
    Convert mid/side to stereo.
    
    Args:
        mid: Mid signal
        side: Side signal
        
    Returns:
        Stereo signal (2D array)
    """
    left = mid + side
    right = mid - side
    
    return np.column_stack([left, right])


# =============================================================================
# ENVELOPE FOLLOWER
# =============================================================================

class EnvelopeFollower:
    """Envelope follower for dynamics processing."""
    
    def __init__(self, attack_ms: float = 10.0, release_ms: float = 100.0, sample_rate: int = 44100):
        """
        Initialize envelope follower.
        
        Args:
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.attack = 1.0 - np.exp(-1.0 / (attack_ms * sample_rate / 1000))
        self.release = 1.0 - np.exp(-1.0 / (release_ms * sample_rate / 1000))
        self.envelope = 0.0
    
    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Process signal to get envelope.
        
        Args:
            signal: Input signal
            
        Returns:
            Envelope signal
        """
        # Get absolute value
        abs_signal = np.abs(signal)
        
        # Apply attack/release
        envelope = np.zeros_like(signal)
        for i, sample in enumerate(abs_signal):
            if sample > self.envelope:
                self.envelope += self.attack * (sample - self.envelope)
            else:
                self.envelope += self.release * (sample - self.envelope)
            envelope[i] = self.envelope
        
        return envelope
    
    def reset(self):
        """Reset envelope to zero."""
        self.envelope = 0.0
