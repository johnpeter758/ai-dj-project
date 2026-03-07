"""
Audio Filter Library for AI DJ Project
Comprehensive collection of audio filters and filtering utilities
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy import signal
from scipy.signal import butter, lfilter, firwin, sosfilt, sosfiltfilt


class AudioFilter:
    """Core audio filter processor with various filter types."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._filter_state = None
        self._sos_state = None
    
    def _normalize(self, audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        """Normalize audio to target dB."""
        if len(audio) == 0:
            return audio
        peak = np.abs(audio).max()
        if peak > 0:
            target_rms = 10 ** (target_db / 20)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                gain = target_rms / rms
                return audio * gain
        return audio
    
    def _ensure_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is stereo (2D array)."""
        if audio.ndim == 1:
            return np.stack([audio, audio])
        return audio
    
    # ==================== BASIC FILTERS ====================
    
    def lowpass(self, audio: np.ndarray, cutoff: float = 1000, 
                order: int = 4, filter_type: str = 'butter') -> np.ndarray:
        """
        Low-pass filter - attenuates frequencies above cutoff.
        
        Parameters:
            audio: Input audio signal
            cutoff: Cutoff frequency in Hz
            order: Filter order (higher = steeper rolloff)
            filter_type: 'butter' (default), 'cheby', 'ellip', 'bessel'
        
        Returns:
            Filtered audio
        """
        nyq = self.sample_rate / 2
        if cutoff >= nyq:
            cutoff = nyq * 0.99
        
        if filter_type == 'butter':
            b, a = butter(order, cutoff / nyq, btype='low')
            return lfilter(b, a, audio)
        elif filter_type == 'cheby':
            b, a = signal.cheby1(order, 0.5, cutoff / nyq, btype='low')
            return lfilter(b, a, audio)
        elif filter_type == 'ellip':
            b, a = signal.ellip(order, 0.5, 40, cutoff / nyq, btype='low')
            return lfilter(b, a, audio)
        else:
            b, a = signal.bessel(order, cutoff / nyq, btype='low')
            return lfilter(b, a, audio)
    
    def highpass(self, audio: np.ndarray, cutoff: float = 100,
                order: int = 4, filter_type: str = 'butter') -> np.ndarray:
        """
        High-pass filter - attenuates frequencies below cutoff.
        
        Parameters:
            audio: Input audio signal
            cutoff: Cutoff frequency in Hz
            order: Filter order
            filter_type: 'butter', 'cheby', 'ellip', 'bessel'
        
        Returns:
            Filtered audio
        """
        nyq = self.sample_rate / 2
        if cutoff <= 0:
            cutoff = 1
        
        if filter_type == 'butter':
            b, a = butter(order, cutoff / nyq, btype='high')
            return lfilter(b, a, audio)
        elif filter_type == 'cheby':
            b, a = signal.cheby1(order, 0.5, cutoff / nyq, btype='high')
            return lfilter(b, a, audio)
        elif filter_type == 'ellip':
            b, a = signal.ellip(order, 0.5, 40, cutoff / nyq, btype='high')
            return lfilter(b, a, audio)
        else:
            b, a = signal.bessel(order, cutoff / nyq, btype='high')
            return lfilter(b, a, audio)
    
    def bandpass(self, audio: np.ndarray, lowcut: float = 200,
                 highcut: float = 2000, order: int = 4) -> np.ndarray:
        """
        Band-pass filter - passes frequencies within range.
        
        Parameters:
            audio: Input audio signal
            lowcut: Lower cutoff frequency (Hz)
            highcut: Upper cutoff frequency (Hz)
            order: Filter order
        
        Returns:
            Filtered audio
        """
        nyq = self.sample_rate / 2
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, audio)
    
    def bandstop(self, audio: np.ndarray, lowcut: float = 50,
                 highcut: float = 60, order: int = 4) -> np.ndarray:
        """
        Band-stop (notch) filter - attenuates frequencies within range.
        
        Parameters:
            audio: Input audio signal
            lowcut: Lower cutoff frequency (Hz)
            highcut: Upper cutoff frequency (Hz)
            order: Filter order
        
        Returns:
            Filtered audio
        """
        nyq = self.sample_rate / 2
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(order, [low, high], btype='bandstop')
        return lfilter(b, a, audio)
    
    # ==================== PARAMETRIC EQ ====================
    
    def parametric_eq(self, audio: np.ndarray, 
                     bands: List[dict]) -> np.ndarray:
        """
        Multi-band parametric equalizer.
        
        Parameters:
            audio: Input audio signal
            bands: List of band configs:
                   [{'freq': 100, 'gain': 3, 'q': 1.0}, ...]
                   freq: Center frequency (Hz)
                   gain: Boost/cut in dB (+/- 12 typical)
                   q: Quality factor (higher = narrower bandwidth)
        
        Returns:
            Filtered audio
        """
        output = audio.copy()
        
        for band in bands:
            freq = band.get('freq', 1000)
            gain_db = band.get('gain', 0)
            q = band.get('q', 1.0)
            
            output = self._peaking_eq(output, freq, gain_db, q)
        
        return output
    
    def _peaking_eq(self, audio: np.ndarray, freq: float, 
                   gain_db: float, q: float = 1.0) -> np.ndarray:
        """Single-band peaking EQ filter."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2 * q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        return lfilter(b, a, audio)
    
    # ==================== SHELVING FILTERS ====================
    
    def low_shelf(self, audio: np.ndarray, freq: float = 200,
                 gain_db: float = 3) -> np.ndarray:
        """
        Low-frequency shelving filter (bass boost/cut).
        
        Parameters:
            audio: Input audio signal
            freq: Shelf corner frequency (Hz)
            gain_db: Boost/cut in dB (+/- 12 typical)
        
        Returns:
            Filtered audio
        """
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2)
        
        cos_w0 = np.cos(w0)
        
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        return lfilter(b, a, audio)
    
    def high_shelf(self, audio: np.ndarray, freq: float = 4000,
                   gain_db: float = 3) -> np.ndarray:
        """
        High-frequency shelving filter (treble boost/cut).
        
        Parameters:
            audio: Input audio signal
            freq: Shelf corner frequency (Hz)
            gain_db: Boost/cut in dB
        
        Returns:
            Filtered audio
        """
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2)
        
        cos_w0 = np.cos(w0)
        
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        return lfilter(b, a, audio)
    
    # ==================== SPECIAL FILTERS ====================
    
    def allpass(self, audio: np.ndarray, freq: float = 1000, 
               q: float = 0.7) -> np.ndarray:
        """
        All-pass filter - changes phase without amplitude.
        
        Parameters:
            audio: Input audio signal
            freq: Frequency of phase shift (Hz)
            q: Quality factor
        
        Returns:
            Filtered audio
        """
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2 * q)
        
        b0 = 1 - alpha
        b1 = -2 * np.cos(w0)
        b2 = 1 + alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        return lfilter(b, a, audio)
    
    def comb_filter(self, audio: np.ndarray, delay_ms: float = 10,
                   feedback: float = 0.5) -> np.ndarray:
        """
        Comb filter - creates resonant peaks.
        
        Parameters:
            audio: Input audio signal
            delay_ms: Delay in milliseconds
            feedback: Feedback amount (-1 to 1)
        
        Returns:
            Filtered audio
        """
        delay_samples = int(self.sample_rate * delay_ms / 1000)
        output = audio.copy()
        
        if delay_samples < len(audio):
            output[delay_samples:] += audio[:-delay_samples] * feedback
        
        return output
    
    def formant_filter(self, audio: np.ndarray, 
                      vowel: str = 'a') -> np.ndarray:
        """
        Formant filter - simulates vocal formants.
        
        Parameters:
            audio: Input audio signal
            vowel: Vowel sound ('a', 'e', 'i', 'o', 'u')
        
        Returns:
            Filtered audio
        """
        # Approximate formant frequencies for each vowel
        formants = {
            'a': [730, 1090, 2440],
            'e': [530, 1840, 2480],
            'i': [390, 1990, 2550],
            'o': [570, 840, 2410],
            'u': [440, 1020, 2240]
        }
        
        freqs = formants.get(vowel, formants['a'])
        output = audio.copy()
        
        for freq in freqs:
            # Bandpass around formant frequency
            output = self.bandpass(output, freq * 0.9, freq * 1.1, order=3)
        
        return output
    
    # ==================== MULTI-POINT FILTERS ====================
    
    def graphic_eq(self, audio: np.ndarray, 
                  gains_db: List[float]) -> np.ndarray:
        """
        Graphic equalizer with fixed bands.
        
        Parameters:
            audio: Input audio signal
            gains_db: List of gain values for each band (dB)
                     Standard: [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000] Hz
        
        Returns:
            Filtered audio
        """
        # Standard EQ frequencies
        frequencies = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        
        if len(gains_db) != len(frequencies):
            raise ValueError(f"Expected {len(frequencies)} gain values")
        
        output = audio.copy()
        
        for freq, gain in zip(frequencies, gains_db):
            if abs(gain) > 0.01:
                # Use wider Q for graphic EQ
                q = 1.4 if freq < 1000 else 3.0
                output = self._peaking_eq(output, freq, gain, q)
        
        return output
    
    def vocal_cut(self, audio: np.ndarray, 
                  sensitivity: float = 0.5) -> np.ndarray:
        """
        Reduce vocal frequencies (de-essing + presence reduction).
        
        Parameters:
            audio: Input audio signal
            sensitivity: Amount of reduction (0-1)
        
        Returns:
            Filtered audio with reduced vocals
        """
        output = audio.copy()
        
        # Reduce presence range (1-4kHz)
        presence_gain = -6 * sensitivity
        output = self._peaking_eq(output, 2000, presence_gain, q=2.0)
        output = self._peaking_eq(output, 3000, presence_gain * 0.8, q=2.5)
        
        # Slight high-pass to reduce proximity effect
        output = self.highpass(output, cutoff=80 + int(40 * sensitivity), order=2)
        
        return output
    
    def bass_boost(self, audio: np.ndarray, 
                   amount: float = 0.5, freq: float = 80) -> np.ndarray:
        """
        Boost low frequencies.
        
        Parameters:
            audio: Input audio signal
            amount: Boost amount (0-1)
            freq: Center frequency for boost
        
        Returns:
            Filtered audio
        """
        gain_db = 6 + 6 * amount  # 6-12 dB boost
        return self._peaking_eq(audio, freq, gain_db, q=0.5)
    
    def presence_boost(self, audio: np.ndarray, 
                       amount: float = 0.5) -> np.ndarray:
        """
        Boost presence/vocal clarity frequencies.
        
        Parameters:
            audio: Input audio signal
            amount: Boost amount (0-1)
        
        Returns:
            Filtered audio
        """
        gain_db = 3 + 3 * amount  # 3-6 dB boost
        
        output = audio.copy()
        output = self._peaking_eq(output, 2000, gain_db, q=1.5)
        output = self._peaking_eq(output, 4000, gain_db * 0.7, q=2.0)
        
        return output
    
    # ==================== ANALYSIS FILTERS ====================
    
    def get_frequency_response(self, filter_func, 
                              num_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frequency response of a filter.
        
        Parameters:
            filter_func: Filter coefficients (b, a) or callable
            num_points: Number of frequency points
        
        Returns:
            (frequencies, magnitudes) arrays
        """
        freqs = np.linspace(0, self.sample_rate / 2, num_points)
        w = 2 * np.pi * freqs / self.sample_rate
        
        if callable(filter_func):
            # Assume it's a method of this class
            # Create impulse response and get frequency response
            impulse = np.zeros(1024)
            impulse[0] = 1
            # This is simplified - actual implementation would use stored coeffs
            mag = np.ones(num_points)
        else:
            b, a = filter_func
            w_coef, h = signal.freqz(b, a, worN=w, whole=False)
            mag = np.abs(h)
        
        return freqs, mag
    
    def create_linkwitz_riley_cross(self, audio: np.ndarray, 
                                   crossover_freq: float,
                                   order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create Linkwitz-Riley crossover filters.
        
        Parameters:
            audio: Input audio signal
            crossover_freq: Crossover frequency (Hz)
            order: Filter order (2, 4, or 8)
        
        Returns:
            (low_pass, high_pass) audio
        """
        nyq = self.sample_rate / 2
        fc = crossover_freq / nyq
        
        # Linkwitz-Riley is Butterworth squared
        n = order // 2
        b, a = butter(n, fc, btype='low')
        
        # Low-pass
        low = lfilter(b, a, lfilter(b, a, audio))
        
        # High-pass (same, but subtract low from original)
        high = audio - low
        
        return low, high
    
    # ==================== ZERO-PHASE FILTERS ====================
    
    def lowpass_zerophase(self, audio: np.ndarray, cutoff: float = 1000,
                         order: int = 4) -> np.ndarray:
        """
        Zero-phase low-pass filter (no phase distortion).
        """
        nyq = self.sample_rate / 2
        b, a = butter(order, cutoff / nyq, btype='low')
        return sosfiltfilt(signal.butter(order, cutoff / nyq, btype='low', output='sos'), audio)
    
    def highpass_zerophase(self, audio: np.ndarray, cutoff: float = 100,
                          order: int = 4) -> np.ndarray:
        """
        Zero-phase high-pass filter (no phase distortion).
        """
        nyq = self.sample_rate / 2
        return sosfiltfilt(signal.butter(order, cutoff / nyq, btype='high', output='sos'), audio)
    
    def bandpass_zerophase(self, audio: np.ndarray, lowcut: float = 200,
                          highcut: float = 2000, order: int = 4) -> np.ndarray:
        """
        Zero-phase band-pass filter (no phase distortion).
        """
        nyq = self.sample_rate / 2
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        return sosfiltfilt(sos, audio)
    
    # ==================== CONVENIENCE PRESETS ====================
    
    def apply_filter_preset(self, audio: np.ndarray, 
                           preset: str) -> np.ndarray:
        """
        Apply predefined filter preset.
        
        Parameters:
            audio: Input audio signal
            preset: Preset name:
                   - 'radio': AM radio simulation
                   - 'telephone': Telephone bandwidth
                   - 'club': Bass-heavy club sound
                   - 'warm': Vintage warmth
                   - 'bright': Enhanced highs
                   - 'vocal': Enhanced vocals
                   - 'telephone_enhanced': Full-range with telephone Character
        
        Returns:
            Filtered audio
        """
        presets = {
            'radio': lambda x: self.bandpass(x, 200, 3500),
            'telephone': lambda x: self.bandpass(x, 300, 3400),
            'club': [
                {'freq': 60, 'gain': 4, 'q': 0.5},
                {'freq': 250, 'gain': 1, 'q': 0.7},
                {'freq': 4000, 'gain': 2, 'q': 1.0},
            ],
            'warm': [
                {'freq': 100, 'gain': 2, 'q': 0.5},
                {'freq': 3000, 'gain': -1, 'q': 1.0},
            ],
            'bright': [
                {'freq': 250, 'gain': -1, 'q': 0.7},
                {'freq': 4000, 'gain': 3, 'q': 1.0},
                {'freq': 8000, 'gain': 2, 'q': 1.5},
            ],
            'vocal': [
                {'freq': 200, 'gain': -1, 'q': 0.7},
                {'freq': 500, 'gain': 1, 'q': 1.0},
                {'freq': 2000, 'gain': 3, 'q': 1.5},
                {'freq': 4000, 'gain': 2, 'q': 2.0},
            ],
            'telephone_enhanced': lambda x: self._telephone_enhanced(x),
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        
        preset_func = presets[preset]
        
        if callable(preset_func):
            return preset_func(audio)
        else:
            return self.parametric_eq(audio, preset_func)
    
    def _telephone_enhanced(self, audio: np.ndarray) -> np.ndarray:
        """Telephone effect with subtle harmonics."""
        # Base telephone band
        output = self.bandpass(audio, 300, 3400)
        
        # Add slight warmth
        output = self._peaking_eq(output, 150, 1, q=0.5)
        
        return output


# Standalone functions for simple filtering
def simple_lowpass(audio: np.ndarray, cutoff: float, 
                  sample_rate: int = 44100) -> np.ndarray:
    """Simple low-pass filter (standalone function)."""
    filt = AudioFilter(sample_rate)
    return filt.lowpass(audio, cutoff)


def simple_highpass(audio: np.ndarray, cutoff: float,
                   sample_rate: int = 44100) -> np.ndarray:
    """Simple high-pass filter (standalone function)."""
    filt = AudioFilter(sample_rate)
    return filt.highpass(audio, cutoff)


def simple_bandpass(audio: np.ndarray, lowcut: float, highcut: float,
                   sample_rate: int = 44100) -> np.ndarray:
    """Simple band-pass filter (standalone function)."""
    filt = AudioFilter(sample_rate)
    return filt.bandpass(audio, lowcut, highcut)
