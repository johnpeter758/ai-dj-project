"""
Parametric Equalizer
Advanced parametric EQ with multiple filter bands for precise frequency control.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from scipy import signal
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt


@dataclass
class EQBand:
    """Represents a single parametric EQ band."""
    frequency: float       # Center frequency in Hz
    gain: float            # Gain in dB (-12 to +12)
    q: float               # Q factor (0.1 to 10)
    filter_type: str       # 'peaking', 'lowshelf', 'highshelf', 'lowpass', 'highpass', 'bandpass'
    enabled: bool = True
    
    def __post_init__(self):
        """Validate and clamp parameters."""
        self.gain = max(-12, min(12, self.gain))
        self.q = max(0.1, min(10, self.q))
        self.frequency = max(20, min(20000, self.frequency))
        
        valid_types = ['peaking', 'lowshelf', 'highshelf', 'lowpass', 'highpass', 'bandpass']
        if self.filter_type not in valid_types:
            self.filter_type = 'peaking'


@dataclass
class EQPreset:
    """Predefined EQ presets for different scenarios."""
    name: str
    bands: List[EQBand]
    
    @classmethod
    def flat(cls) -> 'EQPreset':
        """Flat EQ - no changes."""
        return cls(name="Flat", bands=[
            EQBand(frequency=100, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=300, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=1000, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=3000, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=10000, gain=0, q=1.0, filter_type='peaking'),
        ])
    
    @classmethod
    def bass_boost(cls) -> 'EQPreset':
        """Enhanced bass."""
        return cls(name="Bass Boost", bands=[
            EQBand(frequency=60, gain=6, q=0.7, filter_type='lowshelf'),
            EQBand(frequency=250, gain=3, q=1.0, filter_type='peaking'),
            EQBand(frequency=1000, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=3000, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=10000, gain=-2, q=1.0, filter_type='peaking'),
        ])
    
    @classmethod
    def vocal_presence(cls) -> 'EQPreset':
        """Enhanced vocal clarity."""
        return cls(name="Vocal Presence", bands=[
            EQBand(frequency=80, gain=-3, q=0.7, filter_type='lowshelf'),
            EQBand(frequency=300, gain=2, q=1.0, filter_type='peaking'),
            EQBand(frequency=1000, gain=1, q=1.4, filter_type='peaking'),
            EQBand(frequency=3500, gain=4, q=1.5, filter_type='peaking'),
            EQBand(frequency=8000, gain=2, q=1.0, filter_type='peaking'),
        ])
    
    @classmethod
    def treble_boost(cls) -> 'EQPreset':
        """Enhanced high frequencies."""
        return cls(name="Treble Boost", bands=[
            EQBand(frequency=100, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=1000, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=4000, gain=3, q=1.0, filter_type='peaking'),
            EQBand(frequency=8000, gain=5, q=1.2, filter_type='highshelf'),
            EQBand(frequency=12000, gain=4, q=1.0, filter_type='highshelf'),
        ])
    
    @classmethod
    def warmth(cls) -> 'EQPreset':
        """Warm, vintage-style sound."""
        return cls(name="Warmth", bands=[
            EQBand(frequency=60, gain=3, q=0.5, filter_type='lowshelf'),
            EQBand(frequency=250, gain=2, q=0.8, filter_type='peaking'),
            EQBand(frequency=1000, gain=-1, q=1.0, filter_type='peaking'),
            EQBand(frequency=4000, gain=-2, q=1.0, filter_type='peaking'),
            EQBand(frequency=10000, gain=-4, q=1.0, filter_type='highshelf'),
        ])
    
    @classmethod
    def presence_boost(cls) -> 'EQPreset':
        """Enhanced presence and clarity."""
        return cls(name="Presence Boost", bands=[
            EQBand(frequency=100, gain=-2, q=0.7, filter_type='peaking'),
            EQBand(frequency=500, gain=1, q=1.0, filter_type='peaking'),
            EQBand(frequency=2000, gain=3, q=1.5, filter_type='peaking'),
            EQBand(frequency=5000, gain=4, q=2.0, filter_type='peaking'),
            EQBand(frequency=10000, gain=2, q=1.5, filter_type='peaking'),
        ])
    
    @classmethod
    def bass_cut(cls) -> 'EQPreset':
        """Reduce bass (for busy mixes)."""
        return cls(name="Bass Cut", bands=[
            EQBand(frequency=60, gain=-6, q=0.5, filter_type='lowshelf'),
            EQBand(frequency=200, gain=-3, q=0.8, filter_type='peaking'),
            EQBand(frequency=1000, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=4000, gain=0, q=1.0, filter_type='peaking'),
            EQBand(frequency=10000, gain=0, q=1.0, filter_type='peaking'),
        ])
    
    @classmethod
    def telephone(cls) -> 'EQPreset':
        """Telephone-style filtered sound."""
        return cls(name="Telephone", bands=[
            EQBand(frequency=300, gain=0, q=0.8, filter_type='lowpass'),
            EQBand(frequency=3000, gain=0, q=0.8, filter_type='highpass'),
            EQBand(frequency=1000, gain=-2, q=1.0, filter_type='peaking'),
            EQBand(frequency=500, gain=2, q=1.0, filter_type='peaking'),
            EQBand(frequency=100, gain=0, q=1.0, filter_type='peaking'),
        ])
    
    @classmethod
    def lofi(cls) -> 'EQPreset':
        """Lo-fi retro sound."""
        return cls(name="Lo-Fi", bands=[
            EQBand(frequency=100, gain=-2, q=0.5, filter_type='lowshelf'),
            EQBand(frequency=700, gain=1, q=0.7, filter_type='peaking'),
            EQBand(frequency=2500, gain=-2, q=1.0, filter_type='peaking'),
            EQBand(frequency=6000, gain=-4, q=1.0, filter_type='highshelf'),
            EQBand(frequency=10000, gain=-6, q=1.0, filter_type='highshelf'),
        ])


class ParametricEQ:
    """
    Parametric Equalizer with multiple bands of adjustable filters.
    
    Supports:
    - Peaking filters (bell curves)
    - Low/high shelf filters
    - Low/high pass filters
    - Bandpass filters
    - Real-time filter coefficient calculation
    - Bypass mode
    - Preset management
    """
    
    # Default frequency bands for standard parametric EQ
    DEFAULT_BANDS = [
        EQBand(frequency=32, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=64, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=125, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=250, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=500, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=1000, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=2000, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=4000, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=8000, gain=0, q=1.0, filter_type='peaking'),
        EQBand(frequency=16000, gain=0, q=1.0, filter_type='peaking'),
    ]
    
    def __init__(self, sample_rate: int = 44100, bands: Optional[List[EQBand]] = None):
        """
        Initialize parametric EQ.
        
        Args:
            sample_rate: Audio sample rate in Hz
            bands: List of EQ bands (uses default 10-band if not provided)
        """
        self.sample_rate = sample_rate
        self.bands = bands if bands is not None else self.DEFAULT_BANDS.copy()
        self.bypass = False
        self._sos_cache: Dict[int, np.ndarray] = {}  # Cache for filter coefficients
        
    def _design_peaking_filter(self, freq: float, gain: float, q: float) -> np.ndarray:
        """
        Design a peaking (bell) filter using second-order sections.
        
        Args:
            freq: Center frequency in Hz
            gain: Gain in dB
            q: Q factor
            
        Returns:
            Second-order sections representation
        """
        w0 = 2 * np.pi * freq / self.sample_rate
        A = 10 ** (gain / 40)  # sqrt of amplitude
        alpha = np.sin(w0) / (2 * q)
        
        # Peaking filter coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        return signal.tf2sos(b, a)
    
    def _design_shelf_filter(self, freq: float, gain: float, q: float, 
                             shelf_type: str) -> np.ndarray:
        """
        Design a shelving filter (low or high shelf).
        
        Args:
            freq: Corner frequency in Hz
            gain: Gain in dB
            q: Q factor
            shelf_type: 'low' or 'high'
            
        Returns:
            Second-order sections representation
        """
        w0 = 2 * np.pi * freq / self.sample_rate
        A = 10 ** (gain / 40)
        alpha = np.sin(w0) / (2 * q)
        
        if shelf_type == 'low':
            # Low shelf coefficients
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        else:  # high shelf
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        
        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        return signal.tf2sos(b, a)
    
    def _design_pass_filter(self, freq: float, q: float, 
                           pass_type: str) -> np.ndarray:
        """
        Design high-pass or low-pass filter.
        
        Args:
            freq: Cutoff frequency in Hz
            q: Q factor
            pass_type: 'low' or 'high'
            
        Returns:
            Second-order sections representation
        """
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2 * q)
        
        if pass_type == 'low':
            b0 = (1 - np.cos(w0)) / 2
            b1 = 1 - np.cos(w0)
            b2 = (1 - np.cos(w0)) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
        else:  # high
            b0 = (1 + np.cos(w0)) / 2
            b1 = -(1 + np.cos(w0))
            b2 = (1 + np.cos(w0)) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        return signal.tf2sos(b, a)
    
    def _design_bandpass_filter(self, freq: float, q: float) -> np.ndarray:
        """Design a bandpass filter."""
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2 * q)
        
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        return signal.tf2sos(b, a)
    
    def _get_filter_sos(self, band: EQBand) -> np.ndarray:
        """Get or compute filter coefficients for a band."""
        # Create cache key from band parameters
        cache_key = hash((
            round(band.frequency, 1), 
            round(band.gain, 1), 
            round(band.q, 2), 
            band.filter_type
        ))
        
        if cache_key in self._sos_cache:
            return self._sos_cache[cache_key]
        
        # Design the appropriate filter
        if band.filter_type == 'peaking':
            sos = self._design_peaking_filter(band.frequency, band.gain, band.q)
        elif band.filter_type == 'lowshelf':
            sos = self._design_shelf_filter(band.frequency, band.gain, band.q, 'low')
        elif band.filter_type == 'highshelf':
            sos = self._design_shelf_filter(band.frequency, band.gain, band.q, 'high')
        elif band.filter_type == 'lowpass':
            sos = self._design_pass_filter(band.frequency, band.q, 'low')
        elif band.filter_type == 'highpass':
            sos = self._design_pass_filter(band.frequency, band.q, 'high')
        elif band.filter_type == 'bandpass':
            sos = self._design_bandpass_filter(band.frequency, band.q)
        else:
            # Default to peaking
            sos = self._design_peaking_filter(band.frequency, band.gain, band.q)
        
        # Cache the result
        self._sos_cache[cache_key] = sos
        return sos
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply parametric EQ to audio signal.
        
        Args:
            audio: Input audio (mono or stereo). Shape: (samples,) or (samples, channels)
            
        Returns:
            EQ-processed audio
        """
        if self.bypass:
            return audio.copy()
        
        # Handle stereo
        is_stereo = audio.ndim == 2
        if is_stereo:
            output = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                output[:, ch] = self._process_mono(audio[:, ch])
        else:
            output = self._process_mono(audio)
        
        return output
    
    def _process_mono(self, audio: np.ndarray) -> np.ndarray:
        """Process mono audio through all EQ bands."""
        output = audio.copy()
        
        for band in self.bands:
            if not band.enabled:
                continue
            
            # Skip if no effect
            if band.filter_type == 'peaking' and abs(band.gain) < 0.1:
                continue
            if band.filter_type in ['lowshelf', 'highshelf'] and abs(band.gain) < 0.1:
                continue
            
            try:
                sos = self._get_filter_sos(band)
                output = sosfiltfilt(sos, output)
            except Exception:
                # Fallback: skip this band if filter design fails
                continue
        
        return output
    
    def set_band(self, index: int, frequency: Optional[float] = None,
                 gain: Optional[float] = None, q: Optional[float] = None,
                 enabled: Optional[bool] = None) -> None:
        """
        Update a specific EQ band.
        
        Args:
            index: Band index (0-based)
            frequency: New center frequency in Hz
            gain: New gain in dB
            q: New Q factor
            enabled: Enable/disable band
        """
        if index < 0 or index >= len(self.bands):
            raise IndexError(f"Band index {index} out of range")
        
        band = self.bands[index]
        
        if frequency is not None:
            band.frequency = max(20, min(20000, frequency))
        if gain is not None:
            band.gain = max(-12, min(12, gain))
        if q is not None:
            band.q = max(0.1, min(10, q))
        if enabled is not None:
            band.enabled = enabled
        
        # Clear cache when parameters change
        self._sos_cache.clear()
    
    def set_all_gains(self, gains: List[float]) -> None:
        """
        Set gain for all bands.
        
        Args:
            gains: List of gains in dB (one per band)
        """
        for i, gain in enumerate(gains):
            if i < len(self.bands):
                self.bands[i].gain = max(-12, min(12, gain))
        self._sos_cache.clear()
    
    def get_frequency_response(self, frequencies: Optional[np.ndarray] = None,
                               n_freqs: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency response of the EQ.
        
        Args:
            frequencies: Specific frequencies to evaluate (Hz)
            n_freqs: Number of frequencies if not specified
            
        Returns:
            Tuple of (frequencies, magnitudes in dB)
        """
        if frequencies is None:
            frequencies = np.logspace(np.log10(20), np.log10(20000), n_freqs)
        
        magnitudes = np.ones_like(frequencies)
        
        for band in self.bands:
            if not band.enabled:
                continue
            
            try:
                sos = self._get_filter_sos(band)
                # Get frequency response
                w, h = signal.sosfreqz(sos, worN=2 * np.pi * frequencies / self.sample_rate)
                magnitudes *= np.abs(h)
            except Exception:
                continue
        
        # Convert to dB
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)
        
        return frequencies, magnitudes_db
    
    def get_band_gains(self) -> List[float]:
        """Get current gains for all bands."""
        return [band.gain for band in self.bands]
    
    def get_band_frequencies(self) -> List[float]:
        """Get frequencies for all bands."""
        return [band.frequency for band in self.bands]
    
    def bypass_on(self) -> None:
        """Enable bypass."""
        self.bypass = True
    
    def bypass_off(self) -> None:
        """Disable bypass."""
        self.bypass = False
    
    def toggle_bypass(self) -> bool:
        """Toggle bypass state. Returns new bypass state."""
        self.bypass = not self.bypass
        return self.bypass
    
    def load_preset(self, preset: EQPreset) -> None:
        """
        Load an EQ preset.
        
        Args:
            preset: EQPreset to load
        """
        self.bands = preset.bands.copy()
        self._sos_cache.clear()
    
    def apply_preset_by_name(self, name: str) -> bool:
        """
        Apply a preset by name.
        
        Args:
            name: Preset name (case-insensitive)
            
        Returns:
            True if preset was found and applied
        """
        presets = {
            'flat': EQPreset.flat,
            'bass boost': EQPreset.bass_boost,
            'bass_boost': EQPreset.bass_boost,
            'vocal presence': EQPreset.vocal_presence,
            'vocal_presence': EQPreset.vocal_presence,
            'treble boost': EQPreset.treble_boost,
            'treble_boost': EQPreset.treble_boost,
            'warmth': EQPreset.warmth,
            'presence boost': EQPreset.presence_boost,
            'presence_boost': EQPreset.presence_boost,
            'bass cut': EQPreset.bass_cut,
            'bass_cut': EQPreset.bass_cut,
            'telephone': EQPreset.telephone,
            'lofi': EQPreset.lofi,
            'lo-fi': EQPreset.lofi,
        }
        
        key = name.lower()
        if key in presets:
            self.load_preset(presets[key]())
            return True
        return False
    
    def reset(self) -> None:
        """Reset all bands to flat (0 dB)."""
        for band in self.bands:
            band.gain = 0
            band.enabled = True
        self._sos_cache.clear()
        self.bypass = False
    
    def __repr__(self) -> str:
        return (f"ParametricEQ(sample_rate={self.sample_rate}, "
                f"bands={len(self.bands)}, bypass={self.bypass})")


class DynamicParametricEQ(ParametricEQ):
    """
    Dynamic parametric EQ with frequency-dependent gain reduction.
    Useful for frequency-specific compression/expansion.
    """
    
    def __init__(self, sample_rate: int = 44100, bands: Optional[List[EQBand]] = None,
                 threshold_db: float = -20, ratio: float = 4.0,
                 attack_ms: float = 10, release_ms: float = 100):
        """
        Initialize dynamic parametric EQ.
        
        Args:
            sample_rate: Audio sample rate
            bands: EQ bands
            threshold_db: Threshold in dB
            ratio: Compression ratio
            attack_ms: Attack time in ms
            release_ms: Release time in ms
        """
        super().__init__(sample_rate, bands)
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        
        # Envelope follower state
        self._envelope = 0.0
        self._alpha_attack = np.exp(-1.0 / (attack_ms * sample_rate / 1000))
        self._alpha_release = np.exp(-1.0 / (release_ms * sample_rate / 1000))
    
    def _compute_gain_reduction(self, input_level_db: float) -> float:
        """Compute gain reduction based on input level."""
        if input_level_db > self.threshold_db:
            # Above threshold: apply compression
            excess = input_level_db - self.threshold_db
            reduction = excess * (1 - 1 / self.ratio)
            return -reduction
        return 0
    
    def process_dynamic(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio with dynamic EQ (frequency-specific compression).
        
        Note: This is a simplified version. Full implementation would analyze
        each frequency band separately and apply band-specific dynamics.
        """
        if self.bypass:
            return audio.copy()
        
        # For each band, apply dynamic gain
        output = audio.copy()
        
        for band in self.bands:
            if not band.enabled or band.filter_type not in ['peaking', 'lowshelf', 'highshelf']:
                continue
            
            # Simple envelope following
            input_rms = np.sqrt(np.mean(audio ** 2))
            input_db = 20 * np.log10(input_rms + 1e-10)
            
            # Update envelope
            if input_db > self._envelope:
                self._envelope = self._alpha_attack * self._envelope + (1 - self._alpha_attack) * input_db
            else:
                self._envelope = self._alpha_release * self._envelope + (1 - self._alpha_release) * input_db
            
            # Compute gain reduction
            gr = self._compute_gain_reduction(self._envelope)
            
            # Apply to band
            adjusted_gain = band.gain + gr
            adjusted_gain = max(-12, min(12, adjusted_gain))
            
            # Create temporary band with adjusted gain
            temp_band = EQBand(
                frequency=band.frequency,
                gain=adjusted_gain,
                q=band.q,
                filter_type=band.filter_type,
                enabled=True
            )
            
            try:
                sos = self._get_filter_sos(temp_band)
                output = sosfiltfilt(sos, output)
            except Exception:
                continue
        
        return output


# Convenience functions

def create_eq(sample_rate: int = 44100, num_bands: int = 10) -> ParametricEQ:
    """
    Create a parametric EQ with standard frequency bands.
    
    Args:
        sample_rate: Audio sample rate
        num_bands: Number of EQ bands (default 10)
        
    Returns:
        ParametricEQ instance
    """
    # Standard frequencies: 31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k (ISO frequencies)
    iso_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    
    bands = [
        EQBand(frequency=f, gain=0, q=1.0, filter_type='peaking')
        for f in iso_freqs[:num_bands]
    ]
    
    return ParametricEQ(sample_rate=sample_rate, bands=bands)


def apply_eq_curve(audio: np.ndarray, sample_rate: int,
                   curve: Dict[str, float]) -> np.ndarray:
    """
    Apply a simple EQ curve to audio.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        curve: Dictionary mapping frequency names to gain values.
               Keys: 'sub', 'bass', 'low_mid', 'mid', 'high_mid', 'presence', 'brilliance'
               
    Returns:
        Processed audio
    """
    # Frequency ranges for curve keys
    freq_ranges = {
        'sub': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 500),
        'mid': (500, 2000),
        'high_mid': (2000, 4000),
        'presence': (4000, 6000),
        'brilliance': (6000, 20000)
    }
    
    eq = ParametricEQ(sample_rate=sample_rate)
    
    # Apply gains to corresponding bands
    for key, gain in curve.items():
        if key in freq_ranges:
            # Find closest band
            low, high = freq_ranges[key]
            center = (low + high) / 2
            
            # Apply to band closest to center frequency
            band_freqs = eq.get_band_frequencies()
            closest_idx = min(range(len(band_freqs)), 
                            key=lambda i: abs(band_freqs[i] - center))
            
            eq.set_band(closest_idx, gain=gain)
    
    return eq.process(audio)


if __name__ == "__main__":
    # Demo/test
    import soundfile as sf
    
    # Create a simple test signal (sine wave sweep)
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Frequency sweep from 20Hz to 20kHz
    freq_sweep = 20 * (1000 ** (t / duration))  # Exponential sweep
    test_signal = np.sin(2 * np.pi * freq_sweep * t).astype(np.float32)
    
    # Test parametric EQ
    eq = ParametricEQ(sample_rate=sample_rate)
    print(f"Created: {eq}")
    
    # Apply bass boost preset
    eq.load_preset(EQPreset.bass_boost())
    print(f"Loaded preset: Bass Boost")
    
    # Process audio
    output = eq.process(test_signal)
    
    # Get frequency response
    freqs, mag_db = eq.get_frequency_response()
    print(f"Frequency response at 1kHz: {mag_db[np.argmin(np.abs(freqs - 1000))]:.1f} dB")
    print(f"Frequency response at 60Hz: {mag_db[np.argmin(np.abs(freqs - 60))]:.1f} dB")
    
    # Test bypass
    eq.bypass_on()
    output_bypassed = eq.process(test_signal)
    eq.bypass_off()
    
    print("Parametric EQ module ready!")
