"""
Dynamic Range Compressor for AI DJ Project
Real-time and offline dynamic range compression with multiple modes
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt


class DynamicCompressor:
    """
    Dynamic range compressor with multiple compression modes.
    
    Supports:
    - Standard feedforward compression
    - Feedback compression
    - Multiband compression
    - Soft/hard knee options
    - Peak/RMS detection
    - Sidechain input support
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._envelope_state = 0.0
        self._gain_reduction = 0.0
        self._buffer = None
        
        # Default parameters
        self.threshold_db = -20.0
        self.ratio = 4.0
        self.attack_ms = 10.0
        self.release_ms = 100.0
        self.knee_db = 6.0
        self.makeup_gain_db = 0.0
        self.detector_mode = "rms"  # "peak" or "rms"
        self.knee_type = "soft"     # "soft" or "hard"
        
        # Pre-computed coefficients
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Update filter coefficients based on attack/release settings."""
        # Convert to time constants for envelope follower
        self._attack_coef = np.exp(-1.0 / (self.attack_ms * self.sample_rate / 1000))
        self._release_coef = np.exp(-1.0 / (self.release_ms * self.sample_rate / 1000))
    
    def set_parameters(self, threshold_db: float = None, ratio: float = None,
                       attack_ms: float = None, release_ms: float = None,
                       knee_db: float = None, makeup_gain_db: float = None,
                       detector_mode: str = None, knee_type: str = None):
        """
        Set compressor parameters.
        
        Parameters:
            threshold_db: Level in dB above which compression starts (-60 to 0)
            ratio: Compression ratio (1:1 to 20:1)
            attack_ms: Attack time in milliseconds (0.1 to 100)
            release_ms: Release time in milliseconds (10 to 1000)
            knee_db: Knee width in dB (0 for hard knee, 1 to 24 for soft)
            makeup_gain_db: Makeup gain in dB (-24 to 24)
            detector_mode: "peak" or "rms"
            knee_type: "soft" or "hard"
        """
        if threshold_db is not None:
            self.threshold_db = np.clip(threshold_db, -60.0, 0.0)
        if ratio is not None:
            self.ratio = np.clip(ratio, 1.0, 20.0)
        if attack_ms is not None:
            self.attack_ms = np.clip(attack_ms, 0.1, 100.0)
            self._update_coefficients()
        if release_ms is not None:
            self.release_ms = np.clip(release_ms, 10.0, 1000.0)
            self._update_coefficients()
        if knee_db is not None:
            self.knee_db = np.clip(knee_db, 0.0, 24.0)
        if makeup_gain_db is not None:
            self.makeup_gain_db = np.clip(makeup_gain_db, -24.0, 24.0)
        if detector_mode is not None:
            self.detector_mode = detector_mode
        if knee_type is not None:
            self.knee_type = knee_type
    
    def _compute_gain_db(self, input_level_db: float) -> float:
        """
        Compute gain reduction in dB based on input level and compressor settings.
        """
        # Apply knee
        if self.knee_type == "soft" and self.knee_db > 0:
            threshold_low = self.threshold_db - self.knee_db / 2
            threshold_high = self.threshold_db + self.knee_db / 2
            
            if input_level_db < threshold_low:
                # Below knee: no compression
                gain_reduction = 0.0
            elif input_level_db > threshold_high:
                # Above knee: full compression
                excess = input_level_db - self.threshold_db
                gain_reduction = excess * (1 - 1 / self.ratio)
            else:
                # In knee: gradual transition
                excess = input_level_db - threshold_low
                knee_range = self.knee_db
                gain_reduction = (excess / knee_range) ** 2 * (1 - 1 / self.ratio) * excess
        else:
            # Hard knee
            if input_level_db > self.threshold_db:
                excess = input_level_db - self.threshold_db
                gain_reduction = excess * (1 - 1 / self.ratio)
            else:
                gain_reduction = 0.0
        
        return gain_reduction
    
    def _detect_level(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect signal level using peak or RMS detection.
        """
        if self.detector_mode == "rms":
            # RMS detection (smoother)
            level = np.sqrt(np.mean(audio ** 2))
        else:
            # Peak detection (faster response)
            level = np.max(np.abs(audio))
        
        return level
    
    def process(self, audio: np.ndarray, sidechain: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process audio through the compressor.
        
        Parameters:
            audio: Input audio signal (mono or stereo)
            sidechain: Optional sidechain signal for external triggering
            
        Returns:
            Compressed audio signal
        """
        # Ensure stereo handling
        if audio.ndim == 1:
            return self._process_mono(audio, sidechain)
        else:
            # Process each channel
            output = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                output[ch] = self._process_mono(audio[ch], 
                    sidechain[ch] if sidechain is not None and sidechain.ndim > 1 else sidechain)
            return output
    
    def _process_mono(self, audio: np.ndarray, sidechain: Optional[np.ndarray] = None) -> np.ndarray:
        """Process mono audio through compressor."""
        # Use sidechain or input as detector source
        detector_input = sidechain if sidechain is not None else audio
        
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Get current sample level
            if self.detector_mode == "rms":
                # Use small buffer for RMS
                window_size = min(512, i + 1)
                if window_size > 0:
                    level = np.sqrt(np.mean(detector_input[max(0, i-window_size+1):i+1] ** 2))
                else:
                    level = 0.0
            else:
                level = abs(detector_input[i])
            
            # Convert to dB
            if level > 1e-10:
                level_db = 20 * np.log10(level)
            else:
                level_db = -100.0
            
            # Compute gain reduction
            gain_reduction_db = self._compute_gain_db(level_db)
            
            # Apply envelope (attack/release)
            if gain_reduction_db > self._envelope_state:
                # Attack (gain reduction increasing)
                self._envelope_state = (self._attack_coef * self._envelope_state + 
                                       (1 - self._attack_coef) * gain_reduction_db)
            else:
                # Release (gain reduction decreasing)
                self._envelope_state = (self._release_coef * self._envelope_state + 
                                       (1 - self._release_coef) * gain_reduction_db)
            
            # Apply gain
            gain_linear = 10 ** ((self._envelope_state + self.makeup_gain_db) / 20)
            output[i] = audio[i] * gain_linear
        
        # Soft clipping for limiting
        output = self._soft_clip(output)
        
        return output
    
    def _soft_clip(self, audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """Apply soft clipping to prevent harsh distortion."""
        # TanH soft clipping
        return np.tanh(audio / threshold) * threshold
    
    def get_gain_reduction(self) -> float:
        """Get current gain reduction in dB."""
        return -self._envelope_state
    
    def reset(self):
        """Reset compressor state."""
        self._envelope_state = 0.0
        self._gain_reduction = 0.0


class MultibandCompressor:
    """
    Multiband dynamic range compressor.
    Splits audio into frequency bands and compresses each independently.
    """
    
    def __init__(self, sample_rate: int = 44100, num_bands: int = 4):
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        
        # Default crossover frequencies (Hz)
        self.crossover_frequencies = [100, 1000, 4000, 10000]
        
        # Create compressors for each band
        self.bands = [DynamicCompressor(sample_rate) for _ in range(num_bands)]
        
        # Default band settings
        self._set_default_bands()
        
        # Crossover filters
        self._design_crossover_filters()
    
    def _set_default_bands(self):
        """Set default compression settings for each band."""
        # Bass: slower attack, higher threshold
        self.bands[0].set_parameters(
            threshold_db=-18, ratio=3, attack_ms=20, release_ms=200,
            knee_db=6, makeup_gain_db=2
        )
        
        # Low-mids: balanced
        self.bands[1].set_parameters(
            threshold_db=-20, ratio=4, attack_ms=10, release_ms=150,
            knee_db=6, makeup_gain_db=0
        )
        
        # High-mids: faster
        self.bands[2].set_parameters(
            threshold_db=-22, ratio=4, attack_ms=5, release_ms=100,
            knee_db=4, makeup_gain_db=-1
        )
        
        # Highs: gentle compression
        self.bands[3].set_parameters(
            threshold_db=-24, ratio=2, attack_ms=3, release_ms=80,
            knee_db=3, makeup_gain_db=-2
        )
    
    def _design_crossover_filters(self):
        """Design Linkwitz-Riley crossover filters."""
        self.lowpass_filters = []
        self.highpass_filters = []
        
        for i, freq in enumerate(self.crossover_frequencies):
            # 4th order Linkwitz-Riley (24dB/octave)
            b, a = butter(4, freq / (self.sample_rate / 2), btype='low')
            self.lowpass_filters.append((b, a))
            
            b, a = butter(4, freq / (self.sample_rate / 2), btype='high')
            self.highpass_filters.append((b, a))
    
    def set_band_parameters(self, band: int, **kwargs):
        """Set parameters for a specific band."""
        if 0 <= band < self.num_bands:
            self.bands[band].set_parameters(**kwargs)
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through multiband compressor.
        
        Parameters:
            audio: Input audio signal
            
        Returns:
            Compressed audio signal
        """
        if audio.ndim == 1:
            return self._process_mono(audio)
        else:
            output = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                output[ch] = self._process_mono(audio[ch])
            return output
    
    def _process_mono(self, audio: np.ndarray) -> np.ndarray:
        """Process mono audio through multiband compressor."""
        # Split into bands
        bands = self._split_bands(audio)
        
        # Compress each band
        compressed_bands = [self.bands[i].process(bands[i]) for i in range(self.num_bands)]
        
        # Recombine bands
        return self._combine_bands(compressed_bands)
    
    def _split_bands(self, audio: np.ndarray) -> List[np.ndarray]:
        """Split audio into frequency bands."""
        bands = []
        current = audio.copy()
        
        for i in range(self.num_bands):
            # Apply highpass to get high frequencies
            high = lfilter(self.highpass_filters[i][0], 
                          self.highpass_filters[i][1], 
                          current)
            
            # Store as this band's content
            bands.append(high)
            
            # Lowpass to keep low frequencies for next band
            current = lfilter(self.lowpass_filters[i][0], 
                             self.lowpass_filters[i][1], 
                             current)
        
        # Add final low frequency band
        bands.insert(0, current)
        
        return bands
    
    def _combine_bands(self, bands: List[np.ndarray]) -> np.ndarray:
        """Combine frequency bands back into full range signal."""
        # Sum all bands
        output = np.zeros_like(bands[0])
        for band in bands:
            output += band
        
        return output
    
    def get_band_levels(self) -> List[float]:
        """Get current gain reduction levels for each band."""
        return [band.get_gain_reduction() for band in self.bands]


class VintageCompressor:
    """
    Vintage-style compressor emulation with various modes.
    Models classic analog compressor behaviors.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.mode = "vca"  # vca, opt FET, vari-mu
        
        # VCA mode parameters
        self.threshold_db = -20.0
        self.ratio = 4.0
        self.attack_ms = 10.0
        self.release_ms = 100.0
        self.makeup_gain_db = 0.0
        
        # State
        self._envelope = 0.0
        self._filter_state = 0.0
        
        # Update coefficients
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Update envelope follower coefficients based on mode."""
        if self.mode == "vca":
            # Fast, precise VCA
            self._attack_coef = np.exp(-1.0 / (self.attack_ms * self.sample_rate / 1000))
            self._release_coef = np.exp(-1.0 / (self.release_ms * self.sample_rate / 1000))
        elif self.mode == "opt":
            # Optical compressor: slower, smoother
            self._attack_coef = np.exp(-1.0 / (max(self.attack_ms, 20) * self.sample_rate / 1000))
            self._release_coef = np.exp(-1.0 / (self.release_ms * 2 * self.sample_rate / 1000))
        else:  # vari-mu
            # Variable mu: slower, more gradual
            self._attack_coef = np.exp(-1.0 / (self.attack_ms * 1.5 * self.sample_rate / 1000))
            self._release_coef = np.exp(-1.0 / (self.release_ms * 0.8 * self.sample_rate / 1000))
    
    def set_mode(self, mode: str):
        """Set compressor mode: 'vca', 'opt', or 'vari-mu'."""
        if mode in ["vca", "opt", "vari-mu"]:
            self.mode = mode
            self._update_coefficients()
    
    def set_parameters(self, **kwargs):
        """Set compressor parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._update_coefficients()
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through vintage compressor.
        
        Parameters:
            audio: Input audio signal
            
        Returns:
            Compressed audio signal
        """
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Input level
            input_level = abs(audio[i])
            input_db = 20 * np.log10(max(input_level, 1e-10))
            
            # Gain computation based on mode
            if input_db > self.threshold_db:
                excess = input_db - self.threshold_db
                
                if self.mode == "vca":
                    # Standard VCA compression
                    gain_reduction = excess * (1 - 1 / self.ratio)
                elif self.mode == "opt":
                    # Optical: more gradual
                    gain_reduction = excess * (1 - 1 / self.ratio) * 0.8
                else:  # vari-mu
                    # Variable mu: soft knee, gentle ratio
                    gain_reduction = excess * 0.7 * (1 - np.exp(-excess / 10))
            else:
                gain_reduction = 0.0
            
            # Envelope follower
            if gain_reduction > self._envelope:
                self._envelope = (self._attack_coef * self._envelope + 
                                 (1 - self._attack_coef) * gain_reduction)
            else:
                self._envelope = (self._release_coef * self._envelope + 
                                 (1 - self._release_coef) * gain_reduction)
            
            # Apply gain with makeup
            gain_linear = 10 ** ((self._envelope + self.makeup_gain_db) / 20)
            output[i] = audio[i] * gain_linear
        
        # Slight coloration for vintage feel
        output = self._add_harmonics(output)
        
        return output
    
    def _add_harmonics(self, audio: np.ndarray) -> np.ndarray:
        """Add subtle harmonic distortion for vintage character."""
        # Second harmonic at very low level
        harmonics = audio + 0.002 * audio ** 2 * np.sign(audio)
        return harmonics * 0.998  # Slight overall reduction


# Convenience function for simple compression
def compress(audio: np.ndarray, threshold_db: float = -20, ratio: float = 4,
            attack_ms: float = 10, release_ms: float = 100, 
            makeup_gain_db: float = 0, sample_rate: int = 44100) -> np.ndarray:
    """
    Simple one-shot audio compression.
    
    Parameters:
        audio: Input audio signal
        threshold_db: Threshold in dB
        ratio: Compression ratio
        attack_ms: Attack time in ms
        release_ms: Release time in ms
        makeup_gain_db: Makeup gain in dB
        sample_rate: Audio sample rate
        
    Returns:
        Compressed audio
    """
    compressor = DynamicCompressor(sample_rate)
    compressor.set_parameters(
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        makeup_gain_db=makeup_gain_db
    )
    return compressor.process(audio)


# Example usage and testing
if __name__ == "__main__":
    # Test the compressor
    import time
    
    # Generate test signal: sine sweep with impulses
    sample_rate = 44100
    duration = 3.0
    samples = int(sample_rate * duration)
    
    # Create test signal: varying amplitude
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    # Add some louder sections
    loud_start = int(1.0 * sample_rate)
    loud_end = int(1.5 * sample_rate)
    audio[loud_start:loud_end] = np.sin(2 * np.pi * 440 * t[loud_start:loud_end]) * 0.8
    
    # Add impulses
    impulse_positions = [int(0.5 * sample_rate), int(2.0 * sample_rate)]
    for pos in impulse_positions:
        audio[pos:pos+100] += np.sin(2 * np.pi * 1000 * t[pos:pos+100]) * 0.9
    
    print("Testing DynamicCompressor...")
    
    # Test basic compressor
    compressor = DynamicCompressor(sample_rate)
    compressor.set_parameters(
        threshold_db=-15,
        ratio=6,
        attack_ms=5,
        release_ms=50,
        makeup_gain_db=3
    )
    
    start_time = time.time()
    compressed = compressor.process(audio)
    elapsed = time.time() - start_time
    
    print(f"Processed {samples} samples in {elapsed*1000:.2f}ms")
    print(f"Gain reduction: {compressor.get_gain_reduction():.2f} dB")
    
    # Test multiband compressor
    print("\nTesting MultibandCompressor...")
    multiband = MultibandCompressor(sample_rate, num_bands=4)
    compressed_multi = multiband.process(audio)
    print(f"Band levels: {multiband.get_band_levels()}")
    
    # Test vintage compressor
    print("\nTesting VintageCompressor...")
    vintage = VintageCompressor(sample_rate)
    vintage.set_mode("opt")
    vintage.set_parameters(threshold_db=-18, ratio=4, makeup_gain_db=2)
    compressed_vintage = vintage.process(audio)
    
    print("\nAll tests completed!")
