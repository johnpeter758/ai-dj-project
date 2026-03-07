"""
Stereo Widener for AI DJ Project
Provides stereo width enhancement using mid/side, Haas effect, and phase manipulation.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import signal
from scipy.signal import butter, lfilter


class StereoWidener:
    """
    Stereo width enhancement processor using multiple techniques.
    Supports mid/side processing, Haas effect, and phase manipulation.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._mid_buffer = None
        self._side_buffer = None
    
    def process(self, audio: np.ndarray, width: float = 1.0,
                technique: str = "mid_side") -> np.ndarray:
        """
        Apply stereo widening to audio.
        
        Parameters:
            audio: Stereo audio array (N, 2) or (2, N)
            width: Width factor (0.0 = mono, 1.0 = normal, >1.0 = wider)
            technique: Widening technique ("mid_side", "haas", "phase")
        
        Returns:
            Stereo audio with enhanced width
        """
        # Ensure stereo input
        if audio.ndim == 1:
            # Mono - convert to stereo first
            audio = np.stack([audio, audio], axis=0).T
        
        # Handle (2, N) format
        if audio.shape[0] == 2 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        
        if technique == "mid_side":
            return self._mid_side_process(audio, width)
        elif technique == "haas":
            return self._haas_effect(audio, width)
        elif technique == "phase":
            return self._phase_manipulation(audio, width)
        else:
            raise ValueError(f"Unknown technique: {technique}")
    
    def _mid_side_process(self, audio: np.ndarray, width: float) -> np.ndarray:
        """
        Mid/Side stereo widening.
        Boosts the side signal to increase stereo width.
        """
        # Convert to mid/side
        left = audio[:, 0]
        right = audio[:, 1]
        
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        
        # Apply width to side channel
        side = side * width
        
        # Convert back to left/right
        left_out = mid + side
        right_out = mid - side
        
        # Normalize to prevent clipping
        max_val = max(np.abs(left_out).max(), np.abs(right_out).max())
        if max_val > 1.0:
            left_out /= max_val
            right_out /= max_val
        
        return np.stack([left_out, right_out], axis=1)
    
    def _haas_effect(self, audio: np.ndarray, width: float) -> np.ndarray:
        """
        Haas effect stereo widening.
        Adds a slight delay to one channel to create spaciousness.
        """
        # Delay in samples (width maps to 0-30ms)
        delay_samples = int(width * 0.030 * self.sample_rate)
        delay_samples = max(1, min(delay_samples, 1000))  # Clamp to 1-1000
        
        left = audio[:, 0].copy()
        right = audio[:, 1].copy()
        
        # Apply delay to right channel
        if delay_samples > 0:
            delayed_right = np.zeros_like(right)
            delayed_right[delay_samples:] = right[:-delay_samples]
            delayed_right[:delay_samples] = right[:delay_samples]
            right = delayed_right
        
        # Slight high-shelf boost on delayed side for clarity
        b, a = butter(2, 3000 / (self.sample_rate / 2), 'high')
        right = lfilter(b, a, right)
        
        return np.stack([left, right], axis=1)
    
    def _phase_manipulation(self, audio: np.ndarray, width: float) -> np.ndarray:
        """
        Phase manipulation stereo widening.
        Rotates phase slightly differently for each channel.
        """
        left = audio[:, 0]
        right = audio[:, 1]
        
        # Generate phase rotation coefficients
        # Width affects how much we rotate
        rotation = width * 0.1  # Small rotation angle
        
        # Apply all-pass-like phase shift
        left_processed = self._allpass_filter(left, rotation)
        right_processed = self._allpass_filter(right, -rotation)
        
        return np.stack([left_processed, right_processed], axis=1)
    
    def _allpass_filter(self, audio: np.ndarray, coefficient: float) -> np.ndarray:
        """Simple all-pass filter for phase rotation."""
        output = np.zeros_like(audio)
        
        for i in range(1, len(audio)):
            output[i] = -coefficient * output[i-1] + audio[i] - coefficient * audio[i-1]
        
        return output
    
    def auto_widen(self, audio: np.ndarray, target_width: float = 1.5) -> np.ndarray:
        """
        Automatically widen stereo image based on analysis.
        
        Parameters:
            audio: Stereo audio input
            target_width: Target stereo width (1.0 = original, >1.0 = wider)
        
        Returns:
            Widened stereo audio
        """
        # Analyze current stereo width
        current_width = self.analyze_width(audio)
        
        # Calculate required boost
        if current_width > 0:
            width_ratio = target_width / current_width
        else:
            width_ratio = target_width
        
        # Apply mid/side widening (most natural)
        return self.process(audio, width=width_ratio, technique="mid_side")
    
    def analyze_width(self, audio: np.ndarray) -> float:
        """
        Analyze current stereo width of audio.
        Returns width factor (0 = mono, 1 = normal stereo, >1 = wide).
        """
        if audio.shape[1] < 2:
            return 0.0
        
        left = audio[:, 0]
        right = audio[:, 1]
        
        # Calculate correlation (mono compatibility)
        correlation = np.corrcoef(left, right)[0, 1]
        
        # Convert correlation to width estimate
        # -1 = perfectly out of phase (very wide), 1 = identical (mono)
        if np.isnan(correlation):
            return 1.0
        
        # Map correlation to width (approximate)
        width = 1.0 - correlation * 0.8
        return max(0.0, min(2.0, width))
    
    def squash_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Collapse stereo to mono.
        """
        if audio.shape[1] < 2:
            return audio
        
        mono = (audio[:, 0] + audio[:, 1]) / 2.0
        return np.stack([mono, mono], axis=1)
    
    def enhance_stereo_field(self, audio: np.ndarray,
                              low_width: float = 1.0,
                              mid_width: float = 1.5,
                              high_width: float = 2.0) -> np.ndarray:
        """
        Frequency-dependent stereo widening.
        Different width for low, mid, and high frequencies.
        """
        # Convert to mid/side
        left = audio[:, 0]
        right = audio[:, 1]
        
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        
        # Create frequency bands using crossover
        # Low: < 200 Hz
        # Mid: 200 Hz - 2000 Hz
        # High: > 2000 Hz
        
        # Low band
        b_low, a_low = butter(4, 200 / (self.sample_rate / 2), 'low')
        mid_low = lfilter(b_low, a_low, mid)
        side_low = lfilter(b_low, a_low, side)
        side_low *= low_width
        
        # Mid band  
        b_mid1, a_mid1 = butter(4, 200 / (self.sample_rate / 2), 'high')
        b_mid2, a_mid2 = butter(4, 2000 / (self.sample_rate / 2), 'low')
        mid_mid = lfilter(b_mid2, a_mid2, lfilter(b_mid1, a_mid1, mid))
        side_mid = lfilter(b_mid2, a_mid2, lfilter(b_mid1, a_mid1, side))
        side_mid *= mid_width
        
        # High band
        b_high, a_high = butter(4, 2000 / (self.sample_rate / 2), 'high')
        mid_high = lfilter(b_high, a_high, mid)
        side_high = lfilter(b_high, a_high, side)
        side_high *= high_width
        
        # Combine bands
        side_combined = side_low + side_mid + side_high
        
        # Convert back to stereo
        left_out = mid + side_combined
        right_out = mid - side_combined
        
        # Normalize
        max_val = max(np.abs(left_out).max(), np.abs(right_out).max())
        if max_val > 1.0:
            left_out /= max_val
            right_out /= max_val
        
        return np.stack([left_out, right_out], axis=1)
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            return audio / max_val
        return audio


def widen_stereo(audio: np.ndarray, width: float = 1.5,
                 sample_rate: int = 44100) -> np.ndarray:
    """
    Convenience function for stereo widening.
    
    Parameters:
        audio: Stereo audio (N, 2) or (2, N)
        width: Width factor
        sample_rate: Audio sample rate
    
    Returns:
        Widened stereo audio
    """
    widener = StereoWidener(sample_rate)
    return widener.process(audio, width=width, technique="mid_side")
