"""
Clipper/Limiter - Audio Clipping and Limiting for AI DJ Project

Provides hard and soft clipping, lookahead limiting, and multiband limiting
for dynamic range control and loudness maximization.

Features:
- Hard clipping with configurable threshold
- Soft clipping with exponential/sigmoid curves
- Lookahead limiter with attack/release
- Multiband limiter (3-band)
- Brickwall limiting for mastering
"""

import numpy as np
from typing import Optional, Tuple
from scipy import signal
from scipy.signal import butter


class Clipper:
    """Hard and soft audio clipper."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def hard_clip(self, audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """
        Hard clip audio signal at threshold.
        
        Args:
            audio: Input audio signal (numpy array)
            threshold: Clip threshold in range [-1, 1]
        
        Returns:
            Hard-clipped audio signal
        """
        return np.clip(audio, -threshold, threshold)
    
    def soft_clip_tanh(self, audio: np.ndarray, threshold: float = 0.8,
                        makeup_gain: float = 1.0) -> np.ndarray:
        """
        Soft clip using hyperbolic tangent (tanh) curve.
        Produces smooth, musical clipping with odd harmonics.
        
        Args:
            audio: Input audio signal
            threshold: Soft clip threshold
            makeup_gain: Gain to apply after clipping
        
        Returns:
            Soft-clipped audio signal
        """
        # Scale input to create soft transition around threshold
        k = 10  # Steepness factor
        scaled = audio / threshold
        clipped = np.tanh(k * scaled) * threshold
        return clipped * makeup_gain
    
    def soft_clip_exponential(self, audio: np.ndarray, threshold: float = 0.8,
                               knee_width: float = 0.1) -> np.ndarray:
        """
        Soft clip with exponential curve and configurable knee.
        
        Args:
            audio: Input audio signal
            threshold: Clip threshold
            knee_width: Width of soft knee region
        
        Returns:
            Soft-clipped audio signal
        """
        output = np.zeros_like(audio)
        abs_audio = np.abs(audio)
        
        # Linear region (below knee)
        linear_mask = abs_audio <= (threshold - knee_width)
        output[linear_mask] = audio[linear_mask]
        
        # Knee region
        knee_mask = (abs_audio > (threshold - knee_width)) & (abs_audio <= threshold)
        x = abs_audio[knee_mask]
        knee_center = threshold - knee_width
        # Smooth transition using cosine interpolation
        t = (x - knee_center) / knee_width * np.pi / 2
        knee_factor = np.sin(t)
        output[knee_mask] = audio[knee_mask] * knee_factor + \
                           audio[knee_mask] * (1 - knee_factor) * (x / knee_center)
        
        # Soft clipping region (above threshold)
        clip_mask = abs_audio > threshold
        # Exponential soft clip
        sign = np.sign(audio[clip_mask])
        excess = abs_audio[clip_mask] - threshold
        output[clip_mask] = sign * (threshold + (1 - np.exp(-excess * 2)) * 0.1)
        
        return output
    
    def asymmetric_clip(self, audio: np.ndarray, 
                        pos_threshold: float = 0.9,
                        neg_threshold: float = -0.95) -> np.ndarray:
        """
        Asymmetric clipping - different limits for positive/negative peaks.
        Useful for mastering where negative peaks can be pushed harder.
        
        Args:
            audio: Input audio signal
            pos_threshold: Maximum positive amplitude
            neg_threshold: Maximum negative amplitude (should be negative)
        
        Returns:
            Asymmetrically clipped audio
        """
        return np.clip(audio, neg_threshold, pos_threshold)


class Limiter:
    """Professional lookahead limiter for dynamic range control."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._envelope = 0.0
        self._gain_reduction = 0.0
        self._delay_buffer = None
        self._lookahead_samples = 0
    
    def limit(self, audio: np.ndarray, threshold: float = -0.1,
              ratio: float = 10.0, attack: float = 0.001,
              release: float = 0.1, makeup_gain: float = 1.0,
              lookahead: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply lookahead limiting to audio signal.
        
        Args:
            audio: Input audio signal (numpy array, float32)
            threshold: Limiting threshold in linear scale (e.g., -0.1 = -20dB)
            ratio: Limiting ratio (higher = harder limiting)
            attack: Attack time in seconds
            release: Release time in seconds
            makeup_gain: Gain to apply after limiting
            lookahead: Lookahead time in seconds
        
        Returns:
            Tuple of (limited audio, gain reduction in dB)
        """
        # Convert threshold from dB to linear if needed
        if threshold > 0:
            threshold = 10 ** (threshold / 20)
        
        # Calculate lookahead samples
        self._lookahead_samples = int(lookahead * self.sample_rate)
        
        # Initialize delay buffer for lookahead
        if self._delay_buffer is None or len(self._delay_buffer) != self._lookahead_samples:
            self._delay_buffer = np.zeros(self._lookahead_samples)
        
        # Calculate attack and release coefficients
        alpha_attack = np.exp(-1 / (self.sample_rate * attack))
        alpha_release = np.exp(-1 / (self.sample_rate * release))
        
        # Envelope follower and gain computer
        output = np.zeros_like(audio)
        gain_reduction = np.zeros_like(audio)
        envelope = self._envelope
        
        for i in range(len(audio)):
            # Get input (with lookahead delay applied via buffer)
            if i < self._lookahead_samples:
                # Use buffered sample from end of previous block
                input_level = self._delay_buffer[i] if i < len(self._delay_buffer) else 0
            else:
                input_level = audio[i - self._lookahead_samples]
            
            # Envelope follower
            input_abs = abs(input_level)
            if input_abs > envelope:
                envelope = alpha_attack * envelope + (1 - alpha_attack) * input_abs
            else:
                envelope = alpha_release * envelope + (1 - alpha_release) * input_abs
            
            # Gain computer (soft knee)
            if envelope > threshold and threshold > 0:
                # Above threshold - apply limiting
                db_over = 20 * np.log10(envelope / threshold)
                gain_reduction_db = (db_over * (1 - 1/ratio))
                gain = 10 ** (-gain_reduction_db / 20)
            else:
                # Below threshold - no gain reduction
                gain = 1.0
            
            # Apply gain
            output[i] = audio[i] * gain
            gain_reduction[i] = 20 * np.log10(gain) if gain > 0 else 0
        
        # Store envelope state for next block
        self._envelope = envelope
        
        # Apply makeup gain
        output *= makeup_gain
        
        return output, gain_reduction
    
    def brickwall_limit(self, audio: np.ndarray, threshold: float = -0.5,
                        lookahead: float = 0.005) -> np.ndarray:
        """
        Brickwall limiter - hard limiting with no permitted overs.
        Uses look-ahead to ensure no samples exceed threshold.
        
        Args:
            audio: Input audio signal
            threshold: Maximum output level (linear)
            lookahead: Lookahead time in seconds
        
        Returns:
            Brickwall-limited audio
        """
        lookahead_samples = int(lookahead * self.sample_rate)
        
        # Calculate maximum lookahead amplitude for each sample
        max_lookahead = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Look ahead to find maximum amplitude
            end_idx = min(i + lookahead_samples + 1, len(audio))
            max_lookahead[i] = np.max(np.abs(audio[i:end_idx]))
        
        # Calculate required gain reduction
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = np.where(max_lookahead > threshold,
                           threshold / max_lookahead,
                           1.0)
        
        # Apply gain with smooth attack
        gain = np.minimum(gain, 1.0)
        
        # Smooth the gain changes
        if len(gain) > 1:
            gain = signal.filtfilt([1], [1, 0.99], gain)
            gain = np.clip(gain, 0, 1)
        
        return audio * gain


class MultibandLimiter:
    """3-band multiband limiter for frequency-dependent limiting."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._crossovers = [250, 4000]  # Default crossover frequencies
        
        # Create crossover filters
        self._low_filter = self._create_lowpass(250)
        self._mid_filter = self._create_bandpass(250, 4000)
        self._high_filter = self._create_highpass(4000)
        
        # Create high-pass for mid (to combine bands)
        self._mid_highpass = self._create_highpass(250)
        
        # Limiters for each band
        self._low_limiter = Limiter(sample_rate)
        self._mid_limiter = Limiter(sample_rate)
        self._high_limiter = Limiter(sample_rate)
    
    def _create_lowpass(self, freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create lowpass filter coefficients."""
        nyq = self.sample_rate / 2
        if freq >= nyq:
            freq = nyq * 0.99
        b, a = butter(4, freq / nyq, btype='low')
        return b, a
    
    def _create_highpass(self, freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create highpass filter coefficients."""
        nyq = self.sample_rate / 2
        if freq <= 0:
            freq = 20
        b, a = butter(4, freq / nyq, btype='high')
        return b, a
    
    def _create_bandpass(self, low: float, high: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create bandpass filter coefficients."""
        nyq = self.sample_rate / 2
        if high >= nyq:
            high = nyq * 0.99
        b, a = butter(4, [low / nyq, high / nyq], btype='band')
        return b, a
    
    def limit(self, audio: np.ndarray,
              low_threshold: float = -3.0, mid_threshold: float = -1.0,
              high_threshold: float = -2.0,
              low_ratio: float = 8.0, mid_ratio: float = 6.0, high_ratio: float = 4.0,
              makeup_gain: float = 1.2) -> np.ndarray:
        """
        Apply multiband limiting.
        
        Args:
            audio: Input audio signal
            low_threshold: Threshold for low frequencies (dB)
            mid_threshold: Threshold for mid frequencies (dB)
            high_threshold: Threshold for high frequencies (dB)
            low_ratio: Ratio for low frequencies
            mid_ratio: Ratio for mid frequencies
            high_ratio: Ratio for high frequencies
            makeup_gain: Master makeup gain
        
        Returns:
            Multiband-limited audio
        """
        # Split into bands
        low_band = signal.filtfilt(self._low_filter[0], self._low_filter[1], audio)
        high_band = signal.filtfilt(self._high_filter[0], self._high_filter[1], audio)
        
        # Mid is what's left after removing low and high
        mid_band = audio - low_band - high_band
        
        # Apply limiting to each band
        low_limited, _ = self._low_limiter.limit(
            low_band, threshold=low_threshold, ratio=low_ratio,
            attack=0.005, release=0.2
        )
        mid_limited, _ = self._mid_limiter.limit(
            mid_band, threshold=mid_threshold, ratio=mid_ratio,
            attack=0.002, release=0.1
        )
        high_limited, _ = self._high_limiter.limit(
            high_band, threshold=high_threshold, ratio=high_ratio,
            attack=0.001, release=0.05
        )
        
        # Combine bands with makeup gain
        output = (low_limited + mid_limited + high_limited) * makeup_gain
        
        return output
    
    def set_crossovers(self, low_mid: float, mid_high: float):
        """Update crossover frequencies."""
        self._crossovers = [low_mid, mid_high]
        
        # Recreate filters
        self._low_filter = self._create_lowpass(low_mid)
        self._mid_filter = self._create_bandpass(low_mid, mid_high)
        self._high_filter = self._create_highpass(mid_high)


class LoudnessMaximizer:
    """Complete loudness maximization chain for mastering."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.clipper = Clipper(sample_rate)
        self.limiter = Limiter(sample_rate)
        self.multiband = MultibandLimiter(sample_rate)
        
        # Default settings
        self.target_lufs = -14.0  # Common streaming platform target
        self.true_peak_max = -1.0  # dBTP
    
    def maximize(self, audio: np.ndarray,
                 use_multiband: bool = True,
                 soft_clip: bool = True,
                 limiter_threshold: float = -0.5,
                 limiter_ratio: float = 10.0) -> np.ndarray:
        """
        Full loudness maximization chain.
        
        Args:
            audio: Input audio signal
            use_multiband: Whether to use multiband limiting
            soft_clip: Whether to apply soft clipping before limiting
            limiter_threshold: Limiter threshold in dB
            limiter_ratio: Limiter ratio
        
        Returns:
            Maximized audio signal
        """
        output = audio.copy()
        
        # 1. Soft clipper (adds saturation, increases loudness)
        if soft_clip:
            output = self.clipper.soft_clip_tanh(output, threshold=0.8, makeup_gain=1.1)
        
        # 2. Multiband limiting (optional, more musical)
        if use_multiband:
            output = self.multiband.limit(
                output,
                low_threshold=-3.0,
                mid_threshold=-1.0,
                high_threshold=-2.0,
                makeup_gain=1.15
            )
        
        # 3. Brickwall limiter (ensures no overs)
        output, _ = self.limiter.limit(
            output,
            threshold=limiter_threshold,
            ratio=limiter_ratio,
            attack=0.001,
            release=0.05,
            lookahead=0.003
        )
        
        # 4. Final brickwall clip
        output = self.clipper.hard_clip(output, threshold=0.99)
        
        # 5. Normalize to target peak
        peak = np.max(np.abs(output))
        if peak > 0:
            target_peak = 10 ** (self.true_peak_max / 20)
            output = output * (target_peak / peak)
        
        return output


# Convenience functions

def clip_audio(audio: np.ndarray, threshold: float = 0.9,
               soft: bool = True) -> np.ndarray:
    """
    Simple audio clipping function.
    
    Args:
        audio: Input audio
        threshold: Clip threshold (0-1)
        soft: Use soft clipping
    
    Returns:
        Clipped audio
    """
    clipper = Clipper()
    if soft:
        return clipper.soft_clip_tanh(audio, threshold=threshold)
    return clipper.hard_clip(audio, threshold=threshold)


def limit_audio(audio: np.ndarray, threshold: float = -0.5,
                ratio: float = 10.0, lookahead: float = 0.005) -> np.ndarray:
    """
    Simple audio limiting function.
    
    Args:
        audio: Input audio
        threshold: Limiting threshold in dB
        ratio: Limiting ratio
        lookahead: Lookahead time in seconds
    
    Returns:
        Limited audio
    """
    limiter = Limiter()
    limited, _ = limiter.limit(audio, threshold=threshold, ratio=ratio,
                               lookahead=lookahead)
    return limited


if __name__ == "__main__":
    # Simple test
    import soundfile as sf
    
    # Generate test signal (sine wave with occasional peaks)
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Signal with gradual increase and peaks
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    # Add some peaks
    peak_times = [0.25, 0.5, 0.75, 1.0]
    for pt in peak_times:
        idx = int(pt * sr)
        audio[idx:idx+100] = 1.2  # Overshoot
    
    print(f"Input peak: {np.max(np.abs(audio)):.3f}")
    
    # Test clipper
    clipper = Clipper(sr)
    clipped = clipper.soft_clip_tanh(audio, threshold=0.8, makeup_gain=1.1)
    print(f"After soft clip peak: {np.max(np.abs(clipped)):.3f}")
    
    # Test limiter
    limiter = Limiter(sr)
    limited, gr = limiter.limit(audio, threshold=-3.0, ratio=10.0,
                                lookahead=0.005)
    print(f"After limit peak: {np.max(np.abs(limited)):.3f}")
    
    # Test maximizer
    maximizer = LoudnessMaximizer(sr)
    maximized = maximizer.maximize(audio)
    print(f"After maximize peak: {np.max(np.abs(maximized)):.3f}")
    
    print("Test complete!")
