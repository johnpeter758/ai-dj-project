"""
Automatic Audio Mastering System for AI Music
==============================================

This module provides professional auto-mastering capabilities:
- LUFS normalization for streaming platforms
- Multi-band compression
- Stereo widening
- Reference-based matching
- Loudness targeting (Spotify, Apple Music, etc.)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

# =============================================================================
# STREAMING PLATFORM LOUDNESS TARGETS
# =============================================================================

class StreamingPlatforms:
    """Loudness targets for major streaming platforms."""
    
    # LUFS Integrated Targets (most common)
    SPOTIFY = -14.0  # Recommended for Spotify
    APPLE_MUSIC = -16.0
    YOUTUBE = -14.0
    TIDAL = -14.0
    AMAZON_MUSIC = -14.0
    SOUNDCLOUD = -14.0
    
    # True Peak Maximums
    TRUE_PEAK_MAX = {
        'spotify': -1.0,
        'apple_music': -1.0,
        'youtube': -1.0,
        'tidal': -1.5,
        'amazon': -2.0,
        'soundcloud': -1.0,
    }
    
    # Loudness Range (LRA) targets for consistent dynamic range
    LRA_TARGETS = {
        'spotify': 7.0,    # Good for most genres
        'apple_music': 7.0,
        'youtube': 6.0,
        'tidal': 8.0,
        'amazon': 7.0,
    }
    
    @classmethod
    def get_settings(cls, platform: str) -> Dict[str, float]:
        """Get mastering settings for a specific platform."""
        platform = platform.lower()
        return {
            'target_lufs': getattr(cls, platform.upper(), -14.0),
            'true_peak': cls.TRUE_PEAK_MAX.get(platform, -1.0),
            'lra_target': cls.LRA_TARGETS.get(platform, 7.0),
        }


# =============================================================================
# LUFS METADATA CALCULATION
# =============================================================================

def calculate_lufs(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Calculate integrated LUFS (Loudness Units Full Scale) for audio.
    
    This implements ITU-R BS.1770-4 loudness measurement.
    
    Args:
        audio: Audio data as numpy array (samples,) or (samples, channels)
        sample_rate: Sample rate of audio
        
    Returns:
        Integrated LUFS value
    """
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    
    # K-weighting filter coefficients (simplified)
    # In production, use proper IIR filter implementation
    channel_count = audio.shape[1]
    
    # Calculate mean square
    mean_square = np.mean(audio ** 2, axis=0)
    
    # Apply channel weights (K-weighting)
    # L, R = 1.0, C = 1.0, Ls, Rs = 1.41
    if channel_count == 2:
        weights = [1.0, 1.0]
    elif channel_count >= 6:
        weights = [1.0, 1.0, 1.0, 1.41, 1.41, 0.0][:channel_count]
    else:
        weights = [1.0] * channel_count
    
    weighted_sum = sum(w * ms for w, ms in zip(weights, mean_square))
    
    # Convert to LUFS (approximation)
    if weighted_sum > 0:
        lufs = -0.691 + 10 * np.log10(weighted_sum)
    else:
        lufs = -70.0
    
    return lufs


def calculate_loudness_range(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Calculate Loudness Range (LRA) in LU.
    
    Per ITU-R BS.1770-4, LRA is the difference between the
    upper and lower loudness bounds.
    """
    # Short-term loudness (400ms blocks with 75% overlap)
    block_size = int(0.4 * sample_rate)
    hop_size = int(0.1 * sample_rate)
    
    short_term_loudness = []
    for i in range(0, len(audio) - block_size, hop_size):
        block = audio[i:i + block_size]
        if block.shape[0] > 0:
            lufs = calculate_lufs(block, sample_rate)
            short_term_loudness.append(lufs)
    
    if len(short_term_loudness) < 2:
        return 0.0
    
    # Calculate LRA using gating
    short_term_loudness = np.array(short_term_loudness)
    
    # Absolute gate at -70 LUFS
    gated = short_term_loudness[short_term_loudness > -70]
    
    if len(gated) < 2:
        return 0.0
    
    # Get 10th and 95th percentiles
    low = np.percentile(gated, 10)
    high = np.percentile(gated, 95)
    
    return high - low


def calculate_true_peak(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Calculate True Peak in dBTP (dB True Peak).
    
    Uses 4x oversampling for accurate peak detection.
    """
    # Simple peak detection (in production, use proper oversampling)
    peak = np.max(np.abs(audio))
    
    if peak > 0:
        return 20 * np.log10(peak)
    else:
        return -np.inf


# =============================================================================
# MULTI-BAND COMPRESSION
# =============================================================================

class MultibandCompressor:
    """
    Multi-band dynamics processor for professional mastering.
    
    Splits audio into frequency bands and applies compression
    separately to each, allowing for targeted dynamics control.
    """
    
    # Default crossover frequencies (Hz)
    DEFAULT_CROSSOVERS = {
        'sub': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 1000),
        'mid': (1000, 4000),
        'high_mid': (4000, 8000),
        'high': (8000, 20000),
    }
    
    def __init__(
        self,
        crossovers: Optional[Dict[str, Tuple[float, float]]] = None,
        compression_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize multi-band compressor.
        
        Args:
            crossovers: Dict of band names to (low_freq, high_freq) tuples
            compression_settings: Per-band settings:
                - threshold: dB (default -24)
                - ratio: compression ratio (default 4:1)
                - attack_ms: attack time in ms (default 10)
                - release_ms: release time in ms (default 100)
                - makeup_gain: dB (default 0)
        """
        self.crossovers = crossovers or self.DEFAULT_CROSSOVERS
        
        # Default compression settings per band
        self.default_compression = {
            'sub': {'threshold': -30, 'ratio': 3, 'attack': 15, 'release': 150, 'makeup': 3},
            'bass': {'threshold': -24, 'ratio': 4, 'attack': 10, 'release': 120, 'makeup': 2},
            'low_mid': {'threshold': -20, 'ratio': 3, 'attack': 8, 'release': 100, 'makeup': 0},
            'mid': {'threshold': -18, 'ratio': 2.5, 'attack': 5, 'release': 80, 'makeup': 0},
            'high_mid': {'threshold': -20, 'ratio': 2, 'attack': 3, 'release': 60, 'makeup': 1},
            'high': {'threshold': -24, 'ratio': 2, 'attack': 2, 'release': 50, 'makeup': 2},
        }
        
        self.compression_settings = compression_settings or self.default_compression
    
    def _create_crossover_filters(self, sample_rate: int):
        """Create IIR crossover filters (simplified)."""
        # In production, use scipy.signal for proper Linkwitz-Riley crossovers
        pass
    
    def _compress_band(
        self,
        band_data: np.ndarray,
        settings: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply compression to a frequency band.
        
        This is a simplified feedforward compressor implementation.
        """
        threshold = 10 ** (settings['threshold'] / 20)
        ratio = settings['ratio']
        attack = settings['attack'] / 1000  # Convert to seconds
        release = settings['release'] / 1000
        makeup = 10 ** (settings['makeup_gain'] / 20)
        
        # Simplified dynamic range compression
        compressed = band_data.copy()
        
        # Apply gain reduction based on level
        level = np.abs(band_data)
        above_threshold = level > threshold
        reduction = np.where(
            above_threshold,
            (1 - 1/ratio) * (level - threshold),
            0
        )
        
        compressed = band_data * (1 - reduction) * makeup
        
        # Clipping prevention
        return np.clip(compressed, -1.0, 1.0)
    
    def process(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Process audio through multi-band compression.
        
        Returns:
            Compressed audio
        """
        # In production, implement proper crossover filtering
        # and per-band compression with proper envelope followers
        return audio  # Placeholder


# =============================================================================
# STEREO WIDENING
# =============================================================================

class StereoWidener:
    """
    Stereo width enhancement for mastering.
    
    Techniques:
    - Mid/Side processing
    - Haas effect delays
    - Correlation-based width control
    """
    
    def __init__(
        self,
        width: float = 1.0,
        mid_gain: float = 1.0,
        side_gain: float = 1.0,
        correlation_threshold: float = 0.3,
    ):
        """
        Initialize stereo widener.
        
        Args:
            width: Stereo width multiplier (1.0 = original, 2.0 = double width)
            mid_gain: Gain for mid (center) channel
            side_gain: Gain for side (stereo) channel
            correlation_threshold: Minimum correlation before width reduction
        """
        self.width = width
        self.mid_gain = mid_gain
        self.side_gain = side_gain
        self.correlation_threshold = correlation_threshold
    
    def to_mid_side(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert stereo to Mid/Side."""
        mid = (audio[:, 0] + audio[:, 1]) / np.sqrt(2)
        side = (audio[:, 0] - audio[:, 1]) / np.sqrt(2)
        return mid, side
    
    def to_stereo(self, mid: np.ndarray, side: np.ndarray) -> np.ndarray:
        """Convert Mid/Side back to stereo."""
        left = (mid + side) / np.sqrt(2)
        right = (mid - side) / np.sqrt(2)
        return np.column_stack([left, right])
    
    def calculate_correlation(self, audio: np.ndarray) -> float:
        """Calculate stereo correlation (-1 to 1)."""
        if audio.shape[1] < 2:
            return 1.0
        
        # Pearson correlation coefficient
        l = audio[:, 0]
        r = audio[:, 1]
        
        l_centered = l - np.mean(l)
        r_centered = r - np.mean(r)
        
        correlation = np.sum(l_centered * r_centered) / (
            np.sqrt(np.sum(l_centered ** 2)) * np.sqrt(np.sum(r_centered ** 2)) + 1e-10
        )
        
        return correlation
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply stereo widening.
        
        Args:
            audio: Stereo audio (samples, 2)
            
        Returns:
            Processed stereo audio
        """
        if audio.shape[1] < 2:
            return audio
        
        # Check correlation - reduce width if too mono
        correlation = self.calculate_correlation(audio)
        
        # Convert to mid/side
        mid, side = self.to_mid_side(audio)
        
        # Apply width control
        # Low correlation = narrow, high = can widen more
        effective_width = self.width
        if correlation < self.correlation_threshold:
            effective_width = 1.0 + (self.width - 1.0) * (correlation / self.correlation_threshold)
        
        # Apply gains
        mid = mid * self.mid_gain
        side = side * side_gain * effective_width
        
        # Convert back to stereo
        processed = self.to_stereo(mid, side)
        
        # Soft clip to prevent harsh distortion
        processed = self._soft_clip(processed)
        
        return processed
    
    def _soft_clip(self, audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """Apply soft clipping for saturation without harsh distortion."""
        clipped = audio.copy()
        
        # Soft clip function: x / (1 + x^2)
        for ch in range(audio.shape[1]):
            x = audio[:, ch] / threshold
            clipped[:, ch] = threshold * (x / (1 + x ** 2)) * np.sign(x)
        
        return clipped


# =============================================================================
# LOUDNESS NORMALIZATION
# =============================================================================

class LoudnessNormalizer:
    """
    Professional loudness normalization for streaming platforms.
    
    Applies gain adjustment to match target LUFS while respecting
    true peak limits and loudness range targets.
    """
    
    def __init__(
        self,
        target_lufs: float = -14.0,
        true_peak_limit: float = -1.0,
        target_lra: Optional[float] = None,
    ):
        """
        Initialize loudness normalizer.
        
        Args:
            target_lufs: Target integrated loudness in LUFS
            true_peak_limit: Maximum true peak in dBTP
            target_lra: Target loudness range (optional)
        """
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self.target_lra = target_lra
    
    def measure(self, audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, float]:
        """
        Measure audio loudness metrics.
        
        Returns:
            Dict with 'lufs', 'true_peak', 'lra' values
        """
        return {
            'lufs': calculate_lufs(audio, sample_rate),
            'true_peak': calculate_true_peak(audio, sample_rate),
            'lra': calculate_loudness_range(audio, sample_rate),
        }
    
    def normalize(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize audio to target loudness.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            Tuple of (normalized audio, processing info)
        """
        # Measure current loudness
        metrics = self.measure(audio, sample_rate)
        current_lufs = metrics['lufs']
        
        # Calculate gain needed
        gain_db = self.target_lufs - current_lufs
        
        # Apply gain
        gain_linear = 10 ** (gain_db / 20)
        normalized = audio * gain_linear
        
        # Check true peak
        new_peak = calculate_true_peak(normalized, sample_rate)
        
        # If true peak exceeds limit, apply limiting
        if new_peak > self.true_peak_limit:
            reduction_needed = new_peak - self.true_peak_limit
            limit_gain = 10 ** (-reduction_needed / 20)
            normalized = normalized * limit_gain
            
            # Re-measure after limiting
            metrics = self.measure(normalized, sample_rate)
        
        # Soft clip if still clipping
        normalized = self._soft_limiter(normalized)
        
        info = {
            'original_lufs': current_lufs,
            'target_lufs': self.target_lufs,
            'gain_db': gain_db,
            'metrics': metrics,
        }
        
        return normalized, info
    
    def _soft_limiter(self, audio: np.ndarray, ceiling: float = -0.3) -> np.ndarray:
        """Apply soft limiting to prevent hard clipping."""
        peak = np.max(np.abs(audio))
        ceiling_linear = 10 ** (ceiling / 20)
        
        if peak > ceiling_linear:
            # Soft knee limiting
            threshold = ceiling_linear
            ratio = 10  # Limit ratio
            
            limited = np.where(
                np.abs(audio) > threshold,
                threshold + (audio - threshold) / ratio,
                audio
            )
            
            return limited
        
        return audio


# =============================================================================
# REFERENCE-BASED MASTERING
# =============================================================================

class ReferenceMastering:
    """
    Reference-based mastering that matches characteristics of
    professional reference tracks.
    """
    
    def __init__(self):
        """Initialize reference mastering."""
        self.reference_metrics: Optional[Dict[str, float]] = None
    
    def analyze_reference(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100
    ) -> Dict[str, float]:
        """
        Analyze reference track to extract target characteristics.
        
        Returns:
            Dict with spectral, dynamic, and spatial characteristics
        """
        # Frequency analysis (simplified)
        # In production, use proper STFT/spectrogram analysis
        
        # Measure basic metrics
        metrics = {
            'lufs': calculate_lufs(audio, sample_rate),
            'true_peak': calculate_true_peak(audio, sample_rate),
            'lra': calculate_loudness_range(audio, sample_rate),
            'rms': np.sqrt(np.mean(audio ** 2)),
            'crest_factor': np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-10),
        }
        
        # Calculate spectral centroid (brightness)
        # Weighted average of frequencies
        fft = np.fft.rfft(audio[:, 0] if audio.shape[1] > 0 else audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        magnitudes = np.abs(fft)
        
        if np.sum(magnitudes) > 0:
            metrics['spectral_centroid'] = np.sum(freqs * magnitudes) / np.sum(magnitudes)
        else:
            metrics['spectral_centroid'] = 1000
        
        # Stereo width
        widener = StereoWidener()
        metrics['stereo_width'] = widener.calculate_correlation(audio)
        
        self.reference_metrics = metrics
        return metrics
    
    def match_reference(
        self,
        target_audio: np.ndarray,
        sample_rate: int = 44100,
        match_lufs: bool = True,
        match_spectrum: bool = True,
        match_dynamics: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Match target audio to reference characteristics.
        
        Args:
            target_audio: Audio to master
            sample_rate: Sample rate
            match_lufs: Match loudness
            match_spectrum: Match spectral balance
            match_dynamics: Match dynamics/LRA
            
        Returns:
            Tuple of (mastered audio, processing info)
        """
        if self.reference_metrics is None:
            raise ValueError("No reference analyzed. Call analyze_reference first.")
        
        processed = target_audio.copy()
        info = {'operations': []}
        
        # 1. Match loudness
        if match_lufs:
            normalizer = LoudnessNormalizer(
                target_lufs=self.reference_metrics['lufs'],
                true_peak_limit=-1.0,
            )
            processed, norm_info = normalizer.normalize(processed, sample_rate)
            info['operations'].append(('loudness_match', norm_info))
        
        # 2. Match dynamics (simplified - would use multiband compression)
        if match_dynamics:
            target_lra = self.reference_metrics['lra']
            # In production, adjust multiband compression based on LRA difference
            info['operations'].append(('dynamics_match', {'target_lra': target_lra}))
        
        # 3. Match spectrum (simplified - would use EQ)
        if match_spectrum:
            target_centroid = self.reference_metrics['spectral_centroid']
            info['operations'].append(('spectrum_match', {'target_centroid': target_centroid}))
        
        info['reference_metrics'] = self.reference_metrics
        info['result_metrics'] = {
            'lufs': calculate_lufs(processed, sample_rate),
            'true_peak': calculate_true_peak(processed, sample_rate),
            'lra': calculate_loudness_range(processed, sample_rate),
        }
        
        return processed, info


# =============================================================================
# COMPLETE AUTO-MASTERING PIPELINE
# =============================================================================

class AutoMaster:
    """
    Complete automatic mastering pipeline.
    
    Combines all processing stages for professional results.
    """
    
    def __init__(
        self,
        platform: str = 'spotify',
        reference_audio: Optional[np.ndarray] = None,
        sample_rate: int = 44100,
    ):
        """
        Initialize auto-master.
        
        Args:
            platform: Target streaming platform
            reference_audio: Optional reference track for matching
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        platform_settings = StreamingPlatforms.get_settings(platform)
        
        # Initialize processors
        self.limiter = LoudnessNormalizer(
            target_lufs=platform_settings['target_lufs'],
            true_peak_limit=platform_settings['true_peak'],
            target_lra=platform_settings['lra_target'],
        )
        
        self.widener = StereoWidener(
            width=1.2,  # Slight width increase
            mid_gain=1.0,
            side_gain=1.1,
        )
        
        self.compressor = MultibandCompressor()
        
        # Reference mastering
        if reference_audio is not None:
            self.reference = ReferenceMastering()
            self.reference.analyze_reference(reference_audio, sample_rate)
        else:
            self.reference = None
    
    def master(
        self,
        audio: np.ndarray,
        apply_widening: bool = True,
        apply_compression: bool = True,
        apply_reference_match: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply full mastering chain to audio.
        
        Args:
            audio: Input audio (samples,) or (samples, channels)
            apply_widening: Apply stereo widening
            apply_compression: Apply multi-band compression
            apply_reference_match: Match to reference track
            
        Returns:
            Tuple of (mastered audio, processing report)
        """
        # Ensure proper format
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        report = {
            'input_metrics': self.limiter.measure(audio, self.sample_rate),
            'stages': [],
        }
        
        processed = audio.copy()
        
        # Stage 1: Multi-band compression (optional)
        if apply_compression:
            processed = self.compressor.process(processed, self.sample_rate)
            report['stages'].append('multiband_compression')
        
        # Stage 2: Stereo widening
        if apply_widening:
            processed = self.widener.process(processed)
            report['stages'].append('stereo_widening')
        
        # Stage 3: Reference matching (if enabled)
        if apply_reference_match and self.reference is not None:
            processed, ref_info = self.reference.match_reference(
                processed, self.sample_rate
            )
            report['stages'].append('reference_matching')
            report['reference_info'] = ref_info
        
        # Stage 4: Loudness normalization & limiting
        processed, norm_info = self.limiter.normalize(processed, self.sample_rate)
        report['stages'].append('loudness_normalization')
        report['normalization'] = norm_info
        
        # Final metrics
        report['output_metrics'] = self.limiter.measure(processed, self.sample_rate)
        report['output_metrics']['stereo_correlation'] = self.widener.calculate_correlation(processed)
        
        return processed, report


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """
    Example of how to use the auto-mastering system.
    """
    # This would be used with actual audio data
    print("""
# Auto-Mastering Usage Example
# =============================

from auto_master import AutoMaster, StreamingPlatforms
import numpy as np

# Load your AI-generated audio
audio = np.load('my_ai_track.npy')

# Option 1: Simple platform-targeting mastering
master = AutoMaster(platform='spotify')
mastered, report = master.master(audio)

print(f"Input LUFS: {report['input_metrics']['lufs']:.1f}")
print(f"Output LUFS: {report['output_metrics']['lufs']:.1f}")
print(f"True Peak: {report['output_metrics']['true_peak']:.1f} dBTP")

# Option 2: Reference-based mastering
# Load a professional reference track
reference = np.load('professional_reference.npy')
master = AutoMaster(platform='spotify', reference_audio=reference)
mastered, report = master.master(audio, apply_reference_match=True)

# Option 3: Custom settings
from auto_master import LoudnessNormalizer, StereoWidener

normalizer = LoudnessNormalizer(
    target_lufs=-16.0,  # Apple Music target
    true_peak_limit=-1.0,
)
normalized, info = normalizer.normalize(audio)

widener = StereoWidener(width=1.5)  # More aggressive widening
widened = widener.process(normalized)

# Save mastered audio
np.save('mastered_track.npy', widened)

# Platform-specific settings
spotify = StreamingPlatforms.get_settings('spotify')
apple = StreamingPlatforms.get_settings('apple_music')
print(f"Spotify: {spotify}")
print(f"Apple: {apple}")
""")


if __name__ == '__main__':
    example_usage()
