"""
Frequency Analyzer
Analyzes frequency content of audio files using FFT and spectral analysis.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

# Audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


@dataclass
class FrequencyBand:
    """Represents a frequency band with its range and energy."""
    name: str
    low_freq: float
    high_freq: float
    energy: float
    dominant_freq: float


@dataclass
class SpectralFeatures:
    """Spectral features extracted from audio."""
    centroid: float      # Spectral centroid (brightness)
    flux: float         # Spectral flux (change between frames)
    rolloff: float      # Spectral rolloff (frequency below which 85% of energy)
    flatness: float     # Spectral flatness (tonal vs noise)
    entropy: float      # Spectral entropy


class FrequencyAnalyzer:
    """Analyzes frequency content of audio files."""
    
    # Standard frequency bands (in Hz)
    BANDS = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 500),
        'mid': (500, 2000),
        'upper_mid': (2000, 4000),
        'presence': (4000, 6000),
        'brilliance': (6000, 20000)
    }
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize frequency analyzer.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa not installed. Run: pip install librosa")
    
    def analyze(self, audio_path: str) -> Dict:
        """
        Perform full frequency analysis on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing:
                - bands: List of FrequencyBand objects
                - spectral: SpectralFeatures object
                - spectrum: Full frequency spectrum (magnitudes)
                - frequencies: Frequency bins
                - dominant_freq: Overall dominant frequency
                - fundamental_freq: Estimated fundamental frequency
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Compute STFT
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Get frequency bins
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Analyze frequency bands
        bands = self._analyze_bands(magnitude, frequencies)
        
        # Extract spectral features
        spectral = self._extract_spectral_features(magnitude, frequencies)
        
        # Get overall dominant frequency
        mean_magnitude = np.mean(magnitude, axis=1)
        dominant_idx = np.argmax(mean_magnitude)
        dominant_freq = frequencies[dominant_idx] if dominant_idx < len(frequencies) else 0
        
        # Estimate fundamental frequency
        fundamental_freq = self._estimate_fundamental(y, sr)
        
        return {
            'bands': bands,
            'spectral': spectral,
            'spectrum': mean_magnitude,
            'frequencies': frequencies,
            'dominant_freq': dominant_freq,
            'fundamental_freq': fundamental_freq
        }
    
    def _analyze_bands(self, magnitude: np.ndarray, frequencies: np.ndarray) -> List[FrequencyBand]:
        """Analyze energy in each frequency band."""
        bands = []
        
        for band_name, (low, high) in self.BANDS.items():
            # Find frequency indices
            low_idx = np.searchsorted(frequencies, low)
            high_idx = np.searchsorted(frequencies, high)
            
            # Get magnitudes in this band
            band_magnitude = magnitude[low_idx:high_idx]
            
            if len(band_magnitude) > 0:
                # Calculate total energy
                energy = np.sum(band_magnitude ** 2)
                
                # Find dominant frequency in band
                band_freqs = frequencies[low_idx:high_idx]
                if len(band_magnitude) > 0:
                    dom_idx = np.argmax(np.mean(band_magnitude, axis=1))
                    dominant_freq = band_freqs[dom_idx] if dom_idx < len(band_freqs) else 0
                else:
                    dominant_freq = 0
                
                bands.append(FrequencyBand(
                    name=band_name,
                    low_freq=low,
                    high_freq=high,
                    energy=float(energy),
                    dominant_freq=dominant_freq
                ))
        
        return bands
    
    def _extract_spectral_features(self, magnitude: np.ndarray, frequencies: np.ndarray) -> SpectralFeatures:
        """Extract spectral features from magnitude spectrogram."""
        # Spectral centroid
        centroid = float(librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sample_rate, n_fft=self.n_fft
        ).mean())
        
        # Spectral flux
        flux = float(librosa.feature.spectral_flux(S=magnitude).mean())
        
        # Spectral rolloff
        rolloff = float(librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sample_rate, n_fft=self.n_fft
        ).mean())
        
        # Spectral flatness
        flatness = float(librosa.feature.spectral_flatness(S=magnitude).mean())
        
        # Spectral entropy
        # Normalize to get probability distribution
        mag_sum = np.sum(magnitude, axis=0)
        mag_sum[mag_sum == 0] = 1  # Avoid division by zero
        p = magnitude / mag_sum
        entropy = -np.sum(p * np.log2(p + 1e-10), axis=0).mean()
        
        return SpectralFeatures(
            centroid=centroid,
            flux=flux,
            rolloff=rolloff,
            flatness=flatness,
            entropy=float(entropy)
        )
    
    def _estimate_fundamental(self, y: np.ndarray, sr: int) -> float:
        """Estimate fundamental frequency (pitch) using autocorrelation."""
        # Use autocorrelation-based pitch tracking
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get the pitch at each frame
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                return float(np.median(pitch_values))
        except Exception:
            pass
        
        return 0.0
    
    def get_band_ratios(self, audio_path: str) -> Dict[str, float]:
        """
        Get relative energy ratios between frequency bands.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of band name to ratio (0-1 scale)
        """
        result = self.analyze(audio_path)
        
        total_energy = sum(band.energy for band in result['bands'])
        
        if total_energy == 0:
            return {name: 0.0 for name in self.BANDS.keys()}
        
        return {
            band.name: band.energy / total_energy 
            for band in result['bands']
        }
    
    def get_eq_recommendations(self, audio_path: str) -> Dict:
        """
        Get EQ recommendations based on frequency analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with recommended EQ adjustments
        """
        result = self.analyze(audio_path)
        ratios = self.get_band_ratios(audio_path)
        
        recommendations = {}
        
        # Bass balance
        bass_ratio = ratios.get('bass', 0) + ratios.get('sub_bass', 0)
        if bass_ratio > 0.5:
            recommendations['bass'] = 'reduce'
        elif bass_ratio < 0.2:
            recommendations['bass'] = 'boost'
        
        # Mid presence
        mid_ratio = ratios.get('mid', 0) + ratios.get('upper_mid', 0)
        if mid_ratio > 0.4:
            recommendations['mids'] = 'reduce'
        elif mid_ratio < 0.2:
            recommendations['mids'] = 'boost'
        
        # High frequency brightness
        high_ratio = ratios.get('presence', 0) + ratios.get('brilliance', 0)
        if high_ratio > 0.25:
            recommendations['highs'] = 'reduce'
        elif high_ratio < 0.1:
            recommendations['highs'] = 'boost'
        
        # Overall tonal balance based on spectral centroid
        if result['spectral'].centroid < 500:
            recommendations['overall'] = 'dark'
        elif result['spectral'].centroid > 4000:
            recommendations['overall'] = 'bright'
        else:
            recommendations['overall'] = 'balanced'
        
        return recommendations
    
    def get_frequencies_at_threshold(self, audio_path: str, 
                                      threshold_db: float = -60) -> np.ndarray:
        """
        Get frequencies that exceed a magnitude threshold.
        
        Args:
            audio_path: Path to audio file
            threshold_db: Threshold in decibels
            
        Returns:
            Array of frequencies above threshold
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Convert to dB
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Get mean across time
        mean_db = np.mean(magnitude_db, axis=1)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Filter by threshold
        active_indices = np.where(mean_db > threshold_db)[0]
        
        return frequencies[active_indices]


def analyze_audio_frequency(audio_path: str, **kwargs) -> Dict:
    """
    Convenience function for quick frequency analysis.
    
    Args:
        audio_path: Path to audio file
        **kwargs: Additional arguments passed to FrequencyAnalyzer
    
    Returns:
        Analysis results dictionary
    """
    analyzer = FrequencyAnalyzer(**kwargs)
    return analyzer.analyze(audio_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python freq_analyzer.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(audio_file)
    
    print(f"Frequency Analysis: {audio_file}")
    print(f"\nDominant Frequency: {result['dominant_freq']:.1f} Hz")
    print(f"Fundamental Frequency: {result['fundamental_freq']:.1f} Hz")
    
    print("\nFrequency Bands:")
    for band in result['bands']:
        print(f"  {band.name:12s}: {band.low_freq:6.0f} - {band.high_freq:6.0f} Hz "
              f"(energy: {band.energy:.2e}, dominant: {band.dominant_freq:.1f} Hz)")
    
    print("\nSpectral Features:")
    spectral = result['spectral']
    print(f"  Centroid:  {spectral.centroid:.1f} Hz")
    print(f"  Flux:      {spectral.flux:.4f}")
    print(f"  Rolloff:   {spectral.rolloff:.1f} Hz")
    print(f"  Flatness:  {spectral.flatness:.4f}")
    print(f"  Entropy:   {spectral.entropy:.4f}")
    
    print("\nBand Ratios:")
    ratios = analyzer.get_band_ratios(audio_file)
    for name, ratio in ratios.items():
        print(f"  {name:12s}: {ratio:.2%}")
    
    print("\nEQ Recommendations:")
    eq = analyzer.get_eq_recommendations(audio_file)
    for param, action in eq.items():
        print(f"  {param:12s}: {action}")
