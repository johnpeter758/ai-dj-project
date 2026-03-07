#!/usr/bin/env python3
"""
Octave Divider - Split Audio into Octave Bands
Divides audio into separate octave frequency bands for independent processing
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import signal
from dataclasses import dataclass


@dataclass
class OctaveBand:
    """Represents a single octave frequency band"""
    name: str          # e.g., "Sub", "Bass", "Mid", "High-Mid", "Treble"
    low_freq: float    # Low frequency cutoff in Hz
    high_freq: float   # High frequency cutoff in Hz
    audio: np.ndarray  # The audio data for this band


class OctaveDivider:
    """
    Split audio into octave frequency bands for independent processing.
    
    Useful for DJ mixing, applying different effects to different frequency
    ranges, or isolating specific octave ranges for analysis.
    """
    
    # Standard octave band definitions (based on ISO standards)
    OCTAVE_BANDS = [
        {"name": "Sub", "low": 20, "high": 60},
        {"name": "Bass", "low": 60, "high": 250},
        {"name": "Low-Mid", "low": 250, "high": 500},
        {"name": "Mid", "low": 500, "high": 2000},
        {"name": "High-Mid", "low": 2000, "high": 4000},
        {"name": "Presence", "low": 4000, "high": 6000},
        {"name": "Treble", "low": 6000, "high": 12000},
        {"name": "Air", "low": 12000, "high": 20000},
    ]
    
    # 1/3 octave bands (more granular)
    THIRD_OCTAVE_BANDS = [
        {"name": "Sub 1", "low": 20, "high": 25},
        {"name": "Sub 2", "low": 25, "high": 31.5},
        {"name": "Sub 3", "low": 31.5, "high": 40},
        {"name": "Bass 1", "low": 40, "high": 50},
        {"name": "Bass 2", "low": 50, "high": 63},
        {"name": "Bass 3", "low": 63, "high": 80},
        {"name": "Low 1", "low": 80, "high": 100},
        {"name": "Low 2", "low": 100, "high": 125},
        {"name": "Low 3", "low": 125, "high": 160},
        {"name": "Low-Mid 1", "low": 160, "high": 200},
        {"name": "Low-Mid 2", "low": 200, "high": 250},
        {"name": "Low-Mid 3", "low": 250, "high": 315},
        {"name": "Mid 1", "low": 315, "high": 400},
        {"name": "Mid 2", "low": 400, "high": 500},
        {"name": "Mid 3", "low": 500, "high": 630},
        {"name": "Mid-High 1", "low": 630, "high": 800},
        {"name": "Mid-High 2", "low": 800, "high": 1000},
        {"name": "Mid-High 3", "low": 1000, "high": 1250},
        {"name": "High-Mid 1", "low": 1250, "high": 1600},
        {"name": "High-Mid 2", "low": 1600, "high": 2000},
        {"name": "High-Mid 3", "low": 2000, "high": 2500},
        {"name": "Presence 1", "low": 2500, "high": 3150},
        {"name": "Presence 2", "low": 3150, "high": 4000},
        {"name": "Presence 3", "low": 4000, "high": 5000},
        {"name": "Treble 1", "low": 5000, "high": 6300},
        {"name": "Treble 2", "low": 6300, "high": 8000},
        {"name": "Treble 3", "low": 8000, "high": 10000},
        {"name": "Air 1", "low": 10000, "high": 12500},
        {"name": "Air 2", "low": 12500, "high": 16000},
        {"name": "Air 3", "low": 16000, "high": 20000},
    ]
    
    def __init__(self, sample_rate: int = 44100, 
                 num_bands: int = 8,
                 use_third_octave: bool = False):
        """
        Initialize the octave divider.
        
        Args:
            sample_rate: Audio sample rate in Hz
            num_bands: Number of octave bands (default 8 for full range)
            use_third_octave: Use 1/3 octave bands instead of full octave
        """
        self.sample_rate = sample_rate
        self.use_third_octave = use_third_octave
        
        if use_third_octave:
            self.bands = self.THIRD_OCTAVE_BANDS
            self.num_bands = len(self.THIRD_OCTAVE_BANDS)
        else:
            self.bands = self.OCTAVE_BANDS[:num_bands]
            self.num_bands = num_bands
        
        # Precompute filter coefficients
        self._filter_coeffs = self._design_filters()
    
    def _design_filters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Design Butterworth bandpass filters for each octave band.
        
        Returns:
            List of (b, a) filter coefficient tuples
        """
        filters = []
        nyquist = self.sample_rate / 2
        
        for band in self.bands:
            # Convert frequencies to normalized range (0-1)
            low_norm = band["low"] / nyquist
            high_norm = min(band["high"] / nyquist, 0.99)
            
            # Design bandpass filter
            # Higher order for better separation
            b, a = signal.butter(
                4,  # 4th order filter
                [low_norm, high_norm],
                btype='bandpass'
            )
            filters.append((b, a))
        
        return filters
    
    def split(self, audio: np.ndarray) -> List[OctaveBand]:
        """
        Split audio into separate octave bands.
        
        Args:
            audio: Input audio signal (mono or stereo)
            
        Returns:
            List of OctaveBand objects, one for each frequency band
        """
        # Handle stereo
        if audio.ndim == 2:
            # Process each channel and average the energy
            left_bands = self._process_channel(audio[:, 0])
            right_bands = self._process_channel(audio[:, 1])
            
            result = []
            for i, band in enumerate(self.bands):
                stereo_band = np.stack([
                    left_bands[i],
                    right_bands[i]
                ], axis=1)
                result.append(OctaveBand(
                    name=band["name"],
                    low_freq=band["low"],
                    high_freq=band["high"],
                    audio=stereo_band
                ))
            return result
        else:
            return self._process_channel(audio, is_stereo=False)
    
    def _process_channel(self, audio: np.ndarray, is_stereo: bool = True
                         ) -> List[OctaveBand]:
        """Process a single channel of audio."""
        result = []
        
        for i, (b, a) in enumerate(self._filter_coeffs):
            # Apply filter
            filtered = signal.filtfilt(b, a, audio)
            result.append(OctaveBand(
                name=self.bands[i]["name"],
                low_freq=self.bands[i]["low"],
                high_freq=self.bands[i]["high"],
                audio=filtered
            ))
        
        return result
    
    def merge(self, bands: List[OctaveBand]) -> np.ndarray:
        """
        Merge octave bands back into a single audio signal.
        
        Args:
            bands: List of OctaveBand objects to merge
            
        Returns:
            Merged audio signal
        """
        if not bands:
            raise ValueError("No bands provided")
        
        # Sum all bands (they should already be filtered to non-overlapping ranges)
        if bands[0].audio.ndim == 2:
            # Stereo
            merged = np.zeros_like(bands[0].audio)
            for band in bands:
                merged += band.audio
        else:
            # Mono
            merged = np.zeros(len(bands[0].audio))
            for band in bands:
                merged += band.audio
        
        return merged
    
    def get_band(self, audio: np.ndarray, band_name: str) -> np.ndarray:
        """
        Extract a specific frequency band from audio.
        
        Args:
            audio: Input audio
            band_name: Name of the band to extract
            
        Returns:
            Audio for the specified band
        """
        for i, band in enumerate(self.bands):
            if band["name"] == band_name:
                b, a = self._filter_coeffs[i]
                return signal.filtfilt(b, a, audio)
        
        raise ValueError(f"Band '{band_name}' not found")
    
    def apply_gain_per_band(self, audio: np.ndarray, 
                            gains: Dict[str, float]) -> np.ndarray:
        """
        Apply different gain to each octave band.
        
        Args:
            audio: Input audio
            gains: Dictionary mapping band names to gain values (in dB)
            
        Returns:
            Audio with per-band gain applied
        """
        bands = self.split(audio)
        
        for band in bands:
            if band.name in gains:
                gain_linear = 10 ** (gains[band.name] / 20.0)
                band.audio *= gain_linear
        
        return self.merge(bands)
    
    def get_band_energies(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Calculate the energy in each octave band.
        
        Args:
            audio: Input audio
            
        Returns:
            Dictionary mapping band names to RMS energy values
        """
        bands = self.split(audio)
        energies = {}
        
        for band in bands:
            if band.audio.ndim == 2:
                # Stereo - use mean of both channels
                energy = np.mean(np.sqrt(np.mean(band.audio ** 2, axis=0)))
            else:
                energy = np.sqrt(np.mean(band.audio ** 2))
            energies[band.name] = energy
        
        return energies
    
    def visualize_bands(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Get frequency band levels for visualization.
        
        Args:
            audio: Input audio
            
        Returns:
            Dictionary with band info and levels (in dB)
        """
        energies = self.get_band_energies(audio)
        
        result = {}
        for band in self.bands:
            name = band["name"]
            energy = energies.get(name, 0)
            
            # Convert to dB
            if energy > 0:
                db = 20 * np.log10(energy)
            else:
                db = -96  # Minimum dB value
            
            result[name] = {
                "low_freq": band["low"],
                "high_freq": band["high"],
                "energy": energy,
                "db": db
            }
        
        return result


def octave_divide(audio: np.ndarray, sample_rate: int = 44100,
                  num_bands: int = 8) -> List[OctaveBand]:
    """
    Convenience function to divide audio into octave bands.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        num_bands: Number of octave bands
        
    Returns:
        List of OctaveBand objects
    """
    divider = OctaveDivider(sample_rate=sample_rate, num_bands=num_bands)
    return divider.split(audio)


def octave_merge(bands: List[OctaveBand]) -> np.ndarray:
    """
    Convenience function to merge octave bands back to audio.
    
    Args:
        bands: List of OctaveBand objects
        
    Returns:
        Merged audio signal
    """
    divider = OctaveDivider()
    return divider.merge(bands)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Load audio file
        try:
            import librosa
            audio, sr = librosa.load(sys.argv[1], sr=44100, mono=False)
            
            # Handle librosa returning (2, n) for stereo
            if audio.shape[0] == 2:
                audio = audio.T
            
            divider = OctaveDivider(sample_rate=sr)
            bands = divider.split(audio)
            
            print(f"Split audio into {len(bands)} octave bands:")
            for band in bands:
                print(f"  {band.name}: {band.low_freq}-{band.high_freq} Hz")
            
            energies = divider.get_band_energies(audio)
            print("\nBand energies:")
            for name, energy in energies.items():
                db = 20 * np.log10(energy) if energy > 0 else -96
                print(f"  {name}: {db:.1f} dB")
                
        except ImportError:
            print("librosa not installed. Install with: pip install librosa")
    else:
        # Generate test signal
        print("Running demo with synthetic audio...")
        
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio with multiple frequency components
        audio = (
            0.3 * np.sin(2 * np.pi * 50 * t) +    # Sub bass
            0.3 * np.sin(2 * np.pi * 200 * t) +    # Bass
            0.2 * np.sin(2 * np.pi * 1000 * t) +   # Mid
            0.2 * np.sin(2 * np.pi * 5000 * t)     # Treble
        )
        
        divider = OctaveDivider(sample_rate=sample_rate)
        bands = divider.split(audio)
        
        print(f"Split audio into {len(bands)} octave bands:")
        for band in bands:
            energy = np.sqrt(np.mean(band.audio ** 2))
            db = 20 * np.log10(energy) if energy > 0 else -96
            print(f"  {band.name} ({band.low_freq}-{band.high_freq} Hz): {db:.1f} dB")
        
        # Test gain application
        print("\nApplying +6dB to bass, -6dB to treble...")
        modified = divider.apply_gain_per_band(audio, {
            "Bass": 6,
            "Treble": -6
        })
        
        print("Done!")
