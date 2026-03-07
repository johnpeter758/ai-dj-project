"""
Audio Metering System - Comprehensive audio level and quality measurement

Provides multiple metering types:
- Peak/RMS level metering
- LUFS loudness measurement (EBU R128)
- Spectrum analysis (FFT)
- Phase correlation
- Stereo width metering
- Crest factor

Dependencies: numpy, scipy, soundfile
Install: pip install numpy scipy soundfile
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    import numpy as np
    from scipy import signal
    from scipy.fft import rfft, rfftfreq
    import soundfile as sf
except ImportError as e:
    print(f"Error: Missing required package. Install with: pip install numpy scipy soundfile")
    print(f"Import error: {e}")
    sys.exit(1)


@dataclass
class MeterData:
    """Container for metering results."""
    peak_db: float
    rms_db: float
    crest_factor: float
    integrated_lufs: float
    true_peak_db: float
    loudness_range: float
    momentary_lufs: float
    short_term_lufs: float
    correlation: float
    stereo_width: float
    spectrum: Optional[np.ndarray] = None
    frequencies: Optional[np.ndarray] = None


class AudioMeter:
    """Comprehensive audio metering class."""
    
    def __init__(self, sample_rate: int = 44100, block_size: int = 2048):
        """
        Initialize the audio meter.
        
        Args:
            sample_rate: Audio sample rate in Hz
            block_size: FFT block size for analysis
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.hop_size = block_size // 2
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return samples and sample rate."""
        data, sr = sf.read(audio_path)
        
        # Convert stereo to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            
        return data, sr
    
    def db_to_linear(self, db: float) -> float:
        """Convert dB to linear scale."""
        return 10 ** (db / 20)
    
    def linear_to_db(self, linear: float) -> float:
        """Convert linear to dB scale."""
        if linear <= 0:
            return -np.inf
        return 20 * np.log10(linear)
    
    def measure_levels(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """
        Measure peak and RMS levels.
        
        Returns:
            Tuple of (peak_db, rms_db, crest_factor_db)
        """
        # Peak level
        peak_linear = np.max(np.abs(audio))
        peak_db = self.linear_to_db(peak_linear)
        
        # RMS level
        rms_linear = np.sqrt(np.mean(audio ** 2))
        rms_db = self.linear_to_db(rms_linear)
        
        # Crest factor (peak to RMS ratio)
        crest_factor = peak_linear / rms_linear if rms_linear > 0 else np.inf
        crest_factor_db = self.linear_to_db(crest_factor)
        
        return peak_db, rms_db, crest_factor_db
    
    def measure_lufs(self, audio: np.ndarray, sr: int) -> dict:
        """
        Measure LUFS loudness metrics (EBU R128).
        
        Returns:
            Dictionary with integrated, momentary, short-term LUFS and true peak
        """
        # K-weighting filter coefficients (approximate)
        # Pre-filter
        b_pre, a_pre = signal.butter(2, [38.5 / (sr / 2)], btype='high')
        # Post-filter  
        b_post, a_post = signal.butter(2, [5380 / (sr / 2)], btype='low')
        
        # Apply K-weighting
        try:
            audio_k = signal.filtfilt(b_pre, a_pre, audio)
            audio_k = signal.filtfilt(b_post, a_post, audio_k)
        except:
            audio_k = audio  # Fallback if filtering fails
            
        # Block size for LUFS (400ms)
        block_samples = int(0.4 * sr)
        
        # Calculate mean square per block
        blocks = len(audio_k) // block_samples
        if blocks == 0:
            blocks = 1
            block_samples = len(audio_k)
            
        mean_squares = []
        for i in range(blocks):
            start = i * block_samples
            end = min(start + block_samples, len(audio_k))
            block = audio_k[start:end]
            mean_squares.append(np.mean(block ** 2))
        
        mean_squares = np.array(mean_squares)
        
        # Convert to LUFS (simplified)
        # -0.691 is the offset for LKFS
        integrated_lufs = -0.691 + self.linear_to_db(np.sqrt(np.mean(mean_squares)))
        
        # Momentary loudness (400ms)
        momentary_lufs = -0.691 + self.linear_to_db(np.sqrt(np.mean(audio_k[-block_samples:] ** 2)))
        
        # Short-term loudness (3s)
        short_block = int(3.0 * sr)
        if len(audio_k) >= short_block:
            short_term_lufs = -0.691 + self.linear_to_db(np.sqrt(np.mean(audio_k[-short_block:] ** 2)))
        else:
            short_term_lufs = momentary_lufs
            
        # True peak (oversampled)
        try:
            # 4x oversampling for true peak detection
            up_ratio = 4
            audio_up = signal.resample(audio, len(audio) * up_ratio)
            true_peak_db = self.linear_to_db(np.max(np.abs(audio_up)))
        except:
            true_peak_db = self.linear_to_db(np.max(np.abs(audio)))
            
        # Loudness range (simplified - std of momentary)
        loudness_range = np.std(mean_squares) * 10  # Simplified approximation
        
        return {
            "integrated": integrated_lufs,
            "momentary": momentary_lufs,
            "short_term": short_term_lufs,
            "true_peak": true_peak_db,
            "loudness_range": loudness_range
        }
    
    def measure_spectrum(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT spectrum.
        
        Returns:
            Tuple of (magnitude_spectrum_db, frequencies)
        """
        # Apply Hann window
        window = signal.hann(self.block_size)
        
        # Pad or truncate to block size
        if len(audio) < self.block_size:
            audio = np.pad(audio, (0, self.block_size - len(audio)))
        
        # Extract overlapping frames
        frames = []
        for i in range(0, len(audio) - self.block_size + 1, self.hop_size):
            frame = audio[i:i + self.block_size] * window
            frames.append(frame)
        
        if not frames:
            frames = [audio[:self.block_size] * window]
            
        # Average spectrum
        spectrum = np.mean([np.abs(rfft(frame)) for frame in frames], axis=0)
        
        # Convert to dB
        spectrum_db = self.linear_to_db(spectrum + 1e-10)
        
        # Frequency bins
        freqs = rfftfreq(self.block_size, 1 / self.sample_rate)
        
        return spectrum_db, freqs
    
    def measure_stereo_correlation(self, audio: np.ndarray) -> float:
        """
        Measure stereo phase correlation (-1 to +1).
        
        Returns:
            Correlation coefficient (-1 = out of phase, 0 = mono, 1 = in phase)
        """
        # This requires stereo audio - handle mono case
        if len(audio.shape) == 1:
            return 1.0  # Mono is perfectly correlated
            
        if audio.shape[1] < 2:
            return 1.0
            
        left = audio[:, 0]
        right = audio[:, 1]
        
        # Calculate correlation
        correlation = np.mean(left * right) / (
            np.sqrt(np.mean(left ** 2)) * np.sqrt(np.mean(right ** 2)) + 1e-10
        )
        
        return np.clip(correlation, -1, 1)
    
    def measure_stereo_width(self, audio: np.ndarray) -> float:
        """
        Measure stereo width (0 = mono, 1+ = wide).
        
        Returns:
            Stereo width factor
        """
        if len(audio.shape) == 1:
            return 0.0
            
        left = audio[:, 0]
        right = audio[:, 1]
        
        mid = (left + right) / 2
        side = (left - right) / 2
        
        rms_mid = np.sqrt(np.mean(mid ** 2))
        rms_side = np.sqrt(np.mean(side ** 2))
        
        if rms_mid < 1e-10:
            return 0.0
            
        # Width = mid / (mid + side) normalized
        width = rms_side / (rms_mid + 1e-10)
        
        return width
    
    def analyze(self, audio_path: str, include_spectrum: bool = True) -> MeterData:
        """
        Perform complete meter analysis on audio file.
        
        Args:
            audio_path: Path to audio file
            include_spectrum: Whether to compute FFT spectrum
            
        Returns:
            MeterData object with all measurements
        """
        # Load audio
        data, sr = self.sample_rate = sr
        
        # Check if stereo
        original_shape = data.shape
        is_stereo = len(original_shape) > 1 and original_shape[1] >= 2
        
        # Get mono for level measurement
        if is_stereo:
            mono = np.mean(data, axis=1)
        else:
            mono = data
            
        # Measure levels
        peak_db, rms_db, crest_factor = self.measure_levels(mono)
        
        # Measure LUFS
        lufs = self.measure_lufs(mono, sr)
        
        # Measure stereo
        if is_stereo:
            correlation = self.measure_stereo_correlation(data)
            width = self.measure_stereo_width(data)
        else:
            correlation = 1.0
            width = 0.0
            
        # Measure spectrum
        spectrum = None
        frequencies = None
        if include_spectrum:
            spectrum, frequencies = self.measure_spectrum(mono)
            
        return MeterData(
            peak_db=peak_db,
            rms_db=rms_db,
            crest_factor=crest_factor,
            integrated_lufs=lufs["integrated"],
            true_peak_db=lufs["true_peak"],
            loudness_range=lufs["loudness_range"],
            momentary_lufs=lufs["momentary"],
            short_term_lufs=lufs["short_term"],
            correlation=correlation,
            stereo_width=width,
            spectrum=spectrum,
            frequencies=frequencies
        )
    
    def analyze_blocks(self, audio: np.ndarray, sr: int) -> List[dict]:
        """
        Analyze audio in blocks for real-time metering.
        
        Args:
            audio: Audio samples
            sr: Sample rate
            
        Returns:
            List of meter readings per block
        """
        block_samples = int(0.1 * sr)  # 100ms blocks
        
        results = []
        for i in range(0, len(audio) - block_samples + 1, block_samples):
            block = audio[i:i + block_samples]
            
            peak_db, rms_db, crest = self.measure_levels(block)
            lufs = self.measure_lufs(block, sr)
            
            results.append({
                "position": i / sr,  # Time in seconds
                "peak_db": peak_db,
                "rms_db": rms_db,
                "momentary_lufs": lufs["momentary"]
            })
            
        return results


def print_meter_report(meter_data: MeterData, filename: str = ""):
    """Print formatted metering report."""
    if filename:
        print(f"\n📊 Audio Metering Report: {filename}")
        print("=" * 50)
    
    print("\n🎚️  Level Metering:")
    print(f"   Peak Level:      {meter_data.peak_db:+.1f} dB")
    print(f"   RMS Level:       {meter_data.rms_db:+.1f} dB")
    print(f"   Crest Factor:    {meter_data.crest_factor:+.1f} dB")
    
    print("\n🔊 Loudness (LUFS):")
    print(f"   Integrated:      {meter_data.integrated_lufs:.1f} LUFS")
    print(f"   True Peak:       {meter_data.true_peak_db:.1f} dBTP")
    print(f"   Loudness Range:  {meter_data.loudness_range:.1f} LU")
    print(f"   Momentary:       {meter_data.momentary_lufs:.1f} LUFS")
    print(f"   Short-term:      {meter_data.short_term_lufs:.1f} LUFS")
    
    print("\n🎚️  Stereo:")
    print(f"   Correlation:     {meter_data.correlation:+.2f}")
    print(f"   Width:           {meter_data.stereo_width:.2f}")
    
    if meter_data.spectrum is not None:
        # Find dominant frequencies
        spectrum = meter_data.spectrum
        freqs = meter_data.frequencies
        top_indices = np.argsort(spectrum)[-5:][::-1]
        
        print("\n🎵 Top Frequencies:")
        for idx in top_indices:
            if freqs[idx] > 20 and freqs[idx] < 20000:
                print(f"   {freqs[idx]/1000:.1f} kHz: {spectrum[idx]:.1f} dB")
    
    print("=" * 50)


# Convenience functions
def measure(audio_path: str, include_spectrum: bool = True) -> MeterData:
    """Quick measure function."""
    meter = AudioMeter()
    return meter.analyze(audio_path, include_spectrum)


def measure_levels(audio_path: str) -> Tuple[float, float]:
    """Quick peak/RMS measurement."""
    meter = AudioMeter()
    data, sr = meter.load_audio(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return meter.measure_levels(data)


def measure_lufs(audio_path: str) -> dict:
    """Quick LUFS measurement."""
    meter = AudioMeter()
    data, sr = meter.load_audio(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return meter.measure_lufs(data, sr)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Metering System")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--no-spectrum", action="store_true", 
                        help="Skip spectrum analysis")
    parser.add_argument("--blocks", action="store_true",
                        help="Show per-block analysis")
    
    args = parser.parse_args()
    
    if not Path(args.audio_file).exists():
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)
    
    meter = AudioMeter()
    data, sr = meter.load_audio(args.audio_file)
    
    # Full analysis
    print("Performing full analysis...")
    meter_data = meter.analyze(args.audio_file, not args.no_spectrum)
    print_meter_report(meter_data, args.audio_file)
    
    # Block analysis
    if args.blocks:
        print("\n📈 Block Analysis (100ms blocks):")
        print("-" * 40)
        blocks = meter.analyze_blocks(data, sr)
        
        # Show first 10 blocks
        for block in blocks[:10]:
            peak_bar = "█" * int(max(0, (block["peak_db"] + 60) / 2))
            rms_bar = "▓" * int(max(0, (block["rms_db"] + 60) / 2))
            print(f"{block['position']:5.2f}s | {peak_bar:30} {block['peak_db']:+.1f} dB")
            print(f"         | {rms_bar:30} {block['rms_db']:+.1f} dB  ({block['momentary_lufs']:.1f} LUFS)")
