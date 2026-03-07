"""
Waveform Generator
Generates visual waveforms from audio files for display in DJ interfaces.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union
import json

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class WaveformGenerator:
    """Generates visual waveforms from audio files."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        resolution: str = "medium"
    ):
        """
        Initialize waveform generator.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            n_fft: FFT window size for frequency analysis
            hop_length: Hop length for STFT
            resolution: Resolution level - 'low', 'medium', 'high'
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.resolution = resolution
        
        # Resolution presets
        self.resolution_presets = {
            "low": 100,
            "medium": 500,
            "high": 2000
        }
        self.target_points = self.resolution_presets.get(resolution, 500)
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if LIBROSA_AVAILABLE:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio, sr
        elif SOUNDFILE_AVAILABLE:
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        else:
            raise ImportError("No audio library available. Install librosa or soundfile.")
    
    def generate_waveform(
        self,
        audio: np.ndarray,
        target_points: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate waveform data from audio samples.
        
        Args:
            audio: Audio samples
            target_points: Number of points in output waveform
            
        Returns:
            Normalized waveform array
        """
        if target_points is None:
            target_points = self.target_points
        
        # Calculate samples per point
        samples_per_point = len(audio) // target_points
        
        if samples_per_point == 0:
            return np.abs(audio[:target_points]) if len(audio) >= target_points else np.pad(
                np.abs(audio), (0, target_points - len(audio))
            )
        
        # Reshape and compute RMS for each chunk
        waveform = []
        for i in range(0, len(audio) - samples_per_point, samples_per_point):
            chunk = audio[i:i + samples_per_point]
            rms = np.sqrt(np.mean(chunk ** 2))
            waveform.append(rms)
        
        # Handle remainder
        remainder = len(audio) % samples_per_point
        if remainder > 0:
            chunk = audio[-remainder:]
            rms = np.sqrt(np.mean(chunk ** 2))
            waveform.append(rms)
        
        waveform = np.array(waveform)
        
        # Normalize to 0-1 range
        if waveform.max() > 0:
            waveform = waveform / waveform.max()
        
        return waveform
    
    def generate_stereo_waveform(
        self,
        audio: np.ndarray,
        target_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stereo waveform data (left and right channels).
        
        Args:
            audio: Audio samples (mono or stereo)
            target_points: Number of points in output waveform
            
        Returns:
            Tuple of (left_waveform, right_waveform)
        """
        # Ensure stereo (duplicate mono)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        left = self.generate_waveform(audio[0], target_points)
        right = self.generate_waveform(audio[1], target_points)
        
        return left, right
    
    def generate_spectral_waveform(
        self,
        audio: np.ndarray,
        target_points: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate spectral waveform showing frequency content over time.
        
        Args:
            audio: Audio samples
            target_points: Number of points in output waveform
            
        Returns:
            Spectral waveform array (frequency bands over time)
        """
        if not LIBROSA_AVAILABLE:
            # Fallback to amplitude waveform
            return self.generate_waveform(audio, target_points)
        
        if target_points is None:
            target_points = self.target_points
        
        # Compute mel spectrogram
        n_mels = 64
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=n_mels
        )
        
        # Convert to log scale
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Resample to target points
        if S_db.shape[1] > target_points:
            indices = np.linspace(0, S_db.shape[1] - 1, target_points).astype(int)
            S_db = S_db[:, indices]
        elif S_db.shape[1] < target_points:
            S_db = np.pad(S_db, ((0, 0), (0, target_points - S_db.shape[1])), mode='edge')
        
        # Normalize each frequency band
        spectral = S_db / (np.abs(S_db).max() + 1e-8)
        
        return spectral
    
    def generate_waveform_image_data(
        self,
        waveform: np.ndarray,
        height: int = 200,
        color_scheme: str = "default"
    ) -> List[Dict]:
        """
        Convert waveform to image-ready data (bars for visualization).
        
        Args:
            waveform: Waveform array
            height: Height of each bar in pixels
            color_scheme: Color scheme - 'default', 'neon', 'monochrome'
            
        Returns:
            List of bar dictionaries with position, height, color
        """
        color_schemes = {
            "default": {
                "low": [0, 200, 100],
                "mid": [0, 150, 255],
                "high": [255, 50, 100]
            },
            "neon": {
                "low": [255, 0, 255],
                "mid": [0, 255, 255],
                "high": [255, 255, 0]
            },
            "monochrome": {
                "low": [100, 100, 100],
                "mid": [150, 150, 150],
                "high": [200, 200, 200]
            }
        }
        
        colors = color_schemes.get(color_scheme, color_schemes["default"])
        
        bars = []
        n_points = len(waveform)
        
        for i, amplitude in enumerate(waveform):
            # Determine frequency region (low/mid/high)
            region_idx = i / n_points
            if region_idx < 0.33:
                color = colors["low"]
            elif region_idx < 0.66:
                color = colors["mid"]
            else:
                color = colors["high"]
            
            bar = {
                "index": i,
                "x": i,
                "height": int(amplitude * height),
                "amplitude": float(amplitude),
                "color": color
            }
            bars.append(bar)
        
        return bars
    
    def process_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        format: str = "json"
    ) -> Dict:
        """
        Process audio file and generate waveform data.
        
        Args:
            file_path: Path to audio file
            output_path: Optional path to save waveform data
            format: Output format - 'json', 'numpy', 'image'
            
        Returns:
            Dictionary containing waveform data and metadata
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Generate waveforms
        mono_waveform = self.generate_waveform(audio)
        left_waveform, right_waveform = self.generate_stereo_waveform(audio)
        
        result = {
            "metadata": {
                "source_file": str(file_path),
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "resolution": self.resolution,
                "target_points": self.target_points
            },
            "waveform": {
                "mono": mono_waveform.tolist(),
                "stereo": {
                    "left": left_waveform.tolist(),
                    "right": right_waveform.tolist()
                }
            }
        }
        
        # Add spectral data if librosa available
        if LIBROSA_AVAILABLE:
            spectral = self.generate_spectral_waveform(audio)
            result["waveform"]["spectral"] = spectral.tolist()
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(result, f)
            elif format == "numpy":
                np.save(output_path, result["waveform"])
        
        return result
    
    def get_waveform_summary(
        self,
        waveform: np.ndarray
    ) -> Dict:
        """
        Get summary statistics for waveform.
        
        Args:
            waveform: Waveform array
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            "length": len(waveform),
            "mean": float(np.mean(waveform)),
            "std": float(np.std(waveform)),
            "max": float(np.max(waveform)),
            "min": float(np.min(waveform)),
            "dynamic_range": float(np.max(waveform) - np.min(waveform))
        }


def generate_waveform(
    file_path: str,
    output_path: Optional[str] = None,
    resolution: str = "medium",
    sample_rate: int = 22050
) -> Dict:
    """
    Convenience function to generate waveform from audio file.
    
    Args:
        file_path: Path to audio file
        output_path: Optional path to save output
        resolution: Resolution - 'low', 'medium', 'high'
        sample_rate: Sample rate for loading audio
        
    Returns:
        Waveform data dictionary
    """
    generator = WaveformGenerator(
        sample_rate=sample_rate,
        resolution=resolution
    )
    return generator.process_file(file_path, output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python waveform_generator.py <audio_file> [output_file]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = generate_waveform(audio_file, output_file)
        print(f"Generated waveform: {result['metadata']['target_points']} points")
        print(f"Duration: {result['metadata']['duration']:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
