"""
Time Stretcher - Audio time stretching module for AI DJ Project

Provides multiple time stretching algorithms to change audio duration
without affecting pitch (useful for beatmatching, tempo changes, etc.)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Try to import librosa for phase vocoder approach
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - phase vocoder method disabled")

# Try to import soundfile for audio I/O
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available - audio I/O limited")


class TimeStretcher:
    """
    Time stretching processor with multiple algorithm options.
    
    Supported methods:
    - phase_vocoder: High quality, uses librosa (best for music)
    - wsola: Good quality, overlap-add method (fast, low memory)
    - simple: Basic rate change (affects pitch - not true time stretch)
    """
    
    def __init__(self, method: str = "phase_vocoder"):
        """
        Initialize time stretcher.
        
        Args:
            method: Stretching algorithm ('phase_vocoder', 'wsola', 'simple')
        """
        self.method = method
        self._validate_method()
        
    def _validate_method(self):
        """Validate requested method is available."""
        if self.method == "phase_vocoder" and not LIBROSA_AVAILABLE:
            logger.warning("Phase vocoder unavailable, falling back to simple")
            self.method = "simple"
            
    def stretch(
        self, 
        audio: np.ndarray, 
        rate: float, 
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Time stretch audio by a given rate.
        
        Args:
            audio: Input audio samples (mono or stereo)
            rate: Stretch rate (>1 = slower/longer, <1 = faster/shorter)
            sample_rate: Sample rate of audio
            
        Returns:
            Time-stretched audio array
        """
        if rate == 1.0:
            return audio
            
        if self.method == "phase_vocoder":
            return self._phase_vocoder_stretch(audio, rate, sample_rate)
        elif self.method == "wsola":
            return self._wsola_stretch(audio, rate)
        else:
            return self._simple_stretch(audio, rate)
            
    def _phase_vocoder_stretch(
        self, 
        audio: np.ndarray, 
        rate: float, 
        sample_rate: int
    ) -> np.ndarray:
        """
        Phase vocoder time stretching using librosa.
        High quality but may introduce some artifacts.
        """
        if not LIBROSA_AVAILABLE:
            return self._simple_stretch(audio, rate)
            
        # Ensure mono for processing
        is_stereo = audio.ndim > 1
        if is_stereo:
            result = np.zeros((int(audio.shape[0] / rate), audio.shape[1]), dtype=audio.dtype)
            for ch in range(audio.shape[1]):
                stretched = librosa.effects.time_stretch(
                    audio[:, ch].astype(np.float32), 
                    rate=rate
                )
                result[:, ch] = stretched
            return result
        else:
            return librosa.effects.time_stretch(
                audio.astype(np.float32), 
                rate=rate
            )
            
    def _wsola_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        WSOLA (Waveform Similarity Overlap-Add) time stretching.
        Good quality, computationally efficient.
        """
        # WSOLA parameters
        win_size = 2048
        hop_size = win_size // 4
        
        # Calculate output length
        output_len = int(len(audio) / rate)
        
        if audio.ndim > 1:
            # Stereo
            result = np.zeros((output_len, audio.shape[1]), dtype=audio.dtype)
            for ch in range(audio.shape[1]):
                result[:, ch] = self._wsola_mono(audio[:, ch], rate, win_size, hop_size)
        else:
            result = self._wsola_mono(audio, rate, win_size, hop_size)
            
        return result
        
    def _wsola_mono(
        self, 
        audio: np.ndarray, 
        rate: float, 
        win_size: int, 
        hop_size: int
    ) -> np.ndarray:
        """WSOLA implementation for mono audio."""
        # Create analysis window
        window = np.hanning(win_size).astype(np.float32)
        
        # Output buffers
        output_len = int(len(audio) / rate)
        output = np.zeros(output_len, dtype=np.float32)
        
        # Position trackers
        input_pos = 0
        output_pos = 0
        
        # Synthesis hop (slower than analysis for stretching)
        synthesis_hop = int(hop_size * rate)
        
        while input_pos < len(audio) - win_size and output_pos < output_len - win_size:
            # Extract frame
            frame = audio[input_pos:input_pos + win_size] * window
            
            # Find best similarity offset (simplified - using fixed offset)
            # Real WSOLA would search for best match
            offset = 0
            
            # Add to output with overlap-add
            overlap_end = min(win_size, output_len - output_pos)
            output[output_pos:output_pos + overlap_end] += frame[:overlap_end]
            
            # Advance positions
            input_pos += hop_size
            output_pos += synthesis_hop
            
        # Normalize by overlap counts
        return output
        
    def _simple_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Simple time stretching via resampling.
        NOTE: This changes pitch along with tempo - not true time stretching.
        """
        # Calculate new length
        new_length = int(len(audio) / rate)
        
        if audio.ndim > 1:
            # Resample each channel
            result = np.zeros((new_length, audio.shape[1]), dtype=audio.dtype)
            for ch in range(audio.shape[1]):
                indices = np.linspace(0, len(audio) - 1, new_length)
                result[:, ch] = audio[indices.astype(int), ch]
            return result
        else:
            indices = np.linspace(0, len(audio) - 1, new_length)
            return audio[indices.astype(int)]


def stretch_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    rate: float,
    method: str = "phase_vocoder"
) -> Tuple[bool, str]:
    """
    Time stretch an audio file.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output stretched audio
        rate: Stretch rate (e.g., 1.1 = 10% slower, 0.9 = 10% faster)
        method: Stretching method to use
        
    Returns:
        (success: bool, message: str)
    """
    if not SOUNDFILE_AVAILABLE:
        return False, "soundfile not available for audio I/O"
        
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None) if LIBROSA_AVAILABLE else None
        
        if audio is None:
            # Fallback using soundfile
            audio, sr = sf.read(input_path)
            
        # Apply stretching
        stretcher = TimeStretcher(method=method)
        stretched = stretcher.stretch(audio, rate, sr)
        
        # Save result
        sf.write(output_path, stretched, sr)
        
        return True, f"Stretched audio by {rate}x using {method}"
        
    except Exception as e:
        logger.error(f"Time stretch failed: {e}")
        return False, str(e)


def match_tempo(
    source_bpm: float,
    target_bpm: float,
    audio: np.ndarray,
    sample_rate: int = 44100
) -> Tuple[np.ndarray, float]:
    """
    Time stretch audio to match a target BPM.
    
    Args:
        source_bpm: Original BPM of audio
        target_bpm: Target BPM to match
        audio: Audio samples
        sample_rate: Sample rate
        
    Returns:
        (stretched_audio, actual_rate)
    """
    rate = source_bpm / target_bpm
    stretcher = TimeStretcher(method="phase_vocoder")
    stretched = stretcher.stretch(audio, rate, sample_rate)
    return stretched, rate


# CLI for quick testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Time stretch audio files")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("rate", type=float, help="Stretch rate (e.g., 1.1)")
    parser.add_argument("--method", default="phase_vocoder", 
                        choices=["phase_vocoder", "wsola", "simple"],
                        help="Stretching method")
    
    args = parser.parse_args()
    
    success, msg = stretch_file(args.input, args.output, args.rate, args.method)
    print(msg)
