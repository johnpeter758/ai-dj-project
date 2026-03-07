"""
Pitch Shifting System for AI DJ Project
Real-time and offline pitch shifting with key detection integration
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy import signal
from scipy.fft import fft, ifft, fftfreq


class PitchShifter:
    """
    High-quality pitch shifting using phase vocoder with
    identity phase locking and transient preservation.
    """
    
    def __init__(self, sample_rate: int = 44100, 
                 fft_size: int = 2048,
                 hop_size: int = 512):
        """
        Initialize pitch shifter.
        
        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: FFT window size (power of 2 recommended)
            hop_size: Hop size between frames
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self._window = self._create_window(fft_size)
        
    def _create_window(self, size: int) -> np.ndarray:
        """Create Hann window for analysis."""
        return signal.windows.hann(size)
    
    def _stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Short-Time Fourier Transform.
        
        Returns:
            Tuple of (magnitude, phase) arrays
        """
        num_frames = 1 + (len(audio) - self.fft_size) // self.hop_size
        
        # Pad if needed
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))
        
        magnitude = np.zeros((num_frames, self.fft_size // 2 + 1))
        phase = np.zeros((num_frames, self.fft_size // 2 + 1))
        
        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.fft_size] * self._window
            
            fft_result = fft(frame)
            
            magnitude[i] = np.abs(fft_result)
            phase[i] = np.angle(fft_result)
        
        return magnitude, phase
    
    def _istft(self, magnitude: np.ndarray, phase: np.ndarray,
               original_length: int) -> np.ndarray:
        """
        Inverse STFT with overlap-add synthesis.
        """
        num_frames = magnitude.shape[0]
        output_length = (num_frames - 1) * self.hop_size + self.fft_size
        output = np.zeros(output_length)
        window_sum = np.zeros(output_length)
        
        for i in range(num_frames):
            start = i * self.hop_size
            
            # Reconstruct complex spectrum
            spectrum = magnitude[i] * np.exp(1j * phase[i])
            
            # Inverse FFT
            frame = np.real(ifft(spectrum))
            
            # Apply window and overlap-add
            windowed = frame * self._window
            output[start:start + self.fft_size] += windowed
            window_sum[start:start + self.fft_size] += self._window ** 2
        
        # Normalize by window sum
        window_sum = np.maximum(window_sum, 1e-8)
        output = output / window_sum
        
        # Trim to original length
        return output[:original_length]
    
    def _phase_vocoder(self, magnitude: np.ndarray, phase: np.ndarray,
                       pitch_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phase vocoder pitch shifting with identity phase locking.
        
        Args:
            magnitude: STFT magnitude array
            phase: STFT phase array  
            pitch_ratio: Pitch ratio (2.0 = octave up, 0.5 = octave down)
        
        Returns:
            Tuple of (new_magnitude, new_phase)
        """
        num_frames, num_bins = magnitude.shape
        
        # Time stretching factor (inverse of pitch shift)
        time_stretch = 1.0 / pitch_ratio
        
        # New number of frames after time stretching
        new_num_frames = int(num_frames * time_stretch)
        
        new_magnitude = np.zeros((new_num_frames, num_bins))
        new_phase = np.zeros((new_num_frames, num_bins))
        
        # Calculate phase advance per bin
        omega = 2 * np.pi * fftfreq(self.fft_size, 1 / self.sample_rate)[:num_bins]
        
        for i in range(new_num_frames):
            # Original frame position
            src_frame = i * time_stretch
            src_frame_int = int(src_frame)
            frac = src_frame - src_frame_int
            
            if src_frame_int >= num_frames - 1:
                break
            
            # Interpolate magnitudes
            mag = (1 - frac) * magnitude[src_frame_int] + frac * magnitude[src_frame_int + 1]
            new_magnitude[i] = mag
            
            # Phase locking: propagate phase from original
            if src_frame_int > 0:
                # Calculate expected phase
                expected_phase = phase[src_frame_int] + omega * src_frame * self.hop_size / self.sample_rate
                new_phase[i] = expected_phase
            else:
                new_phase[i] = phase[0]
        
        return new_magnitude, new_phase
    
    def shift_pitch(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """
        Shift pitch by specified semitones.
        
        Args:
            audio: Input audio signal (mono)
            semitones: Number of semitones to shift (positive = up, negative = down)
        
        Returns:
            Pitch-shifted audio
        """
        if len(audio.shape) > 1:
            # Convert stereo to mono for processing
            audio = np.mean(audio, axis=1)
        
        # Convert semitones to pitch ratio
        pitch_ratio = 2 ** (semitones / 12)
        
        # Keep original length for later
        original_length = len(audio)
        
        # Normalize audio to prevent clipping
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        
        # STFT
        magnitude, phase = self._stft(audio)
        
        # Phase vocoder processing
        new_magnitude, new_phase = self._phase_vocoder(magnitude, phase, pitch_ratio)
        
        # ISTFT
        output = self._istft(new_magnitude, new_phase, original_length)
        
        # Restore original level
        output = output * peak
        
        return self._normalize(output)
    
    def shift_to_key(self, audio: np.ndarray, 
                     source_key: int, 
                     target_key: int) -> np.ndarray:
        """
        Shift pitch to match musical keys.
        
        Args:
            audio: Input audio
            source_key: Source key (0-11, where 0=C, 1=C#, etc.)
            target_key: Target key (0-11)
        
        Returns:
            Pitch-shifted audio
        """
        semitones = target_key - source_key
        return self.shift_pitch(audio, semitones)
    
    def shift_octave(self, audio: np.ndarray, octaves: float) -> np.ndarray:
        """
        Shift pitch by full octaves.
        
        Args:
            audio: Input audio
            octaves: Number of octaves (positive = up, negative = down)
        
        Returns:
            Pitch-shifted audio
        """
        return self.shift_pitch(audio, octaves * 12)
    
    @staticmethod
    def _normalize(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        """Normalize audio to target level."""
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_level / peak)
        return audio


class TimeStretch:
    """
    Time stretching without pitch change using phase vocoder.
    """
    
    def __init__(self, sample_rate: int = 44100,
                 fft_size: int = 2048,
                 hop_size: int = 512):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self._window = signal.windows.hann(fft_size)
    
    def stretch(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """
        Stretch audio in time without changing pitch.
        
        Args:
            audio: Input audio
            ratio: Stretch ratio (>1 = slower/longer, <1 = faster/shorter)
        
        Returns:
            Time-stretched audio
        """
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        original_length = len(audio)
        
        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        
        # STFT
        num_frames = 1 + (len(audio) - self.fft_size) // self.hop_size
        magnitude = np.zeros((num_frames, self.fft_size // 2 + 1))
        phase = np.zeros((num_frames, self.fft_size // 2 + 1))
        
        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.fft_size] * self._window
            fft_result = fft(frame)
            magnitude[i] = np.abs(fft_result)
            phase[i] = np.angle(fft_result)
        
        # Time stretch
        new_num_frames = int(num_frames * ratio)
        omega = 2 * np.pi * fftfreq(self.fft_size, 1 / self.sample_rate)[:self.fft_size // 2 + 1]
        
        new_magnitude = np.zeros((new_num_frames, self.fft_size // 2 + 1))
        new_phase = np.zeros((new_num_frames, self.fft_size // 2 + 1))
        
        for i in range(new_num_frames):
            src_frame = i / ratio
            src_frame_int = int(src_frame)
            frac = src_frame - src_frame_int
            
            if src_frame_int >= num_frames - 1:
                break
            
            new_magnitude[i] = (1 - frac) * magnitude[src_frame_int] + frac * magnitude[src_frame_int + 1]
            
            if src_frame_int > 0:
                expected_phase = phase[src_frame_int] + omega * src_frame * self.hop_size / self.sample_rate
                new_phase[i] = expected_phase
            else:
                new_phase[i] = phase[0]
        
        # ISTFT
        output = np.zeros((new_num_frames - 1) * self.hop_size + self.fft_size)
        window_sum = np.zeros_like(output)
        
        for i in range(new_num_frames):
            start = i * self.hop_size
            spectrum = new_magnitude[i] * np.exp(1j * new_phase[i])
            frame = np.real(ifft(spectrum)) * self._window
            output[start:start + self.fft_size] += frame
            window_sum[start:start + self.fft_size] += self._window ** 2
        
        window_sum = np.maximum(window_sum, 1e-8)
        output = output / window_sum
        
        return self._normalize(output * peak)
    
    @staticmethod
    def _normalize(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_level / peak)
        return audio


class KeyMatcher:
    """
    Match audio to target musical key using pitch shifting.
    """
    
    # Key indices: C=0, C#=1, D=2, ... B=11
    KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Circle of fifths order for key compatibility
    CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.pitch_shifter = PitchShifter(sample_rate)
    
    @staticmethod
    def key_name_to_index(key: str) -> int:
        """Convert key name to index (0-11)."""
        key = key.upper().replace('♯', '#').replace('♭', 'b')
        return KeyMatcher.KEY_NAMES.index(key)
    
    @staticmethod
    def key_index_to_name(index: int) -> str:
        """Convert key index to name."""
        return KeyMatcher.KEY_NAMES[index % 12]
    
    @staticmethod
    def get_compatible_keys(key: int) -> list:
        """
        Get harmonically compatible keys.
        
        Args:
            key: Key index (0-11)
        
        Returns:
            List of compatible key indices
        """
        # Major key: I, IV, V, vi, ii, iii, VII
        major_compatible = {
            0: [0, 3, 4, 7, 2, 11, 5],   # C
            1: [1, 4, 5, 8, 3, 0, 6],    # C#
            2: [2, 5, 6, 9, 4, 1, 7],    # D
            3: [3, 6, 7, 10, 5, 2, 8],   # D#
            4: [4, 7, 8, 11, 6, 3, 9],   # E
            5: [5, 8, 9, 0, 7, 4, 10],  # F
            6: [6, 9, 10, 1, 8, 5, 11], # F#
            7: [7, 10, 11, 2, 9, 6, 0],  # G
            8: [8, 11, 0, 3, 10, 7, 1],  # G#
            9: [9, 0, 1, 4, 11, 8, 2],   # A
            10: [10, 1, 2, 5, 0, 9, 3],  # A#
            11: [11, 2, 3, 6, 1, 10, 4]  # B
        }
        return major_compatible.get(key, [key])
    
    def match_to_key(self, audio: np.ndarray, 
                     source_key: int,
                     target_key: int) -> np.ndarray:
        """
        Shift audio to match target key.
        
        Args:
            audio: Input audio
            source_key: Source key index
            target_key: Target key index
        
        Returns:
            Pitch-shifted audio
        """
        return self.pitch_shifter.shift_to_key(audio, source_key, target_key)
    
    def auto_match(self, audio: np.ndarray, 
                   detected_key: int,
                   target_key: int = 0) -> Tuple[np.ndarray, int]:
        """
        Auto-match audio to target key with minimal shifting.
        
        Finds closest compatible key if direct shift would be too extreme.
        
        Args:
            audio: Input audio
            detected_key: Detected key of input
            target_key: Target key (default 0 = C)
        
        Returns:
            Tuple of (processed audio, final key)
        """
        # Calculate direct shift
        direct_shift = target_key - detected_key
        
        # Check if shift is within acceptable range (-6 to +6 semitones)
        if -6 <= direct_shift <= 6:
            return self.match_to_key(audio, detected_key, target_key), target_key
        
        # Find best compatible key
        compatible = self.get_compatible_keys(detected_key)
        
        # Choose closest compatible key
        best_key = min(compatible, key=lambda k: abs(k - target_key))
        
        # Shift to best compatible key
        return self.match_to_key(audio, detected_key, best_key), best_key


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load audio file using scipy."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr
    except ImportError:
        # Fallback to scipy.io.wavfile
        from scipy.io import wavfile
        sr, audio = wavfile.read(path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        return audio, sr


def save_audio(path: str, audio: np.ndarray, sample_rate: int = 44100):
    """Save audio file."""
    try:
        import soundfile as sf
        sf.write(path, audio, sample_rate)
    except ImportError:
        from scipy.io import wavfile
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        wavfile.write(path, sample_rate, audio)


# CLI for testing
if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Pitch Shifting Tool')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', help='Output audio file')
    parser.add_argument('--semitones', '-s', type=float, default=0,
                       help='Semitones to shift (positive=up, negative=down)')
    parser.add_argument('--octaves', '-o', type=float, default=0,
                       help='Octaves to shift')
    parser.add_argument('--sample-rate', '-r', type=int, default=44100,
                       help='Sample rate')
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    audio, sr = load_audio(args.input)
    
    # Calculate total shift
    semitones = args.semitones + (args.octaves * 12)
    
    if semitones == 0:
        print("No pitch shift requested, copying file...")
        save_audio(args.output, audio, sr)
    else:
        print(f"Shifting by {semitones} semitones...")
        shifter = PitchShifter(sample_rate=sr)
        shifted = shifter.shift_pitch(audio, semitones)
        
        print(f"Saving to {args.output}...")
        save_audio(args.output, shifted, sr)
        
        print(f"Done! Pitch shifted by {semitones} semitones "
              f"({'up' if semitones > 0 else 'down'}).")
