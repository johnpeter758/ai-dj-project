"""
Delay Effects Module for AI DJ Project
Comprehensive collection of delay/echo effects
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.signal import butter, lfilter, filtfilt


class DelayEffect:
    """Delay effect processor with multiple delay types."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._buffer = None
        self._buffer_size = 0
        self._write_pos = 0
    
    def _ensure_buffer(self, size: int):
        """Ensure delay buffer is large enough."""
        if self._buffer is None or self._buffer_size < size:
            self._buffer = np.zeros(size)
            self._buffer_size = size
    
    def _normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target dB."""
        if len(audio) == 0:
            return audio
        peak = np.abs(audio).max()
        if peak > 0:
            target_peak = 10 ** (target_db / 20)
            return audio * (target_peak / peak)
        return audio
    
    # ==================== BASIC DELAY ====================
    
    def simple_delay(self, audio: np.ndarray, delay_ms: float = 250,
                    feedback: float = 0.4, wet_dry: float = 0.5) -> np.ndarray:
        """
        Basic delay effect with feedback.
        
        Parameters:
            audio: Input audio signal
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount (0-0.95)
            wet_dry: Wet/dry mix (0=dry, 1=wet)
        
        Returns:
            Delayed audio signal
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        self._ensure_buffer(delay_samples + len(audio))
        
        output = audio.copy()
        
        for i in range(delay_samples, len(audio)):
            output[i] += output[i - delay_samples] * feedback
        
        return self._normalize(audio * (1 - wet_dry) + output * wet_dry)
    
    def multi_tap_delay(self, audio: np.ndarray, 
                       taps: List[Tuple[float, float]] = None,
                       wet_dry: float = 0.5) -> np.ndarray:
        """
        Multi-tap delay - multiple delay times mixed together.
        
        Parameters:
            audio: Input audio signal
            taps: List of (delay_ms, gain) tuples
            wet_dry: Wet/dry mix
        
        Returns:
            Delayed audio with multiple taps
        """
        if taps is None:
            taps = [
                (125, 0.6),   # Early reflection
                (250, 0.5),   # First repeat
                (375, 0.4),   # Second repeat
                (500, 0.3),   # Third repeat
            ]
        
        output = audio.copy()
        
        for delay_ms, gain in taps:
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            if len(audio) > delay_samples:
                output[delay_samples:] += audio[:-delay_samples] * gain
        
        return self._normalize(audio * (1 - wet_dry) + output * wet_dry)
    
    # ==================== SPECIALIZED DELAYS ====================
    
    def tape_delay(self, audio: np.ndarray, delay_ms: float = 150,
                   feedback: float = 0.5, wow_flutter: float = 0.002) -> np.ndarray:
        """
        Tape delay with wow and flutter modulation for analog character.
        
        Parameters:
            audio: Input audio signal
            delay_ms: Base delay time in milliseconds
            feedback: Feedback amount
            wow_flutter: Amount of pitch modulation
        
        Returns:
            Tape-style delayed audio
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        self._ensure_buffer(delay_samples + len(audio))
        
        output = audio.copy()
        
        # Generate modulation signal (wow + flutter)
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        # Wow: slow modulation (0.5 Hz)
        # Flutter: faster modulation (5 Hz)
        mod = 1 + wow_flutter * (np.sin(2 * np.pi * 0.5 * t) + 
                                  0.3 * np.sin(2 * np.pi * 5 * t))
        
        # Apply modulated delay
        for i in range(len(audio)):
            idx = i - int(delay_samples * mod[i])
            if idx >= 0 and idx < len(audio):
                output[i] += audio[idx] * feedback
        
        return self._normalize(output)
    
    def ping_pong_delay(self, audio: np.ndarray, delay_ms: float = 200,
                        feedback: float = 0.35, wet_dry: float = 0.5) -> np.ndarray:
        """
        Ping-pong delay - alternates between left and right channels.
        
        Parameters:
            audio: Input audio (mono or stereo)
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount
            wet_dry: Wet/dry mix
        
        Returns:
            Stereo delayed audio (2D array)
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        # Create stereo output
        if audio.ndim == 1:
            stereo = np.zeros((2, len(audio)))
            stereo[0] = audio
            stereo[1] = audio
        else:
            stereo = audio.copy()
            if stereo.shape[0] != 2:
                stereo = stereo[:2, :]  # Take first 2 channels
        
        # Apply ping-pong feedback
        for i in range(delay_samples, stereo.shape[1]):
            # Left receives from right, right receives from left
            stereo[0, i] += stereo[1, i - delay_samples] * feedback
            stereo[1, i] += stereo[0, i - delay_samples] * feedback
        
        # Mix wet/dry
        if audio.ndim == 1:
            dry_mixed = np.array([audio, audio])
        else:
            dry_mixed = audio[:2, :] if audio.shape[0] >= 2 else np.vstack([audio, audio])
        
        stereo = dry_mixed * (1 - wet_dry) + stereo * wet_dry
        return self._normalize(stereo)
    
    def filter_delay(self, audio: np.ndarray, delay_ms: float = 300,
                     feedback: float = 0.5, cutoff_hz: float = 2000,
                     wet_dry: float = 0.5) -> np.ndarray:
        """
        Filtered delay - delay with lowpass filter on feedback loop.
        
        Parameters:
            audio: Input audio signal
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount
            cutoff_hz: Lowpass filter cutoff frequency
            wet_dry: Wet/dry mix
        
        Returns:
            Filtered delayed audio
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        self._ensure_buffer(delay_samples + len(audio))
        
        output = audio.copy()
        
        # Design lowpass filter
        b, a = butter(2, cutoff_hz / (self.sample_rate / 2), 'low')
        
        # Process with filtered feedback
        feedback_signal = np.zeros_like(audio)
        for i in range(delay_samples, len(audio)):
            feedback_signal[i] = output[i - delay_samples]
        
        # Apply filter to feedback
        feedback_filtered = lfilter(b, a, feedback_signal)
        
        for i in range(delay_samples, len(audio)):
            output[i] += feedback_filtered[i - delay_samples] * feedback
        
        return self._normalize(audio * (1 - wet_dry) + output * wet_dry)
    
    def sync_delay(self, audio: np.ndarray, bpm: float = 120,
                   division: str = "eighth", feedback: float = 0.4,
                   wet_dry: float = 0.5) -> np.ndarray:
        """
        Sync delay - delay time synced to musical tempo.
        
        Parameters:
            audio: Input audio signal
            bpm: Beats per minute
            division: Note division ("quarter", "eighth", "sixteenth", "triplet")
            feedback: Feedback amount
            wet_dry: Wet/dry mix
        
        Returns:
            Tempo-synced delayed audio
        """
        # Calculate delay based on BPM and division
        beat_duration = 60.0 / bpm
        
        divisions = {
            "quarter": 1.0,
            "eighth": 0.5,
            "sixteenth": 0.25,
            "triplet": 1/3,
            "dotted_eighth": 0.75,
            "dotted_sixteenth": 0.375,
        }
        
        delay_beats = divisions.get(division, 0.5)
        delay_ms = (beat_duration * delay_beats) * 1000
        
        return self.simple_delay(audio, delay_ms, feedback, wet_dry)
    
    def reverse_delay(self, audio: np.ndarray, delay_ms: float = 200,
                      feedback: float = 0.3, wet_dry: float = 0.5) -> np.ndarray:
        """
        Reverse delay - plays delayed signal in reverse for ethereal effect.
        
        Parameters:
            audio: Input audio signal
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount
            wet_dry: Wet/dry mix
        
        Returns:
            Reverse delay effect
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        output = audio.copy()
        
        for i in range(delay_samples, len(audio)):
            # Create reversed grain from delay buffer
            start_idx = i - delay_samples
            end_idx = min(i, len(audio))
            if start_idx < end_idx:
                grain = audio[start_idx:end_idx][::-1]
                output[i - len(grain):i] += grain * feedback
        
        return self._normalize(audio * (1 - wet_dry) + output * wet_dry)
    
    # ==================== ADVANCED EFFECTS ====================
    
    def granular_delay(self, audio: np.ndarray, grain_size_ms: float = 50,
                       pitch_shift: float = 1.0, density: float = 0.5,
                       wet_dry: float = 0.5) -> np.ndarray:
        """
        Granular delay - creates clouds of delayed grains.
        
        Parameters:
            audio: Input audio signal
            grain_size_ms: Size of each grain in milliseconds
            pitch_shift: Pitch shift factor for grains
            density: Density of grains (0-1)
            wet_dry: Wet/dry mix
        
        Returns:
            Granular delayed audio
        """
        grain_size = int(grain_size_ms * self.sample_rate / 1000)
        output = audio.copy()
        
        # Random grain positions based on density
        num_grains = int(len(audio) / grain_size * density)
        
        for _ in range(num_grains):
            # Random start position
            start = np.random.randint(0, max(1, len(audio) - grain_size))
            grain = audio[start:start + grain_size].copy()
            
            # Apply pitch shift via simple resampling
            if pitch_shift != 1.0:
                indices = np.linspace(0, len(grain) - 1, int(len(grain) / pitch_shift))
                grain = np.interp(np.arange(len(grain)), indices, grain)
            
            # Random placement in output
            place_pos = np.random.randint(0, max(1, len(output) - len(grain)))
            output[place_pos:place_pos + len(grain)] += grain * 0.3
        
        return self._normalize(audio * (1 - wet_dry) + output * wet_dry)
    
    def chorus_delay(self, audio: np.ndarray, delay_ms: float = 30,
                     depth: float = 0.5, rate_hz: float = 1.5,
                     wet_dry: float = 0.5) -> np.ndarray:
        """
        Chorus effect using short modulated delays.
        
        Parameters:
            audio: Input audio signal
            delay_ms: Base delay time
            depth: Modulation depth
            rate_hz: Modulation rate in Hz
            wet_dry: Wet/dry mix
        
        Returns:
            Chorus effect audio
        """
        base_delay = int(delay_ms * self.sample_rate / 1000)
        output = audio.copy()
        
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        # Create modulated delay
        mod = depth * delay_ms * np.sin(2 * np.pi * rate_hz * t)
        
        for i in range(len(audio)):
            delayed_idx = i - base_delay - int(mod[i] * self.sample_rate / 1000)
            if 0 <= delayed_idx < len(audio):
                output[i] += audio[delayed_idx]
        
        return self._normalize(audio * (1 - wet_dry) + output * wet_dry)
    
    def slap_back_delay(self, audio: np.ndarray, delay_ms: float = 100,
                        feedback: float = 0.2, wet_dry: float = 0.4) -> np.ndarray:
        """
        Slapback delay - short, single-repeat delay (rockabilly style).
        
        Parameters:
            audio: Input audio signal
            delay_ms: Delay time (typically 70-150ms)
            feedback: Feedback amount (low for slapback)
            wet_dry: Wet/dry mix
        
        Returns:
            Slapback delayed audio
        """
        return self.simple_delay(audio, delay_ms, feedback, wet_dry)
    
    # ==================== STEREO DELAYS ====================
    
    def stereo_widened_delay(self, audio: np.ndarray, 
                              delay_l_ms: float = 200,
                              delay_r_ms: float = 233,
                              feedback: float = 0.35,
                              wet_dry: float = 0.5) -> np.ndarray:
        """
        Stereo delay with different left/right times for width.
        
        Parameters:
            audio: Input audio (mono)
            delay_l_ms: Left channel delay
            delay_r_ms: Right channel delay
            feedback: Feedback amount
            wet_dry: Wet/dry mix
        
        Returns:
            Stereo delayed audio
        """
        delay_l = int(delay_l_ms * self.sample_rate / 1000)
        delay_r = int(delay_r_ms * self.sample_rate / 1000)
        
        stereo = np.zeros((2, len(audio)))
        stereo[0] = audio
        stereo[1] = audio
        
        # Process each channel with its own delay
        for i in range(delay_l, len(audio)):
            stereo[0, i] += stereo[0, i - delay_l] * feedback
        
        for i in range(delay_r, len(audio)):
            stereo[1, i] += stereo[1, i - delay_r] * feedback
        
        # Cross-feedback for more stereo width
        for i in range(max(delay_l, delay_r), len(audio)):
            stereo[0, i] += stereo[1, i - delay_r] * feedback * 0.3
            stereo[1, i] += stereo[0, i - delay_l] * feedback * 0.3
        
        return self._normalize(stereo * wet_dry + np.vstack([audio, audio]) * (1 - wet_dry))
    
    # ==================== UTILITY ====================
    
    def dub_delay(self, audio: np.ndarray, initial_delay_ms: float = 400,
                   feedback: float = 0.7, decay: float = 0.8,
                   wet_dry: float = 0.6) -> np.ndarray:
        """
        Dub delay - heavy feedback with filtering (dub reggae style).
        
        Parameters:
            audio: Input audio signal
            initial_delay_ms: Initial delay time
            feedback: High feedback for regeneration
            decay: Additional decay per repeat
            wet_dry: Wet/dry mix
        
        Returns:
            Dub-style delayed audio
        """
        delay_samples = int(initial_delay_ms * self.sample_rate / 1000)
        self._ensure_buffer(delay_samples + len(audio))
        
        output = audio.copy()
        
        # Multiple feedback passes with decay
        current_feedback = feedback
        for _ in range(3):  # 3 passes of regeneration
            for i in range(delay_samples, len(audio)):
                output[i] += output[i - delay_samples] * current_feedback
            current_feedback *= decay
        
        # Add highpass filter to thin repeats (dub characteristic)
        b, a = butter(2, 400 / (self.sample_rate / 2), 'high')
        wet = lfilter(b, a, output)
        
        return self._normalize(audio * (1 - wet_dry) + wet * wet_dry)


def create_delay_processor(sample_rate: int = 44100) -> DelayEffect:
    """Factory function to create a delay processor."""
    return DelayEffect(sample_rate)
