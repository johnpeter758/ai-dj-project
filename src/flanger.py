"""
Flanger Effect Module for AI DJ Project
Classic and advanced flanging effects with feedback, modulation, and stereo widening
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.signal import butter, lfilter, filtfilt


class FlangerEffect:
    """Flanger effect processor with multiple flanging modes."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._buffer = None
        self._buffer_size = 0
        self._write_pos = 0
        self._lfo_phase = 0.0
    
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
    
    def _generate_lfo(self, rate_hz: float, depth_ms: float, 
                      shape: str = 'sine', samples: int = 1) -> np.ndarray:
        """
        Generate LFO modulation signal.
        
        Args:
            rate_hz: LFO rate in Hz
            depth_ms: Depth in milliseconds
            shape: Waveform shape ('sine', 'triangle', 'sawtooth', 'square')
            samples: Number of samples to generate
            
        Returns:
            LFO modulation values
        """
        depth_samples = depth_ms * self.sample_rate / 1000
        
        if shape == 'sine':
            phase = np.linspace(0, 2 * np.pi * rate_hz * samples / self.sample_rate, samples)
            lfo = np.sin(phase + self._lfo_phase) * depth_samples
        elif shape == 'triangle':
            phase = (np.linspace(0, 2 * np.pi * rate_hz * samples / self.sample_rate, samples) + self._lfo_phase) % (2 * np.pi)
            lfo = (2 * np.abs(2 * phase / np.pi - 1) - 1) * depth_samples
        elif shape == 'sawtooth':
            phase = (np.linspace(0, rate_hz * samples / self.sample_rate, samples) + self._lfo_phase / (2 * np.pi)) % 1
            lfo = (2 * phase - 1) * depth_samples
        elif shape == 'square':
            phase = (np.linspace(0, rate_hz * samples / self.sample_rate, samples) + self._lfo_phase / (2 * np.pi)) % 1
            lfo = (np.where(phase < 0.5, 1, -1)) * depth_samples
        else:
            lfo = np.zeros(samples)
        
        self._lfo_phase += 2 * np.pi * rate_hz * samples / self.sample_rate
        self._lfo_phase %= 2 * np.pi
        
        return lfo
    
    # ==================== BASIC FLANGER ====================
    
    def simple_flanger(self, audio: np.ndarray, 
                       delay_ms: float = 5.0,
                       rate_hz: float = 0.5,
                       depth_ms: float = 3.0,
                       feedback: float = 0.5,
                       wet_dry: float = 0.5) -> np.ndarray:
        """
        Classic flanger effect with modulated delay.
        
        Parameters:
            audio: Input audio signal (mono)
            delay_ms: Base delay in milliseconds (0-20)
            rate_hz: LFO rate in Hz
            depth_ms: LFO depth in milliseconds
            feedback: Feedback amount (0-0.95)
            wet_dry: Wet/dry mix (0=dry, 1=wet)
        
        Returns:
            Flanged audio signal
        """
        max_delay_samples = int((delay_ms + depth_ms) * self.sample_rate / 1000) + 100
        self._ensure_buffer(max_delay_samples + len(audio))
        
        output = audio.copy().astype(np.float64)
        
        base_delay = int(delay_ms * self.sample_rate / 1000)
        
        for i in range(len(audio)):
            # Generate LFO value for this sample
            lfo_val = self._generate_lfo(rate_hz, depth_ms, 'sine', 1)[0]
            delay = int(base_delay + lfo_val)
            delay = max(1, min(delay, self._buffer_size - 1))
            
            # Read from delay buffer
            read_pos = self._write_pos - delay
            if read_pos < 0:
                read_pos += self._buffer_size
            
            delayed = self._buffer[read_pos]
            
            # Write to buffer with feedback
            self._buffer[self._write_pos] = audio[i] + delayed * feedback
            
            # Mix wet and dry
            output[i] = audio[i] * (1 - wet_dry) + delayed * wet_dry
            
            # Advance write position
            self._write_pos = (self._write_pos + 1) % self._buffer_size
        
        return self._normalize(output.astype(np.float32))
    
    def process(self, audio: np.ndarray,
                delay_ms: float = 5.0,
                rate_hz: float = 0.5,
                depth_ms: float = 3.0,
                feedback: float = 0.5,
                wet_dry: float = 0.5,
                shape: str = 'sine') -> np.ndarray:
        """
        Process audio through flanger effect.
        
        Args:
            audio: Input audio (mono or stereo)
            delay_ms: Base delay in milliseconds
            rate_hz: LFO rate in Hz
            depth_ms: LFO depth in milliseconds
            feedback: Feedback amount (0-0.95)
            wet_dry: Wet/dry mix (0-1)
            shape: LFO waveform shape
        
        Returns:
            Flanged audio
        """
        is_stereo = audio.ndim == 2
        
        if is_stereo:
            # Stereo flanging with different LFO phases
            left = self._process_channel(
                audio[0], delay_ms, rate_hz, depth_ms, feedback, wet_dry, shape, phase_offset=0
            )
            right = self._process_channel(
                audio[1], delay_ms, rate_hz, depth_ms, feedback, wet_dry, shape, phase_offset=np.pi/4
            )
            return np.array([left, right])
        else:
            return self._process_channel(
                audio, delay_ms, rate_hz, depth_ms, feedback, wet_dry, shape, phase_offset=0
            )
    
    def _process_channel(self, audio: np.ndarray, delay_ms: float,
                         rate_hz: float, depth_ms: float, feedback: float,
                         wet_dry: float, shape: str, phase_offset: float = 0) -> np.ndarray:
        """Process a single channel with flanging."""
        max_delay_samples = int((delay_ms + depth_ms) * self.sample_rate / 1000) + 100
        self._ensure_buffer(max_delay_samples + len(audio))
        
        output = audio.copy().astype(np.float64)
        base_delay = int(delay_ms * self.sample_rate / 1000)
        
        # Store and restore LFO phase for stereo
        saved_phase = self._lfo_phase
        self._lfo_phase = phase_offset
        
        for i in range(len(audio)):
            lfo_val = self._generate_lfo(rate_hz, depth_ms, shape, 1)[0]
            delay = int(base_delay + lfo_val)
            delay = max(1, min(delay, self._buffer_size - 1))
            
            read_pos = self._write_pos - delay
            if read_pos < 0:
                read_pos += self._buffer_size
            
            delayed = self._buffer[read_pos]
            self._buffer[self._write_pos] = audio[i] + delayed * feedback
            output[i] = audio[i] * (1 - wet_dry) + delayed * wet_dry
            self._write_pos = (self._write_pos + 1) % self._buffer_size
        
        self._lfo_phase = saved_phase
        return self._normalize(output.astype(np.float32))
    
    # ==================== ADVANCED FLANGERS ====================
    
    def negative_feedback_flanger(self, audio: np.ndarray,
                                   delay_ms: float = 3.0,
                                   rate_hz: float = 0.3,
                                   depth_ms: float = 2.5,
                                   feedback: float = -0.5,
                                   wet_dry: float = 0.5) -> np.ndarray:
        """
        Negative feedback flanger - creates a more pronounced sweeping effect.
        
        Parameters:
            audio: Input audio
            delay_ms: Base delay
            rate_hz: LFO rate
            depth_ms: LFO depth
            feedback: Negative feedback (-1 to 0)
            wet_dry: Wet/dry mix
        
        Returns:
            Negative feedback flanged audio
        """
        return self.simple_flanger(
            audio, delay_ms, rate_hz, depth_ms, 
            max(-0.95, min(-0.1, feedback)), wet_dry
        )
    
    def double_flanger(self, audio: np.ndarray,
                       delay_ms: float = 5.0,
                       rate_hz: float = 0.4,
                       depth_ms: float = 3.0,
                       feedback: float = 0.4,
                       wet_dry: float = 0.5,
                       separation_ms: float = 10.0) -> np.ndarray:
        """
        Double flanger - two flangers slightly detuned for thicker sound.
        
        Parameters:
            audio: Input audio
            delay_ms: Base delay
            rate_hz: Primary LFO rate
            depth_ms: LFO depth
            feedback: Feedback amount
            wet_dry: Wet/dry mix
            separation_ms: Delay separation between flangers
        
        Returns:
            Double-flanged audio
        """
        # First flanger
        flanger1 = self.simple_flanger(
            audio, delay_ms, rate_hz, depth_ms, feedback, 1.0
        )
        
        # Second flanger with offset
        flanger2 = self.simple_flanger(
            audio, delay_ms + separation_ms, rate_hz * 1.01, depth_ms * 1.1, feedback, 1.0
        )
        
        # Mix both
        return self._normalize(audio * (1 - wet_dry) + (flanger1 + flanger2) * 0.5 * wet_dry)
    
    def tape_flanger(self, audio: np.ndarray,
                     delay_ms: float = 4.0,
                     rate_hz: float = 0.2,
                     depth_ms: float = 2.0,
                     feedback: float = 0.4,
                     wet_dry: float = 0.5,
                     wow_flutter: float = 0.003) -> np.ndarray:
        """
        Tape-style flanger with subtle wow and flutter modulation.
        
        Parameters:
            audio: Input audio
            delay_ms: Base delay
            rate_hz: Primary LFO rate
            depth_ms: LFO depth
            feedback: Feedback amount
            wet_dry: Wet/dry mix
            wow_flutter: Wow/flutter amount
        
        Returns:
            Tape-flanged audio
        """
        max_delay = int((delay_ms + depth_ms + wow_flutter * 100) * self.sample_rate / 1000) + 100
        self._ensure_buffer(max_delay + len(audio))
        
        output = audio.copy().astype(np.float64)
        base_delay = int(delay_ms * self.sample_rate / 1000)
        
        # Add extra LFO for wow/flutter
        flutter_lfo = np.sin(np.arange(len(audio)) * 2 * np.pi * 8 / self.sample_rate) * wow_flutter * self.sample_rate / 1000
        
        for i in range(len(audio)):
            # Main LFO
            lfo_val = self._generate_lfo(rate_hz, depth_ms, 'sine', 1)[0]
            # Add flutter
            total_delay = int(base_delay + lfo_val + flutter_lfo[i])
            total_delay = max(1, min(total_delay, self._buffer_size - 1))
            
            read_pos = self._write_pos - total_delay
            if read_pos < 0:
                read_pos += self._buffer_size
            
            delayed = self._buffer[read_pos]
            self._buffer[self._write_pos] = audio[i] + delayed * feedback
            output[i] = audio[i] * (1 - wet_dry) + delayed * wet_dry
            self._write_pos = (self._write_pos + 1) % self._buffer_size
        
        return self._normalize(output.astype(np.float32))
    
    def resonant_flanger(self, audio: np.ndarray,
                         delay_ms: float = 5.0,
                         rate_hz: float = 0.5,
                         depth_ms: float = 4.0,
                         feedback: float = 0.7,
                         wet_dry: float = 0.5,
                         resonance: float = 0.5) -> np.ndarray:
        """
        Resonant flanger with emphasis on specific frequencies.
        
        Parameters:
            audio: Input audio
            delay_ms: Base delay
            rate_hz: LFO rate
            depth_ms: LFO depth
            feedback: Feedback amount (higher for more resonance)
            wet_dry: Wet/dry mix
            resonance: Resonance peak intensity (0-1)
        
        Returns:
            Resonant flanged audio
        """
        # Apply pre-emphasis for resonance
        b, a = butter(2, 2000 / (self.sample_rate / 2), 'high')
        if resonance > 0:
            pre_emphasized = lfilter(b, a, audio)
            audio = audio + pre_emphasized * resonance * 0.3
        
        # Process through flanger
        output = self.simple_flanger(audio, delay_ms, rate_hz, depth_ms, feedback, wet_dry)
        
        # Apply post-emphasis
        if resonance > 0:
            output = output + lfilter(b, a, output) * resonance * 0.2
        
        return self._normalize(output)
    
    # ==================== PRESET FLANGERS ====================
    
    def through_zero_flanger(self, audio: np.ndarray,
                            wet_dry: float = 0.5) -> np.ndarray:
        """
        Through-zero flanger - classic through-zero modulation.
        Creates dramatic cancellation at zero-crossing of LFO.
        """
        return self.simple_flanger(
            audio,
            delay_ms=0.5,      # Very short base delay
            rate_hz=0.3,
            depth_ms=4.0,
            feedback=0.6,
            wet_dry=wet_dry
        )
    
    def jet_flanger(self, audio: np.ndarray,
                    wet_dry: float = 0.5) -> np.ndarray:
        """
        Jet flanger - airplane jet sound simulation.
        """
        return self.tape_flanger(
            audio,
            delay_ms=6.0,
            rate_hz=0.15,
            depth_ms=5.0,
            feedback=0.7,
            wet_dry=wet_dry,
            wow_flutter=0.005
        )
    
    def syncopation_flanger(self, audio: np.ndarray,
                           wet_dry: float = 0.5) -> np.ndarray:
        """
        Syncopation flanger - rhythmic, stutter-like effect.
        """
        return self.double_flanger(
            audio,
            delay_ms=3.0,
            rate_hz=4.0,       # Fast rate for rhythmic effect
            depth_ms=1.5,
            feedback=0.3,
            wet_dry=wet_dry,
            separation_ms=5.0
        )
    
    def subtle_shimmer(self, audio: np.ndarray,
                      wet_dry: float = 0.3) -> np.ndarray:
        """
        Subtle shimmer - gentle, airy flanging for mix enhancement.
        """
        return self.simple_flanger(
            audio,
            delay_ms=2.0,
            rate_hz=0.2,
            depth_ms=1.5,
            feedback=0.2,
            wet_dry=wet_dry
        )


class MultiBandFlanger:
    """Multi-band flanger - different flanging for different frequency ranges."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.flanger = FlangerEffect(sample_rate)
        self._crossover_freqs = [200, 800, 3200]
    
    def _create_crossover_filters(self):
        """Create crossover filters for frequency bands."""
        filters = []
        for i, freq in enumerate(self._crossover_freqs):
            if i == 0:
                b, a = butter(4, freq / (self.sample_rate / 2), 'low')
            elif i == len(self._crossover_freqs):
                b, a = butter(4, self._crossover_freqs[i-1] / (self.sample_rate / 2), 'high')
            else:
                b, a = butter(4, [self._crossover_freqs[i-1] / (self.sample_rate / 2),
                                 freq / (self.sample_rate / 2)], 'bandpass')
            filters.append((b, a))
        return filters
    
    def process(self, audio: np.ndarray,
               delay_ms: float = 5.0,
               rate_hz: float = 0.5,
               depth_ms: float = 3.0,
               feedback: float = 0.5,
               wet_dry: float = 0.5,
               band_intensity: List[float] = None) -> np.ndarray:
        """
        Process audio with band-specific flanging.
        
        Args:
            audio: Input audio
            delay_ms: Base delay
            rate_hz: LFO rate
            depth_ms: LFO depth
            feedback: Feedback amount
            wet_dry: Wet/dry mix
            band_intensity: Intensity per band (low, mid, high, very high)
        
        Returns:
            Multi-band flanged audio
        """
        if band_intensity is None:
            band_intensity = [1.0, 0.8, 0.6, 0.4]
        
        filters = self._create_crossover_filters()
        
        # Split into bands
        bands = []
        prev_output = audio
        for i, (b, a) in enumerate(filters):
            if i == 0:
                band = lfilter(b, a, audio)
                bands.append(band)
                prev_output = audio - band
            elif i == len(filters) - 1:
                bands.append(prev_output)
            else:
                band = lfilter(b, a, prev_output)
                bands.append(band)
                prev_output = prev_output - band
        
        # Process each band with different settings
        processed = []
        for i, band in enumerate(bands):
            # Vary flanger settings per band
            band_delay = delay_ms * (1 + i * 0.3)
            band_depth = depth_ms * (1 + i * 0.2)
            band_rate = rate_hz * (1 + i * 0.15)
            
            processed_band = self.flanger.simple_flanger(
                band, band_delay, band_rate, band_depth, feedback, 1.0
            )
            
            # Apply band intensity
            intensity = band_intensity[min(i, len(band_intensity) - 1)]
            processed.append(processed_band * intensity)
        
        # Sum bands and mix
        output = sum(processed)
        
        return audio * (1 - wet_dry) + output * wet_dry


# ==================== CONVENIENCE FUNCTIONS ====================

def flanger(audio: np.ndarray, 
            sample_rate: int = 44100,
            delay_ms: float = 5.0,
            rate_hz: float = 0.5,
            depth_ms: float = 3.0,
            feedback: float = 0.5,
            wet_dry: float = 0.5) -> np.ndarray:
    """
    Convenience function for basic flanging.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        delay_ms: Base delay (0-20ms)
        rate_hz: LFO rate (0.1-10 Hz)
        depth_ms: LFO depth (0-10ms)
        feedback: Feedback (0-0.95)
        wet_dry: Wet/dry mix (0-1)
    
    Returns:
        Flanged audio
    """
    flanger_effect = FlangerEffect(sample_rate)
    return flanger_effect.process(audio, delay_ms, rate_hz, depth_ms, feedback, wet_dry)
