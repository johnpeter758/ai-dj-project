"""
Tremolo Effect Module for AI DJ Project
Advanced tremolo effects with LFO modulation, stereowidening, and various waveforms
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.signal import butter, lfilter, filtfilt


class TremoloEffect:
    """Tremolo effect processor with multiple modulation types."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._lfo_phase = 0.0
    
    def _generate_lfo(self, length: int, rate_hz: float, depth: float,
                      shape: str = 'sine', phase: float = 0.0) -> np.ndarray:
        """
        Generate LFO modulation signal.
        
        Args:
            length: Number of samples
            rate_hz: LFO frequency in Hz
            depth: Modulation depth (0-1)
            shape: LFO waveform ('sine', 'triangle', 'square', 'sawtooth', 'squared_sine')
            phase: Starting phase offset
            
        Returns:
            LFO modulation array
        """
        t = np.arange(length) / self.sample_rate
        phase_offset = 2 * np.pi * rate_hz * t + phase
        
        if shape == 'sine':
            lfo = np.sin(phase_offset)
        elif shape == 'triangle':
            lfo = 2 * np.abs(2 * ((phase_offset / (2 * np.pi)) % 1) - 1) - 1
        elif shape == 'square':
            lfo = np.sign(np.sin(phase_offset))
        elif shape == 'sawtooth':
            lfo = 2 * ((phase_offset / (2 * np.pi)) % 1) - 1
        elif shape == 'squared_sine':
            # Sine squared for smoother tremolo (less harsh)
            lfo = np.sin(phase_offset) ** 2
        else:
            lfo = np.sin(phase_offset)
        
        # Normalize to 0-1 range based on depth
        lfo = (lfo + 1) / 2  # Now 0 to 1
        lfo = 1 - (lfo * depth)  # Invert so depth=1 gives full modulation
        
        return lfo
    
    def process(self, audio: np.ndarray, rate_hz: float = 5.0,
                depth: float = 0.5, shape: str = 'sine', mix: float = 1.0) -> np.ndarray:
        """
        Apply tremolo effect to audio.
        
        Args:
            audio: Input audio (mono or stereo)
            rate_hz: LFO rate in Hz
            depth: Modulation depth (0=no modulation, 1=full)
            shape: LFO waveform ('sine', 'triangle', 'square', 'sawtooth', 'squared_sine')
            mix: Wet/dry mix (0=dry, 1=wet)
            
        Returns:
            Tremolo-processed audio
        """
        is_stereo = audio.ndim == 2
        length = audio.shape[1] if is_stereo else len(audio)
        
        # Generate LFO
        lfo = self._generate_lfo(length, rate_hz, depth, shape)
        
        if is_stereo:
            # Stereo processing with different phases
            lfo_left = self._generate_lfo(length, rate_hz, depth, shape, phase=0)
            lfo_right = self._generate_lfo(length, rate_hz, depth, shape, phase=np.pi/2)
            
            wet_left = audio[0] * lfo_left
            wet_right = audio[1] * lfo_right
            wet = np.array([wet_left, wet_right])
        else:
            wet = audio * lfo
        
        # Mix wet and dry
        output = audio * (1 - mix) + wet * mix
        
        return output
    
    # ==================== PULSE TREMOLO ====================
    
    def pulse_tremolo(self, audio: np.ndarray, rate_hz: float = 8.0,
                      depth: float = 0.7, duty_cycle: float = 0.5) -> np.ndarray:
        """
        Pulse/square wave tremolo for vintage sound.
        
        Args:
            audio: Input audio
            rate_hz: Pulse rate in Hz
            depth: Modulation depth (0-1)
            duty_cycle: On/off ratio (0-1)
            
        Returns:
            Pulse tremolo audio
        """
        length = len(audio)
        t = np.arange(length) / self.sample_rate
        phase = 2 * np.pi * rate_hz * t
        
        # Create pulse wave
        pulse = (np.sin(phase) >= np.sin(2 * np.pi * duty_cycle)).astype(float)
        pulse = 1 - (pulse * depth)  # Apply depth
        
        if audio.ndim == 2:
            return audio * pulse[np.newaxis, :]
        return audio * pulse
    
    # ==================== STEREO TREMOLO ====================
    
    def stereo_tremolo(self, audio: np.ndarray, rate_hz: float = 5.0,
                       depth: float = 0.5, phase_offset: float = np.pi/2) -> np.ndarray:
        """
        Stereo tremolo with phase difference between channels.
        
        Args:
            audio: Input stereo audio
            rate_hz: LFO rate in Hz
            depth: Modulation depth (0-1)
            phase_offset: Phase difference between L/R (radians)
            
        Returns:
            Stereo tremolo audio
        """
        if audio.ndim != 2:
            audio = np.array([audio, audio])
        
        length = audio.shape[1]
        
        # Generate phase-shifted LFOs
        lfo_left = self._generate_lfo(length, rate_hz, depth, 'sine', phase=0)
        lfo_right = self._generate_lfo(length, rate_hz, depth, 'sine', phase=phase_offset)
        
        # Apply to each channel
        left_tremolo = audio[0] * lfo_left
        right_tremolo = audio[1] * lfo_right
        
        return np.array([left_tremolo, right_tremolo])
    
    # ==================== VIBRATO (PITCH TREMOLO) ====================
    
    def vibrato(self, audio: np.ndarray, rate_hz: float = 5.0,
                depth_ms: float = 5.0) -> np.ndarray:
        """
        Vibrato effect (pitch modulation, not volume).
        Note: This is a simplified version using delay modulation.
        
        Args:
            audio: Input audio
            rate_hz: Vibrato rate in Hz
            depth_ms: Vibrato depth in milliseconds
            
        Returns:
            Vibrato audio
        """
        is_stereo = audio.ndim == 2
        length = audio.shape[1] if is_stereo else len(audio)
        
        # Create delay-modulated output
        max_delay_samples = int(depth_ms * self.sample_rate / 1000)
        
        t = np.arange(length) / self.sample_rate
        phase = 2 * np.pi * rate_hz * t
        
        # Delay modulation
        delay_mod = np.sin(phase) * max_delay_samples
        
        if is_stereo:
            output = np.zeros_like(audio)
            for ch in range(2):
                for i in range(length):
                    delay = int(delay_mod[i])
                    if delay > 0 and i >= delay:
                        output[ch, i] = audio[ch, i - delay]
                    else:
                        output[ch, i] = audio[ch, i]
            return output
        else:
            output = np.zeros(length)
            for i in range(length):
                delay = int(delay_mod[i])
                if delay > 0 and i >= delay:
                    output[i] = audio[i - delay]
                else:
                    output[i] = audio[i]
            return output
    
    # ==================== RING MODULATOR ====================
    
    def ring_modulator(self, audio: np.ndarray, carrier_hz: float = 440.0) -> np.ndarray:
        """
        Ring modulator - multiplies audio by a carrier signal.
        
        Args:
            audio: Input audio
            carrier_hz: Carrier frequency in Hz
            
        Returns:
            Ring modulated audio
        """
        length = len(audio)
        t = np.arange(length) / self.sample_rate
        carrier = np.sin(2 * np.pi * carrier_hz * t)
        
        if audio.ndim == 2:
            return audio * carrier[np.newaxis, :]
        return audio * carrier
    
    # ==================== AMP SIMULATION ====================
    
    def amp_tremolo(self, audio: np.ndarray, amp_model: str = 'twin_reverb') -> np.ndarray:
        """
        Classic amplifier tremolo simulation.
        
        Args:
            audio: Input audio
            amp_model: Amplifier model ('twin_reverb', 'deluxe_reverb', 'bassman')
            
        Returns:
            Amplifier tremolo audio
        """
        models = {
            'twin_reverb': {'rate': 6.5, 'depth': 0.4, 'shape': 'sine'},
            'deluxe_reverb': {'rate': 4.0, 'depth': 0.5, 'shape': 'triangle'},
            'bassman': {'rate': 3.5, 'depth': 0.35, 'shape': 'squared_sine'}
        }
        
        params = models.get(amp_model, models['twin_reverb'])
        return self.process(audio, **params)
    
    # ==================== SWELL TREMOLO ====================
    
    def swell(self, audio: np.ndarray, attack_ms: float = 100.0,
              release_ms: float = 500.0) -> np.ndarray:
        """
        Volume swell effect - gradual fade in.
        
        Args:
            audio: Input audio
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            
        Returns:
            Swelled audio
        """
        length = len(audio)
        envelope = np.zeros(length)
        
        attack_samples = int(attack_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)
        
        # Attack portion
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Sustain portion
        if length > attack_samples + release_samples:
            envelope[attack_samples:-release_samples] = 1
        
        # Release portion
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        if audio.ndim == 2:
            return audio * envelope[np.newaxis, :]
        return audio * envelope
    
    # ==================== GATED TREMOLO ====================
    
    def gated(self, audio: np.ndarray, rate_hz: float = 2.0,
              gate_level: float = 0.3) -> np.ndarray:
        """
        Rhythmic gating effect.
        
        Args:
            audio: Input audio
            rate_hz: Gate rate in Hz
            gate_level: Minimum volume when gate is closed
            
        Returns:
            Gated tremolo audio
        """
        return self.process(audio, rate_hz, depth=1-gate_level, shape='square')


class MultiTremolo:
    """Multi-voice tremolo with parallel processors."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.tremolos: List[TremoloEffect] = []
    
    def add_voice(self, rate_hz: float = 5.0, depth: float = 0.5,
                  shape: str = 'sine', phase: float = 0.0) -> 'MultiTremolo':
        """Add a tremolo voice."""
        tremolo = TremoloEffect(self.sample_rate)
        self.tremolos.append({
            'processor': tremolo,
            'rate': rate_hz,
            'depth': depth,
            'shape': shape,
            'phase': phase
        })
        return self
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process through all tremolo voices."""
        if not self.tremolos:
            return audio
        
        output = audio.copy()
        for voice in self.tremolos:
            tremolo = voice['processor']
            wet = tremolo.process(
                audio,
                rate_hz=voice['rate'],
                depth=voice['depth'],
                shape=voice['shape']
            )
            output += wet
        
        return output / (len(self.tremolos) + 1)
    
    @classmethod
    def chord_tremolo(cls, sample_rate: int = 44100) -> 'MultiTremolo':
        """Create multi-voice chord tremolo."""
        mt = cls(sample_rate)
        # Add multiple voices with harmonic rates
        mt.add_voice(rate_hz=5.0, depth=0.5, shape='sine', phase=0)
        mt.add_voice(rate_hz=7.5, depth=0.3, shape='sine', phase=np.pi/4)
        mt.add_voice(rate_hz=10.0, depth=0.2, shape='squared_sine', phase=np.pi/2)
        return mt
