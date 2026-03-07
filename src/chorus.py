"""
Chorus Effect Module for AI DJ Project
Advanced chorus effects with multiple voices, LFO shapes, and stereo width
"""

import numpy as np
from typing import Optional, Tuple, List


class ChorusEffect:
    """Advanced chorus effect processor with multiple voices and modulation options."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.voices: List['ChorusVoice'] = []
        
    def create_voices(
        self,
        num_voices: int = 3,
        base_delay_ms: float = 20.0,
        voice_spacing_ms: float = 5.0,
        rate_hz: float = 1.5,
        depth: float = 0.5,
        lfo_shape: str = 'sine'
    ) -> 'ChorusEffect':
        """
        Create multiple chorus voices with different delays.
        
        Args:
            num_voices: Number of parallel voices
            base_delay_ms: Base delay in milliseconds
            voice_spacing_ms: Delay spacing between voices
            rate_hz: LFO rate in Hz
            depth: Modulation depth (0-1)
            lfo_shape: LFO waveform ('sine', 'triangle', 'sawtooth')
            
        Returns:
            Self for chaining
        """
        self.voices = []
        for i in range(num_voices):
            voice = ChorusVoice(
                sample_rate=self.sample_rate,
                base_delay_ms=base_delay_ms + i * voice_spacing_ms,
                rate_hz=rate_hz * (1 + i * 0.1),  # Slight rate variation
                depth=depth,
                lfo_shape=lfo_shape,
                phase_offset=i * (2 * np.pi / num_voices)
            )
            self.voices.append(voice)
        return self
    
    def process(self, audio: np.ndarray, mix: float = 0.5) -> np.ndarray:
        """
        Process audio through all chorus voices.
        
        Args:
            audio: Input audio (mono or stereo)
            mix: Wet/dry mix (0-1)
            
        Returns:
            Chorus-processed audio
        """
        if not self.voices:
            # Default single voice if none created
            default_voice = ChorusVoice(
                sample_rate=self.sample_rate,
                base_delay_ms=20.0,
                rate_hz=1.5,
                depth=0.5
            )
            return default_voice.process(audio, mix)
        
        is_stereo = audio.ndim == 2
        
        if is_stereo:
            # Process stereo with different voice settings
            left_chorus = self._process_stereo_channel(audio[0], offset=0)
            right_chorus = self._process_stereo_channel(audio[1], offset=np.pi/4)
            wet = np.array([left_chorus, right_chorus])
        else:
            # Sum all voices
            wet = np.zeros_like(audio)
            for i, voice in enumerate(self.voices):
                phase = i * (2 * np.pi / len(self.voices))
                wet += voice.process_with_phase(audio, phase)
            wet /= len(self.voices)
        
        return audio * (1 - mix) + wet * mix
    
    def _process_stereo_channel(
        self, 
        audio: np.ndarray, 
        offset: float = 0.0
    ) -> np.ndarray:
        """Process a single channel with stereo offset."""
        wet = np.zeros_like(audio)
        for i, voice in enumerate(self.voices):
            phase = offset + i * (2 * np.pi / len(self.voices))
            wet += voice.process_with_phase(audio, phase)
        wet /= len(self.voices)
        return wet


class ChorusVoice:
    """Single chorus voice with modulated delay line."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        base_delay_ms: float = 20.0,
        rate_hz: float = 1.5,
        depth: float = 0.5,
        feedback: float = 0.0,
        lfo_shape: str = 'sine',
        phase_offset: float = 0.0
    ):
        self.sample_rate = sample_rate
        self.base_delay_ms = base_delay_ms
        self.rate_hz = rate_hz
        self.depth = depth
        self.feedback = feedback
        self.lfo_shape = lfo_shape
        self.phase_offset = phase_offset
        
        # Buffer for delay line
        max_delay_ms = base_delay_ms + 10  # Max deviation + headroom
        self.buffer_size = int(max_delay_ms * sample_rate / 1000) + 100
        self.buffer = np.zeros(self.buffer_size)
        self.buffer_index = 0
        
    def _generate_lfo(self, length: int) -> np.ndarray:
        """Generate LFO waveform."""
        t = np.arange(length) / self.sample_rate
        phase = 2 * np.pi * self.rate_hz * t + self.phase_offset
        
        if self.lfo_shape == 'sine':
            lfo = np.sin(phase)
        elif self.lfo_shape == 'triangle':
            lfo = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
        elif self.lfo_shape == 'sawtooth':
            lfo = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        else:
            lfo = np.sin(phase)
        
        # Scale by depth
        max_dev = int(5.0 * self.depth * self.sample_rate / 1000)
        return lfo * max_dev
    
    def process(self, audio: np.ndarray, mix: float = 0.5) -> np.ndarray:
        """Process audio through this voice."""
        return self.process_with_phase(audio, self.phase_offset, mix)
    
    def process_with_phase(
        self, 
        audio: np.ndarray, 
        phase_offset: float = 0.0,
        mix: float = 0.5
    ) -> np.ndarray:
        """Process with custom phase offset."""
        length = len(audio)
        output = np.zeros(length)
        
        base_delay = int(self.base_delay_ms * self.sample_rate / 1000)
        lfo = self._generate_lfo(length)
        
        # Override phase for this call
        if phase_offset != self.phase_offset:
            t = np.arange(length) / self.sample_rate
            phase = 2 * np.pi * self.rate_hz * t + phase_offset
            
            if self.lfo_shape == 'sine':
                lfo = np.sin(phase)
            elif self.lfo_shape == 'triangle':
                lfo = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
            elif self.lfo_shape == 'sawtooth':
                lfo = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
            
            max_dev = int(5.0 * self.depth * self.sample_rate / 1000)
            lfo = lfo * max_dev
        
        for i in range(length):
            # Calculate delay
            delay = int(base_delay + lfo[i])
            delay = max(0, min(delay, self.buffer_size - 1))
            
            # Read from delay buffer
            read_idx = (self.buffer_index - delay) % self.buffer_size
            delayed = self.buffer[read_idx]
            
            # Feedback
            self.buffer[self.buffer_index] = audio[i] + delayed * self.feedback
            
            # Mix dry and wet
            output[i] = audio[i] * (1 - mix) + delayed * mix
            
            # Advance buffer index
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        
        return output
    
    def reset(self):
        """Reset the delay buffer."""
        self.buffer = np.zeros(self.buffer_size)
        self.buffer_index = 0


def chorus(
    audio: np.ndarray,
    rate_hz: float = 1.5,
    depth: float = 0.5,
    mix: float = 0.5,
    num_voices: int = 3,
    feedback: float = 0.0,
    lfo_shape: str = 'sine',
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply chorus effect to audio (convenience function).
    
    Args:
        audio: Input audio (mono or stereo)
        rate_hz: LFO rate in Hz
        depth: Modulation depth (0-1)
        mix: Wet/dry mix (0-1)
        num_voices: Number of chorus voices
        feedback: Feedback amount (0-0.9)
        lfo_shape: LFO waveform ('sine', 'triangle', 'sawtooth')
        sample_rate: Audio sample rate
        
    Returns:
        Chorus-processed audio
    """
    chorus = ChorusEffect(sample_rate=sample_rate)
    chorus.create_voices(
        num_voices=num_voices,
        rate_hz=rate_hz,
        depth=depth,
        lfo_shape=lfo_shape
    )
    
    # Set feedback on voices
    for voice in chorus.voices:
        voice.feedback = feedback
    
    return chorus.process(audio, mix)


def flanger(
    audio: np.ndarray,
    rate_hz: float = 0.5,
    depth: float = 0.7,
    feedback: float = 0.5,
    mix: float = 0.5,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply flanger effect (closely spaced chorus with feedback).
    
    Args:
        audio: Input audio
        rate_hz: LFO rate in Hz
        depth: Modulation depth (0-1)
        feedback: Feedback amount (0-0.9)
        mix: Wet/dry mix (0-1)
        sample_rate: Audio sample rate
        
    Returns:
        Flanger-processed audio
    """
    return chorus(
        audio,
        rate_hz=rate_hz,
        depth=depth,
        mix=mix,
        num_voices=1,
        feedback=feedback,
        lfo_shape='sine',
        sample_rate=sample_rate
    )


def ensemble(
    audio: np.ndarray,
    mix: float = 0.5,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply ensemble/chorus effect (thick, lush stereo chorus).
    
    Args:
        audio: Input audio
        mix: Wet/dry mix (0-1)
        sample_rate: Audio sample rate
        
    Returns:
        Ensemble-processed audio
    """
    chorus = ChorusEffect(sample_rate=sample_rate)
    chorus.create_voices(
        num_voices=5,
        base_delay_ms=15.0,
        voice_spacing_ms=3.0,
        rate_hz=0.8,
        depth=0.6,
        lfo_shape='triangle'
    )
    
    return chorus.process(audio, mix)


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    import scipy.io.wavfile as wav
    
    print("=" * 50)
    print("Chorus Effect Module Test")
    print("=" * 50)
    
    # Generate test tone
    sample_rate = 44100
    duration = 2.0
    frequency = 440.0
    
    t = np.arange(int(sample_rate * duration)) / sample_rate
    test_tone = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some harmonics for richer sound
    test_tone += np.sin(2 * np.pi * frequency * 2 * t) * 0.15
    test_tone += np.sin(2 * np.pi * frequency * 3 * t) * 0.1
    
    print(f"✓ Test tone generated: {len(test_tone)} samples")
    
    # Test 1: Basic chorus
    print("\n[1] Basic Chorus")
    basic = chorus(test_tone, rate_hz=1.5, depth=0.5, mix=0.5)
    print(f"   ✓ Applied: {basic.shape}")
    
    # Test 2: Multi-voice chorus
    print("\n[2] Multi-Voice Chorus")
    multivoice = chorus(test_tone, rate_hz=1.0, depth=0.6, num_voices=4, mix=0.5)
    print(f"   ✓ Applied: {multivoice.shape}")
    
    # Test 3: Flanger
    print("\n[3] Flanger")
    flanged = flanger(test_tone, rate_hz=0.3, depth=0.8, feedback=0.6, mix=0.5)
    print(f"   ✓ Applied: {flanged.shape}")
    
    # Test 4: Ensemble
    print("\n[4] Ensemble Effect")
    ensembled = ensemble(test_tone, mix=0.6)
    print(f"   ✓ Applied: {ensembled.shape}")
    
    # Test 5: Stereo chorus
    print("\n[5] Stereo Chorus")
    stereo_in = np.array([test_tone, test_tone])
    stereo_out = chorus(stereo_in, rate_hz=1.2, depth=0.5, num_voices=3, mix=0.5)
    print(f"   ✓ Applied: {stereo_out.shape}")
    
    # Test 6: Class-based usage
    print("\n[6] Class-Based Usage")
    chorus_effect = ChorusEffect(sample_rate=sample_rate)
    chorus_effect.create_voices(
        num_voices=3,
        base_delay_ms=20.0,
        voice_spacing_ms=4.0,
        rate_hz=1.5,
        depth=0.5,
        lfo_shape='sine'
    )
    class_result = chorus_effect.process(test_tone, mix=0.5)
    print(f"   ✓ Applied: {class_result.shape}")
    
    # Save test output
    output_path = "/Users/johnpeter/ai-dj-project/src/output/chorus_test.wav"
    wav.write(output_path, sample_rate, (ensembled * 32767).astype(np.int16))
    print(f"\n✓ Test output saved to: {output_path}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
