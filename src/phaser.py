"""
Phaser Effect Module for AI DJ Project
Advanced phaser effects with multiple stages, LFO modulation, and stereo processing
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class PhaserType(Enum):
    """Phaser algorithm types."""
    ANALOG = "analog"
    DIGITAL = "digital"
    VINTAGE = "vintage"
    MODERN = "modern"
    STEP = "step"


class LFOShape(Enum):
    """LFO waveform shapes."""
    SINE = "sine"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"
    REVERSE_SAW = "reverse_saw"
    SQUARE = "square"
    SAMPLE_HOLD = "sample_hold"


@dataclass
class PhaserParams:
    """Parameters for phaser effect."""
    phaser_type: PhaserType = PhaserType.ANALOG
    stages: int = 4                    # Number of allpass filter stages (2-12)
    rate_hz: float = 0.5               # LFO rate in Hz (0.01 - 20)
    depth: float = 1.0                 # Modulation depth (0-1)
    mix: float = 0.5                  # Wet/dry mix (0-1)
    feedback: float = 0.0             # Feedback amount (-1 to 1)
    frequency_min_hz: float = 200      # Minimum cutoff frequency (20-2000)
    frequency_max_hz: float = 2000     # Maximum cutoff frequency (200-10000)
    stereo_offset: float = 0.0         # Stereo phase offset in radians
    lfo_shape: LFOShape = LFOShape.SINE
    lfo_sync: bool = False             # Sync to tempo
    lfo_quantize: bool = False         # Quantize LFO to note values
    sample_rate: int = 44100


class PhaserStage:
    """
    Single allpass filter stage for phaser effect.
    
    Implements a first-order allpass filter with configurable delay.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        base_delay: float = 0.001,  # Base delay in seconds
        feedback: float = 0.0
    ):
        self.sample_rate = sample_rate
        self.base_delay = base_delay
        self.feedback = feedback
        
        # Buffer for delay line
        max_delay_samples = int(sample_rate * 0.01)  # 10ms max
        self.buffer = np.zeros(max_delay_samples)
        self.buffer_index = 0
        
        # State for allpass filter
        self.z1 = 0.0
    
    def process(
        self, 
        audio: np.ndarray, 
        delay_fraction: float
    ) -> np.ndarray:
        """
        Process audio through allpass filter.
        
        Args:
            audio: Input audio samples
            delay_fraction: Delay as fraction of base delay (0-1)
            
        Returns:
            Filtered audio
        """
        output = np.zeros_like(audio)
        
        if len(audio) == 1:
            # Single sample processing
            return self.process_single(audio, delay_fraction)
        
        delay_samples = int(self.base_delay * self.sample_rate * delay_fraction)
        delay_samples = max(1, min(delay_samples, len(self.buffer) - 1))
        
        for i, sample in enumerate(audio):
            # Read from delay line
            read_index = (self.buffer_index - delay_samples) % len(self.buffer)
            delayed = self.buffer[read_index]
            
            # First-order allpass filter
            # y[n] = x[n] + a * y[n-1]
            # where a = delay_fraction
            alpha = delay_fraction * 2.0 - 1.0
            
            output[i] = delayed * (-alpha) + self.z1
            self.z1 = output[i] * alpha + delayed
            
            # Write to delay line with feedback
            self.buffer[self.buffer_index] = sample + delayed * self.feedback
            
            # Advance buffer index
            self.buffer_index = (self.buffer_index + 1) % len(self.buffer)
        
        return output
    
    def process_single(
        self, 
        audio: np.ndarray, 
        delay_fraction: float
    ) -> np.ndarray:
        """
        Process a single sample through allpass filter.
        
        Args:
            audio: Single input sample (array of 1)
            delay_fraction: Delay as fraction of base delay (0-1)
            
        Returns:
            Filtered single sample
        """
        delay_samples = int(self.base_delay * self.sample_rate * delay_fraction)
        delay_samples = max(1, min(delay_samples, len(self.buffer) - 1))
        
        sample = audio[0]
        
        # Read from delay line
        read_index = (self.buffer_index - delay_samples) % len(self.buffer)
        delayed = self.buffer[read_index]
        
        # First-order allpass filter
        alpha = delay_fraction * 2.0 - 1.0
        
        output = delayed * (-alpha) + self.z1
        self.z1 = output * alpha + delayed
        
        # Write to delay line with feedback
        self.buffer[self.buffer_index] = sample + delayed * self.feedback
        
        # Advance buffer index
        self.buffer_index = (self.buffer_index + 1) % len(self.buffer)
        
        return np.array([output])
    
    def reset(self):
        """Reset filter state."""
        self.buffer.fill(0)
        self.z1 = 0.0
        self.buffer_index = 0


class LFO:
    """
    Low Frequency Oscillator for phaser modulation.
    
    Supports various waveforms and optional sync to tempo.
    """
    
    def __init__(
        self,
        rate_hz: float = 0.5,
        shape: LFOShape = LFOShape.SINE,
        sample_rate: int = 44100
    ):
        self.rate_hz = rate_hz
        self.shape = shape
        self.sample_rate = sample_rate
        
        self.phase = 0.0
        self.phase_increment = rate_hz / sample_rate
    
    def process(self, num_samples: int) -> np.ndarray:
        """
        Generate LFO waveform.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            LFO waveform (normalized to 0-1)
        """
        phases = np.arange(num_samples)
        phases = (phases * self.phase_increment + self.phase) % 1.0
        
        # Generate waveform based on shape
        if self.shape == LFOShape.SINE:
            output = 0.5 + 0.5 * np.sin(2 * np.pi * phases)
        elif self.shape == LFOShape.TRIANGLE:
            output = np.where(
                phases < 0.5,
                2 * phases,
                2 * (1 - phases)
            )
        elif self.shape == LFOShape.SAWTOOTH:
            output = phases
        elif self.shape == LFOShape.REVERSE_SAW:
            output = 1 - phases
        elif self.shape == LFOShape.SQUARE:
            output = np.where(phases < 0.5, 1.0, 0.0)
        elif self.shape == LFOShape.SAMPLE_HOLD:
            # Sample and hold - random values held for each cycle
            output = np.zeros(num_samples)
            hold_value = 0.5
            for i in range(num_samples):
                if phases[i] < self.phase_increment:
                    hold_value = np.random.rand()
                output[i] = hold_value
        else:
            output = 0.5 + 0.5 * np.sin(2 * np.pi * phases)
        
        # Update phase for next call
        self.phase = (self.phase + num_samples * self.phase_increment) % 1.0
        
        return output
    
    def set_rate(self, rate_hz: float):
        """Update LFO rate."""
        self.rate_hz = rate_hz
        self.phase_increment = rate_hz / self.sample_rate
    
    def reset(self):
        """Reset LFO phase."""
        self.phase = 0.0


class PhaserEffect:
    """
    Advanced phaser effect processor with multiple stages and modulation.
    
    Implements:
    - Multi-stage allpass filters
    - Multiple LFO waveforms
    - Feedback path
    - Frequency range control
    - Stereo processing with offset
    - Tempo sync (when integrated with tempo system)
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._init_stages(4)
        self._init_lfo()
        
        # Processing state
        self.params = PhaserParams()
        self.stages_enabled = True
        self.bypass = False
    
    def _init_stages(self, num_stages: int):
        """Initialize phaser stages."""
        self.stages: List[PhaserStage] = []
        for i in range(num_stages):
            stage = PhaserStage(
                sample_rate=self.sample_rate,
                base_delay=0.001 * (1 + i * 0.5),  # Staggered base delays
                feedback=0.0
            )
            self.stages.append(stage)
    
    def _init_lfo(self):
        """Initialize LFO generators."""
        self.lfo = LFO(
            rate_hz=0.5,
            shape=LFOShape.SINE,
            sample_rate=self.sample_rate
        )
        # Secondary LFO for stereo
        self.lfo_right = LFO(
            rate_hz=0.5,
            shape=LFOShape.SINE,
            sample_rate=self.sample_rate
        )
    
    def configure(
        self,
        stages: int = 4,
        rate_hz: float = 0.5,
        depth: float = 1.0,
        feedback: float = 0.0,
        frequency_min_hz: float = 200,
        frequency_max_hz: float = 2000,
        mix: float = 0.5,
        lfo_shape: LFOShape = LFOShape.SINE,
        stereo_offset: float = 0.0,
        phaser_type: PhaserType = PhaserType.ANALOG
    ) -> 'PhaserEffect':
        """
        Configure phaser parameters.
        
        Args:
            stages: Number of allpass filter stages (2-12)
            rate_hz: LFO rate in Hz
            depth: Modulation depth (0-1)
            feedback: Feedback amount (-1 to 1)
            frequency_min_hz: Minimum cutoff frequency
            frequency_max_hz: Maximum cutoff frequency
            mix: Wet/dry mix (0-1)
            lfo_shape: LFO waveform shape
            stereo_offset: Stereo phase offset
            phaser_type: Phaser algorithm type
            
        Returns:
            Self for chaining
        """
        # Clamp stages to valid range
        stages = max(2, min(12, stages))
        
        # Reinitialize stages if count changed
        if len(self.stages) != stages:
            self._init_stages(stages)
        
        # Update LFO shape
        self.lfo.shape = lfo_shape
        self.lfo_right.shape = lfo_shape
        
        # Update LFO rates
        self.lfo.set_rate(rate_hz)
        self.lfo_right.set_rate(rate_hz * 1.01)  # Slight detune for stereo
        
        # Store parameters
        self.params.stages = stages
        self.params.rate_hz = rate_hz
        self.params.depth = depth
        self.params.feedback = feedback
        self.params.frequency_min_hz = frequency_min_hz
        self.params.frequency_max_hz = frequency_max_hz
        self.params.mix = mix
        self.params.lfo_shape = lfo_shape
        self.params.stereo_offset = stereo_offset
        self.params.phaser_type = phaser_type
        
        # Update stage feedback
        for stage in self.stages:
            stage.feedback = feedback
        
        return self
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through phaser effect.
        
        Args:
            audio: Input audio (mono or stereo)
            
        Returns:
            Phaser-processed audio
        """
        if self.bypass:
            return audio
        
        is_stereo = audio.ndim == 2
        num_samples = len(audio[0]) if is_stereo else len(audio)
        
        # Generate LFO modulation
        lfo_left = self.lfo.process(num_samples)
        
        if is_stereo:
            # Stereo processing with offset
            phase_offset = self.params.stereo_offset
            lfo_right = self._generate_phased_lfo(num_samples, phase_offset)
            
            # Process each channel
            left_out = self._process_channel(audio[0], lfo_left)
            right_out = self._process_channel(audio[1], lfo_right)
            
            wet = np.array([left_out, right_out])
        else:
            # Mono processing
            wet = self._process_channel(audio, lfo_left)
        
        # Mix wet and dry
        dry = audio if not is_stereo else audio
        output = dry * (1 - self.params.mix) + wet * self.params.mix
        
        return output
    
    def _generate_phased_lfo(
        self, 
        num_samples: int, 
        phase_offset: float
    ) -> np.ndarray:
        """Generate LFO with phase offset."""
        # Reset and generate with offset
        original_phase = self.lfo_right.phase
        self.lfo_right.phase = phase_offset
        lfo = self.lfo_right.process(num_samples)
        self.lfo_right.phase = original_phase
        return lfo
    
    def _process_channel(
        self, 
        audio: np.ndarray, 
        lfo: np.ndarray
    ) -> np.ndarray:
        """Process a single channel through all phaser stages."""
        output = audio.copy()
        
        # Map LFO to frequency range
        freq_min = self.params.frequency_min_hz
        freq_max = self.params.frequency_max_hz
        
        # Convert frequency to delay fraction based on phaser type
        for stage in self.stages:
            stage_output = np.zeros_like(output)
            
            for i in range(len(output)):
                # Map LFO to delay fraction
                lfo_val = lfo[i] * self.params.depth
                
                if self.params.phaser_type == PhaserType.ANALOG:
                    # Analog style: smooth frequency mapping
                    delay_fraction = lfo_val
                elif self.params.phaser_type == PhaserType.DIGITAL:
                    # Digital style: linear mapping
                    delay_fraction = lfo_val * 0.8 + 0.1
                elif self.params.phaser_type == PhaserType.VINTAGE:
                    # Vintage style: more pronounced notches
                    delay_fraction = lfo_val * 0.6 + 0.2
                elif self.params.phaser_type == PhaserType.MODERN:
                    # Modern style: extended range
                    delay_fraction = lfo_val * 0.9 + 0.05
                elif self.params.phaser_type == PhaserType.STEP:
                    # Stepped LFO (for retro sound)
                    step = int(lfo_val * 8) / 8.0
                    delay_fraction = step * 0.7 + 0.15
                else:
                    delay_fraction = lfo_val * 0.5 + 0.25
                
                # Process through stage
                stage_output[i] = stage.process_single(
                    output[i:i+1], 
                    delay_fraction
                )[0]
            
            output = stage_output
        
        return output
    
    def set_bypass(self, bypass: bool):
        """Enable or bypass the effect."""
        self.bypass = bypass
    
    def reset(self):
        """Reset all filter states."""
        for stage in self.stages:
            stage.reset()
        self.lfo.reset()
        self.lfo_right.reset()
    
    def get_params(self) -> PhaserParams:
        """Get current parameters."""
        return self.params
    
    def set_params(self, params: PhaserParams):
        """Set parameters from PhaserParams dataclass."""
        self.configure(
            stages=params.stages,
            rate_hz=params.rate_hz,
            depth=params.depth,
            feedback=params.feedback,
            frequency_min_hz=params.frequency_min_hz,
            frequency_max_hz=params.frequency_max_hz,
            mix=params.mix,
            lfo_shape=params.lfo_shape,
            stereo_offset=params.stereo_offset,
            phaser_type=params.phaser_type
        )


def create_phaser(
    sample_rate: int = 44100,
    stages: int = 4,
    rate_hz: float = 0.5,
    depth: float = 1.0,
    feedback: float = 0.0,
    mix: float = 0.5
) -> PhaserEffect:
    """
    Factory function to create a configured phaser.
    
    Args:
        sample_rate: Audio sample rate
        stages: Number of allpass filter stages
        rate_hz: LFO rate in Hz
        depth: Modulation depth (0-1)
        feedback: Feedback amount (-1 to 1)
        mix: Wet/dry mix (0-1)
        
    Returns:
        Configured PhaserEffect instance
    """
    phaser = PhaserEffect(sample_rate)
    phaser.configure(
        stages=stages,
        rate_hz=rate_hz,
        depth=depth,
        feedback=feedback,
        mix=mix
    )
    return phaser


# Example usage
if __name__ == "__main__":
    # Create phaser with default settings
    phaser = create_phaser(
        sample_rate=44100,
        stages=4,
        rate_hz=0.5,
        depth=1.0,
        mix=0.5
    )
    
    # Process test signal
    import numpy as np
    
    # Generate test signal (sine wave)
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(44100 * duration))
    test_signal = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Process through phaser
    output = phaser.process(test_signal)
    
    print(f"Phaser effect created successfully")
    print(f"Stages: {phaser.params.stages}")
    print(f"Rate: {phaser.params.rate_hz} Hz")
    print(f"Depth: {phaser.params.depth}")
    print(f"Input samples: {len(test_signal)}")
    print(f"Output samples: {len(output)}")
