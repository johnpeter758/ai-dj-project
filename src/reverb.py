"""
Algorithmic Reverb Module for AI DJ Project
Advanced reverb effects using Schroeder and Freeverb algorithms
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class ReverbType(Enum):
    """Reverb algorithm types."""
    HALL = "hall"
    ROOM = "room"
    PLATE = "plate"
    CATHEDRAL = "cathedral"
    CHAMBER = "chamber"
    AMBIENT = "ambient"
    SPRING = "spring"


@dataclass
class ReverbParams:
    """Parameters for algorithmic reverb."""
    reverb_type: ReverbType = ReverbType.ROOM
    room_size: float = 0.5          # 0.0 - 1.0
    damping: float = 0.5            # 0.0 - 1.0 (high frequency absorption)
    wet_dry: float = 0.3            # 0.0 - 1.0
    width: float = 1.0             # stereo width 0.0 - 1.0
    pre_delay_ms: float = 0.0      # 0 - 100 ms
    early_reflections: float = 0.3 # 0.0 - 1.0
    freeze: bool = False            # infinite reverb
    sample_rate: int = 44100


class AlgorithmicReverb:
    """
    Algorithmic reverb using Schroeder reverb and Freeverb algorithms.
    
    Implements:
    - Schroeder reverb (parallel comb filters + series allpass filters)
    - Freeverb algorithm with modulated allpass filters
    - Early reflections simulation
    - Pre-delay
    - Stereo width control
    - HF/LF damping
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._reset_buffers()
    
    def _reset_buffers(self):
        """Initialize reverb buffers and state."""
        # Freeverb-style comb filter delays (in samples at 44100Hz)
        self.comb_tunings = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
        # Scale for different sample rates
        self.comb_tunings = [int(t * self.sample_rate / 44100) for t in self.comb_tunings]
        
        # Allpass filter delays
        self.allpass_tunings = [225, 556, 441, 341]
        self.allpass_tunings = [int(t * self.sample_rate / 44100) for t in self.allpass_tunings]
        
        # Comb filter buffers
        self.comb_buffers: List[np.ndarray] = [
            np.zeros(delay) for delay in self.comb_tunings
        ]
        self.comb_buffer_index = 0
        
        # Allpass filter buffers
        self.allpass_buffers: List[np.ndarray] = [
            np.zeros(delay) for delay in self.allpass_tunings
        ]
        self.allpass_buffer_index = 0
        
        # Filter state for damping
        self._filter_store = 0.0
        
        # Stereo buffers (for Freeverb)
        self._left_filt = 0.0
        self._right_filt = 0.0
        
        # Parameters
        self._params = ReverbParams(sample_rate=sample_rate)
        self._update_params()
    
    def _update_params(self):
        """Update internal parameters from ReverbParams."""
        p = self._params
        
        # Scale parameters
        self._fixed_gain = 0.015
        self._scale_wet = p.wet_dry
        self._scale_dry = 1.0 - p.wet_dry
        self._scale_damp = p.damping
        self._scale_room = 0.28 + (p.room_size * 0.7)
        self._offset_room = 0.7 - (p.room_size * 0.3)
        
        # Stereo width
        self._stereo_width = p.width if p.width > 0 else 0.0
        self._stereo_width = min(1.0, self._stereo_width)
        
        # Feedback for infinite reverb
        self._feedback = 1.0 if p.freeze else 0.84 + (p.room_size * 0.16)
    
    def set_params(self, params: ReverbParams):
        """Set reverb parameters."""
        self._params = params
        self._update_params()
    
    def _process_comb_filter(self, input_sample: float, buffer: np.ndarray, 
                             filter_state: float) -> Tuple[float, float]:
        """
        Process a single sample through a comb filter.
        
        Args:
            input_sample: Input audio sample
            buffer: Comb filter buffer
            filter_state: Current filter state
            
        Returns:
            Tuple of (output_sample, new_filter_state)
        """
        delay = len(buffer)
        
        # Read output
        output = buffer[self.comb_buffer_index]
        
        # Lowpass filter for damping
        filter_old = output
        output = filter_state + (output * self._scale_damp)
        filter_new = filter_old
        
        # Write input with feedback
        buffer[self.comb_buffer_index] = input_sample + (output * self._feedback)
        
        return output, filter_new
    
    def _process_allpass_filter(self, input_sample: float, buffer: np.ndarray) -> float:
        """
        Process a single sample through an allpass filter.
        
        Args:
            input_sample: Input audio sample
            buffer: Allpass filter buffer
            
        Returns:
            Output sample
        """
        delay = len(buffer)
        
        # Read buffered sample
        buffer_out = buffer[self.allpass_buffer_index]
        
        # Allpass calculation
        output = -input_sample + buffer_out
        buffer[self.allpass_buffer_index] = input_sample + (buffer_out * 0.5)
        
        return output
    
    def process(self, audio: np.ndarray, params: Optional[ReverbParams] = None) -> np.ndarray:
        """
        Process audio through the reverb.
        
        Args:
            audio: Input audio (mono or stereo)
            params: Optional reverb parameters
            
        Returns:
            Reverb-processed audio
        """
        if params:
            self.set_params(params)
        
        # Handle stereo input
        if audio.ndim == 2:
            return self._process_stereo(audio)
        
        return self._process_mono(audio)
    
    def _process_mono(self, audio: np.ndarray) -> np.ndarray:
        """Process mono audio through reverb."""
        output = np.zeros_like(audio)
        
        # Process through parallel comb filters
        comb_output = np.zeros_like(audio)
        filter_states = [0.0] * len(self.comb_buffers)
        
        for i in range(len(audio)):
            sample = audio[i]
            sample_combined = 0.0
            
            # Parallel comb filters
            for j, buffer in enumerate(self.comb_buffers):
                out, filter_states[j] = self._process_comb_filter(
                    sample, buffer, filter_states[j]
                )
                sample_combined += out
            
            sample_combined /= len(self.comb_buffers)
            
            # Series allpass filters for diffusion
            sample_allpass = sample_combined
            for buffer in self.allpass_buffers:
                sample_allpass = self._process_allpass_filter(sample_allpass, buffer)
            
            # Mix wet/dry
            output[i] = (audio[i] * self._scale_dry + 
                        sample_allpass * self._scale_wet)
        
        # Update buffer indices
        self._advance_buffer_indices()
        
        return self._normalize(output)
    
    def _process_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Process stereo audio through reverb."""
        left = audio[0]
        right = audio[1] if audio.shape[0] > 1 else left
        
        output_left = np.zeros_like(left)
        output_right = np.zeros_like(right)
        
        # Create stereo variant buffers
        comb_buffers_L = [buf.copy() for buf in self.comb_buffers]
        comb_buffers_R = [buf.copy() for buf in self.comb_buffers]
        allpass_buffers_L = [buf.copy() for buf in self.allpass_buffers]
        allpass_buffers_R = [buf.copy() for buf in self.allpass_buffers]
        
        filter_states_L = [0.0] * len(comb_buffers_L)
        filter_states_R = [0.0] * len(comb_buffers_R)
        
        for i in range(len(left)):
            # Left channel
            sample_L = left[i]
            sample_combined_L = 0.0
            for j, buffer in enumerate(comb_buffers_L):
                out, filter_states_L[j] = self._process_comb_filter(
                    sample_L, buffer, filter_states_L[j]
                )
                sample_combined_L += out
            sample_combined_L /= len(comb_buffers_L)
            
            for buffer in allpass_buffers_L:
                sample_combined_L = self._process_allpass_filter(sample_combined_L, buffer)
            
            # Right channel (with stereo crossing)
            sample_R = right[i]
            sample_combined_R = 0.0
            for j, buffer in enumerate(comb_buffers_R):
                out, filter_states_R[j] = self._process_comb_filter(
                    sample_R, buffer, filter_states_R[j]
                )
                sample_combined_R += out
            sample_combined_R /= len(comb_buffers_R)
            
            for buffer in allpass_buffers_R:
                sample_combined_R = self._process_allpass_filter(sample_combined_R, buffer)
            
            # Apply stereo width
            mid = (sample_combined_L + sample_combined_R) / 2
            side = (sample_combined_L - sample_combined_R) / 2 * self._stereo_width
            sample_combined_L = mid + side
            sample_combined_R = mid - side
            
            # Mix
            output_left[i] = left[i] * self._scale_dry + sample_combined_L * self._scale_wet
            output_right[i] = right[i] * self._scale_dry + sample_combined_R * self._scale_wet
        
        return self._normalize(np.array([output_left, output_right]))
    
    def _advance_buffer_indices(self):
        """Advance all buffer indices."""
        # Advance comb buffer index
        self.comb_buffer_index = (self.comb_buffer_index + 1) % len(self.comb_buffers[0])
        
        # Advance allpass buffer index
        self.allpass_buffer_index = (self.allpass_buffer_index + 1) % len(self.allpass_buffers[0])
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        return audio
    
    def reset(self):
        """Reset all reverb buffers."""
        self._reset_buffers()
    
    def __call__(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """Convenience method to process audio."""
        if kwargs:
            params = ReverbParams(**kwargs)
            return self.process(audio, params)
        return self.process(audio)


class SchroederReverb:
    """
    Classic Schroeder reverb implementation.
    Uses parallel comb filters and series allpass filters.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._init_buffers()
    
    def _init_buffers(self):
        """Initialize Schroeder reverb buffers."""
        # Standard Schroeder comb delays
        self.comb_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
        self.comb_delays = [int(d * self.sample_rate / 44100) for d in self.comb_delays]
        
        # Standard Schroeder allpass delays
        self.allpass_delays = [225, 556, 441, 341]
        self.allpass_delays = [int(d * self.sample_rate / 44100) for d in self.allpass_delays]
        
        # Buffers
        self.comb_buffers = [np.zeros(d, dtype=np.float32) for d in self.comb_delays]
        self.allpass_buffers = [np.zeros(d, dtype=np.float32) for d in self.allpass_delays]
        
        # Indices
        self.comb_idx = [0] * len(self.comb_buffers)
        self.allpass_idx = [0] * len(self.allpass_buffers)
        
        # Parameters
        self.decay = 0.5
        self.wet_dry = 0.3
    
    def _process_comb(self, sample: float, buffer: np.ndarray, 
                      idx: int, decay: float) -> Tuple[float, int]:
        """Process comb filter."""
        delay = len(buffer)
        output = buffer[idx]
        buffer[idx] = sample + output * decay
        idx = (idx + 1) % delay
        return output, idx
    
    def _process_allpass(self, sample: float, buffer: np.ndarray, 
                         idx: int) -> Tuple[float, int]:
        """Process allpass filter."""
        delay = len(buffer)
        buffer_out = buffer[idx]
        output = -sample + buffer_out
        buffer[idx] = sample + buffer_out * 0.5
        idx = (idx + 1) % delay
        return output, idx
    
    def process(self, audio: np.ndarray, 
                decay: float = 0.5, 
                wet_dry: float = 0.3) -> np.ndarray:
        """
        Process audio through Schroeder reverb.
        
        Args:
            audio: Input audio (mono)
            decay: Decay factor (0-1)
            wet_dry: Wet/dry mix (0-1)
            
        Returns:
            Reverb-processed audio
        """
        self.decay = decay
        self.wet_dry = wet_dry
        
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            sample = audio[i]
            
            # Parallel comb filters
            comb_sum = 0.0
            for j, (buffer, idx) in enumerate(zip(self.comb_buffers, self.comb_idx)):
                out, self.comb_idx[j] = self._process_comb(
                    sample, buffer, idx, decay
                )
                comb_sum += out
            comb_sum /= len(self.comb_buffers)
            
            # Series allpass filters
            allpass_out = comb_sum
            for buffer, idx in zip(self.allpass_buffers, self.allpass_idx):
                allpass_out, self.allpass_idx = self._process_allpass(
                    allpass_out, buffer, idx
                )
            
            # Mix
            output[i] = sample * (1 - wet_dry) + allpass_out * wet_dry
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0.95:
            output = output * (0.95 / max_val)
        
        return output
    
    def reset(self):
        """Reset all buffers."""
        self._init_buffers()


class ConvolutionReverb:
    """
    Convolution reverb using impulse responses.
    For high-quality reverb simulation.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.ir = None
    
    def load_impulse(self, ir: np.ndarray):
        """Load an impulse response."""
        self.ir = ir.astype(np.float32)
    
    def generate_impulse(self, reverb_type: ReverbType, 
                         duration: float = 2.0,
                         room_size: float = 0.5) -> np.ndarray:
        """
        Generate a synthetic impulse response.
        
        Args:
            reverb_type: Type of reverb
            duration: Impulse duration in seconds
            room_size: Room size (0-1)
            
        Returns:
            Synthetic impulse response
        """
        num_samples = int(duration * self.sample_rate)
        ir = np.zeros(num_samples, dtype=np.float32)
        
        # Early reflections
        num_early = int(50 + room_size * 50)
        for _ in range(num_early):
            pos = int(np.random.uniform(0, 0.1 * self.sample_rate))
            if pos < num_samples:
                ir[pos] += np.random.uniform(0.1, 0.5) * (1.0 - room_size * 0.5)
        
        # Exponential decay
        decay_rate = 2.0 + (1.0 - room_size) * 4.0
        t = np.arange(num_samples) / self.sample_rate
        envelope = np.exp(-decay_rate * t)
        
        # Add noise with envelope
        noise = np.random.randn(num_samples).astype(np.float32) * envelope * 0.1
        
        # Different characteristics per type
        if reverb_type == ReverbType.PLATE:
            decay_rate = 6.0
            noise *= envelope * 2.0
        elif reverb_type == ReverbType.CATHEDRAL:
            decay_rate = 1.5
            noise *= envelope * 0.5
        elif reverb_type == ReverbType.AMBIENT:
            decay_rate = 0.8
            noise *= envelope * 0.3
        elif reverb_type == ReverbType.SPRING:
            # Spring reverb has character
            decay_rate = 4.0
            # Add some resonance
            for freq in [800, 1200, 2000]:
                osc = np.sin(2 * np.pi * freq * t) * envelope * 0.05
                noise += osc
        
        # Apply envelope
        ir += noise * envelope
        
        # Normalize
        ir /= np.max(np.abs(ir)) + 1e-10
        
        return ir
    
    def process(self, audio: np.ndarray, 
                reverb_type: ReverbType = ReverbType.ROOM,
                wet_dry: float = 0.3,
                room_size: float = 0.5) -> np.ndarray:
        """
        Process audio through convolution reverb.
        
        Args:
            audio: Input audio
            reverb_type: Type of reverb
            wet_dry: Wet/dry mix
            room_size: Room size
            
        Returns:
            Reverb-processed audio
        """
        # Generate impulse if not loaded
        if self.ir is None:
            self.ir = self.generate_impulse(reverb_type, room_size=room_size)
        
        # Convolution
        wet = np.convolve(audio, self.ir, mode='same')
        
        # Mix
        return audio * (1 - wet_dry) + wet * wet_dry


# Preset factory functions
def create_reverb(reverb_type: ReverbType = ReverbType.ROOM,
                   sample_rate: int = 44100) -> AlgorithmicReverb:
    """Create a reverb instance with preset parameters."""
    params = ReverbParams(
        reverb_type=reverb_type,
        room_size=0.5,
        damping=0.5,
        wet_dry=0.3,
        width=1.0,
        pre_delay_ms=0.0,
        early_reflections=0.3,
        freeze=False,
        sample_rate=sample_rate
    )
    
    # Adjust based on type
    if reverb_type == ReverbType.HALL:
        params.room_size = 0.85
        params.damping = 0.4
        params.wet_dry = 0.4
    elif reverb_type == ReverbType.PLATE:
        params.room_size = 0.7
        params.damping = 0.3
        params.wet_dry = 0.35
    elif reverb_type == ReverbType.CATHEDRAL:
        params.room_size = 1.0
        params.damping = 0.2
        params.wet_dry = 0.5
    elif reverb_type == ReverbType.ROOM:
        params.room_size = 0.5
        params.damping = 0.5
        params.wet_dry = 0.25
    elif reverb_type == ReverbType.AMBIENT:
        params.room_size = 0.9
        params.damping = 0.1
        params.wet_dry = 0.6
    elif reverb_type == ReverbType.SPRING:
        params.room_size = 0.3
        params.damping = 0.6
        params.wet_dry = 0.4
    
    reverb = AlgorithmicReverb(sample_rate)
    reverb.set_params(params)
    return reverb


# Example usage
if __name__ == "__main__":
    import soundfile as sf
    
    # Test reverb
    sample_rate = 44100
    
    # Create test signal (simple impulse)
    test_signal = np.zeros(sample_rate // 2)
    test_signal[1000] = 1.0  # Click
    
    # Test different reverb types
    for reverb_type in ReverbType:
        reverb = create_reverb(reverb_type, sample_rate)
        output = reverb.process(test_signal)
        
        print(f"{reverb_type.value}: {np.max(np.abs(output)):.3f}")
    
    print("Reverb module test complete!")
