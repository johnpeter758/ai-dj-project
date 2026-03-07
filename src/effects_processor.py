"""
Effects Processor for AI DJ Project
Audio effects using numpy for processing
"""

import numpy as np
from typing import Optional, Tuple


class EffectsProcessor:
    """Audio effects processor with reverb, delay, chorus, distortion, and filters."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    # =========================================================================
    # REVERB
    # =========================================================================
    
    def reverb(
        self, 
        audio: np.ndarray, 
        decay: float = 0.5, 
        wet_gain: float = 0.3,
        room_size: float = 0.5
    ) -> np.ndarray:
        """
        Algorithmic reverb using comb and allpass filters.
        
        Args:
            audio: Input audio (mono or stereo)
            decay: Decay factor (0-1)
            wet_gain: Wet signal mix (0-1)
            room_size: Room size simulation
            
        Returns:
            Reverb-processed audio
        """
        is_stereo = audio.ndim == 2
        
        # Comb filter delays (in samples) for different room sizes
        delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
        delays = [int(d * room_size) for d in delays]
        
        # Allpass filter delays
        allpass_delays = [225, 556, 441, 341]
        
        def comb_process(audio_mono: np.ndarray, delay: int, decay_val: float) -> np.ndarray:
            """Single comb filter for reverb."""
            output = np.zeros_like(audio_mono)
            buffer = np.zeros(delay)
            
            for i in range(len(audio_mono)):
                buffer[i % delay] = audio_mono[i] + buffer[i % delay] * decay_val
                output[i] = buffer[i % delay]
            
            return output
        
        def allpass_process(audio_mono: np.ndarray, delay: int, decay_val: float) -> np.ndarray:
            """Single allpass filter for diffusion."""
            output = np.zeros_like(audio_mono)
            buffer = np.zeros(delay)
            
            for i in range(len(audio_mono)):
                buffered = buffer[i % delay]
                buffer[i % delay] = output[i] + buffered * decay_val
                output[i] = buffered - output[i] * decay_val
            
            return output
        
        if is_stereo:
            left = self.reverb(audio[0], decay, wet_gain, room_size)
            right = self.reverb(audio[1], decay, wet_gain, room_size)
            return np.array([left, right])
        
        # Process through comb filters in parallel
        wet = np.zeros_like(audio)
        for delay in delays:
            wet += comb_process(audio, delay, decay)
        wet /= len(delays)
        
        # Process through allpass filters
        for delay in allpass_delays:
            wet = allpass_process(wet, delay, 0.5)
        
        # Mix wet and dry
        return audio + wet * wet_gain
    
    def convolution_reverb(
        self, 
        audio: np.ndarray, 
        impulse_response: np.ndarray,
        mix: float = 0.5
    ) -> np.ndarray:
        """
        Convolution reverb using an impulse response.
        
        Args:
            audio: Input audio
            impulse_response: IR file (load with scipy.io.wavfile)
            mix: Wet/dry mix (0 = dry, 1 = wet)
            
        Returns:
            Convolved audio
        """
        # Simple convolution using FFT
        ir = impulse_response[:]  # Copy to avoid modifying original
        ir = ir / (np.max(np.abs(ir)) + 1e-8)  # Normalize
        
        # Use FFT convolution for efficiency
        n = len(audio) + len(ir) - 1
        A = np.fft.rfft(audio, n)
        B = np.fft.rfft(ir, n)
        convolved = np.fft.irfft(A * B, n)
        
        # Mix wet and dry
        return audio * (1 - mix) + convolved * mix
    
    # =========================================================================
    # DELAY
    # =========================================================================
    
    def delay_mono(
        self,
        audio: np.ndarray,
        delay_ms: float = 250.0,
        feedback: float = 0.4,
        mix: float = 0.5
    ) -> np.ndarray:
        """
        Mono delay effect.
        
        Args:
            audio: Input audio
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount (0-1)
            mix: Wet/dry mix (0-1)
            
        Returns:
            Delayed audio
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        output = np.copy(audio)
        
        for i in range(delay_samples, len(audio)):
            output[i] += output[i - delay_samples] * feedback
        
        return audio * (1 - mix) + output * mix
    
    def delay_stereo(
        self,
        audio: np.ndarray,
        delay_ms: float = 250.0,
        feedback: float = 0.4,
        mix: float = 0.5,
        offset_ms: float = 0.0
    ) -> np.ndarray:
        """
        Stereo delay with optional offset between channels.
        
        Args:
            audio: Input audio (stereo 2D array or mono 1D)
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount (0-1)
            mix: Wet/dry mix (0-1)
            offset_ms: Offset between left/right channels
            
        Returns:
            Stereo delayed audio
        """
        is_stereo = audio.ndim == 2
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        offset_samples = int(offset_ms * self.sample_rate / 1000)
        
        if is_stereo:
            left = self.delay_mono(audio[0], delay_ms, feedback, mix)
            right = self.delay_mono(
                audio[1], delay_ms + offset_ms, feedback, mix
            ) if offset_ms > 0 else self.delay_mono(audio[1], delay_ms, feedback, mix)
            return np.array([left, right])
        
        # Mono input - create stereo output
        left = self.delay_mono(audio, delay_ms, feedback, mix)
        right = self.delay_mono(audio, delay_ms + offset_ms, feedback, mix)
        return np.array([left, right])
    
    def delay_ping_pong(
        self,
        audio: np.ndarray,
        delay_ms: float = 250.0,
        feedback: float = 0.4,
        mix: float = 0.5
    ) -> np.ndarray:
        """
        Ping-pong delay - bounces between left and right channels.
        
        Args:
            audio: Input audio (mono)
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount (0-1)
            mix: Wet/dry mix (0-1)
            
        Returns:
            Stereo ping-pong delayed audio
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        left = np.zeros_like(audio)
        right = np.zeros_like(audio)
        
        left_buffer = np.zeros(delay_samples)
        right_buffer = np.zeros(delay_samples)
        
        for i in range(len(audio)):
            # Read from buffers
            l_out = left_buffer[i % delay_samples]
            r_out = right_buffer[i % delay_samples]
            
            # Write to output (alternate)
            if i % 2 == 0:
                left[i] = audio[i] + r_out * feedback
                right[i] = r_out
                left_buffer[i % delay_samples] = left[i]
            else:
                right[i] = audio[i] + l_out * feedback
                left[i] = l_out
                right_buffer[i % delay_samples] = right[i]
        
        wet = np.array([left, right])
        return audio * (1 - mix) + wet.mean(axis=0) * mix
    
    # =========================================================================
    # CHORUS
    # =========================================================================
    
    def chorus(
        self,
        audio: np.ndarray,
        rate_hz: float = 1.5,
        depth: float = 0.5,
        mix: float = 0.5
    ) -> np.ndarray:
        """
        Chorus effect using modulated delay.
        
        Args:
            audio: Input audio
            rate_hz: LFO rate in Hz
            depth: Modulation depth (0-1)
            mix: Wet/dry mix (0-1)
            
        Returns:
            Chorus-processed audio
        """
        is_stereo = audio.ndim == 2
        
        # Delay parameters
        base_delay_ms = 20.0  # Base delay in ms
        max_deviation_ms = 5.0  # Maximum LFO deviation
        
        base_delay = int(base_delay_ms * self.sample_rate / 1000)
        max_dev = int(max_deviation_ms * depth * self.sample_rate / 1000)
        
        # Generate LFO
        t = np.arange(len(audio)) / self.sample_rate
        lfo = np.sin(2 * np.pi * rate_hz * t) * max_dev
        
        def process_mono(audio_mono: np.ndarray) -> np.ndarray:
            output = np.zeros_like(audio_mono)
            buffer = np.zeros(base_delay + max_dev + 100)
            
            for i in range(len(audio_mono)):
                # Calculate delay for this sample
                delay = int(base_delay + lfo[i])
                delay = max(0, min(delay, len(buffer) - 1))
                
                # Read from delayed position
                output[i] = audio_mono[i] + buffer[delay] * 0.5
                
                # Write to current position
                buffer[i % len(buffer)] = audio_mono[i]
            
            return output
        
        if is_stereo:
            # Slightly different LFO for each channel for width
            left = self.chorus(audio[0], rate_hz, depth, mix)
            right = self.chorus(audio[1], rate_hz * 1.01, depth * 1.1, mix)
            return np.array([left, right])
        
        return process_mono(audio) * mix + audio * (1 - mix)
    
    # =========================================================================
    # DISTORTION
    # =========================================================================
    
    def distortion_soft_clip(
        self,
        audio: np.ndarray,
        drive: float = 1.0,
        mix: float = 1.0
    ) -> np.ndarray:
        """
        Soft clipping distortion (smooth saturation).
        
        Args:
            audio: Input audio
            drive: Amount of drive (0-10+)
            mix: Wet/dry mix (0-1)
            
        Returns:
            Soft-clipped audio
        """
        # Soft clip using exponential function
        def soft_clip(x: np.ndarray) -> np.ndarray:
            return 1 - np.exp(-drive * x)
        
        # Handle positive and negative separately for asymmetric clipping
        positive = (audio > 0) * (1 - np.exp(-drive * audio))
        negative = (audio < 0) * (-1 + np.exp(drive * audio))
        
        distorted = positive + negative
        
        # Normalize output
        distorted = distorted / (np.max(np.abs(distorted)) + 1e-8)
        
        return audio * (1 - mix) + distorted * mix
    
    def distortion_hard_clip(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        mix: float = 1.0
    ) -> np.ndarray:
        """
        Hard clipping distortion.
        
        Args:
            audio: Input audio
            threshold: Clipping threshold (0-1)
            mix: Wet/dry mix (0-1)
            
        Returns:
            Hard-clipped audio
        """
        # Normalize first
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio_norm = audio / peak
        else:
            audio_norm = audio
        
        # Hard clip at threshold
        distorted = np.clip(audio_norm, -threshold, threshold)
        
        # Boost to compensate
        distorted = distorted / (threshold + 1e-8)
        
        return audio * (1 - mix) + distorted * mix
    
    def distortion_tanh(
        self,
        audio: np.ndarray,
        drive: float = 2.0,
        mix: float = 1.0
    ) -> np.ndarray:
        """
        Tanh (sigmoid) distortion - smooth, tube-like saturation.
        
        Args:
            audio: Input audio
            drive: Amount of drive (>0)
            mix: Wet/dry mix (0-1)
            
        Returns:
            Tanh-distorted audio
        """
        distorted = np.tanh(audio * drive)
        
        return audio * (1 - mix) + distorted * mix
    
    def distortion(
        self,
        audio: np.ndarray,
        type: str = "tanh",
        drive: float = 2.0,
        mix: float = 1.0
    ) -> np.ndarray:
        """
        Universal distortion with type selection.
        
        Args:
            audio: Input audio
            type: Distortion type ('soft', 'hard', 'tanh')
            drive: Amount of drive
            mix: Wet/dry mix
            
        Returns:
            Distorted audio
        """
        if type == "soft":
            return self.distortion_soft_clip(audio, drive, mix)
        elif type == "hard":
            return self.distortion_hard_clip(audio, drive, mix)
        elif type == "tanh":
            return self.distortion_tanh(audio, drive, mix)
        else:
            raise ValueError(f"Unknown distortion type: {type}")
    
    # =========================================================================
    # FILTERS
    # =========================================================================
    
    def filter_lowpass(
        self,
        audio: np.ndarray,
        cutoff_hz: float = 1000.0,
        resonance: float = 0.0
    ) -> np.ndarray:
        """
        Low-pass filter.
        
        Args:
            audio: Input audio
            cutoff_hz: Cutoff frequency in Hz
            resonance: Resonance/Q factor (0-10)
            
        Returns:
            Low-pass filtered audio
        """
        return self._filter_biquad(audio, cutoff_hz, "lowpass", resonance)
    
    def filter_highpass(
        self,
        audio: np.ndarray,
        cutoff_hz: float = 1000.0,
        resonance: float = 0.0
    ) -> np.ndarray:
        """
        High-pass filter.
        
        Args:
            audio: Input audio
            cutoff_hz: Cutoff frequency in Hz
            resonance: Resonance/Q factor (0-10)
            
        Returns:
            High-pass filtered audio
        """
        return self._filter_biquad(audio, cutoff_hz, "highpass", resonance)
    
    def filter_bandpass(
        self,
        audio: np.ndarray,
        center_hz: float = 1000.0,
        q: float = 1.0
    ) -> np.ndarray:
        """
        Band-pass filter.
        
        Args:
            audio: Input audio
            center_hz: Center frequency in Hz
            q: Q factor (bandwidth)
            
        Returns:
            Band-pass filtered audio
        """
        return self._filter_biquad(audio, center_hz, "bandpass", q)
    
    def _filter_biquad(
        self,
        audio: np.ndarray,
        freq: float,
        filter_type: str,
        q: float = 0.707
    ) -> np.ndarray:
        """
        Biquad (second-order) filter implementation.
        
        Args:
            audio: Input audio
            freq: Frequency parameter (cutoff or center)
            filter_type: 'lowpass', 'highpass', 'bandpass'
            q: Q factor
            
        Returns:
            Filtered audio
        """
        is_stereo = audio.ndim == 2
        
        if is_stereo:
            left = self._filter_biquad(audio[0], freq, filter_type, q)
            right = self._filter_biquad(audio[1], freq, filter_type, q)
            return np.array([left, right])
        
        # Calculate biquad coefficients
        w0 = 2 * np.pi * freq / self.sample_rate
        q_safe = max(q, 0.001)  # Avoid division by zero
        alpha = np.sin(w0) / (2 * q_safe)
        
        cos_w0 = np.cos(w0)
        
        if filter_type == "lowpass":
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = (1 - cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif filter_type == "highpass":
            b0 = (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 = (1 + cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif filter_type == "bandpass":
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Normalize coefficients
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        
        # Apply filter
        output = np.zeros_like(audio)
        x1, x2 = 0.0, 0.0
        y1, y2 = 0.0, 0.0
        
        for i in range(len(audio)):
            output[i] = b0 * audio[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, audio[i]
            y2, y1 = y1, output[i]
        
        return output
    
    # =========================================================================
    # CHAIN PROCESSING
    # =========================================================================
    
    def process_chain(
        self,
        audio: np.ndarray,
        effects: list,
        **params
    ) -> np.ndarray:
        """
        Chain multiple effects together.
        
        Args:
            audio: Input audio
            effects: List of effect names in order
            **params: Parameters for each effect
            
        Returns:
            Processed audio
        """
        result = audio
        
        effect_map = {
            "reverb": self.reverb,
            "delay": self.delay_mono,
            "delay_stereo": self.delay_stereo,
            "delay_pingpong": self.delay_ping_pong,
            "chorus": self.chorus,
            "distortion": self.distortion,
            "distortion_tanh": self.distortion_tanh,
            "distortion_soft": self.distortion_soft_clip,
            "distortion_hard": self.distortion_hard_clip,
            "lowpass": self.filter_lowpass,
            "highpass": self.filter_highpass,
            "bandpass": self.filter_bandpass,
        }
        
        for effect in effects:
            if effect in effect_map:
                effect_params = params.get(effect, {})
                result = effect_map[effect](result, **effect_params)
        
        return result


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def generate_test_signal(duration: float = 2.0, sample_rate: int = 44100) -> np.ndarray:
    """Generate a test signal (sine wave sweep + noise)."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Sweep from 100Hz to 2000Hz
    freq = 100 + (2000 - 100) * (t / duration)
    signal = np.sin(2 * np.pi * freq * t) * 0.3
    
    # Add some noise
    noise = np.random.randn(len(t)) * 0.05
    
    # Add some transients (drum-like)
    impulse_times = [0.5, 1.0, 1.5]
    for imp_t in impulse_times:
        idx = int(imp_t * sample_rate)
        if idx < len(signal):
            signal[idx:idx+100] += np.sin(np.linspace(0, 20, 100)) * 0.5
    
    return signal + noise


def example_basic_effects():
    """Example: Basic effects usage."""
    print("=== Basic Effects Example ===\n")
    
    # Initialize processor
    fx = EffectsProcessor(sample_rate=44100)
    
    # Generate test signal
    audio = generate_test_signal(duration=2.0)
    stereo_audio = np.array([audio, audio])  # Stereo version
    
    print(f"Input signal shape: {audio.shape}")
    print(f"Stereo signal shape: {stereo_audio.shape}")
    
    # Reverb
    with_reverb = fx.reverb(audio, decay=0.5, wet_gain=0.3)
    print(f"✓ Reverb applied: {with_reverb.shape}")
    
    # Delay
    with_delay = fx.delay_mono(audio, delay_ms=250, feedback=0.4, mix=0.5)
    print(f"✓ Delay (mono) applied: {with_delay.shape}")
    
    # Stereo delay
    with_stereo_delay = fx.delay_stereo(stereo_audio, delay_ms=200, offset_ms=20)
    print(f"✓ Delay (stereo) applied: {with_stereo_delay.shape}")
    
    # Ping-pong delay
    with_pingpong = fx.delay_ping_pong(audio, delay_ms=300)
    print(f"✓ Ping-pong delay applied: {with_pingpong.shape}")
    
    # Chorus
    with_chorus = fx.chorus(audio, rate_hz=1.5, depth=0.5)
    print(f"✓ Chorus applied: {with_chorus.shape}")
    
    # Distortion
    with_dist_tanh = fx.distortion_tanh(audio, drive=2.0)
    print(f"✓ Distortion (tanh) applied: {with_dist_tanh.shape}")
    
    with_dist_soft = fx.distortion_soft_clip(audio, drive=3.0)
    print(f"✓ Distortion (soft clip) applied: {with_dist_soft.shape}")
    
    with_dist_hard = fx.distortion_hard_clip(audio, threshold=0.5)
    print(f"✓ Distortion (hard clip) applied: {with_dist_hard.shape}")
    
    # Filters
    lowpass = fx.filter_lowpass(audio, cutoff_hz=2000)
    print(f"✓ Low-pass filter applied: {lowpass.shape}")
    
    highpass = fx.filter_highpass(audio, cutoff_hz=200)
    print(f"✓ High-pass filter applied: {highpass.shape}")
    
    bandpass = fx.filter_bandpass(audio, center_hz=1000, q=2.0)
    print(f"✓ Band-pass filter applied: {bandpass.shape}")


def example_effect_chain():
    """Example: Chaining effects."""
    print("\n=== Effect Chain Example ===\n")
    
    fx = EffectsProcessor(sample_rate=44100)
    audio = generate_test_signal(duration=1.0)
    
    # Create a chain: High-pass -> Distortion -> Reverb
    processed = fx.process_chain(
        audio,
        effects=["highpass", "distortion", "reverb"],
        highpass={"cutoff_hz": 200},
        distortion={"type": "tanh", "drive": 1.5},
        reverb={"decay": 0.4, "wet_gain": 0.3}
    )
    
    print(f"✓ Chain processed: {audio.shape} -> {processed.shape}")


def example_guitar_chain():
    """Example: Typical guitar effects chain."""
    print("\n=== Guitar Effects Chain Example ===\n")
    
    fx = EffectsProcessor(sample_rate=44100)
    audio = generate_test_signal(duration=2.0)
    
    # Classic guitar chain: Overdrive -> Chorus -> Delay -> Reverb
    guitar_tone = fx.process_chain(
        audio,
        effects=["distortion", "chorus", "delay", "reverb"],
        distortion={"type": "tanh", "drive": 2.5, "mix": 0.8},
        chorus={"rate_hz": 1.2, "depth": 0.4, "mix": 0.3},
        delay={"delay_ms": 300, "feedback": 0.3, "mix": 0.4},
        reverb={"decay": 0.4, "wet_gain": 0.25}
    )
    
    print(f"✓ Guitar chain applied: {audio.shape} -> {guitar_tone.shape}")


def example_drum_bus():
    """Example: Drum bus processing."""
    print("\n=== Drum Bus Processing Example ===\n")
    
    fx = EffectsProcessor(sample_rate=44100)
    audio = generate_test_signal(duration=1.5)
    
    # Drum bus: Parallel compression simulation via distortion + low-pass
    drum_bus = fx.process_chain(
        audio,
        effects=["highpass", "distortion_hard", "lowpass"],
        highpass={"cutoff_hz": 60},  # Sub removal
        distortion_hard={"threshold": 0.7, "mix": 0.15},  # Slight grit
        lowpass={"cutoff_hz": 8000}  # Air
    )
    
    print(f"✓ Drum bus processed: {audio.shape} -> {drum_bus.shape}")


def example_stereo_widening():
    """Example: Stereo widening effect."""
    print("\n=== Stereo Widening Example ===\n")
    
    fx = EffectsProcessor(sample_rate=44100)
    audio = generate_test_signal(duration=1.0)
    
    # Convert to stereo and apply stereo effects
    stereo = np.array([audio, audio])
    
    # Slight chorus + delay offset for width
    widened = fx.delay_stereo(
        stereo,
        delay_ms=15,
        feedback=0.2,
        mix=0.3,
        offset_ms=10
    )
    
    # Add chorus
    widened = fx.chorus(widened, rate_hz=0.8, depth=0.2, mix=0.2)
    
    print(f"✓ Stereo widened: {stereo.shape} -> {widened.shape}")


if __name__ == "__main__":
    # Run all examples
    example_basic_effects()
    example_effect_chain()
    example_guitar_chain()
    example_drum_bus()
    example_stereo_widening()
    
    print("\n" + "="*50)
    print("All examples completed successfully!")
    print("="*50)
