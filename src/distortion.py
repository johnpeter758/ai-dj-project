"""
Distortion Effects - Audio Distortion and Waveshaping for AI DJ Project

Provides various distortion, overdrive, fuzz, and waveshaping effects
for creative audio processing and sound design.

Features:
- Hard/soft clipping distortion
- Waveshaping with custom transfer curves
- Bitcrusher (bit reduction)
- Sample rate reduction
- Fuzz effects (Germanium, Silicon, LED)
- Ring modulation
- Polynomial distortion
- Asymmetric distortion
- Multiband distortion
"""

import numpy as np
from typing import Optional, Tuple, Callable
from scipy import signal
from scipy.special import expit  # sigmoid function


class Distortion:
    """Audio distortion and waveshaping processor."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._buffer = None
    
    def _normalize(self, audio: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio
    
    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """Ensure mono processing."""
        if audio.ndim > 1:
            return np.mean(audio, axis=1)
        return audio
    
    # ==================== CLIPPING DISTORTION ====================
    
    def hard_clip(self, audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """
        Hard clip audio signal at threshold.
        
        Args:
            audio: Input audio signal
            threshold: Clip threshold in range [-1, 1]
        
        Returns:
            Hard-clipped audio signal
        """
        return np.clip(audio, -threshold, threshold)
    
    def soft_clip_tanh(self, audio: np.ndarray, drive: float = 1.0,
                       makeup_gain: float = 1.0) -> np.ndarray:
        """
        Soft clip using hyperbolic tangent (tanh).
        Produces smooth clipping with odd harmonics - classic tube sound.
        
        Args:
            audio: Input audio signal
            drive: Amount of distortion (1 = clean, >1 = more distortion)
            makeup_gain: Gain to apply after clipping
        
        Returns:
            Soft-clipped audio signal
        """
        # Scale input, apply tanh, scale back
        clipped = np.tanh(audio * drive) / np.tanh(drive)
        return self._normalize(clipped * makeup_gain)
    
    def soft_clip_sigmoid(self, audio: np.ndarray, drive: float = 1.0,
                          makeup_gain: float = 1.0) -> np.ndarray:
        """
        Soft clip using sigmoid function.
        Produces smooth transition with softer knee than tanh.
        
        Args:
            audio: Input audio signal
            drive: Amount of distortion
            makeup_gain: Gain to apply after clipping
        
        Returns:
            Soft-clipped audio signal
        """
        clipped = expit(audio * drive * 2 - 1) * 2 - 1
        return self._normalize(clipped * makeup_gain)
    
    def soft_clip_exponential(self, audio: np.ndarray, threshold: float = 0.8,
                               knee_width: float = 0.1) -> np.ndarray:
        """
        Soft clip with exponential curve and configurable knee.
        
        Args:
            audio: Input audio signal
            threshold: Clip threshold
            knee_width: Width of soft knee region
        
        Returns:
            Soft-clipped audio signal
        """
        output = np.zeros_like(audio)
        abs_audio = np.abs(audio)
        
        # Linear region (below knee)
        linear_mask = abs_audio <= (threshold - knee_width)
        output[linear_mask] = audio[linear_mask]
        
        # Knee region (smooth transition)
        knee_mask = (abs_audio > (threshold - knee_width)) & (abs_audio <= threshold)
        if np.any(knee_mask):
            x = (abs_audio[knee_mask] - (threshold - knee_width)) / knee_width
            knee_factor = 1 - np.exp(-x * np.pi)
            output[knee_mask] = np.sign(audio[knee_mask]) * (
                (threshold - knee_width) + knee_width * knee_factor
            )
        
        # Clipping region
        clip_mask = abs_audio > threshold
        output[clip_mask] = np.sign(audio[clip_mask]) * threshold
        
        return self._normalize(output)
    
    # ==================== WAVE SHAPING ====================
    
    def waveshape(self, audio: np.ndarray, curve: Callable[[np.ndarray], np.ndarray],
                  drive: float = 1.0) -> np.ndarray:
        """
        Generic waveshaping with custom transfer function.
        
        Args:
            audio: Input audio signal
            curve: Transfer function (takes and returns numpy array)
            drive: Input gain before waveshaping
        
        Returns:
            Waveshaped audio signal
        """
        driven = audio * drive
        shaped = curve(driven)
        return self._normalize(shaped)
    
    def chebyshev_distortion(self, audio: np.ndarray, order: int = 4,
                             drive: float = 1.0) -> np.ndarray:
        """
        Distortion using Chebyshev polynomials.
        Generates specific harmonic content based on polynomial order.
        
        Args:
            audio: Input audio signal
            order: Chebyshev polynomial order (2-8)
            drive: Input drive amount
        
        Returns:
            Distorted audio with harmonics
        """
        # Scale to valid Chebyshev range
        x = np.tanh(audio * drive)
        
        # Chebyshev polynomials of the first kind
        if order == 1:
            return x
        elif order == 2:
            return 2 * x**2 - 1
        elif order == 3:
            return 4 * x**3 - 3 * x
        elif order == 4:
            return 8 * x**4 - 8 * x**2 + 1
        elif order == 5:
            return 16 * x**5 - 20 * x**3 + 5 * x
        elif order == 6:
            return 32 * x**6 - 48 * x**4 + 18 * x**2 - 1
        elif order == 7:
            return 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x
        else:  # order 8
            return 128 * x**8 - 256 * x**6 + 160 * x**4 - 32 * x**2 + 1
    
    def cubic_distortion(self, audio: np.ndarray, drive: float = 2.0,
                         mix: float = 1.0) -> np.ndarray:
        """
        Cubic distortion - simple polynomial waveshaping.
        Adds odd harmonics (classic overdrive character).
        
        Args:
            audio: Input audio signal
            drive: Distortion amount
            mix: Wet/dry mix (0=dry, 1=full distortion)
        
        Returns:
            Cubic distorted audio
        """
        dry = audio
        distorted = audio + drive * audio**3
        return self._normalize(dry * (1 - mix) + distorted * mix)
    
    def sine_shaper(self, audio: np.ndarray, drive: float = 1.0) -> np.ndarray:
        """
        Sine wave shaper - smooth, clean overdrive.
        Excellent for subtle saturation.
        
        Args:
            audio: Input audio signal
            drive: Amount of shaping
        
        Returns:
            Sine-shaped audio
        """
        return self._normalize(np.sin(audio * np.pi * drive) / np.pi)
    
    # ==================== FUZZ EFFECTS ====================
    
    def fuzz(self, audio: np.ndarray, intensity: float = 0.5,
             bias: float = 0.0) -> np.ndarray:
        """
        Classic fuzz effect - extreme clipping with gain.
        
        Args:
            audio: Input audio signal
            intensity: Fuzz intensity (0-1)
            bias: DC bias to shift the clipping point
        
        Returns:
            Fuzz-distorted audio
        """
        # High gain before clipping
        gain = 1 + intensity * 10
        boosted = audio * gain + bias
        
        # Hard clip with asymmetric character
        threshold = 0.3 + (1 - intensity) * 0.5
        clipped = np.clip(boosted, -threshold, threshold)
        
        # Add slight asymmetry for character
        asymmetric = np.where(
            clipped > 0,
            clipped * (1 + intensity * 0.3),
            clipped
        )
        
        return self._normalize(asymmetric)
    
    def germanium_fuzz(self, audio: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Germanium transistor fuzz - warm, smooth, vintage character.
        
        Args:
            audio: Input audio signal
            intensity: Fuzz intensity (0-1)
        
        Returns:
            Germanium fuzz audio
        """
        # Germanium transistors have soft, warm clipping
        drive = 1 + intensity * 4
        biased = audio * drive
        
        # Soft knee clipping with warmth
        soft = np.tanh(biased * 1.2)
        
        # Add slight low-frequency emphasis (Germanium character)
        b, a = butter(2, 200 / (self.sample_rate / 2), 'low')
        lows = signal.lfilter(b, a, soft)
        
        # Blend with original
        return self._normalize(soft * 0.7 + lows * 0.3 * intensity)
    
    def silicon_fuzz(self, audio: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Silicon transistor fuzz - harsh, aggressive, modern character.
        
        Args:
            audio: Input audio signal
            intensity: Fuzz intensity (0-1)
        
        Returns:
            Silicon fuzz audio
        """
        drive = 1 + intensity * 6
        biased = audio * drive
        
        # Harder clipping than Germanium
        clipped = np.clip(biased, -0.7, 0.7)
        
        # Add high-frequency sizzle
        b, a = butter(2, 3000 / (self.sample_rate / 2), 'high')
        highs = signal.lfilter(b, a, clipped)
        
        return self._normalize(clipped + highs * intensity * 0.5)
    
    def led_fuzz(self, audio: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        LED fuzz - soft, squishy compression with overdrive.
        
        Args:
            audio: Input audio signal
            intensity: Fuzz intensity (0-1)
        
        Returns:
            LED fuzz audio
        """
        # LED has very soft knee characteristic
        drive = 1 + intensity * 3
        biased = audio * drive
        
        # Multiple soft clipping stages for LED feel
        result = np.tanh(biased * 1.5)
        result = np.tanh(result * 1.5)
        result = np.tanh(result * 1.5)
        
        return self._normalize(result)
    
    # ==================== BITCRUSHER & SAMPLE REDUCTION ====================
    
    def bitcrush(self, audio: np.ndarray, bits: int = 8,
                 normalize: bool = True) -> np.ndarray:
        """
        Bit crusher - reduce bit depth for lo-fi distortion.
        
        Args:
            audio: Input audio signal
            bits: Target bit depth (1-16)
            normalize: Whether to normalize output
        
        Returns:
            Bit-crushed audio
        """
        if bits >= 16:
            return audio
        
        # Quantize to target bit depth
        levels = 2 ** bits
        crushed = np.floor(audio * levels + 0.5) / levels
        
        if normalize:
            return self._normalize(crushed)
        return crushed
    
    def sample_reduce(self, audio: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Sample rate reduction - creates digital aliasing artifacts.
        
        Args:
            audio: Input audio signal
            factor: Reduction factor (2 = half sample rate effect)
        
        Returns:
            Sample-reduced audio
        """
        if factor <= 1:
            return audio
        
        # Downsample
        reduced = audio[::factor]
        
        # Replicate samples to restore length
        output = np.zeros_like(audio)
        output[:len(reduced)*factor:factor] = reduced
        
        # Smooth the transitions slightly
        b, a = butter(2, 0.9 / factor, 'low')
        output = signal.lfilter(b, a, output)
        
        return self._normalize(output)
    
    def decimate(self, audio: np.ndarray, rate: int = 4) -> np.ndarray:
        """
        Decimation - raw sample dropping without smoothing.
        Creates harsh, digital aliasing.
        
        Args:
            audio: Input audio signal
            rate: Sample drop rate
        
        Returns:
            Decimated audio
        """
        if rate <= 1:
            return audio
        
        # Keep only every nth sample
        output = np.zeros_like(audio)
        output[::rate] = audio[::rate]
        
        # Zero-hold interpolation
        for i in range(rate):
            output[i::rate] = output[i::rate]
        
        return self._normalize(output)
    
    # ==================== RING MODULATION ====================
    
    def ring_modulate(self, audio: np.ndarray, freq: float = 440.0,
                      mix: float = 0.5) -> np.ndarray:
        """
        Ring modulation - multiply signal with sine wave.
        Creates metallic, bell-like tones.
        
        Args:
            audio: Input audio signal
            freq: Modulator frequency in Hz
            mix: Wet/dry mix (0=dry, 1=full ring mod)
        
        Returns:
            Ring-modulated audio
        """
        # Generate modulator
        t = np.arange(len(audio)) / self.sample_rate
        modulator = np.sin(2 * np.pi * freq * t)
        
        # Ring modulation = audio * modulator
        modulated = audio * modulator
        
        return self._normalize(audio * (1 - mix) + modulated * mix)
    
    def ring_mod_amp(self, audio: np.ndarray, mod_freq: float = 440.0,
                     mod_depth: float = 1.0) -> np.ndarray:
        """
        Amplitude ring modulation with configurable depth.
        
        Args:
            audio: Input audio signal
            mod_freq: Modulator frequency
            mod_depth: Modulation depth (0-1)
        
        Returns:
            Ring-modulated audio
        """
        t = np.arange(len(audio)) / self.sample_rate
        modulator = (1 - mod_depth) + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        
        return self._normalize(audio * modulator)
    
    def ring_mod_fm(self, audio: np.ndarray, carrier_freq: float = 800.0,
                    mod_index: float = 1.0) -> np.ndarray:
        """
        Combined FM and ring modulation - complex spectral content.
        
        Args:
            audio: Input audio signal
            carrier_freq: Carrier frequency
            mod_index: FM modulation index
        
        Returns:
            FM-ring modulated audio
        """
        t = np.arange(len(audio)) / self.sample_rate
        
        # FM component
        fm_mod = mod_index * np.sin(2 * np.pi * carrier_freq * t * 0.1)
        fm_signal = np.sin(2 * np.pi * carrier_freq * t + fm_mod)
        
        # Ring modulation with FM
        return self._normalize(audio * fm_signal)
    
    # ==================== ASYMMETRIC DISTORTION ====================
    
    def asymmetric_distort(self, audio: np.ndarray, pos_drive: float = 2.0,
                           neg_drive: float = 1.0) -> np.ndarray:
        """
        Asymmetric distortion - different clipping for positive/negative.
        Creates even harmonics and adds character.
        
        Args:
            audio: Input audio signal
            pos_drive: Drive amount for positive half-cycle
            neg_drive: Drive amount for negative half-cycle
        
        Returns:
            Asymmetrically distorted audio
        """
        positive = np.where(
            audio > 0,
            np.tanh(audio * pos_drive),
            audio
        )
        negative = np.where(
            audio <= 0,
            np.tanh(audio * neg_drive),
            positive
        )
        return self._normalize(negative)
    
    def rectify(self, audio: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """
        Rectification distortion - fold signal at zero crossing.
        
        Args:
            audio: Input audio signal
            amount: Rectification amount (0=none, 1=full)
        
        Returns:
            Rectified audio
        """
        # Full wave rectification
        abs_audio = np.abs(audio)
        
        # Blend between original and rectified
        return self._normalize(audio * (1 - amount) + abs_audio * amount)
    
    # ==================== MULTIBAND DISTORTION ====================
    
    def multiband_distort(self, audio: np.ndarray, drive_low: float = 1.5,
                          drive_mid: float = 2.0, drive_high: float = 1.0,
                          crossover_low: float = 200,
                          crossover_high: float = 2000) -> np.ndarray:
        """
        Multiband distortion - different drive per frequency band.
        
        Args:
            audio: Input audio signal
            drive_low: Drive for low frequencies
            drive_mid: Drive for mid frequencies
            drive_high: Drive for high frequencies
            crossover_low: Low/mid crossover frequency
            crossover_high: Mid/high crossover frequency
        
        Returns:
            Multiband distorted audio
        """
        # Create filters
        nyq = self.sample_rate / 2
        
        # Low band
        b_low, a_low = butter(4, crossover_low / nyq, 'low')
        low = signal.lfilter(b_low, a_low, audio)
        low = np.tanh(low * drive_low)
        
        # Mid band
        b_mid, a_mid = butter(4, [crossover_low / nyq, crossover_high / nyq], 'bandpass')
        mid = signal.lfilter(b_mid, a_mid, audio)
        mid = np.tanh(mid * drive_mid)
        
        # High band
        b_high, a_high = butter(4, crossover_high / nyq, 'high')
        high = signal.lfilter(b_high, a_high, audio)
        high = np.tanh(high * drive_high)
        
        # Recombine
        return self._normalize(low + mid + high)
    
    # ==================== SPECIAL EFFECTS ====================
    
    def grunge(self, audio: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Grunge distortion - harsh, noisy, aggressive.
        Combines bit crushing with harsh filtering.
        
        Args:
            audio: Input audio signal
            intensity: Effect intensity (0-1)
        
        Returns:
            Grunge-distorted audio
        """
        # Bit crush
        bits = int(8 - intensity * 6)  # 8 to 2 bits
        crushed = self.bitcrush(audio, bits, normalize=False)
        
        # Harsh highpass
        b, a = butter(4, 100 / (self.sample_rate / 2), 'high')
        filtered = signal.lfilter(b, a, crushed)
        
        # Add noise
        noise = np.random.randn(len(audio)) * intensity * 0.1
        
        return self._normalize(filtered + noise)
    
    def saturn(self, audio: np.ndarray, drive: float = 0.5,
               freq: float = 200) -> np.ndarray:
        """
        Saturn-style ring mod distortion.
        Subtle ring modulation with drive for saturation.
        
        Args:
            audio: Input audio signal
            drive: Saturation drive
            freq: Ring mod frequency
        
        Returns:
            Saturn-distorted audio
        """
        # First apply soft clipping
        saturated = self.soft_clip_tanh(audio, drive + 1)
        
        # Then ring modulate
        return self.ring_modulate(saturated, freq, mix=drive * 0.3)
    
    def rat(self, audio: np.ndarray, gain: float = 0.7, tone: float = 0.5) -> np.ndarray:
        """
        RAT-style pedal distortion.
        Classic guitar pedal distortion character.
        
        Args:
            audio: Input audio signal
            gain: Distortion gain
            tone: Tone control (0=dark, 1=bright)
        
        Returns:
            RAT-style distorted audio
        """
        # Heavy clipping
        driven = audio * (1 + gain * 50)
        clipped = np.tanh(driven)
        
        # Tone control (lowpass filter)
        cutoff = 500 + tone * 7000
        b, a = butter(2, cutoff / (self.sample_rate / 2), 'low')
        filtered = signal.lfilter(b, a, clipped)
        
        return self._normalize(filtered)
    
    # ==================== CONVENIENCE PRESETS ====================
    
    def overdrive(self, audio: np.ndarray, gain: float = 0.5) -> np.ndarray:
        """
        Classic overdrive - gentle soft clipping.
        
        Args:
            audio: Input audio signal
            gain: Overdrive gain (0-1)
        
        Returns:
            Overdriven audio
        """
        drive = 1 + gain * 4
        return self.soft_clip_tanh(audio, drive)
    
    def distortion(self, audio: np.ndarray, gain: float = 0.5) -> np.ndarray:
        """
        Standard distortion - harder clipping than overdrive.
        
        Args:
            audio: Input audio signal
            gain: Distortion gain (0-1)
        
        Returns:
            Distorted audio
        """
        drive = 1 + gain * 8
        return self.soft_clip_tanh(audio, drive)
    
    def metal(self, audio: np.ndarray, gain: float = 0.5) -> np.ndarray:
        """
        Metal distortion - high gain, aggressive.
        
        Args:
            audio: Input audio signal
            gain: Distortion gain (0-1)
        
        Returns:
            Metal-distorted audio
        """
        # Heavy multi-stage clipping
        result = audio * (1 + gain * 20)
        for _ in range(3):
            result = np.tanh(result)
        return self._normalize(result)
    
    def lofi(self, audio: np.ndarray, bits: int = 4, 
             sample_factor: int = 4) -> np.ndarray:
        """
        Lo-fi effect - bit crushing + sample reduction.
        
        Args:
            audio: Input audio signal
            bits: Bit depth
            sample_factor: Sample reduction factor
        
        Returns:
            Lo-fi audio
        """
        crushed = self.bitcrush(audio, bits, normalize=False)
        reduced = self.sample_reduce(crushed, sample_factor)
        return self._normalize(reduced)


def create_distortion_processor(sample_rate: int = 44100) -> Distortion:
    """Factory function to create a Distortion processor."""
    return Distortion(sample_rate=sample_rate)
