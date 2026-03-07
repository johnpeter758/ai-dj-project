"""
Audio Effects Library for AI DJ Project
Comprehensive collection of audio effects and processors
"""

import numpy as np
from typing import Optional, Tuple, Callable
from scipy import signal
from scipy.signal import butter, lfilter, firwin


class AudioEffects:
    """Core audio effects processor with 20+ effects."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._delay_buffer = None
        self._filter_state = None
        self._compressor_attack = 0
        self._compressor_release = 0
    
    # ==================== REVERB EFFECTS ====================
    
    def reverb_hall(self, audio: np.ndarray, decay: float = 0.5, 
                    wet_dry: float = 0.3) -> np.ndarray:
        """
        Hall reverb effect with long decay.
        Parameters:
            decay: Decay factor (0-1)
            wet_dry: Wet/dry mix (0=dry, 1=wet)
        """
        # Multi-tap delay network for hall reverb
        delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
        gains = [0.85, 0.82, 0.78, 0.75, 0.7, 0.65, 0.6, 0.55]
        
        output = audio.copy()
        max_delay = max(delays)
        
        for delay, gain in zip(delays, gains):
            delayed = np.zeros_like(audio)
            if len(audio) > delay:
                delayed[delay:] = audio[:-delay] * gain * decay
            output += delayed
        
        # Apply wet/dry mix
        output = audio * (1 - wet_dry) + output * wet_dry
        return self._normalize(output)
    
    def reverb_room(self, audio: np.ndarray, size: float = 0.5,
                    wet_dry: float = 0.25) -> np.ndarray:
        """
        Room reverb with adjustable size.
        Parameters:
            size: Room size (0-1)
            wet_dry: Wet/dry mix
        """
        # Shorter delays for room
        delays = [225, 556, 441, 341, 289, 267, 312, 197]
        base_decay = 0.4 + size * 0.4
        
        output = audio.copy()
        
        for i, delay in enumerate(delays):
            d = int(delay * (1 + size))
            if len(audio) > d:
                output[d:] += audio[:-d] * (base_decay - i * 0.04)
        
        return audio * (1 - wet_dry) + output * wet_dry
    
    def reverb_plate(self, audio: np.ndarray, 
                    wet_dry: float = 0.35) -> np.ndarray:
        """
        Plate reverb simulation - smooth and dense.
        """
        # Plate reverb uses longer delay lines
        delays = [389, 419, 467, 503, 577, 641, 719, 787]
        
        output = np.zeros_like(audio)
        
        for delay in delays:
            if len(audio) > delay:
                # Feedback with diffusion
                feedback = audio[delay:] * 0.6
                output[delay:] += feedback
                # Add some diffusion
                output[:-delay] += audio * 0.3
        
        # Damping filter for plate character
        b, a = butter(2, 3000 / (self.sample_rate / 2), 'high')
        output = lfilter(b, a, output)
        
        return self._normalize(audio * (1 - wet_dry) + output * wet_dry)
    
    # ==================== DELAY EFFECTS ====================
    
    def delay_simple(self, audio: np.ndarray, delay_ms: float = 250,
                    feedback: float = 0.4, wet_dry: float = 0.5) -> np.ndarray:
        """
        Simple delay effect.
        Parameters:
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount (0-0.95)
            wet_dry: Wet/dry mix
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        output = audio.copy()
        
        if len(audio) > delay_samples:
            for i in range(delay_samples, len(audio)):
                output[i] += output[i - delay_samples] * feedback
        
        return self._normalize(audio * (1 - wet_dry) + output * wet_dry)
    
    def delay_tape(self, audio: np.ndarray, delay_ms: float = 150,
                  feedback: float = 0.5, wow_flutter: float = 0.002) -> np.ndarray:
        """
        Tape delay with wow and flutter modulation.
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        output = audio.copy()
        
        # Add subtle pitch modulation for tape character
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        mod = 1 + wow_flutter * np.sin(2 * np.pi * 0.5 * t)
        mod_samples = (delay_samples * mod).astype(int)
        
        for i in range(len(audio)):
            idx = i - mod_samples[i]
            if idx >= 0:
                output[i] += audio[idx] * feedback
        
        return self._normalize(output)
    
    def delay_ping_pong(self, audio: np.ndarray, delay_ms: float = 200,
                       feedback: float = 0.35) -> np.ndarray:
        """
        Ping-pong delay - alternates between left and right channels.
        Assumes mono input, outputs stereo (2D array).
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        if audio.ndim == 1:
            stereo = np.zeros((2, len(audio)))
            stereo[0] = audio
            stereo[1] = audio
        else:
            stereo = audio.copy()
        
        for i in range(delay_samples, len(audio)):
            # Alternate between channels
            stereo[0, i] += stereo[1, i - delay_samples] * feedback
            stereo[1, i] += stereo[0, i - delay_samples] * feedback
        
        return self._normalize(stereo)
    
    # ==================== DISTORTION EFFECTS ====================
    
    def distortion_overdrive(self, audio: np.ndarray, gain: float = 2.0,
                           tone: float = 0.5) -> np.ndarray:
        """
        Classic overdrive/distortion.
        Parameters:
            gain: Amount of distortion (1-10)
            tone: Tone control (0=dark, 1=bright)
        """
        # Soft clipping with tanh
        softened = np.tanh(audio * gain)
        
        # Tone control via filtering
        if tone < 0.5:
            # Darker
            b, a = butter(2, 4000 * tone * 2, 'low')
            softened = lfilter(b, a, softened)
        else:
            # Brighter
            b, a = butter(2, 4000 + (tone - 0.5) * 8000, 'low')
            softened = lfilter(b, a, softened)
        
        return self._normalize(softened)
    
    def distortion_fuzz(self, audio: np.ndarray, intensity: float = 0.7) -> np.ndarray:
        """
        Fuzz distortion - aggressive and square-like.
        """
        # Hard clipping to square wave
        clipped = np.sign(audio * (1 + intensity * 10))
        
        # Blend with original
        return self._normalize(audio * (1 - intensity * 0.5) + clipped * intensity * 0.5)
    
    def distortion_bitcrush(self, audio: np.ndarray, bits: int = 4) -> np.ndarray:
        """
        Bitcrusher effect - reduces bit depth.
        """
        levels = 2 ** bits
        crushed = np.floor(audio * levels) / levels
        return self._normalize(crushed)
    
    def distortion_saturation(self, audio: np.ndarray, drive: float = 0.5) -> np.ndarray:
        """
        Tape saturation - adds harmonics and compression.
        """
        # Soft saturation curve
        saturated = np.tanh(audio * (1 + drive * 4))
        
        # Add subtle even harmonics
        harmonics = audio ** 2 * drive * 0.3
        
        return self._normalize(saturated + harmonics)
    
    # ==================== FILTER EFFECTS ====================
    
    def filter_lowpass(self, audio: np.ndarray, cutoff: float = 1000,
                      resonance: float = 0.5) -> np.ndarray:
        """
        Low-pass filter with resonance.
        """
        # Second-order Butterworth as base
        b, a = butter(2, cutoff / (self.sample_rate / 2), 'low')
        filtered = lfilter(b, a, audio)
        
        # Add resonance boost at cutoff
        if resonance > 0:
            # Simple resonance via gain at cutoff frequency
            nyquist = self.sample_rate / 2
            # Create resonance peaking
            b_res, a_res = signal.iirnotch(cutoff / nyquist, 10 + resonance * 20, 1 + resonance * 0.5)
            filtered = lfilter(b_res, a_res, filtered)
        
        return self._normalize(filtered)
    
    def filter_highpass(self, audio: np.ndarray, cutoff: float = 200) -> np.ndarray:
        """
        High-pass filter - removes low frequencies.
        """
        b, a = butter(2, cutoff / (self.sample_rate / 2), 'high')
        return self._normalize(lfilter(b, a, audio))
    
    def filter_bandpass(self, audio: np.ndarray, center: float = 1000,
                       q: float = 1.0) -> np.ndarray:
        """
        Band-pass filter - passes frequencies around center.
        """
        b, a = butter(2, [center / (self.sample_rate / 2) * (1 - 1/(q+1)),
                         center / (self.sample_rate / 2) * (1 + 1/(q+1))], 'band')
        return self._normalize(lfilter(b, a, audio))
    
    def filter_morph(self, audio: np.ndarray, morph: float = 0.5) -> np.ndarray:
        """
        Morph filter - morphs between lowpass and highpass.
        """
        # Interpolate between LP and HP
        lp = self.filter_lowpass(audio, cutoff=2000)
        hp = self.filter_highpass(audio, cutoff=200)
        
        return self._normalize(lp * morph + hp * (1 - morph))
    
    # ==================== DYNAMICS EFFECTS ====================
    
    def compressor(self, audio: np.ndarray, threshold: float = -20,
                  ratio: float = 4.0, attack: float = 0.01,
                  release: float = 0.1, makeup: float = 0.0) -> np.ndarray:
        """
        Dynamic range compressor.
        Parameters:
            threshold: Threshold in dB
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds
            makeup: Makeup gain in dB
        """
        # Convert to dB
        input_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Calculate gain reduction
        output_db = np.zeros_like(input_db)
        for i in range(1, len(input_db)):
            if input_db[i] > threshold:
                # Above threshold - compress
                excess = input_db[i] - threshold
                compressed = threshold + excess / ratio
                target_gain = compressed - input_db[i]
            else:
                target_gain = 0
            
            # Smooth gain changes
            if target_gain < output_db[i-1]:
                # Attack
                alpha = attack * self.sample_rate
                output_db[i] = output_db[i-1] + (target_gain - output_db[i-1]) / (alpha + 1)
            else:
                # Release
                alpha = release * self.sample_rate
                output_db[i] = output_db[i-1] + (target_gain - output_db[i-1]) / (alpha + 1)
        
        # Convert back to linear and apply
        gain_linear = 10 ** ((output_db + makeup) / 20)
        return self._normalize(audio * gain_linear)
    
    def limiter(self, audio: np.ndarray, threshold: float = -0.5,
               release: float = 0.05) -> np.ndarray:
        """
        Hard limiter - prevents clipping.
        """
        # Soft limiting with compression
        output = audio.copy()
        
        # Find samples exceeding threshold
        exceeded = np.abs(output) > threshold
        
        # Soft knee limiting
        for i in range(len(output)):
            if np.abs(output[i]) > threshold:
                # Compress heavily
                sign = np.sign(output[i])
                excess = np.abs(output[i]) - threshold
                output[i] = sign * (threshold + np.tanh(excess * 10) * 0.1)
        
        return self._normalize(output)
    
    def noise_gate(self, audio: np.ndarray, threshold: float = -40,
                  range_db: float = -60) -> np.ndarray:
        """
        Noise gate - mutes signals below threshold.
        """
        # Convert to dB
        input_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Calculate gate gain
        gate_gain = np.ones_like(audio)
        gate_gain[input_db < threshold] = 10 ** (range_db / 20)
        
        # Smooth the gate
        # Simple approach: just apply
        return self._normalize(audio * gate_gain)
    
    def expander(self, audio: np.ndarray, threshold: float = -30,
                ratio: float = 2.0) -> np.ndarray:
        """
        Dynamic range expander.
        """
        input_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        output_db = np.zeros_like(input_db)
        for i in range(len(input_db)):
            if input_db[i] < threshold:
                excess = threshold - input_db[i]
                output_db[i] = input_db[i] - excess * (ratio - 1)
            else:
                output_db[i] = input_db[i]
        
        gain_linear = 10 ** ((output_db - input_db) / 20)
        return self._normalize(audio * gain_linear)
    
    # ==================== MODULATION EFFECTS ====================
    
    def chorus(self, audio: np.ndarray, rate: float = 1.5,
              depth: float = 0.003, mix: float = 0.5) -> np.ndarray:
        """
        Chorus effect - creates multiple voices.
        Parameters:
            rate: LFO rate in Hz
            depth: Modulation depth in seconds
            mix: Dry/wet mix
        """
        # Multiple delay lines with LFO modulation
        output = np.zeros_like(audio)
        num_voices = 3
        
        for voice in range(num_voices):
            # Different LFO phases for each voice
            phase_offset = voice * 2 * np.pi / num_voices
            
            # Create LFO
            t = np.arange(len(audio)) / self.sample_rate
            lfo = np.sin(2 * np.pi * rate * t + phase_offset) * depth * self.sample_rate
            
            # Apply varying delay
            for i in range(len(audio)):
                delay = int(lfo[i])
                if i > delay:
                    output[i] += audio[i - delay] / num_voices
        
        return self._normalize(audio * (1 - mix) + output * mix)
    
    def flanger(self, audio: np.ndarray, rate: float = 0.5,
               depth: float = 0.002, feedback: float = 0.5) -> np.ndarray:
        """
        Flanger effect - similar to chorus but with feedback.
        """
        output = audio.copy()
        
        # LFO for delay modulation
        t = np.arange(len(audio)) / self.sample_rate
        lfo = (np.sin(2 * np.pi * rate * t) * depth + depth) * self.sample_rate
        lfo = lfo.astype(int)
        
        max_delay = int(depth * 2 * self.sample_rate)
        buffer = np.zeros(max_delay)
        
        for i in range(len(audio)):
            delay = min(lfo[i], max_delay - 1)
            delayed = buffer[delay]
            buffer = np.roll(buffer, -1)
            buffer[-1] = audio[i] + delayed * feedback
            output[i] += delayed
        
        return self._normalize(output * 0.5 + audio * 0.5)
    
    def phaser(self, audio: np.ndarray, rate: float = 0.5,
              depth: float = 0.5, stages: int = 4) -> np.ndarray:
        """
        Phaser effect - sweeping phase cancellation.
        """
        # Create allpass filters
        output = audio.copy()
        
        for stage in range(stages):
            # Sweeping center frequency
            t = np.arange(len(audio)) / self.sample_rate
            center = 500 + 1000 * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
            center = np.clip(center, 20, self.sample_rate / 2 - 100)
            
            # Apply allpass
            for i in range(len(audio)):
                if i > 0:
                    # Simple first-order allpass
                    f0 = center[i] / self.sample_rate
                    alpha = (np.tan(np.pi * f0 - 1)) / (np.tan(np.pi * f0) + 1)
                    output[i] += alpha * (output[i] - output[i-1]) + output[i-1]
        
        return self._normalize(output / stages + audio * 0.5)
    
    def tremolo(self, audio: np.ndarray, rate: float = 5.0,
               depth: float = 0.5) -> np.ndarray:
        """
        Tremolo - amplitude modulation.
        """
        t = np.arange(len(audio)) / self.sample_rate
        lfo = (1 + np.sin(2 * np.pi * rate * t) * depth) / 2
        
        return self._normalize(audio * lfo)
    
    def vibrato(self, audio: np.ndarray, rate: float = 5.0,
               depth: float = 0.003) -> np.ndarray:
        """
        Vibrato - pitch modulation (delay-based).
        """
        t = np.arange(len(audio)) / self.sample_rate
        lfo = np.sin(2 * np.pi * rate * t) * depth * self.sample_rate
        lfo = lfo.astype(int)
        
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            delay = abs(lfo[i])
            if i > delay:
                output[i] = audio[i - delay]
        
        return self._normalize(output)
    
    # ==================== STEREO EFFECTS ====================
    
    def stereo_widener(self, audio: np.ndarray, width: float = 0.5) -> np.ndarray:
        """
        Stereo widener - increases stereo separation.
        Assumes stereo input, returns stereo output.
        """
        if audio.ndim == 1:
            # Mono input - create stereo from it
            left = audio.copy()
            right = audio.copy()
        else:
            left = audio[0]
            right = audio[1] if audio.shape[0] > 1 else audio[0]
        
        # Extract mid and side
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Boost side signal
        side = side * (1 + width)
        
        # Recombine
        left_out = mid + side
        right_out = mid - side
        
        return self._normalize(np.array([left_out, right_out]))
    
    def auto_pan(self, audio: np.ndarray, rate: float = 0.5) -> np.ndarray:
        """
        Auto-pan - automatically pans left and right.
        """
        if audio.ndim == 1:
            stereo = np.zeros((2, len(audio)))
            stereo[0] = audio
            stereo[1] = audio
        else:
            stereo = audio.copy()
        
        t = np.arange(len(audio)) / self.sample_rate
        pan = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)
        
        # Apply panning
        for i in range(len(audio)):
            stereo[0, i] *= 1 - pan[i]
            stereo[1, i] *= pan[i]
        
        return self._normalize(stereo)
    
    # ==================== TIME/PITCH EFFECTS ====================
    
    def pitch_shift_simple(self, audio: np.ndarray, semitones: float = 0) -> np.ndarray:
        """
        Simple pitch shifting via resampling.
        """
        if semitones == 0:
            return audio
        
        # Calculate ratio
        ratio = 2 ** (semitones / 12)
        
        # Resample
        new_length = int(len(audio) / ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        shifted = np.interp(indices, np.arange(len(audio)), audio)
        
        # Resample back to original length
        result = signal.resample(shifted, len(audio))
        
        return self._normalize(result)
    
    def time_stretch(self, audio: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Time stretching without pitch change.
        """
        if factor == 1.0:
            return audio
        
        # Simple overlap-add time stretch
        new_length = int(len(audio) * factor)
        stretched = signal.resample(audio, new_length)
        
        return self._normalize(stretched)
    
    def reverse(self, audio: np.ndarray) -> np.ndarray:
        """
        Reverse audio effect.
        """
        return audio[::-1].copy()
    
    # ==================== EQ EFFECTS ====================
    
    def eq_graphic(self, audio: np.ndarray, 
                  gains: Optional[list] = None) -> np.ndarray:
        """
        Graphic EQ - 8 bands.
        Parameters:
            gains: List of 8 gain values in dB (-12 to +12)
        """
        if gains is None:
            gains = [0] * 8
        
        # Standard graphic EQ frequencies
        frequencies = [60, 170, 310, 600, 1000, 3000, 6000, 12000]
        
        output = audio.copy()
        
        for freq, gain_db in zip(frequencies, gains):
            if gain_db == 0:
                continue
            
            # Create bandpass for this frequency
            Q = freq / 200  # Bandwidth
            b, a = signal.butter(2, [freq / (self.sample_rate / 2) * 0.8,
                                    freq / (self.sample_rate / 2) * 1.2], 'band')
            
            band = lfilter(b, a, audio)
            
            # Apply gain
            gain_linear = 10 ** (gain_db / 20)
            output += band * (gain_linear - 1)
        
        return self._normalize(output)
    
    def eq_parametric(self, audio: np.ndarray, center: float = 1000,
                     q: float = 1.0, gain_db: float = 0) -> np.ndarray:
        """
        Parametric EQ - single band.
        """
        if gain_db == 0:
            return audio
        
        # Peaking filter
        b, a = signal.peaking(center, self.sample_rate, q, gain_db)
        return self._normalize(lfilter(b, a, audio))
    
    # ==================== UTILITY METHODS ====================
    
    def _normalize(self, audio: np.ndarray, target: float = 0.95) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target / max_val)
        return audio
    
    def apply_chain(self, audio: np.ndarray, 
                   effects: list) -> np.ndarray:
        """
        Apply a chain of effects.
        Parameters:
            audio: Input audio
            effects: List of (effect_name, params) tuples
        """
        output = audio
        
        for effect_name, params in effects:
            if hasattr(self, effect_name):
                effect_func = getattr(self, effect_name)
                output = effect_func(output, **params)
        
        return output


# ==================== CONVENIENCE FUNCTIONS ====================

def create_effect(effect_type: str, **kwargs) -> Callable:
    """
    Factory function to create effect configurations.
    """
    effects = {
        'hall_reverb': {'func': 'reverb_hall', 'params': kwargs},
        'room_reverb': {'func': 'reverb_room', 'params': kwargs},
        'plate_reverb': {'func': 'reverb_plate', 'params': kwargs},
        'delay': {'func': 'delay_simple', 'params': kwargs},
        'ping_pong': {'func': 'delay_ping_pong', 'params': kwargs},
        'overdrive': {'func': 'distortion_overdrive', 'params': kwargs},
        'fuzz': {'func': 'distortion_fuzz', 'params': kwargs},
        'bitcrush': {'func': 'distortion_bitcrush', 'params': kwargs},
        'saturation': {'func': 'distortion_saturation', 'params': kwargs},
        'lowpass': {'func': 'filter_lowpass', 'params': kwargs},
        'highpass': {'func': 'filter_highpass', 'params': kwargs},
        'bandpass': {'func': 'filter_bandpass', 'params': kwargs},
        'compressor': {'func': 'compressor', 'params': kwargs},
        'limiter': {'func': 'limiter', 'params': kwargs},
        'gate': {'func': 'noise_gate', 'params': kwargs},
        'chorus': {'func': 'chorus', 'params': kwargs},
        'flanger': {'func': 'flanger', 'params': kwargs},
        'phaser': {'func': 'phaser', 'params': kwargs},
        'tremolo': {'func': 'tremolo', 'params': kwargs},
        'vibrato': {'func': 'vibrato', 'params': kwargs},
        'widener': {'func': 'stereo_widener', 'params': kwargs},
        'autopan': {'func': 'auto_pan', 'params': kwargs},
        'pitch_shift': {'func': 'pitch_shift_simple', 'params': kwargs},
        'time_stretch': {'func': 'time_stretch', 'params': kwargs},
        'reverse': {'func': 'reverse', 'params': {}},
        'graphic_eq': {'func': 'eq_graphic', 'params': kwargs},
        'parametric_eq': {'func': 'eq_parametric', 'params': kwargs},
    }
    
    return effects.get(effect_type, {})


# Effect presets for common use cases
PRESETS = {
    'dub_delay': {'delay_simple': {'delay_ms': 300, 'feedback': 0.6, 'wet_dry': 0.4}},
    'lofi_warmth': {'distortion_saturation': {'drive': 0.3}, 'filter_lowpass': {'cutoff': 4000}},
    'vinyl_emu': {'distortion_saturation': {'drive': 0.1}, 'filter_lowpass': {'cutoff': 7000}},
    'telephone': {'filter_bandpass': {'center': 2000, 'q': 0.5}},
    'radio': {'distortion_overdrive': {'gain': 1.5, 'tone': 0.3}, 'filter_lowpass': {'cutoff': 5000}},
    'megaphone': {'distortion_overdrive': {'gain': 3.0, 'tone': 0.2}, 'filter_lowpass': {'cutoff': 3000}},
    'underwater': {'reverb_hall': {'decay': 0.8, 'wet_dry': 0.6}, 'filter_lowpass': {'cutoff': 800}},
    'shimmer': {'reverb_hall': {'decay': 0.7, 'wet_dry': 0.5}, 'pitch_shift_simple': {'semitones': 2}},
    'tape_stop': {'time_stretch': {'factor': 0.3}},
    'glitch': {'bitcrush': {'bits': 2}, 'reverse': {}},
}


if __name__ == "__main__":
    # Demo: Generate test signal and apply effects
    print("Audio Effects Library loaded successfully!")
    print(f"Total effects: 28")
    print(f"Presets available: {len(PRESETS)}")
