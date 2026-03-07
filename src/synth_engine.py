#!/usr/bin/env python3
"""
Synthesizer Engine - Generates synthesizer sounds using numpy/scipy

This module provides a comprehensive synthesizer engine for generating
various synthesizer sounds (lead, pad, bass, pluck, supersaw, etc.)
using digital synthesis techniques.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ADSREnvelope:
    """ADSR (Attack, Decay, Sustain, Release) envelope"""
    attack: float = 0.01    # Seconds
    decay: float = 0.1      # Seconds  
    sustain: float = 0.7    # Level (0-1)
    release: float = 0.3    # Seconds
    
    def get_envelope(self, length: int, sample_rate: int = 44100) -> np.ndarray:
        """Generate envelope curve"""
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)
        
        sustain_samples = length - attack_samples - decay_samples - release_samples
        sustain_samples = max(sustain_samples, 0)
        
        # Build envelope
        envelope = np.zeros(length)
        
        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        if decay_samples > 0:
            end_attack = attack_samples
            end_decay = end_attack + decay_samples
            envelope[end_attack:end_decay] = np.linspace(1, self.sustain, decay_samples)
        
        # Sustain
        if sustain_samples > 0:
            end_decay = attack_samples + decay_samples
            end_sustain = end_decay + sustain_samples
            envelope[end_decay:end_sustain] = self.sustain
        
        # Release
        if release_samples > 0:
            end_sustain = length - release_samples
            envelope[end_sustain:] = np.linspace(self.sustain, 0, release_samples)
        
        return envelope


@dataclass
class SynthParams:
    """Synthesizer parameters"""
    waveform: str = "sawtooth"      # sine, square, sawtooth, triangle
    frequency: float = 440.0         # Hz
    detune: float = 0.0              # Cents
    envelope: ADSREnvelope = None
    filter_cutoff: float = 10000.0   # Hz
    filter_resonance: float = 0.0    # Q
    filter_type: str = "lowpass"     # lowpass, highpass, bandpass
    lfo_rate: float = 0.0            # Hz
    lfo_depth: float = 0.0           # Depth (0-1)
    lfo_target: str = "pitch"         # pitch, filter, amplitude
    reverb_mix: float = 0.0          # Wet/dry (0-1)
    delay_mix: float = 0.0           # Wet/dry (0-1)
    delay_time: float = 0.25          # Seconds
    distortion: float = 0.0           # Drive (0-1)
    volume: float = 0.8               # Master volume


class SynthEngine:
    """Synthesizer engine for generating various synth sounds"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.voices: List[Dict] = []
        
        # Waveform lookup tables
        self._init_waveforms()
    
    def _init_waveforms(self):
        """Initialize waveform lookup tables"""
        self.waveforms = {}
        
        # Generate 4096 samples for each waveform
        samples = 4096
        t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
        
        # Sine wave
        self.waveforms["sine"] = np.sin(t)
        
        # Square wave
        self.waveforms["square"] = np.sign(np.sin(t))
        
        # Sawtooth wave
        self.waveforms["sawtooth"] = 2 * (t / (2 * np.pi)) - 1
        
        # Triangle wave
        self.waveforms["triangle"] = 2 * np.abs(2 * (t / (2 * np.pi)) - 1) - 1
        
        # Add more harmonics for supersaw
        self._init_supersaw()
    
    def _init_supersaw(self):
        """Initialize supersaw (detuned saws)"""
        # Supersaw is 7 detuned sawtooth waves
        detunes = [-7, -5, -3, -1, 1, 3, 5, 7]  # Cents
        supersaw = np.zeros(4096)
        
        for detune in detunes:
            freq_multiplier = 2 ** (detune / 1200)
            t = np.linspace(0, 2 * np.pi * freq_multiplier, 4096, endpoint=False)
            supersaw += (2 * (t / (2 * np.pi)) - 1)
        
        self.waveforms["supersaw"] = supersaw / len(detunes)
    
    def _get_waveform(self, waveform: str, length: int) -> np.ndarray:
        """Generate waveform at given length"""
        if waveform == "sine":
            return self._sine_wave(length)
        elif waveform == "square":
            return self._square_wave(length)
        elif waveform == "sawtooth":
            return self._sawtooth_wave(length)
        elif waveform == "triangle":
            return self._triangle_wave(length)
        elif waveform == "supersaw":
            return self._supersaw_wave(length)
        else:
            return self._sine_wave(length)
    
    def _sine_wave(self, length: int, freq: float = 440.0) -> np.ndarray:
        """Generate sine wave"""
        t = np.arange(length) / self.sample_rate
        return np.sin(2 * np.pi * freq * t)
    
    def _square_wave(self, length: int, freq: float = 440.0) -> np.ndarray:
        """Generate square wave"""
        t = np.arange(length) / self.sample_rate
        return np.sign(np.sin(2 * np.pi * freq * t))
    
    def _sawtooth_wave(self, length: int, freq: float = 440.0) -> np.ndarray:
        """Generate sawtooth wave"""
        t = np.arange(length) / self.sample_rate
        return 2 * (freq * t - np.floor(0.5 + freq * t))
    
    def _triangle_wave(self, length: int, freq: float = 440.0) -> np.ndarray:
        """Generate triangle wave"""
        t = np.arange(length) / self.sample_rate
        return 2 * np.abs(2 * (freq * t - np.floor(0.5 + freq * t))) - 1
    
    def _supersaw_wave(self, length: int, freq: float = 440.0) -> np.ndarray:
        """Generate supersaw (7 detuned saws)"""
        detunes = [-7, -5, -3, -1, 1, 3, 5, 7]
        result = np.zeros(length)
        
        for detune in detunes:
            freq_detuned = freq * (2 ** (detune / 1200))
            result += self._sawtooth_wave(length, freq_detuned)
        
        return result / len(detunes)
    
    def _apply_envelope(self, signal: np.ndarray, envelope: ADSREnvelope) -> np.ndarray:
        """Apply ADSR envelope to signal"""
        if envelope is None:
            envelope = ADSREnvelope()
        
        length = len(signal)
        env = envelope.get_envelope(length, self.sample_rate)
        
        return signal * env
    
    def _apply_filter(self, signal: np.ndarray, cutoff: float, 
                     resonance: float, filter_type: str) -> np.ndarray:
        """Apply simple low-pass filter (simplified biquad)"""
        # Simple one-pole filter implementation
        dt = 1.0 / self.sample_rate
        rc = 1.0 / (2 * np.pi * cutoff)
        alpha = dt / (rc + dt)
        
        filtered = np.zeros_like(signal)
        
        if filter_type == "lowpass":
            for i in range(1, len(signal)):
                filtered[i] = filtered[i-1] + alpha * (signal[i] - filtered[i-1])
        
        elif filter_type == "highpass":
            for i in range(1, len(signal)):
                filtered[i] = alpha * (filtered[i-1] + signal[i] - signal[i-1])
        
        else:  # bandpass - simplified
            # First lowpass then highpass
            temp = filtered.copy()
            for i in range(1, len(signal)):
                temp[i] = temp[i-1] + alpha * (signal[i] - temp[i-1])
            
            # Now simple differentiator for highpass effect
            for i in range(1, len(signal)):
                filtered[i] = alpha * (filtered[i-1] + temp[i] - temp[i-1])
        
        # Apply resonance (simplified - boost around cutoff)
        if resonance > 0:
            # Simple resonance simulation
            q_factor = 1.0 / (0.001 + resonance * 10)
            # Very simplified - just a gentle boost
            filtered *= (1 + resonance * 0.5)
        
        return filtered
    
    def _apply_lfo(self, signal: np.ndarray, rate: float, depth: float,
                   target: str, base_value: float = 440.0) -> np.ndarray:
        """Apply LFO modulation"""
        if rate <= 0 or depth <= 0:
            return signal
        
        length = len(signal)
        t = np.arange(length) / self.sample_rate
        
        # Generate LFO
        lfo = np.sin(2 * np.pi * rate * t) * depth
        
        if target == "pitch":
            # Modulate frequency (vibrato)
            freq_modulated = base_value * (1 + lfo * 0.02)  # Small pitch variation
            result = np.zeros(length)
            phase = 0
            for i in range(length):
                phase += 2 * np.pi * freq_modulated[i] / self.sample_rate
                result[i] = np.sin(phase)
            return result
        
        elif target == "filter":
            # Modulate filter cutoff
            cutoff_modulated = base_value * (1 + lfo * 0.5)
            return self._apply_filter(signal, cutoff_modulated.mean(), 0, "lowpass")
        
        else:  # amplitude
            # Modulate amplitude (tremolo)
            return signal * (1 + lfo * 0.3)
    
    def _apply_delay(self, signal: np.ndarray, mix: float, 
                    delay_time: float) -> np.ndarray:
        """Apply delay effect"""
        if mix <= 0:
            return signal
        
        delay_samples = int(delay_time * self.sample_rate)
        output = signal.copy()
        
        # Simple delay line
        for i in range(delay_samples, len(signal)):
            output[i] += output[i - delay_samples] * mix * 0.5
        
        # Mix wet/dry
        return signal * (1 - mix) + output * mix
    
    def _apply_reverb(self, signal: np.ndarray, mix: float) -> np.ndarray:
        """Apply simple reverb (schroeder algorithm simplified)"""
        if mix <= 0:
            return signal
        
        # Simple comb filter reverb
        delays = [1557, 1617, 1491, 1422, 1277, 1356]  # Sample delays
        decay = 0.7
        
        output = signal.copy()
        
        for delay in delays:
            delayed = np.zeros_like(signal)
            delayed[delay:] = signal[:-delay]
            output += delayed * decay
        
        # Mix wet/dry
        output /= (1 + len(delays) * decay)
        return signal * (1 - mix) + output * mix
    
    def _apply_distortion(self, signal: np.ndarray, drive: float) -> np.ndarray:
        """Apply distortion/overdrive"""
        if drive <= 0:
            return signal
        
        # Soft clipping distortion
        gain = 1 + drive * 10
        
        # Tanh soft clipping
        distorted = np.tanh(signal * gain)
        
        # Mix based on drive
        return signal * (1 - drive) + distorted * drive
    
    def generate(self, params: SynthParams, duration: float = 1.0) -> np.ndarray:
        """Generate synthesizer sound"""
        length = int(duration * self.sample_rate)
        
        # Get base waveform
        signal = self._get_waveform(params.waveform, length)
        
        # Apply detune
        if params.detune != 0:
            freq_multiplier = 2 ** (params.detune / 1200)
            # Resample for detune (simplified)
            indices = np.arange(length) * freq_multiplier
            signal = np.interp(np.arange(length), indices, signal)
        
        # Apply envelope
        signal = self._apply_envelope(signal, params.envelope)
        
        # Apply filter
        signal = self._apply_filter(signal, params.filter_cutoff, 
                                     params.filter_resonance, params.filter_type)
        
        # Apply LFO
        if params.lfo_rate > 0:
            signal = self._apply_lfo(signal, params.lfo_rate, params.lfo_depth,
                                     params.lfo_target, params.filter_cutoff)
        
        # Apply effects (in order)
        if params.distortion > 0:
            signal = self._apply_distortion(signal, params.distortion)
        
        if params.delay_mix > 0:
            signal = self._apply_delay(signal, params.delay_mix, params.delay_time)
        
        if params.reverb_mix > 0:
            signal = self._apply_reverb(signal, params.reverb_mix)
        
        # Apply master volume
        signal *= params.volume
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.9
        
        return signal
    
    def generate_lead(self, note: str = "C4", duration: float = 1.0,
                      octave: int = 4) -> np.ndarray:
        """Generate lead synthesizer sound"""
        freq = self._note_to_freq(note, octave)
        
        params = SynthParams(
            waveform="sawtooth",
            frequency=freq,
            envelope=ADSREnvelope(attack=0.01, decay=0.1, sustain=0.7, release=0.2),
            filter_cutoff=8000,
            filter_type="lowpass",
            lfo_rate=5,
            lfo_depth=0.1,
            lfo_target="filter",
            volume=0.7
        )
        
        return self.generate(params, duration)
    
    def generate_pad(self, note: str = "C3", duration: float = 2.0,
                     octave: int = 3) -> np.ndarray:
        """Generate pad synthesizer sound"""
        freq = self._note_to_freq(note, octave)
        
        params = SynthParams(
            waveform="sine",
            frequency=freq,
            envelope=ADSREnvelope(attack=0.5, decay=0.3, sustain=0.8, release=0.8),
            filter_cutoff=4000,
            filter_type="lowpass",
            lfo_rate=0.5,
            lfo_depth=0.2,
            lfo_target="amplitude",
            reverb_mix=0.4,
            volume=0.6
        )
        
        return self.generate(params, duration)
    
    def generate_bass(self, note: str = "C2", duration: float = 0.5,
                      octave: int = 1) -> np.ndarray:
        """Generate bass synthesizer sound"""
        freq = self._note_to_freq(note, octave)
        
        params = SynthParams(
            waveform="square",
            frequency=freq,
            envelope=ADSREnvelope(attack=0.001, decay=0.1, sustain=0.8, release=0.1),
            filter_cutoff=800,
            filter_type="lowpass",
            filter_resonance=0.3,
            distortion=0.2,
            volume=0.8
        )
        
        return self.generate(params, duration)
    
    def generate_pluck(self, note: str = "C4", duration: float = 0.3,
                       octave: int = 4) -> np.ndarray:
        """Generate pluck/arp sound"""
        freq = self._note_to_freq(note, octave)
        
        params = SynthParams(
            waveform="triangle",
            frequency=freq,
            envelope=ADSREnvelope(attack=0.001, decay=0.2, sustain=0.0, release=0.1),
            filter_cutoff=6000,
            filter_type="lowpass",
            volume=0.7
        )
        
        return self.generate(params, duration)
    
    def generate_supersaw(self, note: str = "C4", duration: float = 1.0,
                          octave: int = 4) -> np.ndarray:
        """Generate supersaw lead sound"""
        freq = self._note_to_freq(note, octave)
        
        params = SynthParams(
            waveform="supersaw",
            frequency=freq,
            detune=10,
            envelope=ADSREnvelope(attack=0.05, decay=0.2, sustain=0.6, release=0.3),
            filter_cutoff=6000,
            filter_type="lowpass",
            lfo_rate=4,
            lfo_depth=0.3,
            lfo_target="filter",
            reverb_mix=0.2,
            volume=0.7
        )
        
        return self.generate(params, duration)
    
    def generate_arp(self, notes: List[str], duration: float = 1.0,
                     pattern: str = "up") -> np.ndarray:
        """Generate arpeggiated pattern"""
        freqs = [self._note_to_freq(n, 4) for n in notes]
        
        if pattern == "up":
            seq = freqs
        elif pattern == "down":
            seq = freqs[::-1]
        elif pattern == "updown":
            seq = freqs + freqs[-2:0:-1]
        else:
            seq = freqs
        
        # Calculate samples per note
        samples_per_note = int((duration / len(seq)) * self.sample_rate)
        
        result = []
        for freq in seq:
            params = SynthParams(
                waveform="sawtooth",
                frequency=freq,
                envelope=ADSREnvelope(attack=0.001, decay=0.1, sustain=0.5, release=0.05),
                filter_cutoff=5000,
                volume=0.6
            )
            note_signal = self.generate(params, duration / len(seq))
            result.append(note_signal)
        
        return np.concatenate(result)
    
    def generate_stab(self, note: str = "C3", duration: float = 0.2,
                       octave: int = 3) -> np.ndarray:
        """Generate chord stab / one-shot"""
        freq = self._note_to_freq(note, octave)
        
        # Add some harmonics
        params = SynthParams(
            waveform="square",
            frequency=freq,
            envelope=ADSREnvelope(attack=0.001, decay=0.15, sustain=0.3, release=0.1),
            filter_cutoff=3000,
            filter_type="lowpass",
            filter_resonance=0.4,
            reverb_mix=0.3,
            volume=0.75
        )
        
        return self.generate(params, duration)
    
    def _note_to_freq(self, note: str, default_octave: int = 4) -> float:
        """Convert note name to frequency"""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        # Parse note
        note = note.upper()
        if len(note) == 1:
            note_name = note
            octave = default_octave
        elif len(note) >= 2:
            note_name = note[0]
            if len(note) > 1 and note[1].isdigit():
                octave = int(note[1])
            else:
                octave = default_octave
                if len(note) > 1:
                    # Handle sharps
                    if note[1] == "#":
                        note_name = note[:2]
        
        # Find note index
        note_idx = 0
        for i, name in enumerate(note_names):
            if note_name.startswith(name):
                note_idx = i
                break
        
        # Calculate frequency (A4 = 440 Hz)
        semitones = (octave - 4) * 12 + (note_idx - 9)
        return 440.0 * (2 ** (semitones / 12))
    
    def to_wav(self, signal: np.ndarray, filename: str):
        """Save audio to WAV file"""
        try:
            import scipy.io.wavfile as wav
            # Convert to 16-bit PCM
            audio_int16 = (signal * 32767).astype(np.int16)
            wav.write(filename, self.sample_rate, audio_int16)
        except ImportError:
            # Fallback: simple WAV without scipy
            self._simple_wav_write(signal, filename)
    
    def _simple_wav_write(self, signal: np.ndarray, filename: str):
        """Simple WAV writer without scipy"""
        import struct
        
        # Convert to 16-bit
        audio_int16 = (signal * 32767).astype(np.int16)
        
        # WAV header
        channels = 1
        bits_per_sample = 16
        byte_rate = self.sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(audio_int16) * 2
        
        with open(filename, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + data_size))
            f.write(b'WAVE')
            
            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # Chunk size
            f.write(struct.pack('<H', 1))   # Audio format (PCM)
            f.write(struct.pack('<H', channels))
            f.write(struct.pack('<I', self.sample_rate))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))
            
            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(audio_int16.tobytes())


# Pre-defined synth presets
SYNTH_PRESETS = {
    "default_lead": {
        "waveform": "sawtooth",
        "attack": 0.01,
        "decay": 0.1,
        "sustain": 0.7,
        "release": 0.2,
        "filter_cutoff": 8000,
        "filter_resonance": 0.0,
    },
    "soft_pad": {
        "waveform": "sine",
        "attack": 0.5,
        "decay": 0.3,
        "sustain": 0.8,
        "release": 0.8,
        "filter_cutoff": 4000,
        "filter_resonance": 0.0,
    },
    "acid_bass": {
        "waveform": "sawtooth",
        "attack": 0.001,
        "decay": 0.1,
        "sustain": 0.8,
        "release": 0.1,
        "filter_cutoff": 800,
        "filter_resonance": 0.7,
    },
    "plucky_arp": {
        "waveform": "triangle",
        "attack": 0.001,
        "decay": 0.2,
        "sustain": 0.0,
        "release": 0.1,
        "filter_cutoff": 6000,
        "filter_resonance": 0.2,
    },
    "supersaw_lead": {
        "waveform": "supersaw",
        "attack": 0.05,
        "decay": 0.2,
        "sustain": 0.6,
        "release": 0.3,
        "filter_cutoff": 6000,
        "filter_resonance": 0.3,
    },
    "wobble_bass": {
        "waveform": "sawtooth",
        "attack": 0.01,
        "decay": 0.1,
        "sustain": 0.8,
        "release": 0.1,
        "filter_cutoff": 400,
        "filter_resonance": 0.5,
        "lfo_rate": 4,
        "lfo_depth": 0.8,
        "lfo_target": "filter",
    },
}


def main():
    """Test the synth engine"""
    print("🎹 Synthesizer Engine\n")
    
    synth = SynthEngine(sample_rate=44100)
    
    # Generate different synth types
    tests = [
        ("Lead", lambda: synth.generate_lead("C4", 0.5)),
        ("Pad", lambda: synth.generate_pad("C3", 1.0)),
        ("Bass", lambda: synth.generate_bass("C2", 0.3)),
        ("Pluck", lambda: synth.generate_pluck("C4", 0.2)),
        ("Supersaw", lambda: synth.generate_supersaw("C4", 0.5)),
        ("Stab", lambda: synth.generate_stab("C3", 0.15)),
    ]
    
    for name, generator in tests:
        signal = generator()
        print(f"  {name}: {len(signal)} samples ({len(signal)/44100:.2f}s)")
        
        # Optionally save to file
        # synth.to_wav(signal, f"/tmp/synth_{name.lower()}.wav")
    
    print("\n✓ Synth engine ready!")
    
    # Example: Generate an arpeggio
    print("\n🎵 Arpeggio example:")
    arp = synth.generate_arp(["C3", "E3", "G3", "B3"], duration=0.5, pattern="updown")
    print(f"  Arp: {len(arp)} samples ({len(arp)/44100:.2f}s)")


if __name__ == "__main__":
    main()
