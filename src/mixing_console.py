"""
Mixing Console - Audio Mixing Pipeline
=======================================
A Python audio mixing console with channel strips, EQ, compression,
pan, sends, master bus processing, and mixing workflow.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EQSettings:
    """3-band parametric EQ settings."""
    low_gain: float = 0.0      # dB
    low_freq: float = 100.0    # Hz
    mid_gain: float = 0.0      # dB
    mid_freq: float = 1000.0   # Hz
    mid_q: float = 1.0         # Q factor
    high_gain: float = 0.0     # dB
    high_freq: float = 8000.0 # Hz


@dataclass
class CompressorSettings:
    """Dynamic range compressor settings."""
    threshold: float = -20.0   # dB
    ratio: float = 4.0         # compression ratio
    attack: float = 10.0       # ms
    release: float = 100.0     # ms
    makeup_gain: float = 0.0  # dB


@dataclass
class ChannelSettings:
    """Complete channel strip settings."""
    gain: float = 0.0         # dB
    pan: float = 0.0          # -1 (left) to 1 (right)
    mute: bool = False
    solo: bool = False
    fader: float = 0.0        # dB (-inf to +6)
    eq: EQSettings = field(default_factory=EQSettings)
    compressor: CompressorSettings = field(default_factory=CompressorSettings)
    sends: Dict[str, float] = field(default_factory=dict)  # send_name -> level (dB)


class BiquadFilter:
    """Biquad filter implementation for EQ."""
    
    def __init__(self, sample_rate: float = 44100.0):
        self.sample_rate = sample_rate
        self.reset()
    
    def reset(self):
        self.x1, self.x2 = 0.0, 0.0
        self.y1, self.y2 = 0.0, 0.0
    
    def process(self, audio: np.ndarray, freq: float, gain_db: float, 
                q: float = 1.0, filter_type: str = 'peaking') -> np.ndarray:
        """Apply biquad filter to audio."""
        w0 = 2.0 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2.0 * q)
        A = 10.0 ** (gain_db / 40.0)
        
        if filter_type == 'peaking':
            b0 = 1.0 + alpha * A
            b1 = -2.0 * np.cos(w0)
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * np.cos(w0)
            a2 = 1.0 - alpha / A
        elif filter_type == 'lowshelf':
            sin_w0 = np.sin(w0)
            cos_w0 = np.cos(w0)
            alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / q - 1.0) + 2.0)
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha
        elif filter_type == 'highshelf':
            sin_w0 = np.sin(w0)
            cos_w0 = np.cos(w0)
            alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / q - 1.0) + 2.0)
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha
        else:
            return audio
        
        # Normalize coefficients
        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0
        
        # Apply filter using direct form II
        output = np.zeros_like(audio)
        for i in range(len(audio)):
            output[i] = b0 * audio[i] + b1 * self.x1 + b2 * self.x2 - a1 * self.y1 - a2 * self.y2
            self.x2, self.x1 = self.x1, audio[i]
            self.y2, self.y1 = self.y1, output[i]
        
        return output


class DynamicsProcessor:
    """Compressor/Limiter for dynamic range control."""
    
    def __init__(self, sample_rate: float = 44100.0, settings: CompressorSettings = None):
        self.sample_rate = sample_rate
        self.settings = settings if settings else CompressorSettings()
        self.envelope = 0.0
    
    def db_to_linear(self, db: float) -> float:
        return 10.0 ** (db / 20.0)
    
    def linear_to_db(self, linear: float) -> float:
        return 20.0 * np.log10(max(linear, 1e-10))
    
    def process(self, audio: np.ndarray, settings: CompressorSettings) -> np.ndarray:
        """Apply compression to audio."""
        threshold_linear = self.db_to_linear(settings.threshold)
        makeup_linear = self.db_to_linear(settings.makeup_gain)
        
        attack_coeff = np.exp(-1.0 / (settings.attack * 1e-3 * self.sample_rate))
        release_coeff = np.exp(-1.0 / (settings.release * 1e-3 * self.sample_rate))
        
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Get input level
            input_level = abs(audio[i])
            
            # Envelope follower
            if input_level > self.envelope:
                self.envelope = attack_coeff * self.envelope + (1 - attack_coeff) * input_level
            else:
                self.envelope = release_coeff * self.envelope + (1 - release_coeff) * input_level
            
            # Compute gain reduction
            if self.envelope > threshold_linear:
                over_threshold = self.envelope - threshold_linear
                compressed = threshold_linear + over_threshold / settings.ratio
                gain = compressed / max(self.envelope, 1e-10)
            else:
                gain = 1.0
            
            # Apply gain with makeup
            output[i] = audio[i] * gain * makeup_linear
        
        return output


class Panner:
    """Stereo panner using constant power panning."""
    
    def process(self, audio: np.ndarray, pan: float) -> np.ndarray:
        """Pan mono audio to stereo."""
        # Constant power panning law
        pan_rad = (pan + 1.0) * np.pi / 4.0  # -1 to 1 -> 0 to pi/2
        left_gain = np.cos(pan_rad)
        right_gain = np.sin(pan_rad)
        
        # Return stereo array
        return np.stack([audio * left_gain, audio * right_gain], axis=-1)


class Send:
    """Aux send for routing to bus/aux."""
    
    def __init__(self, name: str = "Aux"):
        self.name = name
        self.buffer: Optional[np.ndarray] = None
    
    def process(self, audio: np.ndarray, level_db: float) -> np.ndarray:
        """Route audio to send at specified level."""
        level_linear = 10.0 ** (level_db / 20.0)
        return audio * level_linear


class ChannelStrip:
    """Complete channel strip with gain, EQ, compressor, pan, and sends."""
    
    def __init__(self, channel_id: int, name: str = "Channel", 
                 sample_rate: float = 44100.0):
        self.channel_id = channel_id
        self.name = name
        self.sample_rate = sample_rate
        
        self.settings = ChannelSettings()
        self.eq = BiquadFilter(sample_rate)
        self.compressor = DynamicsProcessor(sample_rate)
        self.panner = Panner()
        self.sends: Dict[str, Send] = {}
    
    def add_send(self, name: str) -> Send:
        """Add an aux send to this channel."""
        send = Send(name)
        self.sends[name] = send
        self.settings.sends[name] = -60.0  # Start at -inf dB
        return send
    
    def set_gain(self, db: float):
        """Set input gain in dB."""
        self.settings.gain = db
    
    def set_pan(self, value: float):
        """Set pan value (-1 to 1)."""
        self.settings.pan = np.clip(value, -1.0, 1.0)
    
    def set_mute(self, muted: bool):
        """Set mute state."""
        self.settings.mute = muted
    
    def set_solo(self, solo: bool):
        """Set solo state."""
        self.settings.solo = solo
    
    def set_fader(self, db: float):
        """Set channel fader in dB."""
        self.settings.fader = np.clip(db, -60.0, 6.0)
    
    def set_eq(self, low_gain: float = None, mid_gain: float = None, 
               high_gain: float = None):
        """Set EQ gains in dB."""
        if low_gain is not None:
            self.settings.eq.low_gain = np.clip(low_gain, -12.0, 12.0)
        if mid_gain is not None:
            self.settings.eq.mid_gain = np.clip(mid_gain, -12.0, 12.0)
        if high_gain is not None:
            self.settings.eq.high_gain = np.clip(high_gain, -12.0, 12.0)
    
    def set_compressor(self, threshold: float = None, ratio: float = None,
                       attack: float = None, release: float = None,
                       makeup: float = None):
        """Set compressor parameters."""
        if threshold is not None:
            self.settings.compressor.threshold = threshold
        if ratio is not None:
            self.settings.compressor.ratio = max(1.0, ratio)
        if attack is not None:
            self.settings.compressor.attack = max(0.1, attack)
        if release is not None:
            self.settings.compressor.release = max(10.0, release)
        if makeup is not None:
            self.settings.compressor.makeup_gain = makeup
    
    def process(self, audio: np.ndarray) -> tuple:
        """
        Process audio through the channel strip.
        Returns: (main_output [stereo], send_outputs [dict])
        """
        # Ensure mono input
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
        # 1. Input gain
        gain_linear = 10.0 ** (self.settings.gain / 20.0)
        audio = audio * gain_linear
        
        # 2. EQ processing
        if self.settings.eq.low_gain != 0:
            audio = self.eq.process(audio, self.settings.eq.low_freq,
                                    self.settings.eq.low_gain, 0.707, 'lowshelf')
        if self.settings.eq.mid_gain != 0:
            audio = self.eq.process(audio, self.settings.eq.mid_freq,
                                    self.settings.eq.mid_gain, 
                                    self.settings.eq.mid_q, 'peaking')
        if self.settings.eq.high_gain != 0:
            audio = self.eq.process(audio, self.settings.eq.high_freq,
                                    self.settings.eq.high_gain, 0.707, 'highshelf')
        
        # 3. Compression
        audio = self.compressor.process(audio, self.settings.compressor)
        
        # 4. Apply mute
        if self.settings.mute:
            audio = np.zeros_like(audio)
        
        # 5. Fader
        fader_linear = 10.0 ** (self.settings.fader / 20.0)
        audio = audio * fader_linear
        
        # 6. Panning (to stereo)
        main_output = self.panner.process(audio, self.settings.pan)
        
        # 7. Process sends
        send_outputs = {}
        for send_name, send in self.sends.items():
            level = self.settings.sends.get(send_name, -60.0)
            send_outputs[send_name] = send.process(audio, level)
        
        return main_output, send_outputs


class MasterBus:
    """Master bus with summing and master processing."""
    
    def __init__(self, name: str = "Master", sample_rate: float = 44100.0):
        self.name = name
        self.sample_rate = sample_rate
        
        # Master processing
        self.eq = BiquadFilter(sample_rate)
        self.compressor = DynamicsProcessor(sample_rate)
        self.limiter_threshold = -0.5  # dB
        self.output_gain: float = 0.0
        
        # Bus state
        self.channels: List[ChannelStrip] = []
        self.aux_buses: Dict[str, np.ndarray] = {}
        self.solo_active: bool = False
    
    def add_channel(self, channel: ChannelStrip):
        """Add a channel to the master bus."""
        self.channels.append(channel)
    
    def create_aux_bus(self, name: str):
        """Create an aux bus (for effects returns)."""
        self.aux_buses[name] = np.zeros(0)  # Will be sized on first use
    
    def set_master_eq(self, low_gain: float = None, mid_gain: float = None,
                      high_gain: float = None):
        """Set master bus EQ."""
        # Would store and apply similar to channel EQ
        pass
    
    def set_master_compressor(self, threshold: float = -6.0, ratio: float = 2.0,
                              attack: float = 10.0, release: float = 100.0,
                              makeup: float = 0.0):
        """Set master bus compressor."""
        self.compressor.settings.threshold = threshold
        self.compressor.settings.ratio = ratio
        self.compressor.settings.attack = attack
        self.compressor.settings.release = release
        self.compressor.settings.makeup_gain = makeup
    
    def set_limiter(self, threshold_db: float = -0.5):
        """Set limiter threshold."""
        self.limiter_threshold = threshold_db
    
    def set_output_gain(self, db: float):
        """Set master output gain."""
        self.output_gain = db
    
    def mix(self, audio_length: int) -> np.ndarray:
        """Mix all channels together."""
        # Initialize output buffers
        stereo_output = np.zeros((audio_length, 2))
        
        # Check for soloed channels
        soloed = [ch for ch in self.channels if ch.settings.solo]
        if soloed:
            self.solo_active = True
            channels_to_mix = soloed
        else:
            self.solo_active = False
            channels_to_mix = [ch for ch in self.channels if not ch.settings.mute]
        
        # Mix channels
        for channel in channels_to_mix:
            # Create dummy audio for processing (in real use, would be actual audio)
            dummy_audio = np.random.randn(audio_length) * 0.01
            ch_output, ch_sends = channel.process(dummy_audio)
            stereo_output += ch_output
        
        # Sum aux buses
        for aux_name, aux_audio in self.aux_buses.items():
            if len(aux_audio) == audio_length:
                stereo_output += np.stack([aux_audio, aux_audio], axis=-1)
        
        # Master processing - EQ
        # (Would apply EQ to stereo here)
        
        # Master processing - Compressor (applied to stereo sum)
        # Convert to mono for gain detection, then apply to both
        mono_sum = np.mean(stereo_output, axis=-1)
        compressed_mono = self.compressor.process(mono_sum, self.compressor.settings)
        
        # Re-stereoize with same gain reduction
        envelope_ratio = np.divide(compressed_mono, mono_sum + 1e-10,
                                    out=np.ones_like(mono_sum),
                                    where=mono_sum != 0)
        stereo_output = stereo_output * envelope_ratio[:, np.newaxis]
        
        # Limiter (brick wall)
        max_sample = np.max(np.abs(stereo_output))
        if max_sample > 10.0 ** (self.limiter_threshold / 20.0):
            # Soft clip/limiter
            stereo_output = np.tanh(stereo_output)
        
        # Output gain
        output_linear = 10.0 ** (self.output_gain / 20.0)
        stereo_output *= output_linear
        
        return stereo_output


class MixingConsole:
    """
    Complete mixing console with multiple channels and master bus.
    """
    
    def __init__(self, num_channels: int = 8, sample_rate: float = 44100.0):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        
        # Create channels
        self.channels: List[ChannelStrip] = [
            ChannelStrip(i, f"Ch {i+1}", sample_rate) 
            for i in range(num_channels)
        ]
        
        # Create master bus
        self.master = MasterBus("Master", sample_rate)
        
        # Connect channels to master
        for ch in self.channels:
            self.master.add_channel(ch)
        
        # Create default aux buses
        self.aux_buses: Dict[str, np.ndarray] = {
            "Reverb": np.array([]),
            "Delay": np.array([])
        }
    
    def get_channel(self, idx: int) -> Optional[ChannelStrip]:
        """Get channel by index."""
        if 0 <= idx < len(self.channels):
            return self.channels[idx]
        return None
    
    def process_audio(self, audio_inputs: List[np.ndarray]) -> np.ndarray:
        """
        Process multiple audio sources through the mixer.
        
        Args:
            audio_inputs: List of audio arrays (one per channel)
        
        Returns:
            Stereo mixed output
        """
        # Ensure we have enough channels
        while len(audio_inputs) > len(self.channels):
            new_ch = ChannelStrip(len(self.channels), f"Ch {len(self.channels)+1}",
                                   self.sample_rate)
            self.channels.append(new_ch)
            self.master.add_channel(new_ch)
        
        audio_length = max(len(a) for a in audio_inputs) if audio_inputs else 0
        
        # Process each channel
        stereo_output = np.zeros((audio_length, 2))
        
        # Check for solo
        soloed = [ch for ch in self.channels if ch.settings.solo]
        channels_to_mix = soloed if soloed else self.channels
        
        for i, channel in enumerate(channels_to_mix):
            if i < len(audio_inputs) and len(audio_inputs[i]) > 0:
                audio = audio_inputs[i]
            else:
                continue
            
            # Apply channel processing
            ch_output, ch_sends = channel.process(audio)
            
            # Add to mix (skip muted)
            if not channel.settings.mute:
                stereo_output[:len(ch_output)] += ch_output
        
        # Apply master processing
        mono_sum = np.mean(stereo_output, axis=-1)
        
        # Master compressor
        if self.master.compressor.settings.ratio > 1:
            compressed = self.master.compressor.process(
                mono_sum, self.master.compressor.settings
            )
            # Apply gain reduction
            gr = np.divide(compressed, mono_sum + 1e-10,
                          out=np.ones_like(mono_sum),
                          where=mono_sum != 0)
            stereo_output *= gr[:, np.newaxis]
        
        # Master limiter
        max_val = np.max(np.abs(stereo_output))
        limit_lin = 10.0 ** (self.master.limiter_threshold / 20.0)
        if max_val > limit_lin:
            stereo_output = np.tanh(stereo_output * 2.0) * 0.5
        
        # Output gain
        out_gain = 10.0 ** (self.master.output_gain / 20.0)
        stereo_output *= out_gain
        
        # Soft clip
        stereo_output = np.tanh(stereo_output)
        
        return stereo_output
    
    def get_levels(self) -> Dict[str, Any]:
        """Get current meter levels for all channels."""
        levels = {"channels": [], "master": 0.0}
        
        for ch in self.channels:
            ch_level = {
                "id": ch.channel_id,
                "name": ch.name,
                "mute": ch.settings.mute,
                "solo": ch.settings.solo,
                "fader": ch.settings.fader,
                "pan": ch.settings.pan,
                "eq": {
                    "low": ch.settings.eq.low_gain,
                    "mid": ch.settings.eq.mid_gain,
                    "high": ch.settings.eq.high_gain
                },
                "compressor": {
                    "threshold": ch.settings.compressor.threshold,
                    "ratio": ch.settings.compressor.ratio
                }
            }
            levels["channels"].append(ch_level)
        
        return levels


# =============================================================================
# Example Usage
# =============================================================================

def generate_test_tones(sample_rate: float = 44100.0, duration: float = 3.0):
    """Generate test tones for each channel."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Different frequencies for different "tracks"
    tones = {
        "kick": np.sin(2 * np.pi * 60 * t) * 0.8,
        "bass": np.sin(2 * np.pi * 80 * t) * 0.5 + np.sin(2 * np.pi * 120 * t) * 0.3,
        "snare": np.random.randn(len(t)) * 0.3,
        "hihat": np.random.randn(len(t)) * 0.15,
        "guitar": np.sin(2 * np.pi * 330 * t) * 0.4 + np.sin(2 * np.pi * 440 * t) * 0.3,
        "synth": np.sin(2 * np.pi * 220 * t) * 0.35 + np.sin(2 * np.pi * 330 * t) * 0.25,
        "piano": np.sin(2 * np.pi * 440 * t) * 0.3,
        "vocal": np.sin(2 * np.pi * 180 * t) * 0.4,
    }
    
    return list(tones.values())


def example_mixing_workflow():
    """Demonstrate a typical mixing workflow."""
    print("=" * 60)
    print("MIXING CONSOLE - Example Workflow")
    print("=" * 60)
    
    # Create console with 8 channels
    console = MixingConsole(num_channels=8, sample_rate=44100.0)
    
    # Generate test audio (simulating stems)
    print("\n[1] Loading audio stems...")
    audio_stems = generate_test_tones(duration=5.0)
    print(f"    Generated {len(audio_stems)} stems, {len(audio_stems[0])} samples each")
    
    # Channel naming and setup
    channel_names = ["Kick", "Bass", "Snare", "HiHat", "Guitar", "Synth", "Piano", "Vocal"]
    for i, name in enumerate(channel_names):
        console.channels[i].name = name
    
    # Set up mixing - Channel 1 (Kick)
    print("\n[2] Setting up channel processing...")
    ch1 = console.get_channel(0)
    ch1.set_gain(3.0)
    ch1.set_eq(low_gain=3.0, high_gain=-2.0)  # Boost low, cut high
    ch1.set_compressor(threshold=-12.0, ratio=4.0, attack=5.0, release=50.0, makeup=2.0)
    ch1.set_pan(0.0)  # Center
    ch1.set_fader(-3.0)
    print(f"    {ch1.name}: Gain={ch1.settings.gain}dB, EQ(Lo={ch1.settings.eq.low_gain}dB, Hi={ch1.settings.eq.high_gain}dB), Comp(Thresh={ch1.settings.compressor.threshold}dB, Ratio={ch1.settings.compressor.ratio}:1)")
    
    # Channel 2 (Bass) - more compression
    ch2 = console.get_channel(1)
    ch2.set_gain(2.0)
    ch2.set_eq(low_gain=2.0, mid_gain=-1.0)
    ch2.set_compressor(threshold=-18.0, ratio=6.0, attack=10.0, release=100.0, makeup=4.0)
    ch2.set_pan(-0.2)  # Slightly left
    ch2.set_fader(-2.0)
    print(f"    {ch2.name}: Gain={ch2.settings.gain}dB, Pan={ch2.settings.pan}, Fader={ch2.settings.fader}dB")
    
    # Channel 3-4 (Drums) - panned left/right
    ch3 = console.get_channel(2)
    ch3.set_gain(0.0)
    ch3.set_eq(mid_gain=2.0, high_gain=3.0)
    ch3.set_pan(-0.5)
    ch3.set_fader(-4.0)
    
    ch4 = console.get_channel(3)
    ch4.set_gain(-2.0)
    ch4.set_eq(high_gain=4.0)  # Bright
    ch4.set_pan(0.5)
    ch4.set_fader(-6.0)
    print(f"    {ch3.name}: Pan={ch3.settings.pan}, Fader={ch3.settings.fader}dB")
    print(f"    {ch4.name}: Pan={ch4.settings.pan}, Fader={ch4.settings.fader}dB")
    
    # Channel 5-6 (Guitars/Synths)
    ch5 = console.get_channel(4)
    ch5.set_gain(-2.0)
    ch5.set_pan(-0.6)
    ch5.set_fader(-5.0)
    
    ch6 = console.get_channel(5)
    ch6.set_gain(-3.0)
    ch6.set_pan(0.6)
    ch6.set_fader(-5.0)
    print(f"    {ch5.name}: Pan={ch5.settings.pan}, Fader={ch5.settings.fader}dB")
    print(f"    {ch6.name}: Pan={ch6.settings.pan}, Fader={ch6.settings.fader}dB")
    
    # Channel 7-8 (Keys/Vocals) - higher in mix
    ch7 = console.get_channel(6)
    ch7.set_gain(0.0)
    ch7.set_eq(mid_gain=2.0)
    ch7.set_pan(0.0)
    ch7.set_fader(-2.0)
    
    ch8 = console.get_channel(7)
    ch8.set_gain(1.0)
    ch8.set_eq(low_gain=-2.0, mid_gain=3.0, high_gain=1.0)  # Presence
    ch8.set_compressor(threshold=-20.0, ratio=3.0, attack=2.0, release=80.0, makeup=3.0)
    ch8.set_pan(0.0)
    ch8.set_fader(0.0)  # Vocal up
    print(f"    {ch7.name}: Pan={ch7.settings.pan}, Fader={ch7.settings.fader}dB")
    print(f"    {ch8.name}: Gain={ch8.settings.gain}dB, Fader={ch8.settings.fader}dB (Vocal lead)")
    
    # Master bus setup
    print("\n[3] Setting up master bus...")
    console.master.set_master_compressor(threshold=-8.0, ratio=2.0, 
                                          attack=10.0, release=100.0, makeup=1.0)
    console.master.set_limiter(-0.5)
    console.master.set_output_gain(-3.0)
    print(f"    Compressor: {console.master.compressor.settings.threshold}dB threshold, {console.master.compressor.settings.ratio}:1 ratio")
    print(f"    Limiter: {console.master.limiter_threshold}dB")
    print(f"    Output gain: {console.master.output_gain}dB")
    
    # Process the mix
    print("\n[4] Processing mix...")
    final_mix = console.process_audio(audio_stems)
    
    # Get levels
    print("\n[5] Channel levels:")
    levels = console.get_levels()
    for ch in levels["channels"]:
        fader_str = f"{ch['fader']:+.1f}dB"
        mute_str = "M" if ch["mute"] else " "
        solo_str = "S" if ch["solo"] else " "
        print(f"    {ch['name']:8s} [{mute_str}{solo_str}] Fader: {fader_str:>6s}  Pan: {ch['pan']:>5.1f}")
    
    print(f"\n[6] Output levels:")
    peak_db = 20.0 * np.log10(max(np.max(np.abs(final_mix)), 1e-10))
    rms_db = 20.0 * np.log10(max(np.sqrt(np.mean(final_mix**2)), 1e-10))
    print(f"    Peak:  {peak_db:>6.1f} dB")
    print(f"    RMS:   {rms_db:>6.1f} dB")
    print(f"    Duration: {len(final_mix) / 44100.0:.2f} seconds")
    
    # Demonstrate solo/mute
    print("\n[7] Testing mute/solo...")
    
    # Mute drums (channels 2-3)
    console.get_channel(2).set_mute(True)
    console.get_channel(3).set_mute(True)
    print("    Muted: Snare, HiHat")
    
    # Solo vocal
    console.get_channel(7).set_solo(True)
    print("    Solo: Vocal")
    
    print("\n" + "=" * 60)
    print("Mix complete!")
    print("=" * 60)
    
    return console, final_mix


def example_detailed_channel_strip():
    """Show detailed channel strip capabilities."""
    print("\n" + "=" * 60)
    print("CHANNEL STRIP DETAILED EXAMPLE")
    print("=" * 60)
    
    # Create a single channel
    ch = ChannelStrip(0, "Test Channel", sample_rate=44100.0)
    
    # Generate test signal
    t = np.linspace(0, 1.0, 44100)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    print(f"\nInput signal: 440 Hz sine wave, amplitude 0.5")
    
    # Process with different settings
    ch.set_gain(6.0)  # +6dB
    print(f"\n[1] Input Gain: +6 dB")
    
    ch.set_eq(low_gain=4.0, mid_gain=-2.0, high_gain=3.0)
    print(f"    EQ: Lo +4dB, Mid -2dB, Hi +3dB")
    
    ch.set_compressor(threshold=-12.0, ratio=4.0, attack=10.0, release=100.0, makeup=3.0)
    print(f"    Compressor: -12dB threshold, 4:1 ratio, 10ms attack, 100ms release, +3dB makeup")
    
    ch.set_pan(0.3)  # Slightly right
    print(f"    Pan: +0.3 (right)")
    
    ch.set_fader(-3.0)
    print(f"    Fader: -3dB")
    
    # Process
    output, sends = ch.process(audio)
    print(f"\n[2] Output:")
    print(f"    Stereo channels: {output.shape}")
    print(f"    L/R amplitude: {np.max(np.abs(output), axis=0)}")
    print(f"    Send buses: {list(sends.keys())}")
    
    # Add sends
    ch.add_send("Reverb")
    ch.add_send("Delay")
    ch.settings.sends["Reverb"] = -10.0  # -10dB to reverb
    ch.settings.sends["Delay"] = -15.0   # -15dB to delay
    
    output, sends = ch.process(audio)
    print(f"\n[3] With sends:")
    print(f"    Reverb send: {sends['Reverb'][0]:.3f} (first sample)")
    print(f"    Delay send: {sends['Delay'][0]:.3f} (first sample)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run examples
    console, mix = example_mixing_workflow()
    example_detailed_channel_strip()
    
    print("\n✓ Mixing console created successfully!")
    print("  Use console.process_audio(audio_stems) to mix your tracks.")
