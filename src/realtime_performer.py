"""
Real-Time Performance System for AI DJ
======================================
Live mixing, beat-synced transitions, real-time effects,
and DJ performance mode with dual deck control.
"""

import numpy as np
import time
import threading
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# =============================================================================
# BEAT DETECTION & BPM ANALYSIS
# =============================================================================

class BeatDetector:
    """Real-time beat detection and BPM analysis."""
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.history = deque(maxlen=2048)
        self.beat_times = deque(maxlen=100)
        self.last_beat_time = 0
        self.energy_history = deque(maxlen=43)
        self.bpm = 120.0
        self.beat_phase = 0.0
        
    def analyze_energy(self, audio: np.ndarray) -> float:
        return np.sqrt(np.mean(audio ** 2))
    
    def detect_beat(self, audio: np.ndarray) -> bool:
        energy = self.analyze_energy(audio)
        self.energy_history.append(energy)
        
        if len(self.energy_history) < 43:
            return False
        
        avg_energy = np.mean(self.energy_history)
        threshold = avg_energy * 1.3
        current_time = time.time()
        
        if energy > threshold and (current_time - self.last_beat_time) > 0.1:
            self.last_beat_time = current_time
            self.beat_times.append(current_time)
            
            if len(self.beat_times) >= 4:
                intervals = np.diff(list(self.beat_times))
                avg_interval = np.mean(intervals)
                if 0.2 < avg_interval < 2.0:
                    self.bpm = 60.0 / avg_interval
            return True
        return False
    
    def update_beat_phase(self) -> float:
        current_time = time.time()
        if len(self.beat_times) >= 2:
            last_beat = self.beat_times[-1]
            beat_interval = 60.0 / max(self.bpm, 60)
            time_since_beat = current_time - last_beat
            self.beat_phase = (time_since_beat % beat_interval) / beat_interval
        else:
            self.beat_phase = 0.0
        return self.beat_phase
    
    def get_samples_to_beat(self) -> int:
        phase = self.update_beat_phase()
        if phase < 1.0:
            samples_until = int((1.0 - phase) * 60.0 / self.bpm * self.sample_rate)
            return samples_until
        return 0


# =============================================================================
# CROSSFADER & MIXER
# =============================================================================

class Crossfader:
    def __init__(self):
        self.position = 0.0
        self.curve = 0.5
        
    def get_gains(self) -> Tuple[float, float]:
        x = (self.position + 1) / 2
        if self.curve == 0:
            gain_a = 1.0 - x
            gain_b = x
        else:
            gain_a = np.cos(x * np.pi / 2)
            gain_b = np.sin(x * np.pi / 2)
        return gain_a, gain_b
    
    def set_position(self, position: float):
        self.position = np.clip(position, -1.0, 1.0)
    
    def set_curve(self, curve: float):
        self.curve = np.clip(curve, 0.0, 1.0)


# =============================================================================
# DJ DECK
# =============================================================================

class DeckState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    CUEING = "cueing"


@dataclass
class HotCue:
    position: float = 0.0
    enabled: bool = False
    color: str = "red"


@dataclass
class Loop:
    start: float = 0.0
    end: float = 0.0
    active: bool = False
    beats: int = 0


class DJDeck:
    def __init__(self, deck_id: str, sample_rate: int = 44100, buffer_size: int = 1024):
        self.deck_id = deck_id
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.state = DeckState.STOPPED
        self.playhead = 0.0
        self.playing = False
        self.speed = 1.0
        self.pitch = 0.0
        self.key = 0
        self.audio_buffer: Optional[np.ndarray] = None
        self.output_buffer = np.zeros(buffer_size * 2)
        self.beat_detector = BeatDetector(sample_rate, buffer_size)
        self.hot_cues: List[HotCue] = [HotCue() for _ in range(8)]
        self.loop = Loop()
        self.loop_active = False
        self.gain = 1.0
        self.volume = 1.0
        self.mute = False
        self.filter_freq = 20000.0
        self.filter_resonance = 0.0
        self.effect_sends: Dict[str, float] = {"reverb": 0.0, "delay": 0.0, "chorus": 0.0}
        self.vinyl_mode = False
        self.vinyl_position = 0.0
        
    def load_audio(self, audio: np.ndarray):
        self.audio_buffer = audio.flatten() if audio.ndim > 1 else audio
        self.playhead = 0.0
        
    def play(self):
        self.playing = True
        self.state = DeckState.PLAYING
        
    def pause(self):
        self.playing = False
        self.state = DeckState.PAUSED
        
    def stop(self):
        self.playing = False
        self.playhead = 0.0
        self.state = DeckState.STOPPED
        
    def seek(self, position: float):
        if self.audio_buffer is not None:
            max_pos = len(self.audio_buffer) / self.sample_rate
            self.playhead = np.clip(position, 0, max_pos)
            
    def set_speed(self, speed: float):
        self.speed = np.clip(speed, 0.5, 2.0)
        
    def set_pitch(self, pitch_percent: float):
        self.pitch = np.clip(pitch_percent, -50.0, 50.0)
        self.speed = 1.0 + (self.pitch / 100.0)
        
    def set_volume(self, volume: float):
        self.volume = np.clip(volume, 0.0, 1.0)
        
    def set_filter(self, freq: float, resonance: float = 0.0):
        self.filter_freq = np.clip(freq, 20.0, 20000.0)
        self.filter_resonance = np.clip(resonance, 0.0, 10.0)
        
    def set_hot_cue(self, index: int, position: float = None, enabled: bool = True):
        if 0 <= index < 8:
            if position is not None:
                self.hot_cues[index].position = position
            self.hot_cues[index].enabled = enabled
            
    def trigger_hot_cue(self, index: int):
        if 0 <= index < 8 and self.hot_cues[index].enabled:
            self.seek(self.hot_cues[index].position)
            if not self.playing:
                self.play()
                
    def clear_hot_cue(self, index: int):
        if 0 <= index < 8:
            self.hot_cues[index].enabled = False
            
    def set_loop(self, start: float, end: float):
        self.loop.start = start
        self.loop.end = end
        self.loop.active = True
        self.loop_active = True
        
    def set_loop_beats(self, beats: int):
        current = self.get_current_time()
        beat_duration = 60.0 / self.beat_detector.bpm
        self.loop.start = current
        self.loop.end = current + (beats * beat_duration)
        self.loop.beats = beats
        self.loop_active = True
        
    def toggle_loop(self):
        self.loop_active = not self.loop_active
        
    def exit_loop(self):
        self.loop_active = False
        self.loop.active = False
        
    def get_current_time(self) -> float:
        return self.playhead
        
    def get_current_beat(self) -> int:
        beat_duration = 60.0 / max(self.beat_detector.bpm, 60)
        return int(self.playhead / beat_duration) + 1
        
    def get_phase_to_beat(self) -> float:
        beat_duration = 60.0 / max(self.beat_detector.bpm, 60)
        return (self.playhead % beat_duration) / beat_duration
        
    def process(self, num_samples: int) -> np.ndarray:
        if self.audio_buffer is None or len(self.audio_buffer) == 0:
            return np.zeros(num_samples)
        
        start_sample = int(self.playhead * self.sample_rate)
        audio_length = len(self.audio_buffer)
        
        if self.loop_active and self.loop.active:
            loop_start_sample = int(self.loop.start * self.sample_rate)
            loop_end_sample = int(self.loop.end * self.sample_rate)
            loop_length = loop_end_sample - loop_start_sample
            
            if loop_length > 0:
                output = np.zeros(num_samples)
                remaining = num_samples
                pos = start_sample
                
                while remaining > 0:
                    if loop_start_sample <= pos < loop_end_sample:
                        avail = min(remaining, loop_end_sample - pos)
                        source_start = pos % audio_length
                        end_pos = num_samples - remaining + avail
                        output[num_samples - remaining:end_pos] = self.audio_buffer[source_start:source_start + avail]
                        pos += int(avail * self.speed)
                        remaining -= avail
                    else:
                        avail = min(remaining, audio_length - (pos % audio_length))
                        source_start = pos % audio_length
                        end_pos = num_samples - remaining + avail
                        output[num_samples - remaining:end_pos] = self.audio_buffer[source_start:source_start + avail]
                        pos += int(avail * self.speed)
                        remaining -= avail
                audio = output
            else:
                end_sample = min(start_sample + int(num_samples * self.speed), audio_length)
                audio = self.audio_buffer[start_sample:end_sample]
                if len(audio) < num_samples:
                    audio = np.pad(audio, (0, num_samples - len(audio)))
        else:
            end_sample = min(start_sample + int(num_samples * self.speed), audio_length)
            audio = self.audio_buffer[start_sample:end_sample]
            if len(audio) < num_samples:
                audio = np.pad(audio, (0, num_samples - len(audio)))
        
        if not self.mute:
            audio = audio * self.gain * self.volume
        else:
            audio = np.zeros(num_samples)
            
        self.playhead += (num_samples / self.sample_rate) * self.speed
        if self.playhead * self.sample_rate >= audio_length:
            self.playhead = 0.0
            
        self.beat_detector.detect_beat(audio)
        
        if self.filter_freq < 20000:
            audio = self._apply_filter(audio)
            
        return audio
    
    def _apply_filter(self, audio: np.ndarray) -> np.ndarray:
        alpha = self.filter_freq / (self.filter_freq + self.sample_rate / (2 * np.pi))
        filtered = np.zeros_like(audio)
        for i in range(1, len(audio)):
            filtered[i] = filtered[i-1] + alpha * (audio[i] - filtered[i-1])
        return filtered


# =============================================================================
# BEAT-SYNCED TRANSITION
# =============================================================================

class TransitionEngine:
    def __init__(self):
        self.transition_duration = 8.0
        self.transition_type = "linear"
        self.active = False
        self.progress = 0.0
        self.start_time = 0.0
        
    def start_transition(self, from_bpm: float, to_bpm: float, duration: float = 8.0, transition_type: str = "sync"):
        self.from_bpm = from_bpm
        self.to_bpm = to_bpm
        self.transition_duration = duration
        self.transition_type = transition_type
        self.active = True
        self.progress = 0.0
        self.start_time = time.time()
        
    def update(self) -> float:
        if not self.active:
            return 0.0
        elapsed = time.time() - self.start_time
        self.progress = min(elapsed / self.transition_duration, 1.0)
        if self.progress >= 1.0:
            self.active = False
        return self.progress
        
    def get_speed_multiplier(self, deck_speed: float) -> float:
        if not self.active:
            return deck_speed
        if self.transition_type == "sync":
            return self.from_bpm + (self.to_bpm - self.from_bpm) * self.progress
        elif self.transition_type == "blend":
            blend_factor = self._ease_curve(self.progress)
            return deck_speed * (1 - blend_factor) + (self.to_bpm / self.from_bpm) * deck_speed * blend_factor
        return deck_speed
            
    def get_transition_volume(self, to_deck: bool = False) -> float:
        if not self.active:
            return 1.0 if to_deck else 0.0
        if self.transition_type == "sync" or self.transition_type == "blend":
            if to_deck:
                return self._ease_curve(self.progress)
            else:
                return 1.0 - self._ease_curve(self.progress)
        return 0.5
        
    def _ease_curve(self, t: float) -> float:
        if self.transition_type == "expo":
            return t ** 2 if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
        elif self.transition_type == "smooth":
            return t * t * (3 - 2 * t)
        return t


# =============================================================================
# REAL-TIME EFFECTS PROCESSOR
# =============================================================================

class RealtimeEffects:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.effects_chain = []
        self.reverb_wet = 0.0
        self.reverb_decay = 0.5
        self.reverb_room_size = 0.5
        self.delay_wet = 0.0
        self.delay_time = 0.5
        self.delay_feedback = 0.3
        self.delay_tempo_sync = False
        self.filter_low = 20.0
        self.filter_high = 20000.0
        self.filter_type = "lowpass"
        self.distortion_drive = 0.0
        self.distortion_type = "soft"
        self.chorus_wet = 0.0
        self.chorus_rate = 1.5
        self.chorus_depth = 0.5
        self._init_buffers()
        
    def _init_buffers(self):
        max_delay_samples = int(self.sample_rate * 2.0)
        self.delay_buffer = np.zeros(max_delay_samples)
        self.delay_write_pos = 0
        self.chorus_lfo_phase = 0.0
        
    def set_reverb(self, wet: float, decay: float = None, room_size: float = None):
        self.reverb_wet = np.clip(wet, 0.0, 1.0)
        if decay is not None:
            self.reverb_decay = np.clip(decay, 0.0, 0.99)
        if room_size is not None:
            self.reverb_room_size = np.clip(room_size, 0.1, 1.0)
            
    def set_delay(self, wet: float, time_seconds: float = None, feedback: float = None, tempo_sync: bool = None):
        self.delay_wet = np.clip(wet, 0.0, 1.0)
        if time_seconds is not None:
            self.delay_time = np.clip(time_seconds, 0.01, 2.0)
        if feedback is not None:
            self.delay_feedback = np.clip(feedback, 0.0, 0.95)
        if tempo_sync is not None:
            self.delay_tempo_sync = tempo_sync
            
    def set_filter(self, low: float = None, high: float = None, filter_type: str = None):
        if low is not None:
            self.filter_low = np.clip(low, 20.0, 20000.0)
        if high is not None:
            self.filter_high = np.clip(high, 20.0, 20000.0)
        if filter_type is not None:
            self.filter_type = filter_type
            
    def set_distortion(self, drive: float, dist_type: str = "soft"):
        self.distortion_drive = np.clip(drive, 0.0, 1.0)
        self.distortion_type = dist_type
        
    def set_chorus(self, wet: float, rate: float = None, depth: float = None):
        self.chorus_wet = np.clip(wet, 0.0, 1.0)
        if rate is not None:
            self.chorus_rate = np.clip(rate, 0.1, 10.0)
        if depth is not None:
            self.chorus_depth = np.clip(depth, 0.0, 1.0)
            
    def process(self, audio: np.ndarray, bpm: float = 120.0) -> np.ndarray:
        output = audio.copy()
        
        if self.distortion_drive > 0:
            output = self._apply_distortion(output)
        if self.filter_low > 20 or self.filter_high < 20000:
            output = self._apply_filter(output)
        if self.delay_wet > 0:
            output = self._apply_delay(output, bpm)
        if self.chorus_wet > 0:
            output = self._apply_chorus(output)
        if self.reverb_wet > 0:
            output = self._apply_reverb(output)
        return output
        
    def _apply_distortion(self, audio: np.ndarray) -> np.ndarray:
        gain = 1.0 + self.distortion_drive * 20
        if self.distortion_type == "soft":
            output = np.tanh(audio * gain)
        elif self.distortion_type == "hard":
            output = np.clip(audio * gain, -1, 1)
        elif self.distortion_type == "sigmoid":
            output = 2 / (1 + np.exp(-2 * audio * gain)) - 1
        else:
            output = audio * gain
        return output
        
    def _apply_filter(self, audio: np.ndarray) -> np.ndarray:
        if self.filter_type == "lowpass":
            alpha = self.filter_high / (self.filter_high + self.sample_rate / (2 * np.pi))
            output = np.zeros_like(audio)
            for i in range(1, len(audio)):
                output[i] = output[i-1] + alpha * (audio[i] - output[i-1])
            return output
        elif self.filter_type == "highpass":
            alpha = self.filter_low / (self.filter_low + self.sample_rate / (2 * np.pi))
            output = np.zeros_like(audio)
            prev = 0.0
            for i in range(len(audio)):
                curr = audio[i]
                output[i] = alpha * (prev + curr - output[i-1])
                prev = curr
            return output
        return audio
            
    def _apply_delay(self, audio: np.ndarray, bpm: float) -> np.ndarray:
        if self.delay_tempo_sync:
            beat_time = 60.0 / bpm
            delay_samples = int(beat_time * self.sample_rate)
        else:
            delay_samples = int(self.delay_time * self.sample_rate)
        delay_samples = min(delay_samples, len(self.delay_buffer) - 1)
        output = np.zeros_like(audio)
        for i in range(len(audio)):
            read_pos = (self.delay_write_pos - delay_samples) % len(self.delay_buffer)
            delayed = self.delay_buffer[read_pos]
            output[i] = audio[i] + delayed * self.delay_wet
            self.delay_buffer[self.delay_write_pos] = output[i] + delayed * self.delay_feedback
            self.delay_write_pos = (self.delay_write_pos + 1) % len(self.delay_buffer)
        return output
        
    def _apply_chorus(self, audio: np.ndarray) -> np.ndarray:
        output = np.zeros_like(audio)
        modulation_rate = self.chorus_rate / self.sample_rate
        delay_range = int(0.03 * self.sample_rate)
        for i in range(len(audio)):
            self.chorus_lfo_phase += modulation_rate
            if self.chorus_lfo_phase > 1.0:
                self.chorus_lfo_phase -= 1.0
            lfo_value = np.sin(2 * np.pi * self.chorus_lfo_phase) * self.chorus_depth
            delay_offset = int(lfo_value * delay_range)
            read_pos = (self.delay_write_pos + delay_offset) % len(self.delay_buffer)
            delayed = self.delay_buffer[read_pos]
            output[i] = audio[i] + delayed * self.chorus_wet * 0.5
            self.delay_buffer[self.delay_write_pos] = audio[i]
            self.delay_write_pos = (self.delay_write_pos + 1) % len(self.delay_buffer)
        return output
        
    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        delays = [1557, 1617, 1491, 1422, 1277, 1356]
        decay = self.reverb_decay
        wet = self.reverb_wet
        output = audio.copy()
        for delay in delays:
            delay = int(delay * self.reverb_room_size)
            delayed = np.zeros_like(audio)
            for i in range(delay, len(audio)):
                delayed[i] = audio[i - delay] + delayed[i - delay] * decay
            output += delayed * wet / len(delays)
        return np.clip(output, -1, 1)


# =============================================================================
# DJ PERFORMANCE CONTROLLER
# =============================================================================

class PerformanceMode(Enum):
    DUAL_DECK = "dual_deck"
    SPLIT_CUE = "split_cue"
    BLEND = "blend"
    BATTLE = "battle"


class DJPerformanceController:
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.deck_a = DJDeck("A", sample_rate, buffer_size)
        self.deck_b = DJDeck("B", sample_rate, buffer_size)
        self.crossfader = Crossfader()
        self.transition = TransitionEngine()
        self.effects = RealtimeEffects(sample_rate)
        self.mode = PerformanceMode.BLEND
        self.master_volume = 1.0
        self.on_beat: Optional[Callable] = None
        self.on_transition_complete: Optional[Callable] = None
        self.sync_enabled = False
        self.sync_deck = "A"
        self.master_bpm = 120.0
        self.playing = False
        self.current_time = 0.0
        self.output_buffer = np.zeros(buffer_size * 2)
        self.lock = threading.Lock()
        
    def load_track_deck_a(self, audio: np.ndarray, bpm: float = None):
        with self.lock:
            self.deck_a.load_audio(audio)
            if bpm:
                self.deck_a.beat_detector.bpm = bpm
            else:
                self._analyze_bpm(self.deck_a)
                
    def load_track_deck_b(self, audio: np.ndarray, bpm: float = None):
        with self.lock:
            self.deck_b.load_audio(audio)
            if bpm:
                self.deck_b.beat_detector.bpm = bpm
            else:
                self._analyze_bpm(self.deck_b)
                
    def _analyze_bpm(self, deck: DJDeck):
        if deck.audio_buffer is None:
            return
        chunk_size = self.sample_rate // 10
        energies = []
        for i in range(0, min(len(deck.audio_buffer), self.sample_rate * 30), chunk_size):
            chunk = deck.audio_buffer[i:i+chunk_size]
            energies.append(np.sqrt(np.mean(chunk ** 2)))
        threshold = np.mean(energies) * 1.5
        peaks = [i for i, e in enumerate(energies) if e > threshold]
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            avg_interval = np.mean(intervals) * 0.1
            if 0.2 < avg_interval < 2.0:
                deck.beat_detector.bpm = 60.0 / avg_interval
        self.master_bpm = max(self.deck_a.beat_detector.bpm, self.deck_b.beat_detector.bpm)
        
    def play_deck_a(self):
        self.deck_a.play()
        self.playing = True
        
    def play_deck_b(self):
        self.deck_b.play()
        self.playing = True
        
    def pause_deck_a(self):
        self.deck_a.pause()
        
    def pause_deck_b(self):
        self.deck_b.pause()
        
    def stop_all(self):
        self.deck_a.stop()
        self.deck_b.stop()
        self.playing = False
        
    def set_crossfader(self, position: float):
        self.crossfader.set_position(position)
        
    def set_master_volume(self, volume: float):
        self.master_volume = np.clip(volume, 0.0, 1.0)
        
    def sync_decks(self, enable: bool = True, source: str = "A"):
        self.sync_enabled = enable
        self.sync_deck = source
        if enable:
            if source == "A":
                self.master_bpm = self.deck_a.beat_detector.bpm
                self.deck_b.set_speed(self.master_bpm / self.deck_b.beat_detector.bpm)
            else:
                self.master_bpm = self.deck_b.beat_detector.bpm
                self.deck_a.set_speed(self.master_bpm / self.deck_a.beat_detector.bpm)
                
    def start_transition(self, duration: float = 8.0, transition_type: str = "sync"):
        from_bpm = self.deck_a.beat_detector.bpm if self.deck_a.playing else 0
        to_bpm = self.deck_b.beat_detector.bpm if self.deck_b.playing else self.master_bpm
        self.transition.start_transition(from_bpm, to_bpm, duration, transition_type)
        
    def trigger_transition(self, duration: float = 8.0):
        self.start_transition(duration, "blend")
        
    def set_performance_mode(self, mode: PerformanceMode):
        self.mode = mode
        
    def process(self, num_samples: int = None) -> np.ndarray:
        if num_samples is None:
            num_samples = self.buffer_size
        with self.lock:
            audio_a = self.deck_a.process(num_samples)
            audio_b = self.deck_b.process(num_samples)
            gain_a, gain_b = self.crossfader.get_gains()
            if self.transition.active:
                progress = self.transition.update()
                gain_a *= (1 - progress)
                gain_b *= self.transition.get_transition_volume(to_deck=True)
                if progress >= 1.0 and self.on_transition_complete:
                    self.on_transition_complete()
            mixed = (audio_a * gain_a) + (audio_b * gain_b)
            bpm = max(self.deck_a.beat_detector.bpm, self.deck_b.beat_detector.bpm)
            mixed = self.effects.process(mixed, bpm)
            mixed *= self.master_volume
            mixed = np.clip(mixed, -1.0, 1.0)
            if mixed.ndim == 1:
                stereo = np.zeros(len(mixed) * 2)
                stereo[::2] = mixed
                stereo[1::2] = mixed
                mixed = stereo
            self.output_buffer = mixed
            return mixed
            
    def get_status(self) -> Dict:
        return {
            "mode": self.mode.value,
            "deck_a": {
                "playing": self.deck_a.playing,
                "bpm": round(self.deck_a.beat_detector.bpm, 1),
                "position": round(self.deck_a.get_current_time(), 2),
                "beat": self.deck_a.get_current_beat(),
                "pitch": self.deck_a.pitch,
                "volume": self.deck_a.volume,
                "hot_cues": [hc.enabled for hc in self.deck_a.hot_cues],
                "loop_active": self.deck_a.loop_active
            },
            "deck_b": {
                "playing": self.deck_b.playing,
                "bpm": round(self.deck_b.beat_detector.bpm, 1),
                "position": round(self.deck_b.get_current_time(), 2),
                "beat": self.deck_b.get_current_beat(),
                "pitch": self.deck_b.pitch,
                "volume": self.deck_b.volume,
                "hot_cues": [hc.enabled for hc in self.deck_b.hot_cues],
                "loop_active": self.deck_b.loop_active
            },
            "crossfader": round(self.crossfader.position, 2),
            "sync": self.sync_enabled,
            "transition_active": self.transition.active,
            "master_volume": round(self.master_volume, 2)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("AI DJ Real-Time Performance System Demo")
    print("=" * 60)
    
    controller = DJPerformanceController(sample_rate=44100, buffer_size=1024)
    sample_rate = 44100
    duration = 5
    
    bpm_a = 120
    samples_per_beat = int(60 / bpm_a * sample_rate)
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_a = np.zeros(len(t))
    
    for i in range(0, len(audio_a), samples_per_beat):
        kick_len = min(samples_per_beat // 4, len(audio_a) - i)
        kick_env = np.exp(-np.linspace(0, 5, kick_len))
        kick_freq = 60 * np.exp(-np.linspace(0, 3, kick_len))
        kick = np.sin(2 * np.pi * kick_freq * np.linspace(0, kick_len/sample_rate, kick_len)) * kick_env
        audio_a[i:i+kick_len] += kick * 0.8
    
    bpm_b = 128
    samples_per_beat_b = int(60 / bpm_b * sample_rate)
    audio_b = np.zeros(len(t))
    
    for i in range(0, len(audio_b), samples_per_beat_b):
        kick_len = min(samples_per_beat_b // 4, len(audio_b) - i)
        kick_env = np.exp(-np.linspace(0, 5, kick_len))
        kick_freq = 60 * np.exp(-np.linspace(0, 3, kick_len))
        kick = np.sin(2 * np.pi * kick_freq * np.linspace(0, kick_len/sample_rate, kick_len)) * kick_env
        audio_b[i:i+kick_len] += kick * 0.8
    
    controller.load_track_deck_a(audio_a, bpm=120)
    controller.load_track_deck_b(audio_b, bpm=128)
    
    print(f"Loaded tracks: Deck A @ {bpm_a} BPM, Deck B @ {bpm_b} BPM")
    
    controller.play_deck_a()
    controller.set_crossfader(-1.0)
    print(f"Playing Deck A, crossfader at -1.0 (A only)")
    
    for i in range(5):
        output = controller.process(1024)
        status = controller.get_status()
        print(f"  Frame {i+1}: pos={status['deck_a']['position']}s, beat={status['deck_a']['beat']}")
    
    print("Fading to Deck B...")
    controller.play_deck_b()
    controller.trigger_transition(duration=2.0)
    
    for i in range(10):
        output = controller.process(1024)
        status = controller.get_status()
        trans = "active" if status['transition_active'] else "complete"
        print(f"  Frame {i+1}: A={status['deck_a']['position']:.2f}s B={status['deck_b']['position']:.2f}s trans={trans}")
    
    print("Testing effects...")
    controller.effects.set_delay(wet=0.3, time_seconds=0.25, feedback=0.4)
    controller.effects.set_filter(high=2000.0)
    output = controller.process(1024)
    print("  Delay: wet=0.3, Filter: high=2000Hz")
    
    print("Testing hot cues...")
    controller.deck_a.set_hot_cue(0, position=1.0, enabled=True)
    controller.deck_a.set_hot_cue(1, position=2.0, enabled=True)
    print("  Set hot cue 0 at 1.0s, hot cue 1 at 2.0s")
    
    print("Testing loop...")
    controller.deck_a.set_loop_beats(4)
    print("  Set 4-beat loop at current position")
    
    status = controller.get_status()
    print(f"Final: Mode={status['mode']} Crossfader={status['crossfader']} Sync={status['sync']}")
    print("Demo complete!")
