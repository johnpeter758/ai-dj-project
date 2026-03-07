#!/usr/bin/env python3
"""
Video Sync System - Syncs Music to Video

Synchronizes video playback and effects with music analysis:
- Beat detection and sync
- Energy-based visual intensity
- BPM synchronization
- Key detection for mood matching
- Transition effects for crossfades
- Visual effects triggered by drops and buildups
"""

import os
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum
from datetime import datetime

# Audio analysis imports (graceful fallback)
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class VisualEffect(Enum):
    """Visual effects that can be synced to music"""
    PULSE = "pulse"
    STROBE = "strobe"
    FADE = "fade"
    ZOOM = "zoom"
    COLOR_SHIFT = "color_shift"
    BLUR = "blur"
    GLITCH = "glitch"
    SWELL = "swell"
    DROP_INTENSITY = "drop_intensity"
    BUILDUP = "buildup"
    TRANSITION = "transition"


class SyncMode(Enum):
    """Video synchronization modes"""
    BEAT = "beat"           # Sync to individual beats
    BAR = "bar"             # Sync to musical bars
    PHRASE = "phrase"       # Sync to phrase boundaries
    ENERGY = "energy"       # Sync to energy changes
    MANUAL = "manual"       # Manual trigger


@dataclass
class BeatEvent:
    """Represents a detected beat event"""
    time: float           # Time in seconds
    beat_number: int      # Beat count within bar
    bar_number: int       # Bar count
    strength: float       # Beat strength (0-1)
    is_downbeat: bool     # First beat of bar


@dataclass
class EnergyEvent:
    """Represents an energy change in the music"""
    time: float
    energy: float         # Energy level (0-1)
    delta: float          # Energy change from previous
    event_type: str       # "buildup", "drop", "sustain", "break"


@dataclass
class VideoFrame:
    """Represents a video frame with sync metadata"""
    timestamp: float
    effect: Optional[VisualEffect] = None
    effect_params: Dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0
    color_shift: float = 0.0
    zoom: float = 1.0
    blur: float = 0.0


@dataclass
class SyncConfig:
    """Configuration for video sync"""
    mode: SyncMode = SyncMode.BEAT
    bpm: float = 128.0
    beats_per_bar: int = 4
    bars_per_phrase: int = 8
    transition_duration: float = 5.0
    effect_intensity: float = 1.0
    strobe_rate: float = 4.0        # Hz for strobe effect
    pulse_sensitivity: float = 1.0
    auto_detect_bpm: bool = True
    min_bpm: float = 60.0
    max_bpm: float = 200.0


class MusicAnalyzer:
    """Analyzes music for sync purposes"""
    
    def __init__(self, audio_file: str = None):
        self.audio_file = audio_file
        self.duration: float = 0.0
        self.bpm: float = 128.0
        self.key: str = "C"
        self.beats: List[BeatEvent] = []
        self.energy_events: List[EnergyEvent] = []
        self.tempo_curve: List[float] = []
        self._waveform: Optional[np.ndarray] = None if HAS_NUMPY else None
        
    def load_audio(self, audio_file: str) -> bool:
        """Load audio file for analysis"""
        if not HAS_LIBROSA:
            print("Warning: librosa not available, using synthetic analysis")
            self.audio_file = audio_file
            self.duration = 180.0  # Default 3 min
            self.bpm = 128.0
            return False
            
        try:
            self.audio_file = audio_file
            y, sr = librosa.load(audio_file, sr=None)
            self.duration = librosa.get_duration(y=y, sr=sr)
            self._waveform = y
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
    
    def detect_bpm(self) -> float:
        """Detect BPM from audio"""
        if not HAS_LIBROSA or self._waveform is None:
            return self.bpm
            
        try:
            tempo, beats = librosa.beat.beat_track(y=self._waveform)
            self.bpm = float(tempo)
            return self.bpm
        except Exception as e:
            print(f"BPM detection error: {e}")
            return self.bpm
    
    def detect_key(self) -> str:
        """Detect musical key"""
        if not HAS_LIBROSA or self._waveform is None:
            return self.key
            
        try:
            chroma = librosa.feature.chroma_cqt(y=self._waveform)
            key_idx = np.argmax(np.sum(chroma, axis=1))
            keys = ["C", "C#", "D", "D#", "E", "F", 
                    "F#", "G", "G#", "A", "A#", "B"]
            self.key = keys[key_idx]
            return self.key
        except Exception as e:
            print(f"Key detection error: {e}")
            return self.key
    
    def detect_beats(self, config: SyncConfig = None) -> List[BeatEvent]:
        """Detect all beat events in the track"""
        config = config or SyncConfig()
        self.beats = []
        
        if not HAS_LIBROSA or self._waveform is None:
            # Generate synthetic beats based on BPM
            beat_interval = 60.0 / config.bpm
            current_time = 0.0
            beat_num = 0
            bar_num = 0
            
            while current_time < self.duration:
                is_downbeat = (beat_num % config.beats_per_bar) == 0
                strength = 1.0 if is_downbeat else 0.7
                
                self.beats.append(BeatEvent(
                    time=current_time,
                    beat_number=beat_num % config.beats_per_bar,
                    bar_number=bar_num,
                    strength=strength,
                    is_downbeat=is_downbeat
                ))
                
                current_time += beat_interval
                beat_num += 1
                if beat_num % config.beats_per_bar == 0:
                    bar_num += 1
                    
            return self.beats
        
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=self._waveform)
            beat_times = librosa.frames_to_time(beat_frames)
            
            beat_num = 0
            bar_num = 0
            
            for i, bt in enumerate(beat_times):
                is_downbeat = (beat_num % config.beats_per_bar) == 0
                
                # Get strength from onset envelope
                strength = 0.8 if is_downbeat else 0.6
                
                self.beats.append(BeatEvent(
                    time=bt,
                    beat_number=beat_num % config.beats_per_bar,
                    bar_number=bar_num,
                    strength=strength,
                    is_downbeat=is_downbeat
                ))
                
                beat_num += 1
                if beat_num % config.beats_per_bar == 0:
                    bar_num += 1
                    
            return self.beats
        except Exception as e:
            print(f"Beat detection error: {e}")
            return self.beats
    
    def detect_energy(self, frame_duration: float = 0.1) -> List[EnergyEvent]:
        """Detect energy changes throughout the track"""
        self.energy_events = []
        
        if not HAS_LIBROSA or self._waveform is None:
            # Generate synthetic energy curve
            num_frames = int(self.duration / frame_duration)
            energy_values = []
            
            for i in range(num_frames):
                t = i * frame_duration
                # Simulate energy: buildup-drop pattern
                phase = (t / 30.0) % 1.0  # Every 30 seconds
                if phase < 0.7:
                    energy = 0.3 + phase * 0.7  # Buildup
                else:
                    energy = 1.0 - (phase - 0.7) * 2.0  # Drop
                energy = max(0.1, min(1.0, energy + (hash(str(i)) % 20 - 10) / 50))
                energy_values.append((t, energy))
            
            for i, (t, e) in enumerate(energy_values):
                delta = e - energy_values[i-1][1] if i > 0 else 0
                
                if delta > 0.1:
                    event_type = "buildup"
                elif delta < -0.1:
                    event_type = "drop"
                elif e < 0.3:
                    event_type = "break"
                else:
                    event_type = "sustain"
                    
                self.energy_events.append(EnergyEvent(
                    time=t,
                    energy=e,
                    delta=delta,
                    event_type=event_type
                ))
            
            return self.energy_events
        
        try:
            # Use RMS energy
            rms = librosa.feature.rms(y=self._waveform)[0]
            frame_times = librosa.frames_to_time(range(len(rms)), 
                                                  frame_length=2048,
                                                  hop_length=512)
            
            # Normalize
            rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
            
            for i, (t, e) in enumerate(zip(frame_times, rms_norm)):
                delta = e - rms_norm[i-1] if i > 0 else 0
                
                if delta > 0.1:
                    event_type = "buildup"
                elif delta < -0.1:
                    event_type = "drop"
                elif e < 0.3:
                    event_type = "break"
                else:
                    event_type = "sustain"
                    
                self.energy_events.append(EnergyEvent(
                    time=t,
                    energy=e,
                    delta=delta,
                    event_type=event_type
                ))
                
            return self.energy_events
        except Exception as e:
            print(f"Energy detection error: {e}")
            return self.energy_events
    
    def analyze(self, audio_file: str = None, config: SyncConfig = None) -> Dict:
        """Full analysis of audio file"""
        audio_file = audio_file or self.audio_file
        if not audio_file:
            return {"error": "No audio file specified"}
            
        self.load_audio(audio_file)
        config = config or SyncConfig()
        
        if config.auto_detect_bpm:
            self.detect_bpm()
        
        self.detect_key()
        self.detect_beats(config)
        self.detect_energy()
        
        return {
            "audio_file": audio_file,
            "duration": self.duration,
            "bpm": self.bpm,
            "key": self.key,
            "num_beats": len(self.beats),
            "num_bars": self.beats[-1].bar_number + 1 if self.beats else 0,
            "num_energy_events": len(self.energy_events),
            "config": {
                "mode": config.mode.value,
                "beats_per_bar": config.beats_per_bar,
                "bars_per_phrase": config.bars_per_phrase
            }
        }


class VideoSync:
    """Main video synchronization engine"""
    
    def __init__(self, config: SyncConfig = None):
        self.config = config or SyncConfig()
        self.analyzer = MusicAnalyzer()
        self.current_time: float = 0.0
        self.is_playing: bool = False
        self.effect_callback: Optional[Callable] = None
        self.frames: List[VideoFrame] = []
        self._current_beat_idx: int = 0
        self._current_energy_idx: int = 0
        
    def load_track(self, audio_file: str) -> Dict:
        """Load and analyze a track"""
        result = self.analyzer.analyze(audio_file, self.config)
        
        # Override BPM if detected
        if result.get("bpm"):
            self.config.bpm = result["bpm"]
            
        return result
    
    def set_effect_callback(self, callback: Callable[[VideoFrame], None]):
        """Set callback for effect rendering"""
        self.effect_callback = callback
    
    def generate_frame_at(self, timestamp: float) -> VideoFrame:
        """Generate video frame data at specific timestamp"""
        frame = VideoFrame(timestamp=timestamp)
        
        # Find current beat
        beat = self._find_beat_at(timestamp)
        if beat:
            frame.intensity = beat.strength * self.config.effect_intensity
            
            if beat.is_downbeat:
                frame.effect = VisualEffect.PULSE
                frame.effect_params = {"strength": beat.strength}
        
        # Find current energy event
        energy = self._find_energy_at(timestamp)
        if energy:
            frame.intensity = energy.energy
            
            if energy.event_type == "drop":
                frame.effect = VisualEffect.DROP_INTENSITY
                frame.effect_params = {"intensity": energy.energy}
            elif energy.event_type == "buildup":
                frame.effect = VisualEffect.BUILDUP
                frame.effect_params = {"intensity": energy.energy}
        
        # Apply sync mode effects
        if self.config.mode == SyncMode.BEAT:
            self._apply_beat_effects(frame, timestamp)
        elif self.config.mode == SyncMode.BAR:
            self._apply_bar_effects(frame, timestamp)
        elif self.config.mode == SyncMode.PHRASE:
            self._apply_phrase_effects(frame, timestamp)
        elif self.config.mode == SyncMode.ENERGY:
            self._apply_energy_effects(frame, timestamp)
            
        return frame
    
    def _find_beat_at(self, timestamp: float) -> Optional[BeatEvent]:
        """Find the beat event at a given timestamp"""
        beats = self.analyzer.beats
        if not beats:
            return None
            
        # Find closest beat
        closest = min(beats, key=lambda b: abs(b.time - timestamp))
        if abs(closest.time - timestamp) < 0.1:
            return closest
        return None
    
    def _find_energy_at(self, timestamp: float) -> Optional[EnergyEvent]:
        """Find energy event at timestamp"""
        events = self.analyzer.energy_events
        if not events:
            return None
            
        for event in events:
            if event.time <= timestamp:
                current = event
            else:
                break
        return current if 'current' in locals() else None
    
    def _apply_beat_effects(self, frame: VideoFrame, timestamp: float):
        """Apply effects based on beat sync"""
        beat_interval = 60.0 / self.config.bpm
        phase = (timestamp % beat_interval) / beat_interval
        
        # Strobe effect
        if phase < 0.1:
            frame.effect = VisualEffect.STROBE
            frame.effect_params = {"rate": self.config.strobe_rate}
        
        # Pulse on downbeats
        if frame.intensity > 0.8:
            frame.zoom = 1.0 + (frame.intensity - 0.8) * 0.2
    
    def _apply_bar_effects(self, frame: VideoFrame, timestamp: float):
        """Apply effects based on bar sync"""
        beat_interval = 60.0 / self.config.bpm
        bar_duration = beat_interval * self.config.beats_per_bar
        bar_phase = (timestamp % bar_duration) / bar_duration
        
        # Fade in/out across bar
        frame.intensity = 0.7 + 0.3 * math.sin(bar_phase * math.pi)
        
        if bar_phase < 0.1:
            frame.effect = VisualEffect.PULSE
            frame.color_shift = 0.1
    
    def _apply_phrase_effects(self, frame: VideoFrame, timestamp: float):
        """Apply effects based on phrase sync"""
        beat_interval = 60.0 / self.config.bpm
        phrase_duration = beat_interval * self.config.beats_per_bar * self.config.bars_per_phrase
        phrase_phase = (timestamp % phrase_duration) / phrase_duration
        
        # Color shift across phrase
        frame.color_shift = phrase_phase
        frame.intensity = 0.5 + 0.5 * math.sin(phrase_phase * math.pi)
        
        # Transition at phrase boundaries
        if phrase_phase < 0.05:
            frame.effect = VisualEffect.TRANSITION
            frame.effect_params = {"duration": self.config.transition_duration}
    
    def _apply_energy_effects(self, frame: VideoFrame, timestamp: float):
        """Apply effects based on energy analysis"""
        energy = self._find_energy_at(timestamp)
        if not energy:
            return
            
        frame.intensity = energy.energy
        frame.zoom = 1.0 + energy.energy * 0.1
        
        if energy.event_type == "drop":
            frame.effect = VisualEffect.DROP_INTENSITY
            frame.blur = 0.2
        elif energy.event_type == "buildup":
            frame.effect = VisualEffect.BUILDUP
            frame.zoom = 1.0 + frame.intensity * 0.15
    
    def generate_all_frames(self, frame_rate: float = 30.0) -> List[VideoFrame]:
        """Pre-generate all frames for the track"""
        self.frames = []
        duration = self.analyzer.duration
        
        num_frames = int(duration * frame_rate)
        for i in range(num_frames):
            timestamp = i / frame_rate
            frame = self.generate_frame_at(timestamp)
            self.frames.append(frame)
            
        return self.frames
    
    def play(self):
        """Start synchronized playback"""
        self.is_playing = True
        self.current_time = 0.0
    
    def pause(self):
        """Pause playback"""
        self.is_playing = False
    
    def seek(self, timestamp: float):
        """Seek to timestamp"""
        self.current_time = max(0, min(timestamp, self.analyzer.duration))
        
        # Update current indices
        self._current_beat_idx = 0
        for i, beat in enumerate(self.analyzer.beats):
            if beat.time <= self.current_time:
                self._current_beat_idx = i
    
    def get_current_frame(self) -> VideoFrame:
        """Get frame at current playback time"""
        return self.generate_frame_at(self.current_time)
    
    def update(self, delta_time: float = 0.016):
        """Update - call each frame"""
        if not self.is_playing:
            return
            
        self.current_time += delta_time
        
        if self.current_time >= self.analyzer.duration:
            self.is_playing = False
            return
            
        frame = self.get_current_frame()
        
        if self.effect_callback:
            self.effect_callback(frame)
    
    def export_sync_data(self, output_file: str = None) -> Dict:
        """Export sync data as JSON"""
        data = {
            "config": {
                "mode": self.config.mode.value,
                "bpm": self.config.bpm,
                "beats_per_bar": self.config.beats_per_bar,
                "bars_per_phrase": self.config.bars_per_phrase
            },
            "track": {
                "duration": self.analyzer.duration,
                "bpm": self.analyzer.bpm,
                "key": self.analyzer.key
            },
            "beats": [
                {
                    "time": b.time,
                    "bar": b.bar_number,
                    "beat": b.beat_number,
                    "strength": b.strength,
                    "downbeat": b.is_downbeat
                }
                for b in self.analyzer.beats
            ],
            "energy": [
                {
                    "time": e.time,
                    "energy": e.energy,
                    "type": e.event_type
                }
                for e in self.analyzer.energy_events
            ],
            "frames": [
                {
                    "timestamp": f.timestamp,
                    "intensity": f.intensity,
                    "zoom": f.zoom,
                    "color_shift": f.color_shift,
                    "effect": f.effect.value if f.effect else None
                }
                for f in self.frames
            ] if self.frames else []
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        return data
    
    def get_visualization_data(self) -> Dict:
        """Get real-time visualization data"""
        frame = self.get_current_frame()
        
        return {
            "timestamp": self.current_time,
            "bpm": self.config.bpm,
            "beat": self._current_beat_idx % self.config.beats_per_bar,
            "bar": self._current_beat_idx // self.config.beats_per_bar,
            "intensity": frame.intensity,
            "zoom": frame.zoom,
            "color_shift": frame.color_shift,
            "effect": frame.effect.value if frame.effect else None,
            "is_playing": self.is_playing
        }


class VisualTransition:
    """Handles visual transitions between tracks"""
    
    @staticmethod
    def crossfade(start_frame: VideoFrame, end_frame: VideoFrame, 
                  progress: float) -> VideoFrame:
        """Crossfade between two frames"""
        return VideoFrame(
            timestamp=start_frame.timestamp + (end_frame.timestamp - start_frame.timestamp) * progress,
            intensity=start_frame.intensity * (1 - progress) + end_frame.intensity * progress,
            zoom=start_frame.zoom * (1 - progress) + end_frame.zoom * progress,
            color_shift=start_frame.color_shift * (1 - progress) + end_frame.color_shift * progress,
            blur=start_frame.blur * (1 - progress) + end_frame.blur * progress,
            effect=end_frame.effect if progress > 0.5 else start_frame.effect
        )
    
    @staticmethod
    def beat_jump(current_beat: BeatEvent, target_time: float) -> VideoFrame:
        """Create beat-synchronized jump transition"""
        return VideoFrame(
            timestamp=target_time,
            effect=VisualEffect.PULSE,
            effect_params={"strength": current_beat.strength},
            intensity=current_beat.strength,
            zoom=1.0 + current_beat.strength * 0.1
        )
    
    @staticmethod
    def drop_transition(energy: EnergyEvent, duration: float) -> List[VideoFrame]:
        """Create frames for drop transition"""
        frames = []
        steps = int(duration * 30)
        
        for i in range(steps):
            progress = i / steps
            t = energy.time + progress * duration
            
            frames.append(VideoFrame(
                timestamp=t,
                effect=VisualEffect.DROP_INTENSITY,
                effect_params={"intensity": 1.0 - progress * 0.5},
                intensity=1.0,
                zoom=1.0 + (1.0 - progress) * 0.2,
                blur=0.3 * (1 - progress)
            ))
            
        return frames


# Demo / test
def main():
    """Demo the video sync system"""
    print("=" * 50)
    print("🎬 Video Sync System - Demo")
    print("=" * 50)
    
    # Create sync engine with default config
    config = SyncConfig(
        mode=SyncMode.BEAT,
        bpm=128.0,
        beats_per_bar=4,
        bars_per_phrase=8,
        effect_intensity=0.8
    )
    
    sync = VideoSync(config)
    
    # Load demo track
    audio_file = "/Users/johnpeter/ai-dj-project/music/demo.wav"
    
    print(f"\n📂 Loading: {audio_file}")
    result = sync.load_track(audio_file)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Duration: {result.get('duration', 0):.1f}s")
    print(f"   BPM: {result.get('bpm', 0)}")
    print(f"   Key: {result.get('key', 'Unknown')}")
    print(f"   Beats: {result.get('num_beats', 0)}")
    print(f"   Bars: {result.get('num_bars', 0)}")
    
    # Generate some frames
    print(f"\n🎬 Generating sample frames...")
    sample_times = [0.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    
    for t in sample_times:
        if t < sync.analyzer.duration:
            frame = sync.generate_frame_at(t)
            effect_name = frame.effect.value if frame.effect else "none"
            print(f"   t={t:5.1f}s: intensity={frame.intensity:.2f}, "
                  f"zoom={frame.zoom:.2f}, effect={effect_name}")
    
    # Export sync data
    output_file = "/Users/johnpeter/ai-dj-project/src/output/video_sync.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    sync.frames = sync.generate_all_frames(frame_rate=10)  # Lower rate for demo
    sync.export_sync_data(output_file)
    print(f"\n💾 Exported sync data to: {output_file}")
    
    # Visualization data
    print(f"\n📈 Visualization Data (at t=0):")
    sync.current_time = 0
    viz = sync.get_visualization_data()
    for k, v in viz.items():
        print(f"   {k}: {v}")
    
    print("\n✅ Video Sync System ready!")


if __name__ == "__main__":
    main()
