"""
Live Performance System for AI DJ
=================================
High-level live show management, intelligent track selection,
crowd interaction, real-time visualization, and broadcast capabilities.
"""

import numpy as np
import time
import threading
import json
import os
from typing import Optional, Dict, List, Callable, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from pathlib import Path


# =============================================================================
# TRACK & SET MANAGEMENT
# =============================================================================

class TrackSource(Enum):
    LOCAL = "local"
    GENERATED = "generated"
    STEMS = "stems"
    LIVE_INPUT = "live_input"


@dataclass
class Track:
    """Represents a track in the live performance."""
    id: str
    name: str
    artist: str = ""
    duration: float = 0.0
    bpm: float = 120.0
    key: int = 0
    key_name: str = "C"
    energy: float = 0.5
    mood: str = "neutral"
    genre: str = ""
    audio_data: Optional[np.ndarray] = None
    source: TrackSource = TrackSource.LOCAL
    file_path: str = ""
    hot_cue_positions: List[float] = field(default_factory=list)
    loop_points: List[Tuple[float, float]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"track_{int(time.time() * 1000)}"


@dataclass
class Setlist:
    """A planned set/list of tracks for a live performance."""
    id: str
    name: str
    tracks: List[Track] = field(default_factory=list)
    start_time: Optional[float] = None
    expected_duration: float = 0.0
    notes: str = ""
    
    def total_duration(self) -> float:
        return sum(t.duration for t in self.tracks)
    
    def add_track(self, track: Track, position: int = None):
        if position is None:
            self.tracks.append(track)
        else:
            self.tracks.insert(position, track)
    
    def remove_track(self, track_id: str):
        self.tracks = [t for t in self.tracks if t.id != track_id]
    
    def reorder(self, from_index: int, to_index: int):
        if 0 <= from_index < len(self.tr0 <= to_index < len(self.tracks) and acks):
            track = self.tracks.pop(from_index)
            self.tracks.insert(to_index, track)


# =============================================================================
# ENERGY & CROWD ANALYSIS
# =============================================================================

class CrowdEnergyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PEAK = "peak"


class CrowdAnalyzer:
    """Analyzes crowd energy and recommends track selections."""
    
    def __init__(self, history_size: int = 60):
        self.energy_history = deque(maxlen=history_size)
        self.avg_energy = 0.5
        self.energy_trend: List[str] = []
        self.last_energy_update = 0
        self.smoothing_factor = 0.1
        
    def update_energy(self, energy: float, timestamp: float = None):
        """Update with new energy reading (0.0 to 1.0)."""
        if timestamp is None:
            timestamp = time.time()
        
        smoothed = energy
        if self.energy_history:
            smoothed = self.smoothing_factor * energy + (1 - self.smoothing_factor) * self.avg_energy
        
        self.energy_history.append({
            "energy": smoothed,
            "timestamp": timestamp
        })
        self.avg_energy = smoothed
        self.last_energy_update = timestamp
        
        self._update_trend()
        
    def _update_trend(self):
        """Calculate energy trend from recent history."""
        if len(self.energy_history) < 10:
            self.energy_trend = ["stable"]
            return
            
        recent = list(self.energy_history)[-10:]
        first_half = np.mean([e["energy"] for e in recent[:5]])
        second_half = np.mean([e["energy"] for e in recent[5:]])
        
        diff = second_half - first_half
        if diff > 0.1:
            self.energy_trend = ["rising"]
        elif diff < -0.1:
            self.energy_trend = ["falling"]
        else:
            self.energy_trend = ["stable"]
    
    def get_energy_level(self) -> CrowdEnergyLevel:
        """Get current energy level category."""
        if self.avg_energy < 0.3:
            return CrowdEnergyLevel.LOW
        elif self.avg_energy < 0.6:
            return CrowdEnergyLevel.MEDIUM
        elif self.avg_energy < 0.85:
            return CrowdEnergyLevel.HIGH
        else:
            return CrowdEnergyLevel.PEAK
    
    def get_recommendation(self) -> str:
        """Get track recommendation based on energy analysis."""
        level = self.get_energy_level()
        
        if level == CrowdEnergyLevel.LOW:
            return "build_up"
        elif level == CrowdEnergyLevel.MEDIUM:
            return "maintain_or_build"
        elif level == CrowdEnergyLevel.HIGH:
            return "maintain_or_peak"
        else:
            return "peak_or_drop"
    def get_stats(self) -> Dict:
        return {
            "current_energy": round(self.avg_energy, 2),
            "level": self.get_energy_level().value,
            "trend": self.energy_trend[-1] if self.energy_trend else "unknown",
            "samples": len(self.energy_history)
        }


# =============================================================================
# INTELLIGENT TRACK SELECTION
# =============================================================================

class TrackSelector:
    """Intelligent track selection for live performances."""
    
    def __init__(self, library: List[Track] = None):
        self.library = library or []
        self.played_tracks: set = set()
        
    def add_to_library(self, track: Track):
        self.library.append(track)
    
    def remove_from_library(self, track_id: str):
        self.library = [t for t in self.library if t.id != track_id]
    
    def select_next_track(
        self,
        current_track: Optional[Track] = None,
        energy_target: str = "maintain",
        key_compatibility: bool = True,
        genre_variety: bool = True
    ) -> Optional[Track]:
        """Select the next track based on current track and targets."""
        
        if not self.library:
            return None
        
        # Filter out already played tracks
        candidates = [t for t in self.library if t.id not in self.played_tracks]
        
        if not candidates:
            # Reset if we've played everything
            self.played_tracks.clear()
            candidates = self.library
        
        if not candidates:
            return None
        
        # Score each candidate
        scored = []
        for track in candidates:
            score = self._calculate_track_score(
                track, current_track, energy_target, 
                key_compatibility, genre_variety
            )
            scored.append((track, score))
        
        # Sort by score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[0][0] if scored else None
    
    def _calculate_track_score(
        self,
        track: Track,
        current_track: Optional[Track],
        energy_target: str,
        key_compatible: bool,
        genre_variety: bool
    ) -> float:
        score = 50.0  # Base score
        
        if current_track:
            # Energy matching
            energy_diff = abs(track.energy - current_track.energy)
            if energy_target == "maintain":
                score -= energy_diff * 30
            elif energy_target == "build":
                if track.energy > current_track.energy:
                    score += 20
                else:
                    score -= 20
            elif energy_target == "drop":
                if track.energy < current_track.energy:
                    score += 15
                else:
                    score -= 10
            
            # Key compatibility (Camelot wheel)
            if key_compatible:
                key_score = self._key_compatibility(current_track.key, track.key)
                score += key_score * 10
            
            # Genre variety
            if genre_variety and current_track.genre:
                if track.genre != current_track.genre:
                    score += 5
        
        # Random factor to avoid repetitive patterns
        score += np.random.uniform(-5, 5)
        
        return score
    
    def _key_compatibility(self, key1: int, key2: int) -> float:
        """Calculate Camelot wheel key compatibility (simplified)."""
        # Same key = perfect
        if key1 == key2:
            return 1.0
        
        # Relative major/minor (+1 or -1 in Camelot)
        wheel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # C through B
        diff = abs(key1 - key2)
        
        if diff == 1 or diff == 11:  # Adjacent
            return 0.8
        elif diff == 7:  # Perfect 5th
            return 0.9
        elif diff == 5:  # Perfect 4th
            return 0.9
        else:
            return 0.3
    
    def mark_played(self, track_id: str):
        self.played_tracks.add(track_id)
    
    def reset_history(self):
        self.played_tracks.clear()


# =============================================================================
# VISUAL PERFORMANCE FEEDBACK
# =============================================================================

class VisualFeedback:
    """Visual feedback system for live performances."""
    
    def __init__(self):
        self.beat_pulse = 0.0
        self.transition_flash = 0.0
        self.energy_glow = 0.5
        self.waveform_data: Optional[np.ndarray] = None
        self.spectrum_data: Optional[np.ndarray] = None
        self.beat_indicator = False
        self.last_beat_time = 0
        self.beats_since_last = 0
        
    def on_beat(self, beat_number: int):
        """Called on each beat."""
        self.beat_pulse = 1.0
        self.beat_indicator = True
        self.beats_since_last = 0
        self.last_beat_time = time.time()
        
    def update(self, dt: float):
        """Update visual state (call each frame)."""
        # Decay beat pulse
        self.beat_pulse = max(0.0, self.beat_pulse - dt * 5)
        
        # Decay transition flash
        self.transition_flash = max(0.0, self.transition_flash - dt * 2)
        
        # Update beat indicator
        if time.time() - self.last_beat_time > 0.5:
            self.beat_indicator = False
        
        self.beats_since_last += 1
    
    def trigger_transition(self):
        """Trigger transition visual effect."""
        self.transition_flash = 1.0
    
    def update_energy(self, energy: float):
        """Update energy glow (0.0 to 1.0)."""
        self.energy_glow = energy
    
    def update_waveform(self, audio: np.ndarray):
        """Update waveform display data."""
        # Downsample for display
        if len(audio) > 512:
            step = len(audio) // 512
            self.waveform_data = audio[::step][:512]
        else:
            self.waveform_data = audio[:512]
    
    def update_spectrum(self, spectrum: np.ndarray):
        """Update spectrum analyzer data."""
        if len(spectrum) > 64:
            step = len(spectrum) // 64
            self.spectrum_data = spectrum[::step][:64]
        else:
            self.spectrum_data = spectrum[:64]
    
    def get_display_data(self) -> Dict:
        return {
            "beat_pulse": round(self.beat_pulse, 2),
            "beat_on": self.beat_indicator,
            "transition_flash": round(self.transition_flash, 2),
            "energy_glow": round(self.energy_glow, 2),
            "waveform": self.waveform_data.tolist() if self.waveform_data is not None else [],
            "spectrum": self.spectrum_data.tolist() if self.spectrum_data is not None else []
        }


# =============================================================================
# RECORDING & BROADCASTING
# =============================================================================

class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


class RecordingSystem:
    """Records live performances."""
    
    def __init__(self, output_dir: str = "./recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = RecordingState.IDLE
        self.audio_buffer: List[np.ndarray] = []
        self.start_time: Optional[float] = None
        self.duration = 0.0
        self.current_file: Optional[Path] = None
        self.sample_rate = 44100
        
    def start_recording(self, session_name: str = None) -> str:
        """Start a new recording session."""
        if self.state == RecordingState.RECORDING:
            return str(self.current_file) if self.current_file else ""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if session_name:
            filename = f"{session_name}_{timestamp}.wav"
        else:
            filename = f"live_set_{timestamp}.wav"
        
        self.current_file = self.output_dir / filename
        self.audio_buffer = []
        self.start_time = time.time()
        self.state = RecordingState.RECORDING
        
        return str(self.current_file)
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save to file."""
        if self.state == RecordingState.IDLE:
            return None
        
        if self.audio_buffer:
            self._save_audio()
        
        filepath = str(self.current_file) if self.current_file else None
        self.state = RecordingState.IDLE
        self.duration = time.time() - self.start_time if self.start_time else 0
        
        return filepath
    
    def pause_recording(self):
        """Pause the current recording."""
        if self.state == RecordingState.RECORDING:
            self.state = RecordingState.PAUSED
    
    def resume_recording(self):
        """Resume a paused recording."""
        if self.state == RecordingState.PAUSED:
            self.state = RecordingState.RECORDING
    
    def add_audio(self, audio: np.ndarray):
        """Add audio data to the recording."""
        if self.state == RecordingState.RECORDING:
            self.audio_buffer.append(audio.copy())
            self.duration = time.time() - self.start_time if self.start_time else 0
    
    def _save_audio(self):
        """Save accumulated audio to file."""
        if not self.audio_buffer:
            return
        
        try:
            import scipy.io.wavfile as wav
            audio = np.concatenate(self.audio_buffer)
            # Convert stereo to mono if needed
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Normalize
            audio = audio / max(np.abs(audio).max(), 1.0)
            wav.write(str(self.current_file), self.sample_rate, audio.astype(np.float32))
        except Exception as e:
            print(f"Error saving recording: {e}")
    
    def get_status(self) -> Dict:
        return {
            "state": self.state.value,
            "file": str(self.current_file) if self.current_file else None,
            "duration": round(self.duration, 2),
            "samples": sum(len(a) for a in self.audio_buffer)
        }


# =============================================================================
# LIVE PERFORMANCE ORCHESTRATOR
# =============================================================================

class PerformanceState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    PLAYING = "playing"
    TRANSITIONING = "transitioning"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass
class PerformanceEvent:
    """An event in the live performance."""
    timestamp: float
    event_type: str  # "track_change", "effect_change", "energy_update", etc.
    data: Dict = field(default_factory=dict)


class LivePerformance:
    """
    Main orchestrator for live performances.
    Coordinates tracks, transitions, analysis, and feedback.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
        output_dir: str = "./recordings"
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Track management
        self.setlist: Optional[Setlist] = None
        self.current_track_index = -1
        self.next_track_index = -1
        
        # Analysis & selection
        self.crowd_analyzer = CrowdAnalyzer()
        self.track_selector = TrackSelector()
        
        # Visual & recording
        self.visual = VisualFeedback()
        self.recording = RecordingSystem(output_dir)
        
        # Performance state
        self.state = PerformanceState.IDLE
        self.start_time: Optional[float] = None
        self.elapsed_time = 0.0
        self.is_playing = False
        
        # Events
        self.events: List[PerformanceEvent] = []
        self.on_track_change: Optional[Callable[[Track, Track], None]] = None
        self.on_energy_change: Optional[Callable[[float], None]] = None
        
        # Audio processing
        self._audio_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
    def load_setlist(self, setlist: Setlist):
        """Load a setlist for performance."""
        self.setlist = setlist
        self.track_selector.library = setlist.tracks
        self.current_track_index = -1
        self.next_track_index = 0 if setlist.tracks else -1
        self._log_event("setlist_loaded", {"track_count": len(setlist.tracks)})
    
    def load_track(self, track: Track) -> bool:
        """Load a single track for immediate playback."""
        with self._lock:
            if self.setlist and self.current_track_index >= 0:
                # Update selector that current track is being played
                self.track_selector.mark_played(track.id)
            
            self.current_track_index = self.next_track_index
            self._log_event("track_loaded", {"track": track.name, "bpm": track.bpm})
            return True
    
    def play(self):
        """Start/resume playback."""
        if not self.setlist or not self.setlist.tracks:
            return
        
        self.is_playing = True
        self.state = PerformanceState.PLAYING
        
        if self.start_time is None:
            self.start_time = time.time()
        
        self._log_event("play", {"track_index": self.current_track_index})
    
    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.state = PerformanceState.PAUSED
        self._log_event("pause", {})
    
    def stop(self):
        """Stop playback completely."""
        self.is_playing = False
        self.state = PerformanceState.IDLE
        self.start_time = None
        self._log_event("stop", {})
    
    def next_track(self, auto_select: bool = True) -> Optional[Track]:
        """Move to the next track."""
        if not self.setlist:
            return None
        
        old_track = None
        if self.current_track_index >= 0 and self.current_track_index < len(self.setlist.tracks):
            old_track = self.setlist.tracks[self.current_track_index]
            self.track_selector.mark_played(old_track.id)
        
        if auto_select:
            # Auto-select based on crowd energy
            recommendation = self.crowd_analyzer.get_recommendation()
            energy_target = "build" if recommendation == "build_up" else "maintain"
            
            current = self.setlist.tracks[self.current_track_index] if self.current_track_index >= 0 else None
            next_track = self.track_selector.select_next_track(
                current, energy_target=energy_target
            )
            
            if next_track and self.on_track_change and old_track:
                self.on_track_change(old_track, next_track)
            
            return next_track
        else:
            # Manual - go to next in setlist
            if self.current_track_index + 1 < len(self.setlist.tracks):
                self.current_track_index += 1
                return self.setlist.tracks[self.current_track_index]
        
        return None
    
    def previous_track(self) -> Optional[Track]:
        """Go to the previous track."""
        if not self.setlist or self.current_track_index <= 0:
            return None
        
        self.current_track_index -= 1
        return self.setlist.tracks[self.current_track_index]
    
    def jump_to_track(self, index: int) -> Optional[Track]:
        """Jump to a specific track in the setlist."""
        if not self.setlist or index < 0 or index >= len(self.setlist.tracks):
            return None
        
        self.current_track_index = index
        return self.setlist.tracks[index]
    
    def update_energy(self, energy: float):
        """Update crowd energy reading."""
        self.crowd_analyzer.update_energy(energy)
        self.visual.update_energy(energy)
        
        if self.on_energy_change:
            self.on_energy_change(energy)
    
    def start_recording(self, session_name: str = None) -> str:
        """Start recording the performance."""
        name = session_name or (self.setlist.name if self.setlist else "live")
        return self.recording.start_recording(name)
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save."""
        return self.recording.stop_recording()
    
    def get_current_track(self) -> Optional[Track]:
        """Get the currently playing track."""
        if self.setlist and 0 <= self.current_track_index < len(self.setlist.tracks):
            return self.setlist.tracks[self.current_track_index]
        return None
    
    def get_status(self) -> Dict:
        """Get comprehensive performance status."""
        current = self.get_current_track()
        
        return {
            "state": self.state.value,
            "is_playing": self.is_playing,
            "elapsed_time": round(self.elapsed_time, 2),
            "current_track": {
                "name": current.name if current else None,
                "artist": current.artist if current else None,
                "bpm": current.bpm if current else None,
                "duration": current.duration if current else None,
                "index": self.current_track_index,
                "total": len(self.setlist.tracks) if self.setlist else 0
            } if current else None,
            "next_track_index": self.next_track_index,
            "crowd": self.crowd_analyzer.get_stats(),
            "visual": self.visual.get_display_data(),
            "recording": self.recording.get_status(),
            "event_count": len(self.events)
        }
    
    def _log_event(self, event_type: str, data: Dict):
        """Log a performance event."""
        event = PerformanceEvent(
            timestamp=time.time() - (self.start_time or time.time()),
            event_type=event_type,
            data=data
        )
        self.events.append(event)
    
    def export_setlist(self, filepath: str):
        """Export setlist to JSON file."""
        if not self.setlist:
            return
        
        data = {
            "id": self.setlist.id,
            "name": self.setlist.name,
            "expected_duration": self.setlist.expected_duration,
            "notes": self.setlist.notes,
            "tracks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "artist": t.artist,
                    "duration": t.duration,
                    "bpm": t.bpm,
                    "key": t.key_name,
                    "energy": t.energy,
                    "genre": t.genre,
                    "tags": t.tags
                }
                for t in self.setlist.tracks
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_setlist(self, filepath: str) -> Setlist:
        """Import setlist from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracks = [
            Track(
                id=t["id"],
                name=t["name"],
                artist=t.get("artist", ""),
                duration=t.get("duration", 0),
                bpm=t.get("bpm", 120),
                key=t.get("key", 0),
                key_name=t.get("key", "C"),
                energy=t.get("energy", 0.5),
                genre=t.get("genre", ""),
                tags=t.get("tags", [])
            )
            for t in data.get("tracks", [])
        ]
        
        setlist = Setlist(
            id=data.get("id", ""),
            name=data.get("name", "Imported Set"),
            tracks=tracks,
            expected_duration=data.get("expected_duration", 0),
            notes=data.get("notes", "")
        )
        
        self.load_setlist(setlist)
        return setlist


# =============================================================================
# SIMPLE DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI DJ Live Performance System Demo")
    print("=" * 60)
    
    # Create demo tracks
    tracks = [
        Track(
            id="t1",
            name="Opening Energy",
            artist="AI Generator",
            duration=300,
            bpm=120,
            key=0,
            key_name="C",
            energy=0.4,
            genre="house"
        ),
        Track(
            id="t2",
            name="Building Up",
            artist="AI Generator",
            duration=320,
            bpm=124,
            key=7,
            key_name="G",
            energy=0.6,
            genre="house"
        ),
        Track(
            id="t3",
            name="Peak Time",
            artist="AI Generator",
            duration=340,
            bpm=128,
            key=2,
            key_name="D",
            energy=0.85,
            genre="techno"
        ),
        Track(
            id="t4",
            name="Cool Down",
            artist="AI Generator",
            duration=280,
            bpm=116,
            key=9,
            key_name="A",
            energy=0.35,
            genre="melodic"
        ),
    ]
    
    # Create setlist
    setlist = Setlist(
        id="demo_set_001",
        name="Demo Live Set",
        tracks=tracks,
        expected_duration=1240
    )
    
    # Create performance system
    performance = LivePerformance(sample_rate=44100, buffer_size=1024)
    performance.load_setlist(setlist)
    
    print(f"\nLoaded setlist: {setlist.name}")
    print(f"  Tracks: {len(setlist.tracks)}")
    print(f"  Duration: {setlist.total_duration() // 60}min")
    
    # Simulate some energy readings
    print("\nSimulating crowd energy...")
    energies = [0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    for e in energies:
        performance.update_energy(e)
        stats = performance.crowd_analyzer.get_stats()
        print(f"  Energy: {e:.2f} -> Level: {stats['level']}, Trend: {stats['trend']}")
    
    # Test track selection
    print("\nTesting intelligent track selection...")
    current = tracks[0]
    for i in range(3):
        next_track = performance.track_selector.select_next_track(
            current,
            energy_target="build",
            key_compatibility=True
        )
        if next_track:
            print(f"  {current.name} -> {next_track.name} (BPM: {next_track.bpm}, Energy: {next_track.energy})")
            current = next_track
    
    # Test recording
    print("\nTesting recording system...")
    rec_file = performance.start_recording("demo_session")
    print(f"  Recording started: {rec_file}")
    
    # Simulate some audio
    dummy_audio = np.random.randn(1024).astype(np.float32) * 0.1
    performance.recording.add_audio(dummy_audio)
    performance.recording.add_audio(dummy_audio)
    
    rec_file = performance.stop_recording()
    print(f"  Recording stopped: {rec_file}")
    
    # Get full status
    print("\nPerformance Status:")
    status = performance.get_status()
    print(f"  State: {status['state']}")
    print(f"  Crowd Energy: {status['crowd']['current_energy']}")
    print(f"  Visual Beat Pulse: {status['visual']['beat_pulse']}")
    print(f"  Events Logged: {status['event_count']}")
    
    # Export setlist
    print("\nExporting setlist...")
    performance.export_setlist("/tmp/demo_setlist.json")
    print("  Saved to /tmp/demo_setlist.json")
    
    print("\nDemo complete!")
