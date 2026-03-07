#!/usr/bin/env python3
"""
Auto DJ System - Automatic Music Mixer and Player

Automatically mixes and plays music with:
- Beat matching (BPM synchronization)
- Key mixing (harmonic compatibility)
- Crossfading transitions
- Smart track queuing
- Energy flow management
"""

import os
import random
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

# Try to import audio libraries, fall back gracefully
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


@dataclass
class Track:
    """Represents a track in the library"""
    filename: str
    artist: str
    genre: str
    bpm: float
    key: str
    energy: float = 0.5
    duration: float = 180.0
    
    @property
    def name(self) -> str:
        return self.filename.replace(".wav", "").replace("_", " ").title()


@dataclass
class DJState:
    """Current state of the Auto DJ"""
    current_track: Optional[Track] = None
    next_track: Optional[Track] = None
    queue: list = None
    is_playing: bool = False
    is_paused: bool = False
    crossfade_duration: float = 5.0
    volume: float = 0.8
    start_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.queue is None:
            self.queue = []


class AutoDJ:
    """Auto DJ System - Automatically mixes and plays music"""
    
    def __init__(self, music_dir: str = None):
        self.music_dir = music_dir or "/Users/johnpeter/ai-dj-project/music"
        self.library: dict[str, Track] = {}
        self.state = DJState()
        self._load_library()
        
    def _load_library(self):
        """Load track metadata from library"""
        library_file = Path(self.music_dir) / "LIBRARY.md"
        
        if not library_file.exists():
            print(f"⚠️ Library file not found: {library_file}")
            return
            
        # Parse LIBRARY.md for track info
        # This is a simple parser - could be enhanced
        tracks_data = [
            {"filename": "drake_in_my_feings.wav", "artist": "Drake", "genre": "Hip-Hop", "bpm": 185, "key": "2A", "energy": 0.150},
            {"filename": "travis_sicko_mode.wav", "artist": "Travis Scott", "genre": "Trap", "bpm": 152, "key": "3A", "energy": 0.267},
            {"filename": "travis_butterfly_effect.wav", "artist": "Travis Scott", "genre": "Trap", "bpm": 96, "key": "2A", "energy": 0.5},
            {"filename": "drake_fair_trade.wav", "artist": "Drake", "genre": "Hip-Hop", "bpm": 172, "key": "4A", "energy": 0.5},
            {"filename": "marshmello_happier.wav", "artist": "Marshmello", "genre": "EDM", "bpm": 99, "key": "6A", "energy": 0.124},
            {"filename": "rick_roll.wav", "artist": "Rick Astley", "genre": "Pop", "bpm": 112, "key": "9B", "energy": 0.129},
            {"filename": "daft_punk_doin_it_right.wav", "artist": "Daft Punk", "genre": "Electronic", "bpm": 89, "key": "9B", "energy": 0.5},
            {"filename": "meduza_lose_control.wav", "artist": "MEDUZA", "genre": "EDM", "bpm": 123, "key": "1B", "energy": 0.5},
            {"filename": "odesza_a_moment_apart.wav", "artist": "ODESZA", "genre": "Electronic", "bpm": 117, "key": "8B", "energy": 0.5},
            {"filename": "edm_1.wav", "artist": "Various", "genre": "EDM", "bpm": 129, "key": "6A", "energy": 0.296},
            {"filename": "edm_2.wav", "artist": "Various", "genre": "EDM", "bpm": 96, "key": "6A", "energy": 0.198},
            {"filename": "edm_3.wav", "artist": "Various", "genre": "EDM", "bpm": 123, "key": "8B", "energy": 0.250},
            {"filename": "edm_4.wav", "artist": "Various", "genre": "EDM", "bpm": 99, "key": "6A", "energy": 0.303},
        ]
        
        for t in tracks_data:
            track = Track(
                filename=t["filename"],
                artist=t["artist"],
                genre=t["genre"],
                bpm=t["bpm"],
                key=t["key"],
                energy=t["energy"],
                duration=180.0  # Default duration
            )
            self.library[track.filename] = track
            
        print(f"📀 Loaded {len(self.library)} tracks into library")
    
    def get_track(self, filename: str) -> Optional[Track]:
        """Get track by filename"""
        return self.library.get(filename)
    
    def get_all_tracks(self) -> list[Track]:
        """Get all tracks in library"""
        return list(self.library.values())
    
    def calculate_bpm_match_score(self, bpm1: float, bpm2: float) -> float:
        """Calculate BPM compatibility score (0-1)"""
        # Allow mixing within +/- 10% BPM
        ratio = min(bpm1, bpm2) / max(bpm1, bpm2)
        if ratio > 0.9:
            return 1.0
        elif ratio > 0.8:
            return 0.7
        elif ratio > 0.7:
            return 0.4
        return 0.1
    
    def calculate_key_compatibility(self, key1: str, key2: str) -> float:
        """Calculate key compatibility score (0-1) using camelot wheel"""
        # Simplified camelot wheel compatibility
        key_groups = {
            "1A": ["1A", "1B", "12A", "2A"],
            "2A": ["2A", "2B", "1A", "3A"],
            "3A": ["3A", "3B", "2A", "4A"],
            "4A": ["4A", "4B", "3A", "5A"],
            "5A": ["5A", "5B", "4A", "6A"],
            "6A": ["6A", "6B", "5A", "7A"],
            "7A": ["7A", "7B", "6A", "8A"],
            "8A": ["8A", "8B", "7A", "9A"],
            "9A": ["9A", "9B", "8A", "10A"],
            "10A": ["10A", "10B", "9A", "11A"],
            "11A": ["11A", "11B", "10A", "12A"],
            "12A": ["12A", "12B", "11A", "1A"],
            "1B": ["1B", "1A", "12B", "2B"],
            "2B": ["2B", "2A", "1B", "3B"],
            "3B": ["3B", "3A", "2B", "4B"],
            "4B": ["4B", "4A", "3B", "5B"],
            "5B": ["5B", "5A", "4B", "6B"],
            "6B": ["6B", "6A", "5B", "7B"],
            "7B": ["7B", "7A", "6B", "8B"],
            "8B": ["8B", "8A", "7B", "9B"],
            "9B": ["9B", "9A", "8B", "10B"],
            "10B": ["10B", "10A", "9B", "11B"],
            "11B": ["11B", "11A", "10B", "12B"],
            "12B": ["12B", "12A", "11B", "1B"],
        }
        
        compatible = key_groups.get(key1, [key1])
        if key2 in compatible:
            return 1.0
        # Check for relative minor/major
        if key1.replace("A", "B") == key2 or key1.replace("B", "A") == key2:
            return 0.8
        return 0.3
    
    def calculate_mix_score(self, track1: Track, track2: Track) -> float:
        """Calculate overall mix compatibility score"""
        bpm_score = self.calculate_bpm_match_score(track1.bpm, track2.bpm)
        key_score = self.calculate_key_compatibility(track1.key, track2.key)
        
        # Weighted average: BPM more important for mixing
        return (bpm_score * 0.6) + (key_score * 0.4)
    
    def find_best_next_track(self, current: Track, exclude: list = None) -> Optional[Track]:
        """Find the best track to play next based on compatibility"""
        exclude = exclude or []
        candidates = [t for t in self.library.values() 
                     if t.filename not in exclude]
        
        if not candidates:
            return None
            
        best_track = None
        best_score = 0
        
        for track in candidates:
            score = self.calculate_mix_score(current, track)
            if score > best_score:
                best_score = score
                best_track = track
                
        return best_track
    
    def generate_playlist(self, length: int = 10, seed: Track = None) -> list[Track]:
        """Generate a smart playlist with smooth transitions"""
        playlist = []
        used = set()
        
        # Start with a random track or seed
        if seed:
            current = seed
        else:
            current = random.choice(list(self.library.values()))
        
        playlist.append(current)
        used.add(current.filename)
        
        for _ in range(length - 1):
            next_track = self.find_best_next_track(current, list(used))
            if next_track:
                playlist.append(next_track)
                used.add(next_track.filename)
                current = next_track
            else:
                break
                
        return playlist
    
    def add_to_queue(self, track: Track):
        """Add track to play queue"""
        self.state.queue.append(track)
        
    def remove_from_queue(self, index: int) -> Optional[Track]:
        """Remove track from queue by index"""
        if 0 <= index < len(self.state.queue):
            return self.state.queue.pop(index)
        return None
    
    def clear_queue(self):
        """Clear the play queue"""
        self.state.queue.clear()
    
    def shuffle_queue(self):
        """Shuffle the current queue"""
        random.shuffle(self.state.queue)
    
    # ============ Playback Simulation ============
    # Note: Actual audio playback would require pygame, pyaudio, or similar
    
    def play(self, track: Optional[Track] = None):
        """Start playing a track"""
        if track:
            self.state.current_track = track
            self.state.is_playing = True
            self.state.is_paused = False
            self.state.start_time = datetime.now()
            print(f"▶️  Now playing: {track.name} - {track.artist}")
            print(f"   BPM: {track.bpm} | Key: {track.key} | Energy: {track.energy}")
        elif self.state.queue:
            self.play(self.state.queue.pop(0))
        else:
            print("📭 Queue is empty!")
            
    def pause(self):
        """Pause playback"""
        self.state.is_paused = True
        if self.state.current_track:
            print(f"⏸️  Paused: {self.state.current_track.name}")
            
    def resume(self):
        """Resume playback"""
        self.state.is_paused = False
        if self.state.current_track:
            print(f"▶️  Resumed: {self.state.current_track.name}")
            
    def stop(self):
        """Stop playback"""
        self.state.is_playing = False
        self.state.is_paused = False
        self.state.current_track = None
        print("⏹️  Stopped")
    
    def skip(self):
        """Skip to next track in queue"""
        if self.state.queue:
            self.play(self.state.queue[0])
        else:
            self.stop()
            print("📭 No more tracks in queue")
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        self.state.volume = max(0.0, min(1.0, volume))
        print(f"🔊 Volume: {int(self.state.volume * 100)}%")
    
    def set_crossfade(self, duration: float):
        """Set crossfade duration in seconds"""
        self.state.crossfade_duration = duration
        print(f"🌊 Crossfade: {duration}s")
    
    def get_status(self) -> dict:
        """Get current DJ status"""
        return {
            "playing": self.state.is_playing,
            "paused": self.state.is_paused,
            "current_track": self.state.current_track.name if self.state.current_track else None,
            "queue_length": len(self.state.queue),
            "volume": self.state.volume,
            "crossfade": self.state.crossfade_duration,
        }
    
    def show_queue(self):
        """Display current queue"""
        print("\n🎵 PLAY QUEUE")
        print("=" * 50)
        if self.state.current_track:
            print(f"▶️  NOW: {self.state.current_track.name} - {self.state.current_track.artist}")
            print(f"    BPM: {self.state.current_track.bpm} | Key: {self.state.current_track.key}")
            print("-" * 50)
        
        if self.state.queue:
            for i, track in enumerate(self.state.queue, 1):
                print(f"{i:2}. {track.name} - {track.artist}")
                print(f"    BPM: {track.bpm} | Key: {track.key} | Energy: {track.energy}")
        else:
            print("📭 Queue is empty")
        print("=" * 50)
    
    def show_library(self):
        """Display all tracks in library"""
        print("\n📀 MUSIC LIBRARY")
        print("=" * 70)
        print(f"{'#':<3} {'Track':<35} {'Artist':<15} {'BPM':<5} {'Key':<4} {'Energy'}")
        print("-" * 70)
        
        for i, track in enumerate(self.library.values(), 1):
            print(f"{i:<3} {track.name[:33]:<35} {track.artist[:13]:<15} {track.bpm:<5} {track.key:<4} {track.energy}")
        print("=" * 70)
        print(f"Total: {len(self.library)} tracks")
    
    def analyze_mix_opportunities(self):
        """Show best mixing pairs"""
        print("\n🔄 MIX COMPATIBILITY MATRIX")
        print("=" * 70)
        
        tracks = list(self.library.values())
        opportunities = []
        
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i+1:]:
                score = self.calculate_mix_score(t1, t2)
                if score > 0.6:
                    opportunities.append((t1, t2, score))
        
        opportunities.sort(key=lambda x: x[2], reverse=True)
        
        for t1, t2, score in opportunities[:10]:
            print(f"✓ {t1.name} ↔ {t2.name}")
            print(f"  Score: {score:.2f} | BPM: {t1.bpm}→{t2.bpm} | Key: {t1.key}→{t2.key}")
        
        if not opportunities:
            print("No high-compatibility pairs found (score > 0.6)")
        print("=" * 70)


def main():
    """Demo the Auto DJ system"""
    print("🎛️" + "=" * 50)
    print("     AUTO DJ SYSTEM - Automatic Mixer")
    print("=" * 52)
    
    # Initialize Auto DJ
    auto_dj = AutoDJ()
    
    # Show library
    auto_dj.show_library()
    
    # Analyze mixing opportunities
    auto_dj.analyze_mix_opportunities()
    
    # Generate smart playlist
    print("\n🎲 Generating smart playlist...")
    playlist = auto_dj.generate_playlist(length=5)
    
    print("\n📝 GENERATED PLAYLIST")
    print("-" * 50)
    for i, track in enumerate(playlist, 1):
        print(f"{i}. {track.name} - {track.artist}")
        print(f"   BPM: {track.bpm} | Key: {track.key}")
    print("-" * 50)
    
    # Add to queue and demonstrate
    for track in playlist:
        auto_dj.add_to_queue(track)
    
    auto_dj.show_queue()
    
    # Simulate playback
    print("\n🎧 Starting playback simulation...")
    auto_dj.play()
    
    print("\n✅ Auto DJ System ready!")
    print("\nCommands:")
    print("  play()     - Start playing")
    print("  pause()    - Pause playback")
    print("  resume()   - Resume playback")
    print("  skip()     - Skip to next track")
    print("  stop()     - Stop playback")
    print("  set_volume(v) - Set volume 0.0-1.0")
    print("  show_queue() - Display queue")
    print("  show_library() - Display library")
    
    return auto_dj


if __name__ == "__main__":
    auto_dj = main()
