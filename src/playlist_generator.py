#!/usr/bin/env python3
"""
Playlist Generator for AI DJ Project

Generates playlists based on genre, mood, energy level, BPM, duration,
and transition compatibility. Supports mixing genres with smooth transitions.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os

# Try to import from existing modules
try:
    from genre_db import Genre, GENRES, EnergyLevel, Mood
    _HAS_GENRE_DB = True
except ImportError:
    _HAS_GENRE_DB = False
    # Fallback enums if genre_db not available
    class EnergyLevel:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        BUILDUP = "buildup"
        DROPCENTER = "drop_center"
    
    class Mood:
        CALM = "calm"
        NEUTRAL = "neutral"
        UPBEAT = "upbeat"
        AGGRESSIVE = "aggressive"
        DARK = "dark"
        EUPHORIC = "euphoric"
        MELANCHOLIC = "melancholic"
        MYSTERIOUS = "mysterious"


@dataclass
class Track:
    """Represents a track in the playlist."""
    id: str
    title: str
    artist: str
    genre: str
    bpm: int
    key: str
    duration_seconds: int
    energy: str = "medium"
    mood: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    @property
    def duration_formatted(self) -> str:
        """Return duration as MM:SS format."""
        minutes = self.duration_seconds // 60
        seconds = self.duration_seconds % 60
        return f"{minutes}:{seconds:02d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert track to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "genre": self.genre,
            "bpm": self.bpm,
            "key": self.key,
            "duration_seconds": self.duration_seconds,
            "duration_formatted": self.duration_formatted,
            "energy": self.energy,
            "mood": self.mood,
            "file_path": self.file_path,
            "tags": self.tags
        }


@dataclass
class Playlist:
    """Represents a generated playlist."""
    name: str
    description: str = ""
    tracks: List[Track] = field(default_factory=list)
    total_duration_seconds: int = 0
    genres: List[str] = field(default_factory=list)
    average_bpm: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def duration_formatted(self) -> str:
        """Return total duration as HH:MM:SS or MM:SS format."""
        hours = self.total_duration_seconds // 3600
        minutes = (self.total_duration_seconds % 3600) // 60
        seconds = self.total_duration_seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
    
    @property
    def track_count(self) -> int:
        """Return number of tracks."""
        return len(self.tracks)
    
    def add_track(self, track: Track) -> None:
        """Add a track to the playlist."""
        self.tracks.append(track)
        self.total_duration_seconds += track.duration_seconds
        if track.genre not in self.genres:
            self.genres.append(track.genre)
        self._recalculate()
    
    def _recalculate(self) -> None:
        """Recalculate playlist statistics."""
        if self.tracks:
            self.average_bpm = sum(t.bpm for t in self.tracks) / len(self.tracks)
        else:
            self.average_bpm = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert playlist to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "tracks": [t.to_dict() for t in self.tracks],
            "track_count": self.track_count,
            "total_duration_seconds": self.total_duration_seconds,
            "total_duration_formatted": self.duration_formatted,
            "genres": self.genres,
            "average_bpm": round(self.average_bpm, 1),
            "created_at": self.created_at
        }


class PlaylistGenerator:
    """Generates playlists based on various criteria."""
    
    # Musical key compatibility for harmonic mixing
    KEY_COMPATIBILITY = {
        # Major keys - compatible keys for smooth transitions
        "C": ["C", "G", "F", "Am", "Em", "Dm"],
        "G": ["G", "D", "C", "Em", "Bm", "Am"],
        "D": ["D", "A", "G", "Bm", "F#m", "Em"],
        "A": ["A", "E", "D", "F#m", "C#m", "Bm"],
        "E": ["E", "B", "A", "C#m", "G#m", "F#m"],
        "B": ["B", "F#", "E", "G#m", "D#m", "C#m"],
        "F": ["F", "C", "Bb", "Dm", "Am", "Gm"],
        "Bb": ["Bb", "F", "Eb", "Gm", "Dm", "Cm"],
        "Eb": ["Eb", "Bb", "Ab", "Cm", "Fm", "Gm"],
        "Ab": ["Ab", "Eb", "Db", "Fm", "Bbm", "Cm"],
        "Db": ["Db", "Ab", "Gb", "Bbm", "Ebm", "Fm"],
        "Gb": ["Gb", "Db", "Ebm", "Bbm", "Abm", "Bbm"],
        # Minor keys
        "Am": ["Am", "Em", "Dm", "C", "G", "F"],
        "Dm": ["Dm", "Am", "Gm", "F", "A", "C"],
        "Em": ["Em", "Bm", "Am", "G", "D", "A"],
        "Bm": ["Bm", "F#m", "Em", "D", "A", "E"],
        "F#m": ["F#m", "C#m", "Bm", "A", "E", "B"],
        "Gm": ["Gm", "Dm", "Cm", "Bb", "F", "Eb"],
        "Fm": ["Fm", "Cm", "Bb", "Ab", "Eb", "Db"],
        "Cm": ["Cm", "Gm", "Fm", "Eb", "Bb", "Ab"],
        "Bbm": ["Bbm", "Fm", "Ebm", "Db", "Gb", "Ab"],
        "Ebm": ["Ebm", "Bbm", "Ab", "Gb", "Db", "Gb"],
        "Abm": ["Abm", "Ebm", "Bbm", "Gb", "Db", "Ebm"],
    }
    
    def __init__(self, library_path: Optional[str] = None):
        """Initialize the playlist generator.
        
        Args:
            library_path: Path to track library JSON file. If None, uses demo tracks.
        """
        self.library_path = library_path
        self.tracks: List[Track] = []
        self._load_library()
    
    def _load_library(self) -> None:
        """Load track library from file or use demo tracks."""
        if self.library_path and Path(self.library_path).exists():
            with open(self.library_path, 'r') as f:
                data = json.load(f)
                self.tracks = [Track(**t) for t in data.get('tracks', [])]
        else:
            self._load_demo_tracks()
    
    def _load_demo_tracks(self) -> None:
        """Load demo tracks for testing."""
        demo_genres = ["house", "techno", "deep_house", "tech_house", "trance", 
                       "dubstep", "drum_and_bass", "hip_hop", "r_and_b", "pop"]
        
        demo_artists = ["DJ Alpha", "Synthwave Collective", "Bass Mechanics", 
                       "Neon Pulse", "Midnight Runners", "Electric Dreams",
                       "Groove Factory", "Rhythm Republic", "Beat Lab", "Sound Architects"]
        
        titles = [
            "Midnight Groove", "Electric Dreams", "Bass Drop", "Neon Nights",
            "Deep Connection", "Pulse", "Velocity", "Momentum", "Ascension",
            "Descent", "Euphoria", "Momentum", "Rhythm Nation", "Frequency",
            "Wavelength", "Amplitude", "Resonance", "Harmony", "Melody", "Drop Zone"
        ]
        
        keys = ["C", "G", "D", "A", "E", "F", "Bb", "Eb", "Am", "Em", "Dm", "Bm"]
        moods = ["upbeat", "calm", "aggressive", "euphoric", "dark", "mysterious", "melancholic"]
        energies = ["low", "medium", "high", "buildup", "drop_center"]
        
        # Generate demo tracks
        for i in range(50):
            genre = random.choice(demo_genres)
            bpm_range = self._get_bpm_range(genre)
            
            track = Track(
                id=f"track_{i+1:03d}",
                title=f"{random.choice(titles)} {i+1}",
                artist=random.choice(demo_artists),
                genre=genre,
                bpm=random.randint(bpm_range[0], bpm_range[1]),
                key=random.choice(keys),
                duration_seconds=random.randint(180, 420),
                energy=random.choice(energies),
                mood=[random.choice(moods), random.choice(moods) if random.random() > 0.5 else ""],
                file_path=f"/music/{genre}/track_{i+1:03d}.mp3",
                tags=[genre, random.choice(moods)]
            )
            self.tracks.append(track)
    
    def _get_bpm_range(self, genre: str) -> tuple:
        """Get BPM range for a genre."""
        bpm_ranges = {
            "house": (118, 130),
            "techno": (125, 140),
            "deep_house": (120, 128),
            "tech_house": (125, 132),
            "trance": (130, 145),
            "dubstep": (140, 160),
            "drum_and_bass": (160, 180),
            "hip_hop": (70, 100),
            "r_and_b": (60, 90),
            "pop": (100, 130),
            "ambient": (60, 100),
        }
        return bpm_ranges.get(genre, (100, 140))
    
    def _is_key_compatible(self, key1: str, key2: str) -> bool:
        """Check if two keys are compatible for mixing."""
        if key1 == key2:
            return True
        compatible_keys = self.KEY_COMPATIBILITY.get(key1, [])
        return key2 in compatible_keys
    
    def _is_energy_compatible(self, energy1: str, energy2: str) -> bool:
        """Check if two energy levels are compatible for smooth transition."""
        # Define energy transition graph
        compatible = {
            "low": ["low", "medium", "buildup"],
            "medium": ["low", "medium", "high", "buildup"],
            "high": ["medium", "high", "drop_center"],
            "buildup": ["low", "medium", "high", "buildup", "drop_center"],
            "drop_center": ["high", "drop_center", "medium"],
        }
        return energy2 in compatible.get(energy1, [])
    
    def _filter_tracks(
        self,
        genres: Optional[List[str]] = None,
        bpm_range: Optional[tuple] = None,
        moods: Optional[List[str]] = None,
        energy: Optional[str] = None,
        min_duration: Optional[int] = None,
        max_duration: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Track]:
        """Filter tracks based on criteria."""
        filtered = self.tracks.copy()
        
        if genres:
            filtered = [t for t in filtered if t.genre in genres]
        
        if bpm_range:
            filtered = [t for t in filtered if bpm_range[0] <= t.bpm <= bpm_range[1]]
        
        if moods:
            filtered = [t for t in filtered if any(m in t.mood for m in moods)]
        
        if energy:
            filtered = [t for t in filtered if t.energy == energy]
        
        if min_duration:
            filtered = [t for t in filtered if t.duration_seconds >= min_duration]
        
        if max_duration:
            filtered = [t for t in filtered if t.duration_seconds <= max_duration]
        
        if tags:
            filtered = [t for t in filtered if any(tag in t.tags for tag in tags)]
        
        return filtered
    
    def generate(
        self,
        name: str = "Generated Playlist",
        description: str = "",
        genres: Optional[List[str]] = None,
        bpm: Optional[int] = None,
        bpm_tolerance: int = 10,
        moods: Optional[List[str]] = None,
        energy: Optional[str] = None,
        track_count: Optional[int] = None,
        total_duration_minutes: Optional[int] = None,
        transition_style: str = "smooth",  # "smooth", "energy_peak", "genre_mix"
        include_genre_transitions: bool = True,
    ) -> Playlist:
        """Generate a playlist based on criteria.
        
        Args:
            name: Playlist name
            description: Playlist description
            genres: List of genres to include
            bpm: Target BPM for the playlist
            bpm_tolerance: Allowed BPM deviation
            moods: List of moods to include
            energy: Target energy level
            track_count: Number of tracks to generate
            total_duration_minutes: Target total duration in minutes
            transition_style: How tracks transition ("smooth", "energy_peak", "genre_mix")
            include_genre_transitions: Whether to include genre transitions
            
        Returns:
            Generated Playlist object
        """
        playlist = Playlist(name=name, description=description)
        
        # Determine filtering criteria
        target_bpm_range = None
        if bpm:
            target_bpm_range = (bpm - bpm_tolerance, bpm + bpm_tolerance)
        
        # Get filtered tracks
        available_tracks = self._filter_tracks(
            genres=genres,
            bpm_range=target_bpm_range,
            moods=moods,
            energy=energy,
        )
        
        if not available_tracks:
            # Relax constraints if no tracks found
            available_tracks = self._filter_tracks(genres=genres)
        
        if not available_tracks:
            return playlist
        
        random.shuffle(available_tracks)
        
        # Determine number of tracks
        if track_count and total_duration_minutes:
            target_tracks = min(track_count, len(available_tracks))
        elif track_count:
            target_tracks = track_count
        elif total_duration_minutes:
            target_tracks = len(available_tracks)
        else:
            target_tracks = min(10, len(available_tracks))
        
        # Select tracks based on transition style
        selected_tracks = []
        used_ids = set()
        
        for track in available_tracks:
            if len(selected_tracks) >= target_tracks:
                break
            
            if track.id in used_ids:
                continue
            
            # Check transition compatibility with previous track
            if selected_tracks:
                prev_track = selected_tracks[-1]
                
                if transition_style == "smooth":
                    # Strict BPM and key compatibility
                    if abs(track.bpm - prev_track.bpm) > bpm_tolerance:
                        continue
                    if not self._is_key_compatible(prev_track.key, track.key):
                        continue
                
                elif transition_style == "energy_peak":
                    # Allow bigger jumps but respect energy flow
                    if not self._is_energy_compatible(prev_track.energy, track.energy):
                        continue
                
                elif transition_style == "genre_mix":
                    # More lenient, allow genre variety
                    pass
            
            selected_tracks.append(track)
            used_ids.add(track.id)
        
        # Add tracks to playlist
        for track in selected_tracks:
            playlist.add_track(track)
        
        # Trim to exact duration if specified
        if total_duration_minutes and playlist.total_duration_seconds > total_duration_minutes * 60:
            # Remove tracks from the end to fit duration
            while playlist.total_duration_seconds > total_duration_minutes * 60 and playlist.tracks:
                removed = playlist.tracks.pop()
                playlist.total_duration_seconds -= removed.duration_seconds
            playlist._recalculate()
        
        return playlist
    
    def generate_energy_journey(
        self,
        name: str = "Energy Journey",
        start_energy: str = "low",
        peak_energy: str = "high",
        genres: Optional[List[str]] = None,
        track_count: int = 10,
    ) -> Playlist:
        """Generate a playlist with an energy arc (buildup to peak).
        
        Args:
            name: Playlist name
            start_energy: Starting energy level
            peak_energy: Peak energy level
            genres: List of genres to include
            track_count: Number of tracks
            
        Returns:
            Generated Playlist with energy arc
        """
        playlist = Playlist(name=name, description=f"Energy journey from {start_energy} to {peak_energy}")
        
        energy_levels = ["low", "medium", "buildup", "high", "drop_center"]
        
        # Find start and peak indices
        try:
            start_idx = energy_levels.index(start_energy)
            peak_idx = energy_levels.index(peak_energy)
        except ValueError:
            start_idx, peak_idx = 0, 3
        
        if start_idx > peak_idx:
            start_idx, peak_idx = peak_idx, start_idx
        
        # Create energy progression
        if track_count <= 4:
            energy_progression = [energy_levels[start_idx]] * track_count
        else:
            # Build gradual progression
            steps = peak_idx - start_idx
            progression = []
            for i in range(track_count):
                prog_idx = start_idx + int((i / (track_count - 1)) * steps)
                prog_idx = min(prog_idx, peak_idx)
                progression.append(energy_levels[prog_idx])
            energy_progression = progression
        
        # Select tracks matching energy progression
        for target_energy in energy_progression:
            matching_tracks = self._filter_tracks(
                genres=genres,
                energy=target_energy,
            )
            
            if matching_tracks:
                track = random.choice(matching_tracks)
                # Avoid duplicates
                if track.id not in [t.id for t in playlist.tracks]:
                    playlist.add_track(track)
        
        return playlist
    
    def generate_mood_based(
        self,
        mood: str,
        genres: Optional[List[str]] = None,
        track_count: int = 10,
    ) -> Playlist:
        """Generate a playlist based on a specific mood.
        
        Args:
            mood: Target mood
            genres: Optional list of genres
            track_count: Number of tracks
            
        Returns:
            Generated Playlist
        """
        return self.generate(
            name=f"{mood.title()} Mix",
            description=f"Playlist curated for {mood} mood",
            moods=[mood],
            genres=genres,
            track_count=track_count,
            transition_style="smooth",
        )
    
    def generate_genre_mix(
        self,
        genres: List[str],
        track_count: int = 12,
        bpm: Optional[int] = None,
    ) -> Playlist:
        """Generate a playlist mixing multiple genres.
        
        Args:
            genres: List of genres to mix
            track_count: Total number of tracks
            bpm: Optional target BPM
            
        Returns:
            Generated Playlist
        """
        playlist = Playlist(
            name="Genre Mix",
            description=f"Mix of {', '.join(genres)}"
        )
        
        # Distribute tracks across genres
        tracks_per_genre = max(1, track_count // len(genres))
        remaining = track_count % len(genres)
        
        for genre in genres:
            count = tracks_per_genre + (1 if remaining > 0 else 0)
            remaining -= 1
            
            genre_playlist = self.generate(
                name=f"Temp {genre}",
                genres=[genre],
                track_count=count,
                bpm=bpm,
                bpm_tolerance=15,
                transition_style="genre_mix",
            )
            
            for track in genre_playlist.tracks:
                playlist.add_track(track)
        
        return playlist
    
    def export_m3u(self, playlist: Playlist, output_path: str) -> str:
        """Export playlist to M3U format.
        
        Args:
            playlist: Playlist to export
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        lines = ["#EXTM3U", f"#PLAYLIST:{playlist.name}"]
        
        for track in playlist.tracks:
            # M3U format: #EXTINF:duration,title - artist
            duration = track.duration_seconds
            title = f"{track.title} - {track.artist}"
            lines.append(f"#EXTINF:{duration},{title}")
            
            if track.file_path:
                lines.append(track.file_path)
            else:
                lines.append(f"{track.title}.mp3")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(output_path)
    
    def export_json(self, playlist: Playlist, output_path: str) -> str:
        """Export playlist to JSON format.
        
        Args:
            playlist: Playlist to export
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(playlist.to_dict(), f, indent=2)
        
        return str(output_path)
    
    def export_csv(self, playlist: Playlist, output_path: str) -> str:
        """Export playlist to CSV format.
        
        Args:
            playlist: Playlist to export
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "Artist", "Genre", "BPM", "Key", "Duration", "Energy", "Mood"])
            
            for track in playlist.tracks:
                writer.writerow([
                    track.title,
                    track.artist,
                    track.genre,
                    track.bpm,
                    track.key,
                    track.duration_formatted,
                    track.energy,
                    ", ".join(track.mood)
                ])
        
        return str(output_path)


def main():
    """CLI interface for playlist generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI DJ Playlist Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a house playlist with 10 tracks
  playlist_generator.py --genre house --tracks 10
  
  # Generate a mood-based playlist
  playlist_generator.py --mood upbeat --tracks 15
  
  # Generate energy journey from low to high
  playlist_generator.py --energy-journey --start-energy low --peak-energy high
  
  # Mix multiple genres
  playlist_generator.py --mix-genres house techno trance --tracks 12
  
  # Export to M3U
  playlist_generator.py --genre house --output playlist.m3u
        """
    )
    
    parser.add_argument("--genre", "-g", help="Genre(s) to include (comma-separated)")
    parser.add_argument("--mood", "-m", help="Target mood")
    parser.add_argument("--energy", "-e", help="Target energy level (low, medium, high, buildup, drop_center)")
    parser.add_argument("--bpm", "-b", type=int, help="Target BPM")
    parser.add_argument("--tracks", "-t", type=int, help="Number of tracks")
    parser.add_argument("--duration", "-d", type=int, help="Total duration in minutes")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=["m3u", "json", "csv"], default="json", help="Output format")
    parser.add_argument("--transition", choices=["smooth", "energy_peak", "genre_mix"], default="smooth",
                        help="Transition style")
    parser.add_argument("--energy-journey", action="store_true", help="Generate energy journey playlist")
    parser.add_argument("--start-energy", default="low", help="Starting energy for journey")
    parser.add_argument("--peak-energy", default="high", help="Peak energy for journey")
    parser.add_argument("--mix-genres", nargs="+", help="Genres to mix")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PlaylistGenerator()
    
    # Generate playlist based on arguments
    if args.energy_journey:
        playlist = generator.generate_energy_journey(
            start_energy=args.start_energy,
            peak_energy=args.peak_energy,
            genres=args.genre.split(",") if args.genre else None,
            track_count=args.tracks or 10,
        )
    elif args.mix_genres:
        playlist = generator.generate_genre_mix(
            genres=args.mix_genres,
            track_count=args.tracks or 12,
            bpm=args.bpm,
        )
    elif args.mood:
        playlist = generator.generate_mood_based(
            mood=args.mood,
            genres=args.genre.split(",") if args.genre else None,
            track_count=args.tracks or 10,
        )
    else:
        playlist = generator.generate(
            name=args.genre.title() + " Mix" if args.genre else "Generated Playlist",
            genres=args.genre.split(",") if args.genre else None,
            bpm=args.bpm,
            moods=[args.mood] if args.mood else None,
            energy=args.energy,
            track_count=args.tracks,
            total_duration_minutes=args.duration,
            transition_style=args.transition,
        )
    
    # Print playlist info
    print(f"\n{'='*50}")
    print(f"Playlist: {playlist.name}")
    print(f"{'='*50}")
    print(f"Description: {playlist.description}")
    print(f"Tracks: {playlist.track_count}")
    print(f"Duration: {playlist.duration_formatted}")
    print(f"Average BPM: {playlist.average_bpm:.1f}")
    print(f"Genres: {', '.join(playlist.genres)}")
    print(f"\nTrack List:")
    print("-"*50)
    
    for i, track in enumerate(playlist.tracks, 1):
        print(f"{i:2d}. {track.title}")
        print(f"    {track.artist} | {track.genre} | {track.bpm} BPM | {track.key} | {track.duration_formatted} | {track.energy}")
    
    # Export if output path specified
    if args.output:
        if args.format == "m3u":
            path = generator.export_m3u(playlist, args.output)
        elif args.format == "csv":
            path = generator.export_csv(playlist, args.output)
        else:
            path = generator.export_json(playlist, args.output)
        
        print(f"\nExported to: {path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
