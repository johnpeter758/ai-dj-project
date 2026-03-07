#!/usr/bin/env python3
"""
Playlist Management Module for AI DJ Project

Handles playlist lifecycle: creation, loading, saving, editing,
queue management, and playback control.
"""

import json
import uuid
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum
from copy import deepcopy


class PlaybackMode(Enum):
    """Playlist playback modes."""
    LINEAR = "linear"           # Play tracks in order
    LOOP = "loop"               # Loop through playlist
    SHUFFLE = "shuffle"         # Shuffle and play
    SINGLE = "single"           # Repeat single track
    LOOP_SINGLE = "loop_single" # Loop single track


class PlaylistError(Exception):
    """Base exception for playlist operations."""
    pass


class TrackNotFoundError(PlaylistError):
    """Raised when a track is not found in playlist."""
    pass


class InvalidOperationError(PlaylistError):
    """Raised when an invalid operation is attempted."""
    pass


@dataclass
class Track:
    """Represents a track in a managed playlist."""
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
    source: str = "library"  # "library", "generated", "imported"
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())
    play_count: int = 0
    last_played: Optional[str] = None
    
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
            "tags": self.tags,
            "source": self.source,
            "added_at": self.added_at,
            "play_count": self.play_count,
            "last_played": self.last_played,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Track":
        """Create track from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "Unknown"),
            artist=data.get("artist", "Unknown"),
            genre=data.get("genre", "unknown"),
            bpm=data.get("bpm", 120),
            key=data.get("key", "C"),
            duration_seconds=data.get("duration_seconds", 180),
            energy=data.get("energy", "medium"),
            mood=data.get("mood", []),
            file_path=data.get("file_path"),
            tags=data.get("tags", []),
            source=data.get("source", "library"),
            added_at=data.get("added_at", datetime.now().isoformat()),
            play_count=data.get("play_count", 0),
            last_played=data.get("last_played"),
        )


@dataclass
class Playlist:
    """Represents a managed playlist."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Playlist"
    description: str = ""
    tracks: List[Track] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    genre: Optional[str] = None
    cover_art: Optional[str] = None
    
    # Playback state
    current_index: int = 0
    playback_mode: PlaybackMode = PlaybackMode.LINEAR
    
    @property
    def track_count(self) -> int:
        """Return number of tracks."""
        return len(self.tracks)
    
    @property
    def total_duration_seconds(self) -> int:
        """Return total playlist duration."""
        return sum(t.duration_seconds for t in self.tracks)
    
    @property
    def duration_formatted(self) -> str:
        """Return total duration as HH:MM:SS or MM:SS."""
        total = self.total_duration_seconds
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
    
    @property
    def average_bpm(self) -> float:
        """Return average BPM of playlist."""
        if not self.tracks:
            return 0.0
        return sum(t.bpm for t in self.tracks) / len(self.tracks)
    
    @property
    def current_track(self) -> Optional[Track]:
        """Get current track based on current_index."""
        if 0 <= self.current_index < len(self.tracks):
            return self.tracks[self.current_index]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert playlist to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tracks": [t.to_dict() for t in self.tracks],
            "track_count": self.track_count,
            "total_duration_seconds": self.total_duration_seconds,
            "duration_formatted": self.duration_formatted,
            "average_bpm": round(self.average_bpm, 1),
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "created_by": self.created_by,
            "tags": self.tags,
            "genre": self.genre,
            "cover_art": self.cover_art,
            "current_index": self.current_index,
            "playback_mode": self.playback_mode.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Playlist":
        """Create playlist from dictionary."""
        tracks = [Track.from_dict(t) for t in data.get("tracks", [])]
        playlist = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            tracks=tracks,
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            created_by=data.get("created_by", "system"),
            tags=data.get("tags", []),
            genre=data.get("genre"),
            cover_art=data.get("cover_art"),
            current_index=data.get("current_index", 0),
        )
        if "playback_mode" in data:
            playlist.playback_mode = PlaybackMode(data["playback_mode"])
        return playlist


class PlaylistManager:
    """Manages playlists - CRUD operations, persistence, queue control."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize playlist manager.
        
        Args:
            storage_dir: Directory for playlist storage. Defaults to ./playlists
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path("./playlists")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.playlists: Dict[str, Playlist] = {}
        self.active_playlist: Optional[Playlist] = None
        self.queue: List[Track] = []  # Upcoming tracks (priority queue)
        
        # Callbacks for playlist events
        self.on_track_change: Optional[Callable[[Track], None]] = None
        self.on_playlist_change: Optional[Callable[[Playlist], None]] = None
        
        # Load all saved playlists
        self._load_all_playlists()
    
    def _get_playlist_path(self, playlist_id: str) -> Path:
        """Get file path for playlist storage."""
        return self.storage_dir / f"{playlist_id}.json"
    
    def _load_all_playlists(self) -> None:
        """Load all playlists from storage directory."""
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    playlist = Playlist.from_dict(data)
                    self.playlists[playlist.id] = playlist
            except Exception as e:
                print(f"⚠️ Failed to load playlist {file_path}: {e}")
    
    def create(
        self,
        name: str,
        description: str = "",
        tags: List[str] = None,
        genre: str = None,
    ) -> Playlist:
        """Create a new playlist.
        
        Args:
            name: Playlist name
            description: Playlist description
            tags: Optional tags
            genre: Optional genre
            
        Returns:
            Created playlist
        """
        playlist = Playlist(
            name=name,
            description=description,
            tags=tags or [],
            genre=genre,
        )
        self.playlists[playlist.id] = playlist
        self._save_playlist(playlist)
        
        if self.on_playlist_change:
            self.on_playlist_change(playlist)
        
        return playlist
    
    def get(self, playlist_id: str) -> Optional[Playlist]:
        """Get playlist by ID."""
        return self.playlists.get(playlist_id)
    
    def list_playlists(self) -> List[Playlist]:
        """List all playlists."""
        return list(self.playlists.values())
    
    def delete(self, playlist_id: str) -> bool:
        """Delete a playlist.
        
        Args:
            playlist_id: ID of playlist to delete
            
        Returns:
            True if deleted, False if not found
        """
        if playlist_id not in self.playlists:
            return False
        
        # Remove from active if active
        if self.active_playlist and self.active_playlist.id == playlist_id:
            self.active_playlist = None
        
        # Delete file
        playlist_path = self._get_playlist_path(playlist_id)
        if playlist_path.exists():
            playlist_path.unlink()
        
        del self.playlists[playlist_id]
        return True
    
    def _save_playlist(self, playlist: Playlist) -> None:
        """Save playlist to disk."""
        playlist.modified_at = datetime.now().isoformat()
        playlist_path = self._get_playlist_path(playlist.id)
        
        with open(playlist_path, 'w') as f:
            json.dump(playlist.to_dict(), f, indent=2)
    
    def add_track(self, playlist_id: str, track: Track) -> bool:
        """Add a track to a playlist.
        
        Args:
            playlist_id: Target playlist ID
            track: Track to add
            
        Returns:
            True if added successfully
        """
        playlist = self.playlists.get(playlist_id)
        if not playlist:
            return False
        
        playlist.tracks.append(track)
        self._save_playlist(playlist)
        
        if self.on_playlist_change:
            self.on_playlist_change(playlist)
        
        return True
    
    def add_tracks(self, playlist_id: str, tracks: List[Track]) -> int:
        """Add multiple tracks to a playlist.
        
        Args:
            playlist_id: Target playlist ID
            tracks: Tracks to add
            
        Returns:
            Number of tracks added
        """
        playlist = self.playlists.get(playlist_id)
        if not playlist:
            return 0
        
        playlist.tracks.extend(tracks)
        self._save_playlist(playlist)
        
        if self.on_playlist_change:
            self.on_playlist_change(playlist)
        
        return len(tracks)
    
    def remove_track(self, playlist_id: str, track_id: str) -> bool:
        """Remove a track from playlist.
        
        Args:
            playlist_id: Playlist ID
            track_id: Track ID to remove
            
        Returns:
            True if removed, False if not found
        """
        playlist = self.playlists.get(playlist_id)
        if not playlist:
            return False
        
        for i, track in enumerate(playlist.tracks):
            if track.id == track_id:
                playlist.tracks.pop(i)
                # Adjust current index if needed
                if playlist.current_index >= len(playlist.tracks):
                    playlist.current_index = max(0, len(playlist.tracks) - 1)
                self._save_playlist(playlist)
                
                if self.on_playlist_change:
                    self.on_playlist_change(playlist)
                return True
        
        return False
    
    def reorder_track(self, playlist_id: str, from_index: int, to_index: int) -> bool:
        """Reorder a track within playlist.
        
        Args:
            playlist_id: Playlist ID
            from_index: Current track index
            to_index: Target index
            
        Returns:
            True if reordered successfully
        """
        playlist = self.playlists.get(playlist_id)
        if not playlist:
            return False
        
        if not (0 <= from_index < len(playlist.tracks)):
            return False
        if not (0 <= to_index < len(playlist.tracks)):
            return False
        
        track = playlist.tracks.pop(from_index)
        playlist.tracks.insert(to_index, track)
        
        # Update current index if needed
        if playlist.current_index == from_index:
            playlist.current_index = to_index
        elif from_index < playlist.current_index <= to_index:
            playlist.current_index -= 1
        elif to_index <= playlist.current_index < from_index:
            playlist.current_index += 1
        
        self._save_playlist(playlist)
        return True
    
    def set_active(self, playlist_id: str) -> bool:
        """Set active playlist for playback.
        
        Args:
            playlist_id: Playlist ID to activate
            
        Returns:
            True if successful
        """
        playlist = self.playlists.get(playlist_id)
        if not playlist:
            return False
        
        self.active_playlist = playlist
        self.queue.clear()
        return True
    
    def get_active(self) -> Optional[Playlist]:
        """Get currently active playlist."""
        return self.active_playlist
    
    # ========== Queue Management ==========
    
    def add_to_queue(self, track: Track) -> None:
        """Add track to playback queue (next up)."""
        self.queue.append(track)
    
    def add_to_queue_front(self, track: Track) -> None:
        """Add track to front of queue (play next)."""
        self.queue.insert(0, track)
    
    def get_queue(self) -> List[Track]:
        """Get current queue."""
        return self.queue.copy()
    
    def clear_queue(self) -> None:
        """Clear the queue."""
        self.queue.clear()
    
    def remove_from_queue(self, index: int) -> bool:
        """Remove track from queue by index."""
        if 0 <= index < len(self.queue):
            self.queue.pop(index)
            return True
        return False
    
    # ========== Playback Control ==========
    
    def get_next_track(self) -> Optional[Track]:
        """Get next track based on playback mode.
        
        Returns:
            Next track or None
        """
        if not self.active_playlist:
            return None
        
        playlist = self.active_playlist
        tracks = playlist.tracks
        
        if not tracks:
            return None
        
        # If queue has tracks, play from queue first
        if self.queue:
            return self.queue.pop(0)
        
        # Determine next index based on mode
        if playlist.playback_mode == PlaybackMode.SINGLE:
            return playlist.current_track
        
        elif playlist.playback_mode == PlaybackMode.LOOP_SINGLE:
            track = playlist.current_track
            if track:
                track.play_count += 1
                track.last_played = datetime.now().isoformat()
            return track
        
        elif playlist.playback_mode == PlaybackMode.LOOP:
            playlist.current_index = (playlist.current_index + 1) % len(tracks)
        
        elif playlist.playback_mode == PlaybackMode.SHUFFLE:
            if len(tracks) > 1:
                # Get next random (excluding current)
                available = [i for i in range(len(tracks)) if i != playlist.current_index]
                if available:
                    playlist.current_index = random.choice(available)
                else:
                    playlist.current_index = 0
            else:
                playlist.current_index = 0
        
        else:  # LINEAR
            if playlist.current_index < len(tracks) - 1:
                playlist.current_index += 1
            else:
                return None  # End of playlist
        
        track = playlist.current_track
        if track:
            track.play_count += 1
            track.last_played = datetime.now().isoformat()
            self._save_playlist(playlist)
        
        if self.on_track_change:
            self.on_track_change(track)
        
        return track
    
    def get_previous_track(self) -> Optional[Track]:
        """Get previous track based on playback mode."""
        if not self.active_playlist:
            return None
        
        playlist = self.active_playlist
        tracks = playlist.tracks
        
        if not tracks:
            return None
        
        if playlist.playback_mode == PlaybackMode.SINGLE:
            return playlist.current_track
        
        if playlist.playback_mode == PlaybackMode.LOOP_SINGLE:
            return playlist.current_track
        
        # Go to previous
        playlist.current_index = (playlist.current_index - 1) % len(tracks)
        
        track = playlist.current_track
        if self.on_track_change:
            self.on_track_change(track)
        
        return track
    
    def jump_to_track(self, index: int) -> bool:
        """Jump to specific track by index."""
        if not self.active_playlist:
            return False
        
        playlist = self.active_playlist
        if 0 <= index < len(playlist.tracks):
            playlist.current_index = index
            
            track = playlist.current_track
            if track:
                track.play_count += 1
                track.last_played = datetime.now().isoformat()
                self._save_playlist(playlist)
            
            if self.on_track_change:
                self.on_track_change(track)
            return True
        return False
    
    def jump_to_track_id(self, track_id: str) -> bool:
        """Jump to specific track by ID."""
        if not self.active_playlist:
            return False
        
        playlist = self.active_playlist
        for i, track in enumerate(playlist.tracks):
            if track.id == track_id:
                return self.jump_to_track(i)
        return False
    
    def set_playback_mode(self, mode: PlaybackMode) -> bool:
        """Set playback mode for active playlist."""
        if not self.active_playlist:
            return False
        
        self.active_playlist.playback_mode = mode
        self._save_playlist(self.active_playlist)
        return True
    
    # ========== Playlist Operations ==========
    
    def shuffle(self, playlist_id: str) -> bool:
        """Shuffle playlist tracks."""
        playlist = self.playlists.get(playlist_id)
        if not playlist or len(playlist.tracks) <= 1:
            return False
        
        random.shuffle(playlist.tracks)
        playlist.current_index = 0
        self._save_playlist(playlist)
        
        if self.on_playlist_change:
            self.on_playlist_change(playlist)
        
        return True
    
    def sort(self, playlist_id: str, by: str = "bpm", reverse: bool = False) -> bool:
        """Sort playlist by attribute.
        
        Args:
            playlist_id: Playlist ID
            by: Sort attribute (bpm, title, artist, genre, duration, energy)
            reverse: Reverse sort order
            
        Returns:
            True if sorted successfully
        """
        playlist = self.playlists.get(playlist_id)
        if not playlist:
            return False
        
        valid_attrs = {
            "bpm": lambda t: t.bpm,
            "title": lambda t: t.title.lower(),
            "artist": lambda t: t.artist.lower(),
            "genre": lambda t: t.genre.lower(),
            "duration": lambda t: t.duration_seconds,
            "energy": lambda t: t.energy,
            "added": lambda t: t.added_at,
        }
        
        if by not in valid_attrs:
            return False
        
        playlist.tracks.sort(key=valid_attrs[by], reverse=reverse)
        playlist.current_index = 0
        self._save_playlist(playlist)
        
        if self.on_playlist_change:
            self.on_playlist_change(playlist)
        
        return True
    
    def duplicate(self, playlist_id: str, new_name: Optional[str] = None) -> Optional[Playlist]:
        """Duplicate a playlist.
        
        Args:
            playlist_id: Source playlist ID
            new_name: Name for new playlist
            
        Returns:
            New playlist or None if source not found
        """
        source = self.playlists.get(playlist_id)
        if not source:
            return None
        
        new_playlist = Playlist(
            name=new_name or f"{source.name} (Copy)",
            description=source.description,
            tags=source.tags.copy(),
            genre=source.genre,
            tracks=deepcopy(source.tracks),
        )
        
        self.playlists[new_playlist.id] = new_playlist
        self._save_playlist(new_playlist)
        
        return new_playlist
    
    def merge(self, playlist_ids: List[str], new_name: str) -> Optional[Playlist]:
        """Merge multiple playlists into one.
        
        Args:
            playlist_ids: List of playlist IDs to merge
            new_name: Name for merged playlist
            
        Returns:
            New merged playlist or None if any not found
        """
        all_tracks = []
        
        for pid in playlist_ids:
            playlist = self.playlists.get(pid)
            if not playlist:
                return None
            all_tracks.extend(playlist.tracks)
        
        if not all_tracks:
            return None
        
        merged = Playlist(
            name=new_name,
            description=f"Merged from {len(playlist_ids)} playlists",
            tracks=all_tracks,
        )
        
        self.playlists[merged.id] = merged
        self._save_playlist(merged)
        
        return merged
    
    def search(self, query: str, playlist_id: Optional[str] = None) -> List[Track]:
        """Search tracks in playlist or all playlists.
        
        Args:
            query: Search query
            playlist_id: Optional playlist to search in
            
        Returns:
            Matching tracks
        """
        query_lower = query.lower()
        results = []
        
        if playlist_id:
            playlists = [self.playlists.get(playlist_id)] if self.playlists.get(playlist_id) else []
        else:
            playlists = self.playlists.values()
        
        for playlist in playlists:
            if not playlist:
                continue
            for track in playlist.tracks:
                if (query_lower in track.title.lower() or
                    query_lower in track.artist.lower() or
                    query_lower in track.genre.lower() or
                    any(query_lower in tag.lower() for tag in track.tags)):
                    results.append(track)
        
        return results
    
    # ========== Import/Export ==========
    
    def export_playlist(self, playlist_id: str, format: str = "json") -> Optional[str]:
        """Export playlist to file.
        
        Args:
            playlist_id: Playlist to export
            format: Export format (json, m3u)
            
        Returns:
            Path to exported file or None
        """
        playlist = self.playlists.get(playlist_id)
        if not playlist:
            return None
        
        if format == "json":
            output_path = self.storage_dir / f"{playlist.name}.json"
            with open(output_path, 'w') as f:
                json.dump(playlist.to_dict(), f, indent=2)
            return str(output_path)
        
        elif format == "m3u":
            output_path = self.storage_dir / f"{playlist.name}.m3u"
            lines = ["#EXTM3U", f"#PLAYLIST:{playlist.name}"]
            
            for track in playlist.tracks:
                lines.append(f"#EXTINF:{track.duration_seconds},{track.title} - {track.artist}")
                lines.append(track.file_path or f"{track.title}.mp3")
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            return str(output_path)
        
        return None
    
    def import_playlist(self, file_path: str) -> Optional[Playlist]:
        """Import playlist from file.
        
        Args:
            file_path: Path to playlist file
            
        Returns:
            Imported playlist or None
        """
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        try:
            if path.suffix == ".json":
                with open(path, 'r') as f:
                    data = json.load(f)
                    playlist = Playlist.from_dict(data)
                    # Assign new ID to avoid conflicts
                    playlist.id = str(uuid.uuid4())
                    self.playlists[playlist.id] = playlist
                    self._save_playlist(playlist)
                    return playlist
            
            elif path.suffix == ".m3u":
                return self._import_m3u(path)
        
        except Exception as e:
            print(f"⚠️ Failed to import playlist: {e}")
        
        return None
    
    def _import_m3u(self, path: Path) -> Optional[Playlist]:
        """Import M3U playlist file."""
        tracks = []
        
        with open(path, 'r') as f:
            lines = f.readlines()
        
        current_title = None
        current_duration = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("#EXTINF:"):
                # Parse: #EXTINF:duration,title - artist
                parts = line[8:].split(",", 1)
                if len(parts) == 2:
                    try:
                        current_duration = int(parts[0])
                    except ValueError:
                        current_duration = 180
                    current_title = parts[1]
            
            elif line and not line.startswith("#"):
                # This is a file path
                title = current_title or path.stem
                artist = "Unknown"
                
                if " - " in title:
                    parts = title.split(" - ", 1)
                    artist = parts[0].strip()
                    title = parts[1].strip()
                
                track = Track(
                    id=str(uuid.uuid4()),
                    title=title,
                    artist=artist,
                    genre="unknown",
                    bpm=120,
                    key="C",
                    duration_seconds=current_duration or 180,
                    file_path=line,
                    source="imported",
                )
                tracks.append(track)
                current_title = None
                current_duration = None
        
        if not tracks:
            return None
        
        playlist = Playlist(
            name=path.stem,
            description="Imported playlist",
            tracks=tracks,
        )
        
        self.playlists[playlist.id] = playlist
        self._save_playlist(playlist)
        
        return playlist


# ========== Demo / Testing ==========

def create_demo_manager() -> PlaylistManager:
    """Create a demo playlist manager with sample data."""
    manager = PlaylistManager()
    
    # Create some demo playlists
    house = manager.create("House Mix", "Deep and tech house vibes", genre="house")
    trance = manager.create("Trance Journey", "Uplifting trance tracks", genre="trance")
    hiphop = manager.create("Hip Hop Classics", "Old school hip hop", genre="hip_hop")
    
    # Add demo tracks
    demo_tracks = [
        Track(id=str(uuid.uuid4()), title="Midnight Groove", artist="DJ Alpha", 
              genre="house", bpm=124, key="Am", duration_seconds=320, energy="medium"),
        Track(id=str(uuid.uuid4()), title="Deep Dreams", artist="Synthwave Collective",
              genre="house", bpm=122, key="G", duration_seconds=280, energy="low"),
        Track(id=str(uuid.uuid4()), title="Electric Pulse", artist="Neon Pulse",
              genre="tech_house", bpm=128, key="Em", duration_seconds=360, energy="high"),
        Track(id=str(uuid.uuid4()), title="Ascension", artist="Beat Lab",
              genre="trance", bpm=138, key="A", duration_seconds=420, energy="buildup"),
        Track(id=str(uuid.uuid4()), title="Euphoria", artist="Rhythm Republic",
              genre="trance", bpm=140, key="B", duration_seconds=380, energy="drop_center"),
        Track(id=str(uuid.uuid4()), title="Golden Era", artist="Old School",
              genre="hip_hop", bpm=92, key="C", duration_seconds=240, energy="medium"),
    ]
    
    # Add tracks to house playlist
    for track in demo_tracks[:3]:
        manager.add_track(house.id, track)
    
    # Add tracks to trance playlist
    for track in demo_tracks[3:5]:
        manager.add_track(trance.id, track)
    
    # Add tracks to hiphop playlist
    for track in demo_tracks[5:]:
        manager.add_track(hiphop.id, track)
    
    return manager


def main():
    """Demo CLI for playlist manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI DJ Playlist Manager")
    parser.add_argument("--create", "-c", help="Create new playlist")
    parser.add_argument("--list", "-l", action="store_true", help="List all playlists")
    parser.add_argument("--active", "-a", help="Set active playlist by ID")
    parser.add_argument("--next", "-n", action="store_true", help="Get next track")
    parser.add_argument("--shuffle", "-s", help="Shuffle playlist by ID")
    parser.add_argument("--storage", default="./playlists", help="Storage directory")
    
    args = parser.parse_args()
    
    manager = PlaylistManager(args.storage)
    
    if args.create:
        playlist = manager.create(args.create)
        print(f"✓ Created playlist: {playlist.name} ({playlist.id})")
    
    if args.list:
        print("\n📋 Playlists:")
        for p in manager.list_playlists():
            print(f"  • {p.name} ({p.track_count} tracks, {p.duration_formatted})")
    
    if args.active:
        if manager.set_active(args.active):
            print(f"✓ Active playlist: {manager.active_playlist.name}")
        else:
            print("✗ Playlist not found")
    
    if args.next:
        track = manager.get_next_track()
        if track:
            print(f"▶ Next: {track.title} - {track.artist}")
        else:
            print("◼ No track")
    
    if args.shuffle:
        if manager.shuffle(args.shuffle):
            print("✓ Shuffled")
        else:
            print("✗ Not found")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
