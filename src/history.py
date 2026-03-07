"""
AI DJ Project - Play History System
Tracks played songs, playlists, and listening analytics.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class PlayEntry:
    """Single play history entry."""
    song_id: str
    song_title: str
    artist: str
    played_at: str  # ISO format timestamp
    duration_seconds: int
    source: str  # playlist, auto_dj, manual, shuffle
    playlist_id: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    bpm: Optional[int] = None
    key: Optional[str] = None
    skip: bool = False  # If user skipped the song


class PlayHistory:
    """Manages play history for the AI DJ system."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or "/Users/johnpeter/ai-dj-project/src")
        self.output_dir = self.data_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.history_file = self.output_dir / "play_history.json"
        
        # Load existing history
        self.history: list[dict] = self._load_history()
    
    def _load_history(self) -> list[dict]:
        """Load play history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_history(self) -> None:
        """Save play history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_play(
        self,
        song_id: str,
        song_title: str,
        artist: str,
        duration_seconds: int,
        source: str = "manual",
        playlist_id: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
        skip: bool = False
    ) -> dict:
        """Add a new play entry to history."""
        entry = PlayEntry(
            song_id=song_id,
            song_title=song_title,
            artist=artist,
            played_at=datetime.now().isoformat(),
            duration_seconds=duration_seconds,
            source=source,
            playlist_id=playlist_id,
            genre=genre,
            mood=mood,
            bpm=bpm,
            key=key,
            skip=skip
        )
        
        self.history.append(asdict(entry))
        self._save_history()
        
        return asdict(entry)
    
    def mark_skipped(self, song_id: str, played_at: str) -> bool:
        """Mark a song as skipped by user."""
        for entry in reversed(self.history):
            if entry.get("song_id") == song_id and entry.get("played_at") == played_at:
                entry["skip"] = True
                self._save_history()
                return True
        return False
    
    def get_recent_plays(self, limit: int = 50) -> list[dict]:
        """Get most recent plays."""
        return self.history[-limit:] if self.history else []
    
    def get_plays_by_source(self, source: str) -> list[dict]:
        """Get plays filtered by source (playlist, auto_dj, manual, shuffle)."""
        return [e for e in self.history if e.get("source") == source]
    
    def get_plays_by_playlist(self, playlist_id: str) -> list[dict]:
        """Get all plays from a specific playlist."""
        return [e for e in self.history if e.get("playlist_id") == playlist_id]
    
    def get_top_songs(self, limit: int = 10, skip_skipped: bool = True) -> list[dict]:
        """Get most played songs."""
        song_counts: dict[str, dict] = {}
        
        for entry in self.history:
            if skip_skipped and entry.get("skip"):
                continue
            
            song_id = entry.get("song_id")
            if song_id not in song_counts:
                song_counts[song_id] = {
                    "song_id": song_id,
                    "song_title": entry.get("song_title"),
                    "artist": entry.get("artist"),
                    "play_count": 0,
                    "total_duration": 0,
                    "last_played": entry.get("played_at")
                }
            
            song_counts[song_id]["play_count"] += 1
            song_counts[song_id]["total_duration"] += entry.get("duration_seconds", 0)
            
            # Update last played if more recent
            if entry.get("played_at") > song_counts[song_id]["last_played"]:
                song_counts[song_id]["last_played"] = entry.get("played_at")
        
        # Sort by play count
        sorted_songs = sorted(song_counts.values(), key=lambda x: x["play_count"], reverse=True)
        return sorted_songs[:limit]
    
    def get_genre_breakdown(self) -> dict:
        """Get play count breakdown by genre."""
        genre_counts: dict[str, int] = {}
        
        for entry in self.history:
            genre = entry.get("genre") or "unknown"
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        return genre_counts
    
    def get_listening_stats(self, days: int = 7) -> dict:
        """Get listening statistics for the past N days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        recent_plays = [
            e for e in self.history
            if datetime.fromisoformat(e.get("played_at", "2000-01-01")).timestamp() > cutoff
        ]
        
        total_plays = len(recent_plays)
        total_skips = sum(1 for e in recent_plays if e.get("skip"))
        total_duration = sum(e.get("duration_seconds", 0) for e in recent_plays)
        
        # Group by day
        daily_plays: dict[str, int] = {}
        for entry in recent_plays:
            day = entry.get("played_at", "")[:10]  # YYYY-MM-DD
            daily_plays[day] = daily_plays.get(day, 0) + 1
        
        return {
            "period_days": days,
            "total_plays": total_plays,
            "total_skips": total_skips,
            "skip_rate": round(total_skips / max(total_plays, 1) * 100, 1),
            "total_listening_seconds": total_duration,
            "total_listening_formatted": self._format_duration(total_duration),
            "daily_plays": daily_plays
        }
    
    def _format_duration(self, seconds: int) -> str:
        """Format seconds as human-readable duration."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    
    def clear_history(self) -> None:
        """Clear all play history."""
        self.history = []
        self._save_history()
    
    def export_history(self, filepath: str) -> None:
        """Export history to a specific file path."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


# Standalone usage example
if __name__ == "__main__":
    # Demo usage
    history = PlayHistory()
    
    # Add some demo entries
    history.add_play(
        song_id="song_001",
        song_title="Midnight Drive",
        artist="Synthwave Master",
        duration_seconds=240,
        source="playlist",
        playlist_id="chill_vibes",
        genre="synthwave",
        mood="chill",
        bpm=110,
        key="Am"
    )
    
    history.add_play(
        song_id="song_002",
        song_title="Electric Dreams",
        artist="Neon Rider",
        duration_seconds=195,
        source="auto_dj",
        genre="electronic",
        mood="energetic",
        bpm=128,
        key="G"
    )
    
    # Get stats
    print("Recent plays:", history.get_recent_plays(5))
    print("Top songs:", history.get_top_songs(5))
    print("Genre breakdown:", history.get_genre_breakdown())
    print("Listening stats (7 days):", history.get_listening_stats(7))
