#!/usr/bin/env python3
"""
AI DJ Project - Statistics System

Comprehensive statistics tracking and aggregation for the AI DJ system.
Provides insights into listening patterns, popular tracks, genre distribution,
and user engagement metrics.
"""

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import existing modules for data aggregation
try:
    from history import PlayHistory, PlayEntry
    from favorites import Favorite
    from playlist import Playlist
    from queue import PlayQueue
except ImportError:
    PlayHistory = None
    Favorite = None
    Playlist = None
    PlayQueue = None


@dataclass
class ListeningStats:
    """Aggregated listening statistics."""
    total_plays: int = 0
    total_listening_time_seconds: int = 0
    total_listening_time_formatted: str = "0:00:00"
    unique_songs_played: int = 0
    unique_artists: int = 0
    playlists_created: int = 0
    favorites_count: int = 0
    songs_in_queue: int = 0
    average_song_duration_seconds: float = 0.0
    skip_rate: float = 0.0  # Percentage of skipped songs


@dataclass
class TopItem:
    """Represents a top item (song/artist/genre) with count."""
    id: str
    name: str
    count: int
    percentage: float


@dataclass
class TimeDistribution:
    """Listening distribution by time period."""
    hour: int
    day_of_week: str
    plays: int
    percentage: float


@dataclass 
class GenreStats:
    """Genre distribution statistics."""
    genre: str
    plays: int
    percentage: float
    average_bpm: float
    average_energy: float


class StatsManager:
    """Main statistics manager for AI DJ system."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or "/Users/johnpeter/ai-dj-project/src")
        self.output_dir = self.data_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Data files
        self.history_file = self.output_dir / "play_history.json"
        self.stats_cache_file = self.output_dir / "stats_cache.json"
        
        # Initialize components
        self._history = None
        self._playlist_dir = self.data_dir / "playlists"
        
        # Cache for stats
        self._cache: Dict[str, Any] = {}
        self._cache_valid = False
    
    @property
    def history(self) -> Optional[PlayHistory]:
        """Lazy-load PlayHistory."""
        if self._history is None and PlayHistory:
            self._history = PlayHistory(str(self.data_dir))
        return self._history
    
    def _load_history(self) -> List[Dict]:
        """Load play history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _load_favorites(self) -> List[Dict]:
        """Load favorites from file."""
        favorites_file = self.data_dir / "data" / "favorites.json"
        if favorites_file.exists():
            try:
                with open(favorites_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _load_playlists(self) -> List[Dict]:
        """Load all playlists."""
        playlists = []
        if self._playlist_dir.exists():
            for playlist_file in self._playlist_dir.glob("*.json"):
                try:
                    with open(playlist_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            playlists.extend(data)
                        elif isinstance(data, dict):
                            playlists.append(data)
                except (json.JSONDecodeError, IOError):
                    continue
        return playlists
    
    def _load_queue(self) -> List[Dict]:
        """Load current queue."""
        queue_file = self.output_dir / "queue.json"
        if queue_file.exists():
            try:
                with open(queue_file, 'r') as f:
                    data = json.load(f)
                    return data.get("items", []) if isinstance(data, dict) else data
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _format_duration(self, seconds: int) -> str:
        """Format seconds to HH:MM:SS."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"
    
    def invalidate_cache(self) -> None:
        """Invalidate stats cache to force recalculation."""
        self._cache_valid = False
        self._cache = {}
    
    def get_listening_stats(self) -> ListeningStats:
        """Get overall listening statistics."""
        history = self._load_history()
        
        if not history:
            return ListeningStats()
        
        # Calculate basic stats
        total_plays = len(history)
        total_listening_time = sum(
            entry.get("duration_seconds", 0) for entry in history
        )
        
        # Count unique songs and artists
        unique_songs = len(set(entry.get("song_id") for entry in history))
        unique_artists = len(set(entry.get("artist") for entry in history if entry.get("artist")))
        
        # Get favorites count
        favorites = self._load_favorites()
        favorites_count = len(favorites)
        
        # Get playlist count
        playlists = self._load_playlists()
        playlist_count = len(playlists)
        
        # Get queue size
        queue = self._load_queue()
        queue_count = len(queue)
        
        # Calculate skip rate
        skips = sum(1 for entry in history if entry.get("skip", False))
        skip_rate = (skips / total_plays * 100) if total_plays > 0 else 0.0
        
        # Average duration
        avg_duration = (
            total_listening_time / total_plays if total_plays > 0 else 0.0
        )
        
        return ListeningStats(
            total_plays=total_plays,
            total_listening_time_seconds=total_listening_time,
            total_listening_time_formatted=self._format_duration(total_listening_time),
            unique_songs_played=unique_songs,
            unique_artists=unique_artists,
            playlists_created=playlist_count,
            favorites_count=favorites_count,
            songs_in_queue=queue_count,
            average_song_duration_seconds=round(avg_duration, 1),
            skip_rate=round(skip_rate, 1)
        )
    
    def get_top_songs(self, limit: int = 10) -> List[TopItem]:
        """Get most played songs."""
        history = self._load_history()
        
        if not history:
            return []
        
        # Count song plays
        song_counts = Counter()
        for entry in history:
            if not entry.get("skip", False):  # Exclude skipped songs
                song_id = entry.get("song_id", "")
                song_counts[song_id] += 1
        
        total_plays = sum(song_counts.values())
        
        # Build top songs list
        top_songs = []
        for song_id, count in song_counts.most_common(limit):
            # Get song title from history
            song_title = next(
                (e.get("song_title", "Unknown") for e in history 
                 if e.get("song_id") == song_id),
                "Unknown"
            )
            percentage = (count / total_plays * 100) if total_plays > 0 else 0
            top_songs.append(TopItem(
                id=song_id,
                name=song_title,
                count=count,
                percentage=round(percentage, 1)
            ))
        
        return top_songs
    
    def get_top_artists(self, limit: int = 10) -> List[TopItem]:
        """Get most played artists."""
        history = self._load_history()
        
        if not history:
            return []
        
        # Count artist plays
        artist_counts = Counter()
        for entry in history:
            if not entry.get("skip", False):
                artist = entry.get("artist", "Unknown")
                if artist:
                    artist_counts[artist] += 1
        
        total_plays = sum(artist_counts.values())
        
        top_artists = []
        for artist, count in artist_counts.most_common(limit):
            percentage = (count / total_plays * 100) if total_plays > 0 else 0
            top_artists.append(TopItem(
                id=artist,
                name=artist,
                count=count,
                percentage=round(percentage, 1)
            ))
        
        return top_artists
    
    def get_genre_distribution(self) -> List[GenreStats]:
        """Get genre distribution statistics."""
        history = self._load_history()
        
        if not history:
            return []
        
        # Group by genre
        genre_plays = defaultdict(list)
        for entry in history:
            if not entry.get("skip", False):
                genre = entry.get("genre", "Unknown")
                bpm = entry.get("bpm", 0)
                energy = entry.get("energy", 0.0)
                genre_plays[genre].append({"bpm": bpm, "energy": energy})
        
        total_plays = sum(len(plays) for plays in genre_plays.values())
        
        genre_stats = []
        for genre, plays in genre_plays.items():
            play_count = len(plays)
            avg_bpm = sum(p["bpm"] for p in plays if p["bpm"]) / max(play_count, 1)
            avg_energy = sum(p["energy"] for p in plays if p["energy"]) / max(play_count, 1)
            percentage = (play_count / total_plays * 100) if total_plays > 0 else 0
            
            genre_stats.append(GenreStats(
                genre=genre,
                plays=play_count,
                percentage=round(percentage, 1),
                average_bpm=round(avg_bpm, 0),
                average_energy=round(avg_energy, 2)
            ))
        
        # Sort by plays descending
        genre_stats.sort(key=lambda x: x.plays, reverse=True)
        return genre_stats
    
    def get_time_distribution(self) -> Dict[str, List[TimeDistribution]]:
        """Get listening distribution by time of day and day of week."""
        history = self._load_history()
        
        if not history:
            return {"by_hour": [], "by_day": []}
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                     "Friday", "Saturday", "Sunday"]
        
        # Count by hour
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)
        
        for entry in history:
            if entry.get("skip", False):
                continue
            
            played_at = entry.get("played_at", "")
            if played_at:
                try:
                    dt = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
                    hour_counts[dt.hour] += 1
                    day_counts[day_names[dt.weekday()]] += 1
                except (ValueError, TypeError):
                    continue
        
        total_plays = sum(hour_counts.values())
        
        # Build hour distribution
        by_hour = []
        for hour in range(24):
            plays = hour_counts[hour]
            percentage = (plays / total_plays * 100) if total_plays > 0 else 0
            by_hour.append(TimeDistribution(
                hour=hour,
                day_of_week="",
                plays=plays,
                percentage=round(percentage, 1)
            ))
        
        # Build day distribution
        by_day = []
        for day in day_names:
            plays = day_counts[day]
            percentage = (plays / total_plays * 100) if total_plays > 0 else 0
            by_day.append(TimeDistribution(
                hour=0,
                day_of_week=day,
                plays=plays,
                percentage=round(percentage, 1)
            ))
        
        return {"by_hour": by_hour, "by_day": by_day}
    
    def get_recent_activity(self, days: int = 7) -> Dict[str, Any]:
        """Get recent listening activity."""
        history = self._load_history()
        
        if not history:
            return {"plays_today": 0, "plays_this_week": 0, "recent_plays": []}
        
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=days - 1)
        
        plays_today = 0
        plays_this_week = 0
        recent_plays = []
        
        for entry in reversed(history):  # Most recent last
            if entry.get("skip", False):
                continue
            
            played_at = entry.get("played_at", "")
            if played_at:
                try:
                    dt = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
                    
                    if dt >= today_start:
                        plays_today += 1
                    
                    if dt >= week_start:
                        plays_this_week += 1
                    
                    # Collect recent plays (last 10)
                    if len(recent_plays) < 10:
                        recent_plays.append({
                            "song_id": entry.get("song_id"),
                            "song_title": entry.get("song_title", "Unknown"),
                            "artist": entry.get("artist", "Unknown"),
                            "played_at": entry.get("played_at"),
                            "genre": entry.get("genre"),
                            "bpm": entry.get("bpm")
                        })
                except (ValueError, TypeError):
                    continue
        
        return {
            "plays_today": plays_today,
            "plays_this_week": plays_this_week,
            "recent_plays": recent_plays
        }
    
    def get_bpm_distribution(self) -> Dict[str, Any]:
        """Get BPM distribution statistics."""
        history = self._load_history()
        
        bpms = [entry.get("bpm", 0) for entry in history 
                if entry.get("bpm") and not entry.get("skip", False)]
        
        if not bpms:
            return {"average": 0, "min": 0, "max": 0, "distribution": {}}
        
        bpms.sort()
        distribution = {
            "low (<100)": 0,
            "medium (100-130)": 0,
            "high (130-160)": 0,
            "very_high (>160)": 0
        }
        
        for bpm in bpms:
            if bpm < 100:
                distribution["low (<100)"] += 1
            elif bpm < 130:
                distribution["medium (100-130)"] += 1
            elif bpm < 160:
                distribution["high (130-160)"] += 1
            else:
                distribution["very_high (>160)"] += 1
        
        return {
            "average": round(sum(bpms) / len(bpms), 0),
            "min": min(bpms),
            "max": max(bpms),
            "median": bpms[len(bpms) // 2],
            "distribution": distribution
        }
    
    def get_mood_distribution(self) -> List[Dict[str, Any]]:
        """Get mood distribution from history."""
        history = self._load_history()
        
        if not history:
            return []
        
        mood_counts = Counter()
        for entry in history:
            if not entry.get("skip", False):
                mood = entry.get("mood", "Unknown")
                if mood:
                    mood_counts[mood] += 1
        
        total = sum(mood_counts.values())
        
        return [
            {
                "mood": mood,
                "count": count,
                "percentage": round(count / total * 100, 1)
            }
            for mood, count in mood_counts.most_common()
        ]
    
    def get_source_breakdown(self) -> List[Dict[str, Any]]:
        """Get breakdown by play source (playlist, auto_dj, manual, shuffle)."""
        history = self._load_history()
        
        if not history:
            return []
        
        source_counts = Counter()
        for entry in history:
            source = entry.get("source", "unknown")
            source_counts[source] += 1
        
        total = sum(source_counts.values())
        
        return [
            {
                "source": source,
                "count": count,
                "percentage": round(count / total * 100, 1)
            }
            for source, count in source_counts.most_common()
        ]
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get all statistics in one comprehensive report."""
        return {
            "generated_at": datetime.now().isoformat(),
            "listening": asdict(self.get_listening_stats()),
            "top_songs": [asdict(s) for s in self.get_top_songs(10)],
            "top_artists": [asdict(a) for a in self.get_top_artists(10)],
            "genre_distribution": [asdict(g) for g in self.get_genre_distribution()],
            "time_distribution": {
                "by_hour": [asdict(h) for h in self.get_time_distribution()["by_hour"]],
                "by_day": [asdict(d) for d in self.get_time_distribution()["by_day"]]
            },
            "recent_activity": self.get_recent_activity(7),
            "bpm_stats": self.get_bpm_distribution(),
            "mood_distribution": self.get_mood_distribution(),
            "source_breakdown": self.get_source_breakdown()
        }
    
    def save_stats_report(self, filepath: str = None) -> str:
        """Generate and save a stats report to file."""
        if filepath is None:
            filepath = str(self.output_dir / f"stats_report_{datetime.now().strftime('%Y%m%d')}.json")
        
        stats = self.get_comprehensive_stats()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return filepath
    
    def get_quick_summary(self) -> str:
        """Get a quick text summary of key stats."""
        stats = self.get_listening_stats()
        
        summary = f"""📊 AI DJ Statistics

🎵 Total Plays: {stats.total_plays}
⏱️ Listening Time: {stats.total_listening_time_formatted}
🎤 Unique Artists: {stats.unique_artists}
❤️ Favorites: {stats.favorites_count}
📋 Queue: {stats.songs_in_queue} songs
⏭️ Skip Rate: {stats.skip_rate}%
"""
        return summary


# CLI for quick stats
def main():
    """CLI entry point for stats."""
    import sys
    
    manager = StatsManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "summary":
            print(manager.get_quick_summary())
        elif command == "report":
            filepath = manager.save_stats_report()
            print(f"✅ Report saved to: {filepath}")
        elif command == "top-songs":
            for song in manager.get_top_songs(5):
                print(f"  {song.count}x - {song.name}")
        elif command == "top-artists":
            for artist in manager.get_top_artists(5):
                print(f"  {artist.count}x - {artist.name}")
        else:
            print(f"Unknown command: {command}")
            print("Available: summary, report, top-songs, top-artists")
    else:
        # Default: print summary
        print(manager.get_quick_summary())


if __name__ == "__main__":
    main()
