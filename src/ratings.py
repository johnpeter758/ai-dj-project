#!/usr/bin/env python3
"""
Ratings System for AI DJ Project

Manages user ratings for tracks - allowing users to rate songs on multiple
dimensions (overall quality, energy, mood match, danceability, etc.) and 
use these ratings for smart playlist generation and recommendations.
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

# Database integration
try:
    from database import DB_PATH
    import sqlite3
    HAS_DB = True
except ImportError:
    HAS_DB = False

# Local ratings file as fallback/primary storage
RATINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "data", 
    "ratings.json"
)


# Rating categories/scales
class RatingScale:
    """Standard rating scales for different criteria"""
    # 1-5 star scale
    STARS = (1, 2, 3, 4, 5)
    
    # 0-10 numerical scale
    NUMERICAL = tuple(range(0, 11))
    
    # Percentage scale
    PERCENTAGE = tuple(range(0, 101, 5))  # 0, 5, 10, ..., 100


@dataclass
class TrackRating:
    """Represents a rating entry for a track"""
    id: Optional[int] = None
    song_id: int = 0
    song_name: str = ""
    song_path: str = ""
    artist: str = ""
    genre: str = ""
    
    # Main rating (1-5 stars)
    overall: int = 0
    
    # Detailed ratings (0-10 scale)
    energy: float = 0.0       # How energetic is the track
    mood: float = 0.0         # How well it matches desired mood
    danceability: float = 0.0 # How danceable
    quality: float = 0.0      # Production quality
    originality: float = 0.0 # How original/unique
    
    # User-specific tags
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Timestamps
    rated_at: str = ""
    updated_at: str = ""
    
    # Rating count (for averaging multiple ratings of same track)
    rating_count: int = 1
    
    def __post_init__(self):
        if not self.rated_at:
            self.rated_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.rated_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackRating':
        """Create TrackRating from dictionary"""
        # Handle tags field
        if 'tags' in data and isinstance(data['tags'], str):
            data['tags'] = [t.strip() for t in data['tags'].split(',') if t.strip()]
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    def get_average_rating(self) -> float:
        """Calculate average of all detailed ratings"""
        ratings = [self.energy, self.mood, self.danceability, self.quality, self.originality]
        valid_ratings = [r for r in ratings if r > 0]
        if not valid_ratings:
            return 0.0
        return sum(valid_ratings) / len(valid_ratings)
    
    def get_weighted_rating(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted average rating"""
        if weights is None:
            weights = {
                'overall': 0.4,
                'energy': 0.15,
                'mood': 0.15,
                'danceability': 0.1,
                'quality': 0.1,
                'originality': 0.1
            }
        
        total = 0.0
        weight_sum = 0.0
        
        for key, weight in weights.items():
            value = getattr(self, key, 0)
            if value > 0:
                total += value * weight
                weight_sum += weight
        
        return total / weight_sum if weight_sum > 0 else 0.0


class RatingsManager:
    """Manages ratings for tracks"""
    
    def __init__(self, use_db: bool = True):
        self.use_db = use_db and HAS_DB
        self._ratings: List[TrackRating] = []
        self._next_id = 1
        self._load()
    
    def _load(self) -> None:
        """Load ratings from storage"""
        if self.use_db:
            self._load_from_db()
        else:
            self._load_from_file()
    
    def _load_from_db(self) -> None:
        """Load ratings from SQLite database"""
        if not os.path.exists(DB_PATH):
            self.use_db = False
            self._load_from_file()
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create ratings table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER NOT NULL,
                    song_name TEXT NOT NULL,
                    song_path TEXT,
                    artist TEXT,
                    genre TEXT,
                    overall INTEGER DEFAULT 0,
                    energy REAL DEFAULT 0.0,
                    mood REAL DEFAULT 0.0,
                    danceability REAL DEFAULT 0.0,
                    quality REAL DEFAULT 0.0,
                    originality REAL DEFAULT 0.0,
                    tags TEXT,
                    notes TEXT,
                    rated_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    rating_count INTEGER DEFAULT 1
                )
            """)
            
            cursor.execute("SELECT * FROM ratings ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    data = dict(zip(columns, row))
                    # Convert tags string to list
                    if data.get('tags'):
                        data['tags'] = [t.strip() for t in data['tags'].split(',') if t.strip()]
                    else:
                        data['tags'] = []
                    self._ratings.append(TrackRating(**data))
                self._next_id = max(r.id for r in self._ratings) + 1
            else:
                conn.close()
                self.use_db = False
                self._load_from_file()
                return
                
            conn.close()
        except Exception as e:
            print(f"Error loading ratings from DB: {e}")
            self.use_db = False
            self._load_from_file()
    
    def _load_from_file(self) -> None:
        """Load ratings from JSON file"""
        if not os.path.exists(RATINGS_FILE):
            return
        
        try:
            with open(RATINGS_FILE, 'r') as f:
                data = json.load(f)
            
            self._ratings = [TrackRating.from_dict(r) for r in data.get('ratings', [])]
            self._next_id = data.get('next_id', 1)
        except Exception as e:
            print(f"Error loading ratings from file: {e}")
            self._ratings = []
            self._next_id = 1
    
    def _save(self) -> None:
        """Save ratings to storage"""
        if self.use_db:
            self._save_to_db()
        else:
            self._save_to_file()
    
    def _save_to_db(self) -> None:
        """Save ratings to SQLite database"""
        if not HAS_DB:
            self.use_db = False
            self._save_to_file()
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER NOT NULL,
                    song_name TEXT NOT NULL,
                    song_path TEXT,
                    artist TEXT,
                    genre TEXT,
                    overall INTEGER DEFAULT 0,
                    energy REAL DEFAULT 0.0,
                    mood REAL DEFAULT 0.0,
                    danceability REAL DEFAULT 0.0,
                    quality REAL DEFAULT 0.0,
                    originality REAL DEFAULT 0.0,
                    tags TEXT,
                    notes TEXT,
                    rated_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    rating_count INTEGER DEFAULT 1
                )
            """)
            
            # Clear existing and re-insert
            cursor.execute("DELETE FROM ratings")
            
            for rating in self._ratings:
                tags_str = ','.join(rating.tags) if rating.tags else ''
                cursor.execute("""
                    INSERT INTO ratings (
                        song_id, song_name, song_path, artist, genre,
                        overall, energy, mood, danceability, quality, originality,
                        tags, notes, rated_at, updated_at, rating_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rating.song_id, rating.song_name, rating.song_path,
                    rating.artist, rating.genre, rating.overall,
                    rating.energy, rating.mood, rating.danceability,
                    rating.quality, rating.originality, tags_str,
                    rating.notes, rating.rated_at, rating.updated_at,
                    rating.rating_count
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving ratings to DB: {e}")
            self.use_db = False
            self._save_to_file()
    
    def _save_to_file(self) -> None:
        """Save ratings to JSON file"""
        os.makedirs(os.path.dirname(RATINGS_FILE), exist_ok=True)
        
        data = {
            'ratings': [r.to_dict() for r in self._ratings],
            'next_id': self._next_id,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(RATINGS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_or_update(
        self,
        song_id: int,
        song_name: str = "",
        song_path: str = "",
        artist: str = "",
        genre: str = "",
        overall: int = 0,
        energy: float = 0.0,
        mood: float = 0.0,
        danceability: float = 0.0,
        quality: float = 0.0,
        originality: float = 0.0,
        tags: List[str] = None,
        notes: str = ""
    ) -> TrackRating:
        """Add a new rating or update existing one for a song"""
        existing = self.get(song_id)
        
        if existing:
            # Update existing rating
            existing.song_name = song_name or existing.song_name
            existing.song_path = song_path or existing.song_path
            existing.artist = artist or existing.artist
            existing.genre = genre or existing.genre
            
            if overall > 0:
                existing.overall = overall
            if energy > 0:
                existing.energy = energy
            if mood > 0:
                existing.mood = mood
            if danceability > 0:
                existing.danceability = danceability
            if quality > 0:
                existing.quality = quality
            if originality > 0:
                existing.originality = originality
            if tags:
                existing.tags = list(set(existing.tags + tags))
            if notes:
                existing.notes = notes
            
            existing.updated_at = datetime.now().isoformat()
            existing.rating_count += 1
            
            self._save()
            return existing
        
        # Create new rating
        rating = TrackRating(
            id=self._next_id,
            song_id=song_id,
            song_name=song_name,
            song_path=song_path,
            artist=artist,
            genre=genre,
            overall=overall,
            energy=energy,
            mood=mood,
            danceability=danceability,
            quality=quality,
            originality=originality,
            tags=tags or [],
            notes=notes
        )
        
        self._ratings.insert(0, rating)
        self._next_id += 1
        self._save()
        
        return rating
    
    def remove(self, song_id: int) -> bool:
        """Remove a rating by song_id"""
        for i, rating in enumerate(self._ratings):
            if rating.song_id == song_id:
                del self._ratings[i]
                self._save()
                return True
        return False
    
    def remove_by_id(self, rating_id: int) -> bool:
        """Remove a rating by its ID"""
        for i, rating in enumerate(self._ratings):
            if rating.id == rating_id:
                del self._ratings[i]
                self._save()
                return True
        return False
    
    def get(self, song_id: int) -> Optional[TrackRating]:
        """Get a rating by song_id"""
        for rating in self._ratings:
            if rating.song_id == song_id:
                return rating
        return None
    
    def get_by_id(self, rating_id: int) -> Optional[TrackRating]:
        """Get a rating by its ID"""
        for rating in self._ratings:
            if rating.id == rating_id:
                return rating
        return None
    
    def list_all(self) -> List[TrackRating]:
        """Get all ratings, newest first"""
        return list(self._ratings)
    
    def list_by_genre(self, genre: str) -> List[TrackRating]:
        """Get ratings filtered by genre"""
        return [r for r in self._ratings if r.genre.lower() == genre.lower()]
    
    def list_by_artist(self, artist: str) -> List[TrackRating]:
        """Get ratings filtered by artist"""
        return [r for r in self._ratings if artist.lower() in r.artist.lower()]
    
    def list_top_rated(self, limit: int = 10, min_overall: int = 3) -> List[TrackRating]:
        """Get top rated songs (by overall rating)"""
        sorted_ratings = sorted(
            [r for r in self._ratings if r.overall >= min_overall],
            key=lambda r: r.overall,
            reverse=True
        )
        return sorted_ratings[:limit]
    
    def list_by_tags(self, tags: List[str]) -> List[TrackRating]:
        """Get ratings that have any of the specified tags"""
        tags_lower = [t.lower() for t in tags]
        return [
            r for r in self._ratings
            if any(t.lower() in tags_lower for t in r.tags)
        ]
    
    def list_by_weighted_score(self, min_score: float = 5.0, limit: int = 50) -> List[TrackRating]:
        """Get songs by weighted rating score"""
        sorted_ratings = sorted(
            [r for r in self._ratings if r.get_weighted_rating() >= min_score],
            key=lambda r: r.get_weighted_rating(),
            reverse=True
        )
        return sorted_ratings[:limit]
    
    def search(self, query: str) -> List[TrackRating]:
        """Search ratings by name, artist, genre, or notes"""
        query = query.lower()
        return [
            r for r in self._ratings
            if query in r.song_name.lower()
            or query in r.artist.lower()
            or query in r.genre.lower()
            or query in r.notes.lower()
            or any(query in tag.lower() for tag in r.tags)
        ]
    
    def update_notes(self, song_id: int, notes: str) -> bool:
        """Update notes for a rating"""
        for rating in self._ratings:
            if rating.song_id == song_id:
                rating.notes = notes
                rating.updated_at = datetime.now().isoformat()
                self._save()
                return True
        return False
    
    def add_tags(self, song_id: int, tags: List[str]) -> bool:
        """Add tags to a rating"""
        for rating in self._ratings:
            if rating.song_id == song_id:
                rating.tags = list(set(rating.tags + tags))
                rating.updated_at = datetime.now().isoformat()
                self._save()
                return True
        return False
    
    def remove_tags(self, song_id: int, tags: List[str]) -> bool:
        """Remove tags from a rating"""
        for rating in self._ratings:
            if rating.song_id == song_id:
                rating.tags = [t for t in rating.tags if t.lower() not in [tag.lower() for tag in tags]]
                rating.updated_at = datetime.now().isoformat()
                self._save()
                return True
        return False
    
    def get_genre_ratings_summary(self) -> Dict[str, Dict[str, float]]:
        """Get average ratings by genre"""
        genre_data: Dict[str, List[TrackRating]] = {}
        
        for rating in self._ratings:
            if rating.genre:
                if rating.genre not in genre_data:
                    genre_data[rating.genre] = []
                genre_data[rating.genre].append(rating)
        
        summary = {}
        for genre, ratings in genre_data.items():
            count = len(ratings)
            avg_overall = sum(r.overall for r in ratings) / count if count > 0 else 0
            avg_energy = sum(r.energy for r in ratings) / count if count > 0 else 0
            avg_mood = sum(r.mood for r in ratings) / count if count > 0 else 0
            avg_dance = sum(r.danceability for r in ratings) / count if count > 0 else 0
            
            summary[genre] = {
                'count': count,
                'avg_overall': round(avg_overall, 2),
                'avg_energy': round(avg_energy, 2),
                'avg_mood': round(avg_mood, 2),
                'avg_danceability': round(avg_dance, 2)
            }
        
        return summary
    
    def get_all_tags(self) -> List[str]:
        """Get all unique tags across all ratings"""
        all_tags = set()
        for rating in self._ratings:
            all_tags.update(rating.tags)
        return sorted(list(all_tags))
    
    def count(self) -> int:
        """Get total number of ratings"""
        return len(self._ratings)
    
    def clear(self) -> None:
        """Clear all ratings"""
        self._ratings = []
        self._next_id = 1
        self._save()
    
    def export(self) -> List[Dict[str, Any]]:
        """Export all ratings as list of dicts"""
        return [r.to_dict() for r in self._ratings]
    
    def import_ratings(self, ratings_data: List[Dict[str, Any]]) -> int:
        """Import ratings from list of dicts"""
        imported = 0
        for data in ratings_data:
            try:
                rating = TrackRating.from_dict(data)
                rating.id = self._next_id
                self._ratings.append(rating)
                self._next_id += 1
                imported += 1
            except Exception as e:
                print(f"Error importing rating: {e}")
        
        if imported > 0:
            self._save()
        return imported


# Global ratings manager instance
_ratings_manager: Optional[RatingsManager] = None


def get_ratings_manager() -> RatingsManager:
    """Get or create the global ratings manager"""
    global _ratings_manager
    if _ratings_manager is None:
        _ratings_manager = RatingsManager()
    return _ratings_manager


# Convenience functions
def add_rating(
    song_id: int,
    song_name: str = "",
    song_path: str = "",
    artist: str = "",
    genre: str = "",
    overall: int = 0,
    energy: float = 0.0,
    mood: float = 0.0,
    danceability: float = 0.0,
    quality: float = 0.0,
    originality: float = 0.0,
    tags: List[str] = None,
    notes: str = ""
) -> TrackRating:
    """Add or update a rating for a track"""
    return get_ratings_manager().add_or_update(
        song_id, song_name, song_path, artist, genre,
        overall, energy, mood, danceability, quality, originality, tags, notes
    )


def remove_rating(song_id: int) -> bool:
    """Remove a rating"""
    return get_ratings_manager().remove(song_id)


def get_rating(song_id: int) -> Optional[TrackRating]:
    """Get a specific rating"""
    return get_ratings_manager().get(song_id)


def list_ratings() -> List[TrackRating]:
    """List all ratings"""
    return get_ratings_manager().list_all()


def list_top_rated(limit: int = 10) -> List[TrackRating]:
    """List top rated tracks"""
    return get_ratings_manager().list_top_rated(limit)


def search_ratings(query: str) -> List[TrackRating]:
    """Search ratings"""
    return get_ratings_manager().search(query)


def get_genre_summary() -> Dict[str, Dict[str, float]]:
    """Get ratings summary by genre"""
    return get_ratings_manager().get_genre_ratings_summary()


if __name__ == "__main__":
    # Demo/test
    rm = RatingsManager()
    
    print("Ratings System Test")
    print("=" * 40)
    
    # Add some test ratings
    rm.add_or_update(
        song_id=1,
        song_name="Summer Vibes",
        artist="Beach Boys",
        genre="pop",
        overall=5,
        energy=8.5,
        mood=9.0,
        danceability=8.0,
        quality=7.5,
        originality=6.0,
        tags=["summer", "happy", "party"],
        notes="Perfect summer anthem"
    )
    
    rm.add_or_update(
        song_id=2,
        song_name="Midnight Drive",
        artist="Synthwave Master",
        genre="electronic",
        overall=4,
        energy=9.0,
        mood=7.5,
        danceability=7.0,
        quality=8.5,
        originality=8.0,
        tags=["night", "driving", "retro"]
    )
    
    rm.add_or_update(
        song_id=3,
        song_name="Chill Wave",
        artist="Lo-Fi Artist",
        genre="chillwave",
        overall=5,
        energy=3.0,
        mood=9.5,
        danceability=2.0,
        quality=8.0,
        originality=7.5,
        tags=["chill", "study", "relaxing"]
    )
    
    rm.add_or_update(
        song_id=4,
        song_name="Party Starter",
        artist="DJ Mix",
        genre="house",
        overall=4,
        energy=10.0,
        mood=8.0,
        danceability=10.0,
        quality=7.0,
        originality=5.0,
        tags=["party", "workout"]
    )
    
    print(f"Total ratings: {rm.count()}")
    
    print("\nAll ratings:")
    for r in rm.list_all():
        avg = r.get_average_rating()
        weighted = r.get_weighted_rating()
        print(f"  - {r.song_name} ({r.genre}): {r.overall}★ | avg: {avg:.1f} | weighted: {weighted:.1f}")
    
    print("\nTop rated (overall):")
    for r in rm.list_top_rated(3):
        print(f"  - {r.song_name}: {r.overall}★")
    
    print("\nTop rated (weighted):")
    for r in rm.list_by_weighted_score(min_score=5.0, limit=3):
        print(f"  - {r.song_name}: {r.get_weighted_rating():.1f}")
    
    print("\nFiltering by genre 'electronic':")
    for r in rm.list_by_genre("electronic"):
        print(f"  - {r.song_name}")
    
    print("\nSearching for 'chill':")
    for r in rm.search("chill"):
        print(f"  - {r.song_name}")
    
    print("\nAll tags:", rm.get_all_tags())
    
    print("\nGenre summary:")
    for genre, stats in rm.get_genre_ratings_summary().items():
        print(f"  {genre}: {stats['count']} tracks, avg overall: {stats['avg_overall']}★")
    
    # Test update
    print("\nUpdating rating for song_id=1:")
    r = rm.add_or_update(song_id=1, overall=5, energy=9.0)
    print(f"  New rating count: {r.rating_count}")
    
    print("\nTest complete!")
