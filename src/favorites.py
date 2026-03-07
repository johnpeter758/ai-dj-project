#!/usr/bin/env python3
"""
Favorites System for AI DJ Project

Manages user favorites - songs that the user has marked as favorites
for quick access, playlists, or later listening.
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

# Local favorites file as fallback/primary storage
FAVORITES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "data", 
    "favorites.json"
)


@dataclass
class Favorite:
    """Represents a favorite song entry"""
    id: Optional[int] = None
    song_id: int = 0
    song_name: str = ""
    song_path: str = ""
    genre: str = ""
    bpm: int = 128
    key: str = "C"
    duration: int = 0
    energy: float = 0.0
    mood: str = ""
    added_at: str = ""
    notes: str = ""  # User notes about why it's a favorite
    
    def __post_init__(self):
        if not self.added_at:
            self.added_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Favorite':
        """Create Favorite from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class FavoritesManager:
    """Manages the user's favorite songs"""
    
    def __init__(self, use_db: bool = True):
        self.use_db = use_db and HAS_DB
        self._favorites: List[Favorite] = []
        self._next_id = 1
        self._load()
    
    def _load(self) -> None:
        """Load favorites from storage"""
        if self.use_db:
            self._load_from_db()
        else:
            self._load_from_file()
    
    def _load_from_db(self) -> None:
        """Load favorites from SQLite database"""
        if not os.path.exists(DB_PATH):
            self.use_db = False
            self._load_from_file()
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create favorites table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER NOT NULL,
                    song_name TEXT NOT NULL,
                    song_path TEXT,
                    genre TEXT,
                    bpm INTEGER DEFAULT 128,
                    key TEXT DEFAULT 'C',
                    duration INTEGER DEFAULT 0,
                    energy REAL DEFAULT 0.0,
                    mood TEXT,
                    added_at TEXT NOT NULL,
                    notes TEXT
                )
            """)
            
            cursor.execute("SELECT * FROM favorites ORDER BY added_at DESC")
            rows = cursor.fetchall()
            
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    data = dict(zip(columns, row))
                    self._favorites.append(Favorite(**data))
                self._next_id = max(f.id for f in self._favorites) + 1
            else:
                # No DB records, try loading from file
                conn.close()
                self.use_db = False
                self._load_from_file()
                return
                
            conn.close()
        except Exception as e:
            print(f"Error loading favorites from DB: {e}")
            self.use_db = False
            self._load_from_file()
    
    def _load_from_file(self) -> None:
        """Load favorites from JSON file"""
        if not os.path.exists(FAVORITES_FILE):
            return
        
        try:
            with open(FAVORITES_FILE, 'r') as f:
                data = json.load(f)
            
            self._favorites = [Favorite.from_dict(f) for f in data.get('favorites', [])]
            self._next_id = data.get('next_id', 1)
        except Exception as e:
            print(f"Error loading favorites from file: {e}")
            self._favorites = []
            self._next_id = 1
    
    def _save(self) -> None:
        """Save favorites to storage"""
        if self.use_db:
            self._save_to_db()
        else:
            self._save_to_file()
    
    def _save_to_db(self) -> None:
        """Save favorites to SQLite database"""
        if not HAS_DB:
            self.use_db = False
            self._save_to_file()
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER NOT NULL,
                    song_name TEXT NOT NULL,
                    song_path TEXT,
                    genre TEXT,
                    bpm INTEGER DEFAULT 128,
                    key TEXT DEFAULT 'C',
                    duration INTEGER DEFAULT 0,
                    energy REAL DEFAULT 0.0,
                    mood TEXT,
                    added_at TEXT NOT NULL,
                    notes TEXT
                )
            """)
            
            # Clear existing and re-insert (simple sync approach)
            cursor.execute("DELETE FROM favorites")
            
            for fav in self._favorites:
                cursor.execute("""
                    INSERT INTO favorites (
                        song_id, song_name, song_path, genre, bpm, key,
                        duration, energy, mood, added_at, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fav.song_id, fav.song_name, fav.song_path, fav.genre,
                    fav.bpm, fav.key, fav.duration, fav.energy, fav.mood,
                    fav.added_at, fav.notes
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving favorites to DB: {e}")
            self.use_db = False
            self._save_to_file()
    
    def _save_to_file(self) -> None:
        """Save favorites to JSON file"""
        os.makedirs(os.path.dirname(FAVORITES_FILE), exist_ok=True)
        
        data = {
            'favorites': [f.to_dict() for f in self._favorites],
            'next_id': self._next_id,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add(
        self,
        song_id: int,
        song_name: str,
        song_path: str = "",
        genre: str = "",
        bpm: int = 128,
        key: str = "C",
        duration: int = 0,
        energy: float = 0.0,
        mood: str = "",
        notes: str = ""
    ) -> Favorite:
        """Add a song to favorites"""
        # Check if already favorited
        if self.is_favorite(song_id):
            for f in self._favorites:
                if f.song_id == song_id:
                    return f
        
        favorite = Favorite(
            id=self._next_id,
            song_id=song_id,
            song_name=song_name,
            song_path=song_path,
            genre=genre,
            bpm=bpm,
            key=key,
            duration=duration,
            energy=energy,
            mood=mood,
            notes=notes
        )
        
        self._favorites.insert(0, favorite)
        self._next_id += 1
        self._save()
        
        return favorite
    
    def remove(self, song_id: int) -> bool:
        """Remove a song from favorites by song_id"""
        for i, fav in enumerate(self._favorites):
            if fav.song_id == song_id:
                del self._favorites[i]
                self._save()
                return True
        return False
    
    def remove_by_id(self, favorite_id: int) -> bool:
        """Remove a favorite by its ID"""
        for i, fav in enumerate(self._favorites):
            if fav.id == favorite_id:
                del self._favorites[i]
                self._save()
                return True
        return False
    
    def is_favorite(self, song_id: int) -> bool:
        """Check if a song is in favorites"""
        return any(f.song_id == song_id for f in self._favorites)
    
    def get(self, song_id: int) -> Optional[Favorite]:
        """Get a favorite by song_id"""
        for fav in self._favorites:
            if fav.song_id == song_id:
                return fav
        return None
    
    def get_by_id(self, favorite_id: int) -> Optional[Favorite]:
        """Get a favorite by its ID"""
        for fav in self._favorites:
            if fav.id == favorite_id:
                return fav
        return None
    
    def list_all(self) -> List[Favorite]:
        """Get all favorites, newest first"""
        return list(self._favorites)
    
    def list_by_genre(self, genre: str) -> List[Favorite]:
        """Get favorites filtered by genre"""
        return [f for f in self._favorites if f.genre.lower() == genre.lower()]
    
    def list_by_mood(self, mood: str) -> List[Favorite]:
        """Get favorites filtered by mood"""
        return [f for f in self._favorites if f.mood.lower() == mood.lower()]
    
    def search(self, query: str) -> List[Favorite]:
        """Search favorites by name, genre, or notes"""
        query = query.lower()
        return [
            f for f in self._favorites
            if query in f.song_name.lower()
            or query in f.genre.lower()
            or query in f.mood.lower()
            or query in f.notes.lower()
        ]
    
    def update_notes(self, song_id: int, notes: str) -> bool:
        """Update notes for a favorite"""
        for fav in self._favorites:
            if fav.song_id == song_id:
                fav.notes = notes
                self._save()
                return True
        return False
    
    def count(self) -> int:
        """Get total number of favorites"""
        return len(self._favorites)
    
    def clear(self) -> None:
        """Clear all favorites"""
        self._favorites = []
        self._next_id = 1
        self._save()
    
    def export(self) -> List[Dict[str, Any]]:
        """Export all favorites as list of dicts"""
        return [f.to_dict() for f in self._favorites]
    
    def import_favorites(self, favorites_data: List[Dict[str, Any]]) -> int:
        """Import favorites from list of dicts"""
        imported = 0
        for data in favorites_data:
            try:
                fav = Favorite.from_dict(data)
                # Assign new ID to avoid conflicts
                fav.id = self._next_id
                self._favorites.append(fav)
                self._next_id += 1
                imported += 1
            except Exception as e:
                print(f"Error importing favorite: {e}")
        
        if imported > 0:
            self._save()
        return imported


# Global favorites manager instance
_favorites_manager: Optional[FavoritesManager] = None


def get_favorites_manager() -> FavoritesManager:
    """Get or create the global favorites manager"""
    global _favorites_manager
    if _favorites_manager is None:
        _favorites_manager = FavoritesManager()
    return _favorites_manager


# Convenience functions
def add_favorite(
    song_id: int,
    song_name: str,
    song_path: str = "",
    genre: str = "",
    bpm: int = 128,
    key: str = "C",
    duration: int = 0,
    energy: float = 0.0,
    mood: str = "",
    notes: str = ""
) -> Favorite:
    """Add a song to favorites"""
    return get_favorites_manager().add(
        song_id, song_name, song_path, genre, bpm, key, duration, energy, mood, notes
    )


def remove_favorite(song_id: int) -> bool:
    """Remove a song from favorites"""
    return get_favorites_manager().remove(song_id)


def is_favorite(song_id: int) -> bool:
    """Check if a song is a favorite"""
    return get_favorites_manager().is_favorite(song_id)


def list_favorites() -> List[Favorite]:
    """List all favorites"""
    return get_favorites_manager().list_all()


def search_favorites(query: str) -> List[Favorite]:
    """Search favorites"""
    return get_favorites_manager().search(query)


def get_favorite(song_id: int) -> Optional[Favorite]:
    """Get a specific favorite"""
    return get_favorites_manager().get(song_id)


if __name__ == "__main__":
    # Demo/test
    fm = FavoritesManager()
    
    print("Favorites System Test")
    print("=" * 40)
    
    # Add some test favorites
    fm.add(
        song_id=1,
        song_name="Summer Vibes",
        song_path="/path/to/summer.mp3",
        genre="pop",
        bpm=120,
        key="G",
        duration=180,
        energy=0.8,
        mood="happy",
        notes="Great for parties"
    )
    
    fm.add(
        song_id=2,
        song_name="Midnight Drive",
        song_path="/path/to/midnight.mp3",
        genre="electronic",
        bpm=128,
        key="Am",
        duration=240,
        energy=0.9,
        mood="energetic"
    )
    
    fm.add(
        song_id=3,
        song_name="Chill Wave",
        song_path="/path/to/chill.mp3",
        genre="chillwave",
        bpm=90,
        key="Em",
        duration=200,
        energy=0.4,
        mood="relaxed"
    )
    
    print(f"Total favorites: {fm.count()}")
    print("\nAll favorites:")
    for f in fm.list_all():
        print(f"  - {f.song_name} ({f.genre}) - {f.bpm} BPM")
    
    print("\nSearching for 'chill':")
    for f in fm.search("chill"):
        print(f"  - {f.song_name}")
    
    print("\nFiltering by genre 'electronic':")
    for f in fm.list_by_genre("electronic"):
        print(f"  - {f.song_name}")
    
    print("\nIs song_id=1 a favorite?", is_favorite(1))
    print("Is song_id=99 a favorite?", is_favorite(99))
    
    # Remove one
    fm.remove(2)
    print(f"\nAfter removing song_id=2: {fm.count()} favorites")
    
    print("\nTest complete!")
