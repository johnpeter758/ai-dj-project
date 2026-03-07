#!/usr/bin/env python3
"""
Tagging System for AI DJ Project

Provides flexible tagging for tracks with predefined categories,
custom tags, automatic tag inference, and filtering capabilities.
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


# =============================================================================
# TAG CATEGORIES AND PREDEFINED VALUES
# =============================================================================

class TagCategory(Enum):
    """Predefined tag categories"""
    GENRE = "genre"
    MOOD = "mood"
    ENERGY = "energy"
    TEMPO = "tempo"
    KEY = "key"
    STYLE = "style"
    INSTRUMENT = "instrument"
    VOCALS = "vocals"
    ERA = "era"
    CUSTOM = "custom"


# Predefined tag values for common categories
GENRE_TAGS = {
    "electronic", "house", "techno", "trance", "dubstep", "drum_and_bass",
    "trap", "hip_hop", "rap", "pop", "rnb", "rock", "metal", "indie",
    "ambient", "chill", "lofi", "experimental", "classical", "jazz",
    "latin", "afrobeats", "reggae", "country", "folk"
}

MOOD_TAGS = {
    "happy", "sad", "energetic", "calm", "aggressive", "melancholic",
    "uplifting", "dark", "dreamy", "romantic", "intense", "peaceful",
    "nostalgic", "optimistic", "pensive", "anxious", "triumphant"
}

ENERGY_TAGS = {
    "low", "medium", "high", "extreme", "building", "drop", "peak",
    "chill", "moderate", "intense"
}

TEMPO_TAGS = {
    "slow", "moderate", "fast", "upbeat", "downtempo", "ballad",
    "midtempo", "accelerating", "decelerating"
}

KEY_TAGS = {
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
    "Cm", "C#m", "Dbm", "Dm", "D#m", "Ebm", "Em", "Fm", "F#m", "Gbm", "Gm", "G#m", "Abm", "Am", "A#m", "Bbm", "Bm"
}

STYLE_TAGS = {
    "acoustic", "synth", "analog", "digital", "hybrid", "organic",
    "minimal", "maximal", "textured", "clean", "lofi", "hi-fi",
    "retro", "modern", "vintage", "futuristic"
}

VOCALS_TAGS = {
    "instrumental", "vocal", "featured", "spoken_word", "chanting",
    "screaming", "whispered", "harmony", "monophonic", "polyphonic"
}

ERA_TAGS = {
    "70s", "80s", "90s", "2000s", "2010s", "2020s", "future",
    "classic", "timeless", "contemporary"
}


# Map category to predefined values
CATEGORY_VALUES = {
    TagCategory.GENRE: GENRE_TAGS,
    TagCategory.MOOD: MOOD_TAGS,
    TagCategory.ENERGY: ENERGY_TAGS,
    TagCategory.TEMPO: TEMPO_TAGS,
    TagCategory.KEY: KEY_TAGS,
    TagCategory.STYLE: STYLE_TAGS,
    TagCategory.VOCALS: VOCALS_TAGS,
    TagCategory.ERA: ERA_TAGS,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrackTag:
    """Represents a single tag"""
    name: str
    category: str
    confidence: float = 1.0  # For auto-inferred tags
    source: str = "manual"  # manual, auto, imported
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source
        }


@dataclass
class TrackTags:
    """Container for all tags on a track"""
    track_id: Optional[int] = None
    file_path: str = ""
    tags: List[TrackTag] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
    
    def add_tag(self, name: str, category: str = "custom", 
                confidence: float = 1.0, source: str = "manual") -> None:
        """Add a tag to the track"""
        # Check if tag already exists
        for tag in self.tags:
            if tag.name.lower() == name.lower() and tag.category == category:
                return  # Already exists
        self.tags.append(TrackTag(name, category, confidence, source))
        self.updated_at = datetime.now().isoformat()
    
    def remove_tag(self, name: str, category: Optional[str] = None) -> bool:
        """Remove a tag from the track"""
        original_count = len(self.tags)
        if category:
            self.tags = [t for not (t.name.lower() == name.lower() and t.category == category)]
        else:
            self.tags = [t for not t.name.lower() == name.lower()]
        self.updated_at = datetime.now().isoformat()
        return len(self.tags) < original_count
    
    def get_tags_by_category(self, category: str) -> List[TrackTag]:
        """Get all tags in a specific category"""
        return [t for t in self.tags if t.category == category]
    
    def has_tag(self, name: str, category: Optional[str] = None) -> bool:
        """Check if track has a specific tag"""
        if category:
            return any(t.name.lower() == name.lower() and t.category == category 
                      for t in self.tags)
        return any(t.name.lower() == name.lower() for t in self.tags)
    
    def get_all_tag_names(self) -> Set[str]:
        """Get all tag names as a set"""
        return {t.name.lower() for t in self.tags}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "file_path": self.file_path,
            "tags": [t.to_dict() for t in self.tags],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackTags":
        """Create TrackTags from dictionary"""
        tags = [TrackTag(**t) for t in data.get("tags", [])]
        return cls(
            track_id=data.get("track_id"),
            file_path=data.get("file_path", ""),
            tags=tags,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", "")
        )


# =============================================================================
# TAG MANAGER
# =============================================================================

class TagManager:
    """
    Manages tags for tracks with database persistence.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize tag manager with database path"""
        if db_path is None:
            import os
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "data", 
                "ai_dj.db"
            )
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self) -> None:
        """Ensure required database tables exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                file_path TEXT,
                tag_name TEXT NOT NULL,
                tag_category TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (track_id) REFERENCES songs(id) ON DELETE CASCADE
            )
        """)
        
        # Index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_track_tags_track_id 
            ON track_tags(track_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_track_tags_category 
            ON track_tags(tag_category)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_track_tags_name 
            ON track_tags(tag_name)
        """)
        
        conn.commit()
        conn.close()
    
    def _row_to_track_tags(self, row: tuple) -> TrackTags:
        """Convert database row to TrackTags object"""
        track_id, file_path = row[1], row[2]
        
        # Get all tags for this track
        all_tags = self.get_tags_for_track(track_id, file_path)
        return all_tags
    
    def add_tag(self, track_id: Optional[int], file_path: str,
                name: str, category: str = "custom",
                confidence: float = 1.0, source: str = "manual") -> bool:
        """Add a tag to a track"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Check if tag already exists
        cursor.execute("""
            SELECT id FROM track_tags 
            WHERE track_id = ? AND file_path = ? AND tag_name = ? AND tag_category = ?
        """, (track_id, file_path, name.lower(), category))
        
        if cursor.fetchone():
            conn.close()
            return False  # Tag already exists
        
        cursor.execute("""
            INSERT INTO track_tags 
            (track_id, file_path, tag_name, tag_category, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (track_id, file_path, name.lower(), category, confidence, source, now, now))
        
        conn.commit()
        conn.close()
        return True
    
    def remove_tag(self, track_id: Optional[int], file_path: str,
                   name: str, category: Optional[str] = None) -> bool:
        """Remove a tag from a track"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute("""
                DELETE FROM track_tags 
                WHERE track_id = ? AND file_path = ? AND tag_name = ? AND tag_category = ?
            """, (track_id, file_path, name.lower(), category))
        else:
            cursor.execute("""
                DELETE FROM track_tags 
                WHERE track_id = ? AND file_path = ? AND tag_name = ?
            """, (track_id, file_path, name.lower()))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    
    def get_tags_for_track(self, track_id: Optional[int], 
                           file_path: str) -> TrackTags:
        """Get all tags for a specific track"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tag_name, tag_category, confidence, source, created_at, updated_at
            FROM track_tags
            WHERE track_id = ? AND file_path = ?
            ORDER BY tag_category, tag_name
        """, (track_id, file_path))
        
        rows = cursor.fetchall()
        conn.close()
        
        tags = [
            TrackTag(name=row[0], category=row[1], confidence=row[2], source=row[3])
            for row in rows
        ]
        
        created_at = rows[0][4] if rows else ""
        updated_at = rows[0][5] if rows else ""
        
        return TrackTags(
            track_id=track_id,
            file_path=file_path,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at
        )
    
    def get_all_unique_tags(self, category: Optional[str] = None) -> List[str]:
        """Get all unique tag names, optionally filtered by category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute("""
                SELECT DISTINCT tag_name FROM track_tags 
                WHERE tag_category = ?
                ORDER BY tag_name
            """, (category,))
        else:
            cursor.execute("""
                SELECT DISTINCT tag_name FROM track_tags ORDER BY tag_name
            """)
        
        tags = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tags
    
    def find_tracks_by_tag(self, tag_name: str, 
                           category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find all tracks with a specific tag"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute("""
                SELECT DISTINCT track_id, file_path FROM track_tags
                WHERE tag_name = ? AND tag_category = ?
            """, (tag_name.lower(), category))
        else:
            cursor.execute("""
                SELECT DISTINCT track_id, file_path FROM track_tags
                WHERE tag_name = ?
            """, (tag_name.lower(),))
        
        results = [{"track_id": row[0], "file_path": row[1]} for row in cursor.fetchall()]
        conn.close()
        return results
    
    def find_tracks_by_tags(self, tags: List[str], 
                            match_all: bool = False) -> List[Dict[str, Any]]:
        """Find tracks that have any or all of the given tags"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tag_lower = [t.lower() for t in tags]
        
        if match_all:
            # Find tracks that have ALL tags
            placeholders = ",".join("?" * len(tag_lower))
            cursor.execute(f"""
                SELECT track_id, file_path, COUNT(DISTINCT tag_name) as tag_count
                FROM track_tags
                WHERE tag_name IN ({placeholders})
                GROUP BY track_id, file_path
                HAVING tag_count = ?
            """, (*tag_lower, len(tag_lower)))
        else:
            # Find tracks that have ANY of the tags
            placeholders = ",".join("?" * len(tag_lower))
            cursor.execute(f"""
                SELECT DISTINCT track_id, file_path FROM track_tags
                WHERE tag_name IN ({placeholders})
            """, (*tag_lower,))
        
        results = [{"track_id": row[0], "file_path": row[1]} for row in cursor.fetchall()]
        conn.close()
        return results
    
    def update_tag(self, track_id: Optional[int], file_path: str,
                   old_name: str, new_name: str,
                   category: Optional[str] = None) -> bool:
        """Update a tag name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute("""
                UPDATE track_tags SET tag_name = ?, updated_at = ?
                WHERE track_id = ? AND file_path = ? AND tag_name = ? AND tag_category = ?
            """, (new_name.lower(), datetime.now().isoformat(),
                  track_id, file_path, old_name.lower(), category))
        else:
            cursor.execute("""
                UPDATE track_tags SET tag_name = ?, updated_at = ?
                WHERE track_id = ? AND file_path = ? AND tag_name = ?
            """, (new_name.lower(), datetime.now().isoformat(),
                  track_id, file_path, old_name.lower()))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated
    
    def merge_tags(self, source_tag: str, target_tag: str,
                   category: Optional[str] = None) -> int:
        """Merge one tag into another (delete source, optionally update references)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete source tags
        if category:
            cursor.execute("""
                DELETE FROM track_tags WHERE tag_name = ? AND tag_category = ?
            """, (source_tag.lower(), category))
        else:
            cursor.execute("""
                DELETE FROM track_tags WHERE tag_name = ?
            """, (source_tag.lower(),))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted
    
    def get_tag_counts(self, category: Optional[str] = None) -> Dict[str, int]:
        """Get count of tracks for each tag"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute("""
                SELECT tag_name, COUNT(*) FROM track_tags
                WHERE tag_category = ?
                GROUP BY tag_name
                ORDER BY COUNT(*) DESC
            """, (category,))
        else:
            cursor.execute("""
                SELECT tag_name, COUNT(*) FROM track_tags
                GROUP BY tag_name
                ORDER BY COUNT(*) DESC
            """)
        
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return counts
    
    def get_categories_with_tags(self) -> Dict[str, List[str]]:
        """Get all categories and their tags"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT tag_category, tag_name FROM track_tags
            ORDER BY tag_category, tag_name
        """)
        
        categories: Dict[str, Set[str]] = {}
        for row in cursor.fetchall():
            cat, name = row[0], row[1]
            if cat not in categories:
                categories[cat] = set()
            categories[cat].add(name)
        
        conn.close()
        return {k: sorted(v) for k, v in categories.items()}
    
    def delete_all_tags_for_track(self, track_id: Optional[int],
                                   file_path: str) -> int:
        """Delete all tags for a specific track"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM track_tags WHERE track_id = ? AND file_path = ?
        """, (track_id, file_path))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted


# =============================================================================
# AUTO-TAGGING UTILITIES
# =============================================================================

class AutoTagger:
    """
    Automatically generates tags based on track metadata and analysis.
    """
    
    @staticmethod
    def infer_from_metadata(track_id: Optional[int], file_path: str,
                           bpm: Optional[float] = None,
                           key: Optional[str] = None,
                           genre: Optional[str] = None,
                           energy: Optional[float] = None,
                           mood: Optional[str] = None) -> List[TrackTag]:
        """
        Infer tags from track metadata.
        """
        tags = []
        
        # BPM-based tempo tags
        if bpm:
            if bpm < 90:
                tags.append(TrackTag("slow", "tempo", 0.8, "auto"))
            elif bpm < 120:
                tags.append(TrackTag("moderate", "tempo", 0.8, "auto"))
            elif bpm < 140:
                tags.append(TrackTag("fast", "tempo", 0.8, "auto"))
            else:
                tags.append(TrackTag("upbeat", "tempo", 0.8, "auto"))
            
            # Genre-specific BPM tags
            for genre_name, (bpm_min, bpm_max) in BPM_RANGES.items():
                if bpm_min <= bpm <= bpm_max:
                    tags.append(TrackTag(genre_name, "genre", 0.6, "auto"))
                    break
        
        # Key tag
        if key:
            # Normalize key format
            key_normalized = key.replace("minor", "m").replace("major", "")
            if key_normalized in KEY_TAGS:
                tags.append(TrackTag(key_normalized, "key", 0.9, "auto"))
        
        # Genre tag
        if genre:
            tags.append(TrackTag(genre.lower(), "genre", 0.9, "auto"))
        
        # Energy-based tags
        if energy is not None:
            if energy < 0.3:
                tags.append(TrackTag("low", "energy", 0.8, "auto"))
                tags.append(TrackTag("chill", "mood", 0.6, "auto"))
            elif energy < 0.6:
                tags.append(TrackTag("moderate", "energy", 0.8, "auto"))
            elif energy < 0.85:
                tags.append(TrackTag("high", "energy", 0.8, "auto"))
                tags.append(TrackTag("energetic", "mood", 0.7, "auto"))
            else:
                tags.append(TrackTag("extreme", "energy", 0.8, "auto"))
                tags.append(TrackTag("intense", "mood", 0.7, "auto"))
        
        # Mood tag
        if mood:
            tags.append(TrackTag(mood.lower(), "mood", 0.9, "auto"))
        
        return tags
    
    @staticmethod
    def infer_from_filename(file_path: str) -> List[TrackTag]:
        """
        Infer tags from filename patterns.
        """
        tags = []
        filename = Path(file_path).stem.lower()
        
        # Common patterns in filenames
        mood_indicators = {
            "happy": "happy", "joy": "happy", "uplift": "uplifting",
            "sad": "sad", "melancholy": "melancholic", "blue": "sad",
            "dark": "dark", "night": "dark", "shadow": "dark",
            "energy": "energetic", "power": "energetic", "drive": "energetic",
            "chill": "chill", "relax": "calm", "ambient": "ambient",
            "party": "energetic", "club": "energetic", "bass": "energetic",
        }
        
        genre_indicators = {
            "house": "house", "techno": "techno", "trance": "trance",
            "dubstep": "dubstep", "drum": "drum_and_bass", "dnb": "drum_and_bass",
            "trap": "trap", "hiphop": "hip_hop", "rap": "hip_hop",
            "pop": "pop", "edm": "electronic", "electro": "electronic",
            "lofi": "lofi", "lo-fi": "lofi", "chill": "chill",
        }
        
        for indicator, tag in mood_indicators.items():
            if indicator in filename:
                tags.append(TrackTag(tag, "mood", 0.4, "auto"))
                break
        
        for indicator, tag in genre_indicators.items():
            if indicator in filename:
                tags.append(TrackTag(tag, "genre", 0.4, "auto"))
                break
        
        # Era detection
        era_indicators = {"70s": "70s", "80s": "80s", "90s": "90s", 
                         "2000": "2000s", "2010": "2010s", "2020": "2020s"}
        for indicator, era in era_indicators.items():
            if indicator in filename:
                tags.append(TrackTag(era, "era", 0.5, "auto"))
                break
        
        return tags


# BPM ranges for auto-tagging
BPM_RANGES = {
    "dubstep": (70, 75),
    "trap": (140, 180),
    "house": (118, 130),
    "techno": (120, 150),
    "trance": (138, 145),
    "drum_and_bass": (160, 180),
    "hip_hop": (80, 100),
    "pop": (100, 128),
    "edm": (128, 140),
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_tag_manager(db_path: Optional[str] = None) -> TagManager:
    """Create a TagManager instance"""
    return TagManager(db_path)


def quick_tag(track_id: Optional[int], file_path: str, 
              *tag_names: str, category: str = "custom") -> bool:
    """Quickly add multiple tags to a track"""
    manager = TagManager()
    for tag in tag_names:
        manager.add_tag(track_id, file_path, tag, category)
    return True


def quick_search(*tag_names: str, match_all: bool = False) -> List[Dict[str, Any]]:
    """Quickly search tracks by tags"""
    manager = TagManager()
    return manager.find_tracks_by_tags(list(tag_names), match_all)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test the tagging system
    print("Testing Tagging System...")
    
    # Create manager
    manager = TagManager()
    
    # Test adding tags
    test_file = "/test/track.mp3"
    manager.add_tag(1, test_file, "house", "genre")
    manager.add_tag(1, test_file, "energetic", "mood")
    manager.add_tag(1, test_file, "high", "energy")
    manager.add_tag(1, test_file, "128", "tempo")  # BPM as tag
    
    # Get tags
    tags = manager.get_tags_for_track(1, test_file)
    print(f"Tags for track: {[t.name for t in tags.tags]}")
    
    # Auto-tag from metadata
    auto_tags = AutoTagger.infer_from_metadata(
        track_id=1, 
        file_path=test_file,
        bpm=128,
        key="C",
        genre="house",
        energy=0.85
    )
    print(f"Auto-inferred tags: {[t.name for t in auto_tags]}")
    
    # Get all unique tags
    all_tags = manager.get_all_unique_tags()
    print(f"All unique tags: {all_tags}")
    
    # Get tag counts
    counts = manager.get_tag_counts()
    print(f"Tag counts: {counts}")
    
    print("\nTagging system ready!")
