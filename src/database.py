#!/usr/bin/env python3
"""
Database Module for AI DJ Project

Provides SQLite-based storage for songs, fusions, analyses, playlists,
user preferences, and system state.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Database path - stored in project data directory
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ai_dj.db")


@dataclass
class Song:
    """Represents a generated song"""
    id: Optional[int] = None
    name: str = ""
    prompt: str = ""
    genre: str = "pop"
    bpm: int = 128
    key: str = "C"
    duration: int = 180  # seconds
    energy: float = 0.8
    mood: str = "neutral"
    file_path: str = ""
    stem_paths: str = ""  # JSON dict of stem paths
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class Fusion:
    """Represents a song fusion"""
    id: Optional[int] = None
    name: str = ""
    song1_id: int = 0
    song2_id: int = 0
    genre: str = "electronic"
    bpm: int = 128
    key: str = "C"
    duration: int = 240
    energy: float = 0.85
    file_path: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now


@dataclass
class Analysis:
    """Audio analysis results"""
    id: Optional[int] = None
    song_id: Optional[int] = None
    file_path: str = ""
    bpm: float = 0.0
    key: str = ""
    key_confidence: float = 0.0
    energy: float = 0.0
    danceability: float = 0.0
    tempo_curve: str = ""  # JSON array
    spectral_data: str = ""  # JSON array
    waveform: str = ""  # JSON array
    analyzed_at: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.analyzed_at:
            self.analyzed_at = now


@dataclass
class Playlist:
    """Playlist/queue"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    song_ids: str = "[]"  # JSON array
    current_index: int = 0
    is_active: bool = False
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class SystemState:
    """System state snapshot"""
    id: Optional[int] = None
    songs_generated: int = 0
    fusions_created: int = 0
    total_playtime: int = 0  # seconds
    last_song_id: Optional[int] = None
    last_fusion_id: Optional[int] = None
    preferences: str = "{}"  # JSON dict
    state_json: str = "{}"
    saved_at: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.saved_at:
            self.saved_at = now


class Database:
    """Main database interface"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_db()
    
    def _ensure_db_directory(self):
        """Create database directory if needed"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Songs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    prompt TEXT,
                    genre TEXT DEFAULT 'pop',
                    bpm INTEGER DEFAULT 128,
                    key TEXT DEFAULT 'C',
                    duration INTEGER DEFAULT 180,
                    energy REAL DEFAULT 0.8,
                    mood TEXT DEFAULT 'neutral',
                    file_path TEXT,
                    stem_paths TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Fusions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fusions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    song1_id INTEGER NOT NULL,
                    song2_id INTEGER NOT NULL,
                    genre TEXT DEFAULT 'electronic',
                    bpm INTEGER DEFAULT 128,
                    key TEXT DEFAULT 'C',
                    duration INTEGER DEFAULT 240,
                    energy REAL DEFAULT 0.85,
                    file_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (song1_id) REFERENCES songs(id),
                    FOREIGN KEY (song2_id) REFERENCES songs(id)
                )
            """)
            
            # Analyses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER,
                    file_path TEXT NOT NULL,
                    bpm REAL DEFAULT 0.0,
                    key TEXT DEFAULT '',
                    key_confidence REAL DEFAULT 0.0,
                    energy REAL DEFAULT 0.0,
                    danceability REAL DEFAULT 0.0,
                    tempo_curve TEXT,
                    spectral_data TEXT,
                    waveform TEXT,
                    analyzed_at TEXT NOT NULL,
                    FOREIGN KEY (song_id) REFERENCES songs(id)
                )
            """)
            
            # Playlists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS playlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    song_ids TEXT DEFAULT '[]',
                    current_index INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # System states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    songs_generated INTEGER DEFAULT 0,
                    fusions_created INTEGER DEFAULT 0,
                    total_playtime INTEGER DEFAULT 0,
                    last_song_id INTEGER,
                    last_fusion_id INTEGER,
                    preferences TEXT DEFAULT '{}',
                    state_json TEXT DEFAULT '{}',
                    saved_at TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs(genre)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_created ON songs(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fusions_songs ON fusions(song1_id, song2_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_song ON analyses(song_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_playlists_active ON playlists(is_active)")
    
    # ==================== Song Operations ====================
    
    def add_song(self, song: Song) -> int:
        """Add a new song"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO songs (name, prompt, genre, bpm, key, duration, energy, mood, file_path, stem_paths, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song.name, song.prompt, song.genre, song.bpm, song.key,
                song.duration, song.energy, song.mood, song.file_path,
                song.stem_paths, song.created_at, song.updated_at
            ))
            return cursor.lastrowid
    
    def get_song(self, song_id: int) -> Optional[Song]:
        """Get song by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM songs WHERE id = ?", (song_id,))
            row = cursor.fetchone()
            if row:
                return Song(**dict(row))
            return None
    
    def get_songs(self, genre: Optional[str] = None, limit: int = 100) -> List[Song]:
        """Get songs with optional filtering"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if genre:
                cursor.execute("SELECT * FROM songs WHERE genre = ? ORDER BY created_at DESC LIMIT ?", (genre, limit))
            else:
                cursor.execute("SELECT * FROM songs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [Song(**dict(row)) for row in cursor.fetchall()]
    
    def update_song(self, song: Song) -> bool:
        """Update song"""
        song.updated_at = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE songs SET name=?, prompt=?, genre=?, bpm=?, key=?, duration=?, energy=?, mood=?, file_path=?, stem_paths=?, updated_at=?
                WHERE id=?
            """, (
                song.name, song.prompt, song.genre, song.bpm, song.key,
                song.duration, song.energy, song.mood, song.file_path,
                song.stem_paths, song.updated_at, song.id
            ))
            return cursor.rowcount > 0
    
    def delete_song(self, song_id: int) -> bool:
        """Delete song"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM songs WHERE id = ?", (song_id,))
            return cursor.rowcount > 0
    
    def count_songs(self, genre: Optional[str] = None) -> int:
        """Count songs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if genre:
                cursor.execute("SELECT COUNT(*) FROM songs WHERE genre = ?", (genre,))
            else:
                cursor.execute("SELECT COUNT(*) FROM songs")
            return cursor.fetchone()[0]
    
    # ==================== Fusion Operations ====================
    
    def add_fusion(self, fusion: Fusion) -> int:
        """Add a new fusion"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fusions (name, song1_id, song2_id, genre, bpm, key, duration, energy, file_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fusion.name, fusion.song1_id, fusion.song2_id, fusion.genre,
                fusion.bpm, fusion.key, fusion.duration, fusion.energy,
                fusion.file_path, fusion.created_at
            ))
            return cursor.lastrowid
    
    def get_fusion(self, fusion_id: int) -> Optional[Fusion]:
        """Get fusion by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM fusions WHERE id = ?", (fusion_id,))
            row = cursor.fetchone()
            if row:
                return Fusion(**dict(row))
            return None
    
    def get_fusions(self, limit: int = 100) -> List[Fusion]:
        """Get all fusions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM fusions ORDER BY created_at DESC LIMIT ?", (limit,))
            return [Fusion(**dict(row)) for row in cursor.fetchall()]
    
    def get_fusions_by_song(self, song_id: int) -> List[Fusion]:
        """Get fusions involving a specific song"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM fusions WHERE song1_id = ? OR song2_id = ?
                ORDER BY created_at DESC
            """, (song_id, song_id))
            return [Fusion(**dict(row)) for row in cursor.fetchall()]
    
    def count_fusions(self) -> int:
        """Count fusions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fusions")
            return cursor.fetchone()[0]
    
    # ==================== Analysis Operations ====================
    
    def add_analysis(self, analysis: Analysis) -> int:
        """Add new analysis"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analyses (song_id, file_path, bpm, key, key_confidence, energy, danceability, tempo_curve, spectral_data, waveform, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.song_id, analysis.file_path, analysis.bpm, analysis.key,
                analysis.key_confidence, analysis.energy, analysis.danceability,
                analysis.tempo_curve, analysis.spectral_data, analysis.waveform,
                analysis.analyzed_at
            ))
            return cursor.lastrowid
    
    def get_analysis(self, analysis_id: int) -> Optional[Analysis]:
        """Get analysis by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
            row = cursor.fetchone()
            if row:
                return Analysis(**dict(row))
            return None
    
    def get_analysis_by_song(self, song_id: int) -> Optional[Analysis]:
        """Get analysis for a specific song"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE song_id = ?", (song_id,))
            row = cursor.fetchone()
            if row:
                return Analysis(**dict(row))
            return None
    
    # ==================== Playlist Operations ====================
    
    def add_playlist(self, playlist: Playlist) -> int:
        """Add new playlist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO playlists (name, description, song_ids, current_index, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                playlist.name, playlist.description, playlist.song_ids,
                playlist.current_index, int(playlist.is_active),
                playlist.created_at, playlist.updated_at
            ))
            return cursor.lastrowid
    
    def get_playlist(self, playlist_id: int) -> Optional[Playlist]:
        """Get playlist by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM playlists WHERE id = ?", (playlist_id,))
            row = cursor.fetchone()
            if row:
                return Playlist(**dict(row))
            return None
    
    def get_playlists(self) -> List[Playlist]:
        """Get all playlists"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM playlists ORDER BY updated_at DESC")
            return [Playlist(**dict(row)) for row in cursor.fetchall()]
    
    def get_active_playlist(self) -> Optional[Playlist]:
        """Get currently active playlist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM playlists WHERE is_active = 1 LIMIT 1")
            row = cursor.fetchone()
            if row:
                return Playlist(**dict(row))
            return None
    
    def set_active_playlist(self, playlist_id: int) -> bool:
        """Set a playlist as active (deactivates others)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE playlists SET is_active = 0")
            cursor.execute("UPDATE playlists SET is_active = 1 WHERE id = ?", (playlist_id,))
            return cursor.rowcount > 0
    
    def update_playlist(self, playlist: Playlist) -> bool:
        """Update playlist"""
        playlist.updated_at = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE playlists SET name=?, description=?, song_ids=?, current_index=?, is_active=?, updated_at=?
                WHERE id=?
            """, (
                playlist.name, playlist.description, playlist.song_ids,
                playlist.current_index, int(playlist.is_active),
                playlist.updated_at, playlist.id
            ))
            return cursor.rowcount > 0
    
    # ==================== System State Operations ====================
    
    def save_system_state(self, state: SystemState) -> int:
        """Save system state"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_states (songs_generated, fusions_created, total_playtime, last_song_id, last_fusion_id, preferences, state_json, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.songs_generated, state.fusions_created, state.total_playtime,
                state.last_song_id, state.last_fusion_id, state.preferences,
                state.state_json, state.saved_at
            ))
            return cursor.lastrowid
    
    def get_latest_state(self) -> Optional[SystemState]:
        """Get most recent system state"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_states ORDER BY saved_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return SystemState(**dict(row))
            return None
    
    def get_system_stats(self) -> dict:
        """Get current system statistics"""
        stats = {
            "total_songs": self.count_songs(),
            "total_fusions": self.count_fusions(),
        }
        
        # Get latest state
        state = self.get_latest_state()
        if state:
            stats["songs_generated"] = state.songs_generated
            stats["fusions_created"] = state.fusions_created
            stats["total_playtime"] = state.total_playtime
        
        return stats
    
    # ==================== Utility Methods ====================
    
    def export_json(self, table: str) -> str:
        """Export table as JSON"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            return json.dumps([dict(row) for row in rows], indent=2)
    
    def clear_all(self):
        """Clear all data (use with caution)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analyses")
            cursor.execute("DELETE FROM fusions")
            cursor.execute("DELETE FROM playlists")
            cursor.execute("DELETE FROM system_states")
            cursor.execute("DELETE FROM songs")


# Singleton instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create database singleton"""
    global _db
    if _db is None:
        _db = Database()
    return _db


# Convenience functions
def add_song(name: str, prompt: str = "", genre: str = "pop", **kwargs) -> int:
    """Quick add song"""
    song = Song(name=name, prompt=prompt, genre=genre, **kwargs)
    return get_database().add_song(song)


def get_song(song_id: int) -> Optional[Song]:
    """Quick get song"""
    return get_database().get_song(song_id)


def get_songs(genre: Optional[str] = None, limit: int = 100) -> List[Song]:
    """Quick get songs"""
    return get_database().get_songs(genre, limit)


def add_fusion(name: str, song1_id: int, song2_id: int, **kwargs) -> int:
    """Quick add fusion"""
    fusion = Fusion(name=name, song1_id=song1_id, song2_id=song2_id, **kwargs)
    return get_database().add_fusion(fusion)


def get_fusion(fusion_id: int) -> Optional[Fusion]:
    """Quick get fusion"""
    return get_database().get_fusion(fusion_id)


def add_analysis(song_id: Optional[int], file_path: str, **kwargs) -> int:
    """Quick add analysis"""
    analysis = Analysis(song_id=song_id, file_path=file_path, **kwargs)
    return get_database().add_analysis(analysis)


def get_stats() -> dict:
    """Quick get system stats"""
    return get_database().get_system_stats()


if __name__ == "__main__":
    # Test the database
    db = Database()
    
    print("🎵 AI DJ Database")
    print("=" * 40)
    
    # Test adding a song
    song = Song(
        name="Test Song",
        prompt="upbeat pop track with synth",
        genre="pop",
        bpm=128,
        key="C",
        duration=180,
        energy=0.8
    )
    song_id = db.add_song(song)
    print(f"✅ Added song with ID: {song_id}")
    
    # Test retrieving
    retrieved = db.get_song(song_id)
    print(f"✅ Retrieved song: {retrieved.name}")
    
    # Test getting stats
    stats = db.get_system_stats()
    print(f"📊 Stats: {stats}")
    
    print("\n✅ Database working!")
