#!/usr/bin/env python3
"""
Search System for AI DJ Project

Provides flexible search and filtering capabilities for tracks/songs
in the AI DJ library. Supports text search, metadata filtering,
and relevance ranking.
"""

import os
import sqlite3
import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ai_dj.db")


class SortBy(Enum):
    """Sort options for search results."""
    RELEVANCE = "relevance"
    NAME = "name"
    DATE_NEWEST = "date_newest"
    DATE_OLDEST = "date_oldest"
    BPM = "bpm"
    ENERGY = "energy"
    DURATION = "duration"


@dataclass
class SearchFilters:
    """Filter criteria for track search."""
    query: Optional[str] = None  # Text search in name/prompt
    genres: Optional[List[str]] = None
    bpm_min: Optional[int] = None
    bpm_max: Optional[int] = None
    keys: Optional[List[str]] = None
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None
    moods: Optional[List[str]] = None
    duration_min: Optional[int] = None  # seconds
    duration_max: Optional[int] = None  # seconds
    has_file: Optional[bool] = None  # Only tracks with audio files
    date_from: Optional[str] = None  # ISO date string
    date_to: Optional[str] = None    # ISO date string


@dataclass
class SearchResult:
    """Single search result with metadata."""
    id: int
    name: str
    prompt: str
    genre: str
    bpm: int
    key: str
    duration: int
    energy: float
    mood: str
    file_path: str
    created_at: str
    relevance_score: float = 0.0


class TrackSearch:
    """Main search engine for tracks."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        """Ensure database and tables exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create songs table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                prompt TEXT DEFAULT '',
                genre TEXT DEFAULT 'pop',
                bpm INTEGER DEFAULT 128,
                key TEXT DEFAULT 'C',
                duration INTEGER DEFAULT 180,
                energy REAL DEFAULT 0.8,
                mood TEXT DEFAULT 'neutral',
                file_path TEXT DEFAULT '',
                stem_paths TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Create FTS virtual table for full-text search if not exists
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS songs_fts USING fts5(
                name, prompt, genre, mood,
                content='songs',
                content_rowid='id'
            )
        """)
        
        # Rebuild FTS index if out of sync
        try:
            cursor.execute("INSERT INTO songs_fts(songs_fts) VALUES('rebuild')")
        except:
            pass  # Index is fine
        
        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS songs_ai AFTER INSERT ON songs BEGIN
                INSERT INTO songs_fts(rowid, name, prompt, genre, mood)
                VALUES (new.id, new.name, new.prompt, new.genre, new.mood);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS songs_ad AFTER DELETE ON songs BEGIN
                INSERT INTO songs_fts(songs_fts, rowid, name, prompt, genre, mood)
                VALUES ('delete', old.id, old.name, old.prompt, old.genre, old.mood);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS songs_au AFTER UPDATE ON songs BEGIN
                INSERT INTO songs_fts(songs_fts, rowid, name, prompt, genre, mood)
                VALUES ('delete', old.id, old.name, old.prompt, old.genre, old.mood);
                INSERT INTO songs_fts(rowid, name, prompt, genre, mood)
                VALUES (new.id, new.name, new.prompt, new.genre, new.mood);
            END
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def search(
        self,
        filters: SearchFilters,
        sort_by: SortBy = SortBy.RELEVANCE,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[SearchResult], int]:
        """
        Search tracks with given filters.
        
        Returns: (results, total_count)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build WHERE clause
        conditions = []
        params = []
        
        # Full-text search
        if filters.query:
            query = self._prepare_fts_query(filters.query)
            conditions.append(f"""
                id IN (
                    SELECT rowid FROM songs_fts WHERE songs_fts MATCH ?
                )
            """)
            params.append(query)
        
        # Genre filter
        if filters.genres:
            placeholders = ','.join('?' * len(filters.genres))
            conditions.append(f"genre IN ({placeholders})")
            params.extend([g.lower() for g in filters.genres])
        
        # BPM range
        if filters.bpm_min is not None:
            conditions.append("bpm >= ?")
            params.append(filters.bpm_min)
        if filters.bpm_max is not None:
            conditions.append("bpm <= ?")
            params.append(filters.bpm_max)
        
        # Key filter
        if filters.keys:
            placeholders = ','.join('?' * len(filters.keys))
            conditions.append(f"LOWER(`key`) IN ({placeholders})")
            params.extend([k.lower() for k in filters.keys])
        
        # Energy range
        if filters.energy_min is not None:
            conditions.append("energy >= ?")
            params.append(filters.energy_min)
        if filters.energy_max is not None:
            conditions.append("energy <= ?")
            params.append(filters.energy_max)
        
        # Mood filter
        if filters.moods:
            placeholders = ','.join('?' * len(filters.moods))
            conditions.append(f"LOWER(mood) IN ({placeholders})")
            params.extend([m.lower() for m in filters.moods])
        
        # Duration range
        if filters.duration_min is not None:
            conditions.append("duration >= ?")
            params.append(filters.duration_min)
        if filters.duration_max is not None:
            conditions.append("duration <= ?")
            params.append(filters.duration_max)
        
        # Has file
        if filters.has_file is not None:
            if filters.has_file:
                conditions.append("file_path != ''")
            else:
                conditions.append("file_path = ''")
        
        # Date range
        if filters.date_from:
            conditions.append("created_at >= ?")
            params.append(filters.date_from)
        if filters.date_to:
            conditions.append("created_at <= ?")
            params.append(filters.date_to)
        
        # Build query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Get total count
        count_sql = f"SELECT COUNT(*) FROM songs WHERE {where_clause}"
        cursor.execute(count_sql, params)
        total = cursor.fetchone()[0]
        
        # Build ORDER BY
        order_clause = self._get_order_by(sort_by, filters.query)
        
        # Main query
        sql = f"""
            SELECT id, name, prompt, genre, bpm, `key`, duration, 
                   energy, mood, file_path, created_at
            FROM songs
            WHERE {where_clause}
            {order_clause}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to SearchResult
        results = []
        for row in rows:
            score = self._calculate_relevance(row, filters.query) if filters.query else 1.0
            results.append(SearchResult(
                id=row["id"],
                name=row["name"],
                prompt=row["prompt"],
                genre=row["genre"],
                bpm=row["bpm"],
                key=row["key"],
                duration=row["duration"],
                energy=row["energy"],
                mood=row["mood"],
                file_path=row["file_path"],
                created_at=row["created_at"],
                relevance_score=score
            ))
        
        # Sort by relevance if needed
        if sort_by == SortBy.RELEVANCE and filters.query:
            results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results, total
    
    def _prepare_fts_query(self, query: str) -> str:
        """Prepare query for FTS5."""
        # Escape special characters and add wildcards
        query = re.sub(r'[^\w\s]', ' ', query)
        terms = query.split()
        return ' '.join(f'{term}*' for term in terms if term)
    
    def _get_order_by(self, sort_by: SortBy, query: Optional[str]) -> str:
        """Get ORDER BY clause."""
        # Note: We don't use FTS bm25() in ORDER BY since we're querying
        # the songs table, not FTS. Relevance scoring is done in Python
        # after fetching results.
        mapping = {
            SortBy.RELEVANCE: "ORDER BY created_at DESC",  # Will be re-sorted by relevance in Python
            SortBy.NAME: "ORDER BY name COLLATE NOCASE ASC",
            SortBy.DATE_NEWEST: "ORDER BY created_at DESC",
            SortBy.DATE_OLDEST: "ORDER BY created_at ASC",
            SortBy.BPM: "ORDER BY bpm ASC",
            SortBy.ENERGY: "ORDER BY energy DESC",
            SortBy.DURATION: "ORDER BY duration ASC",
        }
        return mapping.get(sort_by, "ORDER BY created_at DESC")
    
    def _calculate_relevance(self, row: sqlite3.Row, query: str) -> float:
        """Calculate simple relevance score."""
        score = 0.0
        query_lower = query.lower()
        
        # Name match (highest weight)
        if query_lower in row["name"].lower():
            score += 10.0
        
        # Genre match
        if query_lower in row["genre"].lower():
            score += 5.0
        
        # Mood match
        if query_lower in row["mood"].lower():
            score += 3.0
        
        # Prompt match
        if query_lower in row["prompt"].lower():
            score += 2.0
        
        return score
    
    def get_by_id(self, track_id: int) -> Optional[SearchResult]:
        """Get single track by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, prompt, genre, bpm, `key`, duration,
                   energy, mood, file_path, created_at
            FROM songs WHERE id = ?
        """, (track_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return SearchResult(
                id=row["id"],
                name=row["name"],
                prompt=row["prompt"],
                genre=row["genre"],
                bpm=row["bpm"],
                key=row["key"],
                duration=row["duration"],
                energy=row["energy"],
                mood=row["mood"],
                file_path=row["file_path"],
                created_at=row["created_at"]
            )
        return None
    
    def get_genres(self) -> List[str]:
        """Get all unique genres."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT genre FROM songs ORDER BY genre")
        genres = [row[0] for row in cursor.fetchall()]
        conn.close()
        return genres
    
    def get_moods(self) -> List[str]:
        """Get all unique moods."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT mood FROM songs ORDER BY mood")
        moods = [row[0] for row in cursor.fetchall()]
        conn.close()
        return moods
    
    def get_keys(self) -> List[str]:
        """Get all musical keys."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT `key` FROM songs ORDER BY `key`")
        keys = [row[0] for row in cursor.fetchall()]
        conn.close()
        return keys
    
    def get_bpm_range(self) -> Tuple[int, int]:
        """Get min/max BPM in library."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(bpm), MAX(bpm) FROM songs")
        row = cursor.fetchone()
        conn.close()
        return (row[0] or 0, row[1] or 200)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM songs")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM songs WHERE file_path != ''")
        with_files = cursor.fetchone()[0]
        
        bpm_min, bpm_max = self.get_bpm_range()
        
        stats = {
            "total_tracks": total,
            "tracks_with_files": with_files,
            "bpm_range": {"min": bpm_min, "max": bpm_max},
            "genres": self.get_genres(),
            "moods": self.get_moods(),
            "keys": self.get_keys()
        }
        
        conn.close()
        return stats
    
    def suggest_similar(
        self,
        track_id: int,
        limit: int = 10
    ) -> List[SearchResult]:
        """Find similar tracks based on genre, BPM, key, energy."""
        track = self.get_by_id(track_id)
        if not track:
            return []
        
        filters = SearchFilters(
            genres=[track.genre],
            bpm_min=max(60, track.bpm - 10),
            bpm_max=track.bpm + 10,
            energy_min=max(0.0, track.energy - 0.2),
            energy_max=min(1.0, track.energy + 0.2),
            keys=[track.key]
        )
        
        results, _ = self.search(filters, limit=limit + 1)
        
        # Exclude the original track
        return [r for r in results if r.id != track_id][:limit]


# Convenience functions
def search_tracks(
    query: Optional[str] = None,
    genre: Optional[str] = None,
    bpm: Optional[int] = None,
    key: Optional[str] = None,
    energy: Optional[float] = None,
    mood: Optional[str] = None,
    limit: int = 50
) -> List[SearchResult]:
    """
    Simple search interface for quick lookups.
    
    Example:
        results = search_tracks("upbeat", genre="house", bpm=128)
    """
    search = TrackSearch()
    
    # Build filters
    filters = SearchFilters(
        query=query,
        genres=[genre] if genre else None,
        bpm_min=bpm,
        bpm_max=bpm,
        keys=[key] if key else None,
        energy_min=energy,
        energy_max=energy,
        moods=[mood] if mood else None
    )
    
    results, _ = search.search(filters, limit=limit)
    return results


if __name__ == "__main__":
    # Demo usage
    search = TrackSearch()
    
    print("=== Library Stats ===")
    stats = search.get_stats()
    print(f"Total tracks: {stats['total_tracks']}")
    print(f"Genres: {stats['genres']}")
    print(f"BPM range: {stats['bpm_range']}")
    
    print("\n=== Search: all tracks ===")
    results, total = search.search(SearchFilters(), limit=5)
    print(f"Found {total} tracks")
    for r in results:
        print(f"  - {r.name} ({r.genre}, {r.bpm} BPM)")
    
    print("\n=== Search: genre=pop ===")
    results, total = search.search(SearchFilters(genres=["pop"]), limit=5)
    print(f"Found {total} tracks")
    for r in results:
        print(f"  - {r.name} ({r.bpm} BPM, energy={r.energy})")
