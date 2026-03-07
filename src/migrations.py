#!/usr/bin/env python3
"""
Database Migrations System for AI DJ Project

Provides version-controlled schema migrations with support for:
- Migration tracking in the database
- Up/down migrations
- Automatic migration on startup
- Migration creation utilities
"""

import os
import sqlite3
import json
import hashlib
import importlib.util
from datetime import datetime
from typing import Optional, List, Dict, Callable, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path


# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ai_dj.db")
MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")


@dataclass
class Migration:
    """Represents a single migration"""
    version: str  # e.g., "001", "002"
    name: str     # e.g., "add_user_preferences"
    description: str = ""
    
    # The migration functions
    up: Callable[[sqlite3.Connection], None] = field(default=lambda conn: None)
    down: Callable[[sqlite3.Connection], None] = field(default=lambda conn: None)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MigrationRecord:
    """Record of an applied migration"""
    version: str
    name: str
    description: str
    applied_at: str
    checksum: str


class Migrations:
    """Database migrations manager"""
    
    def __init__(self, db_path: str = DB_PATH, migrations_dir: str = MIGRATIONS_DIR):
        self.db_path = db_path
        self.migrations_dir = migrations_dir
        self._ensure_migrations_dir()
        self._ensure_schema_version_table()
    
    def _ensure_migrations_dir(self):
        """Create migrations directory if needed"""
        os.makedirs(self.migrations_dir, exist_ok=True)
    
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
    
    def _ensure_schema_version_table(self):
        """Create schema versions table if not exists"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_versions (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    applied_at TEXT NOT NULL,
                    checksum TEXT NOT NULL
                )
            """)
    
    def _get_applied_migrations(self) -> List[MigrationRecord]:
        """Get list of applied migrations"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version, name, description, applied_at, checksum
                FROM schema_versions
                ORDER BY version ASC
            """)
            return [
                MigrationRecord(
                    version=row["version"],
                    name=row["name"],
                    description=row["description"],
                    applied_at=row["applied_at"],
                    checksum=row["checksum"]
                )
                for row in cursor.fetchall()
            ]
    
    def _compute_checksum(self, version: str, name: str) -> str:
        """Compute checksum for migration"""
        data = f"{version}:{name}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _record_migration(self, migration: Migration):
        """Record a successful migration"""
        checksum = self._compute_checksum(migration.version, migration.name)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO schema_versions (version, name, description, applied_at, checksum)
                VALUES (?, ?, ?, ?, ?)
            """, (
                migration.version,
                migration.name,
                migration.description,
                datetime.now().isoformat(),
                checksum
            ))
    
    def _remove_migration_record(self, version: str):
        """Remove migration record (for rollback)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM schema_versions WHERE version = ?", (version,))
    
    def get_current_version(self) -> Optional[str]:
        """Get current database schema version"""
        migrations = self._get_applied_migrations()
        if migrations:
            return migrations[-1].version
        return None
    
    def get_pending_migrations(self, all_migrations: List[Migration]) -> List[Migration]:
        """Get migrations that haven't been applied yet"""
        applied = {m.version for m in self._get_applied_migrations()}
        return [m for m in all_migrations if m.version not in applied]
    
    def migrate(self, all_migrations: List[Migration], target_version: Optional[str] = None) -> Tuple[int, List[str]]:
        """
        Run pending migrations up to target version.
        
        Returns: (number of migrations applied, list of migration versions applied)
        """
        pending = self.get_pending_migrations(all_migrations)
        
        # Filter by target version if specified
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        applied = []
        for migration in pending:
            print(f"📦 Applying migration {migration.version}: {migration.name}")
            print(f"   {migration.description}")
            
            try:
                with self.get_connection() as conn:
                    migration.up(conn)
                self._record_migration(migration)
                applied.append(migration.version)
                print(f"   ✅ Applied successfully")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                raise
        
        return len(applied), applied
    
    def rollback(self, all_migrations: List[Migration], steps: int = 1) -> Tuple[int, List[str]]:
        """
        Roll back the last N migrations.
        
        Returns: (number of migrations rolled back, list of versions rolled back)
        """
        applied = self._get_applied_migrations()
        
        if not applied:
            print("No migrations to roll back")
            return 0, []
        
        # Get migrations to roll back (in reverse order)
        to_rollback = list(reversed(applied[-steps:]))
        
        rolled_back = []
        for record in to_rollback:
            print(f"🔄 Rolling back migration {record.version}: {record.name}")
            
            # Find the migration definition
            migration = next((m for m in all_migrations if m.version == record.version), None)
            if migration is None:
                print(f"   ⚠️  Migration definition not found, removing record only")
                self._remove_migration_record(record.version)
                rolled_back.append(record.version)
                continue
            
            try:
                with self.get_connection() as conn:
                    migration.down(conn)
                self._remove_migration_record(record.version)
                rolled_back.append(record.version)
                print(f"   ✅ Rolled back successfully")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                raise
        
        return len(rolled_back), rolled_back
    
    def reset(self, all_migrations: List[Migration]):
        """Reset database to empty state (remove all migration records)"""
        print("⚠️  Resetting database - this will mark all migrations as pending")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM schema_versions")
        print("✅ Database reset complete")
    
    def status(self, all_migrations: List[Migration]) -> Dict[str, Any]:
        """Show migration status"""
        applied = {m.version for m in self._get_applied_migrations()}
        current = self.get_current_version()
        
        pending = [m for m in all_migrations if m.version not in applied]
        
        return {
            "current_version": current,
            "total_applied": len(applied),
            "total_pending": len(pending),
            "applied_versions": sorted(applied),
            "pending_versions": [m.version for m in pending]
        }
    
    def create_migration_file(self, version: str, name: str, description: str = "") -> str:
        """Create a new migration file template"""
        filename = f"{version}_{name}.py"
        filepath = os.path.join(self.migrations_dir, filename)
        
        if os.path.exists(filepath):
            raise FileExistsError(f"Migration file already exists: {filepath}")
        
        template = f'''"""
Migration: {version} - {name}

{description}
Created: {datetime.now().isoformat()}
"""

from {os.path.basename(self.db_path) or "db"} import MIGRATIONS


def up(conn):
    """Apply this migration"""
    cursor = conn.cursor()
    
    # Example: Add a column
    # cursor.execute("ALTER TABLE songs ADD COLUMN new_column TEXT DEFAULT ''")
    
    pass


def down(conn):
    """Revert this migration"""
    cursor = conn.cursor()
    
    # Example: Drop a column (requires recreate in SQLite)
    # cursor.execute("CREATE TABLE songs_backup AS SELECT id, name, ... FROM songs")
    # cursor.execute("DROP TABLE songs")
    # cursor.execute("ALTER TABLE songs_backup RENAME TO songs")
    
    pass


# Register this migration
MIGRATIONS.append(Migration(
    version="{version}",
    name="{name}",
    description="{description}",
    up=up,
    down=down
))
'''
        with open(filepath, 'w') as f:
            f.write(template)
        
        print(f"✅ Created migration file: {filepath}")
        return filepath


# ==================== Migration Definitions ====================

# Core migrations for the AI DJ database
# Add new migrations here as the schema evolves

def get_core_migrations() -> List[Migration]:
    """Get all core database migrations"""
    migrations = [
        Migration(
            version="001",
            name="initial_schema",
            description="Initial schema with songs, fusions, analyses, playlists, system_states",
            up=lambda conn: _init_initial_schema(conn),
            down=lambda conn: _drop_all_tables(conn)
        ),
    ]
    return migrations


def _init_initial_schema(conn: sqlite3.Connection):
    """Initialize the core database schema (matches database.py)"""
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
            preferences TEXT DEFAULT '{{}}',
            state_json TEXT DEFAULT '{{}}',
            saved_at TEXT NOT NULL
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs(genre)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_created ON songs(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fusions_songs ON fusions(song1_id, song2_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_song ON analyses(song_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_playlists_active ON playlists(is_active)")


def _drop_all_tables(conn: sqlite3.Connection):
    """Drop all core tables (for rollback/reset)"""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS songs")
    cursor.execute("DROP TABLE IF EXISTS fusions")
    cursor.execute("DROP TABLE IF EXISTS analyses")
    cursor.execute("DROP TABLE IF EXISTS playlists")
    cursor.execute("DROP TABLE IF EXISTS system_states")


# Example of how to add more migrations:

def add_example_migration():
    """Example: Adding a new migration for user preferences"""
    return Migration(
        version="002",
        name="add_user_preferences",
        description="Add user preferences table",
        up=lambda conn: _add_user_preferences(conn),
        down=lambda conn: _drop_user_preferences(conn)
    )


def _add_user_preferences(conn: sqlite3.Connection):
    """Add user preferences table"""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL UNIQUE,
            default_genre TEXT DEFAULT 'electronic',
            default_bpm INTEGER DEFAULT 128,
            preferred_keys TEXT DEFAULT '[]',
            energy_range TEXT DEFAULT '[0.3, 0.9]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)


def _drop_user_preferences(conn: sqlite3.Connection):
    """Drop user preferences table"""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS user_preferences")


# ==================== Main API ====================

# Global migrations instance
_migrations: Optional[Migrations] = None


def get_migrations() -> Migrations:
    """Get or create migrations singleton"""
    global _migrations
    if _migrations is None:
        _migrations = Migrations()
    return _migrations


def run_migrations(target_version: Optional[str] = None) -> Tuple[int, List[str]]:
    """
    Run all pending migrations.
    
    This is typically called on application startup.
    
    Args:
        target_version: Optional version to migrate up to (for testing/rollback)
    
    Returns:
        Tuple of (number of migrations applied, list of versions)
    """
    migrations = get_migrations()
    core_migrations = get_core_migrations()
    
    # Could also load migrations from files in migrations directory
    all_migrations = core_migrations + get_file_migrations()
    
    return migrations.migrate(all_migrations, target_version)


def get_file_migrations() -> List[Migration]:
    """Load migrations from files in migrations directory"""
    migrations = []
    migrations_dir = MIGRATIONS_DIR
    
    if not os.path.exists(migrations_dir):
        return migrations
    
    for filename in sorted(os.listdir(migrations_dir)):
        if filename.endswith(".py") and not filename.startswith("_"):
            filepath = os.path.join(migrations_dir, filename)
            
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(
                f"migration_{filename[:-3]}", 
                filepath
            )
            if spec and spec.loader:
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"⚠️  Failed to load migration {filename}: {e}")
    
    return migrations


def migration_status() -> Dict[str, Any]:
    """Get current migration status"""
    return get_migrations().status(get_core_migrations() + get_file_migrations())


def rollback_migrations(steps: int = 1) -> Tuple[int, List[str]]:
    """Roll back the last N migrations"""
    return get_migrations().rollback(get_core_migrations(), steps)


def reset_migrations():
    """Reset all migrations (marks all as pending)"""
    get_migrations().reset(get_core_migrations() + get_file_migrations())


# ==================== CLI ====================

if __name__ == "__main__":
    import sys
    
    migrations = get_migrations()
    all_migrations = get_core_migrations() + get_file_migrations()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            status = migrations.status(all_migrations)
            print("📊 Migration Status")
            print("=" * 40)
            print(f"Current version: {status['current_version'] or 'None'}")
            print(f"Applied: {status['total_applied']}")
            print(f"Pending: {status['total_pending']}")
            if status['pending_versions']:
                print(f"Pending versions: {', '.join(status['pending_versions'])}")
        
        elif command == "migrate":
            applied, versions = migrations.migrate(all_migrations)
            print(f"\n✅ Applied {applied} migration(s)")
            if versions:
                print(f"Versions: {', '.join(versions)}")
        
        elif command == "rollback":
            steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            rolled_back, versions = migrations.rollback(all_migrations, steps)
            print(f"\n✅ Rolled back {rolled_back} migration(s)")
            if versions:
                print(f"Versions: {', '.join(versions)}")
        
        elif command == "reset":
            migrations.reset(all_migrations)
        
        elif command == "create":
            if len(sys.argv) < 4:
                print("Usage: python migrations.py create <version> <name> [description]")
                sys.exit(1)
            version = sys.argv[2]
            name = sys.argv[3]
            description = sys.argv[4] if len(sys.argv) > 4 else ""
            migrations.create_migration_file(version, name, description)
        
        else:
            print(f"Unknown command: {command}")
            print("Available: status, migrate, rollback, reset, create")
            sys.exit(1)
    else:
        # Default: run migrations on startup
        print("🚀 Running database migrations...")
        applied, versions = run_migrations()
        if applied > 0:
            print(f"✅ Applied {applied} migration(s): {', '.join(versions)}")
        else:
            print("✅ Database is up to date")
