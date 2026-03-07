"""
Backup System for AI DJ Project

Provides backup operations for project files, database, configurations, and outputs.
Supports full backups, incremental backups, compression, and retention policies.
"""

import os
import sys
import json
import shutil
import hashlib
import gzip
import tarfile
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/johnpeter/ai-dj-project/logs/backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupDestination(Enum):
    LOCAL = "local"
    GITHUB = "github"
    BOTH = "both"


@dataclass
class BackupConfig:
    """Configuration for backup operations."""
    project_root: str = "/Users/johnpeter/ai-dj-project"
    backup_root: str = "/Users/johnpeter/ai-dj-backups"
    github_repo: str = "https://github.com/johnpeter758/ai-dj-backup.git"
    github_dir: str = "ai-dj-project"
    
    # What to backup
    include_src: bool = True
    include_data: bool = True
    include_database: bool = True
    include_exports: bool = True
    include_fusions: bool = True
    include_logs: bool = False  # Usually don't backup logs
    include_music: bool = False  # Usually don't backup music files
    include_reference: bool = True
    include_research: bool = True
    
    # Exclude patterns
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc",
        "__pycache__",
        "*.log",
        ".git",
        "node_modules",
        "*.wav",
        "*.mp3",
        "*.flac",
        "cache/*",
        "*.tmp",
        ".DS_Store"
    ])
    
    # Retention
    max_full_backups: int = 7
    max_incremental: int = 30
    
    # Compression
    compress: bool = True
    compression_level: int = 6
    
    # Notifications
    notify_on_success: bool = True
    notify_on_failure: bool = True
    discord_channel_id: str = "1479541701923180576"


@dataclass
class BackupManifest:
    """Metadata about a backup."""
    backup_id: str
    backup_type: BackupType
    timestamp: str
    files: List[str]
    total_size: int
    compressed_size: Optional[int] = None
    checksum: Optional[str] = None
    previous_backup_id: Optional[str] = None


class BackupSystem:
    """Main backup system class."""
    
    def __init__(self, config: Optional[BackupConfig] = None):
        self.config = config or BackupConfig()
        self.project_root = Path(self.config.project_root)
        self.backup_root = Path(self.config.backup_root)
        
        # Ensure backup directory exists
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Track backups for incremental/differential
        self.manifest_file = self.backup_root / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """Load backup manifest from disk."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}")
        return {"backups": [], "latest_full": None, "latest_incremental": None}
    
    def _save_manifest(self):
        """Save backup manifest to disk."""
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save manifest: {e}")
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}"
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _get_files_to_backup(self) -> List[Tuple[Path, str]]:
        """Get list of files to backup based on config."""
        files = []
        
        # Define what to include
        includes = []
        if self.config.include_src:
            includes.append(("src", self.project_root / "src"))
        if self.config.include_data:
            includes.append(("data", self.project_root / "data"))
        if self.config.include_database:
            includes.append(("database", self.project_root / "src" / "database.py"))
            db_path = self.project_root / "data" / "database.db"
            if db_path.exists():
                includes.append(("data/database.db", db_path))
        if self.config.include_exports:
            includes.append(("exports", self.project_root / "exports"))
        if self.config.include_fusions:
            includes.append(("fusions", self.project_root / "fusions"))
        if self.config.include_logs:
            includes.append(("logs", self.project_root / "logs"))
        if self.config.include_music:
            includes.append(("music", self.project_root / "music"))
        if self.config.include_reference:
            includes.append(("reference", self.project_root / "reference"))
        if self.config.include_research:
            includes.append(("research", self.project_root / "research"))
        
        for name, path in includes:
            if not path.exists():
                logger.warning(f"Path does not exist: {path}")
                continue
            
            if path.is_file():
                files.append((name, path))
            elif path.is_dir():
                for root, dirs, filenames in os.walk(path):
                    # Filter out excluded patterns
                    dirs[:] = [d for d in dirs if not self._is_excluded(d)]
                    for filename in filenames:
                        if not self._is_excluded(filename):
                            full_path = Path(root) / filename
                            relative_path = full_path.relative_to(self.project_root)
                            files.append((str(relative_path), full_path))
        
        return files
    
    def _is_excluded(self, name: str) -> bool:
        """Check if a file/directory should be excluded."""
        import fnmatch
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
    
    def _create_archive(self, files: List[Tuple[str, Path]], backup_id: str) -> Path:
        """Create a compressed tar archive of the backup."""
        archive_name = f"{backup_id}.tar.gz"
        archive_path = self.backup_root / archive_name
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            for relative_path, full_path in files:
                try:
                    tar.add(full_path, arcname=relative_path)
                except Exception as e:
                    logger.warning(f"Could not add {full_path}: {e}")
        
        return archive_path
    
    def _get_changed_files(self, since_backup_id: Optional[str] = None) -> List[Tuple[Path, str]]:
        """Get files changed since a specific backup."""
        if since_backup_id is None:
            # Get all files
            return self._get_files_to_backup()
        
        # Find the previous backup's manifest
        previous_manifest = None
        for backup in self.manifest.get("backups", []):
            if backup.get("backup_id") == since_backup_id:
                previous_manifest = backup
                break
        
        if previous_manifest is None:
            logger.warning(f"Previous backup {since_backup_id} not found, doing full backup")
            return self._get_files_to_backup()
        
        previous_files = set(previous_manifest.get("files", []))
        all_files = self._get_files_to_backup()
        
        # Filter to only new/modified files
        changed_files = []
        for relative_path, full_path in all_files:
            if relative_path not in previous_files:
                # Check if file was modified
                try:
                    mtime = full_path.stat().st_mtime
                    backup_time = datetime.datetime.fromisoformat(
                        previous_manifest["timestamp"]
                    ).timestamp()
                    if mtime > backup_time:
                        changed_files.append((relative_path, full_path))
                except Exception:
                    changed_files.append((relative_path, full_path))
        
        return changed_files
    
    def create_backup(
        self, 
        backup_type: BackupType = BackupType.FULL,
        destination: BackupDestination = BackupDestination.LOCAL
    ) -> Tuple[bool, Optional[str]]:
        """
        Create a backup.
        
        Args:
            backup_type: Type of backup (FULL, INCREMENTAL, DIFFERENTIAL)
            destination: Where to store the backup
            
        Returns:
            Tuple of (success, backup_id or error_message)
        """
        backup_id = self._generate_backup_id()
        logger.info(f"Starting {backup_type.value} backup: {backup_id}")
        
        try:
            # Determine which files to backup
            if backup_type == BackupType.INCREMENTAL:
                since_id = self.manifest.get("latest_incremental") or self.manifest.get("latest_full")
                files = self._get_changed_files(since_id)
            elif backup_type == BackupType.DIFFERENTIAL:
                since_id = self.manifest.get("latest_full")
                files = self._get_changed_files(since_id)
            else:  # FULL
                files = self._get_files_to_backup()
            
            if not files:
                logger.info("No files to backup")
                return True, backup_id
            
            # Create archive
            if self.config.compress:
                archive_path = self._create_archive(files, backup_id)
                compressed_size = archive_path.stat().st_size
                checksum = self._calculate_checksum(archive_path)
            else:
                # For uncompressed, just use a directory
                archive_path = self.backup_root / backup_id
                archive_path.mkdir(parents=True, exist_ok=True)
                for relative_path, full_path in files:
                    dest = archive_path / relative_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, dest)
                compressed_size = None
                checksum = None
            
            # Calculate total size
            total_size = sum(f.stat().st_size for _, f in files if f.exists())
            
            # Create manifest entry
            manifest_entry = {
                "backup_id": backup_id,
                "backup_type": backup_type.value,
                "timestamp": datetime.datetime.now().isoformat(),
                "files": [f[0] for f in files],
                "total_size": total_size,
                "compressed_size": compressed_size,
                "checksum": checksum,
                "archive_path": str(archive_path)
            }
            
            # Update manifest
            self.manifest["backups"].append(manifest_entry)
            if backup_type == BackupType.FULL:
                self.manifest["latest_full"] = backup_id
            self.manifest["latest_incremental"] = backup_id
            self._save_manifest()
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            # Handle destination
            if destination in (BackupDestination.GITHUB, BackupDestination.BOTH):
                self._sync_to_github(backup_id)
            
            logger.info(f"Backup complete: {backup_id}")
            logger.info(f"  Files: {len(files)}, Size: {total_size / 1024 / 1024:.2f} MB")
            if compressed_size:
                logger.info(f"  Compressed: {compressed_size / 1024 / 1024:.2f} MB")
            
            return True, backup_id
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False, str(e)
    
    def _cleanup_old_backups(self):
        """Remove old backups based on retention policy."""
        backups = self.manifest.get("backups", [])
        
        # Separate by type
        full_backups = [b for b in backups if b["backup_type"] == "full"]
        incremental_backups = [b for b in backups if b["backup_type"] != "full"]
        
        # Remove excess full backups
        if len(full_backups) > self.config.max_full_backups:
            to_remove = full_backups[:-self.config.max_full_backups]
            for backup in to_remove:
                self._remove_backup(backup)
        
        # Remove excess incremental
        if len(incremental_backups) > self.config.max_incremental:
            to_remove = incremental_backups[:-self.config.max_incremental]
            for backup in to_remove:
                self._remove_backup(backup)
    
    def _remove_backup(self, backup: Dict):
        """Remove a backup's files."""
        try:
            archive_path = Path(backup.get("archive_path", ""))
            if archive_path.exists():
                archive_path.unlink()
        except Exception as e:
            logger.warning(f"Could not remove backup file: {e}")
        
        # Remove from manifest
        self.manifest["backups"] = [
            b for b in self.manifest["backups"] 
            if b["backup_id"] != backup["backup_id"]
        ]
    
    def _sync_to_github(self, backup_id: str):
        """Sync backup to GitHub repository."""
        # This would require git operations
        # For now, just log that it would happen
        logger.info(f"Would sync {backup_id} to GitHub: {self.config.github_repo}")
        # TODO: Implement git sync using subprocess
    
    def restore_backup(self, backup_id: str, target_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of the backup to restore
            target_dir: Where to restore to (defaults to project root)
            
        Returns:
            Tuple of (success, message)
        """
        if target_dir is None:
            target_dir = self.project_root
        
        # Find the backup
        backup = None
        for b in self.manifest.get("backups", []):
            if b["backup_id"] == backup_id:
                backup = b
                break
        
        if backup is None:
            return False, f"Backup {backup_id} not found"
        
        archive_path = Path(backup.get("archive_path", ""))
        if not archive_path.exists():
            return False, f"Backup archive not found: {archive_path}"
        
        try:
            # Verify checksum
            if backup.get("checksum"):
                actual_checksum = self._calculate_checksum(archive_path)
                if actual_checksum != backup["checksum"]:
                    return False, "Backup checksum mismatch - file may be corrupted"
            
            # Extract
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(target_dir)
            
            return True, f"Restored backup {backup_id} to {target_dir}"
            
        except Exception as e:
            return False, f"Restore failed: {e}"
    
    def list_backups(self) -> List[Dict]:
        """List all available backups."""
        return self.manifest.get("backups", [])
    
    def get_backup_info(self, backup_id: str) -> Optional[Dict]:
        """Get info about a specific backup."""
        for backup in self.manifest.get("backups", []):
            if backup["backup_id"] == backup_id:
                return backup
        return None


def run_backup(
    backup_type: str = "full",
    destination: str = "local",
    config_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Convenience function to run a backup from command line.
    
    Args:
        backup_type: "full", "incremental", or "differential"
        destination: "local", "github", or "both"
        config_path: Optional path to JSON config file
        
    Returns:
        Tuple of (success, message)
    """
    # Load custom config if provided
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            config = BackupConfig(**config_data)
    
    backup_system = BackupSystem(config)
    
    # Parse types
    bt = BackupType.FULL
    if backup_type == "incremental":
        bt = BackupType.INCREMENTAL
    elif backup_type == "differential":
        bt = BackupType.DIFFERENTIAL
    
    dest = BackupDestination.LOCAL
    if destination == "github":
        dest = BackupDestination.GITHUB
    elif destination == "both":
        dest = BackupDestination.BOTH
    
    return backup_system.create_backup(bt, dest)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI DJ Project Backup System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument(
        "--type", "-t", 
        choices=["full", "incremental", "differential"],
        default="full",
        help="Type of backup"
    )
    backup_parser.add_argument(
        "--destination", "-d",
        choices=["local", "github", "both"],
        default="local",
        help="Backup destination"
    )
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_id", help="ID of backup to restore")
    restore_parser.add_argument(
        "--target", 
        help="Target directory to restore to"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List backups")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get backup info")
    info_parser.add_argument("backup_id", help="ID of backup to inspect")
    
    args = parser.parse_args()
    
    if args.command == "backup":
        success, message = run_backup(args.type, args.destination)
        if success:
            print(f"✓ {message}")
            sys.exit(0)
        else:
            print(f"✗ {message}")
            sys.exit(1)
    
    elif args.command == "restore":
        bs = BackupSystem()
        target = Path(args.target) if args.target else None
        success, message = bs.restore_backup(args.backup_id, target)
        if success:
            print(f"✓ {message}")
            sys.exit(0)
        else:
            print(f"✗ {message}")
            sys.exit(1)
    
    elif args.command == "list":
        bs = BackupSystem()
        backups = bs.list_backups()
        if not backups:
            print("No backups found")
        else:
            print(f"{'Backup ID':<30} {'Type':<12} {'Timestamp':<25} {'Size':<12}")
            print("-" * 80)
            for b in backups:
                size_mb = b.get("total_size", 0) / 1024 / 1024
                print(f"{b['backup_id']:<30} {b['backup_type']:<12} {b['timestamp']:<25} {size_mb:>8.2f} MB")
    
    elif args.command == "info":
        bs = BackupSystem()
        info = bs.get_backup_info(args.backup_id)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print(f"Backup {args.backup_id} not found")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
