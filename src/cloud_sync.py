#!/usr/bin/env python3
"""
Cloud Sync Module for AI DJ Project
Provides cloud backup and synchronization capabilities.

Supports:
- Local file backup with versioning
- Cloud provider integration (Dropbox, S3, Google Drive)
- Incremental sync with change detection
- Encryption options for sensitive files
- Scheduled sync operations

Usage:
    from cloud_sync import CloudSync
    sync = CloudSync()
    sync.backup_all()
    sync.restore("path/to/file", version="latest")
"""

import os
import json
import hashlib
import shutil
import logging
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Sync operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CloudProvider(Enum):
    """Supported cloud providers."""
    LOCAL = "local"
    DROPBOX = "dropbox"
    S3 = "s3"
    GDRIVE = "gdrive"
    ICLOUD = "icloud"


@dataclass
class SyncConfig:
    """Configuration for cloud sync operations."""
    # Local settings
    project_root: str = "/Users/johnpeter/ai-dj-project"
    backup_dir: str = "/Users/johnpeter/ai-dj-backup"
    versions_dir: str = "/Users/johnpeter/ai-dj-backup/versions"
    
    # Cloud settings
    provider: CloudProvider = CloudProvider.LOCAL
    cloud_path: str = ""
    use_encryption: bool = False
    encryption_key: Optional[str] = None
    
    # Sync settings
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", "node_modules",
        "*.log", "*.tmp", ".DS_Store", "Thumbs.db",
        "cache/*", "logs/*", "*.wav", "*.mp3", "*.flac"
    ])
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.json", "*.md", "*.yaml", "*.yml",
        "*.sh", "*.txt", "*.cfg", "*.ini"
    ])
    max_versions: int = 10
    chunk_size: int = 10 * 1024 * 1024  # 10MB chunks
    
    # Scheduling
    auto_sync: bool = False
    sync_interval_minutes: int = 60
    
    # Notifications
    notify_on_complete: bool = True
    notify_on_error: bool = True


@dataclass
class FileInfo:
    """Information about a synced file."""
    path: str
    checksum: str
    size: int
    modified: datetime
    version: int = 1
    synced: Optional[datetime] = None
    status: SyncStatus = SyncStatus.PENDING


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    files_synced: int = 0
    files_failed: int = 0
    bytes_transferred: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class CloudSync:
    """
    Main cloud synchronization class for AI DJ Project.
    
    Provides backup, restore, and sync capabilities with support for
    multiple cloud providers and local versioning.
    """
    
    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig()
        self._sync_lock = threading.Lock()
        self._file_index: Dict[str, FileInfo] = {}
        self._load_index()
        
    def _load_index(self) -> None:
        """Load the file index from disk."""
        index_path = os.path.join(self.config.backup_dir, ".sync_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    for path, info in data.items():
                        info['modified'] = datetime.fromisoformat(info['modified'])
                        if info.get('synced'):
                            info['synced'] = datetime.fromisoformat(info['synced'])
                        self._file_index[path] = FileInfo(**info)
            except Exception as e:
                logger.warning(f"Could not load sync index: {e}")
    
    def _save_index(self) -> None:
        """Save the file index to disk."""
        os.makedirs(self.config.backup_dir, exist_ok=True)
        index_path = os.path.join(self.config.backup_dir, ".sync_index.json")
        data = {
            path: {
                'path': info.path,
                'checksum': info.checksum,
                'size': info.size,
                'modified': info.modified.isoformat(),
                'version': info.version,
                'synced': info.synced.isoformat() if info.synced else None,
                'status': info.status.value
            }
            for path, info in self._file_index.items()
        }
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                while chunk := f.read(self.config.chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {filepath}: {e}")
            return ""
    
    def _should_exclude(self, path: str) -> bool:
        """Check if a file should be excluded based on patterns."""
        from fnmatch import fnmatch
        filename = os.path.basename(path)
        for pattern in self.config.exclude_patterns:
            if fnmatch(filename, pattern) or fnmatch(path, pattern):
                return True
        return False
    
    def _should_include(self, path: str) -> bool:
        """Check if a file should be included based on patterns."""
        if self._should_exclude(path):
            return False
        from fnmatch import fnmatch
        filename = os.path.basename(path)
        if not self.config.include_patterns:
            return True
        for pattern in self.config.include_patterns:
            if fnmatch(filename, pattern) or fnmatch(path, pattern):
                return True
        return False
    
    def _get_files_to_sync(self, directory: str) -> List[str]:
        """Get list of files that should be synced."""
        files = []
        for root, dirs, filenames in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(os.path.join(root, d))]
            
            for filename in filenames:
                filepath = os.path.join(root, filename)
                if self._should_include(filepath):
                    files.append(filepath)
        return files
    
    def _create_version(self, filepath: str) -> str:
        """Create a versioned backup of a file."""
        os.makedirs(self.config.versions_dir, exist_ok=True)
        
        rel_path = os.path.relpath(filepath, self.config.project_root)
        version_path = os.path.join(self.config.versions_dir, rel_path)
        
        # Get version number
        base_version = 1
        if rel_path in self._file_index:
            base_version = self._file_index[rel_path].version + 1
        
        # Create versioned filename
        versioned_name = f"{os.path.splitext(rel_path)[0]}_v{base_version}{os.path.splitext(rel_path)[1]}"
        versioned_path = os.path.join(self.config.versions_dir, versioned_name)
        
        os.makedirs(os.path.dirname(versioned_path), exist_ok=True)
        shutil.copy2(filepath, versioned_path)
        
        # Clean up old versions
        self._cleanup_versions(rel_path)
        
        return versioned_path
    
    def _cleanup_versions(self, rel_path: str) -> None:
        """Remove old versions beyond max_versions."""
        base_name = os.path.splitext(rel_path)[0]
        ext = os.path.splitext(rel_path)[1]
        
        versions = []
        for f in os.listdir(self.config.versions_dir):
            if f.startswith(base_name + "_v") and f.endswith(ext):
                try:
                    v = int(f.split("_v")[1].split(ext)[0])
                    versions.append((v, f))
                except (ValueError, IndexError):
                    continue
        
        versions.sort(reverse=True)
        
        for _, filename in versions[self.config.max_versions:]:
            try:
                os.remove(os.path.join(self.config.versions_dir, filename))
                logger.debug(f"Removed old version: {filename}")
            except Exception as e:
                logger.warning(f"Could not remove old version {filename}: {e}")
    
    def backup_file(self, filepath: str) -> SyncResult:
        """
        Backup a single file to the cloud/backup location.
        
        Args:
            filepath: Path to the file to backup
            
        Returns:
            SyncResult with operation details
        """
        result = SyncResult(success=False, start_time=datetime.now())
        
        if not os.path.exists(filepath):
            result.errors.append(f"File not found: {filepath}")
            return result
        
        if not self._should_include(filepath):
            result.status = SyncStatus.SKIPPED
            result.success = True
            return result
        
        rel_path = os.path.relpath(filepath, self.config.project_root)
        checksum = self._calculate_checksum(filepath)
        
        # Check if file has changed
        if rel_path in self._file_index:
            existing = self._file_index[rel_path]
            if existing.checksum == checksum:
                logger.debug(f"File unchanged: {rel_path}")
                result.success = True
                result.status = SyncStatus.SKIPPED
                return result
        
        # Create backup
        dest_dir = os.path.join(self.config.backup_dir, os.path.dirname(rel_path))
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(self.config.backup_dir, rel_path)
        
        try:
            shutil.copy2(filepath, dest_path)
            file_size = os.path.getsize(filepath)
            
            # Update index
            version = 1
            if rel_path in self._file_index:
                version = self._file_index[rel_path].version + 1
                
            self._file_index[rel_path] = FileInfo(
                path=rel_path,
                checksum=checksum,
                size=file_size,
                modified=datetime.fromtimestamp(os.path.getmtime(filepath)),
                version=version,
                synced=datetime.now(),
                status=SyncStatus.COMPLETED
            )
            
            # Create versioned backup
            self._create_version(filepath)
            
            result.success = True
            result.files_synced = 1
            result.bytes_transferred = file_size
            result.status = SyncStatus.COMPLETED
            logger.info(f"Backed up: {rel_path}")
            
        except Exception as e:
            result.errors.append(f"Failed to backup {rel_path}: {str(e)}")
            result.files_failed = 1
            logger.error(f"Backup failed for {rel_path}: {e}")
        
        result.end_time = datetime.now()
        return result
    
    def backup_all(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> SyncResult:
        """
        Backup all project files to the cloud/backup location.
        
        Args:
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            SyncResult with operation details
        """
        with self._sync_lock:
            result = SyncResult(success=False, start_time=datetime.now())
            
            logger.info("Starting full backup...")
            
            files = self._get_files_to_sync(self.config.project_root)
            total = len(files)
            
            for i, filepath in enumerate(files):
                file_result = self.backup_file(filepath)
                result.files_synced += file_result.files_synced
                result.files_failed += file_result.files_failed
                result.bytes_transferred += file_result.bytes_transferred
                result.errors.extend(file_result.errors)
                
                if progress_callback:
                    progress_callback(i + 1, total)
            
            self._save_index()
            
            result.success = result.files_failed == 0
            result.end_time = datetime.now()
            
            duration = (result.end_time - result.start_time).total_seconds()
            logger.info(f"Backup complete: {result.files_synced} files, "
                       f"{result.bytes_transferred / 1024 / 1024:.2f} MB "
                       f"in {duration:.1f}s")
            
            return result
    
    def restore(self, filepath: str, version: str = "latest") -> SyncResult:
        """
        Restore a file from backup.
        
        Args:
            filepath: Relative path to the file to restore
            version: Version to restore ("latest" or version number)
            
        Returns:
            SyncResult with operation details
        """
        result = SyncResult(success=False, start_time=datetime.now())
        
        rel_path = filepath if not filepath.startswith("/") else os.path.relpath(
            filepath, self.config.project_root
        )
        
        # Find backup file
        backup_path = os.path.join(self.config.backup_dir, rel_path)
        
        if version != "latest":
            # Restore specific version
            base_name = os.path.splitext(rel_path)[0]
            ext = os.path.splitext(rel_path)[1]
            versioned_name = f"{base_name}_v{version}{ext}"
            backup_path = os.path.join(self.config.versions_dir, versioned_name)
        
        if not os.path.exists(backup_path):
            result.errors.append(f"Backup not found: {backup_path}")
            return result
        
        # Restore to original location
        dest_path = os.path.join(self.config.project_root, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        try:
            shutil.copy2(backup_path, dest_path)
            result.success = True
            result.files_synced = 1
            result.bytes_transferred = os.path.getsize(backup_path)
            logger.info(f"Restored: {rel_path} (version: {version})")
        except Exception as e:
            result.errors.append(f"Restore failed: {str(e)}")
            result.files_failed = 1
        
        result.end_time = datetime.now()
        return result
    
    def sync_to_cloud(self) -> SyncResult:
        """
        Sync local backup to cloud provider.
        
        Returns:
            SyncResult with operation details
        """
        result = SyncResult(success=False, start_time=datetime.now())
        
        if self.config.provider == CloudProvider.LOCAL:
            logger.info("Cloud sync: using local backup (no cloud configured)")
            result.success = True
            result.end_time = datetime.now()
            return result
        
        # Cloud provider specific logic
        if self.config.provider == CloudProvider.DROPBOX:
            result = self._sync_dropbox()
        elif self.config.provider == CloudProvider.S3:
            result = self._sync_s3()
        elif self.config.provider == CloudProvider.GDRIVE:
            result = self._sync_gdrive()
        elif self.config.provider == CloudProvider.ICLOUD:
            result = self._sync_icloud()
        
        result.end_time = datetime.now()
        return result
    
    def _sync_dropbox(self) -> SyncResult:
        """Sync to Dropbox."""
        result = SyncResult(success=False, start_time=datetime.now())
        # Dropbox API integration would go here
        # For now, use rclone if available
        try:
            cmd = ["which", "rclone"]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                cmd = [
                    "rclone", "sync", 
                    self.config.backup_dir,
                    f"dropbox:{self.config.cloud_path}",
                    "--progress"
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                result.success = proc.returncode == 0
                if not result.success:
                    result.errors.append(proc.stderr)
        except Exception as e:
            result.errors.append(f"Dropbox sync failed: {str(e)}")
        
        result.end_time = datetime.now()
        return result
    
    def _sync_s3(self) -> SyncResult:
        """Sync to S3."""
        result = SyncResult(success=False, start_time=datetime.now())
        try:
            cmd = ["which", "rclone"]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                cmd = [
                    "rclone", "sync",
                    self.config.backup_dir,
                    f"s3:{self.config.cloud_path}",
                    "--progress"
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                result.success = proc.returncode == 0
                if not result.success:
                    result.errors.append(proc.stderr)
        except Exception as e:
            result.errors.append(f"S3 sync failed: {str(e)}")
        
        result.end_time = datetime.now()
        return result
    
    def _sync_gdrive(self) -> SyncResult:
        """Sync to Google Drive."""
        result = SyncResult(success=False, start_time=datetime.now())
        try:
            cmd = ["which", "rclone"]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                cmd = [
                    "rclone", "sync",
                    self.config.backup_dir,
                    f"gdrive:{self.config.cloud_path}",
                    "--progress"
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                result.success = proc.returncode == 0
                if not result.success:
                    result.errors.append(proc.stderr)
        except Exception as e:
            result.errors.append(f"Google Drive sync failed: {str(e)}")
        
        result.end_time = datetime.now()
        return result
    
    def _sync_icloud(self) -> SyncResult:
        """Sync to iCloud Drive."""
        result = SyncResult(success=False, start_time=datetime.now())
        try:
            # Use macOS built-in iCloud Drive
           icloud_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")
            if os.path.exists(icloud_path):
                dest = os.path.join(icloud_path, self.config.cloud_path)
                os.makedirs(dest, exist_ok=True)
                cmd = ["rsync", "-av", self.config.backup_dir + "/", dest + "/"]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                result.success = proc.returncode == 0
                if not result.success:
                    result.errors.append(proc.stderr)
            else:
                result.errors.append("iCloud Drive not available")
        except Exception as e:
            result.errors.append(f"iCloud sync failed: {str(e)}")
        
        result.end_time = datetime.now()
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        files = self._get_files_to_sync(self.config.project_root)
        
        synced = sum(1 for f in self._file_index.values() 
                   if f.status == SyncStatus.COMPLETED)
        
        # Calculate total size
        total_size = sum(f.size for f in self._file_index.values())
        
        return {
            "provider": self.config.provider.value,
            "project_root": self.config.project_root,
            "backup_dir": self.config.backup_dir,
            "total_files": len(files),
            "synced_files": synced,
            "pending_files": len(files) - synced,
            "total_size_mb": total_size / 1024 / 1024,
            "last_sync": max((f.synced for f in self._file_index.values() 
                            if f.synced), default=None),
            "auto_sync": self.config.auto_sync,
            "sync_interval": self.config.sync_interval_minutes
        }
    
    def list_versions(self, filepath: str) -> List[Dict[str, Any]]:
        """List available versions of a file."""
        rel_path = os.path.relpath(filepath, self.config.project_root)
        base_name = os.path.splitext(rel_path)[0]
        ext = os.path.splitext(rel_path)[0]
        
        versions = []
        
        # Check main backup
        backup_path = os.path.join(self.config.backup_dir, rel_path)
        if os.path.exists(backup_path):
            versions.append({
                "version": "current",
                "path": backup_path,
                "size": os.path.getsize(backup_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(backup_path))
            })
        
        # Check versioned backups
        if os.path.exists(self.config.versions_dir):
            for f in os.listdir(self.config.versions_dir):
                if f.startswith(base_name + "_v") and f.endswith(ext):
                    vpath = os.path.join(self.config.versions_dir, f)
                    try:
                        v = int(f.split("_v")[1].split(ext)[0])
                        versions.append({
                            "version": v,
                            "path": vpath,
                            "size": os.path.getsize(vpath),
                            "modified": datetime.fromtimestamp(os.path.getmtime(vpath))
                        })
                    except (ValueError, IndexError):
                        continue
        
        return sorted(versions, key=lambda x: x.get('version', 0), reverse=True)
    
    def verify_backup(self) -> SyncResult:
        """
        Verify backup integrity by comparing checksums.
        
        Returns:
            SyncResult with verification details
        """
        result = SyncResult(success=False, start_time=datetime.now())
        
        logger.info("Starting backup verification...")
        
        for rel_path, file_info in self._file_index.items():
            backup_path = os.path.join(self.config.backup_dir, rel_path)
            
            if not os.path.exists(backup_path):
                result.errors.append(f"Missing backup: {rel_path}")
                result.files_failed += 1
                continue
            
            current_checksum = self._calculate_checksum(backup_path)
            
            if current_checksum != file_info.checksum:
                result.errors.append(f"Checksum mismatch: {rel_path}")
                result.files_failed += 1
            else:
                result.files_synced += 1
        
        result.success = result.files_failed == 0
        result.end_time = datetime.now()
        
        return result


def create_config(**kwargs) -> SyncConfig:
    """Create a SyncConfig with custom settings."""
    return SyncConfig(**kwargs)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI DJ Cloud Sync")
    parser.add_argument("command", choices=["backup", "restore", "status", "verify", "sync-cloud"],
                       help="Command to execute")
    parser.add_argument("--file", "-f", help="File to restore")
    parser.add_argument("--version", "-v", default="latest", help="Version to restore")
    parser.add_argument("--provider", "-p", default="local", 
                       choices=["local", "dropbox", "s3", "gdrive", "icloud"],
                       help="Cloud provider")
    parser.add_argument("--cloud-path", help="Cloud path/bucket")
    
    args = parser.parse_args()
    
    # Create sync instance
    config = SyncConfig()
    if args.provider != "local":
        config.provider = CloudProvider(args.provider)
        if args.cloud_path:
            config.cloud_path = args.cloud_path
    
    sync = CloudSync(config)
    
    if args.command == "backup":
        result = sync.backup_all()
        print(f"Backup complete: {result.files_synced} files, "
              f"{result.bytes_transferred / 1024 / 1024:.2f} MB")
        if result.errors:
            print(f"Errors: {result.errors}")
            
    elif args.command == "restore":
        if not args.file:
            print("Error: --file required for restore")
            exit(1)
        result = sync.restore(args.file, args.version)
        if result.success:
            print(f"Restored: {args.file}")
        else:
            print(f"Restore failed: {result.errors}")
            
    elif args.command == "status":
        status = sync.get_status()
        print(json.dumps(status, indent=2, default=str))
        
    elif args.command == "verify":
        result = sync.verify_backup()
        print(f"Verification: {result.files_synced} OK, {result.files_failed} failed")
        if result.errors:
            print(f"Errors: {result.errors}")
            
    elif args.command == "sync-cloud":
        result = sync.sync_to_cloud()
        if result.success:
            print("Cloud sync complete")
        else:
            print(f"Cloud sync failed: {result.errors}")
