#!/usr/bin/env python3
"""
Track Import System for AI DJ Project

Handles importing tracks from various sources (local files, URLs, directories),
extracts metadata (BPM, key, duration, energy), and stores track information
in the database.
"""

import os
import json
import hashlib
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
import subprocess

# Audio processing
import numpy as np

# Try to import existing modules
try:
    from database import Database, Song, Analysis
    from audio_utils import get_duration, get_sample_rate
    from key_detector import detect_key
    from beat_detector import detect_bpm
    _HAS_LOCAL_DEPS = True
except ImportError:
    _HAS_LOCAL_DEPS = False


# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff"}


@dataclass
class ImportResult:
    """Result of an import operation"""
    success: bool
    track_id: Optional[int] = None
    file_path: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackMetadata:
    """Extracted track metadata"""
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    duration: float = 0.0  # seconds
    bpm: float = 0.0
    key: str = ""
    key_confidence: float = 0.0
    energy: float = 0.5
    sample_rate: int = 44100
    channels: int = 2
    bitrate: int = 0
    file_size: int = 0
    format: str = ""
    checksum: str = ""


class TrackImporter:
    """Main class for importing tracks into the AI DJ system"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the importer
        
        Args:
            data_dir: Optional custom data directory path
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path("/Users/johnpeter/ai-dj-project/data")
        
        self.tracks_dir = self.data_dir / "tracks"
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        
        self.db = None
        if _HAS_LOCAL_DEPS:
            try:
                self.db = Database()
            except Exception:
                pass
    
    def import_file(self, file_path: str, copy_to_library: bool = True,
                    analyze: bool = True, auto_detect_genre: bool = True) -> ImportResult:
        """Import a single audio file
        
        Args:
            file_path: Path to the audio file
            copy_to_library: Whether to copy file to tracks directory
            analyze: Whether to analyze BPM, key, etc.
            auto_detect_genre: Whether to auto-detect genre
            
        Returns:
            ImportResult with import status and metadata
        """
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            return ImportResult(
                success=False,
                file_path=str(path),
                error=f"File not found: {path}"
            )
        
        # Validate format
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            return ImportResult(
                success=False,
                file_path=str(path),
                error=f"Unsupported format: {path.suffix}"
            )
        
        try:
            # Extract metadata
            metadata = self._extract_metadata(path)
            
            # Copy to library if requested
            if copy_to_library:
                dest_path = self._copy_to_library(path, metadata)
                metadata.file_path = str(dest_path)
            else:
                metadata.file_path = str(path.absolute())
            
            # Analyze if requested
            if analyze:
                analysis = self._analyze_track(metadata)
                metadata.bpm = analysis.get("bpm", metadata.bpm)
                metadata.key = analysis.get("key", metadata.key)
                metadata.key_confidence = analysis.get("key_confidence", 0.0)
                metadata.energy = analysis.get("energy", metadata.energy)
            
            # Auto-detect genre if requested
            if auto_detect_genre and not metadata.genre:
                metadata.genre = self._detect_genre(metadata)
            
            # Save to database
            track_id = self._save_to_database(metadata)
            
            return ImportResult(
                success=True,
                track_id=track_id,
                file_path=metadata.file_path,
                metadata=self._metadata_to_dict(metadata)
            )
            
        except Exception as e:
            return ImportResult(
                success=False,
                file_path=str(path),
                error=str(e)
            )
    
    def import_directory(self, directory: str, recursive: bool = True,
                        copy_to_library: bool = True, analyze: bool = False) -> List[ImportResult]:
        """Import all audio files from a directory
        
        Args:
            directory: Path to directory containing audio files
            recursive: Whether to search subdirectories
            copy_to_library: Whether to copy files to tracks directory
            analyze: Whether to analyze each track
            
        Returns:
            List of ImportResult for each file
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return [ImportResult(
                success=False,
                file_path=directory,
                error=f"Not a directory: {directory}"
            )]
        
        results = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                result = self.import_file(
                    str(file_path),
                    copy_to_library=copy_to_library,
                    analyze=analyze
                )
                results.append(result)
        
        return results
    
    def import_url(self, url: str, analyze: bool = True) -> ImportResult:
        """Import audio from a URL
        
        Args:
            url: URL to audio file
            analyze: Whether to analyze BPM, key, etc.
            
        Returns:
            ImportResult with import status
        """
        parsed = urlparse(url)
        
        # Validate URL
        if not parsed.scheme or not parsed.netloc:
            return ImportResult(
                success=False,
                file_path=url,
                error=f"Invalid URL: {url}"
            )
        
        # Check if supported
        path = Path(parsed.path)
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            return ImportResult(
                success=False,
                file_path=url,
                error=f"Unsupported format from URL: {path.suffix}"
            )
        
        try:
            # Download file
            temp_path = self._download_file(url)
            
            # Import the downloaded file
            result = self.import_file(temp_path, copy_to_library=True, analyze=analyze)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            return ImportResult(
                success=False,
                file_path=url,
                error=f"Download failed: {str(e)}"
            )
    
    def _extract_metadata(self, path: Path) -> TrackMetadata:
        """Extract metadata from audio file
        
        Args:
            path: Path to audio file
            
        Returns:
            TrackMetadata object
        """
        metadata = TrackMetadata()
        
        # Basic file info
        metadata.file_size = path.stat().st_size
        metadata.format = path.suffix.lower()[1:]  # Remove dot
        
        # Generate checksum
        metadata.checksum = self._compute_checksum(path)
        
        # Try to extract ID3 tags using ffprobe
        try:
            tags = self._get_audio_tags(str(path))
            metadata.title = tags.get("title", path.stem)
            metadata.artist = tags.get("artist", "Unknown Artist")
            metadata.album = tags.get("album", "")
            metadata.genre = tags.get("genre", "")
            metadata.bpm = float(tags.get("bpm", 0))
        except:
            metadata.title = path.stem
            metadata.artist = "Unknown Artist"
        
        # Get duration using ffprobe
        try:
            duration = self._get_duration_ffprobe(str(path))
            metadata.duration = duration
        except:
            pass
        
        # Get audio properties
        try:
            audio_info = self._get_audio_info(str(path))
            metadata.sample_rate = audio_info.get("sample_rate", 44100)
            metadata.channels = audio_info.get("channels", 2)
            metadata.bitrate = audio_info.get("bitrate", 0)
        except:
            pass
        
        return metadata
    
    def _get_audio_tags(self, file_path: str) -> Dict[str, str]:
        """Get ID3 tags from audio file using ffprobe
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary of tags
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", file_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                tags = data.get("format", {}).get("tags", {})
                
                # Convert bytes to string if needed
                return {k: v if isinstance(v, str) else str(v) for k, v in tags.items()}
        except:
            pass
        
        return {}
    
    def _get_duration_ffprobe(self, file_path: str) -> float:
        """Get duration using ffprobe
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        return 0.0
    
    def _get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """Get audio stream info using ffprobe
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio properties
        """
        info = {"sample_rate": 44100, "channels": 2, "bitrate": 0}
        
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams", "-select_streams", "a:0",
                    file_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                if streams:
                    stream = streams[0]
                    info["sample_rate"] = int(stream.get("sample_rate", 44100))
                    info["channels"] = stream.get("channels", 2)
                    bitrate = stream.get("bit_rate")
                    if bitrate:
                        info["bitrate"] = int(bitrate)
        except:
            pass
        
        return info
    
    def _compute_checksum(self, path: Path) -> str:
        """Compute MD5 checksum of file
        
        Args:
            path: Path to file
            
        Returns:
            MD5 hex digest
        """
        hash_md5 = hashlib.md5()
        
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
        except:
            return ""
        
        return hash_md5.hexdigest()
    
    def _copy_to_library(self, source: Path, metadata: TrackMetadata) -> Path:
        """Copy file to tracks library
        
        Args:
            source: Source file path
            metadata: Track metadata
            
        Returns:
            Destination path
        """
        # Create filename with checksum to avoid duplicates
        ext = source.suffix
        safe_name = re.sub(r'[^\w\-.]', '_', metadata.title)
        dest_name = f"{safe_name}_{metadata.checksum[:8]}{ext}"
        
        dest_path = self.tracks_dir / dest_name
        
        # Handle duplicate names
        counter = 1
        while dest_path.exists():
            dest_name = f"{safe_name}_{metadata.checksum[:8]}_{counter}{ext}"
            dest_path = self.tracks_dir / dest_name
            counter += 1
        
        # Copy file
        import shutil
        shutil.copy2(source, dest_path)
        
        return dest_path
    
    def _download_file(self, url: str) -> str:
        """Download file from URL to temp location
        
        Args:
            url: URL to download
            
        Returns:
            Path to downloaded temp file
        """
        import urllib.request
        import tempfile
        
        # Get filename from URL
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        
        if not filename:
            filename = f"temp_{uuid.uuid4().hex[:8]}.mp3"
        
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix=Path(filename).suffix)
        os.close(temp_fd)
        
        # Download with progress
        try:
            urllib.request.urlretrieve(url, temp_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
        
        return temp_path
    
    def _analyze_track(self, metadata: TrackMetadata) -> Dict[str, Any]:
        """Analyze track for BPM, key, energy
        
        Args:
            metadata: Track metadata
            
        Returns:
            Analysis results dictionary
        """
        analysis = {
            "bpm": metadata.bpm,
            "key": metadata.key,
            "key_confidence": metadata.key_confidence,
            "energy": metadata.energy
        }
        
        # Load audio if possible
        try:
            import librosa
            
            y, sr = librosa.load(metadata.file_path, sr=22050)
            
            # Detect BPM if not in tags
            if metadata.bpm <= 0:
                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    analysis["bpm"] = float(tempo)
                except:
                    pass
            
            # Detect key if not in tags
            if not metadata.key:
                try:
                    key, confidence = detect_key(y, sr)
                    analysis["key"] = key
                    analysis["key_confidence"] = confidence
                except:
                    pass
            
            # Calculate energy from RMS
            try:
                rms = librosa.feature.rms(y=y)[0]
                # Normalize to 0-1 range
                energy = float(np.clip(np.mean(rms) * 10, 0, 1))
                analysis["energy"] = energy
            except:
                pass
                
        except ImportError:
            # librosa not available, use basic analysis
            pass
        except Exception as e:
            print(f"Analysis warning: {e}")
        
        return analysis
    
    def _detect_genre(self, metadata: TrackMetadata) -> str:
        """Auto-detect genre from metadata and analysis
        
        Args:
            metadata: Track metadata
            
        Returns:
            Detected genre string
        """
        # Simple genre detection based on metadata
        genre_tags = metadata.genre.lower()
        
        # Check common genre patterns
        genre_patterns = {
            "electronic": ["electronic", "edm", "house", "techno", "trance", "dubstep"],
            "rock": ["rock", "metal", "punk"],
            "pop": ["pop", "dance"],
            "hip-hop": ["hip-hop", "hip hop", "rap", "trap"],
            "classical": ["classical", "orchestra", "symphony"],
            "jazz": ["jazz", "blues"],
            "r&b": ["r&b", "soul", "funk"],
            "ambient": ["ambient", "chill", "downtempo"],
        }
        
        for genre, patterns in genre_patterns.items():
            for pattern in patterns:
                if pattern in genre_tags:
                    return genre
        
        # Default based on energy and BPM
        if metadata.bpm > 140:
            return "electronic"
        elif metadata.bpm < 80:
            return "ambient"
        
        return "pop"  # Default
    
    def _save_to_database(self, metadata: TrackMetadata) -> Optional[int]:
        """Save track to database
        
        Args:
            metadata: Track metadata
            
        Returns:
            Database ID if successful
        """
        if not self.db:
            return None
        
        try:
            # Create song record
            song = Song(
                name=metadata.title,
                prompt=f"Imported: {metadata.artist}",
                genre=metadata.genre,
                bpm=int(metadata.bpm) if metadata.bpm else 128,
                key=metadata.key or "C",
                duration=int(metadata.duration),
                energy=metadata.energy,
                mood="neutral",
                file_path=metadata.file_path,
                stem_paths="{}"
            )
            
            song_id = self.db.add_song(song)
            
            # Create analysis record
            if metadata.bpm or metadata.key:
                analysis = Analysis(
                    song_id=song_id,
                    file_path=metadata.file_path,
                    bpm=metadata.bpm,
                    key=metadata.key,
                    key_confidence=metadata.key_confidence,
                    energy=metadata.energy,
                    analyzed_at=datetime.now().isoformat()
                )
                self.db.add_analysis(analysis)
            
            return song_id
            
        except Exception as e:
            print(f"Database save error: {e}")
            return None
    
    def _metadata_to_dict(self, metadata: TrackMetadata) -> Dict[str, Any]:
        """Convert metadata to dictionary
        
        Args:
            metadata: TrackMetadata object
            
        Returns:
            Dictionary representation
        """
        return {
            "title": metadata.title,
            "artist": metadata.artist,
            "album": metadata.album,
            "genre": metadata.genre,
            "duration": metadata.duration,
            "bpm": metadata.bpm,
            "key": metadata.key,
            "energy": metadata.energy,
            "sample_rate": metadata.sample_rate,
            "channels": metadata.channels,
            "bitrate": metadata.bitrate,
            "format": metadata.format,
            "checksum": metadata.checksum
        }
    
    def get_track_count(self) -> int:
        """Get total number of imported tracks
        
        Returns:
            Track count
        """
        if not self.db:
            return 0
        
        try:
            return len(self.db.get_all_songs())
        except:
            return 0
    
    def list_tracks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List imported tracks
        
        Args:
            limit: Maximum number to return
            offset: Number to skip
            
        Returns:
            List of track dictionaries
        """
        if not self.db:
            return []
        
        try:
            songs = self.db.get_all_songs()
            tracks = []
            for song in songs[offset:offset+limit]:
                tracks.append({
                    "id": song.id,
                    "title": song.name,
                    "artist": song.prompt.replace("Imported: ", ""),
                    "genre": song.genre,
                    "bpm": song.bpm,
                    "key": song.key,
                    "duration": song.duration,
                    "energy": song.energy,
                    "file_path": song.file_path
                })
            return tracks
        except:
            return []


def import_track(file_path: str, **kwargs) -> ImportResult:
    """Convenience function to import a single track
    
    Args:
        file_path: Path to audio file
        **kwargs: Additional arguments for TrackImporter.import_file
        
    Returns:
        ImportResult
    """
    importer = TrackImporter()
    return importer.import_file(file_path, **kwargs)


def import_folder(folder_path: str, **kwargs) -> List[ImportResult]:
    """Convenience function to import all tracks from a folder
    
    Args:
        folder_path: Path to folder
        **kwargs: Additional arguments for TrackImporter.import_directory
        
    Returns:
        List of ImportResult
    """
    importer = TrackImporter()
    return importer.import_directory(folder_path, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Import tracks to AI DJ")
    parser.add_argument("path", help="File, folder, or URL to import")
    parser.add_argument("--no-copy", action="store_true", help="Don't copy to library")
    parser.add_argument("--analyze", action="store_true", help="Analyze BPM/key/energy")
    parser.add_argument("--recursive", action="store_true", help="Import recursively")
    
    args = parser.parse_args()
    
    importer = TrackImporter()
    
    if os.path.isfile(args.path):
        result = importer.import_file(
            args.path,
            copy_to_library=not args.no_copy,
            analyze=args.analyze
        )
        print(f"Imported: {result.success}")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Track ID: {result.track_id}")
            
    elif os.path.isdir(args.path):
        results = importer.import_directory(
            args.path,
            recursive=args.recursive,
            copy_to_library=not args.no_copy,
            analyze=args.analyze
        )
        success = sum(1 for r in results if r.success)
        print(f"Imported {success}/{len(results)} tracks")
        
    elif args.path.startswith("http"):
        result = importer.import_url(args.path, analyze=args.analyze)
        print(f"Imported: {result.success}")
        if result.error:
            print(f"Error: {result.error}")
    else:
        print(f"Invalid path: {args.path}")
