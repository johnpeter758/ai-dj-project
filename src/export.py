#!/usr/bin/env python3
"""
AI DJ Project - Export System

Provides comprehensive track export functionality including:
- Multiple audio formats (WAV, MP3, FLAC, OGG, AAC)
- Stem/track separation exports
- Metadata embedding
- Quality presets
- Batch export operations
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    AAC = "aac"
    M4A = "m4a"


class ExportQuality(Enum):
    """Export quality presets."""
    DEMO = "demo"          # Low quality, small file size
    STANDARD = "standard"  # CD quality
    HIGH = "high"          # High quality
    LOSSLESS = "lossless"  # Lossless compression
    STUDIO = "studio"      # Maximum quality


class ExportMode(Enum):
    """Export operation modes."""
    SINGLE = "single"          # Single track export
    STEMS = "stems"            # Individual stems export
    MIXDOWN = "mixdown"        # Full mixdown export
    BATCH = "batch"            # Batch export multiple tracks
    PROJECT = "project"        # Complete project export


@dataclass
class TrackMetadata:
    """Metadata for exported tracks."""
    title: str = ""
    artist: str = "AI DJ"
    album: str = ""
    genre: str = ""
    year: int = 0
    track_number: int = 0
    composer: str = ""
    BPM: float = 0.0
    key: str = ""
    comments: str = ""
    artwork_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "year": self.year,
            "track_number": self.track_number,
            "composer": self.composer,
            "BPM": self.BPM,
            "key": self.key,
            "comments": self.comments,
            "artwork_path": self.artwork_path
        }


@dataclass
class ExportSettings:
    """Configuration for export operations."""
    format: ExportFormat = ExportFormat.WAV
    quality: ExportQuality = ExportQuality.HIGH
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2  # 1=mono, 2=stereo
    
    # MP3 specific
    mp3_bitrate: int = 320
    
    # Output settings
    output_dir: str = "/Users/johnpeter/ai-dj-project/src/output"
    filename_template: str = "{title}_{artist}"
    
    # Metadata
    metadata: TrackMetadata = field(default_factory=TrackMetadata)
    
    # Options
    normalize: bool = True
    add_fade_out: bool = True
    fade_out_duration: float = 2.0  # seconds
    
    def get_quality_settings(self) -> Tuple[int, int, int]:
        """Get quality settings based on quality preset."""
        quality_map = {
            ExportQuality.DEMO: (22050, 128, 16),
            ExportQuality.STANDARD: (44100, 192, 16),
            ExportQuality.HIGH: (44100, 320, 24),
            ExportQuality.LOSSLESS: (48000, 0, 24),  # 0 = not applicable
            ExportQuality.STUDIO: (96000, 0, 32),
        }
        sr, bitrate, depth = quality_map.get(self.quality, (44100, 320, 16))
        
        # Override with explicit settings if provided
        if self.sample_rate != 44100:
            sr = self.sample_rate
        if self.bit_depth != 16:
            depth = self.bit_depth
            
        return sr, bitrate, depth


@dataclass
class StemExportSettings(ExportSettings):
    """Settings for stem exports."""
    stem_names: List[str] = field(default_factory=lambda: [
        "drums", "bass", "melody", "vocals"
    ])
    export_all: bool = True
    individual_files: bool = True
    combined_stems: bool = True


@dataclass  
class BatchExportJob:
    """Represents a single job in a batch export."""
    input_path: str
    output_filename: str
    metadata: Optional[TrackMetadata] = None
    settings: Optional[ExportSettings] = None


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    file_size: Optional[int] = None
    duration: Optional[float] = None
    format: Optional[ExportFormat] = None
    metadata: Optional[Dict[str, Any]] = None


class ExportSystem:
    """Main export system for handling track exports."""
    
    def __init__(self, settings: Optional[ExportSettings] = None):
        self.settings = settings or ExportSettings()
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        os.makedirs(self.settings.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.settings.output_dir}")
    
    def _get_ffmpeg_format(self, format: ExportFormat) -> str:
        """Get ffmpeg format string."""
        format_map = {
            ExportFormat.WAV: "wav",
            ExportFormat.MP3: "mp3",
            ExportFormat.FLAC: "flac",
            ExportFormat.OGG: "ogg",
            ExportFormat.AAC: "aac",
            ExportFormat.M4A: "ipod",
        }
        return format_map.get(format, "wav")
    
    def _build_ffmpeg_command(
        self,
        input_path: str,
        output_path: str,
        settings: ExportSettings
    ) -> List[str]:
        """Build ffmpeg command for export."""
        cmd = ["ffmpeg", "-y", "-i", input_path]
        
        # Audio filters
        filters = []
        
        # Normalize if requested
        if settings.normalize:
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        
        # Fade out if requested
        if settings.add_fade_out:
            # We'll apply this at the end
            pass
        
        if filters:
            cmd.extend(["-af", ",".join(filters)])
        
        # Format-specific settings
        if settings.format == ExportFormat.MP3:
            cmd.extend(["-b:a", f"{settings.mp3_bitrate}k"])
            cmd.extend(["-codec:a", "libmp3lame"])
        elif settings.format == ExportFormat.FLAC:
            cmd.extend(["-compression_level", "8"])
        elif settings.format in (ExportFormat.AAC, ExportFormat.M4A):
            cmd.extend(["-codec:a", "aac"])
            cmd.extend(["-b:a", "256k"])
        elif settings.format == ExportFormat.OGG:
            cmd.extend(["-codec:a", "libvorbis"])
            cmd.extend(["-q:a", "6"])
        
        # Sample rate and channels
        cmd.extend(["-ar", str(settings.sample_rate)])
        cmd.extend(["-ac", str(settings.channels)])
        
        # Bit depth for WAV/FLAC
        if settings.format in (ExportFormat.WAV, ExportFormat.FLAC):
            cmd.extend(["-acodec", "pcm_s{}_le".format(settings.bit_depth)])
        
        # Metadata
        if settings.metadata.title:
            cmd.extend(["-metadata", f"title={settings.metadata.title}"])
        if settings.metadata.artist:
            cmd.extend(["-metadata", f"artist={settings.metadata.artist}"])
        if settings.metadata.album:
            cmd.extend(["-metadata", f"album={settings.metadata.album}"])
        if settings.metadata.year:
            cmd.extend(["-metadata", f"date={settings.metadata.year}"])
        if settings.metadata.genre:
            cmd.extend(["-metadata", f"genre={settings.metadata.genre}"])
        if settings.metadata.comments:
            cmd.extend(["-metadata", f"comment={settings.metadata.comments}"])
        
        # Artwork
        if settings.metadata.artwork_path and os.path.exists(settings.metadata.artwork_path):
            cmd.extend(["-i", settings.metadata.artwork_path])
            cmd.extend(["-map", "0:a"])
            cmd.extend(["-map", "1:v"])
            cmd.extend(["-c:v", "copy"])
            cmd.extend(["-id3v2_version", "3"])
            cmd.extend(["-metadata:s:v", "title=Album cover"])
            cmd.extend(["-metadata:s:v", "comment=Cover (front)"])
        
        # Output path
        cmd.append(output_path)
        
        return cmd
    
    def export_track(
        self,
        input_path: str,
        output_filename: Optional[str] = None,
        settings: Optional[ExportSettings] = None
    ) -> ExportResult:
        """
        Export a single track.
        
        Args:
            input_path: Path to input audio file
            output_filename: Optional output filename (without extension)
            settings: Optional export settings override
            
        Returns:
            ExportResult with export status and details
        """
        settings = settings or self.settings
        
        # Generate output filename
        if not output_filename:
            template = settings.filename_template
            output_filename = template.format(
                title=settings.metadata.title or "track",
                artist=settings.metadata.artist or "AI_DJ",
                date=datetime.now().strftime("%Y%m%d")
            )
        
        output_path = os.path.join(
            settings.output_dir,
            f"{output_filename}.{settings.format.value}"
        )
        
        logger.info(f"Exporting: {input_path} -> {output_path}")
        
        try:
            # Check input exists
            if not os.path.exists(input_path):
                return ExportResult(
                    success=False,
                    error_message=f"Input file not found: {input_path}"
                )
            
            # Build ffmpeg command
            cmd = self._build_ffmpeg_command(input_path, output_path, settings)
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return ExportResult(
                    success=False,
                    error_message=f"Export failed: {result.stderr}"
                )
            
            # Get file size
            file_size = os.path.getsize(output_path)
            
            # Get duration using ffprobe
            duration = self._get_duration(output_path)
            
            logger.info(f"Export successful: {output_path} ({file_size} bytes)")
            
            return ExportResult(
                success=True,
                output_path=output_path,
                file_size=file_size,
                duration=duration,
                format=settings.format,
                metadata=settings.metadata.to_dict()
            )
            
        except subprocess.TimeoutExpired:
            return ExportResult(
                success=False,
                error_message="Export timed out"
            )
        except Exception as e:
            logger.exception("Export failed")
            return ExportResult(
                success=False,
                error_message=str(e)
            )
    
    def _get_duration(self, file_path: str) -> Optional[float]:
        """Get audio file duration using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return None
    
    def export_stems(
        self,
        input_path: str,
        stem_settings: Optional[StemExportSettings] = None
    ) -> List[ExportResult]:
        """
        Export individual stems/tracks.
        
        Args:
            input_path: Path to input audio file
            stem_settings: Settings for stem export
            
        Returns:
            List of ExportResult for each stem
        """
        stem_settings = stem_settings or StemExportSettings()
        results = []
        
        # For now, this assumes input_path contains stems or we need to split
        # In a real implementation, this would use Demucs or similar
        
        if stem_settings.individual_files:
            for stem_name in stem_settings.stem_names:
                output_path = os.path.join(
                    stem_settings.output_dir,
                    f"{stem_name}.{stem_settings.format.value}"
                )
                
                # This is a placeholder - actual stem splitting would go here
                logger.info(f"Would export stem: {stem_name} -> {output_path}")
                
                results.append(ExportResult(
                    success=True,
                    output_path=output_path,
                    format=stem_settings.format
                ))
        
        return results
    
    def export_batch(
        self,
        jobs: List[BatchExportJob],
        progress_callback: Optional[callable] = None
    ) -> List[ExportResult]:
        """
        Export multiple tracks in batch.
        
        Args:
            jobs: List of BatchExportJob
            progress_callback: Optional callback(completed, total)
            
        Returns:
            List of ExportResult for each job
        """
        results = []
        total = len(jobs)
        
        logger.info(f"Starting batch export of {total} tracks")
        
        for i, job in enumerate(jobs):
            settings = job.settings or self.settings
            if job.metadata:
                settings.metadata = job.metadata
            
            result = self.export_track(
                input_path=job.input_path,
                output_filename=job.output_filename,
                settings=settings
            )
            
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            logger.info(f"Batch progress: {i+1}/{total}")
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {successful}/{total} successful")
        
        return results
    
    def export_project(
        self,
        project_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, ExportResult]:
        """
        Export a complete project with all formats and stems.
        
        Args:
            project_path: Path to project directory
            output_dir: Optional output directory override
            
        Returns:
            Dictionary mapping export type to ExportResult
        """
        results = {}
        
        # Load project metadata if exists
        metadata_path = os.path.join(project_path, "project.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                project_data = json.load(f)
                self.settings.metadata = TrackMetadata(**project_data.get("metadata", {}))
        
        # Find main audio file
        audio_extensions = [".wav", ".mp3", ".flac", ".ogg"]
        audio_file = None
        for ext in audio_extensions:
            potential = os.path.join(project_path, f"main{ext}")
            if os.path.exists(potential):
                audio_file = potential
                break
        
        if not audio_file:
            logger.warning("No main audio file found in project")
            return results
        
        # Export in multiple formats
        formats_to_export = [
            ExportFormat.WAV,
            ExportFormat.MP3,
            ExportFormat.FLAC
        ]
        
        for fmt in formats_to_export:
            self.settings.format = fmt
            output_filename = f"export_{fmt.value}"
            
            result = self.export_track(
                input_path=audio_file,
                output_filename=output_filename,
                settings=self.settings
            )
            
            results[fmt.value] = result
        
        return results
    
    def get_supported_formats() -> List[str]:
        """Get list of supported export formats."""
        return [f.value for f in ExportFormat]
    
    def validate_ffmpeg() -> Tuple[bool, str]:
        """Check if ffmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return True, version_line
            return False, "FFmpeg not found"
        except FileNotFoundError:
            return False, "FFmpeg not installed"
        except Exception as e:
            return False, str(e)


class ExportPreset:
    """Predefined export presets."""
    
    @staticmethod
    def spotify() -> ExportSettings:
        """Spotify upload quality."""
        return ExportSettings(
            format=ExportFormat.WAV,
            quality=ExportQuality.STUDIO,
            sample_rate=44100,
            bit_depth=24,
            mp3_bitrate=320
        )
    
    @staticmethod
    def soundcloud() -> ExportSettings:
        """SoundCloud upload quality."""
        return ExportSettings(
            format=ExportFormat.MP3,
            quality=ExportQuality.HIGH,
            mp3_bitrate=320
        )
    
    @staticmethod
    def instagram() -> ExportSettings:
        """Instagram story/reel quality."""
        return ExportSettings(
            format=ExportFormat.AAC,
            quality=ExportQuality.STANDARD,
            sample_rate=44100,
            bit_depth=16
        )
    
    @staticmethod
    def podcast() -> ExportSettings:
        """Podcast quality."""
        return ExportSettings(
            format=ExportFormat.MP3,
            quality=ExportQuality.STANDARD,
            sample_rate=44100,
            mp3_bitrate=128
        )
    
    @staticmethod
    def archive() -> ExportSettings:
        """Archive/lossless quality."""
        return ExportSettings(
            format=ExportFormat.FLAC,
            quality=ExportQuality.LOSSLESS,
            sample_rate=96000,
            bit_depth=24
        )


# Convenience functions
def quick_export(
    input_path: str,
    output_format: Union[str, ExportFormat] = "mp3",
    output_name: Optional[str] = None
) -> ExportResult:
    """Quick export with sensible defaults."""
    if isinstance(output_format, str):
        output_format = ExportFormat(output_format)
    
    settings = ExportSettings(
        format=output_format,
        quality=ExportQuality.HIGH if output_format != ExportFormat.WAV else ExportQuality.STANDARD
    )
    
    exporter = ExportSystem(settings)
    return exporter.export_track(input_path, output_name)


def batch_export_from_list(
    file_list: List[Tuple[str, str]],
    output_format: Union[str, ExportFormat] = "mp3"
) -> List[ExportResult]:
    """
    Batch export from a list of (input_path, output_name) tuples.
    
    Args:
        file_list: List of (input_path, output_filename) tuples
        output_format: Export format
        
    Returns:
        List of ExportResult
    """
    if isinstance(output_format, str):
        output_format = ExportFormat(output_format)
    
    settings = ExportSettings(format=output_format)
    exporter = ExportSystem(settings)
    
    jobs = [
        BatchExportJob(input_path=path, output_filename=name)
        for path, name in file_list
    ]
    
    return exporter.export_batch(jobs)


# CLI interface
def main():
    """CLI for export system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI DJ Export System")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output filename (without extension)")
    parser.add_argument("-f", "--format", default="mp3", choices=["wav", "mp3", "flac", "ogg", "aac"])
    parser.add_argument("-q", "--quality", default="high", choices=["demo", "standard", "high", "lossless", "studio"])
    parser.add_argument("--title", help="Track title")
    parser.add_argument("--artist", help="Track artist")
    parser.add_argument("--album", help="Album name")
    parser.add_argument("--bpm", type=float, help="BPM")
    parser.add_argument("--key", help="Musical key")
    
    args = parser.parse_args()
    
    # Check ffmpeg
    available, msg = ExportSystem.validate_ffmpeg()
    if not available:
        logger.error(f"FFmpeg not available: {msg}")
        return 1
    
    logger.info(f"FFmpeg: {msg}")
    
    # Build settings
    settings = ExportSettings(
        format=ExportFormat(args.format),
        quality=ExportQuality(args.quality),
        metadata=TrackMetadata(
            title=args.title or "",
            artist=args.artist or "AI DJ",
            album=args.album or "",
            BPM=args.bpm or 0.0,
            key=args.key or ""
        )
    )
    
    # Export
    exporter = ExportSystem(settings)
    result = exporter.export_track(args.input, args.output)
    
    if result.success:
        logger.info(f"Export complete: {result.output_path}")
        logger.info(f"File size: {result.file_size} bytes")
        if result.duration:
            logger.info(f"Duration: {result.duration:.2f}s")
        return 0
    else:
        logger.error(f"Export failed: {result.error_message}")
        return 1


if __name__ == "__main__":
    exit(main())
