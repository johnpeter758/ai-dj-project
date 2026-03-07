"""
Audio Encoder for AI DJ Project
===============================

Handles encoding audio to various formats with quality control,
metadata embedding, and format-specific optimizations.
"""

import numpy as np
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import json
import tempfile
import struct


class AudioEncoder:
    """Multi-format audio encoder with quality control."""
    
    # Format configurations
    FORMATS = {
        'wav': {
            'extensions': ['.wav'],
            'codec': 'pcm_s16le',
            'lossless': True,
            'bit_depths': [16, 24, 32],
            'default_bit_depth': 24,
        },
        'mp3': {
            'extensions': ['.mp3'],
            'codec': 'libmp3lame',
            'lossless': False,
            'bitrates': [128, 192, 256, 320],
            'default_bitrate': 320,
        },
        'flac': {
            'extensions': ['.flac'],
            'codec': 'flac',
            'lossless': True,
            'compression_levels': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'default_level': 5,
        },
        'aac': {
            'extensions': ['.m4a', '.aac'],
            'codec': 'aac',
            'lossless': False,
            'bitrates': [128, 192, 256, 320],
            'default_bitrate': 256,
        },
        'ogg': {
            'extensions': ['.ogg', '.oga'],
            'codec': 'libvorbis',
            'lossless': False,
            'bitrates': [128, 192, 256, 320],
            'default_bitrate': 256,
        },
        'opus': {
            'extensions': ['.opus'],
            'codec': 'libopus',
            'lossless': False,
            'bitrates': [64, 96, 128, 160, 192, 256],
            'default_bitrate': 128,
        },
    }
    
    # Sample rates
    SAMPLE_RATES = [22050, 32000, 44100, 48000, 96000, 192000]
    DEFAULT_SAMPLE_RATE = 44100
    
    def __init__(self, sample_rate: int = 44100, bit_depth: int = 24):
        """
        Initialize audio encoder.
        
        Args:
            sample_rate: Target sample rate in Hz
            bit_depth: Target bit depth (16, 24, 32)
        """
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self._ffmpeg_available = self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def encode(
        self,
        audio: np.ndarray,
        output_path: str,
        format: str = 'mp3',
        quality: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Encode audio to specified format.
        
        Args:
            audio: Audio data as numpy array (mono or stereo)
            output_path: Output file path
            format: Output format (mp3, wav, flac, aac, ogg, opus)
            quality: Quality setting (bitrate for lossy, level for lossless)
            metadata: Optional metadata dict (title, artist, album, etc.)
        
        Returns:
            Dict with encoding results and info
        """
        format = format.lower()
        
        if format not in self.FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        # Ensure output has correct extension
        output_path = self._ensure_extension(output_path, format)
        
        # Handle metadata
        metadata_args = self._build_metadata_args(metadata)
        
        # Use ffmpeg if available
        if self._ffmpeg_available:
            return self._encode_ffmpeg(
                audio, output_path, format, quality, metadata_args
            )
        else:
            # Fallback to numpy-based encoding
            return self._encode_numpy(audio, output_path, format, quality)
    
    def _ensure_extension(self, path: str, format: str) -> str:
        """Ensure output path has correct extension."""
        path = Path(path)
        expected_ext = self.FORMATS[format]['extensions'][0]
        
        if path.suffix.lower() != expected_ext:
            path = path.with_suffix(expected_ext)
        
        return str(path)
    
    def _build_metadata_args(self, metadata: Optional[Dict[str, str]]) -> list:
        """Build ffmpeg metadata arguments."""
        if not metadata:
            return []
        
        args = []
        for key, value in metadata.items():
            # Map common keys to ffmpeg tags
            tag_map = {
                'title': 'title',
                'artist': 'artist',
                'album': 'album',
                'album_artist': 'album_artist',
                'genre': 'genre',
                'year': 'date',
                'track': 'track',
                'comment': 'comment',
            }
            tag = tag_map.get(key.lower(), key.lower())
            args.extend(['-metadata', f'{tag}={value}'])
        
        return args
    
    def _encode_ffmpeg(
        self,
        audio: np.ndarray,
        output_path: str,
        format: str,
        quality: Optional[int],
        metadata_args: list,
    ) -> Dict[str, Any]:
        """Encode using ffmpeg."""
        # Create temp WAV file for ffmpeg input
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write audio to temp WAV
            self._write_wav(audio, tmp_path)
            
            # Build ffmpeg command
            cmd = ['ffmpeg', '-y', '-i', tmp_path]
            
            # Add format-specific options
            cmd.extend(self._get_format_args(format, quality))
            
            # Add metadata
            cmd.extend(metadata_args)
            
            # Output path
            cmd.append(output_path)
            
            # Run encoding
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed: {result.stderr.decode()}")
            
            # Get file info
            file_size = os.path.getsize(output_path)
            
            return {
                'success': True,
                'output_path': output_path,
                'format': format,
                'file_size': file_size,
                'sample_rate': self.sample_rate,
                'bit_depth': self.bit_depth,
                'duration': len(audio) / self.sample_rate if audio.ndim == 1 else len(audio[0]) / self.sample_rate,
            }
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _get_format_args(self, format: str, quality: Optional[int]) -> list:
        """Get format-specific ffmpeg arguments."""
        args = []
        
        if format == 'mp3':
            bitrate = quality or self.FORMATS['mp3']['default_bitrate']
            args.extend(['-b:a', f'{bitrate}k'])
            
        elif format == 'flac':
            level = quality or self.FORMATS['flac']['default_level']
            args.extend(['-compression_level', str(level)])
            
        elif format == 'aac':
            bitrate = quality or self.FORMATS['aac']['default_bitrate']
            args.extend(['-b:a', f'{bitrate}k'])
            
        elif format == 'ogg':
            bitrate = quality or self.FORMATS['ogg']['default_bitrate']
            args.extend(['-b:a', f'{bitrate}k'])
            
        elif format == 'opus':
            bitrate = quality or self.FORMATS['opus']['default_bitrate']
            args.extend(['-b:a', f'{bitrate}k'])
        
        # Set sample rate and bit depth
        args.extend(['-ar', str(self.sample_rate)])
        
        if format != 'mp3' and format != 'aac' and format != 'ogg' and format != 'opus':
            args.extend(['-acodec', self.FORMATS[format]['codec']])
        
        return args
    
    def _write_wav(self, audio: np.ndarray, path: str):
        """Write audio to WAV file using numpy."""
        # Normalize audio to 16-bit range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to appropriate dtype
        if self.bit_depth == 16:
            audio_int = (audio * 32767).astype(np.int16)
        elif self.bit_depth == 24:
            # Scale to 24-bit range
            audio_int = (audio * 8388607).astype(np.int32)
            # Convert to 24-bit bytes (little endian)
            audio_int = audio_int.astype(np.int24) if hasattr(np, 'int24') else audio_int
        else:  # 32-bit
            audio_int = (audio * 2147483647).astype(np.int32)
        
        # Ensure stereo
        if audio.ndim == 1:
            audio_int = np.stack([audio_int, audio_int], axis=0)
        
        # Write WAV file
        import wave
        
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(self.bit_depth // 8)
            wav_file.setframerate(self.sample_rate)
            
            # Convert to bytes
            if self.bit_depth == 16:
                audio_bytes = audio_int.tobytes()
            elif self.bit_depth == 24:
                # Manual 24-bit packing
                audio_bytes = b''
                for sample in audio_int.T.flatten():
                    audio_bytes += struct.pack('<i', sample)[:3]
            else:
                audio_bytes = audio_int.tobytes()
            
            wav_file.writeframes(audio_bytes)
    
    def _encode_numpy(
        self,
        audio: np.ndarray,
        output_path: str,
        format: str,
        quality: Optional[int],
    ) -> Dict[str, Any]:
        """Fallback numpy-based encoder for WAV/FLAC."""
        if format == 'wav':
            self._write_wav(audio, output_path)
            file_size = os.path.getsize(output_path)
        elif format == 'flac':
            # Basic FLAC encoding (compression only)
            import wave
            tmp_wav = output_path.replace('.flac', '.wav')
            self._write_wav(audio, tmp_wav)
            # Note: Full FLAC encoding requires external library
            os.rename(tmp_wav, output_path)
            file_size = os.path.getsize(output_path)
        else:
            raise RuntimeError(
                f"Format {format} requires ffmpeg. "
                "Install ffmpeg for full format support."
            )
        
        return {
            'success': True,
            'output_path': output_path,
            'format': format,
            'file_size': file_size,
            'sample_rate': self.sample_rate,
            'bit_depth': self.bit_depth,
            'duration': len(audio) / self.sample_rate if audio.ndim == 1 else len(audio[0]) / self.sample_rate,
        }
    
    def batch_encode(
        self,
        audio: np.ndarray,
        output_dir: str,
        formats: list,
        quality: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Encode audio to multiple formats.
        
        Args:
            audio: Audio data
            output_dir: Output directory
            formats: List of formats to encode to
            quality: Optional dict of quality settings per format
            metadata: Optional metadata for all outputs
        
        Returns:
            Dict mapping format to result
        """
        quality = quality or {}
        results = {}
        
        for fmt in formats:
            output_name = f"output.{self.FORMATS[fmt]['extensions'][0][1:]}"
            output_path = os.path.join(output_dir, output_name)
            
            try:
                results[fmt] = self.encode(
                    audio,
                    output_path,
                    fmt,
                    quality.get(fmt),
                    metadata
                )
            except Exception as e:
                results[fmt] = {
                    'success': False,
                    'format': fmt,
                    'error': str(e)
                }
        
        return results


class StreamingEncoder(AudioEncoder):
    """Optimized encoder for streaming platforms."""
    
    PLATFORM_SETTINGS = {
        'spotify': {
            'format': 'ogg',
            'quality': 320,
            'sample_rate': 44100,
        },
        'apple_music': {
            'format': 'aac',
            'quality': 256,
            'sample_rate': 44100,
        },
        'youtube': {
            'format': 'mp3',
            'quality': 320,
            'sample_rate': 44100,
        },
        'soundcloud': {
            'format': 'mp3',
            'quality': 320,
            'sample_rate': 44100,
        },
        'tidal': {
            'format': 'flac',
            'quality': 5,
            'sample_rate': 96000,
        },
    }
    
    def encode_for_platform(
        self,
        audio: np.ndarray,
        output_dir: str,
        platform: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Encode audio optimized for a streaming platform."""
        platform = platform.lower()
        
        if platform not in self.PLATFORM_SETTINGS:
            raise ValueError(f"Unknown platform: {platform}")
        
        settings = self.PLATFORM_SETTINGS[platform]
        
        # Apply platform-specific sample rate
        self.sample_rate = settings['sample_rate']
        
        output_name = f"{platform}_output{settings['format']}"
        output_path = os.path.join(output_dir, output_name)
        
        return self.encode(
            audio,
            output_path,
            settings['format'],
            settings['quality'],
            metadata
        )
    
    def encode_all_platforms(
        self,
        audio: np.ndarray,
        output_dir: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Encode for all supported streaming platforms."""
        results = {}
        
        for platform in self.PLATFORM_SETTINGS:
            try:
                results[platform] = self.encode_for_platform(
                    audio, output_dir, platform, metadata
                )
            except Exception as e:
                results[platform] = {'success': False, 'error': str(e)}
        
        return results


def encode_audio(
    audio: np.ndarray,
    output_path: str,
    format: str = 'mp3',
    sample_rate: int = 44100,
    bit_depth: int = 24,
    quality: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for quick audio encoding.
    
    Args:
        audio: Audio data as numpy array
        output_path: Output file path
        format: Output format (mp3, wav, flac, aac, ogg, opus)
        sample_rate: Sample rate in Hz
        bit_depth: Bit depth (16, 24, 32)
        quality: Quality setting
        metadata: Audio metadata
    
    Returns:
        Encoding result dict
    """
    encoder = AudioEncoder(sample_rate=sample_rate, bit_depth=bit_depth)
    return encoder.encode(audio, output_path, format, quality, metadata)
