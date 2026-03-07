#!/usr/bin/env python3
"""
AI DJ Project - Common Utilities Library

Provides shared utilities for audio processing, file handling, and common operations
across the AI DJ project.
"""

import os
import json
import hashlib
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Configure logging
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    return logger


# Path utilities
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_path(filename: str, suffix: str = "") -> Path:
    """Get a timestamped output path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = Path(filename).stem
    ext = Path(filename).suffix
    if suffix:
        name = f"{name}_{suffix}"
    return OUTPUT_DIR / f"{name}_{timestamp}{ext}"


# Audio file utilities
def load_audio(path: Union[str, Path], sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio file using librosa (with fallback to soundfile)."""
    try:
        import librosa
        y, sr = librosa.load(path, sr=sr)
        return y, sr
    except ImportError:
        import soundfile as sf
        y, sr = sf.read(path)
        if sr is not None and sr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=sr)
        return y, sr


def save_audio(path: Union[str, Path], y: np.ndarray, sr: int, normalize: bool = True):
    """Save audio file with optional normalization."""
    if normalize:
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    path = Path(path)
    ensure_dir(path.parent)
    try:
        import soundfile as sf
        sf.write(path, y, sr)
    except ImportError:
        import librosa
        librosa.output.write_wav(path, y, sr)


def get_duration(path: Union[str, Path]) -> float:
    """Get audio duration in seconds."""
    try:
        import librosa
        return librosa.get_duration(filename=path)
    except ImportError:
        import soundfile as sf
        info = sf.info(path)
        return info.duration


# File hashing utilities
def hash_file(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_string(s: str, algorithm: str = "sha256") -> str:
    """Compute hash of a string."""
    return hashlib.new(algorithm, s.encode()).hexdigest()


# JSON utilities
def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: Union[str, Path], data: Dict[str, Any], indent: int = 2):
    """Save data to JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


# Configuration utilities
@dataclass
class Config:
    """Base configuration class."""
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON file."""
        data = load_json(path)
        return cls(**data)
    
    def to_json(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        save_json(path, asdict(self))


# Progress tracking
@dataclass
class Progress:
    """Simple progress tracker."""
    current: int = 0
    total: int = 100
    message: str = ""
    
    @property
    def percent(self) -> float:
        return (self.current / self.total * 100) if self.total > 0 else 0
    
    def increment(self, n: int = 1):
        self.current = min(self.current + n, self.total)
    
    def __str__(self) -> str:
        return f"{self.message}: {self.current}/{self.total} ({self.percent:.1f}%)"


# Time utilities
def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_timestamp(ts: str) -> datetime:
    """Parse timestamp string to datetime."""
    return datetime.strptime(ts, "%Y%m%d_%H%M%S")


# Audio processing helpers
def db_to_amplitude(db: float) -> float:
    """Convert decibels to amplitude."""
    return 10 ** (db / 20)


def amplitude_to_db(amplitude: float) -> float:
    """Convert amplitude to decibels."""
    return 20 * np.log10(amplitude + 1e-10)


def normalize_audio(y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio to target dB."""
    peak = np.max(np.abs(y))
    if peak > 0:
        target_amp = db_to_amplitude(target_db)
        y = y * (target_amp / peak)
    return y


def fade_in_out(y: np.ndarray, sr: int, fade_len: float = 0.1) -> np.ndarray:
    """Apply fade in and fade out to audio."""
    fade_samples = int(fade_len * sr)
    if fade_samples * 2 > len(y):
        return y
    
    # Fade in
    fade_in = np.linspace(0, 1, fade_samples)
    y[:fade_samples] *= fade_in
    
    # Fade out
    fade_out = np.linspace(1, 0, fade_samples)
    y[-fade_samples:] *= fade_out
    
    return y


def mix_audio(a: np.ndarray, b: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """Mix two audio arrays with a ratio (0 = all a, 1 = all b)."""
    if len(a) != len(b):
        # Pad shorter one
        max_len = max(len(a), len(b))
        a = np.pad(a, (0, max_len - len(a)))
        b = np.pad(b, (0, max_len - len(b)))
    return (1 - ratio) * a + ratio * b


# Clip utilities
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * clamp(t, 0, 1)


def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map value from one range to another."""
    return out_min + (out_max - out_min) * ((value - in_min) / (in_max - in_min))


# Enum utilities
class LabeledEnum(Enum):
    """Enum with human-readable labels."""
    
    @property
    def label(self) -> str:
        return self.name.replace('_', ' ').title()


# Logging utilities
def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper


# Temp file utilities
def temp_file(suffix: str = "", dir: Optional[Path] = None) -> Path:
    """Create a temporary file."""
    fd, path = tempfile.mkstemp(suffix=suffix, dir=dir)
    os.close(fd)
    return Path(path)


def temp_dir(prefix: str = "aidj_") -> Path:
    """Create a temporary directory."""
    return Path(tempfile.mkdtemp(prefix=prefix))


# Process utilities
def run_command(cmd: List[str], capture: bool = True, timeout: int = 300) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


# Validation utilities
def is_audio_file(path: Union[str, Path]) -> bool:
    """Check if file is an audio file."""
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}
    return Path(path).suffix.lower() in audio_exts


def validate_audio_path(path: Union[str, Path]) -> Path:
    """Validate and return audio path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not is_audio_file(path):
        raise ValueError(f"Not an audio file: {path}")
    return path


# Cache utilities
class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self._ttl):
                return value
            del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self._cache[key] = (value, datetime.now())
    
    def clear(self):
        self._cache.clear()


# Singleton decorator
def singleton(cls):
    """Singleton decorator."""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


# Batch processing
def batch_process(items: List[Any], batch_size: int = 32) -> List[List[Any]]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# Export common items
__all__ = [
    'setup_logger',
    'PROJECT_ROOT',
    'OUTPUT_DIR', 
    'CACHE_DIR',
    'RECORDINGS_DIR',
    'ensure_dir',
    'get_output_path',
    'load_audio',
    'save_audio',
    'get_duration',
    'hash_file',
    'hash_string',
    'load_json',
    'save_json',
    'Config',
    'Progress',
    'format_duration',
    'get_timestamp',
    'parse_timestamp',
    'db_to_amplitude',
    'amplitude_to_db',
    'normalize_audio',
    'fade_in_out',
    'mix_audio',
    'clamp',
    'lerp',
    'map_range',
    'LabeledEnum',
    'log_execution_time',
    'temp_file',
    'temp_dir',
    'run_command',
    'is_audio_file',
    'validate_audio_path',
    'SimpleCache',
    'singleton',
    'batch_process',
]
