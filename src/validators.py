"""
Input validation module for AI DJ Project.

Provides validation functions for all input types including:
- Audio parameters (BPM, sample rate, etc.)
- File paths and formats
- Genre/mood classifications
- Audio processing parameters
- Configuration values
"""

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from constants import (
    BIT_DEPTH_16, BIT_DEPTH_24, BIT_DEPTH_32, BIT_DEPTH_DEFAULT,
    BPM_MAX, BPM_MIN, BUFFER_SIZE_LARGE, BUFFER_SIZE_MEDIUM, BUFFER_SIZE_SMALL,
    CHANNELS_DEFAULT, FORMAT_AAC, FORMAT_FLAC, FORMAT_MP3, FORMAT_OGG, FORMAT_WAV,
    MONO, SAMPLE_RATE_44K, SAMPLE_RATE_48K, SAMPLE_RATE_DEFAULT, STEREO,
    STEM_TYPES, BPM_RANGES
)
from exceptions import (
    ConfigurationException, InvalidFileFormatException
)


# =============================================================================
# AUDIO PARAMETER VALIDATORS
# =============================================================================

def validate_bpm(bpm: Any, min_bpm: int = BPM_MIN, max_bpm: int = BPM_MAX) -> float:
    """
    Validate BPM value.
    
    Args:
        bpm: BPM value to validate
        min_bpm: Minimum allowed BPM (default: 60)
        max_bpm: Maximum allowed BPM (default: 200)
    
    Returns:
        Validated BPM as float
    
    Raises:
        TypeError: If bpm is not a number
        ValueError: If bpm is outside valid range
    """
    if not isinstance(bpm, (int, float)):
        raise TypeError(f"BPM must be a number, got {type(bpm).__name__}")
    
    bpm_float = float(bpm)
    
    if bpm_float < min_bpm or bpm_float > max_bpm:
        raise ValueError(
            f"BPM {bpm_float} is outside valid range ({min_bpm}-{max_bpm})"
        )
    
    return bpm_float


def validate_sample_rate(sample_rate: Any) -> int:
    """
    Validate sample rate value.
    
    Args:
        sample_rate: Sample rate to validate
    
    Returns:
        Validated sample rate as int
    
    Raises:
        TypeError: If sample_rate is not an integer
        ValueError: If sample_rate is not a supported value
    """
    valid_rates = [SAMPLE_RATE_44K, SAMPLE_RATE_48K]
    
    if not isinstance(sample_rate, int):
        raise TypeError(
            f"Sample rate must be an integer, got {type(sample_rate).__name__}"
        )
    
    if sample_rate not in valid_rates:
        raise ValueError(
            f"Invalid sample rate {sample_rate}. "
            f"Supported values: {valid_rates}"
        )
    
    return sample_rate


def validate_bit_depth(bit_depth: Any) -> int:
    """
    Validate bit depth value.
    
    Args:
        bit_depth: Bit depth to validate
    
    Returns:
        Validated bit depth as int
    
    Raises:
        TypeError: If bit_depth is not an integer
        ValueError: If bit_depth is not a supported value
    """
    valid_depths = [BIT_DEPTH_16, BIT_DEPTH_24, BIT_DEPTH_32]
    
    if not isinstance(bit_depth, int):
        raise TypeError(
            f"Bit depth must be an integer, got {type(bit_depth).__name__}"
        )
    
    if bit_depth not in valid_depths:
        raise ValueError(
            f"Invalid bit depth {bit_depth}. "
            f"Supported values: {valid_depths}"
        )
    
    return bit_depth


def validate_channels(channels: Any) -> int:
    """
    Validate channel count.
    
    Args:
        channels: Number of channels to validate
    
    Returns:
        Validated channel count
    
    Raises:
        TypeError: If channels is not an integer
        ValueError: If channels is not supported
    """
    valid_channels = [MONO, STEREO]
    
    if not isinstance(channels, int):
        raise TypeError(
            f"Channels must be an integer, got {type(channels).__name__}"
        )
    
    if channels not in valid_channels:
        raise ValueError(
            f"Invalid channel count {channels}. "
            f"Supported values: {valid_channels}"
        )
    
    return channels


def validate_buffer_size(buffer_size: Any) -> int:
    """
    Validate buffer size.
    
    Args:
        buffer_size: Buffer size to validate
    
    Returns:
        Validated buffer size
    
    Raises:
        TypeError: If buffer_size is not an integer
        ValueError: If buffer_size is not a power of 2 or not in valid range
    """
    valid_sizes = [BUFFER_SIZE_SMALL, BUFFER_SIZE_MEDIUM, BUFFER_SIZE_LARGE]
    
    if not isinstance(buffer_size, int):
        raise TypeError(
            f"Buffer size must be an integer, got {type(buffer_size).__name__}"
        )
    
    # Must be a power of 2
    if buffer_size <= 0 or (buffer_size & (buffer_size - 1)) != 0:
        raise ValueError(
            f"Buffer size must be a positive power of 2, got {buffer_size}"
        )
    
    if buffer_size not in valid_sizes:
        # Allow but warn (could extend valid sizes)
        pass
    
    return buffer_size


# =============================================================================
# FILE VALIDATORS
# =============================================================================

AUDIO_EXTENSIONS = {FORMAT_WAV, FORMAT_MP3, FORMAT_FLAC, FORMAT_OGG, FORMAT_AAC}


def validate_audio_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: Optional[set] = None
) -> Path:
    """
    Validate audio file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must already exist
        allowed_extensions: Set of allowed extensions (default: all audio formats)
    
    Returns:
        Validated Path object
    
    Raises:
        TypeError: If file_path is not a string or Path
        ValueError: If path is invalid or has wrong extension
        FileNotFoundError: If must_exist=True and file doesn't exist
    """
    if not isinstance(file_path, (str, Path)):
        raise TypeError(
            f"File path must be str or Path, got {type(file_path).__name__}"
        )
    
    path = Path(file_path).resolve()
    
    if allowed_extensions is None:
        allowed_extensions = AUDIO_EXTENSIONS
    
    # Check extension
    ext = path.suffix.lstrip('.').lower()
    if ext not in allowed_extensions:
        raise InvalidFileFormatException(
            f"Invalid file format '{ext}'. "
            f"Allowed formats: {', '.join(sorted(allowed_extensions))}"
        )
    
    # Check existence
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    return path


def validate_directory_path(
    dir_path: Union[str, Path],
    must_exist: bool = True,
    create_if_missing: bool = False
) -> Path:
    """
    Validate directory path.
    
    Args:
        dir_path: Directory path to validate
        must_exist: Whether directory must already exist
        create_if_missing: Create directory if it doesn't exist
    
    Returns:
        Validated Path object
    
    Raises:
        TypeError: If dir_path is not a string or Path
        FileNotFoundError: If must_exist=True and directory doesn't exist
        PermissionError: If create_if_missing=True but can't create
    """
    if not isinstance(dir_path, (str, Path)):
        raise TypeError(
            f"Directory path must be str or Path, got {type(dir_path).__name__}"
        )
    
    path = Path(dir_path).resolve()
    
    if not path.exists():
        if must_exist and not create_if_missing:
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise PermissionError(
                    f"Cannot create directory {path}: {e}"
                ) from e
    
    elif not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    return path


# =============================================================================
# GENRE & CLASSIFICATION VALIDATORS
# =============================================================================

def validate_genre(genre: Any, allowed_genres: Optional[dict] = None) -> str:
    """
    Validate genre name.
    
    Args:
        genre: Genre name to validate
        allowed_genres: Dict of allowed genres with BPM ranges
    
    Returns:
        Validated lowercase genre name
    
    Raises:
        TypeError: If genre is not a string
        ValueError: If genre is not in allowed list
    """
    if not isinstance(genre, str):
        raise TypeError(f"Genre must be a string, got {type(genre).__name__}")
    
    genre_lower = genre.lower().strip()
    
    if allowed_genres is None:
        allowed_genres = BPM_RANGES
    
    if genre_lower not in allowed_genres:
        raise ValueError(
            f"Unknown genre '{genre}'. "
            f"Allowed genres: {', '.join(sorted(allowed_genres.keys()))}"
        )
    
    return genre_lower


def validate_mood(mood: Any) -> str:
    """
    Validate mood/tags classification.
    
    Args:
        mood: Mood to validate
    
    Returns:
        Validated lowercase mood
    
    Raises:
        TypeError: If mood is not a string
    """
    valid_moods = {
        "happy", "sad", "energetic", "calm", "dark", "bright",
        "aggressive", "chill", "uplifting", "melancholic",
        "romantic", "angry", "peaceful", "tense", "mysterious"
    }
    
    if not isinstance(mood, str):
        raise TypeError(f"Mood must be a string, got {type(mood).__name__}")
    
    mood_lower = mood.lower().strip()
    
    if mood_lower not in valid_moods:
        # Allow unknown moods but could warn
        pass
    
    return mood_lower


def validate_key(key: Any) -> str:
    """
    Validate musical key.
    
    Args:
        key: Musical key to validate (e.g., "C", "Am", "F#m")
    
    Returns:
        Validated musical key
    
    Raises:
        TypeError: If key is not a string
        ValueError: If key format is invalid
    """
    if not isinstance(key, str):
        raise TypeError(f"Key must be a string, got {type(key).__name__}")
    
    key_clean = key.strip()
    
    # Basic validation: root note + optional mode indicator
    valid_roots = {
        "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b",
        "db", "eb", "gb", "ab", "bb"  # Enharmonic equivalents
    }
    valid_modes = {"", "m", "maj", "min", "maj7", "min7", "7", "sus2", "sus4"}
    
    # Simple pattern matching
    pattern = r'^([A-Ga-g][#b]?)(m|maj|min|maj7|min7|7|sus2|sus4)?$'
    if not re.match(pattern, key_clean):
        raise ValueError(
            f"Invalid musical key '{key}'. "
            f"Expected format: e.g., 'C', 'Am', 'F#m', 'G7'"
        )
    
    return key_clean


def validate_stem_type(stem_type: Any) -> str:
    """
    Validate stem type for separation.
    
    Args:
        stem_type: Stem type to validate
    
    Returns:
        Validated lowercase stem type
    
    Raises:
        TypeError: If stem_type is not a string
        ValueError: If stem_type is not recognized
    """
    if not isinstance(stem_type, str):
        raise TypeError(
            f"Stem type must be a string, got {type(stem_type).__name__}"
        )
    
    stem_lower = stem_type.lower().strip()
    
    if stem_lower not in STEM_TYPES:
        raise ValueError(
            f"Invalid stem type '{stem_type}'. "
            f"Allowed types: {', '.join(STEM_TYPES)}"
        )
    
    return stem_lower


# =============================================================================
# AUDIO PROCESSING PARAMETER VALIDATORS
# =============================================================================

def validate_gain(gain_db: Any, min_gain: float = -60.0, max_gain: float = 20.0) -> float:
    """
    Validate gain in dB.
    
    Args:
        gain_db: Gain value in dB
        min_gain: Minimum allowed gain
        max_gain: Maximum allowed gain
    
    Returns:
        Validated gain in dB
    
    Raises:
        TypeError: If gain_db is not a number
        ValueError: If gain is outside valid range
    """
    if not isinstance(gain_db, (int, float)):
        raise TypeError(f"Gain must be a number, got {type(gain_db).__name__}")
    
    gain = float(gain_db)
    
    if gain < min_gain or gain > max_gain:
        raise ValueError(
            f"Gain {gain}dB is outside valid range ({min_gain}dB to {max_gain}dB)"
        )
    
    return gain


def validate_threshold(threshold: Any, min_db: float = -60.0, max_db: float = 0.0) -> float:
    """
    Validate threshold value in dB.
    
    Args:
        threshold: Threshold value in dB
        min_db: Minimum allowed threshold
        max_db: Maximum allowed threshold
    
    Returns:
        Validated threshold in dB
    
    Raises:
        TypeError: If threshold is not a number
        ValueError: If threshold is outside valid range
    """
    if not isinstance(threshold, (int, float)):
        raise TypeError(
            f"Threshold must be a number, got {type(threshold).__name__}"
        )
    
    threshold_val = float(threshold)
    
    if threshold_val < min_db or threshold_val > max_db:
        raise ValueError(
            f"Threshold {threshold_val}dB is outside valid range "
            f"({min_db}dB to {max_db}dB)"
        )
    
    return threshold_val


def validate_ratio(ratio: Any, min_ratio: float = 1.0, max_ratio: float = 20.0) -> float:
    """
    Validate compression/limiting ratio.
    
    Args:
        ratio: Compression ratio
        min_ratio: Minimum allowed ratio
        max_ratio: Maximum allowed ratio
    
    Returns:
        Validated ratio
    
    Raises:
        TypeError: If ratio is not a number
        ValueError: If ratio is outside valid range
    """
    if not isinstance(ratio, (int, float)):
        raise TypeError(f"Ratio must be a number, got {type(ratio).__name__}")
    
    ratio_val = float(ratio)
    
    if ratio_val < min_ratio or ratio_val > max_ratio:
        raise ValueError(
            f"Ratio {ratio_val}x is outside valid range ({min_ratio}x to {max_ratio}x)"
        )
    
    return ratio_val


def validate_frequency(
    frequency: Any,
    min_freq: float = 20.0,
    max_freq: float = 20000.0
) -> float:
    """
    Validate frequency in Hz.
    
    Args:
        frequency: Frequency in Hz
        min_freq: Minimum allowed frequency
        max_freq: Maximum allowed frequency
    
    Returns:
        Validated frequency in Hz
    
    Raises:
        TypeError: If frequency is not a number
        ValueError: If frequency is outside valid range
    """
    if not isinstance(frequency, (int, float)):
        raise TypeError(
            f"Frequency must be a number, got {type(frequency).__name__}"
        )
    
    freq = float(frequency)
    
    if freq < min_freq or freq > max_freq:
        raise ValueError(
            f"Frequency {freq}Hz is outside valid range ({min_freq}Hz to {max_freq}Hz)"
        )
    
    return freq


def validate_q_factor(q: Any, min_q: float = 0.1, max_q: float = 20.0) -> float:
    """
    Validate Q factor for EQ filters.
    
    Args:
        q: Q factor value
        min_q: Minimum allowed Q
        max_q: Maximum allowed Q
    
    Returns:
        Validated Q factor
    
    Raises:
        TypeError: If q is not a number
        ValueError: If q is outside valid range
    """
    if not isinstance(q, (int, float)):
        raise TypeError(f"Q factor must be a number, got {type(q).__name__}")
    
    q_val = float(q)
    
    if q_val < min_q or q_val > max_q:
        raise ValueError(
            f"Q factor {q_val} is outside valid range ({min_q} to {max_q})"
        )
    
    return q_val


def validate_percentage(
    value: Any,
    min_val: float = 0.0,
    max_val: float = 100.0,
    param_name: str = "value"
) -> float:
    """
    Validate percentage value.
    
    Args:
        value: Percentage value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Name for error messages
    
    Returns:
        Validated percentage
    
    Raises:
        TypeError: If value is not a number
        ValueError: If value is outside valid range
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be a number, got {type(value).__name__}")
    
    val = float(value)
    
    if val < min_val or val > max_val:
        raise ValueError(
            f"{param_name} {val}% is outside valid range ({min_val}% to {max_val}%)"
        )
    
    return val


def validate_duration(
    duration: Any,
    min_duration: float = 0.1,
    max_duration: float = 3600.0
) -> float:
    """
    Validate duration in seconds.
    
    Args:
        duration: Duration in seconds
        min_duration: Minimum allowed duration
        max_duration: Maximum allowed duration
    
    Returns:
        Validated duration in seconds
    
    Raises:
        TypeError: If duration is not a number
        ValueError: If duration is outside valid range
    """
    if not isinstance(duration, (int, float)):
        raise TypeError(
            f"Duration must be a number, got {type(duration).__name__}"
        )
    
    dur = float(duration)
    
    if dur < min_duration or dur > max_duration:
        raise ValueError(
            f"Duration {dur}s is outside valid range "
            f"({min_duration}s to {max_duration}s)"
        )
    
    return dur


def validate_probability(prob: Any) -> float:
    """
    Validate probability value (0.0 to 1.0).
    
    Args:
        prob: Probability value
    
    Returns:
        Validated probability
    
    Raises:
        TypeError: If prob is not a number
        ValueError: If prob is outside [0.0, 1.0] range
    """
    return validate_percentage(prob, 0.0, 100.0, "Probability")


# =============================================================================
# CONFIGURATION VALIDATORS
# =============================================================================

def validate_config_value(
    value: Any,
    expected_type: type,
    allowed_values: Optional[List[Any]] = None,
    param_name: str = "value"
) -> Any:
    """
    Generic configuration value validator.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        allowed_values: Optional list of allowed values
        param_name: Name for error messages
    
    Returns:
        Validated value
    
    Raises:
        TypeError: If value type doesn't match expected
        ValueError: If value not in allowed_values
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{param_name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    
    if allowed_values is not None and value not in allowed_values:
        raise ValueError(
            f"Invalid {param_name} '{value}'. "
            f"Allowed values: {allowed_values}"
        )
    
    return value


def validate_api_key(api_key: Optional[str], param_name: str = "api_key") -> str:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        param_name: Parameter name for errors
    
    Returns:
        Validated API key
    
    Raises:
        ValueError: If API key is empty or invalid format
    """
    if api_key is None:
        raise ValueError(f"{param_name} cannot be None")
    
    if not isinstance(api_key, str):
        raise TypeError(f"{param_name} must be a string")
    
    api_key_stripped = api_key.strip()
    
    if len(api_key_stripped) == 0:
        raise ValueError(f"{param_name} cannot be empty")
    
    if len(api_key_stripped) < 10:
        raise ValueError(f"{param_name} appears to be too short")
    
    return api_key_stripped


def validate_url(url: Any, require_https: bool = True) -> str:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        require_https: Whether to require HTTPS
    
    Returns:
        Validated URL string
    
    Raises:
        TypeError: If url is not a string
        ValueError: If URL format is invalid
    """
    if not isinstance(url, str):
        raise TypeError(f"URL must be a string, got {type(url).__name__}")
    
    url_stripped = url.strip()
    
    if not url_stripped:
        raise ValueError("URL cannot be empty")
    
    # Basic URL pattern
    url_pattern = r'^https?://'
    if require_https and not re.match(url_pattern, url_stripped):
        raise ValueError(f"URL must start with http:// or https://")
    
    if require_https and not url_stripped.startswith('https://'):
        raise ValueError(f"HTTPS is required")
    
    return url_stripped


# =============================================================================
# MIXING & EFFECT VALIDATORS
# =============================================================================

def validate_pan(pan: Any) -> float:
    """
    Validate pan position (-1.0 to 1.0).
    
    Args:
        pan: Pan value (-1.0 = left, 0.0 = center, 1.0 = right)
    
    Returns:
        Validated pan value
    
    Raises:
        TypeError: If pan is not a number
        ValueError: If pan is outside valid range
    """
    return validate_percentage(pan, -100.0, 100.0, "Pan") / 100.0


def validate_mix(mix: Any) -> float:
    """
    Validate dry/wet mix (0.0 to 1.0).
    
    Args:
        mix: Mix value (0.0 = fully dry, 1.0 = fully wet)
    
    Returns:
        Validated mix value
    
    Raises:
        TypeError: If mix is not a number
        ValueError: If mix is outside valid range
    """
    return validate_percentage(mix, 0.0, 100.0, "Mix") / 100.0


def validate_time_signature(time_sig: Any) -> Tuple[int, int]:
    """
    Validate time signature.
    
    Args:
        time_sig: Time signature (e.g., "4/4", "3/4", or (4, 4))
    
    Returns:
        Tuple of (beats_per_bar, beat_unit)
    
    Raises:
        TypeError: If time_sig is not string or tuple
        ValueError: If time signature is invalid
    """
    if isinstance(time_sig, str):
        match = re.match(r'^(\d+)/(\d+)$', time_sig.strip())
        if not match:
            raise ValueError(
                f"Invalid time signature '{time_sig}'. Expected format: '4/4'"
            )
        beats = int(match.group(1))
        unit = int(match.group(2))
    elif isinstance(time_sig, tuple) and len(time_sig) == 2:
        beats, unit = time_sig
        if not isinstance(beats, int) or not isinstance(unit, int):
            raise TypeError("Time signature tuple must contain two integers")
    else:
        raise TypeError(
            f"Time signature must be string or tuple, got {type(time_sig).__name__}"
        )
    
    # Validate ranges
    valid_beats = [2, 3, 4, 5, 6, 7, 8, 9, 12]
    valid_units = [2, 4, 8, 16]
    
    if beats not in valid_beats:
        raise ValueError(
            f"Invalid beats per bar {beats}. Must be one of: {valid_beats}"
        )
    
    if unit not in valid_units:
        raise ValueError(
            f"Invalid beat unit {unit}. Must be one of: {valid_units}"
        )
    
    return (beats, unit)


# =============================================================================
# UTILITY VALIDATORS
# =============================================================================

def validate_range(
    value: Any,
    min_val: Union[int, float],
    max_val: Union[int, float],
    param_name: str = "value"
) -> Union[int, float]:
    """
    Generic numeric range validator.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Parameter name for errors
    
    Returns:
        Validated value
    
    Raises:
        TypeError: If value is not a number
        ValueError: If value is outside range
    """
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{param_name} must be a number, got {type(value).__name__}"
        )
    
    if value < min_val or value > max_val:
        raise ValueError(
            f"{param_name} {value} is outside valid range ({min_val} to {max_val})"
        )
    
    return value


def validate_integer(
    value: Any,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    param_name: str = "value"
) -> int:
    """
    Validate integer value with optional range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        param_name: Parameter name for errors
    
    Returns:
        Validated integer
    
    Raises:
        TypeError: If value is not an integer
        ValueError: If value is outside range
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"{param_name} must be an integer, got {type(value).__name__}"
        )
    
    if min_val is not None and value < min_val:
        raise ValueError(
            f"{param_name} {value} is below minimum {min_val}"
        )
    
    if max_val is not None and value > max_val:
        raise ValueError(
            f"{param_name} {value} is above maximum {max_val}"
        )
    
    return value


def validate_list(
    value: Any,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    element_type: Optional[type] = None,
    param_name: str = "value"
) -> list:
    """
    Validate list with optional constraints.
    
    Args:
        value: List to validate
        min_length: Minimum list length
        max_length: Maximum list length
        element_type: Type all elements must be
        param_name: Parameter name for errors
    
    Returns:
        Validated list
    
    Raises:
        TypeError: If value is not a list or element type is wrong
        ValueError: If list length is outside range
    """
    if not isinstance(value, list):
        raise TypeError(f"{param_name} must be a list, got {type(value).__name__}")
    
    if min_length is not None and len(value) < min_length:
        raise ValueError(
            f"{param_name} has {len(value)} elements, minimum is {min_length}"
        )
    
    if max_length is not None and len(value) > max_length:
        raise ValueError(
            f"{param_name} has {len(value)} elements, maximum is {max_length}"
        )
    
    if element_type is not None:
        for i, item in enumerate(value):
            if not isinstance(item, element_type):
                raise TypeError(
                    f"{param_name}[{i}] must be {element_type.__name__}, "
                    f"got {type(item).__name__}"
                )
    
    return value


def validate_string(
    value: Any,
    min_length: int = 0,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    param_name: str = "value"
) -> str:
    """
    Validate string with optional constraints.
    
    Args:
        value: String to validate
        min_length: Minimum string length
        max_length: Maximum string length
        pattern: Optional regex pattern
        param_name: Parameter name for errors
    
    Returns:
        Validated string
    
    Raises:
        TypeError: If value is not a string
        ValueError: If string doesn't meet constraints
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{param_name} must be a string, got {type(value).__name__}"
        )
    
    if len(value) < min_length:
        raise ValueError(
            f"{param_name} has {len(value)} characters, minimum is {min_length}"
        )
    
    if max_length is not None and len(value) > max_length:
        raise ValueError(
            f"{param_name} has {len(value)} characters, maximum is {max_length}"
        )
    
    if pattern is not None and not re.match(pattern, value):
        raise ValueError(
            f"{param_name} does not match required pattern"
        )
    
    return value
