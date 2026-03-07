"""
Constants for AI DJ Project

Contains all project-wide constants including audio settings,
genre definitions, processing parameters, and configuration values.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT METADATA
# =============================================================================

PROJECT_NAME = "AI DJ Project"
PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "Peter"
PROJECT_DESCRIPTION = "AI-powered DJ system for music generation, mixing, and performance"

# =============================================================================
# AUDIO SETTINGS
# =============================================================================

# Sample rates
SAMPLE_RATE_44K = 44100
SAMPLE_RATE_48K = 48000
SAMPLE_RATE_DEFAULT = SAMPLE_RATE_48K

# Bit depths
BIT_DEPTH_16 = 16
BIT_DEPTH_24 = 24
BIT_DEPTH_32 = 32
BIT_DEPTH_DEFAULT = BIT_DEPTH_24

# Channels
MONO = 1
STEREO = 2
CHANNELS_DEFAULT = STEREO

# Audio formats
FORMAT_WAV = "wav"
FORMAT_MP3 = "mp3"
FORMAT_FLAC = "flac"
FORMAT_OGG = "ogg"
FORMAT_AAC = "aac"

# Buffer sizes
BUFFER_SIZE_SMALL = 256
BUFFER_SIZE_MEDIUM = 512
BUFFER_SIZE_LARGE = 1024
BUFFER_SIZE_DEFAULT = BUFFER_SIZE_MEDIUM

# =============================================================================
# BPM AND TEMPO CONSTANTS
# =============================================================================

BPM_MIN = 60
BPM_MAX = 200
BPM_DEFAULT = 120

# Genre-specific BPM ranges
BPM_RANGES = {
    "dubstep": (70, 75),
    "trap": (140, 180),
    "house": (118, 130),
    "techno": (120, 150),
    "trance": (138, 145),
    "drum_and_bass": (160, 180),
    "hip_hop": (80, 100),
    "pop": (100, 128),
    "edm": (128, 140),
}

# Tempo sync tolerance (percentage)
TEMPO_TOLERANCE = 0.05  # 5%

# =============================================================================
# DIRECTORY PATHS
# =============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = SRC_DIR / "output"
CACHE_DIR = SRC_DIR / "cache"
RECORDINGS_DIR = SRC_DIR / "recordings"
ANALYTICS_DIR = SRC_DIR / "analytics_reports"
WORKFLOW_DIR = SRC_DIR / "workflow"
OFFLINE_DIR = SRC_DIR / "offline"

# Ensure directories exist
for directory in [OUTPUT_DIR, CACHE_DIR, RECORDINGS_DIR, ANALYTICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STEM SEPARATION
# =============================================================================

STEM_TYPES = ["bass", "drums", "vocals", "other"]

STEM_MODEL_DEMUCS = "demucs"
STEM_MODEL_SPLEETER = "spleeter"
STEM_MODEL_DEFAULT = STEM_MODEL_DEMUCS

# =============================================================================
# MIXING CONSTANTS
# =============================================================================

# Volume levels (0.0 to 1.0)
VOLUME_MIN = 0.0
VOLUME_MAX = 1.0
VOLUME_DEFAULT = 0.8

# Crossfade durations (seconds)
CROSSFADE_SHORT = 2
CROSSFADE_MEDIUM = 5
CROSSFADE_LONG = 10
CROSSFADE_DEFAULT = CROSSFADE_MEDIUM

# EQ bands
EQ_LOW_BAND = (20, 250)
EQ_MID_BAND = (250, 4000)
EQ_HIGH_BAND = (4000, 20000)

# Compressor defaults
COMPRESSOR_THRESHOLD_DEFAULT = -20  # dB
COMPRESSOR_RATIO_DEFAULT = 4.0
COMPRESSOR_ATTACK_DEFAULT = 10  # ms
COMPRESSOR_RELEASE_DEFAULT = 100  # ms

# Reverb defaults
REVERB_ROOM_SIZE_DEFAULT = 0.5
REVERB_DAMPING_DEFAULT = 0.5
REVERB_WET_DRY_DEFAULT = 0.3

# =============================================================================
# EFFECTS PROCESSING
# =============================================================================

# Effect wet/dry mix
EFFECT_WET_MIN = 0.0
EFFECT_WET_MAX = 1.0
EFFECT_WET_DEFAULT = 0.5

# Filter types
FILTER_LOWPASS = "lowpass"
FILTER_HIGHPASS = "highpass"
FILTER_BANDPASS = "bandpass"
FILTER_NOTCH = "notch"

# Filter default frequencies
FILTER_LP_DEFAULT = 8000  # Hz
FILTER_HP_DEFAULT = 100   # Hz

# Distortion types
DISTORTION_SOFT = "soft"
DISTORTION_HARD = "hard"
DISTORTION_BITCRUSH = "bitcrush"

# =============================================================================
# KEY AND HARMONY
# =============================================================================

# Musical keys
ICAL_KEYS = [
    "CMUS", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]
MINOR_KEYS = [f"{key}m" for key in MUSICAL_KEYS]
ALL_KEYS = MUSICAL_KEYS + MINOR_KEYS

# Camelot wheel system (for harmonic mixing)
CAMELOT_WHEEL = {
    "8A": ["8A", "9A", "7A", "8B", "9B"],
    "9A": ["9A", "10A", "8A", "9B", "10B"],
    "10A": ["10A", "11A", "9A", "10B", "11B"],
    "11A": ["11A", "12A", "10A", "11B", "12B"],
    "12A": ["12A", "1A", "11A", "12B", "1B"],
    "1A": ["1A", "2A", "12A", "1B", "2B"],
    "2A": ["2A", "3A", "1A", "2B", "3B"],
    "3A": ["3A", "4A", "2A", "3B", "4B"],
    "4A": ["4A", "5A", "3A", "4B", "5B"],
    "5A": ["5A", "6A", "4A", "5B", "6B"],
    "6A": ["6A", "7A", "5A", "6B", "7B"],
    "7A": ["7A", "8A", "6A", "7B", "8B"],
    "8B": ["8B", "9B", "7B", "8A", "9A"],
    "9B": ["9B", "10B", "8B", "9A", "10A"],
    "10B": ["10B", "11B", "9B", "10A", "11A"],
    "11B": ["11B", "12B", "10B", "11A", "12A"],
    "12B": ["12B", "1B", "11B", "12A", "1A"],
    "1B": ["1B", "2B", "12B", "1A", "2A"],
    "2B": ["2B", "3B", "1B", "2A", "3A"],
    "3B": ["3B", "4B", "2B", "3A", "4A"],
    "4B": ["4B", "5B", "3B", "4A", "5A"],
    "5B": ["5B", "6B", "4B", "5A", "6A"],
    "6B": ["6B", "7B", "5B", "6A", "7A"],
    "7B": ["7B", "8B", "6B", "7A", "8A"],
}

# =============================================================================
# ENERGY AND MOOD LEVELS
# =============================================================================

class EnergyLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BUILDUP = "buildup"
    DROP_CENTER = "drop_center"

class Mood:
    CALM = "calm"
    NEUTRAL = "neutral"
    UPBEAT = "upbeat"
    AGGRESSIVE = "aggressive"
    DARK = "dark"
    EUPHORIC = "euphoric"
    MELANCHOLIC = "melancholic"
    MYSTERIOUS = "mysterious"

# =============================================================================
# ANALYSIS CONSTANTS
# =============================================================================

# Feature extraction
FFT_SIZE = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20

# Beat detection
BEAT_THRESHOLD_DEFAULT = 0.3
BEAT_WINDOW_SIZE = 43  # frames

# Loudness standards (LUFS)
LUFS_TARGET_MASTER = -14  # Streaming standard
LUFS_TARGET_MASTERING = -8  # Louder master
LUFS_TRUE_PEAK_MAX = -1.0  # dBTP

# =============================================================================
# AI MODELS
# =============================================================================

# Vocal generation
VOCAL_MODEL_VOCAFUSION = "vocalfusion"
VOCAL_MODEL_DEFAULT = VOCAL_MODEL_VOCAFUSION

# Stem separation models
STEM_MODEL_VERSIONS = {
    "demucs": ["htdemucs", "htdemucs_6s", "mdx_extra", "mdx"],
    "spleeter": ["spleeter:2stems", "spleeter:4stems", "spleeter:5stems"],
}

# Music generation models
GENERATION_MODELS = ["ace-step", "musicgen", "riffusion"]

# =============================================================================
# API AND SERVICE ENDPOINTS (PLACEHOLDERS)
# =============================================================================

# These should be set via environment variables or config file
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

# =============================================================================
# UI AND DISPLAY
# =============================================================================

# Radar display
RADAR_SWEEP_SPEED = 2.0  # seconds per rotation
RADAR_MAX_RANGE = 200  # BPM
RADAR_BLANK_ZONES = 10

# Waveform display
WAVEFORM_HEIGHT = 128
WAVEFORM_COLOR = "#00ff88"
WAVEFORM_BACKGROUND = "#1a1a2e"

# =============================================================================
# CACHE AND PERFORMANCE
# =============================================================================

CACHE_MAX_SIZE_MB = 500
CACHE_EXPIRY_HOURS = 24
CACHE_ENABLED = True

# Threading
MAX_WORKERS = 4
PROCESS_TIMEOUT = 300  # seconds

# =============================================================================
# FILE NAMING
# =============================================================================

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
OUTPUT_FILENAME_PATTERN = "{genre}_{timestamp}_{hash}"

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_FILE_NOT_FOUND = "Audio file not found"
ERROR_INVALID_FORMAT = "Unsupported audio format"
ERROR_PROCESSING_FAILED = "Audio processing failed"
ERROR_MODEL_NOT_FOUND = "AI model not available"

# =============================================================================
# SUCCESS MESSAGES
# =============================================================================

SUCCESS_GENERATED = "Audio generated successfully"
SUCCESS_SPLIT = "Stems separated successfully"
SUCCESS_MIXED = "Mix completed successfully"
SUCCESS_MASTERED = "Mastering completed successfully"
