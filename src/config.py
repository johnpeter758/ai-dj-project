#!/usr/bin/env python3
"""
Configuration System for AI DJ Project
Supports environment variables, config files, and defaults.
"""

import os
import json
from pathlib import Path
from typing import Any, Optional, get_type_hints
from dataclasses import dataclass, field, is_dataclass


# Project root directory
PROJECT_ROOT = Path("/Users/johnpeter/ai-dj-project")
SRC_DIR = PROJECT_ROOT / "src"


@dataclass
class AudioConfig:
    """Audio processing settings"""
    sample_rate: int = 44100
    bit_depth: int = 24
    buffer_size: int = 512
    channels: int = 2
    default_bpm: int = 128
    loudness_target: float = -14.0  # LUFS
    true_peak_limit: float = -1.0    # dB


@dataclass
class ModelConfig:
    """AI model settings"""
    # Stem separation
    demucs_model: str = "htdemucs_6s"
    demucs_device: str = "cuda"
    
    # Vocals/AI
    vocal_model: str = "ACE-Step"
    vocal_device: str = "cuda"
    
    # Generation models
    melody_model: str = "musicgen"
    bass_model: str = "riffusion"
    
    # Model cache directories
    model_cache_dir: str = str(SRC_DIR / "cache" / "models")
    stem_cache_dir: str = str(SRC_DIR / "cache" / "stems")


@dataclass
class EffectConfig:
    """Audio effects default settings"""
    reverb_default_mix: float = 0.3
    reverb_default_decay: float = 2.0
    delay_default_mix: float = 0.25
    delay_default_time: float = 0.5  # seconds
    compressor_threshold: float = -20.0
    compressor_ratio: float = 4.0
    eq_default_preset: str = "flat"


@dataclass
class OutputConfig:
    """Output/export settings"""
    output_dir: str = str(SRC_DIR / "output")
    recordings_dir: str = str(SRC_DIR / "recordings")
    format: str = "wav"
    include_stems: bool = True
    normalize_output: bool = True


@dataclass
class APIConfig:
    """External API settings"""
    # These should be set via environment variables
    openai_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    elevenlabs_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ELEVENLABS_API_KEY")
    )
    spotify_client_id: Optional[str] = field(
        default_factory=lambda: os.getenv("SPOTIFY_CLIENT_ID")
    )
    spotify_client_secret: Optional[str] = field(
        default_factory=lambda: os.getenv("SPOTIFY_CLIENT_SECRET")
    )


@dataclass
class GenreConfig:
    """Genre-specific defaults"""
    available_genres: list = field(default_factory=lambda: [
        "pop", "house", "techno", "trance", "dubstep", 
        "hip-hop", "rnb", "rock", "edm", "ambient"
    ])
    default_genre: str = "house"
    energy_range: tuple = (0.0, 1.0)
    bpm_range: tuple = (70, 180)


@dataclass 
class CloudConfig:
    """Cloud sync settings"""
    enabled: bool = False
    sync_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv("CLOUD_SYNC_ENDPOINT")
    )
    sync_interval: int = 300  # seconds
    backup_enabled: bool = False


class Config:
    """
    Main configuration class that loads from multiple sources:
    1. Default values (lowest priority)
    2. Config file (JSON)
    3. Environment variables (highest priority)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.audio = AudioConfig()
        self.models = ModelConfig()
        self.effects = EffectConfig()
        self.output = OutputConfig()
        self.api = APIConfig()
        self.genres = GenreConfig()
        self.cloud = CloudConfig()
        
        # Load from config file if provided
        if config_file:
            self.load_from_file(config_file)
        else:
            # Try default config locations
            self._load_default_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _load_default_config(self):
        """Try loading from default config locations"""
        default_locations = [
            PROJECT_ROOT / "config.json",
            SRC_DIR / "config.json",
            Path.home() / ".ai-dj" / "config.json",
        ]
        
        for loc in default_locations:
            if loc.exists():
                self.load_from_file(str(loc))
                break
    
    def load_from_file(self, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Apply settings to each section
        for section, values in data.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if is_dataclass(section_obj):
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Audio settings
        if os.getenv("AIDJ_SAMPLE_RATE"):
            self.audio.sample_rate = int(os.getenv("AIDJ_SAMPLE_RATE"))
        if os.getenv("AIDJ_BUFFER_SIZE"):
            self.audio.buffer_size = int(os.getenv("AIDJ_BUFFER_SIZE"))
        if os.getenv("AIDJ_BPM"):
            self.audio.default_bpm = int(os.getenv("AIDJ_BPM"))
        
        # Model settings
        if os.getenv("AIDJ_DEMUCS_DEVICE"):
            self.models.demucs_device = os.getenv("AIDJ_DEMUCS_DEVICE")
        if os.getenv("AIDJ_VOCAL_DEVICE"):
            self.models.vocal_device = os.getenv("AIDJ_VOCAL_DEVICE")
        
        # Output settings
        if os.getenv("AIDJ_OUTPUT_DIR"):
            self.output.output_dir = os.getenv("AIDJ_OUTPUT_DIR")
        
        # Cloud settings
        if os.getenv("AIDJ_CLOUD_ENABLED"):
            self.cloud.enabled = os.getenv("AIDJ_CLOUD_ENABLED").lower() == "true"
    
    def save_to_file(self, path: str):
        """Save current config to JSON file"""
        config_dict = {}
        
        for section_name in ['audio', 'models', 'effects', 'output', 'api', 'genres', 'cloud']:
            section = getattr(self, section_name)
            if is_dataclass(section):
                config_dict[section_name] = {
                    k: v for k, v in vars(section).items() 
                    if not k.startswith('_')
                }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-notation key (e.g., 'audio.sample_rate')"""
        parts = key.split('.')
        obj = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj
    
    def __repr__(self):
        return f"<Config: audio={self.audio.sample_rate}Hz, bpm={self.audio.default_bpm}, device={self.models.demucs_device}>"


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config():
    """Reset global config (useful for testing)"""
    global _config
    _config = None


# Convenience accessors
audio = lambda: get_config().audio
models = lambda: get_config().models
effects = lambda: get_config().effects
output = lambda: get_config().output
api = lambda: get_config().api
genres = lambda: get_config().genres
cloud = lambda: get_config().cloud
