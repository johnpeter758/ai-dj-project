"""
Effect Presets System for AI DJ Project
Pre-configured effect chains and parameter sets for quick sound design
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class PresetCategory(Enum):
    """Categories of effect presets."""
    REVERB = "reverb"
    DELAY = "delay"
    COMPRESSION = "compression"
    EQ = "eq"
    DISTORTION = "distortion"
    FILTER = "filter"
    MODULATION = "modulation"
    MASTERING = "mastering"
    CHAIN = "chain"  # Multi-effect chains
    GENRE = "genre"  # Genre-specific


@dataclass
class EffectPreset:
    """Single effect preset with parameters."""
    name: str
    category: PresetCategory
    description: str
    parameters: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert preset to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "parameters": self.parameters,
            "tags": self.tags
        }


@dataclass 
class EffectChain:
    """Multi-effect chain preset."""
    name: str
    description: str
    effects: List[Dict[str, Any]]  # List of {effect_name, parameters}
    tags: List[str] = field(default_factory=list)
    genre: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "effects": self.effects,
            "tags": self.tags,
            "genre": self.genre
        }


# ==================== REVERB PRESETS ====================

REVERB_PRESETS: List[EffectPreset] = [
    EffectPreset(
        name="Large Hall",
        category=PresetCategory.REVERB,
        description="Spacious hall reverb with long decay",
        parameters={
            "reverb_type": "hall",
            "room_size": 0.9,
            "damping": 0.3,
            "wet_dry": 0.35,
            "width": 1.0,
            "pre_delay_ms": 20.0,
            "early_reflections": 0.4
        },
        tags=["spacious", "large", "concert", "dramatic"]
    ),
    EffectPreset(
        name="Small Room",
        category=PresetCategory.REVERB,
        description="Intimate room reverb for drums",
        parameters={
            "reverb_type": "room",
            "room_size": 0.3,
            "damping": 0.6,
            "wet_dry": 0.2,
            "width": 0.8,
            "pre_delay_ms": 0.0,
            "early_reflections": 0.5
        },
        tags=["tight", "drums", "acoustic", "intimate"]
    ),
    EffectPreset(
        name="Plate Warm",
        category=PresetCategory.REVERB,
        description="Warm plate reverb for vocals",
        parameters={
            "reverb_type": "plate",
            "room_size": 0.6,
            "damping": 0.4,
            "wet_dry": 0.3,
            "width": 0.9,
            "pre_delay_ms": 10.0,
            "early_reflections": 0.25
        },
        tags=["vocal", "warm", "smooth", "classic"]
    ),
    EffectPreset(
        name="Cathedral",
        category=PresetCategory.REVERB,
        description="Massive cathedral reverb for ethereal sounds",
        parameters={
            "reverb_type": "cathedral",
            "room_size": 1.0,
            "damping": 0.2,
            "wet_dry": 0.5,
            "width": 1.0,
            "pre_delay_ms": 40.0,
            "early_reflections": 0.3,
            "freeze": False
        },
        tags=["ethereal", "massive", "ambient", "spiritual"]
    ),
    EffectPreset(
        name="Ambience",
        category=PresetCategory.REVERB,
        description="Subtle ambience for depth",
        parameters={
            "reverb_type": "ambient",
            "room_size": 0.4,
            "damping": 0.7,
            "wet_dry": 0.1,
            "width": 0.7,
            "pre_delay_ms": 0.0,
            "early_reflections": 0.15
        },
        tags=["subtle", "depth", "pad", "background"]
    ),
    EffectPreset(
        name="Spring",
        category=PresetCategory.REVERB,
        description="Retro spring reverb (guitar style)",
        parameters={
            "reverb_type": "spring",
            "room_size": 0.5,
            "damping": 0.5,
            "wet_dry": 0.25,
            "width": 0.6,
            "pre_delay_ms": 0.0,
            "early_reflections": 0.2
        },
        tags=["retro", "guitar", "vintage", "bouncy"]
    ),
]


# ==================== DELAY PRESETS ====================

DELAY_PRESETS: List[EffectPreset] = [
    EffectPreset(
        name="Slap Echo",
        category=PresetCategory.DELAY,
        description="Short slap-back delay for vocals",
        parameters={
            "delay_time_ms": 80,
            "feedback": 0.3,
            "wet_dry": 0.25,
            "sync_to_bpm": True,
            "note_value": "eighth",
            "filter_cutoff": 3000
        },
        tags=["vocal", "slap", "retro", "tight"]
    ),
    EffectPreset(
        name="Ping Pong",
        category=PresetCategory.DELAY,
        description="Stereo ping-pong delay",
        parameters={
            "delay_time_ms": 250,
            "feedback": 0.4,
            "wet_dry": 0.35,
            "sync_to_bpm": True,
            "note_value": "quarter",
            "ping_pong": True,
            "filter_cutoff": 4000
        },
        tags=["stereo", "movement", "groove", "spatial"]
    ),
    EffectPreset(
        name="Tape Echo",
        category=PresetCategory.DELAY,
        description="Warm tape-style echo",
        parameters={
            "delay_time_ms": 375,
            "feedback": 0.5,
            "wet_dry": 0.4,
            "sync_to_bpm": False,
            "wow_flutter": 0.02,
            "saturation": 0.3,
            "filter_cutoff": 2500
        },
        tags=["warm", "analog", "vintage", "tape"]
    ),
    EffectPreset(
        name="Dub Delay",
        category=PresetCategory.DELAY,
        description="Heavy dub delay with filter",
        parameters={
            "delay_time_ms": 500,
            "feedback": 0.65,
            "wet_dry": 0.5,
            "sync_to_bpm": True,
            "note_value": "half",
            "filter_cutoff": 800,
            "filter_resonance": 2.0
        },
        tags=["dub", " reggae", "heavy", "experimental"]
    ),
    EffectPreset(
        name="Quarter Note",
        category=PresetCategory.DELAY,
        description="Simple quarter-note delay",
        parameters={
            "delay_time_ms": 500,
            "feedback": 0.35,
            "wet_dry": 0.3,
            "sync_to_bpm": True,
            "note_value": "quarter",
            "filter_cutoff": 5000
        },
        tags=["simple", "basic", "rhythmic", "fill"]
    ),
]


# ==================== COMPRESSION PRESETS ====================

COMPRESSION_PRESETS: List[EffectPreset] = [
    EffectPreset(
        name="Gentle Bus",
        category=PresetCategory.COMPRESSION,
        description="Gentle bus compression for glue",
        parameters={
            "threshold_db": -18,
            "ratio": 2.0,
            "attack_ms": 10.0,
            "release_ms": 150.0,
            "knee_db": 6.0,
            "makeup_gain_db": 0,
            "gain_reduction_limit_db": 3
        },
        tags=["bus", "glue", "gentle", "mixing"]
    ),
    EffectPreset(
        name="Punchy Drums",
        category=PresetCategory.COMPRESSION,
        description="Aggressive drum compression for impact",
        parameters={
            "threshold_db": -12,
            "ratio": 4.0,
            "attack_ms": 1.0,
            "release_ms": 80.0,
            "knee_db": 3.0,
            "makeup_gain_db": 2,
            "gain_reduction_limit_db": 8
        },
        tags=["drums", "punchy", "impact", "aggressive"]
    ),
    EffectPreset(
        name="Vocal Squeeze",
        category=PresetCategory.COMPRESSION,
        description="Smooth vocal compression",
        parameters={
            "threshold_db": -20,
            "ratio": 3.0,
            "attack_ms": 5.0,
            "release_ms": 100.0,
            "knee_db": 8.0,
            "makeup_gain_db": 3,
            "gain_reduction_limit_db": 6
        },
        tags=["vocal", "smooth", "consistent", "radio"]
    ),
    EffectPreset(
        name="Limiting",
        category=PresetCategory.COMPRESSION,
        description="Hard limiting for max loudness",
        parameters={
            "threshold_db": -3,
            "ratio": 20.0,
            "attack_ms": 0.1,
            "release_ms": 50.0,
            "knee_db": 0.5,
            "makeup_gain_db": 0,
            "gain_reduction_limit_db": 12
        },
        tags=["limiter", "loud", "brickwall", "mastering"]
    ),
    EffectPreset(
        name="Bass Tight",
        category=PresetCategory.COMPRESSION,
        description="Fast compression for tight bass",
        parameters={
            "threshold_db": -15,
            "ratio": 3.5,
            "attack_ms": 2.0,
            "release_ms": 60.0,
            "knee_db": 4.0,
            "makeup_gain_db": 1,
            "gain_reduction_limit_db": 5
        },
        tags=["bass", "tight", "punch", "low-end"]
    ),
]


# ==================== EQ PRESETS ====================

EQ_PRESETS: List[EffectPreset] = [
    EffectPreset(
        name="Vocal Boost",
        category=PresetCategory.EQ,
        description="Enhanced vocal presence",
        parameters={
            "bands": [
                {"freq": 80, "gain_db": -3, "q": 1.0, "type": "low_cut"},
                {"freq": 250, "gain_db": -2, "q": 0.7, "type": "peaking"},
                {"freq": 2000, "gain_db": 3, "q": 1.5, "type": "peaking"},
                {"freq": 5000, "gain_db": 2, "q": 2.0, "type": "peaking"},
                {"freq": 10000, "gain_db": 1, "q": 1.0, "type": "high_shelf"}
            ]
        },
        tags=["vocal", "presence", "boost", "speech"]
    ),
    EffectPreset(
        name="Bass Warmth",
        category=PresetCategory.EQ,
        description="Warm bass enhancement",
        parameters={
            "bands": [
                {"freq": 40, "gain_db": 4, "q": 0.5, "type": "peaking"},
                {"freq": 100, "gain_db": 2, "q": 0.7, "type": "peaking"},
                {"freq": 200, "gain_db": -1, "q": 1.0, "type": "peaking"},
                {"freq": 5000, "gain_db": -2, "q": 0.7, "type": "high_cut"}
            ]
        },
        tags=["bass", "warm", "low-end", "body"]
    ),
    EffectPreset(
        name="Air Boost",
        category=PresetCategory.EQ,
        description="Air and sparkle on highs",
        parameters={
            "bands": [
                {"freq": 8000, "gain_db": 3, "q": 0.7, "type": "high_shelf"},
                {"freq": 12000, "gain_db": 4, "q": 0.5, "type": "high_shelf"},
                {"freq": 16000, "gain_db": 2, "q": 0.3, "type": "high_shelf"}
            ]
        },
        tags=["air", "sparkle", "highs", "bright"]
    ),
    EffectPreset(
        name="Telephone",
        category=PresetCategory.EQ,
        description="Telephone/boxy effect",
        parameters={
            "bands": [
                {"freq": 300, "gain_db": -4, "q": 1.5, "type": "peaking"},
                {"freq": 3000, "gain_db": 5, "q": 2.0, "type": "peaking"},
                {"freq": 3500, "gain_db": -3, "q": 1.0, "type": "peaking"},
                {"freq": 8000, "gain_db": -6, "q": 0.7, "type": "high_cut"}
            ]
        },
        tags=["retro", "telephone", "boxy", "lo-fi"]
    ),
    EffectPreset(
        name="Car System",
        category=PresetCategory.EQ,
        description="EQ for car speaker playback",
        parameters={
            "bands": [
                {"freq": 60, "gain_db": 3, "q": 0.5, "type": "peaking"},
                {"freq": 250, "gain_db": -2, "q": 0.7, "type": "peaking"},
                {"freq": 1000, "gain_db": 1, "q": 1.0, "type": "peaking"},
                {"freq": 4000, "gain_db": -2, "q": 1.0, "type": "peaking"},
                {"freq": 8000, "gain_db": -4, "q": 0.5, "type": "high_cut"}
            ]
        },
        tags=["car", "playback", "portable", "bass-heavy"]
    ),
]


# ==================== DISTORTION PRESETS ====================

DISTORTION_PRESETS: List[EffectPreset] = [
    EffectPreset(
        name="Soft Saturation",
        category=PresetCategory.DISTORTION,
        description="Gentle tape-like saturation",
        parameters={
            "drive": 0.2,
            "tone": 0.5,
            "mix": 0.3,
            "type": "tape"
        },
        tags=["tape", "warm", "saturation", "gentle"]
    ),
    EffectPreset(
        name="Hard Clipping",
        category=PresetCategory.DISTORTION,
        description="Aggressive hard-clip distortion",
        parameters={
            "drive": 0.8,
            "tone": 0.6,
            "mix": 0.6,
            "type": "hard"
        },
        tags=["hard", "clipping", "aggressive", "grit"]
    ),
    EffectPreset(
        name="Bitcrush",
        category=PresetCategory.DISTORTION,
        description="Lo-fi bitcrushing effect",
        parameters={
            "bits": 4,
            "downsample": 2,
            "mix": 0.5,
            "type": "bitcrush"
        },
        tags=["lo-fi", "bitcrush", "retro", "digital"]
    ),
    EffectPreset(
        name="Overdrive",
        category=PresetCategory.DISTORTION,
        description="Guitar overdrive simulation",
        parameters={
            "drive": 0.5,
            "tone": 0.7,
            "mix": 0.5,
            "type": "overdrive"
        },
        tags=["overdrive", "guitar", "rock", "warm"]
    ),
    EffectPreset(
        name="Waveshaper",
        category=PresetCategory.DISTORTION,
        description="Custom waveshaper distortion",
        parameters={
            "drive": 0.6,
            "mix": 0.5,
            "type": "waveshaper",
            "curve": "sigmoid"
        },
        tags=["synth", "electronic", "aggressive", "sculpt"]
    ),
]


# ==================== FILTER PRESETS ====================

FILTER_PRESETS: List[EffectPreset] = [
    EffectPreset(
        name="Low Pass Smooth",
        category=PresetCategory.FILTER,
        description="Gentle low-pass filter",
        parameters={
            "type": "lowpass",
            "cutoff_hz": 2000,
            "resonance": 0.5,
            "slope": "12db"
        },
        tags=["lowpass", "smooth", "warm", "drums"]
    ),
    EffectPreset(
        name="High Pass Tight",
        category=PresetCategory.FILTER,
        description="Tight high-pass for mix clarity",
        parameters={
            "type": "highpass",
            "cutoff_hz": 80,
            "resonance": 0.3,
            "slope": "12db"
        },
        tags=["highpass", "tight", "clear", "bass"]
    ),
    EffectPreset(
        name="Bandpass Resonant",
        category=PresetCategory.FILTER,
        description="Resonant bandpass for effects",
        parameters={
            "type": "bandpass",
            "cutoff_hz": 1000,
            "resonance": 8.0,
            "slope": "24db"
        },
        tags=["resonant", "mono", "effect", "synth"]
    ),
    EffectPreset(
        name="Notch",
        category=PresetCategory.FILTER,
        description="Notch filter for problem frequencies",
        parameters={
            "type": "notch",
            "cutoff_hz": 400,
            "resonance": 5.0,
            "slope": "24db"
        },
        tags=["notch", "remove", "problem", "surgical"]
    ),
]


# ==================== MODULATION PRESETS ====================

MODULATION_PRESETS: List[EffectPreset] = [
    EffectPreset(
        name="Chorus Light",
        category=PresetCategory.MODULATION,
        description="Light chorus for width",
        parameters={
            "type": "chorus",
            "rate_hz": 1.5,
            "depth": 0.3,
            "mix": 0.25,
            "voices": 3
        },
        tags=["chorus", "width", "subtle", "pad"]
    ),
    EffectPreset(
        name="Flanger Deep",
        category=PresetCategory.MODULATION,
        description="Deep flanger effect",
        parameters={
            "type": "flanger",
            "rate_hz": 0.5,
            "depth": 0.8,
            "mix": 0.5,
            "feedback": 0.6,
            "delay_ms": 3.0
        },
        tags=["flanger", "deep", "jet", "intense"]
    ),
    EffectPreset(
        name="Phaser Sweep",
        category=PresetCategory.MODULATION,
        description="Sweeping phaser effect",
        parameters={
            "type": "phaser",
            "rate_hz": 0.3,
            "depth": 0.7,
            "mix": 0.5,
            "stages": 4,
            "feedback": 0.5
        },
        tags=["phaser", "sweep", "motion", "synth"]
    ),
    EffectPreset(
        name="Tremolo Gentle",
        category=PresetCategory.MODULATION,
        description="Gentle tremolo effect",
        parameters={
            "type": "tremolo",
            "rate_hz": 4.0,
            "depth": 0.3,
            "waveform": "sine",
            "sync_to_bpm": True
        },
        tags=["tremolo", "gentle", "pulse", "vintage"]
    ),
]


# ==================== MASTERING PRESETS ====================

MASTERING_PRESETS: List[EffectPreset] = [
    EffectPreset(
        name="Loud Club",
        category=PresetCategory.MASTERING,
        description="Loud mastering for club play",
        parameters={
            "target_lufs": -6.0,
            "true_peak_max": -1.0,
            " compressor": {
                "threshold_db": -12,
                "ratio": 4.0,
                "attack_ms": 3.0,
                "release_ms": 50.0
            },
            "limiter_threshold_db": -1,
            "eq": {
                "low_shelf_db": 2,
                "high_shelf_db": -1
            }
        },
        tags=["club", "loud", "competitive", "dance"]
    ),
    EffectPreset(
        name="Streaming Clean",
        category=PresetCategory.MASTERING,
        description="Clean mastering for streaming",
        parameters={
            "target_lufs": -14.0,
            "true_peak_max": -1.0,
            "compressor": {
                "threshold_db": -18,
                "ratio": 2.0,
                "attack_ms": 5.0,
                "release_ms": 100.0
            },
            "limiter_threshold_db": -1,
            "eq": {
                "low_shelf_db": 0,
                "high_shelf_db": 0
            }
        },
        tags=["streaming", "clean", "balanced", "spotify"]
    ),
    EffectPreset(
        name="Warm Vinyl",
        category=PresetCategory.MASTERING,
        description="Warm vinyl-style mastering",
        parameters={
            "target_lufs": -12.0,
            "true_peak_max": -2.0,
            "saturation": 0.2,
            "eq": {
                "low_shelf_db": 1,
                "high_shelf_db": -2
            },
            "noise_floor_db": -60
        },
        tags=["vinyl", "warm", "analog", "nostalgic"]
    ),
]


# ==================== EFFECT CHAINS ====================

EFFECT_CHAINS: List[EffectChain] = [
    EffectChain(
        name="Vocal Chain",
        description="Complete vocal processing chain",
        effects=[
            {"effect": "compressor", "parameters": COMPRESSION_PRESETS[2].parameters},
            {"effect": "eq", "parameters": EQ_PRESETS[0].parameters},
            {"effect": "deesser", "parameters": {"threshold_db": -15, "ratio": 4.0, "freq_hz": 7000}},
            {"effect": "reverb", "parameters": REVERB_PRESETS[2].parameters}
        ],
        tags=["vocal", "complete", "processing"],
        genre="pop"
    ),
    EffectChain(
        name="Drum Bus",
        description="Drum bus processing chain",
        effects=[
            {"effect": "compressor", "parameters": COMPRESSION_PRESETS[1].parameters},
            {"effect": "eq", "parameters": {"bands": [{"freq": 80, "gain_db": 2, "q": 0.7, "type": "peaking"}]}},
            {"effect": "reverb", "parameters": REVERB_PRESETS[1].parameters}
        ],
        tags=["drums", "bus", "processing"],
        genre="electronic"
    ),
    EffectChain(
        name="Bass Guitar",
        description="Bass guitar processing",
        effects=[
            {"effect": "compressor", "parameters": COMPRESSION_PRESETS[4].parameters},
            {"effect": "eq", "parameters": EQ_PRESETS[1].parameters},
            {"effect": "saturation", "parameters": DISTORTION_PRESETS[0].parameters},
            {"effect": "limiter", "parameters": {"threshold_db": -3, "ratio": 20.0}}
        ],
        tags=["bass", "guitar", "processing"],
        genre="rock"
    ),
    EffectChain(
        name="Synth Lead",
        description="Synth lead processing for mix presence",
        effects=[
            {"effect": "eq", "parameters": EQ_PRESETS[2].parameters},
            {"effect": "compressor", "parameters": COMPRESSION_PRESETS[0].parameters},
            {"effect": "delay", "parameters": DELAY_PRESETS[1].parameters}
        ],
        tags=["synth", "lead", "presence"],
        genre="electronic"
    ),
    EffectChain(
        name="Lo-Fi Hip Hop",
        description="Vintage lo-fi processing chain",
        effects=[
            {"effect": "bitcrush", "parameters": DISTORTION_PRESETS[2].parameters},
            {"effect": "chorus", "parameters": MODULATION_PRESETS[0].parameters},
            {"effect": "reverb", "parameters": REVERB_PRESETS[4].parameters},
            {"effect": "eq", "parameters": EQ_PRESETS[3].parameters}
        ],
        tags=["lofi", "hiphop", "vintage", "chill"],
        genre="hiphop"
    ),
]


# ==================== GENRE PRESETS ====================

GENRE_PRESETS: Dict[str, EffectChain] = {
    "edm": EffectChain(
        name="EDM Master",
        description="High-energy EDM processing",
        effects=[
            {"effect": "compressor", "parameters": {"threshold_db": -10, "ratio": 4.0, "attack_ms": 2.0, "release_ms": 50.0}},
            {"effect": "eq", "parameters": {"bands": [{"freq": 60, "gain_db": 3, "type": "peaking"}, {"freq": 8000, "gain_db": 2, "type": "high_shelf"}]}},
            {"effect": "limiter", "parameters": {"threshold_db": -1, "ratio": 20.0}},
            {"effect": "loudness", "parameters": {"target_lufs": -6.0}}
        ],
        tags=["edm", "loud", "energy"],
        genre="edm"
    ),
    "rock": EffectChain(
        name="Rock Master",
        description="Powerful rock mix",
        effects=[
            {"effect": "compressor", "parameters": {"threshold_db": -12, "ratio": 3.0, "attack_ms": 5.0, "release_ms": 80.0}},
            {"effect": "eq", "parameters": {"bands": [{"freq": 100, "gain_db": 2, "type": "peaking"}, {"freq": 3000, "gain_db": 1, "type": "peaking"}]}},
            {"effect": "saturation", "parameters": {"drive": 0.3, "mix": 0.4, "type": "tape"}},
            {"effect": "limiter", "parameters": {"threshold_db": -1}}
        ],
        tags=["rock", "power", "guitar"],
        genre="rock"
    ),
    "hip_hop": EffectChain(
        name="Hip Hop Master",
        description="Punchy hip hop mix",
        effects=[
            {"effect": "compressor", "parameters": {"threshold_db": -14, "ratio": 6.0, "attack_ms": 1.0, "release_ms": 60.0}},
            {"effect": "eq", "parameters": {"bands": [{"freq": 80, "gain_db": 4, "type": "peaking"}, {"freq": 300, "gain_db": -2, "type": "peaking"}]}},
            {"effect": "sidechain", "parameters": {"threshold_db": -20, "ratio": 4.0}},
            {"effect": "limiter", "parameters": {"threshold_db": -1}}
        ],
        tags=["hiphop", "punchy", "bass"],
        genre="hiphop"
    ),
    "ambient": EffectChain(
        name="Ambient Master",
        description="Spacious ambient mix",
        effects=[
            {"effect": "reverb", "parameters": REVERB_PRESETS[3].parameters},
            {"effect": "eq", "parameters": {"bands": [{"freq": 200, "gain_db": -2, "type": "peaking"}, {"freq": 8000, "gain_db": 2, "type": "high_shelf"}]}},
            {"effect": "compressor", "parameters": {"threshold_db": -20, "ratio": 2.0, "attack_ms": 20.0, "release_ms": 200.0}},
            {"effect": "limiter", "parameters": {"threshold_db": -3}}
        ],
        tags=["ambient", "space", "ethereal"],
        genre="ambient"
    ),
}


# ==================== PRESET MANAGER ====================

class PresetManager:
    """Manages all effect presets."""
    
    def __init__(self):
        self.reverb = REVERB_PRESETS
        self.delay = DELAY_PRESETS
        self.compression = COMPRESSION_PRESETS
        self.eq = EQ_PRESETS
        self.distortion = DISTORTION_PRESETS
        self.filter = FILTER_PRESETS
        self.modulation = MODULATION_PRESETS
        self.mastering = MASTERING_PRESETS
        self.chains = EFFECT_CHAINS
        self.genre = GENRE_PRESETS
    
    def get_all_presets(self) -> Dict[str, List[EffectPreset]]:
        """Get all presets organized by category."""
        return {
            "reverb": self.reverb,
            "delay": self.delay,
            "compression": self.compression,
            "eq": self.eq,
            "distortion": self.distortion,
            "filter": self.filter,
            "modulation": self.modulation,
            "mastering": self.mastering
        }
    
    def get_by_category(self, category: PresetCategory) -> List[EffectPreset]:
        """Get presets by category."""
        category_map = {
            PresetCategory.REVERB: self.reverb,
            PresetCategory.DELAY: self.delay,
            PresetCategory.COMPRESSION: self.compression,
            PresetCategory.EQ: self.eq,
            PresetCategory.DISTORTION: self.distortion,
            PresetCategory.FILTER: self.filter,
            PresetCategory.MODULATION: self.modulation,
            PresetCategory.MASTERING: self.mastering
        }
        return category_map.get(category, [])
    
    def search(self, query: str) -> List[EffectPreset]:
        """Search presets by name, description, or tags."""
        query = query.lower()
        results = []
        
        for preset in self.get_all_presets().values():
            for p in preset:
                if (query in p.name.lower() or 
                    query in p.description.lower() or
                    any(query in tag.lower() for tag in p.tags)):
                    results.append(p)
        
        return results
    
    def find_chain(self, name: str) -> Optional[EffectChain]:
        """Find effect chain by name."""
        for chain in self.chains:
            if chain.name.lower() == name.lower():
                return chain
        return None
    
    def find_genre(self, genre: str) -> Optional[EffectChain]:
        """Find genre preset by name."""
        return self.genre.get(genre.lower())
    
    def to_json(self) -> str:
        """Export all presets to JSON."""
        data = {
            "presets": {cat: [p.to_dict() for p in presets] 
                       for cat, presets in self.get_all_presets().items()},
            "chains": [c.to_dict() for c in self.chains],
            "genre": {k: v.to_dict() for k, v in self.genre.items()}
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PresetManager':
        """Load presets from JSON (future: implement deserialization)."""
        # For now, just return a fresh manager
        # Full deserialization would rebuild EffectPreset objects
        return cls()


# Default instance
presets = PresetManager()
