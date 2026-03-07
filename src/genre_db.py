"""
Genre Database for AI DJ Project

Contains genre characteristics, BPM ranges, instrumentation,
and transition compatibility for the AI DJ system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class EnergyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BUILDUP = "buildup"
    DROPCENTER = "drop_center"


class Mood(Enum):
    CALM = "calm"
    NEUTRAL = "neutral"
    UPBEAT = "upbeat"
    AGGRESSIVE = "aggressive"
    DARK = "dark"
    EUPHORIC = "euphoric"
    MELANCHOLIC = "melancholic"
    MYSTERIOUS = "mysterious"


@dataclass
class Genre:
    """Represents a music genre with all its characteristics."""
    name: str
    bpm_range: tuple[int, int]
    energy: EnergyLevel
    mood: List[Mood]
    instrumentation: List[str]
    characteristics: List[str]
    common_keys: List[str]
    chord_progressions: List[str]
    typical_duration: tuple[int, int]  # (min_seconds, max_seconds)
    danceability: float  # 0.0 to 1.0
    bass_intensity: float  # 0.0 to 1.0
    vocal_present: float  # 0.0 to 1.0
    similar_genres: List[str] = field(default_factory=list)
    compatible_genres: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


# Genre Database
GENRES: Dict[str, Genre] = {
    # Electronic / EDM Genres
    "house": Genre(
        name="House",
        bpm_range=(118, 130),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.EUPHORIC, Mood.NEUTRAL],
        instrumentation=["4-on-the-floor kick", "bass", "synths", "drums", "pads", "vocals"],
        characteristics=["Four-on-the-floor beat", "steady pulse", "soulful elements", "vocals often processed"],
        common_keys=["C", "G", "D", "A"],
        chord_progressions=["I-IV-V-I", "ii-V-I", "I-V-vi-IV"],
        typical_duration=(180, 420),
        danceability=0.85,
        bass_intensity=0.6,
        vocal_present=0.5,
        similar_genres=["deep_house", "tech_house", "progressive_house"],
        compatible_genres=["techno", "deep_house", "progressive_house", "disco"],
        tags=["electronic", "dance", "4/4", "pulsing"]
    ),
    
    "deep_house": Genre(
        name="Deep House",
        bpm_range=(120, 128),
        energy=EnergyLevel.LOW,
        mood=[Mood.CALM, Mood.NEUTRAL, Mood.MELANCHOLIC],
        instrumentation=["deep kick", "warm bass", "pads", "vocals", "synths", "saxophone"],
        characteristics=["Deep, warm sound", "soulful", "atmospheric", "vocals often present"],
        common_keys=["C", "G", "F", "Bb"],
        chord_progressions=["I-vi-IV-V", "ii-V-I", "vi-IV-I-V"],
        typical_duration=(240, 480),
        danceability=0.75,
        bass_intensity=0.5,
        vocal_present=0.6,
        similar_genres=["house", "nu_disco", "lofi_house"],
        compatible_genres=["house", "lofi_house", "nu_disco", "chillout"],
        tags=["electronic", "soulful", "warm", "atmospheric"]
    ),
    
    "tech_house": Genre(
        name="Tech House",
        bpm_range=(125, 132),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.NEUTRAL, Mood.UPBEAT, Mood.MYSTERIOUS],
        instrumentation=["tight kick", "bass", "synths", "drums", "minimal percussion", "effects"],
        characteristics=["Groovy", "minimal", "repetitive", "hypnotic"],
        common_keys=["E", "A", "D", "G"],
        chord_progressions=["I-IV-V", "i-bVII", "i-bVII-IV"],
        typical_duration=(240, 480),
        danceability=0.8,
        bass_intensity=0.7,
        vocal_present=0.3,
        similar_genres=["house", "techno", "minimal_house"],
        compatible_genres=["techno", "house", "minimal_house", "progressive_house"],
        tags=["electronic", "groovy", "minimal", "hypnotic"]
    ),
    
    "progressive_house": Genre(
        name="Progressive House",
        bpm_range=(126, 132),
        energy=EnergyLevel.BUILDUP,
        mood=[Mood.EUPHORIC, Mood.UPBEAT, Mood.NEUTRAL],
        instrumentation=["big kick", "bass", "synths", "pads", "arps", "plucks"],
        characteristics=["Builds and releases", "emotional", "layered", "long breakdowns"],
        common_keys=["C", "G", "D", "E"],
        chord_progressions=["I-V-vi-IV", "vi-IV-I-V", "I-IV-V-I"],
        typical_duration=(300, 540),
        danceability=0.8,
        bass_intensity=0.65,
        vocal_present=0.4,
        similar_genres=["trance", "house", "electro_house"],
        compatible_genres=["trance", "house", "electro_house", "big_room"],
        tags=["electronic", "emotional", "building", "anthem"]
    ),
    
    "techno": Genre(
        name="Techno",
        bpm_range=(125, 145),
        energy=EnergyLevel.HIGH,
        mood=[Mood.AGGRESSIVE, Mood.DARK, Mood.NEUTRAL, Mood.MYSTERIOUS],
        instrumentation=["kick", "bass", "synths", "drums", "effects", "drones"],
        characteristics=["Repetitive", "driving", "dark", "industrial"],
        common_keys=["E", "A", "D", "B"],
        chord_progressions=["i-bVII-bVI-V", "i-IV-vii-III", "minimal changes"],
        typical_duration=(360, 600),
        danceability=0.75,
        bass_intensity=0.8,
        vocal_present=0.2,
        similar_genres=["minimal_techno", "industrial_techno", "tech_house"],
        compatible_genres=["tech_house", "minimal_techno", "industrial", "detroit_techno"],
        tags=["electronic", "dark", "driving", "industrial", "hypnotic"]
    ),
    
    "minimal_techno": Genre(
        name="Minimal Techno",
        bpm_range=(125, 135),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.CALM, Mood.MYSTERIOUS, Mood.NEUTRAL],
        instrumentation=["kick", "bass", "minimal synths", "drums", "effects"],
        characteristics=["Minimal", "spacey", "repetitive", "subtle changes"],
        common_keys=["E", "A", "B", "F#"],
        chord_progressions=["sparse, often single chords", "i-bVII", "minimal"],
        typical_duration=(360, 720),
        danceability=0.7,
        bass_intensity=0.65,
        vocal_present=0.1,
        similar_genres=["techno", "tech_house", "ambient"],
        compatible_genres=["techno", "tech_house", "ambient", "deep_house"],
        tags=["electronic", "minimal", "spacey", "hypnotic"]
    ),
    
    "trance": Genre(
        name="Trance",
        bpm_range=(138, 145),
        energy=EnergyLevel.HIGH,
        mood=[Mood.EUPHORIC, Mood.UPBEAT, Mood.AGGRESSIVE],
        instrumentation=["kick", "bass", "synths", "pads", "arps", "vocals"],
        characteristics=["Uplifting", "emotional", "melodic", "builds to climax"],
        common_keys=["C", "G", "D", "E", "A"],
        chord_progressions=["I-V-vi-IV", "vi-IV-I-V", "I-IV-V-I"],
        typical_duration=(300, 540),
        danceability=0.9,
        bass_intensity=0.7,
        vocal_present=0.4,
        similar_genres=["progressive_trance", "psytrance", "uplifting_trance"],
        compatible_genres=["progressive_house", "psytrance", "big_room", "hardstyle"],
        tags=["electronic", "uplifting", "melodic", "emotional", "anthem"]
    ),
    
    "psytrance": Genre(
        name="Psytrance",
        bpm_range=(140, 150),
        energy=EnergyLevel.HIGH,
        mood=[Mood.AGGRESSIVE, Mood.EUPHORIC],
        instrumentation=["kick", "bass", "synths", "effects", "drums"],
        characteristics=["Psychedelic", "complex", "fast", "spiraling synths"],
        common_keys=["E", "A", "B", "F#"],
        chord_progressions=["i-bVII-bVI-V", "i-IV-vii-III", "complex modulations"],
        typical_duration=(240, 480),
        danceability=0.85,
        bass_intensity=0.9,
        vocal_present=0.15,
        similar_genres=["trance", "progressive_psytrance", "hitech"],
        compatible_genres=["trance", "progressive_house", "dark_psytrance"],
        tags=["electronic", "psychedelic", "fast", "complex"]
    ),
    
    "dubstep": Genre(
        name="Dubstep",
        bpm_range=(138, 142),
        energy=EnergyLevel.DROPCENTER,
        mood=[Mood.DARK, Mood.AGGRESSIVE, Mood.MYSTERIOUS],
        instrumentation=["wobble bass", "drums", "synths", "effects", "vocals"],
        characteristics=["Wobble bass", "drop-centric", "half-time feel", "heavy"],
        common_keys=["C#", "F#", "G#", "D#"],
        chord_progressions=["i-bVII-bVI-V", "i-IV-bVII-V", "minimal"],
        typical_duration=(180, 300),
        danceability=0.7,
        bass_intensity=0.95,
        vocal_present=0.35,
        similar_genres=["riddim", "future_bass"],
        compatible_genres=["drum_and_bass", "future_bass", "riddim", "trap"],
        tags=["electronic", "bass_heavy", "wobble", "dark"]
    ),
    
    "drum_and_bass": Genre(
        name="Drum & Bass",
        bpm_range=(170, 180),
        energy=EnergyLevel.HIGH,
        mood=[Mood.AGGRESSIVE, Mood.UPBEAT, Mood.NEUTRAL],
        instrumentation=["breakbeat", "bass", "synths", "drums"],
        characteristics=["Fast", "breakbeat rhythms", "heavy bass", "energetic"],
        common_keys=["C", "F", "G", "Bb"],
        chord_progressions=["I-IV-V-I", "ii-V-I", "I-V-vi-IV"],
        typical_duration=(180, 300),
        danceability=0.9,
        bass_intensity=0.85,
        vocal_present=0.3,
        similar_genres=["liquid_dnb", "neurofunk", "jungle"],
        compatible_genres=["jungle", "liquid_dnb", "dubstep", "breakbeat"],
        tags=["electronic", "fast", "breakbeat", "energetic"]
    ),
    
    "liquid_dnb": Genre(
        name="Liquid Drum & Bass",
        bpm_range=(170, 180),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.CALM, Mood.MELANCHOLIC, Mood.UPBEAT],
        instrumentation=["breakbeat", "bass", "pads", "synths", "vocals"],
        characteristics=["Smooth", "melodic", "soulful", "relaxed"],
        common_keys=["C", "G", "F", "Bb"],
        chord_progressions=["I-vi-IV-V", "ii-V-I", "vi-IV-I-V"],
        typical_duration=(180, 360),
        danceability=0.75,
        bass_intensity=0.6,
        vocal_present=0.5,
        similar_genres=["drum_and_bass", "chillout", "neurofunk"],
        compatible_genres=["drum_and_bass", "chillout", "deep_house", "lofi"],
        tags=["electronic", "smooth", "melodic", "relaxed"]
    ),
    
    "electro_house": Genre(
        name="Electro House",
        bpm_range=(126, 134),
        energy=EnergyLevel.HIGH,
        mood=[Mood.UPBEAT, Mood.AGGRESSIVE, Mood.EUPHORIC],
        instrumentation=["hard kick", "bass", "synths", "leads", "effects"],
        characteristics=["Big drops", "gritty", "energetic", "four-on-the-floor"],
        common_keys=["E", "A", "D", "G"],
        chord_progressions=["I-V-vi-IV", "i-bVII-bVI-V", "I-IV-V-I"],
        typical_duration=(180, 360),
        danceability=0.85,
        bass_intensity=0.85,
        vocal_present=0.3,
        similar_genres=["big_room", "progressive_house", "future_house"],
        compatible_genres=["big_room", "progressive_house", "dubstep", "trance"],
        tags=["electronic", "big_drop", "gritty", "energetic"]
    ),
    
    "big_room": Genre(
        name="Big Room House",
        bpm_range=(126, 132),
        energy=EnergyLevel.DROPCENTER,
        mood=[Mood.EUPHORIC, Mood.UPBEAT],
        instrumentation=["big kick", "bass", "synths", "leads", "effects"],
        characteristics=["Massive drops", "simple structure", "festival-ready", "anthemic"],
        common_keys=["C", "E", "A", "D"],
        chord_progressions=["I-V-vi-IV", "I-IV-V-I", "vi-IV-I-V"],
        typical_duration=(180, 300),
        danceability=0.85,
        bass_intensity=0.9,
        vocal_present=0.25,
        similar_genres=["electro_house", "progressive_house", "future_bass"],
        compatible_genres=["electro_house", "progressive_house", "trance", "future_bass"],
        tags=["electronic", "festival", "anthem", "big_drop"]
    ),
    
    "future_bass": Genre(
        name="Future Bass",
        bpm_range=(140, 150),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.EUPHORIC, Mood.UPBEAT, Mood.MYSTERIOUS],
        instrumentation=["wobble bass", "chimey synths", "drums", "vocals", "pads"],
        characteristics=["Wobble synths", "emotional", "bright", "layered"],
        common_keys=["C", "F", "G", "Bb"],
        chord_progressions=["I-V-vi-IV", "vi-IV-I-V", "I-IV-V-I"],
        typical_duration=(180, 300),
        danceability=0.8,
        bass_intensity=0.7,
        vocal_present=0.5,
        similar_genres=["dubstep", "trap", "chilltrap"],
        compatible_genres=["dubstep", "trap", "electro_house", "chillout"],
        tags=["electronic", "emotional", "bright", "wobble"]
    ),
    
    "trap": Genre(
        name="Trap",
        bpm_range=(140, 180),
        energy=EnergyLevel.HIGH,
        mood=[Mood.AGGRESSIVE, Mood.DARK, Mood.UPBEAT],
        instrumentation=["808", "hi-hats", "snare", "bass", "synths", "vocals"],
        characteristics=["808 drums", "fast hi-hats", "drops", "Southern influence"],
        common_keys=["C#", "F#", "G#", "D#"],
        chord_progressions=["i-bVII-bVI-V", "i-IV-i-V", "minimal"],
        typical_duration=(180, 300),
        danceability=0.75,
        bass_intensity=0.9,
        vocal_present=0.55,
        similar_genres=["hip_hop", "future_bass", "dubstep"],
        compatible_genres=["hip_hop", "dubstep", "future_bass", "riddim"],
        tags=["hip_hop_electronic", "808", "aggressive", "dark"]
    ),
    
    # Hip Hop / Rap
    "hip_hop": Genre(
        name="Hip Hop",
        bpm_range=(80, 110),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.NEUTRAL, Mood.AGGRESSIVE],
        instrumentation=["drums", "bass", "samples", "synths", "vocals"],
        characteristics=["Boom-bap rhythm", "samples", "rapped vocals", "groovy"],
        common_keys=["C", "F", "G", "Bb"],
        chord_progressions=["I-IV-V-I", "ii-V-I", "I-vi-IV-V"],
        typical_duration=(180, 300),
        danceability=0.75,
        bass_intensity=0.6,
        vocal_present=0.9,
        similar_genres=["trap", "lofi_hip_hop", "gangsta_rap"],
        compatible_genres=["rnb", "trap", "lofi_hip_hop", "disco"],
        tags=["hip_hop", "rap", "samples", "groove"]
    ),
    
    "lofi_hip_hop": Genre(
        name="Lo-Fi Hip Hop",
        bpm_range=(70, 90),
        energy=EnergyLevel.LOW,
        mood=[Mood.CALM, Mood.MELANCHOLIC, Mood.NEUTRAL],
        instrumentation=["drums", "bass", "samples", "piano", "vinyl_effects"],
        characteristics=["Chill", "grainy", "relaxed", "sample-based"],
        common_keys=["C", "F", "G", "Bb"],
        chord_progressions=["I-vi-IV-V", "I-IV-V-I", "vi-IV-I-V"],
        typical_duration=(120, 240),
        danceability=0.5,
        bass_intensity=0.4,
        vocal_present=0.4,
        similar_genres=["chillhop", "instrumental_hip_hop", "jazzy_hip_hop"],
        compatible_genres=["chillout", "jazz", "ambient", "deep_house"],
        tags=["hip_hop", "chill", "relaxed", "sample_based"]
    ),
    
    # Pop / Commercial
    "pop": Genre(
        name="Pop",
        bpm_range=(100, 128),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.EUPHORIC, Mood.NEUTRAL],
        instrumentation=["drums", "bass", "synths", "vocals", "guitar", "piano"],
        characteristics=["Catchy", "accessible", "melodic", "vocals-forward"],
        common_keys=["C", "G", "D", "A", "E"],
        chord_progressions=["I-V-vi-IV", "vi-IV-I-V", "I-IV-V-I"],
        typical_duration=(180, 240),
        danceability=0.8,
        bass_intensity=0.5,
        vocal_present=0.85,
        similar_genres=["dance_pop", "synth_pop", "electropop"],
        compatible_genres=["dance", "rnb", "electronic", "disco"],
        tags=["pop", "mainstream", "catchy", "vocals"]
    ),
    
    "dance_pop": Genre(
        name="Dance Pop",
        bpm_range=(118, 128),
        energy=EnergyLevel.HIGH,
        mood=[Mood.UPBEAT, Mood.EUPHORIC],
        instrumentation=["drums", "synths", "bass", "vocals", "effects"],
        characteristics=["Danceable", "electronic", "upbeat", "radio-friendly"],
        common_keys=["C", "G", "D", "A"],
        chord_progressions=["I-V-vi-IV", "I-IV-V-I", "vi-IV-I-V"],
        typical_duration=(180, 240),
        danceability=0.9,
        bass_intensity=0.6,
        vocal_present=0.8,
        similar_genres=["pop", "house", "edm"],
        compatible_genres=["house", "pop", "electro_house", "trance"],
        tags=["pop", "dance", "electronic", "energetic"]
    ),
    
    # R&B
    "rnb": Genre(
        name="R&B",
        bpm_range=(80, 110),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.CALM, Mood.NEUTRAL, Mood.UPBEAT],
        instrumentation=["drums", "bass", "piano", "synths", "vocals", "guitar"],
        characteristics=["Smooth", "vocals-focused", "groovy", "emotional"],
        common_keys=["C", "G", "F", "Bb", "Eb"],
        chord_progressions=["I-IV-V-I", "ii-V-I", "I-vi-IV-V"],
        typical_duration=(180, 300),
        danceability=0.7,
        bass_intensity=0.5,
        vocal_present=0.9,
        similar_genres=["soul", "hip_hop", "contemporary_rnb"],
        compatible_genres=["hip_hop", "pop", "soul", "dance"],
        tags=["rnb", "smooth", "vocals", "groove"]
    ),
    
    # Rock
    "rock": Genre(
        name="Rock",
        bpm_range=(100, 160),
        energy=EnergyLevel.HIGH,
        mood=[Mood.AGGRESSIVE, Mood.UPBEAT, Mood.NEUTRAL],
        instrumentation=["guitar", "drums", "bass", "vocals"],
        characteristics=["Guitar-driven", "energetic", "live feel", "powerful"],
        common_keys=["E", "A", "G", "D", "C"],
        chord_progressions=["I-IV-V", "I-V-IV", "i-bVII-bVI-V", "ii-V-I"],
        typical_duration=(180, 360),
        danceability=0.7,
        bass_intensity=0.7,
        vocal_present=0.8,
        similar_genres=["alternative_rock", "hard_rock", "indie_rock"],
        compatible_genres=["pop", "metal", "electronic", "punk"],
        tags=["rock", "guitar", "energetic", "live"]
    ),
    
    "indie_rock": Genre(
        name="Indie Rock",
        bpm_range=(100, 140),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.NEUTRAL, Mood.MELANCHOLIC],
        instrumentation=["guitar", "drums", "bass", "vocals", "synths"],
        characteristics=["Alternative", "eclectic", "experimental", "guitar-forward"],
        common_keys=["G", "C", "D", "E", "A"],
        chord_progressions=["I-IV-V", "I-V-vi-IV", "i-bVII-bVI-V"],
        typical_duration=(180, 300),
        danceability=0.65,
        bass_intensity=0.55,
        vocal_present=0.75,
        similar_genres=["rock", "alternative", "dream_pop"],
        compatible_genres=["rock", "pop", "electronic", "shoegaze"],
        tags=["rock", "indie", "alternative", "guitar"]
    ),
    
    # Metal
    "metal": Genre(
        name="Metal",
        bpm_range=(100, 200),
        energy=EnergyLevel.HIGH,
        mood=[Mood.AGGRESSIVE, Mood.DARK],
        instrumentation=["guitar", "drums", "bass", "vocals"],
        characteristics=["Heavy", "distorted", "aggressive", "powerful"],
        common_keys=["E", "A", "D", "G", "C"],
        chord_progressions=["i-bVI-bVII-i", "i-IV-i-V", "i-bVII-IV"],
        typical_duration=(240, 480),
        danceability=0.5,
        bass_intensity=0.9,
        vocal_present=0.7,
        similar_genres=["death_metal", "thrash_metal", "black_metal"],
        compatible_genres=["rock", "industrial", "hardcore"],
        tags=["metal", "heavy", "aggressive", "distorted"]
    ),
    
    # Jazz
    "jazz": Genre(
        name="Jazz",
        bpm_range=(80, 140),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.CALM, Mood.NEUTRAL, Mood.UPBEAT],
        instrumentation=["piano", "drums", "bass", "saxophone", "trumpet", "vocals"],
        characteristics=["Improvisation", "complex chords", "swing", "live"],
        common_keys=["C", "F", "Bb", "G", "D"],
        chord_progressions=["ii-V-I", "I-vi-ii-V", "I-IV-vii-iii-VI-ii-V-I"],
        typical_duration=(240, 480),
        danceability=0.6,
        bass_intensity=0.5,
        vocal_present=0.5,
        similar_genres=["swing", "bossa_nova", "cool_jazz"],
        compatible_genres=["soul", "rnb", "lofi_hip_hop", "chillout"],
        tags=["jazz", "improvisation", "live", "sophisticated"]
    ),
    
    # Classical / Orchestral
    "classical": Genre(
        name="Classical",
        bpm_range=(60, 120),
        energy=EnergyLevel.LOW,
        mood=[Mood.CALM, Mood.NEUTRAL, Mood.EUPHORIC, Mood.MELANCHOLIC],
        instrumentation=["orchestra", "piano", "strings", "winds", "brass"],
        characteristics=["Orchestral", "complex", "emotional", "formal structure"],
        common_keys=["C", "G", "D", "F", "Bb"],
        chord_progressions=["complex classical progressions", "sonata form", "rondo"],
        typical_duration=(300, 900),
        danceability=0.3,
        bass_intensity=0.4,
        vocal_present=0.3,
        similar_genres=["orchestral", "film_score", "neoclassical"],
        compatible_genres=["ambient", "piano", "film_score", "new_age"],
        tags=["classical", "orchestral", "sophisticated", "emotional"]
    ),
    
    # Ambient / Chill
    "ambient": Genre(
        name="Ambient",
        bpm_range=(60, 120),
        energy=EnergyLevel.LOW,
        mood=[Mood.CALM, Mood.MYSTERIOUS, Mood.MELANCHOLIC],
        instrumentation=["pads", "drones", "field_recordings", "synths", "effects"],
        characteristics=["Atmospheric", "textural", "minimal rhythm", "evocative"],
        common_keys=["any", "often atonal"],
        chord_progressions=["minimal or atonal", "sustained chords"],
        typical_duration=(300, 900),
        danceability=0.2,
        bass_intensity=0.3,
        vocal_present=0.2,
        similar_genres=["chillout", "experimental", "soundscape"],
        compatible_genres=["classical", "chillout", "meditation", "film_score"],
        tags=["ambient", "atmospheric", "textural", "minimal"]
    ),
    
    "chillout": Genre(
        name="Chillout",
        bpm_range=(80, 110),
        energy=EnergyLevel.LOW,
        mood=[Mood.CALM, Mood.NEUTRAL],
        instrumentation=["pads", "drums", "bass", "synths", "vocals"],
        characteristics=["Relaxing", "smooth", "downtempo", "accessible ambient"],
        common_keys=["C", "G", "F", "Bb"],
        chord_progressions=["I-vi-IV-V", "I-IV-V-I", "ii-V-I"],
        typical_duration=(240, 480),
        danceability=0.45,
        bass_intensity=0.4,
        vocal_present=0.4,
        similar_genres=["lounge", "downtempo", "ambient"],
        compatible_genres=["deep_house", "ambient", "lofi_hip_hop", "jazz"],
        tags=["chill", "relaxed", "downtempo", "smooth"]
    ),
    
    # Disco / Funk
    "disco": Genre(
        name="Disco",
        bpm_range=(115, 130),
        energy=EnergyLevel.HIGH,
        mood=[Mood.UPBEAT, Mood.EUPHORIC],
        instrumentation=["drums", "bass", "guitar", "strings", "vocals", "horns"],
        characteristics=["Four-on-the-floor", "groovy", "uplifting", "string sections"],
        common_keys=["C", "G", "F", "Bb"],
        chord_progressions=["I-IV-V-I", "ii-V-I", "I-vi-IV-V"],
        typical_duration=(180, 360),
        danceability=0.95,
        bass_intensity=0.65,
        vocal_present=0.75,
        similar_genres=["nu_disco", "funk", "soul"],
        compatible_genres=["house", "funk", "pop", "soul"],
        tags=["disco", "groove", "uplifting", "70s"]
    ),
    
    "funk": Genre(
        name="Funk",
        bpm_range=(90, 120),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.NEUTRAL],
        instrumentation=["drums", "bass", "guitar", "horns", "vocals", "keyboards"],
        characteristics=["Groovy", "syncopated", "bass-heavy", "improvisation"],
        common_keys=["F", "Bb", "G", "C"],
        chord_progressions=["I-IV-V-I", "i-IV-i-V", "ii-V-I"],
        typical_duration=(180, 360),
        danceability=0.9,
        bass_intensity=0.8,
        vocal_present=0.7,
        similar_genres=["disco", "soul", "rnb", "jazz_funk"],
        compatible_genres=["disco", "soul", "rnb", "house"],
        tags=["funk", "groove", "bass_heavy", "syncopated"]
    ),
    
    # Latin
    "latin": Genre(
        name="Latin",
        bpm_range=(85, 120),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.EUPHORIC],
        instrumentation=["percussion", "bass", "guitar", "horns", "vocals"],
        characteristics=["Rhythmic", "danceable", "regional styles", "percussion-driven"],
        common_keys=["C", "G", "D", "A"],
        chord_progressions=["I-IV-V-I", "ii-V-I", "i-bVII-bVI"],
        typical_duration=(180, 360),
        danceability=0.9,
        bass_intensity=0.6,
        vocal_present=0.7,
        similar_genres=["salsa", "reggaeton", "cumbia", "bossa_nova"],
        compatible_genres=["house", "pop", "tropical_house", "reggaeton"],
        tags=["latin", "rhythmic", "danceable", "percussion"]
    ),
    
    "reggaeton": Genre(
        name="Reggaeton",
        bpm_range=(85, 100),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.AGGRESSIVE],
        instrumentation=["drums", "bass", "synths", "vocals", "effects"],
        characteristics=["Dem bow rhythm", "Spanish vocals", "urban", "danceable"],
        common_keys=["C", "F", "G", "Bb"],
        chord_progressions=["I-bVII-IV-I", "i-bVII-bVI-V", "I-IV-V-I"],
        typical_duration=(180, 300),
        danceability=0.9,
        bass_intensity=0.75,
        vocal_present=0.85,
        similar_genres=["latin_pop", "trap_latino", "dancehall"],
        compatible_genres=["latin", "dancehall", "trap", "latin_pop"],
        tags=["reggaeton", "urban", "latin", "danceable"]
    ),
    
    # World / Regional
    "afrobeats": Genre(
        name="Afrobeats",
        bpm_range=(90, 115),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.EUPHORIC],
        instrumentation=["drums", "bass", "guitar", "synths", "vocals", "percussion"],
        characteristics=["Fela Kuti influence", "groovy", "modern African", "danceable"],
        common_keys=["C", "G", "F", "Bb"],
        chord_progressions=["I-IV-V-I", "I-vi-IV-V", "ii-V-I"],
        typical_duration=(180, 300),
        danceability=0.9,
        bass_intensity=0.65,
        vocal_present=0.8,
        similar_genres=["afrobeat", "amapiano", "dancehall"],
        compatible_genres=["dancehall", "hip_hop", "latin", "pop"],
        tags=["afrobeats", "african", "groove", "modern"]
    ),
    
    "amapiano": Genre(
        name="Amapiano",
        bpm_range=(100, 120),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.EUPHORIC],
        instrumentation=["piano", "drums", "bass", "synths", "vocals", "log_drum"],
        characteristics=["South African house", "piano-driven", "deep", "log drum"],
        common_keys=["C", "G", "F", "Bb"],
        chord_progressions=["I-IV-V-I", "i-IV-i-V", "I-vi-IV-V"],
        typical_duration=(240, 480),
        danceability=0.85,
        bass_intensity=0.7,
        vocal_present=0.5,
        similar_genres=["deep_house", "afrobeats", "afrohouse"],
        compatible_genres=["afrobeats", "deep_house", "house", "latin"],
        tags=["amapiano", "south_african", "piano", "groove"]
    ),
    
    # Other / Cross-genre
    "soundtrack": Genre(
        name="Soundtrack / Film Score",
        bpm_range=(60, 180),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.CALM, Mood.NEUTRAL, Mood.EUPHORIC, Mood.DARK, Mood.MELANCHOLIC],
        instrumentation=["orchestra", "synths", "electronics", "voices"],
        characteristics=["Narrative-driven", "emotional", "cinematic", "dynamic"],
        common_keys=["any", "often minor keys"],
        chord_progressions=["variable", "depends on scene"],
        typical_duration=(120, 600),
        danceability=0.3,
        bass_intensity=0.5,
        vocal_present=0.3,
        similar_genres=["classical", "ambient", "video_game_music"],
        compatible_genres=["classical", "ambient", "electronic", "pop"],
        tags=["soundtrack", "cinematic", "emotional", "orchestral"]
    ),
    
    "video_game_music": Genre(
        name="Video Game Music",
        bpm_range=(80, 180),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.UPBEAT, Mood.NEUTRAL, Mood.AGGRESSIVE, Mood.DARK, Mood.EUPHORIC],
        instrumentation=["synths", "orchestra", "chiptune", "electronics", "guitar"],
        characteristics=["Dynamic", "level-driven", "repetitive loops", "emotional"],
        common_keys=["C", "G", "D", "E", "F", "A", "B"],
        chord_progressions=["I-IV-V-I", "I-V-vi-IV", "i-bVII-bVI-V", "vi-IV-I-V"],
        typical_duration=(60, 300),
        danceability=0.5,
        bass_intensity=0.5,
        vocal_present=0.2,
        similar_genres=["soundtrack", "synthwave", "chiptune"],
        compatible_genres=["soundtrack", "synthwave", "electronic", "pop"],
        tags=["video_game", "chiptune", "8bit", "nostalgic", "dynamic"]
    ),
    
    "synthwave": Genre(
        name="Synthwave",
        bpm_range=(90, 128),
        energy=EnergyLevel.MEDIUM,
        mood=[Mood.NEUTRAL, Mood.UPBEAT, Mood.MYSTERIOUS, Mood.DARK],
        instrumentation=["synths", "drums", "bass", "guitar", "vocals"],
        characteristics=["Retro-futuristic", "80s aesthetic", "neon", "driving rhythms"],
        common_keys=["Am", "Em", "F", "C", "G", "D"],
        chord_progressions=["i-bVII-bVI-V", "I-IV-V-I", "i-bVII-IV"],
        typical_duration=(180, 360),
        danceability=0.7,
        bass_intensity=0.65,
        vocal_present=0.4,
        similar_genres=["video_game_music", "retrowave", "outrun"],
        compatible_genres=["video_game_music", "electronic", "pop", "house"],
        tags=["synthwave", "retro", "80s", "neon", "nostalgic"]
    ),
}


# Helper functions for genre operations

def get_genre(genre_key: str) -> Optional[Genre]:
    """Get a genre by its key name."""
    return GENRES.get(genre_key.lower().replace(" ", "_"))


def get_genres_by_bpm(bpm: float) -> List[Genre]:
    """Find all genres that match a given BPM."""
    return [g for g in GENRES.values() if g.bpm_range[0] <= bpm <= g.bpm_range[1]]


def get_genres_by_energy(energy: EnergyLevel) -> List[Genre]:
    """Find all genres with a specific energy level."""
    return [g for g in GENRES.values() if g.energy == energy]


def get_genres_by_mood(mood: Mood) -> List[Genre]:
    """Find all genres that contain a specific mood."""
    return [g for g in GENRES.values() if mood in g.mood]


def get_compatible_genres(genre_key: str) -> List[Genre]:
    """Get genres compatible with a given genre for transitions."""
    genre = get_genre(genre_key)
    if not genre:
        return []
    return [GENRES[g] for g in genre.compatible_genres if g in GENRES]


def get_similar_genres(genre_key: str) -> List[Genre]:
    """Get similar genres to a given genre."""
    genre = get_genre(genre_key)
    if not genre:
        return []
    return [GENRES[g] for g in genre.similar_genres if g in GENRES]


def search_genres_by_tag(tag: str) -> List[Genre]:
    """Search genres by tag."""
    return [g for g in GENRES.values() if tag.lower() in [t.lower() for t in g.tags]]


def search_genres_by_instrument(instrument: str) -> List[Genre]:
    """Search genres by instrumentation."""
    return [g for g in GENRES.values() if instrument.lower() in [i.lower() for i in g.instrumentation]]


def get_all_genre_names() -> List[str]:
    """Get a list of all genre names."""
    return list(GENRES.keys())


def get_all_genre_display_names() -> List[str]:
    """Get a list of all genre display names."""
    return [g.name for g in GENRES.values()]


def calculate_transition_compatibility(genre1_key: str, genre2_key: str) -> float:
    """
    Calculate transition compatibility score between two genres (0.0 to 1.0).
    
    Factors:
    - BPM difference (closer = better)
    - Energy level compatibility
    - Mood overlap
    - Key compatibility (circle of fifths)
    - Danceability compatibility
    """
    g1 = get_genre(genre1_key)
    g2 = get_genre(genre2_key)
    
    if not g1 or not g2:
        return 0.0
    
    score = 0.0
    
    # BPM compatibility (40% weight)
    bpm_mid1 = (g1.bpm_range[0] + g1.bpm_range[1]) / 2
    bpm_mid2 = (g2.bpm_range[0] + g2.bpm_range[1]) / 2
    bpm_diff = abs(bpm_mid1 - bpm_mid2)
    bpm_score = max(0, 1 - (bpm_diff / 50))  # Max 50 BPM difference
    score += bpm_score * 0.4
    
    # Energy compatibility (25% weight)
    energy_diff = abs(list(EnergyLevel).index(g1.energy) - list(EnergyLevel).index(g2.energy))
    energy_score = max(0, 1 - (energy_diff / 4))
    score += energy_score * 0.25
    
    # Mood overlap (15% weight)
    mood_overlap = len(set(g1.mood) & set(g2.mood)) / max(len(g1.mood), len(g2.mood), 1)
    score += mood_overlap * 0.15
    
    # Danceability compatibility (10% weight)
    dance_diff = abs(g1.danceability - g2.danceability)
    dance_score = 1 - dance_diff
    score += dance_score * 0.1
    
    # Explicit compatibility check (10% weight)
    if genre2_key in g1.compatible_genres or genre1_key in g2.compatible_genres:
        score += 0.1
    
    return min(1.0, score)


def suggest_genre_sequence(start_genre: str, num_genres: int = 5) -> List[str]:
    """
    Suggest a genre sequence for a DJ set starting from a given genre.
    Returns a list of genre keys in suggested order.
    """
    current = start_genre
    sequence = [current]
    used = {current}
    
    for _ in range(num_genres - 1):
        compatible = get_compatible_genres(current)
        # Filter out already used genres
        candidates = [g for g in compatible if g.name not in used]
        
        if not candidates:
            # Fall back to similar genres
            similar = get_similar_genres(current)
            candidates = [g for g in similar if g.name not in used]
        
        if not candidates:
            # Fall back to any genre not used
            candidates = [g for g in GENRES.values() if g.name not in used]
        
        if not candidates:
            break
        
        # Score all candidates and pick the best
        best_score = -1
        best_genre = None
        
        for candidate in candidates:
            if len(sequence) > 1:
                prev_genre = sequence[-2]
                score = calculate_transition_compatibility(prev_genre, candidate.name)
            else:
                score = calculate_transition_compatibility(current, candidate.name)
            
            if score > best_score:
                best_score = score
                best_genre = candidate
        
        if best_genre:
            sequence.append(best_genre.name.lower().replace(" ", "_"))
            used.add(best_genre.name)
            current = best_genre.name.lower().replace(" ", "_")
    
    return sequence


# Example usage
if __name__ == "__main__":
    print("=== Genre Database ===")
    print(f"Total genres: {len(GENRES)}")
    print()
    
    # List all genres
    print("Available genres:")
    for key, genre in sorted(GENRES.items()):
        print(f"  - {key}: {genre.name} ({genre.bpm_range[0]}-{genre.bpm_range[1]} BPM)")
    
    print()
    
    # Example: Check house
    house = get_genre("house")
    if house:
        print(f"=== {house.name} ===")
        print(f"BPM Range: {house.bpm_range[0]}-{house.bpm_range[1]}")
        print(f"Energy: {house.energy.value}")
        print(f"Moods: {[m.value for m in house.mood]}")
        print(f"Instrumentation: {', '.join(house.instrumentation)}")
        print(f"Characteristics: {', '.join(house.characteristics)}")
        print(f"Danceability: {house.danceability}")
        print(f"Compatible genres: {', '.join(house.compatible_genres)}")
        print()
    
    # Example: Transition compatibility
    print("=== Transition Examples ===")
    print(f"House -> Techno: {calculate_transition_compatibility('house', 'techno'):.2f}")
    print(f"House -> Trance: {calculate_transition_compatibility('house', 'trance'):.2f}")
    print(f"Dubstep -> Drum and Bass: {calculate_transition_compatibility('dubstep', 'drum_and_bass'):.2f}")
    print(f"Lo-fi Hip Hop -> Ambient: {calculate_transition_compatibility('lofi_hip_hop', 'ambient'):.2f}")