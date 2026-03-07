#!/usr/bin/env python3
"""
AI DJ Orchestrator
==================
The brain that coordinates all music generation components:
1. Arrangement Generator - Creates song structure
2. Drum Generator - Generates rhythmic patterns
3. Bass Generator - Creates bass lines
4. Chord Generator - Generates chord progressions
5. Melody Generator - Creates melodic content
6. Effects Processor - Applies audio effects
7. Auto Master - Professional mastering

Usage:
    orchestrator = Orchestrator()
    song = orchestrator.create_song(
        genre="house",
        key="C",
        bpm=128,
        duration=180
    )
    orchestrator.export(song, "my_song.wav")
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import random

# Import all generators
from arrangement_generator import ArrangementGenerator, SectionType
from drum_generator import DrumPattern
from bass_generator import BassGenerator
from chord_generator import ChordProgression
from melody_generator import MelodyGenerator, ScaleType, HookGenerator

# Import effects and mastering
try:
    from effects_processor import EffectsProcessor
except ImportError:
    EffectsProcessor = None

try:
    from auto_master import AutoMaster, StreamingPlatforms
except ImportError:
    AutoMaster = None
    StreamingPlatforms = None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SongConfig:
    """Configuration for song generation."""
    genre: str = "house"
    key: str = "C"
    scale: str = "minor"
    bpm: int = 128
    duration_sec: int = 180
    title: str = ""
    artist: str = "AI DJ"
    energy: float = 0.8
    mood: str = "energetic"
    
    # Effects settings
    reverb: float = 0.3
    delay: float = 0.2
    chorus: float = 0.1
    compression: float = 0.5
    
    # Mastering settings
    mastering_target: str = "spotify"  # spotify, apple, youtube, etc.
    true_peak: float = -1.0
    
    def __post_init__(self):
        if not self.title:
            self.title = f"AI Generated {self.genre.title()}"


@dataclass
class GeneratedTrack:
    """Container for a generated track with all its components."""
    config: SongConfig
    arrangement: List[dict] = field(default_factory=list)
    drum_patterns: Dict[str, str] = field(default_factory=dict)
    bass_lines: Dict[str, list] = field(default_factory=dict)
    chord_progressions: Dict[str, list] = field(default_factory=list)
    melodies: Dict[str, list] = field(default_factory=dict)
    audio_mix: Optional[np.ndarray] = None
    audio_mastered: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class Orchestrator:
    """
    AI DJ Orchestrator - Coordinates all music generation components.
    
    This is the main brain that:
    1. Generates song arrangements
    2. Creates drum patterns for each section
    3. Generates bass lines
    4. Creates chord progressions
    5. Produces melodies and hooks
    6. Applies effects processing
    7. Runs auto-mastering
    8. Exports final audio
    """
    
    # Genre to scale mapping
    GENRE_SCALES = {
        "house": "minor",
        "techno": "minor",
        "trance": "minor",
        "dubstep": "minor",
        "pop": "major",
        "hip_hop": "minor",
        "trap": "minor",
        "rnb": "minor",
        "jazz": "major",
        "rock": "major",
        "edm": "minor",
    }
    
    # Genre to drum pattern mapping
    GENRE_DRUM_MAP = {
        "house": "house",
        "techno": "techno",
        "trance": "techno",
        "dubstep": "dubstep",
        "pop": "house",
        "hip_hop": "hip_hop",
        "trap": "trap",
        "rnb": "hip_hop",
        "edm": "house",
    }
    
    def __init__(self, sample_rate: int = 44100, output_dir: str = "./output"):
        """
        Initialize the orchestrator.
        
        Args:
            sample_rate: Audio sample rate (default 44100)
            output_dir: Directory for exported audio files
        """
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all generators
        self._init_generators()
        
        # Stats tracking
        self.stats = {
            "songs_created": 0,
            "total_duration": 0.0,
            "genres_used": {},
        }
        
        print(f"🎛️ AI DJ Orchestrator initialized")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Output Dir: {self.output_dir}")
    
    def _init_generators(self):
        """Initialize all component generators."""
        # Core generators
        self.arrangement_gen = ArrangementGenerator(bpm=120)
        self.drum_gen = DrumPattern(bpm=120)
        self.bass_gen = BassGenerator(key="C", scale="minor")
        self.chord_gen = ChordProgression(key="C")
        
        # Melody generator (will be configured per song)
        self.melody_gen = None
        self.hook_gen = None
        
        # Effects and mastering
        self.effects = EffectsProcessor(sample_rate=self.sample_rate) if EffectsProcessor else None
        self.master = AutoMaster(sample_rate=self.sample_rate) if AutoMaster else None
        
        print(f"   ✓ Arrangement Generator")
        print(f"   ✓ Drum Generator")
        print(f"   ✓ Bass Generator")
        print(f"   ✓ Chord Generator")
        print(f"   ✓ Melody Generator")
        print(f"   ✓ Effects Processor" if self.effects else "   ✗ Effects Processor (unavailable)")
        print(f"   ✓ Auto Master" if self.master else "   ✗ Auto Master (unavailable)")
    
    def _configure_for_song(self, config: SongConfig):
        """Configure generators for a specific song."""
        # Update BPM
        self.arrangement_gen.bpm = config.bpm
        self.drum_gen.bpm = config.bpm
        
        # Update key and scale for bass/chords
        self.bass_gen.key = config.key
        self.bass_gen.scale = config.scale
        self.bass_gen.root = self.bass_gen._get_root()
        
        self.chord_gen.key = config.key
        
        # Setup melody generator
        key_to_midi = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
        root_midi = key_to_midi.get(config.key.upper(), 60)
        scale_type = ScaleType.MAJOR if config.scale == "major" else ScaleType.NATURAL_MINOR
        
        self.melody_gen = MelodyGenerator(
            root_note=root_midi,
            scale_type=scale_type,
            tempo=config.bpm,
        )
        self.hook_gen = HookGenerator(self.melody_gen)
    
    def create_song(self, config: Optional[SongConfig] = None, **kwargs) -> GeneratedTrack:
        """
        Create a complete song from scratch.
        
        Args:
            config: SongConfig object with song parameters
            **kwargs: Alternative to config - individual parameters
            
        Returns:
            GeneratedTrack with all components
        """
        # Build config
        if config is None:
            config = SongConfig(**kwargs)
        
        # Auto-detect genre scale if not specified
        if config.scale == "minor" and config.genre in self.GENRE_SCALES:
            config.scale = self.GENRE_SCALES.get(config.genre, "minor")
        
        # Configure generators for this song
        self._configure_for_song(config)
        
        print(f"\n{'='*60}")
        print(f"🎵 Creating Song: {config.title}")
        print(f"{'='*60}")
        print(f"   Genre: {config.genre}")
        print(f"   Key: {config.key} {config.scale}")
        print(f"   BPM: {config.bpm}")
        print(f"   Duration: {config.duration_sec}s")
        
        # Step 1: Generate arrangement
        print(f"\n📋 Step 1: Generating arrangement...")
        arrangement = self._generate_arrangement(config)
        
        # Step 2: Generate drums for each section
        print(f"🥁 Step 2: Generating drum patterns...")
        drum_patterns = self._generate_drums(arrangement, config)
        
        # Step 3: Generate bass lines
        print(f"🎸 Step 3: Generating bass lines...")
        bass_lines = self._generate_bass(arrangement, config)
        
        # Step 4: Generate chord progressions
        print(f"🎹 Step 4: Generating chord progressions...")
        chord_progressions = self._generate_chords(arrangement, config)
        
        # Step 5: Generate melodies
        print(f"🎼 Step 5: Generating melodies...")
        melodies = self._generate_melodies(arrangement, config)
        
        # Step 6: Mix to audio
        print(f"🎚️ Step 6: Mixing to audio...")
        audio_mix = self._mix_to_audio(config, arrangement, drum_patterns, 
                                       bass_lines, chord_progressions, melodies)
        
        # Step 7: Apply effects
        if self.effects and config.reverb > 0:
            print(f"✨ Step 7: Applying effects...")
            audio_mix = self._apply_effects(audio_mix, config)
        
        # Step 8: Master
        if self.master:
            print(f"🏆 Step 8: Mastering...")
            audio_mastered = self._master_audio(audio_mix, config)
        else:
            audio_mastered = audio_mix
        
        # Build track
        track = GeneratedTrack(
            config=config,
            arrangement=arrangement,
            drum_patterns=drum_patterns,
            bass_lines=bass_lines,
            chord_progressions=chord_progressions,
            melodies=melodies,
            audio_mix=audio_mix,
            audio_mastered=audio_mastered,
            metadata={
                "total_bars": sum(s["bars"] for s in arrangement),
                "sections": len(arrangement),
                "created_by": "AI DJ Orchestrator",
            }
        )
        
        # Update stats
        self.stats["songs_created"] += 1
        self.stats["total_duration"] += config.duration_sec
        self.stats["genres_used"][config.genre] = self.stats["genres_used"].get(config.genre, 0) + 1
        
        print(f"\n✅ Song created successfully!")
        print(f"   Total Bars: {track.metadata['total_bars']}")
        print(f"   Sections: {track.metadata['sections']}")
        
        return track
    
    def _generate_arrangement(self, config: SongConfig) -> List[dict]:
        """Generate song arrangement structure."""
        arrangement = self.arrangement_gen.generate(config.genre)
        
        # Calculate total duration and scale to target
        current_duration = sum(s["duration_sec"] for s in arrangement)
        if current_duration > 0:
            scale_factor = config.duration_sec / current_duration
            
            for section in arrangement:
                section["duration_sec"] = round(section["duration_sec"] * scale_factor, 1)
                section["bars"] = max(4, int(section["bars"] * scale_factor))
        
        return arrangement
    
    def _generate_drums(self, arrangement: List[dict], config: SongConfig) -> Dict[str, str]:
        """Generate drum patterns for each section."""
        drum_style = self.GENRE_DRUM_MAP.get(config.genre, "house")
        patterns = {}
        
        for section in arrangement:
            section_type = section["section"]
            bars = section["bars"]
            
            # Vary pattern based on section energy
            if section_type in ["intro", "outro"]:
                pattern = self.drum_gen.generate(drum_style, bars=min(bars, 2))
            elif section_type in ["drop", "chorus"]:
                pattern = self.drum_gen.generate(drum_style, bars=bars)
            else:
                pattern = self.drum_gen.generate(drum_style, bars=max(1, bars // 2))
            
            patterns[section_type] = pattern
        
        return patterns
    
    def _generate_bass(self, arrangement: List[dict], config: SongConfig) -> Dict[str, list]:
        """Generate bass lines for each section."""
        bass_lines = {}
        
        for section in arrangement:
            section_type = section["section"]
            bars = section["bars"]
            
            # Generate bass line
            bass_line = self.bass_gen.generate(config.genre, bars=bars)
            bass_lines[section_type] = bass_line
        
        return bass_lines
    
    def _generate_chords(self, arrangement: List[dict], config: SongConfig) -> Dict[str, list]:
        """Generate chord progressions for each section."""
        chord_progs = {}
        
        for section in arrangement:
            section_type = section["section"]
            bars = section["bars"]
            
            # Generate progression
            progression = self.chord_gen.generate(config.genre, length=bars // 2)
            chord_progs[section_type] = progression
        
        return chord_progs
    
    def _generate_melodies(self, arrangement: List[dict], config: SongConfig) -> Dict[str, list]:
        """Generate melodies for each section."""
        melodies = {}
        
        for section in arrangement:
            section_type = section["section"]
            energy = section["energy_start"]
            
            # Only generate prominent melodies for high-energy sections
            if energy > 0.5 and self.melody_gen:
                num_notes = int(8 * energy)
                melody = self.melody_gen.generate_melody(num_notes=num_notes)
                melodies[section_type] = melody.to_midi_numbers()
            else:
                melodies[section_type] = []
        
        return melodies
    
    def _mix_to_audio(
        self,
        config: SongConfig,
        arrangement: List[dict],
        drum_patterns: Dict[str, str],
        bass_lines: Dict[str, list],
        chord_progressions: Dict[str, list],
        melodies: Dict[str, list]
    ) -> np.ndarray:
        """Mix all components to stereo audio."""
        # Calculate total samples
        total_sec = config.duration_sec
        total_samples = int(total_sec * self.sample_rate)
        
        # Create stereo buffer
        audio = np.zeros((total_samples, 2))
        
        # Generate simple waveforms for each component
        beats_per_sec = config.bpm / 60
        samples_per_beat = int(self.sample_rate / beats_per_sec)
        
        sample_position = 0
        
        for section in arrangement:
            section_type = section["section"]
            duration = section["duration_sec"]
            section_samples = int(duration * self.sample_rate)
            section_end = sample_position + section_samples
            
            # Add drums
            if section_type in drum_patterns:
                drum_pattern = drum_patterns[section_type]
                drum_events = self.drum_gen.to_midi(drum_pattern)
                
                for event in drum_events:
                    event_time = int(event["time"] * samples_per_beat)
                    if sample_position + event_time < section_end:
                        # Simple kick/snare synthesis
                        freq = 60 if event["note"] == 36 else (200 if event["note"] == 38 else 8000)
                        env = np.exp(-np.linspace(0, 0.1, int(0.1 * self.sample_rate)))
                        tone = env * np.sin(2 * np.pi * freq * np.linspace(0, 0.1, int(0.1 * self.sample_rate)))
                        end_pos = min(sample_position + event_time + len(tone), total_samples)
                        if end_pos <= total_samples:
                            audio[sample_position + event_time:end_pos, 0] += tone[:end_pos - sample_position - event_time] * event["velocity"] / 127
                            audio[sample_position + event_time:end_pos, 1] += tone[:end_pos - sample_position - event_time] * event["velocity"] / 127
            
            # Add bass
            if section_type in bass_lines:
                bass_line = bass_lines[section_type]
                bass_notes = self.bass_gen.to_notes(bass_line)
                
                for i, note_event in enumerate(bass_notes):
                    event_time = int(note_event["time"] * samples_per_beat)
                    if sample_position + event_time < section_end:
                        freq = 440 * 2 ** ((note_event["note"] - 69) / 12)
                        env = np.exp(-np.linspace(0, 0.3, int(0.3 * self.sample_rate)))
                        tone = env * np.sin(2 * np.pi * freq * np.linspace(0, 0.3, int(0.3 * self.sample_rate)))
                        end_pos = min(sample_position + event_time + len(tone), total_samples)
                        if end_pos <= total_samples:
                            audio[sample_position + event_time:end_pos, 0] += tone[:end_pos - sample_position - event_time] * note_event["velocity"] / 127 * 0.6
                            audio[sample_position + event_time:end_pos, 1] += tone[:end_pos - sample_position - event_time] * note_event["velocity"] / 127 * 0.6
            
            sample_position = section_end
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def _apply_effects(self, audio: np.ndarray, config: SongConfig) -> np.ndarray:
        """Apply effects chain to audio."""
        if not self.effects:
            return audio
        
        # Apply reverb
        if config.reverb > 0:
            try:
                audio = self.effects.reverb(
                    audio, 
                    decay=0.5, 
                    wet_gain=config.reverb,
                    room_size=0.5
                )
            except Exception as e:
                print(f"   ⚠️ Reverb failed: {e}")
        
        # Apply delay (use delay_mono or delay_stereo)
        if config.delay > 0:
            try:
                if audio.ndim == 1 or audio.shape[1] == 1:
                    audio = self.effects.delay_mono(
                        audio,
                        delay_ms=250,
                        feedback=0.3,
                        mix=config.delay
                    )
                else:
                    audio = self.effects.delay_stereo(
                        audio,
                        delay_ms=250,
                        feedback=0.3,
                        mix=config.delay
                    )
            except Exception as e:
                print(f"   ⚠️ Delay failed: {e}")
        
        return audio
    
    def _master_audio(self, audio: np.ndarray, config: SongConfig) -> np.ndarray:
        """Master the audio for streaming platforms."""
        if not self.master:
            return audio
        
        try:
            audio, meta = self.master.master(
                audio,
                apply_widening=True,
                apply_compression=True,
                apply_reference_match=False,
            )
        except Exception as e:
            print(f"   ⚠️ Mastering failed: {e}")
        
        return audio
    
    def export(self, track: GeneratedTrack, filename: str, format: str = "wav") -> str:
        """
        Export track to audio file.
        
        Args:
            track: GeneratedTrack to export
            filename: Output filename
            format: Audio format (wav, mp3, flac)
            
        Returns:
            Path to exported file
        """
        # Use mastered audio if available, otherwise mix
        audio = track.audio_mastered if track.audio_mastered is not None else track.audio_mix
        
        if audio is None:
            raise ValueError("No audio to export. Generate a track first.")
        
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        # Normalize
        audio = np.clip(audio, -1.0, 1.0)
        
        # Build output path
        output_path = self.output_dir / filename
        
        # Export based on format
        if format == "wav":
            self._export_wav(audio, output_path)
        elif format == "mp3":
            self._export_mp3(audio, output_path)
        elif format == "flac":
            self._export_flac(audio, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save metadata
        self._save_metadata(track, output_path)
        
        print(f"📁 Exported: {output_path}")
        return str(output_path)
    
    def _export_wav(self, audio: np.ndarray, path: Path):
        """Export as WAV file."""
        try:
            import scipy.io.wavfile as wavfile
            # Convert float to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(str(path), self.sample_rate, audio_int16)
        except Exception as e:
            # Fallback: save as numpy array
            np.save(str(path.with_suffix(".npy")), audio)
            print(f"   ⚠️ WAV export failed, saved as numpy: {e}")
    
    def _export_mp3(self, audio: np.ndarray, path: Path):
        """Export as MP3 file (requires pydub or ffmpeg)."""
        try:
            # Try with pydub
            from pydub import AudioSegment
            audio_int16 = (audio * 32767).astype(np.int16)
            # This is a simplified version - full implementation would need proper MP3 encoding
            self._export_wav(audio, path.with_suffix(".wav"))
            print(f"   ⚠️ MP3 not available, exported as WAV")
        except ImportError:
            self._export_wav(audio, path.with_suffix(".wav"))
            print(f"   ⚠️ MP3 not available, exported as WAV")
    
    def _export_flac(self, audio: np.ndarray, path: Path):
        """Export as FLAC file."""
        self._export_wav(audio, path.with_suffix(".wav"))
        print(f"   ⚠️ FLAC not available, exported as WAV")
    
    def _save_metadata(self, track: GeneratedTrack, audio_path: Path):
        """Save song metadata as JSON."""
        metadata = {
            "title": track.config.title,
            "artist": track.config.artist,
            "genre": track.config.genre,
            "key": track.config.key,
            "scale": track.config.scale,
            "bpm": track.config.bpm,
            "duration_sec": track.config.duration_sec,
            "energy": track.config.energy,
            "created_at": track.created_at,
            "arrangement": track.arrangement,
            "metadata": track.metadata,
        }
        
        meta_path = audio_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   📄 Metadata: {meta_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "sample_rate": self.sample_rate,
            "output_dir": str(self.output_dir),
            "songs_created": self.stats["songs_created"],
            "total_duration_sec": self.stats["total_duration"],
            "genres_used": self.stats["genres_used"],
            "components": {
                "effects": self.effects is not None,
                "mastering": self.master is not None,
            }
        }
    
    def create_from_template(
        self,
        template: str = "pop_standard",
        genre: Optional[str] = None,
        **kwargs
    ) -> GeneratedTrack:
        """
        Create a song from a predefined template.
        
        Templates:
            - pop_standard: Standard pop structure
            - edm_drop: EDM with big drops
            - hip_hop_boom: Hip-hop boom bap
            - house_groove: Deep house groove
            - cinematic: Epic cinematic
        """
        templates = {
            "pop_standard": SongConfig(
                genre=genre or "pop",
                key="C",
                scale="major",
                bpm=120,
                duration_sec=210,
                energy=0.75,
            ),
            "edm_drop": SongConfig(
                genre=genre or "edm",
                key="A",
                scale="minor",
                bpm=128,
                duration_sec=240,
                energy=0.95,
                reverb=0.4,
            ),
            "hip_hop_boom": SongConfig(
                genre=genre or "hip_hop",
                key="E",
                scale="minor",
                bpm=90,
                duration_sec=180,
                energy=0.7,
            ),
            "house_groove": SongConfig(
                genre=genre or "house",
                key="F",
                scale="minor",
                bpm=124,
                duration_sec=360,
                energy=0.8,
                reverb=0.2,
            ),
            "cinematic": SongConfig(
                genre="orchestral",
                key="D",
                scale="major",
                bpm=70,
                duration_sec=300,
                energy=0.9,
                reverb=0.5,
            ),
        }
        
        config = templates.get(template, templates["pop_standard"])
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return self.create_song(config)


# =============================================================================
# MAIN / CLI
# =============================================================================

def main():
    """CLI for the AI DJ Orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI DJ Orchestrator")
    parser.add_argument("--genre", "-g", default="house", help="Genre")
    parser.add_argument("--key", "-k", default="C", help="Musical key")
    parser.add_argument("--bpm", "-b", type=int, default=128, help="BPM")
    parser.add_argument("--duration", "-d", type=int, default=180, help="Duration in seconds")
    parser.add_argument("--output", "-o", default="output.wav", help="Output file")
    parser.add_argument("--template", "-t", help="Use template (pop_standard, edm_drop, hip_hop_boom, house_groove, cinematic)")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Create song
    if args.template:
        print(f"\n📀 Using template: {args.template}")
        song = orchestrator.create_from_template(args.template)
    else:
        config = SongConfig(
            genre=args.genre,
            key=args.key,
            bpm=args.bpm,
            duration_sec=args.duration,
            title=f"{args.genre.title()} Track",
        )
        song = orchestrator.create_song(config)
    
    # Export
    orchestrator.export(song, args.output)
    
    # Show status
    status = orchestrator.get_status()
    print(f"\n📊 Status: {status['songs_created']} songs created")


if __name__ == "__main__":
    main()
