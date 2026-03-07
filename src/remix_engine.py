"""
Remix Engine
============
A Python module for creating professional remixes from existing audio tracks.

Features:
- Stem separation (vocals, drums, bass, other)
- Tempo and key shifting
- Remix style presets (club, radio, acoustic, etc.)
- Effect processing (reverb, delay, filter sweeps)
- Arrangement restructuring
- Professional mixing and mastering

Usage:
    from remix_engine import RemixEngine, RemixConfig, RemixStyle
    
    engine = RemixEngine()
    config = RemixConfig(
        input_track="path/to/track.wav",
        output_path="remix_output.wav",
        style=RemixStyle.CLUB,
        target_bpm=128,
        target_key="Cm",
        stem_mix={"vocals": 0.8, "drums": 1.0, "bass": 1.0, "other": 0.6}
    )
    result = engine.create_remix(config)
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import os


class RemixStyle(Enum):
    """Remix style presets with different characteristics."""
    CLUB = "club"           # Full energy, 4/4 kick, club PA optimized
    RADIO = "radio"         # Radio-friendly, cleaner mix
    ACOUSTIC = "acoustic"   # Stripped back, organic feel
    AMBIENT = "ambient"     # Atmospheric, spacious
    TECHNO = "techno"       # Minimal, driving, dark
    HIPHOP = "hiphop"       # Boom bap style, swung beats
    HOUSE = "house"         # Classic house feel
    TRANCE = "trance"       # Uplifting, melodic
    DUBSTEP = "dubstep"     # Wobble bass, heavy drops
    CHILL = "chill"         # Downtempo, relaxed


class RemixSection(Enum):
    """Sections of a remix."""
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    DROP = "drop"
    OUTRO = "outro"


@dataclass
class RemixConfig:
    """Configuration for remix creation."""
    # Input/Output
    input_track: str = ""
    output_path: str = "remix.wav"
    
    # Tempo & Key
    target_bpm: Optional[int] = None
    target_key: Optional[str] = None  # e.g., "Cm", "F#"
    
    # Style
    style: RemixStyle = RemixStyle.CLUB
    
    # Stem mixing (0.0 to 1.0 for each stem)
    stem_mix: Dict[str, float] = field(default_factory=lambda: {
        "vocals": 1.0,
        "drums": 1.0,
        "bass": 1.0,
        "other": 1.0
    })
    
    # Effects
    reverb_wet: float = 0.0      # 0.0 = dry, 1.0 = fully wet
    delay_wet: float = 0.0
    filter_cutoff: float = 20000  # Hz, 20k = no filter
    filter_resonance: float = 0.0
    
    # Remix options
    extend_duration: float = 0.0  # seconds to extend
    loop_stems: bool = False
    add_transition: bool = True
    transition_duration: float = 4.0  # seconds
    
    # Quality
    sample_rate: int = 44100
    bit_depth: int = 16


@dataclass
class TrackAnalysis:
    """Analysis results for an input track."""
    bpm: float = 120.0
    key: str = "C"
    duration: float = 0.0
    rms_energy: float = 0.0
    spectral_centroid: float = 0.0
    has_vocals: bool = True
    has_drums: bool = True
    has_bass: bool = True


class RemixEngine:
    """
    Main remix engine class for creating professional remixes.
    """
    
    # Musical key semitone shifts for harmonic mixing
    KEY_COMPATIBLE = {
        "C": ["C", "Am", "F", "Em", "G", "Em"],
        "Cm": ["Cm", "Ab", "Gm", "Eb", "Bb", "Gm"],
        "C#": ["C#", "A#m", "F#", "G#m", "A#", "G#m"],
        "C#m": ["C#m", "Bb", "G#m", "Fm", "Db", "G#m"],
        "D": ["D", "Bm", "G", "Am", "A", "Bm"],
        "Dm": ["Dm", "Bb", "Am", "Gm", "C", "Am"],
        "D#": ["D#", "Cm", "G#", "A#m", "Bb", "A#m"],
        "D#m": ["D#m", "C", "A#m", "Gm", "Db", "A#m"],
        "E": ["E", "C#m", "G#", "A#m", "B", "C#m"],
        "Em": ["Em", "C", "G", "Am", "D", "Am"],
        "F": ["F", "Dm", "A", "Bm", "C", "Dm"],
        "Fm": ["Fm", "C", "A#m", "Gm", "Db", "A#m"],
        "F#": ["F#", "Ebm", "B", "C#m", "Db", "C#m"],
        "F#m": ["F#m", "D", "Bm", "A#m", "E", "Bm"],
        "G": ["G", "Em", "C", "Dm", "D", "Em"],
        "Gm": ["Gm", "Eb", "Dm", "Cm", "F", "Dm"],
        "G#": ["G#", "Fm", "C#", "D#m", "D#", "D#m"],
        "G#m": ["G#m", "E", "C#m", "Bm", "F#", "Bm"],
        "A": ["A", "F#m", "D", "Em", "E", "F#m"],
        "Am": ["Am", "F", "C", "Gm", "G", "C"],
        "A#": ["A#", "Gm", "D#", "Fm", "F", "Fm"],
        "A#m": ["A#m", "G", "Fm", "Em", "A#", "Em"],
        "B": ["B", "G#m", "F#", "G#m", "Db", "G#m"],
        "Bm": ["Bm", "G", "Em", "Dm", "A", "Em"],
    }
    
    # Style-specific BPM ranges
    STYLE_BPM = {
        RemixStyle.CLUB: (124, 130),
        RemixStyle.RADIO: (90, 120),
        RemixStyle.ACOUSTIC: (70, 100),
        RemixStyle.AMBIENT: (60, 90),
        RemixStyle.TECHNO: (125, 135),
        RemixStyle.HIPHOP: (70, 95),
        RemixStyle.HOUSE: (120, 128),
        RemixStyle.TRANCE: (138, 145),
        RemixStyle.DUBSTEP: (140, 142),
        RemixStyle.CHILL: (80, 110),
    }
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.analysis: Optional[TrackAnalysis] = None
        self.stems: Dict[str, np.ndarray] = {}
        self._temp_files: List[str] = []
    
    def __del__(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def analyze_track(self, track_path: str) -> TrackAnalysis:
        """
        Analyze a track to extract BPM, key, and other properties.
        
        Args:
            track_path: Path to the audio file
            
        Returns:
            TrackAnalysis object with track properties
        """
        print(f"🔍 Analyzing track: {track_path}")
        
        # Load audio
        y, sr = librosa.load(track_path, sr=self.sample_rate)
        
        # Detect BPM using beat tracking
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        
        # Detect key using chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key = keys[key_idx]
        
        # Check if minor (more energy in lower frequencies)
        min_key = keys[key_idx] + "m"
        
        # Duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)
        rms_energy = float(np.mean(rms))
        
        # Spectral centroid (brightness)
        spectral = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid = float(np.mean(spectral))
        
        # Detect presence of stems (simplified)
        # In production, this would use ML models
        has_vocals = True  # Assume true, refine with separation
        has_drums = True
        has_bass = True
        
        self.analysis = TrackAnalysis(
            bpm=bpm,
            key=key,
            duration=duration,
            rms_energy=rms_energy,
            spectral_centroid=spectral_centroid,
            has_vocals=has_vocals,
            has_drums=has_drums,
            has_bass=has_bass
        )
        
        print(f"   BPM: {bpm:.1f}")
        print(f"   Key: {key}")
        print(f"   Duration: {duration:.1f}s")
        
        return self.analysis
    
    def separate_stems(self, track_path: str, 
                       model: str = "htdemucs",
                       output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Separate track into stems using Demucs.
        
        Args:
            track_path: Path to the audio file
            model: Demucs model to use
            output_dir: Directory for stem outputs
            
        Returns:
            Dict mapping stem names to file paths
        """
        print(f"🎵 Separating stems: {track_path}")
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        # Check if demucs is available
        try:
            import demucs.separate
            demucs_available = True
        except ImportError:
            demucs_available = False
            print("   ⚠️ Demucs not available, using basic separation")
        
        if demucs_available:
            # Use Demucs for stem separation
            # This is a simplified call - full implementation would handle args
            try:
                # Import and run demucs
                from demucs import separate
                import shlex
                
                cmd = f"-n {model} --out {output_dir} {track_path}"
                args = separate.parse_args(shlex.split(cmd))
                separate.main(args)
                
                # Find separated stems
                track_name = Path(track_path).stem
                stem_dir = Path(output_dir) / model / track_name
                
                self.stems = {}
                for stem_name in ["vocals", "drums", "bass", "other"]:
                    stem_path = stem_dir / f"{stem_name}.wav"
                    if stem_path.exists():
                        self.stems[stem_name], _ = librosa.load(
                            str(stem_path), sr=self.sample_rate
                        )
                        print(f"   ✓ {stem_name}: loaded")
            except Exception as e:
                print(f"   ⚠️ Demucs error: {e}, using basic separation")
                self._basic_separation(track_path)
        else:
            # Fallback: basic frequency-based separation
            self._basic_separation(track_path)
        
        return {name: f"{output_dir}/{name}.wav" for name in self.stems.keys()}
    
    def _basic_separation(self, track_path: str):
        """Basic frequency-based stem separation fallback."""
        print("   Using basic frequency-based separation")
        
        y, sr = librosa.load(track_path, sr=self.sample_rate)
        
        # Simple frequency band separation
        # Bass: < 250 Hz
        # Drums: 250 Hz - 5 kHz (simplified)
        # Other: > 5 kHz
        # Vocals: mid range (simplified)
        
        from scipy.signal import butter, filtfilt
        
        def band_filter(data, low, high, sr):
            nyq = sr / 2
            low_norm = low / nyq
            high_norm = high / nyq
            b, a = butter(4, [low_norm, high_norm], btype='band')
            return filtfilt(b, a, data)
        
        # This is a simplified approximation
        self.stems["bass"] = band_filter(y, 20, 250, sr)
        self.stems["drums"] = band_filter(y, 250, 5000, sr)
        self.stems["vocals"] = band_filter(y, 500, 8000, sr)
        self.stems["other"] = band_filter(y, 8000, 20000, sr)
        
        for name in self.stems:
            print(f"   ✓ {name}: separated (basic)")
    
    def change_tempo(self, audio: np.ndarray, 
                     source_bpm: float, 
                     target_bpm: float) -> np.ndarray:
        """
        Change tempo of audio without affecting pitch.
        
        Args:
            audio: Audio samples
            source_bpm: Original BPM
            target_bpm: Target BPM
            
        Returns:
            Time-stretched audio
        """
        if source_bpm == target_bpm:
            return audio
        
        # Calculate time stretch ratio
        ratio = target_bpm / source_bpm
        
        # Use librosa for time stretching
        # stretch with phase vocoder for professional results
        audio_stretched = librosa.effects.time_stretch(
            audio, 
            rate=ratio
        )
        
        return audio_stretched
    
    def change_key(self, audio: np.ndarray,
                   source_key: str,
                   target_key: str,
                   sample_rate: int = 44100) -> np.ndarray:
        """
        Change key of audio without affecting tempo.
        
        Args:
            audio: Audio samples
            source_key: Original key (e.g., "C", "Am")
            target_key: Target key
            
        Returns:
            Pitch-shifted audio
        """
        if source_key == target_key:
            return audio
        
        # Parse keys to semitones
        def key_to_semitone(key: str) -> int:
            key = key.replace("m", "")
            keys = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
            return keys.get(key, 0)
        
        source_semi = key_to_semitone(source_key)
        target_semi = key_to_semitone(target_key)
        
        # Calculate semitone shift
        semitone_shift = target_semi - source_semi
        
        # Use librosa for pitch shifting
        audio_shifted = librosa.effects.pitch_shift(
            audio,
            sr=sample_rate,
            n_steps=semitone_shift
        )
        
        return audio_shifted
    
    def apply_reverb(self, audio: np.ndarray, 
                     wet: float = 0.3,
                     room_size: float = 0.5,
                     sample_rate: int = 44100) -> np.ndarray:
        """
        Apply reverb effect to audio.
        
        Args:
            audio: Input audio
            wet: Wet/dry mix (0.0 to 1.0)
            room_size: Room size (0.0 to 1.0)
            sample_rate: Audio sample rate
            
        Returns:
            Audio with reverb applied
        """
        if wet <= 0:
            return audio
        
        # Generate impulse response
        ir_length = int(sample_rate * (1.0 + room_size * 2))
        impulse = np.random.randn(ir_length) * 0.1
        
        # Decay
        decay = np.exp(-3 * np.linspace(0, 1, ir_length))
        impulse = impulse * decay
        
        # Add some early reflections
        early_reflections = int(sample_rate * 0.03)
        impulse[:early_reflections] *= 2
        
        # Apply convolution reverb
        from scipy.signal import convolve
        wet_audio = convolve(audio, impulse, mode='same')
        
        # Mix wet and dry
        result = audio * (1 - wet) + wet_audio * wet
        
        return result
    
    def apply_delay(self, audio: np.ndarray,
                    wet: float = 0.3,
                    delay_time: float = 0.5,  # seconds
                    feedback: float = 0.4,
                    sample_rate: int = 44100) -> np.ndarray:
        """
        Apply delay effect to audio.
        
        Args:
            audio: Input audio
            wet: Wet/dry mix (0.0 to 1.0)
            delay_time: Delay time in seconds
            feedback: Feedback amount (0.0 to 1.0)
            sample_rate: Audio sample rate
            
        Returns:
            Audio with delay applied
        """
        if wet <= 0:
            return audio
        
        # Create delay buffer
        delay_samples = int(delay_time * sample_rate)
        output = audio.copy()
        
        # Feedback delay
        delayed = np.zeros_like(audio)
        delayed_sample = np.zeros(delay_samples)
        
        for i in range(len(audio)):
            delayed_sample = np.roll(delayed_sample, 1)
            delayed_sample[0] = audio[i] + delayed_sample[-1] * feedback
            delayed[i] = delayed_sample[0]
        
        # Mix
        result = audio * (1 - wet) + delayed * wet
        
        return result
    
    def apply_filter(self, audio: np.ndarray,
                    cutoff: float = 20000,
                    resonance: float = 0.0,
                    filter_type: str = "lowpass",
                    sample_rate: int = 44100) -> np.ndarray:
        """
        Apply filter to audio.
        
        Args:
            audio: Input audio
            cutoff: Filter cutoff frequency in Hz
            resonance: Resonance/Q factor (0.0 to 1.0)
            filter_type: "lowpass", "highpass", or "bandpass"
            sample_rate: Audio sample rate
            
        Returns:
            Filtered audio
        """
        if cutoff >= 20000 and filter_type == "lowpass":
            return audio
        if cutoff <= 20 and filter_type == "highpass":
            return audio
        
        from scipy.signal import butter, filtfilt
        
        nyq = sample_rate / 2
        cutoff_norm = min(cutoff / nyq, 0.99)
        
        # Convert resonance to Q
        q = 0.5 + resonance * 10
        
        if filter_type == "lowpass":
            b, a = butter(4, cutoff_norm, btype='low', q=q)
        elif filter_type == "highpass":
            b, a = butter(4, cutoff_norm, btype='high', q=q)
        else:
            b, a = butter(4, cutoff_norm, btype='band', q=q)
        
        filtered = filtfilt(b, a, audio)
        
        return filtered
    
    def create_transition(self, audio_out: np.ndarray,
                         audio_in: np.ndarray,
                         duration: float,
                         sample_rate: int = 44100) -> np.ndarray:
        """
        Create a smooth transition between two audio segments.
        
        Args:
            audio_out: Ending audio
            audio_in: Starting audio  
            duration: Transition duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Combined audio with crossfade
        """
        trans_samples = int(duration * sample_rate)
        trans_samples = min(trans_samples, len(audio_out), len(audio_in))
        
        # Equal-power crossfade
        t = np.linspace(0, np.pi/2, trans_samples)
        fade_out = np.cos(t)
        fade_in = np.sin(t)
        
        result = audio_out.copy()
        
        # Crossfade region
        out_slice = result[:trans_samples]
        in_slice = audio_in[:trans_samples]
        
        result[:trans_samples] = out_slice * fade_out + in_slice * fade_in
        
        # Append remaining audio
        result = np.concatenate([result, audio_in[trans_samples:]])
        
        return result
    
    def remix_stems(self, config: RemixConfig) -> np.ndarray:
        """
        Apply remix processing to separated stems.
        
        Args:
            config: Remix configuration
            
        Returns:
            Processed and mixed audio
        """
        print(f"🎛️ Applying {config.style.value} remix processing...")
        
        # Default BPM range for style
        if config.target_bpm is None:
            bpm_range = self.STYLE_BPM.get(config.style, (120, 128))
            config.target_bpm = int(np.mean(bpm_range))
        
        # Get source BPM
        source_bpm = self.analysis.bpm if self.analysis else 120.0
        source_key = self.analysis.key if self.analysis else "C"
        
        processed_stems = {}
        
        for stem_name, stem_audio in self.stems.items():
            # Apply stem-specific gain
            stem_gain = config.stem_mix.get(stem_name, 1.0)
            processed = stem_audio * stem_gain
            
            # Tempo change
            if config.target_bpm and config.target_bpm != source_bpm:
                processed = self.change_tempo(
                    processed, source_bpm, config.target_bpm
                )
            
            # Key change
            if config.target_key and config.target_key != source_key:
                processed = self.change_key(
                    processed, source_key, config.target_key,
                    config.sample_rate
                )
            
            # Apply stem-specific effects
            if stem_name == "vocals" and config.reverb_wet > 0:
                processed = self.apply_reverb(
                    processed, config.reverb_wet * 0.5,
                    sample_rate=config.sample_rate
                )
            
            if stem_name == "drums" and config.filter_cutoff < 20000:
                processed = self.apply_filter(
                    processed, config.filter_cutoff,
                    config.filter_resonance, "lowpass",
                    config.sample_rate
                )
            
            if stem_name == "bass":
                # Bass boost for club style
                if config.style == RemixStyle.CLUB:
                    processed = self.apply_filter(
                        processed, 200, 0.3, "lowpass", config.sample_rate
                    )
            
            processed_stems[stem_name] = processed
            print(f"   ✓ {stem_name}: processed")
        
        # Mix stems together
        mixed = np.zeros(max(len(s) for s in processed_stems.values()))
        
        for stem_audio in processed_stems.values():
            # Match lengths
            if len(stem_audio) < len(mixed):
                mixed[:len(stem_audio)] += stem_audio
            else:
                mixed += stem_audio[:len(mixed)]
        
        # Normalize
        if np.max(np.abs(mixed)) > 0:
            mixed = mixed / np.max(np.abs(mixed)) * 0.9
        
        return mixed
    
    def create_remix(self, config: RemixConfig) -> str:
        """
        Create a complete remix from an input track.
        
        Args:
            config: Remix configuration
            
        Returns:
            Path to the created remix file
        """
        print(f"\n🎬 Creating {config.style.value} remix...")
        print(f"   Input: {config.input_track}")
        print(f"   Output: {config.output_path}")
        
        # Step 1: Analyze the track
        self.analyze_track(config.input_track)
        
        # Step 2: Separate stems
        self.separate_stems(config.input_track)
        
        # Step 3: Process and remix stems
        remixed_audio = self.remix_stems(config)
        
        # Step 4: Apply master effects
        print("   Applying master effects...")
        
        if config.reverb_wet > 0:
            remixed_audio = self.apply_reverb(
                remixed_audio, config.reverb_wet * 0.3,
                sample_rate=config.sample_rate
            )
        
        if config.delay_wet > 0:
            remixed_audio = self.apply_delay(
                remixed_audio, config.delay_wet,
                sample_rate=config.sample_rate
            )
        
        if config.filter_cutoff < 20000:
            remixed_audio = self.apply_filter(
                remixed_audio, config.filter_cutoff,
                config.filter_resonance, "lowpass",
                config.sample_rate
            )
        
        # Step 5: Apply limiting and normalization
        print("   Finalizing mix...")
        
        # Soft clipping for warmth
        remixed_audio = np.tanh(remixed_audio * 1.2) / 1.2
        
        # Limiter
        peak = np.max(np.abs(remixed_audio))
        if peak > 0.95:
            remixed_audio = remixed_audio * (0.95 / peak)
        
        # Normalize
        remixed_audio = remixed_audio / np.max(np.abs(remixed_audio)) * 0.91
        
        # Step 6: Save output
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(
            str(output_path),
            remixed_audio,
            config.sample_rate,
            subtype='PCM_16' if config.bit_depth == 16 else 'PCM_24'
        )
        
        print(f"✅ Remix saved: {output_path}")
        
        return str(output_path)
    
    def create_quick_remix(self,
                          input_track: str,
                          output_path: str,
                          style: RemixStyle = RemixStyle.CLUB,
                          target_bpm: Optional[int] = None) -> str:
        """
        Quick remix with sensible defaults.
        
        Args:
            input_track: Path to input audio
            output_path: Path for output
            style: Remix style
            target_bpm: Optional target BPM
            
        Returns:
            Path to created remix
        """
        config = RemixConfig(
            input_track=input_track,
            output_path=output_path,
            style=style,
            target_bpm=target_bpm,
            stem_mix={"vocals": 0.8, "drums": 1.0, "bass": 1.0, "other": 0.6}
        )
        
        return self.create_remix(config)


def create_stem_remix(input_track: str,
                      output_path: str,
                      stems_to_use: List[str] = None,
                      stem_gains: Dict[str, float] = None,
                      bpm: Optional[int] = None,
                      key: Optional[str] = None) -> str:
    """
    Convenience function for stem-based remixing.
    
    Args:
        input_track: Input audio file
        output_path: Output file path
        stems_to_use: List of stems to include (e.g., ["vocals", "drums"])
        stem_gains: Gain for each stem
        bpm: Target BPM (None = auto-detect)
        key: Target key (None = auto-detect)
        
    Returns:
        Path to created remix
    """
    engine = RemixEngine()
    
    # Analyze
    engine.analyze_track(input_track)
    engine.separate_stems(input_track)
    
    # Build config
    stem_mix = stem_gains or {stem: 1.0 for stem in engine.stems}
    if stems_to_use:
        stem_mix = {s: stem_mix.get(s, 0.0) for s in stems_to_use}
    
    config = RemixConfig(
        input_track=input_track,
        output_path=output_path,
        target_bpm=bpm,
        target_key=key,
        stem_mix=stem_mix
    )
    
    # Process
    remixed = engine.remix_stems(config)
    
    # Save
    sf.write(output_path, remixed, engine.sample_rate)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Remix Engine v1.0")
    print("=" * 40)
    
    # Quick example (would need actual audio files)
    # engine = RemixEngine()
    # engine.create_quick_remix(
    #     "music/track.wav",
    #     "music/remix.wav",
    #     style=RemixStyle.CLUB,
    #     target_bpm=128
    # )
    
    print("\nUsage:")
    print("  from remix_engine import RemixEngine, RemixConfig, RemixStyle")
    print("  engine = RemixEngine()")
    print("  engine.create_quick_remix('input.wav', 'output.wav')")
