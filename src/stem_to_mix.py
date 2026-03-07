#!/usr/bin/env python3
"""
Stem-to-Mix Workflow - Professional Mix Generation from Audio Stems
===================================================================
Takes separated stems (drums, bass, vocals, melody, etc.) and creates
professional, polished mixes with genre-aware processing.
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Try to import audio libraries, fall back gracefully
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from scipy.io import wavfile
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import existing modules if available
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from mixing_console import MixingConsole, ChannelSettings, EQSettings, CompressorSettings
    from audio_fx import AudioEffects
    HAS_INTERNAL_MODULES = True
except ImportError:
    HAS_INTERNAL_MODULES = False


class Genre(Enum):
    """Supported mixing genres with characteristic processing."""
    HOUSE = "house"
    TECHNO = "techno"
    HIPHOP = "hiphop"
    POP = "pop"
    RNB = "rnb"
    ROCK = "rock"
    EDM = "edm"
    LOFI = "lofi"
    AMBIENT = "ambient"
    GENERIC = "generic"


@dataclass
class StemConfig:
    """Configuration for an individual stem track."""
    name: str
    file_path: str
    volume: float = 0.0  # dB
    pan: float = 0.0     # -1 to 1
    mute: bool = False
    solo: bool = False
    eq_low: float = 0.0   # dB
    eq_mid: float = 0.0   # dB
    eq_high: float = 0.0  # dB
    reverb_send: float = 0.0  # 0 to 1
    delay_send: float = 0.0   # 0 to 1
    compressor_threshold: float = -20.0
    compressor_ratio: float = 4.0
    audio: np.ndarray = None
    sample_rate: int = 44100


@dataclass
class MixConfig:
    """Master mix configuration."""
    genre: Genre = Genre.GENERIC
    tempo: float = 128.0  # BPM
    key: str = "C"  # Musical key
    output_format: str = "wav"
    bit_depth: int = 24
    sample_rate: int = 44100
    master_volume: float = -6.0  # dB
    master_limiter: bool = True
    stereo_width: float = 1.0
    wet_reverb: float = 0.15
    wet_delay: float = 0.1
    transitions: List[Dict] = field(default_factory=list)


class StemToMix:
    """
    Professional stem-to-mix workflow processor.
    
    Takes separated stems and creates polished, radio-ready mixes
    with genre-aware processing, proper gain staging, and creative effects.
    """
    
    # Default stem types to look for
    DEFAULT_STEM_TYPES = [
        'drums', 'drum', 'percussion',
        'bass', 'bassline', 'low',
        'vocals', 'vocal', 'voice', 'acapella', 'acappella',
        'melody', 'lead', 'synth',
        'pads', 'ambient', 'atmosphere',
        'guitar', 'strings', 'brass',
        'fx', 'effects', 'rises', 'impacts'
    ]
    
    # Genre-specific processing presets
    GENRE_PRESETS = {
        Genre.HOUSE: {
            'drums': {'eq_low': 3, 'eq_mid': 0, 'eq_high': 2, 'comp_ratio': 6, 'comp_threshold': -15},
            'bass': {'eq_low': 2, 'eq_mid': -2, 'eq_high': 0, 'comp_ratio': 8, 'comp_threshold': -12},
            'vocals': {'eq_low': -2, 'eq_mid': 3, 'eq_high': 2, 'comp_ratio': 4, 'comp_threshold': -18},
            'melody': {'eq_low': 0, 'eq_mid': 2, 'eq_high': 3, 'comp_ratio': 3, 'comp_threshold': -20},
        },
        Genre.TECHNO: {
            'drums': {'eq_low': 4, 'eq_mid': -2, 'eq_high': 1, 'comp_ratio': 8, 'comp_threshold': -10},
            'bass': {'eq_low': 5, 'eq_mid': -3, 'eq_high': 0, 'comp_ratio': 10, 'comp_threshold': -8},
            'melody': {'eq_low': 0, 'eq_mid': 0, 'eq_high': 4, 'comp_ratio': 2, 'comp_threshold': -25},
        },
        Genre.HIPHOP: {
            'drums': {'eq_low': 2, 'eq_mid': 1, 'eq_high': 3, 'comp_ratio': 4, 'comp_threshold': -15},
            'bass': {'eq_low': 4, 'eq_mid': 0, 'eq_high': -2, 'comp_ratio': 6, 'comp_threshold': -12},
            'vocals': {'eq_low': -3, 'eq_mid': 4, 'eq_high': 2, 'comp_ratio': 3, 'comp_threshold': -20},
            'melody': {'eq_low': 1, 'eq_mid': 2, 'eq_high': 1, 'comp_ratio': 2, 'comp_threshold': -22},
        },
        Genre.POP: {
            'drums': {'eq_low': 1, 'eq_mid': 2, 'eq_high': 3, 'comp_ratio': 4, 'comp_threshold': -18},
            'bass': {'eq_low': 2, 'eq_mid': 1, 'eq_high': 0, 'comp_ratio': 4, 'comp_threshold': -15},
            'vocals': {'eq_low': -2, 'eq_mid': 4, 'eq_high': 3, 'comp_ratio': 3, 'comp_threshold': -18},
            'melody': {'eq_low': 0, 'eq_mid': 3, 'eq_high': 4, 'comp_ratio': 2, 'comp_threshold': -20},
        },
        Genre.LOFI: {
            'drums': {'eq_low': -2, 'eq_mid': 1, 'eq_high': -3, 'comp_ratio': 2, 'comp_threshold': -25},
            'bass': {'eq_low': 3, 'eq_mid': -2, 'eq_high': -4, 'comp_ratio': 2, 'comp_threshold': -20},
            'vocals': {'eq_low': -3, 'eq_mid': 2, 'eq_high': -2, 'comp_ratio': 2, 'comp_threshold': -22},
            'melody': {'eq_low': 0, 'eq_mid': 1, 'eq_high': -2, 'comp_ratio': 1.5, 'comp_threshold': -25},
        },
    }
    
    def __init__(self, config: MixConfig = None):
        self.config = config if config else MixConfig()
        self.stems: Dict[str, StemConfig] = {}
        self.audio_fx = AudioEffects(sample_rate=self.config.sample_rate) if HAS_INTERNAL_MODULES else None
        self._loaded = False
        
    def load_stems(self, stem_paths: Dict[str, str]) -> bool:
        """
        Load stem audio files.
        
        Args:
            stem_paths: Dict mapping stem names to file paths
                       e.g., {'drums': '/path/to/drums.wav', 'bass': '...'}
        
        Returns:
            True if successful, False otherwise
        """
        self.stems = {}
        
        for name, path in stem_paths.items():
            if not os.path.exists(path):
                print(f"Warning: Stem file not found: {path}")
                continue
                
            try:
                if HAS_LIBROSA:
                    audio, sr = librosa.load(path, sr=self.config.sample_rate, mono=True)
                elif HAS_SCIPY:
                    sr, audio = wavfile.read(path)
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32) / 32768.0
                else:
                    print("Error: No audio library available (librosa or scipy)")
                    return False
                
                stem_config = StemConfig(
                    name=name,
                    file_path=path,
                    audio=audio,
                    sample_rate=sr
                )
                self.stems[name] = stem_config
                print(f"Loaded stem: {name} ({len(audio)/sr:.1f}s)")
                
            except Exception as e:
                print(f"Error loading stem {name}: {e}")
                return False
        
        self._loaded = len(self.stems) > 0
        return self._loaded
    
    def load_stem_directory(self, directory: str, auto_detect: bool = True) -> bool:
        """
        Load all stems from a directory.
        
        Args:
            directory: Path to directory containing stem files
            auto_detect: Auto-detect stem types from filenames
        
        Returns:
            True if successful
        """
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return False
        
        stem_paths = {}
        
        for file_path in directory.glob("*"):
            if file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                stem_name = self._detect_stem_type(file_path.stem) if auto_detect else file_path.stem
                stem_paths[stem_name] = str(file_path)
        
        return self.load_stems(stem_paths)
    
    def _detect_stem_type(self, filename: str) -> str:
        """Detect stem type from filename."""
        filename_lower = filename.lower()
        
        for stem_type in self.DEFAULT_STEM_TYPES:
            if stem_type in filename_lower:
                # Map to canonical names
                if stem_type in ['drums', 'drum', 'percussion']:
                    return 'drums'
                elif stem_type in ['bass', 'bassline', 'low']:
                    return 'bass'
                elif stem_type in ['vocals', 'vocal', 'voice', 'acapella', 'acappella']:
                    return 'vocals'
                elif stem_type in ['melody', 'lead', 'synth']:
                    return 'melody'
                elif stem_type in ['pads', 'ambient', 'atmosphere']:
                    return 'pads'
                elif stem_type in ['guitar', 'strings', 'brass']:
                    return 'guitar'
                elif stem_type in ['fx', 'effects', 'rises', 'impacts']:
                    return 'fx'
        
        return filename  # Return original if no match
    
    def apply_genre_presets(self):
        """Apply genre-specific processing to all stems."""
        if self.config.genre not in self.GENRE_PRESETS:
            return
        
        presets = self.GENRE_PRESETS[self.config.genre]
        
        for stem_name, stem in self.stems.items():
            # Find matching preset
            preset = None
            for key in presets:
                if key in stem_name.lower():
                    preset = presets[key]
                    break
            
            if preset:
                stem.eq_low = preset.get('eq_low', 0)
                stem.eq_mid = preset.get('eq_mid', 0)
                stem.eq_high = preset.get('eq_high', 0)
                stem.compressor_threshold = preset.get('comp_threshold', -20)
                stem.compressor_ratio = preset.get('comp_ratio', 4)
                print(f"Applied {self.config.genre.value} preset to {stem_name}")
    
    def process_stem(self, stem: StemConfig) -> np.ndarray:
        """Process a single stem with EQ, compression, and effects."""
        if stem.mute or stem.audio is None:
            return np.zeros(1) if stem.audio is None else np.zeros_like(stem.audio)
        
        audio = stem.audio.copy()
        
        # Apply EQ
        audio = self._apply_eq(audio, stem.eq_low, stem.eq_mid, stem.eq_high, stem.sample_rate)
        
        # Apply compression
        audio = self._apply_compression(
            audio,
            threshold=stem.compressor_threshold,
            ratio=stem.compressor_ratio,
            sample_rate=stem.sample_rate
        )
        
        # Apply volume/trim
        audio *= self.db_to_linear(stem.volume)
        
        # Apply pan (simple gain adjustment for now)
        # Full panning would require proper stereo encoding
        
        return audio
    
    def _apply_eq(self, audio: np.ndarray, low: float, mid: float, 
                  high: float, sample_rate: int) -> np.ndarray:
        """Apply 3-band EQ."""
        if HAS_INTERNAL_MODULES and self.audio_fx:
            # Use internal EQ
            if low != 0:
                audio = self.audio_fx.eq_low_shelf(audio, freq=100, gain=low)
            if mid != 0:
                audio = self.audio_fx.eq_peaking(audio, freq=1000, gain=mid, q=1.0)
            if high != 0:
                audio = self.audio_fx.eq_high_shelf(audio, freq=8000, gain=high)
        else:
            # Simple scipy-based EQ
            if low != 0:
                b, a = butter(2, 200 / (sample_rate / 2), 'low')
                audio = signal.lfilter(b, a, audio) * (1 + low * 0.1)
            if high != 0:
                b, a = butter(2, 6000 / (sample_rate / 2), 'high')
                audio = signal.lfilter(b, a, audio) * (1 + high * 0.1)
        
        return audio
    
    def _apply_compression(self, audio: np.ndarray, threshold: float = -20,
                           ratio: float = 4, sample_rate: int = 44100) -> np.ndarray:
        """Apply dynamic range compression."""
        if not HAS_SCIPY:
            return audio
        
        # Convert threshold to linear
        threshold_linear = self.db_to_linear(threshold)
        
        # Soft knee compression
        output = np.zeros_like(audio)
        for i in range(len(audio)):
            input_level = abs(audio[i])
            
            if input_level > threshold_linear:
                over = input_level - threshold_linear
                compressed = threshold_linear + over / ratio
                gain = compressed / max(input_level, 1e-10)
            else:
                gain = 1.0
            
            output[i] = audio[i] * gain
        
        # Makeup gain (gentle)
        makeup = 1.0 + (threshold + 20) * 0.02
        output *= makeup
        
        return self._normalize(output)
    
    def create_mix(self, output_path: str = None, fade_duration: float = 2.0) -> Optional[str]:
        """
        Create the final mixed audio.
        
        Args:
            output_path: Output file path
            fade_duration: End fade duration in seconds
        
        Returns:
            Output file path if successful
        """
        if not self._loaded or not self.stems:
            print("No stems loaded!")
            return None
        
        # Find longest stem
        max_length = 0
        for stem in self.stems.values():
            if stem.audio is not None:
                max_length = max(max_length, len(stem.audio))
        
        # Process and mix each stem
        mixed_audio = np.zeros(max_length)
        
        for stem_name, stem in self.stems.items():
            processed = self.process_stem(stem)
            
            # Handle different lengths (crossfade/loop/truncate)
            if len(processed) < max_length:
                # Loop or pad
                if stem_name in ['drums', 'bass']:
                    # Loop for rhythmic elements
                    repeated = np.tile(processed, max_length // len(processed) + 1)
                    processed = repeated[:max_length]
                else:
                    # Pad for others
                    processed = np.pad(processed, (0, max_length - len(processed)))
            elif len(processed) > max_length:
                processed = processed[:max_length]
            
            # Apply panning (simple)
            pan_gain = (stem.pan + 1) / 2  # Convert -1..1 to 0..1
            processed *= pan_gain
            
            mixed_audio += processed
        
        # Normalize
        mixed_audio = self._normalize(mixed_audio, target_db=-1)
        
        # Apply master processing
        mixed_audio = self._apply_master_processing(mixed_audio)
        
        # Apply fade out at end
        fade_samples = int(fade_duration * self.config.sample_rate)
        if fade_samples < len(mixed_audio):
            fade_curve = np.linspace(1, 0, fade_samples)
            mixed_audio[-fade_samples:] *= fade_curve
        
        # Apply master volume
        mixed_audio *= self.db_to_linear(self.config.master_volume)
        
        # Save output
        if not output_path:
            output_path = "mix_output.wav"
        
        self._save_audio(mixed_audio, output_path)
        
        print(f"\nMix saved to: {output_path}")
        print(f"Duration: {len(mixed_audio)/self.config.sample_rate:.1f}s")
        
        return output_path
    
    def _apply_master_processing(self, audio: np.ndarray) -> np.ndarray:
        """Apply master bus processing."""
        # Gentle compression on master
        audio = self._apply_compression(audio, threshold=-15, ratio=2)
        
        # Stereo widening (simple)
        if self.config.stereo_width != 1.0:
            audio = self._apply_stereo_width(audio, self.config.stereo_width)
        
        # Limiter if enabled
        if self.config.master_limiter:
            audio = self._apply_limiter(audio)
        
        return audio
    
    def _apply_stereo_width(self, audio: np.ndarray, width: float) -> np.ndarray:
        """Apply stereo width adjustment."""
        # Simple MS processing
        mid = audio * 0.707  # Approximate mid
        side = audio * (width - 1) * 0.707  # Width-adjusted side
        
        # Recombine
        return mid + side
    
    def _apply_limiter(self, audio: np.ndarray, threshold: float = -0.5) -> np.ndarray:
        """Apply limiting to prevent clipping."""
        threshold_linear = self.db_to_linear(threshold)
        
        output = np.copy(audio)
        for i in range(len(output)):
            if abs(output[i]) > threshold_linear:
                output[i] = np.sign(output[i]) * threshold_linear
        
        return output
    
    def _normalize(self, audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        """Normalize audio to target dB."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_linear = self.db_to_linear(target_db)
            audio = audio * (target_linear / max_val)
        return audio
    
    def db_to_linear(self, db: float) -> float:
        """Convert dB to linear gain."""
        return 10.0 ** (db / 20.0)
    
    def _save_audio(self, audio: np.ndarray, path: str):
        """Save audio to file."""
        # Ensure 16-bit range
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        
        if HAS_SCIPY:
            wavfile.write(path, self.config.sample_rate, audio_int16)
        elif HAS_LIBROSA:
            librosa.output.write_wav(path, audio, sr=self.config.sample_rate)
    
    def set_stem_volume(self, stem_name: str, volume_db: float):
        """Set volume for a specific stem."""
        if stem_name in self.stems:
            self.stems[stem_name].volume = volume_db
    
    def set_stem_pan(self, stem_name: str, pan: float):
        """Set pan for a specific stem (-1 left, 1 right)."""
        if stem_name in self.stems:
            self.stems[stem_name].pan = np.clip(pan, -1, 1)
    
    def set_stem_eq(self, stem_name: str, low: float = 0, mid: float = 0, high: float = 0):
        """Set EQ for a specific stem."""
        if stem_name in self.stems:
            self.stems[stem_name].eq_low = low
            self.stems[stem_name].eq_mid = mid
            self.stems[stem_name].eq_high = high
    
    def get_mix_stats(self) -> Dict[str, Any]:
        """Get statistics about the current mix."""
        stats = {
            'loaded_stems': list(self.stems.keys()),
            'genre': self.config.genre.value,
            'tempo': self.config.tempo,
            'key': self.config.key,
            'sample_rate': self.config.sample_rate,
        }
        
        for stem_name, stem in self.stems.items():
            if stem.audio is not None:
                stats[f'{stem_name}_duration'] = len(stem.audio) / stem.sample_rate
                stats[f'{stem_name}_rms'] = float(np.sqrt(np.mean(stem.audio ** 2)))
        
        return stats
    
    def export_config(self, path: str):
        """Export current configuration to JSON."""
        config_dict = {
            'genre': self.config.genre.value,
            'tempo': self.config.tempo,
            'key': self.config.key,
            'master_volume': self.config.master_volume,
            'stems': {}
        }
        
        for name, stem in self.stems.items():
            config_dict['stems'][name] = {
                'volume': stem.volume,
                'pan': stem.pan,
                'eq_low': stem.eq_low,
                'eq_mid': stem.eq_mid,
                'eq_high': stem.eq_high,
                'compressor_threshold': stem.compressor_threshold,
                'compressor_ratio': stem.compressor_ratio,
            }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Config exported to: {path}")
    
    def import_config(self, path: str):
        """Import configuration from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        self.config.genre = Genre(config_dict.get('genre', 'generic'))
        self.config.tempo = config_dict.get('tempo', 128)
        self.config.key = config_dict.get('key', 'C')
        self.config.master_volume = config_dict.get('master_volume', -6)
        
        for name, stem_config in config_dict.get('stems', {}).items():
            if name in self.stems:
                self.stems[name].volume = stem_config.get('volume', 0)
                self.stems[name].pan = stem_config.get('pan', 0)
                self.stems[name].eq_low = stem_config.get('eq_low', 0)
                self.stems[name].eq_mid = stem_config.get('eq_mid', 0)
                self.stems[name].eq_high = stem_config.get('eq_high', 0)
                self.stems[name].compressor_threshold = stem_config.get('compressor_threshold', -20)
                self.stems[name].compressor_ratio = stem_config.get('compressor_ratio', 4)
        
        print(f"Config imported from: {path}")


# Convenience function for simple workflows
def quick_mix(stem_paths: Dict[str, str], output_path: str = "mix.wav", 
              genre: str = "generic") -> Optional[str]:
    """
    Quick mix function for simple workflows.
    
    Args:
        stem_paths: Dict of stem_name -> file_path
        output_path: Output file path
        genre: Genre preset ("house", "techno", "hiphop", "pop", "lofi")
    
    Returns:
        Output file path if successful
    """
    config = MixConfig(genre=Genre(genre))
    processor = StemToMix(config)
    
    if not processor.load_stems(stem_paths):
        return None
    
    processor.apply_genre_presets()
    return processor.create_mix(output_path)


# CLI usage
def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stem to Mix - Professional Mix Generator')
    parser.add_argument('input', nargs='?', help='Input stem directory or JSON config')
    parser.add_argument('-o', '--output', default='mix.wav', help='Output file path')
    parser.add_argument('-g', '--genre', default='generic', 
                       choices=['house', 'techno', 'hiphop', 'pop', 'lofi', 'edm', 'rnb', 'rock', 'ambient', 'generic'],
                       help='Genre preset')
    parser.add_argument('-t', '--tempo', type=float, default=128, help='Tempo (BPM)')
    parser.add_argument('--config-export', help='Export config to JSON')
    parser.add_argument('--config-import', help='Import config from JSON')
    
    args = parser.parse_args()
    
    # Create processor
    config = MixConfig(genre=Genre(args.genre), tempo=args.tempo)
    processor = StemToMix(config)
    
    # Handle config import
    if args.config_import:
        processor.import_config(args.config_import)
    
    # Load stems
    if args.input:
        if os.path.isdir(args.input):
            processor.load_stem_directory(args.input)
        elif args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                stem_paths = json.load(f)
            processor.load_stems(stem_paths)
        else:
            print(f"Unknown input: {args.input}")
            return
    
    if not processor.stems:
        print("No stems loaded!")
        parser.print_help()
        return
    
    # Apply genre presets
    processor.apply_genre_presets()
    
    # Show stats
    stats = processor.get_mix_stats()
    print(f"\nLoaded stems: {', '.join(stats['loaded_stems'])}")
    print(f"Genre: {stats['genre']}, Tempo: {stats['tempo']} BPM")
    
    # Create mix
    output = processor.create_mix(args.output)
    
    # Handle config export
    if args.config_export:
        processor.export_config(args.config_export)


if __name__ == "__main__":
    main()
