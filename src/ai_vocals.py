#!/usr/bin/env python3
"""
AI Vocals Generator - Generate AI-powered vocals for the DJ project
Supports multiple backends: ElevenLabs, Bark, Coqui TTS, RVC, So-Vits-SVC
"""

import os
import json
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Audio processing
import numpy as np
import soundfile as sf

# For vocal processing
try:
    import librosa
except ImportError:
    librosa = None

try:
    import torch
except ImportError:
    torch = None


class VocalModel(Enum):
    """Supported AI vocal models"""
    BARK = "bark"
    COQUI_TTS = "coqui"
    ELEVENLABS = "elevenlabs"
    RVC = "rvc"
    SO_VITS = "so-vits"
    VOICEBOX = "voicebox"


class VoiceStyle(Enum):
    """Vocal style presets"""
    POP = "pop"
    ROCK = "rock"
    RNB = "rnb"
    EDM = "edm"
    HIPHOP = "hiphop"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    spoken = "spoken"


@dataclass
class VocalConfig:
    """Configuration for vocal generation"""
    model: VocalModel = VocalModel.BARK
    voice_preset: str = "v2/en_speaker_6"
    style: VoiceStyle = VoiceStyle.POP
    pitch: float = 0.0  # semitones
    tempo: float = 1.0
    reverb_wet: float = 0.3
    delay_wet: float = 0.1
    harmony_voices: int = 0
    harmony_interval: float = 3.0  # semitones
    formantshift: float = 0.0
    vibrato: float = 0.0
    vibrato_rate: float = 5.0
    output_sr: int = 44100


@dataclass
class GeneratedVocal:
    """Container for generated vocal output"""
    audio: np.ndarray
    sample_rate: int
    config: VocalConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    
    def __post_init__(self):
        if self.audio is not None and len(self.audio) > 0:
            self.duration = len(self.audio) / self.sample_rate


class AIVocalsGenerator:
    """Main class for generating AI vocals"""
    
    def __init__(self, config: Optional[VocalConfig] = None):
        self.config = config or VocalConfig()
        self.model_loaded = False
        
    def _ensure_model(self):
        """Lazy load the model"""
        if not self.model_loaded:
            self._load_model()
            self.model_loaded = True
    
    def _load_model(self):
        """Load the selected AI vocal model"""
        if self.config.model == VocalModel.BARK:
            self._load_bark()
        elif self.config.model == VocalModel.COQUI_TTS:
            self._load_coqui()
        elif self.config.model == VocalModel.ELEVENLABS:
            self._load_elevenlabs()
        elif self.config.model == VocalModel.RVC:
            self._load_rvc()
        elif self.config.model == VocalModel.SO_VITS:
            self._load_so_vits()
        else:
            raise ValueError(f"Unsupported model: {self.config.model}")
    
    def _load_bark(self):
        """Load Bark model"""
        try:
            from bark.generation import load_model as load_bark_model
            from bark import generate_audio
            self.bark_generate = generate_audio
            print("Bark model loaded")
        except ImportError:
            raise ImportError("Bark not installed. Run: pip install bark")
    
    def _load_coqui(self):
        """Load Coqui TTS model"""
        try:
            from TTS.api import TTS
            self.tts = TTS(model_name="xtts_v2", gpu=True)
            print("Coqui TTS model loaded")
        except ImportError:
            raise ImportError("Coqui TTS not installed. Run: pip install TTS")
    
    def _load_elevenlabs(self):
        """Load ElevenLabs (API-based, no local model)"""
        self.elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not self.elevenlabs_key:
            print("Warning: ELEVENLABS_API_KEY not set")
        print("ElevenLabs ready (API-based)")
    
    def _load_rvc(self):
        """Load RVC (Retrieval-based Voice Conversion)"""
        print("RVC model ready (requires .pth and .index files)")
    
    def _load_so_vits(self):
        """Load So-Vits-SVC"""
        print("So-Vits-SVC model ready (requires model checkpoint)")
    
    def generate(
        self,
        text: str,
        output_path: Optional[str] = None,
        config: Optional[VocalConfig] = None
    ) -> GeneratedVocal:
        """
        Generate vocals from text
        
        Args:
            text: Lyrics or text to synthesize
            output_path: Optional path to save audio file
            config: Optional override config
            
        Returns:
            GeneratedVocal object
        """
        if config:
            self.config = config
        
        self._ensure_model()
        
        # Generate base audio
        audio = self._synthesize_text(text)
        
        # Apply processing
        audio = self._apply_pitch_shift(audio)
        audio = self._apply_formant_shift(audio)
        audio = self._apply_vibrato(audio)
        audio = self._apply_reverb(audio)
        audio = self._apply_delay(audio)
        
        # Generate harmonies if requested
        if self.config.harmony_voices > 0:
            audio = self._generate_harmony(audio)
        
        # Normalize
        audio = self._normalize_audio(audio)
        
        result = GeneratedVocal(
            audio=audio,
            sample_rate=self.config.output_sr,
            config=self.config,
            metadata={"text": text},
            duration=len(audio) / self.config.output_sr
        )
        
        # Save if output path provided
        if output_path:
            sf.write(output_path, audio, self.config.output_sr)
            result.metadata["saved_to"] = output_path
        
        return result
    
    def _synthesize_text(self, text: str) -> np.ndarray:
        """Synthesize text using the selected model"""
        if self.config.model == VocalModel.BARK:
            return self._synthesize_bark(text)
        elif self.config.model == VocalModel.COQUI_TTS:
            return self._synthesize_coqui(text)
        elif self.config.model == VocalModel.ELEVENLABS:
            return self._synthesize_elevenlabs(text)
        else:
            # Fallback: generate simple sine wave melody
            return self._synthesize_simple(text)
    
    def _synthesize_bark(self, text: str) -> np.ndarray:
        """Generate using Bark"""
        try:
            from bark import sound
            from bark.generation import generate_audio
            
            # Bark has voice presets
            voice_preset = self.config.voice_preset
            
            audio_array = generate_audio(
                text,
                voice_preset=voice_preset,
                # Add semantic_prompt for music/singing
                semantic_prompt=None
            )
            
            # Resample to output_sr
            if audio_array is not None:
                if librosa:
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=24000, 
                        target_sr=self.config.output_sr
                    )
            return audio_array if audio_array is not None else np.zeros(self.config.output_sr)
        except Exception as e:
            print(f"Bark synthesis error: {e}")
            return self._synthesize_simple(text)
    
    def _synthesize_coqui(self, text: str) -> np.ndarray:
        """Generate using Coqui TTS"""
        try:
            # Generate to file first
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            self.tts.tts_to_file(
                text=text,
                file_path=temp_path,
                voice_dir=None  # Use default voice
            )
            
            # Load the generated file
            audio, sr = sf.read(temp_path)
            os.unlink(temp_path)
            
            # Resample if needed
            if sr != self.config.output_sr and librosa:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.output_sr)
            
            return audio
        except Exception as e:
            print(f"Coqui synthesis error: {e}")
            return self.synthesize_simple(text)
    
    def _synthesize_elevenlabs(self, text: str) -> np.ndarray:
        """Generate using ElevenLabs API"""
        if not self.elevenlabs_key:
            return self._synthesize_simple(text)
        
        try:
            import requests
            
            voice_id = "pNInz6obpgDQGcFmaJgB"  # Default Adam voice
            
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": self.elevenlabs_key
                },
                json={
                    "text": text,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
            )
            
            if response.status_code == 200:
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    temp_path = f.name
                    f.write(response.content)
                
                # Convert to numpy
                import subprocess
                result = subprocess.run(
                    ["ffmpeg", "-i", temp_path, "-ar", str(self.config.output_sr), "-ac", "1", "-f", "wav", "-"],
                    capture_output=True
                )
                os.unlink(temp_path)
                
                if result.returncode == 0:
                    import io
                    audio, sr = sf.read(io.BytesIO(result.stdout))
                    return audio
            
            return self._synthesize_simple(text)
        except Exception as e:
            print(f"ElevenLabs synthesis error: {e}")
            return self._synthesize_simple(text)
    
    def _synthesize_simple(self, text: str) -> np.ndarray:
        """Fallback simple synthesis (sine wave melody)"""
        # Simple musical scale
        notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C major
        duration_per_note = 0.3
        sr = self.config.output_sr
        
        audio = []
        for i, char in enumerate(text[:40]):  # Limit length
            freq = notes[i % len(notes)]
            # Different frequencies for vowels vs consonants
            if char.lower() in 'aeiou':
                freq *= 1.0
            else:
                freq *= 0.8
            
            t = np.linspace(0, duration_per_note, int(sr * duration_per_note))
            note = np.sin(2 * np.pi * freq * t)
            # Apply envelope
            envelope = np.exp(-3 * np.linspace(0, 1, len(note)))
            note *= envelope
            audio.append(note)
        
        return np.concatenate(audio) if audio else np.zeros(sr)
    
    def _apply_pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shifting"""
        if self.config.pitch == 0.0 or not librosa:
            return audio
        
        # Convert semitones to ratio
        pitch_ratio = 2 ** (self.config.pitch / 12)
        
        # Use librosa for pitch shifting
        audio_stretched = librosa.effects.pitch_shift(
            audio, 
            sr=self.config.output_sr,
            n_steps=self.config.pitch
        )
        return audio_stretched
    
    def _apply_formant_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply formant shifting (voice character)"""
        if self.config.formantshift == 0.0 or not librosa:
            return audio
        
        # Formant shifting via pitch-preserving time stretch
        formant_ratio = 2 ** (self.config.formantshift / 12)
        
        # Stretch then compress to preserve pitch
        stretched = librosa.effects.time_stretch(audio, rate=formant_ratio)
        # Then pitch shift back
        audio = librosa.effects.pitch_shift(
            stretched,
            sr=self.config.output_sr,
            n_steps=-self.config.formantshift
        )
        return audio
    
    def _apply_vibrato(self, audio: np.ndarray) -> np.ndarray:
        """Add vibrato effect"""
        if self.config.vibrato == 0.0:
            return audio
        
        sr = self.config.output_sr
        t = np.linspace(0, len(audio) / sr, len(audio))
        
        # Vibrato is frequency modulation
        mod_depth = self.config.vibrato * 20  # Hz deviation
        mod_freq = self.config.vibrato_rate  # Hz
        
        vibrato = 1 + (mod_depth / 440) * np.sin(2 * np.pi * mod_freq * t)
        
        return audio * vibrato
    
    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Apply reverb effect"""
        if self.config.reverb_wet == 0.0:
            return audio
        
        # Simple convolution reverb (using impulse response approximation)
        sr = self.config.output_sr
        reverb_length = int(sr * 2.0)  # 2 second reverb
        
        # Generate simple reverb impulse
        impulse = np.random.randn(reverb_length) * 0.3
        impulse[:int(sr * 0.1)] *= np.exp(-5 * np.linspace(0, 1, int(sr * 0.1)))
        
        # Apply reverb via convolution
        wet = np.convolve(audio, impulse, mode='same')
        
        # Mix wet and dry
        return audio * (1 - self.config.reverb_wet) + wet * self.config.reverb_wet
    
    def _apply_delay(self, audio: np.ndarray) -> np.ndarray:
        """Apply delay effect"""
        if self.config.delay_wet == 0.0:
            return audio
        
        sr = self.config.output_sr
        delay_time = 0.25  # 250ms
        delay_samples = int(sr * delay_time)
        
        # Create delayed signal
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        # Mix
        return audio + delayed * self.config.delay_wet
    
    def _generate_harmony(self, audio: np.ndarray) -> np.ndarray:
        """Generate harmony voices"""
        if self.config.harmony_voices == 0 or not librosa:
            return audio
        
        sr = self.config.output_sr
        harmonies = [audio]
        
        for i in range(self.config.harmony_voices):
            # Shift pitch for harmony
            shift = self.config.harmony_interval * (i + 1)
            harm = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
            
            # Slight volume reduction for harmony
            harm *= 0.7
            harmonies.append(harm)
        
        # Mix all voices
        return np.sum(harmonies, axis=0) / len(harmonies)
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        return audio
    
    def generate_from_lyrics(
        self,
        lyrics: str,
        melody: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> GeneratedVocal:
        """
        Generate vocals with melody line
        
        Args:
            lyrics: Lyrics text (can include timing markers)
            melody: Optional melody array (frequencies over time)
            output_path: Path to save output
            
        Returns:
            GeneratedVocal
        """
        # Split lyrics into lines/phrases
        lines = lyrics.split('\n')
        
        all_audio = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Generate this line
            vocal = self.generate(line)
            all_audio.append(vocal.audio)
            
            # Add small gap between lines
            gap = np.zeros(int(self.config.output_sr * 0.5))
            all_audio.append(gap)
        
        # Concatenate
        full_audio = np.concatenate(all_audio)
        
        # Apply tempo adjustment
        if self.config.tempo != 1.0 and librosa:
            full_audio = librosa.effects.time_stretch(full_audio, rate=self.config.tempo)
        
        result = GeneratedVocal(
            audio=full_audio,
            sample_rate=self.config.output_sr,
            config=self.config,
            metadata={"lyrics": lyrics},
            duration=len(full_audio) / self.config.output_sr
        )
        
        if output_path:
            sf.write(output_path, full_audio, self.config.output_sr)
            result.metadata["saved_to"] = output_path
        
        return result


class VocalStemExtractor:
    """Extract vocal stems from existing audio"""
    
    def __init__(self, model: str = "demucs"):
        self.model = model
    
    def extract(self, audio_path: str, output_dir: str = ".") -> Dict[str, str]:
        """
        Extract vocal stem from audio
        
        Returns:
            Dict with stem paths
        """
        output_path = Path(output_dir)
        
        if self.model == "demucs":
            return self._extract_demucs(audio_path, output_path)
        else:
            raise ValueError(f"Unknown model: {self.model}")
    
    def _extract_demucs(self, audio_path: str, output_path: Path) -> Dict[str, str]:
        """Extract using Demucs"""
        try:
            from demucs.separate import main as demucs_main
            
            # Run demucs
            args = [
                "-n", "htdemucs",
                "--two-stems", "vocals",
                "-o", str(output_path),
                audio_path
            ]
            demucs_main(args)
            
            # Find output
            filename = Path(audio_path).stem
            vocal_path = output_path / "htdemucs" / filename / "vocals.wav"
            
            return {"vocals": str(vocal_path)}
        except ImportError:
            print("Demucs not installed")
            return {}


def create_vocal_generator(
    model: str = "bark",
    voice: str = "v2/en_speaker_6",
    style: str = "pop"
) -> AIVocalsGenerator:
    """Factory function to create a vocal generator"""
    config = VocalConfig(
        model=VocalModel(model.lower()),
        voice_preset=voice,
        style=VoiceStyle(style.lower())
    )
    return AIVocalsGenerator(config)


# CLI
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Vocals Generator")
    parser.add_argument("command", choices=["generate", "extract"])
    parser.add_argument("input", help="Text file or audio file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--model", "-m", default="bark", choices=["bark", "coqui", "elevenlabs"])
    parser.add_argument("--voice", default="v2/en_speaker_6", help="Voice preset")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch shift (semitones)")
    parser.add_argument("--harmony", type=int, default=0, help="Number of harmony voices")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        # Read text input
        text = Path(args.input).read_text()
        
        config = VocalConfig(
            model=VocalModel(args.model),
            voice_preset=args.voice,
            pitch=args.pitch,
            harmony_voices=args.harmony
        )
        
        generator = AIVocalsGenerator(config)
        result = generator.generate(text, args.output)
        
        print(f"Generated {result.duration:.2f}s of audio")
        if args.output:
            print(f"Saved to: {args.output}")
    
    elif args.command == "extract":
        extractor = VocalStemExtractor()
        stems = extractor.extract(args.input, Path(args.output).parent if args.output else ".")
        print(json.dumps(stems, indent=2))


if __name__ == "__main__":
    main()
