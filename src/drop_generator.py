"""
EDM Drop Generator
==================
A Python module for generating professional EDM drop sections with
build-up, drop, and release sections using numpy for audio synthesis.

Features:
- Build-up section with risers and white noise sweeps
- Drop section with bass, synths, and drums
- Release/outro section
- Energy curve tracking throughout the arrangement
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class GenreStyle(Enum):
    """EDM sub-genres with different frequency profiles."""
    HOUSE = "house"
    TECHNO = "techno"
    DRUM_BASS = "drum_bass"
    TRAP = "trap"
    FUTURE_BASS = "future_bass"


@dataclass
class TrackConfig:
    """Configuration for a single audio track."""
    sample_rate: int = 44100
    duration: float = 8.0  # seconds per section
    bpm: int = 128


@dataclass
class DropConfig:
    """Configuration for the entire drop section."""
    build_up_duration: float = 4.0  # seconds
    drop_duration: float = 8.0
    release_duration: float = 4.0
    genre: GenreStyle = GenreStyle.HOUSE
    bpm: int = 128


class EnergyCurve:
    """
    Tracks energy levels throughout the track using an envelope follower.
    Energy ranges from 0.0 (silence) to 1.0 (full intensity).
    """
    
    def __init__(self, sample_rate: int, duration: float):
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples = int(sample_rate * duration)
        self.energy = np.zeros(self.samples)
    
    def add_build_up(self, duration: float, curve: str = "exponential") -> None:
        """Add energy ramp during build-up section."""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, 1, n_samples)
        
        if curve == "exponential":
            ramp = np.power(t, 2)
        elif curve == "linear":
            ramp = t
        elif curve == "sigmoid":
            ramp = 1 / (1 + np.exp(-10 * (t - 0.5)))
        else:
            ramp = t
        
        # Scale to target peak energy (0.85 for build-up)
        ramp = ramp * 0.85
        
        # Add to energy array
        self.energy[:n_samples] = ramp
    
    def add_drop(self, duration: float, peak_energy: float = 1.0) -> None:
        """Set high energy during drop section."""
        n_samples = int(self.sample_rate * duration)
        
        # Calculate offset (after build-up)
        offset = np.sum(self.energy > 0)
        
        # Apply drop energy with slight variation
        base_energy = peak_energy * 0.95
        variation = np.random.uniform(-0.05, 0.05, n_samples)
        drop_energy = base_energy + variation
        
        self.energy[offset:offset + n_samples] = drop_energy
    
    def add_release(self, duration: float, curve: str = "exponential") -> None:
        """Add energy decay during release/outro."""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(1, 0, n_samples)
        
        if curve == "exponential":
            decay = np.power(t, 2)
        elif curve == "linear":
            decay = t
        else:
            decay = t
        
        # Start from current energy level
        offset = np.sum(self.energy > 0)
        start_energy = self.energy[offset - 1] if offset > 0 else 1.0
        
        self.energy[offset:offset + n_samples] = start_energy * decay
    
    def get_segment(self, start: float, end: float) -> np.ndarray:
        """Get energy values for a time segment."""
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate)
        return self.energy[start_idx:end_idx]
    
    def plot(self) -> None:
        """Visualize energy curve with matplotlib."""
        import matplotlib.pyplot as plt
        
        t = np.linspace(0, self.duration, self.samples)
        plt.figure(figsize=(12, 4))
        plt.plot(t, self.energy, 'cyan', linewidth=2)
        plt.fill_between(t, self.energy, alpha=0.3, color='cyan')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.title('EDM Drop Energy Curve')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.show()


class Riser:
    """
    Creates rising sound effects (risers) for build-up sections.
    Uses filtered noise with increasing frequency content.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def generate(
        self,
        duration: float,
        start_freq: float = 200,
        end_freq: float = 8000,
        num_harmonics: int = 8
    ) -> np.ndarray:
        """
        Generate a rising synth riser.
        
        Args:
            duration: Length in seconds
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz  
            num_harmonics: Number of harmonics to layer
        
        Returns:
            Normalized audio array
        """
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        
        # Frequency envelope (logarithmic rise sounds more musical)
        freq_envelope = np.exp(
            np.linspace(np.log(start_freq), np.log(end_freq), n_samples)
        )
        
        # Base oscillator with frequency modulation
        signal = np.zeros(n_samples)
        
        for harmonic in range(1, num_harmonics + 1):
            # Each harmonic rises at a slightly different rate
            harmonic_freq = freq_envelope * (1 + harmonic * 0.1)
            phase = 2 * np.pi * np.cumsum(harmonic_freq) / self.sample_rate
            signal += np.sin(phase * harmonic) / harmonic
        
        # Add some noise for texture
        noise = np.random.randn(n_samples) * 0.1
        window = int(self.sample_rate / start_freq)
        noise = np.convolve(noise, np.ones(window) / window, mode='same') * 2
        
        signal += noise
        
        # Apply envelope (fade in)
        envelope = np.linspace(0, 1, n_samples)
        signal = signal * envelope
        
        return self._normalize(signal)
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize to -1 to 1 range."""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val
        return signal


class WhiteNoiseSweep:
    """Creates white noise sweeps for build-up texture."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def generate(
        self,
        duration: float,
        start_freq: float = 200,
        end_freq: float = 10000,
        fade_in: float = 0.1,
        fade_out: float = 0.1
    ) -> np.ndarray:
        """
        Generate filtered white noise with frequency sweep.
        
        Args:
            duration: Length in seconds
            start_freq: Starting cutoff frequency (Hz)
            end_freq: Ending cutoff frequency (Hz)
            fade_in: Fade in time as fraction of duration
            fade_out: Fade out time as fraction of duration
        
        Returns:
            Normalized audio array
        """
        n_samples = int(self.sample_rate * duration)
        
        # White noise
        noise = np.random.randn(n_samples) * 0.5
        
        # Frequency envelope for filter
        freq_envelope = np.linspace(start_freq, end_freq, n_samples)
        
        # Apply crude lowpass filter with moving cutoff
        filtered = np.zeros(n_samples)
        window = int(self.sample_rate / 1000)
        
        for i in range(n_samples):
            cutoff = freq_envelope[i]
            window_size = max(1, int(self.sample_rate / cutoff))
            start = max(0, i - window_size)
            filtered[i] = np.mean(noise[start:i + 1])
        
        # Apply envelope (fade in/out)
        fade_in_samples = int(n_samples * fade_in)
        fade_out_samples = int(n_samples * fade_out)
        
        envelope = np.ones(n_samples)
        if fade_in_samples > 0:
            envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
        if fade_out_samples > 0:
            envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
        
        return self._normalize(filtered * envelope)
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val
        return signal


class BassGenerator:
    """Generates bass sounds for the drop section."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def generate(
        self,
        duration: float,
        note: str = "A1",
        genre: GenreStyle = GenreStyle.HOUSE,
        distortion: float = 0.3
    ) -> np.ndarray:
        """Generate a bass note."""
        freq = self._note_to_freq(note)
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        
        # Base oscillator
        signal = np.sin(2 * np.pi * freq * t)
        
        # Add sub harmonics based on genre
        if genre == GenreStyle.TECHNO:
            signal += 0.5 * np.sin(2 * np.pi * freq * 0.5 * t)
        elif genre == GenreStyle.TRAP:
            signal = self._apply_distortion(signal, distortion)
            signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        elif genre == GenreStyle.DRUM_BASS:
            signal = self._apply_distortion(signal, distortion * 1.5)
            signal += 0.4 * np.sin(2 * np.pi * freq * 2 * t)
        
        # ADSR envelope
        attack = int(0.01 * n_samples)
        decay = int(0.1 * n_samples)
        sustain = int(0.7 * n_samples)
        release = int(0.2 * n_samples)
        
        envelope = np.zeros(n_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack + decay] = np.linspace(1, 0.7, decay)
        envelope[attack + decay:attack + decay + sustain] = 0.7
        envelope[-release:] = np.linspace(0.7, 0, release)
        
        return self._normalize(signal * envelope)
    
    def generate_pattern(
        self,
        duration: float,
        pattern: str = "x...x...x...x.",
        note: str = "A1",
        genre: GenreStyle = GenreStyle.HOUSE
    ) -> np.ndarray:
        """Generate a bass pattern from a string pattern."""
        n_samples = int(self.sample_rate * duration)
        signal = np.zeros(n_samples)
        
        note_length = len(pattern)
        samples_per_beat = n_samples // note_length
        
        for i, char in enumerate(pattern):
            if char.lower() == 'x':
                start = i * samples_per_beat
                note_samples = int(samples_per_beat * 0.9)
                note_signal = self.generate(
                    samples_per_beat / self.sample_rate, note, genre
                )[:note_samples]
                signal[start:start + len(note_signal)] += note_signal
        
        return self._normalize(signal)
    
    def _note_to_freq(self, note: str) -> float:
        """Convert note name to frequency."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = note.upper()
        
        if '#' in note:
            note_name, octave = note[:2], int(note[2:])
        else:
            note_name, octave = note[0], int(note[1:])
        
        semitone = notes.index(note_name)
        a4_offset = (semitone - 9) + (octave - 4) * 12
        return 440 * (2 ** (a4_offset / 12))
    
    def _apply_distortion(self, signal: np.ndarray, amount: float) -> np.ndarray:
        """Apply soft clipping distortion."""
        return np.tanh(signal * (1 + amount * 5))
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(signal))
        return signal / max_val if max_val > 0 else signal


class SynthGenerator:
    """Generates synth lead/pad sounds for the drop."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def generate(
        self,
        duration: float,
        notes: list = None,
        waveform: str = "sawtooth",
        genre: GenreStyle = GenreStyle.HOUSE
    ) -> np.ndarray:
        """Generate a synth sound."""
        if notes is None:
            notes = ["C4"]
        
        n_samples = int(self.sample_rate * duration)
        signal = np.zeros(n_samples)
        note_duration = duration / len(notes)
        
        for i, note in enumerate(notes):
            start = int(i * note_duration * self.sample_rate)
            note_signal = self._generate_waveform(note_duration, note, waveform, genre)
            signal[start:start + len(note_signal)] += note_signal
        
        return self._normalize(signal)
    
    def _generate_waveform(self, duration: float, note: str, waveform: str, genre: GenreStyle) -> np.ndarray:
        """Generate a single note with specified waveform."""
        freq = self._note_to_freq(note)
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        
        if waveform == "sine":
            signal = np.sin(2 * np.pi * freq * t)
        elif waveform == "square":
            signal = np.sign(np.sin(2 * np.pi * freq * t))
        elif waveform == "sawtooth":
            signal = 2 * (t * freq - np.floor(t * freq + 0.5))
        elif waveform == "triangle":
            signal = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
        else:
            signal = np.sin(2 * np.pi * freq * t)
        
        if genre == GenreStyle.FUTURE_BASS:
            signal += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
            signal += 0.25 * np.sin(2 * np.pi * freq * 3 * t)
        
        # ADSR envelope
        attack = int(0.02 * n_samples)
        decay = int(0.1 * n_samples)
        sustain_level = 0.6 if genre == GenreStyle.FUTURE_BASS else 0.7
        sustain = int(0.6 * n_samples)
        release = int(0.2 * n_samples)
        
        envelope = np.zeros(n_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)
        envelope[attack + decay:attack + decay + sustain] = sustain_level
        envelope[-release:] = np.linspace(sustain_level, 0, release)
        
        return signal * envelope
    
    def _note_to_freq(self, note: str) -> float:
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = note[:2] if '#' in note else note[0]
        octave = int(note[2 if '#' in note else 1:])
        semitone = notes.index(note_name)
        return 440 * (2 ** ((semitone - 9) + (octave - 4) * 12) / 12)
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(signal))
        return signal / max_val if max_val > 0 else signal


class DrumGenerator:
    """Generates kick, snare, and hi-hat patterns."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def kick(self, duration: float, pitch: float = 150, pitch_decay: float = 0.05, punch: float = 0.8) -> np.ndarray:
        """Generate a kick drum."""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        freq = pitch * np.exp(-pitch_decay * t * 20)
        signal = np.sin(2 * np.pi * freq * t)
        signal += np.random.randn(n_samples) * punch * np.exp(-t * 50)
        return self._normalize(signal * np.exp(-t * 8))
    
    def snare(self, duration: float, tone_freq: float = 200, noise_amount: float = 0.5) -> np.ndarray:
        """Generate a snare drum."""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        tone = np.sin(2 * np.pi * tone_freq * t)
        noise = np.random.randn(n_samples)
        window = int(self.sample_rate / 1000)
        noise = np.convolve(noise, np.ones(window) / window, mode='same')
        signal = tone * (1 - noise_amount) + noise * noise_amount
        return self._normalize(signal * np.exp(-t * 15))
    
    def hihat(self, duration: float, open_ratio: float = 0.2) -> np.ndarray:
        """Generate a hi-hat."""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        noise = np.random.randn(n_samples)
        window = int(self.sample_rate / 7000)
        noise = noise - np.convolve(noise, np.ones(window) / window, mode='same')
        decay_rate = 8 if open_ratio > 0.5 else 40
        return self._normalize(noise * np.exp(-t * decay_rate) * 0.3)
    
    def generate_pattern(self, duration: float, bpm: int = 128, pattern: str = "kkkk") -> np.ndarray:
        """Generate a drum pattern. k=kick, s=snare, h=closed hat, H=open hat"""
        n_samples = int(self.sample_rate * duration)
        signal = np.zeros(n_samples)
        samples_per_beat = int(self.sample_rate * 60.0 / bpm)
        
        for i, char in enumerate(pattern):
            start = i * samples_per_beat
            end = min((i + 1) * samples_per_beat, n_samples)
            if char == 'k':
                signal[start:end] += self.kick(samples_per_beat / self.sample_rate)
            elif char == 's':
                signal[start:end] += self.snare(samples_per_beat / self.sample_rate)
            elif char == 'h':
                signal[start:end] += self.hihat(samples_per_beat / self.sample_rate, 0.1)
            elif char == 'H':
                signal[start:end] += self.hihat(samples_per_beat / self.sample_rate, 0.8)
        
        return self._normalize(signal)
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(signal))
        return signal / max_val if max_val > 0 else signal


class DropGenerator:
    """
    Main class for generating complete EDM drops.
    Combines all elements into a cohesive section.
    """
    
    def __init__(self, config: DropConfig = None):
        self.config = config or DropConfig()
        self.sample_rate = 44100
        
        self.riser = Riser(self.sample_rate)
        self.noise_sweep = WhiteNoiseSweep(self.sample_rate)
        self.bass = BassGenerator(self.sample_rate)
        self.synth = SynthGenerator(self.sample_rate)
        self.drums = DrumGenerator(self.sample_rate)
        
        self.total_duration = (
            self.config.build_up_duration +
            self.config.drop_duration +
            self.config.release_duration
        )
        
        self.energy = EnergyCurve(self.sample_rate, self.total_duration)
    
    def generate_build_up(self) -> np.ndarray:
        """Generate the build-up section with risers and sweeps."""
        duration = self.config.build_up_duration
        
        riser_signal = self.riser.generate(duration, start_freq=200, end_freq=8000, num_harmonics=6)
        noise_signal = self.noise_sweep.generate(duration, start_freq=500, end_freq=8000, fade_in=0.05, fade_out=0.0)
        
        build_up = riser_signal * 0.7 + noise_signal * 0.3
        self.energy.add_build_up(duration, curve="sigmoid")
        
        return build_up
    
    def generate_drop(self) -> np.ndarray:
        """Generate the main drop section with bass, synths, and drums."""
        duration = self.config.drop_duration
        genre = self.config.genre
        
        # Bass patterns by genre
        bass_patterns = {
            GenreStyle.HOUSE: "x...x...x...x.",
            GenreStyle.TECHNO: "x...x...x...x.",
            GenreStyle.DRUM_BASS: "x.xx.x.xx.x.xx.",
            GenreStyle.TRAP: "x...x...x...x.",
            GenreStyle.FUTURE_BASS: "x..x..x..x..x..",
        }
        
        bass_notes = {
            GenreStyle.HOUSE: "A1", GenreStyle.TECHNO: "A1",
            GenreStyle.DRUM_BASS: "A1", GenreStyle.TRAP: "A1",
            GenreStyle.FUTURE_BASS: "C2",
        }
        
        pattern = bass_patterns.get(genre, "x...x...x...x.")
        note = bass_notes.get(genre, "A1")
        bass_signal = self.bass.generate_pattern(duration, pattern=pattern, note=note, genre=genre)
        
        # Synth notes by genre
        synth_notes = {
            GenreStyle.HOUSE: ["C4", "E4", "G4", "C5"],
            GenreStyle.TECHNO: ["C4", "G4"],
            GenreStyle.DRUM_BASS: ["C4", "D4", "E4", "F4"],
            GenreStyle.TRAP: ["C4", "G4", "C5"],
            GenreStyle.FUTURE_BASS: ["C4", "E4", "G4", "B4", "C5"],
        }
        
        notes = synth_notes.get(genre, ["C4", "E4", "G4"])
        synth_signal = self.synth.generate(duration, notes=notes, waveform="sawtooth", genre=genre)
        
        # Drum patterns by genre
        drum_patterns = {
            GenreStyle.HOUSE: "kkkk", GenreStyle.TECHNO: "kkkk",
            GenreStyle.DRUM_BASS: "kkkk", GenreStyle.TRAP: "kkss",
            GenreStyle.FUTURE_BASS: "kkkh",
        }
        
        drum_pattern = drum_patterns.get(genre, "kkkk")
        drums_signal = self.drums.generate_pattern(duration, bpm=self.config.bpm, pattern=drum_pattern)
        
        # Mix all elements
        drop = bass_signal * 0.5 + synth_signal * 0.3 + drums_signal * 0.4
        self.energy.add_drop(duration, peak_energy=1.0)
        
        return drop
    
    def generate_release(self) -> np.ndarray:
        """Generate the release/outro section."""
        duration = self.config.release_duration
        
        release = self.noise_sweep.generate(duration, start_freq=8000, end_freq=200, fade_in=0.0, fade_out=0.3)
        
        # Add delay tail
        delay_samples = int(self.sample_rate * 0.25)
        delayed = np.zeros(int(self.sample_rate * duration))
        delayed[delay_samples:] = release[:-delay_samples] * 0.3
        
        release = release + delayed
        self.energy.add_release(duration, curve="exponential")
        
        return release * 0.6
    
    def generate(self) -> Tuple[np.ndarray, EnergyCurve]:
        """Generate the complete drop section."""
        build_up = self.generate_build_up()
        drop = self.generate_drop()
        release = self.generate_release()
        
        full_drop = np.concatenate([build_up, drop, release])
        return full_drop, self.energy
    
    def generate_with_callback(self, callback: Optional[Callable[[str, np.ndarray], None]] = None) -> Tuple[np.ndarray, EnergyCurve]:
        """Generate drop with progress callback."""
        if callback:
            callback("build_up", self.generate_build_up())
            callback("drop", self.generate_drop())
            callback("release", self.generate_release())
            full_drop = np.concatenate([
                self.energy.energy[:int(self.sample_rate * self.config.build_up_duration)],
                self.energy.energy[int(self.sample_rate * self.config.build_up_duration):int(self.sample_rate * (self.config.build_up_duration + self.config.drop_duration))],
                self.energy.energy[int(self.sample_rate * (self.config.build_up_duration + self.config.drop_duration)):]
            ])
            return full_drop[:int(self.sample_rate * self.total_duration)], self.energy
        
        return self.generate()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("EDM Drop Generator - Example Usage")
    print("=" * 50)
    
    # Basic example
    config = DropConfig(
        build_up_duration=4.0,
        drop_duration=8.0,
        release_duration=4.0,
        genre=GenreStyle.HOUSE,
        bpm=128
    )
    
    generator = DropGenerator(config)
    audio, energy = generator.generate()
    
    print(f"\nGenerated drop:")
    print(f"  - Duration: {len(audio) / generator.sample_rate:.2f}s")
    print(f"  - Sample rate: {generator.sample_rate} Hz")
    print(f"  - Samples: {len(audio)}")
    print(f"  - Build-up: {config.build_up_duration}s")
    print(f"  - Drop: {config.drop_duration}s")
    print(f"  - Release: {config.release_duration}s")
    
    print(f"\nEnergy curve stats:")
    print(f"  - Min energy: {np.min(energy.energy):.3f}")
    print(f"  - Max energy: {np.max(energy.energy):.3f}")
    print(f"  - Mean energy: {np.mean(energy.energy):.3f}")
    
    # Genre comparison
    print("\n" + "=" * 50)
    print("Genre Comparison")
    print("=" * 50)
    
    for genre in [GenreStyle.HOUSE, GenreStyle.TECHNO, GenreStyle.DRUM_BASS, GenreStyle.TRAP, GenreStyle.FUTURE_BASS]:
        config = DropConfig(build_up_duration=2.0, drop_duration=4.0, release_duration=2.0, genre=genre, bpm=128)
        gen = DropGenerator(config)
        audio, _ = gen.generate()
        print(f"  {genre.value}: {len(audio)/gen.sample_rate:.2f}s, peak={np.max(np.abs(audio)):.3f}")
    
    # Individual components
    print("\n" + "=" * 50)
    print("Individual Components")
    print("=" * 50)
    
    sample_rate = 44100
    duration = 2.0
    
    riser = Riser(sample_rate)
    riser_audio = riser.generate(duration, start_freq=200, end_freq=8000)
    print(f"  Riser: {len(riser_audio)} samples")
    
    bass = BassGenerator(sample_rate)
    bass_audio = bass.generate_pattern(duration, pattern="x...x...x...x.", note="A1", genre=GenreStyle.TECHNO)
    print(f"  Bass: {len(bass_audio)} samples")
    
    synth = SynthGenerator(sample_rate)
    synth_audio = synth.generate(duration, notes=["C4", "E4", "G4"], waveform="sawtooth", genre=GenreStyle.FUTURE_BASS)
    print(f"  Synth: {len(synth_audio)} samples")
    
    drums = DrumGenerator(sample_rate)
    drums_audio = drums.generate_pattern(duration, bpm=128, pattern="kkss")
    print(f"  Drums: {len(drums_audio)} samples")
    
    print("\nDone! Run energy.plot() to visualize the energy curve.")
