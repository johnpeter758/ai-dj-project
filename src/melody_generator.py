#!/usr/bin/env python3
"""
Melody Generator for AI DJ Project
Generates melodies based on music theory scales with random walk constraints.
Outputs MIDI note numbers for integration with other generators.
"""

import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ScaleType(Enum):
    MAJOR = "major"
    NATURAL_MINOR = "natural_minor"
    HARMONIC_MINOR = "harmonic_minor"
    MAJOR_PENTATONIC = "major_pentatonic"
    MINOR_PENTATONIC = "minor_pentatonic"
    BLUES = "blues"
    DORIAN = "dorian"
    MIXOLYDIAN = "mixolydian"


# MIDI note numbers for one octave (C4 = 60)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class Note:
    """Represents a single note in a melody."""
    midi: int
    duration: float  # in beats
    velocity: int = 100  # 0-127
    
    @property
    def name(self) -> str:
        octave = (self.midi // 12) - 1
        note = NOTE_NAMES[self.midi % 12]
        return f"{note}{octave}"
    
    @property
    def pitch_class(self) -> int:
        return self.midi % 12


@dataclass
class Melody:
    """A sequence of notes forming a melody."""
    notes: List[Note]
    scale: ScaleType
    root_note: int  # MIDI note number of root
    
    def to_midi_numbers(self) -> List[int]:
        """Convert melody to list of MIDI note numbers."""
        return [n.midi for n in self.notes]
    
    def to_midi_with_durations(self) -> List[Tuple[int, float]]:
        """Convert melody to list of (midi, duration) tuples."""
        return [(n.midi, n.duration) for n in self.notes]
    
    def display(self) -> str:
        """Display melody as note names with durations."""
        return " | ".join([f"{n.name}({n.duration})" for n in self.notes])


class Scale:
    """Music theory scale with note intervals."""
    
    # Intervals in semitones from root
    INTERVALS = {
        ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
        ScaleType.NATURAL_MINOR: [0, 2, 3, 5, 7, 8, 10],
        ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
        ScaleType.MAJOR_PENTATONIC: [0, 2, 4, 7, 9],
        ScaleType.MINOR_PENTATONIC: [0, 3, 5, 7, 10],
        ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
        ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
        ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    }
    
    def __init__(self, root_midi: int, scale_type: ScaleType):
        self.root = root_midi
        self.type = scale_type
        self.intervals = self.INTERVALS[scale_type]
    
    def get_note_in_scale(self, scale_degree: int, octave: int = 4) -> int:
        """Get MIDI note for a scale degree in a given octave."""
        # Normalize scale degree to 0-6 range
        degree = scale_degree % len(self.intervals)
        octave_offset = (scale_degree // len(self.intervals)) + octave - 4
        return self.root + self.intervals[degree] + (octave_offset * 12)
    
    def get_scale_notes(self, octave: int = 4, octaves: int = 2) -> List[int]:
        """Get all notes in the scale across multiple octaves."""
        notes = []
        for oct in range(octave, octave + octaves):
            for degree in range(len(self.intervals)):
                notes.append(self.get_note_in_scale(degree, oct))
        return notes
    
    def is_in_scale(self, midi: int) -> bool:
        """Check if a MIDI note is in this scale."""
        pitch_class = midi % 12
        root_class = self.root % 12
        interval = (pitch_class - root_class) % 12
        return interval in self.intervals


class MelodyGenerator:
    """Generates melodies using scale-based random walk."""
    
    def __init__(
        self,
        root_note: int = 60,  # C4
        scale_type: ScaleType = ScaleType.MAJOR,
        tempo: int = 120,
        min_note: int = 48,  # C3
        max_note: int = 84,  # C6
    ):
        self.root_note = root_note
        self.scale_type = scale_type
        self.tempo = tempo
        self.min_note = min_note
        self.max_note = max_note
        self.scale = Scale(root_note, scale_type)
        
        # Random walk constraints
        self.max_step = 5  # Max semitones per step
        self.prefer_conjunct = 0.6  # Probability of small steps
        
        # Note duration options (in beats)
        self.rhythm_patterns = [
            [1.0],  # Quarter notes
            [0.5, 0.5],  # Eighth notes
            [1.0, 0.5, 0.5],  # Quarter + eighths
            [0.5, 0.5, 0.5, 0.5],  # Sixteenth notes
            [1.0, 1.0],  # Half notes
            [0.5, 1.0],  # Eighth + quarter
            [1.0, 0.25, 0.25, 0.5],  # Mixed
        ]
    
    def _get_next_note(self, current_midi: int, allow_repeat: bool = False) -> Optional[int]:
        """Generate next note using constrained random walk."""
        scale_notes = self.scale.get_scale_notes(octave=2, octaves=5)
        scale_notes = [n for n in scale_notes if self.min_note <= n <= self.max_note]
        
        if not scale_notes:
            return None
        
        # Decide step size: small (conjunct) or larger (disjunct)
        if random.random() < self.prefer_conjunct:
            # Small step: -3 to +3 semitones
            step = random.randint(-3, 3)
        else:
            # Larger step: -self.max_step to +self.max_step
            step = random.randint(-self.max_step, self.max_step)
        
        next_midi = current_midi + step
        
        # Clamp to range
        next_midi = max(self.min_note, min(self.max_note, next_midi))
        
        # If note not in scale, find nearest scale note
        if not self.scale.is_in_scale(next_midi):
            # Try to move to nearest scale note
            candidates = sorted(scale_notes, key=lambda n: abs(n - next_midi))
            next_midi = candidates[0]
        
        # Optionally allow repeat
        if not allow_repeat and next_midi == current_midi:
            return self._get_next_note(current_midi, allow_repeat)
        
        return next_midi
    
    def generate_melody(
        self,
        num_notes: int = 8,
        start_note: Optional[int] = None,
        rhythm: Optional[List[float]] = None,
    ) -> Melody:
        """Generate a melody using scale-based random walk."""
        notes = []
        
        # Start note: either specified or random in scale
        if start_note is None:
            scale_notes = self.scale.get_scale_notes(octave=3, octaves=2)
            scale_notes = [n for n in scale_notes if self.min_note <= n <= self.max_note]
            current = random.choice(scale_notes)
        else:
            current = start_note
        
        # Use provided rhythm or generate one
        if rhythm is None:
            rhythm = random.choice(self.rhythm_patterns)
        
        for i in num_notes * [0]:
            # Get duration from rhythm pattern (cycle through)
            duration = rhythm[i % len(rhythm)]
            
            # Add some rests occasionally (10% chance)
            if random.random() < 0.1:
                continue
            
            # Get next note
            current = self._get_next_note(current, allow_repeat=(i % 4 == 0))
            
            if current is not None:
                velocity = random.randint(80, 120)
                notes.append(Note(midi=current, duration=duration, velocity=velocity))
        
        return Melody(notes=notes, scale=self.scale_type, root_note=self.root_note)


class HookGenerator:
    """Generates hook melodies with repetition patterns."""
    
    def __init__(self, melody_generator: MelodyGenerator):
        self.gen = melody_generator
    
    def create_hook(
        self,
        bars: int = 4,
        beats_per_bar: int = 4,
        repetition_factor: int = 2,
        variation: float = 0.2,
    ) -> Melody:
        """
        Create a hook melody with repetition.
        
        Args:
            bars: Number of bars (2-4)
            beats_per_bar: Beats per bar (default 4)
            repetition_factor: How many times to repeat the motif
            variation: Amount of variation in repeated sections (0-1)
        """
        bars = max(2, min(4, bars))  # Clamp to 2-4 bars
        total_beats = bars * beats_per_bar
        
        # Generate base motif (half the length for repetition)
        motif_length = total_beats // repetition_factor
        base_melody = self.gen.generate_melody(
            num_notes=motif_length,
            rhythm=[1.0] * motif_length,
        )
        
        # Build hook by repeating with variation
        hook_notes = []
        for rep in range(repetition_factor):
            for note in base_melody.notes:
                if rep > 0 and variation > 0:
                    # Apply variation: slight pitch shift or rhythm change
                    new_midi = note.midi
                    if random.random() < variation:
                        # Transpose up/down by semitone
                        new_midi += random.choice([-1, 1])
                        # Ensure still in scale
                        if not self.gen.scale.is_in_scale(new_midi):
                            new_midi = note.midi
                    
                    new_duration = note.duration
                    if random.random() < variation * 0.5:
                        # Slightly alter duration
                        new_duration = max(0.25, note.duration * random.choice([0.5, 1.0, 1.5]))
                    
                    hook_notes.append(Note(
                        midi=new_midi,
                        duration=new_duration,
                        velocity=note.velocity,
                    ))
                else:
                    hook_notes.append(note)
        
        return Melody(
            notes=hook_notes,
            scale=self.gen.scale_type,
            root_note=self.gen.root_note,
        )


def generate_example():
    """Generate example melodies demonstrating the generator."""
    print("=" * 60)
    print("MELODY GENERATOR EXAMPLES")
    print("=" * 60)
    
    # Example 1: Major scale melody
    print("\n1. Major Scale Melody (C Major, starting on C4):")
    gen = MelodyGenerator(
        root_note=60,  # C4
        scale_type=ScaleType.MAJOR,
        tempo=120,
    )
    melody = gen.generate_melody(num_notes=8)
    print(f"   Scale: {gen.scale_type.value}")
    print(f"   MIDI: {melody.to_midi_numbers()}")
    print(f"   Notes: {melody.display()}")
    
    # Example 2: Minor pentatonic melody
    print("\n2. Minor Pentatonic Melody (A Minor, starting on A3):")
    gen2 = MelodyGenerator(
        root_note=57,  # A3
        scale_type=ScaleType.MINOR_PENTATONIC,
        tempo=100,
    )
    melody2 = gen2.generate_melody(num_notes=8)
    print(f"   Scale: {gen2.scale_type.value}")
    print(f"   MIDI: {melody2.to_midi_numbers()}")
    print(f"   Notes: {melody2.display()}")
    
    # Example 3: Blues scale
    print("\n3. Blues Melody (C Blues):")
    gen3 = MelodyGenerator(
        root_note=60,
        scale_type=ScaleType.BLUES,
        tempo=80,
    )
    melody3 = gen3.generate_melody(num_notes=8)
    print(f"   Scale: {gen3.scale_type.value}")
    print(f"   MIDI: {melody3.to_midi_numbers()}")
    print(f"   Notes: {melody3.display()}")
    
    # Example 4: Hook creation
    print("\n4. Hook Melody (4 bars, 2x repetition with variation):")
    hook_gen = HookGenerator(gen)
    hook = hook_gen.create_hook(bars=4, repetition_factor=2, variation=0.15)
    print(f"   Scale: {hook.scale.value}")
    print(f"   MIDI: {hook.to_midi_numbers()}")
    print(f"   Notes: {hook.display()}")
    
    # Example 5: Different root, harmonic minor
    print("\n5. Harmonic Minor Melody (E Harmonic Minor):")
    gen4 = MelodyGenerator(
        root_note=64,  # E4
        scale_type=ScaleType.HARMONIC_MINOR,
    )
    melody4 = gen4.generate_melody(num_notes=8)
    print(f"   Scale: {gen4.scale_type.value}")
    print(f"   MIDI: {melody4.to_midi_numbers()}")
    print(f"   Notes: {melody4.display()}")
    
    print("\n" + "=" * 60)
    print("All melodies ready for MIDI conversion!")
    print("=" * 60)
    
    return {
        'major': melody,
        'minor_penta': melody2,
        'blues': melody3,
        'hook': hook,
        'harmonic_minor': melody4,
    }


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI number to note name."""
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


if __name__ == "__main__":
    # Run examples
    melodies = generate_example()
    
    # Demonstrate MIDI output format
    print("\nMIDI OUTPUT FORMAT:")
    print("-" * 40)
    for name, melody in melodies.items():
        print(f"\n{name}:")
        for midi, dur in melody.to_midi_with_durations():
            print(f"  Note: {midi_to_note_name(midi):3s} (MIDI {midi:3d}) - {dur} beats")
