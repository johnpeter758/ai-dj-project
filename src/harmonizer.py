#!/usr/bin/env python3
"""
Harmonizer for AI DJ Project
Generates harmonic arrangements to accompany melodies.
Supports various harmony types: parallel, drop 2, voice leading, counterpoint.
"""

import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class HarmonyType(Enum):
    PARALLEL_THIRDS = "parallel_thirds"
    PARALLEL_SIXTHS = "parallel_sixths"
    DROP_2 = "drop_2"
    ROOT_POSITION = "root_position"
    FIRST_INVERSION = "first_inversion"
    SECOND_INVERSION = "second_inversion"
    VOICE_LEADING = "voice_leading"
    OPEN = "open"
    CLOSED = "closed"


# Interval distances in semitones
INTERVALS = {
    "unison": 0,
    "minor_second": 1,
    "major_second": 2,
    "minor_third": 3,
    "major_third": 4,
    "perfect_fourth": 5,
    "tritone": 6,
    "perfect_fifth": 7,
    "minor_sixth": 8,
    "major_sixth": 9,
    "minor_seventh": 10,
    "major_seventh": 11,
    "octave": 12,
}


@dataclass
class HarmonyVoice:
    """A single voice/part in a harmony."""
    name: str
    notes: List[int]  # MIDI note numbers


@dataclass
class Harmony:
    """A complete harmonic arrangement with multiple voices."""
    voices: List[HarmonyVoice]
    intervals: List[int]  # Intervals above root for each voice
    
    def to_midi_streams(self) -> Dict[str, List[int]]:
        """Convert harmony to dict of voice -> MIDI notes."""
        return {v.name: v.notes for v in self.voices}


class Harmonizer:
    """Generate harmonies for melodies."""
    
    # Scale intervals (semitones from root)
    SCALES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "natural_minor": [0, 2, 3, 5, 7, 8, 10],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "pentatonic_major": [0, 2, 4, 7, 9],
        "pentatonic_minor": [0, 3, 5, 7, 10],
    }
    
    # Common chord tones
    CHORD_TONES = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "diminished": [0, 3, 6],
        "augmented": [0, 4, 8],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "dom7": [0, 4, 7, 10],
    }
    
    def __init__(self, root: str = "C", scale: str = "major"):
        self.root = self._note_to_number(root)
        self.scale_name = scale
        self.scale = self.SCALES.get(scale, self.SCALES["major"])
    
    def _note_to_number(self, note: str) -> int:
        """Convert note name to MIDI number (C4 = 60)."""
        notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        # Handle sharps/flats
        note = note.replace('#', '').replace('b', '')
        octave = 4
        if note[-1].isdigit():
            octave = int(note[-1])
            note = note[:-1]
        base = notes.get(note.upper(), 0)
        return (octave + 1) * 12 + base
    
    def _get_diatonic_chord(self, degree: int, inversion: int = 0) -> List[int]:
        """Get chord tones for a diatonic degree."""
        root = self.scale[degree % 7]
        third = self.scale[(degree + 2) % 7]
        fifth = self.scale[(degree + 4) % 7]
        
        # Handle octave wrapping
        if third < root:
            third += 12
        if fifth < root:
            fifth += 12
        
        chord = [root, third, fifth]
        
        # Apply inversion
        if inversion == 1 and len(chord) > 2:
            chord = [chord[1], chord[2], chord[0] + 12]
        elif inversion == 2 and len(chord) > 2:
            chord = [chord[2], chord[0] + 12, chord[1] + 12]
        
        return [n + self.root for n in chord]
    
    def generate_parallel_harmony(
        self,
        melody: List[int],
        interval: int = 4,
        voices: int = 2
    ) -> Harmony:
        """Generate parallel harmony (same contour, fixed interval)."""
        voice_notes = []
        
        for v in range(voices):
            voice = []
            for note in melody:
                # Add interval for this voice
                harmony_note = note + (interval * (v + 1))
                voice.append(harmony_note)
            voice_notes.append(voice)
        
        voices_list = [
            HarmonyVoice(name=f"voice_{i}", notes=notes)
            for i, notes in enumerate(voice_notes)
        ]
        
        return Harmony(
            voices=voices_list,
            intervals=[interval * (i + 1) for i in range(voices)]
        )
    
    def generate_drop_2_harmony(
        self,
        melody: List[int],
        num_voices: int = 4
    ) -> Harmony:
        """Generate drop 2 voicing (spread voicing common in jazz)."""
        # Start with root position
        chord = self._get_diatonic_chord(0)
        
        # Drop the 2nd note (from top) down an octave
        if len(chord) >= 2:
            chord[1] -= 12
        
        # Expand to requested number of voices
        while len(chord) < num_voices:
            # Add octave above
            chord.append(chord[0] + 12)
        
        # Apply to melody notes
        voice_notes = []
        for i in range(min(num_voices, len(chord))):
            voice = [chord[i] + (note - 60) for note in melody]
            voice_notes.append(voice)
        
        voices_list = [
            HarmonyVoice(name=f"voice_{i}", notes=notes)
            for i, notes in enumerate(voice_notes)
        ]
        
        return Harmony(
            voices=voices_list,
            intervals=list(range(num_voices))
        )
    
    def generate_voice_leaded_harmony(
        self,
        melody: List[int],
        num_voices: int = 3
    ) -> Harmony:
        """Generate voice-led harmony (smooth voice leading)."""
        voice_notes = [[] for _ in range(num_voices)]
        
        for i, root_note in enumerate(melody):
            # Get chord based on scale degree
            degree = (root_note - self.root) % 7
            chord = self._get_diatonic_chord(degree)
            
            # Assign notes to voices with smooth voice leading
            for v in range(num_voices):
                if v < len(chord):
                    voice_notes[v].append(chord[v])
                else:
                    # Fill extra voices with octave
                    voice_notes[v].append(chord[0] + 12 * (v - len(chord) + 1))
        
        voices_list = [
            HarmonyVoice(name=f"voice_{i}", notes=notes)
            for i, notes in enumerate(voice_notes)
        ]
        
        return Harmony(
            voices=voices_list,
            intervals=[0, 3, 7][:num_voices]
        )
    
    def generate_polyphonic_harmony(
        self,
        melody: List[int],
        style: str = "country"
    ) -> Harmony:
        """Generate polyphonic harmony (like country twin guitars)."""
        intervals_by_style = {
            "country": [4, 7],  # thirds and fifths
            "jazz": [4, 10],   # third and seventh
            "gospel": [7, 12],  # fifth and octave
            "rock": [5, 12],   # fourth and octave
        }
        
        intervals = intervals_by_style.get(style, [4, 7])
        
        voice_notes = []
        
        # Lead voice (original melody)
        voice_notes.append(melody)
        
        # Harmony voices
        for interval in intervals:
            voice = [n + interval for n in melody]
            voice_notes.append(voice)
        
        voices_list = [
            HarmonyVoice(name=f"voice_{i}", notes=notes)
            for i, notes in enumerate(voice_notes)
        ]
        
        return Harmony(
            voices=voices_list,
            intervals=[0] + intervals
        )
    
    def generate_block_chords(
        self,
        melody: List[int],
        chord_progression: List[str] = None
    ) -> Harmony:
        """Generate block chords (whole notes under melody)."""
        if chord_progression is None:
            chord_progression = ["I", "IV", "V", "I"]
        
        voice_notes = [[] for _ in range(3)]
        
        for i, note in enumerate(melody):
            # Determine chord from progression
            degree = i % len(chord_progression)
            chord_symbol = chord_progression[degree]
            
            # Get chord tones
            chord = self._get_diatonic_chord(self._roman_to_degree(chord_symbol))
            
            # Block chord: all notes play together
            for v in range(3):
                if v < len(chord):
                    voice_notes[v].append(chord[v])
                else:
                    voice_notes[v].append(chord[0])
        
        voices_list = [
            HarmonyVoice(name=f"voice_{i}", notes=notes)
            for i, notes in enumerate(voice_notes)
        ]
        
        return Harmony(
            voices=voices_list,
            intervals=[0, 4, 7]
        )
    
    def _roman_to_degree(self, roman: str) -> int:
        """Convert roman numeral to scale degree."""
        roman = roman.upper().strip()
        mapping = {
            "I": 0, "II": 1, "III": 2, "IV": 3,
            "V": 4, "VI": 5, "VII": 6
        }
        return mapping.get(roman, 0)
    
    def harmonize(
        self,
        melody: List[int],
        harmony_type: HarmonyType = HarmonyType.PARALLEL_THIRDS,
        num_voices: int = 3
    ) -> Harmony:
        """Main harmonization method."""
        if harmony_type == HarmonyType.PARALLEL_THIRDS:
            return self.generate_parallel_harmony(melody, interval=4, voices=num_voices)
        elif harmony_type == HarmonyType.PARALLEL_SIXTHS:
            return self.generate_parallel_harmony(melody, interval=9, voices=num_voices)
        elif harmony_type == HarmonyType.DROP_2:
            return self.generate_drop_2_harmony(melody, num_voices=num_voices)
        elif harmony_type == HarmonyType.VOICE_LEADING:
            return self.generate_voice_leaded_harmony(melody, num_voices=num_voices)
        elif harmony_type in [HarmonyType.ROOT_POSITION, 
                              HarmonyType.FIRST_INVERSION,
                              HarmonyType.SECOND_INVERSION]:
            inv = {"ROOT_POSITION": 0, "FIRST_INVERSION": 1, "SECOND_INVERSION": 2}
            chord = self._get_diatonic_chord(0, inversion=inv[harmony_type.name])
            voice_notes = [[n for n in chord] * len(melody)]
            return Harmony(
                voices=[HarmonyVoice(name="chord", notes=voice_notes[0][:len(melody)])],
                intervals=[0, 4, 7]
            )
        else:
            # Default to parallel thirds
            return self.generate_parallel_harmony(melody, interval=4, voices=num_voices)


def generate_demo_melody() -> List[int]:
    """Generate a simple demo melody."""
    # C major scale
    scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
    melody = []
    for _ in range(8):
        melody.append(random.choice(scale))
    return melody


def main():
    print("🎼 Harmonizer Demo\n")
    
    # Create harmonizer in C major
    h = Harmonizer(root="C", scale="major")
    
    # Generate demo melody
    melody = generate_demo_melody()
    print(f"Original melody (MIDI): {melody}")
    print(f"Notes: {[['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][n%12] for n in melody]}\n")
    
    # Test different harmony types
    harmony_types = [
        HarmonyType.PARALLEL_THIRDS,
        HarmonyType.PARALLEL_SIXTHS,
        HarmonyType.VOICE_LEADING,
    ]
    
    for ht in harmony_types:
        harmony = h.harmonize(melody, harmony_type=ht, num_voices=2)
        print(f"--- {ht.value} ---")
        for voice in harmony.voices:
            print(f"  {voice.name}: {voice.notes[:4]}...")


if __name__ == "__main__":
    main()
