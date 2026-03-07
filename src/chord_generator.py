#!/usr/bin/env python3
"""
Chord Progression Generator
"""

import random

class ChordProgression:
    """Generate chord progressions"""
    
    # Chord progressions by genre
    PROGRESSIONS = {
        "pop": [
            "I-V-vi-IV",      # Classic pop
            "I-IV-V-I",        # Basic
            "vi-IV-I-V",       # 50s
            "I-vi-IV-V",       # Canon
        ],
        "edm": [
            "i-VII-VI-V",
            "i-III-VII-IV", 
            "i-IV-vii-III",
            "I-bVII-IV",
        ],
        "hip_hop": [
            "I-IV-V-I",
            "I-VI-IV-V",
            "i-bVII-bVI-V",
        ],
        "jazz": [
            "ii-V-I",
            "I-vi-ii-V",
            "IVmaj7-3/4",
            "Imaj7-vi7-ii7-V7",
        ],
        "rock": [
            "I-IV-V-I",
            "I-bVII-IV",
            "i-IV-V-i",
            "I-V-bVII-IV",
        ],
    }
    
    # Chord tones
    CHORDS = {
        "I": [0, 4, 7],
        "i": [0, 3, 7],
        "II": [2, 5, 9],
        "ii": [2, 5, 9],
        "III": [4, 7, 11],
        "iii": [4, 7, 10],
        "IV": [5, 9, 12],
        "iv": [5, 8, 12],
        "V": [7, 11, 14],
        "v": [7, 10, 14],
        "VI": [9, 12, 16],
        "vi": [9, 12, 15],
        "VII": [11, 14, 17],
        "vii": [10, 14, 17],
        "bVII": [10, 14, 17],
        "bVI": [8, 12, 15],
        "bIII": [3, 7, 10],
    }
    
    def __init__(self, key: str = "C"):
        self.key = key
    
    def generate(self, genre: str = "pop", length: int = 8) -> list:
        """Generate chord progression"""
        progressions = self.PROGRESSIONS.get(genre, self.PROGRESSIONS["pop"])
        progression = random.choice(progressions)
        
        chords = progression.split("-")
        
        # Repeat to fill length
        result = []
        while len(result) < length:
            for chord in chords:
                result.append(chord)
                if len(result) >= length:
                    break
        
        return result[:length]
    
    def to_notes(self, chord: str, octave: int = 4) -> list:
        """Convert chord to note numbers"""
        return [n + octave * 12 for n in self.CHORDS.get(chord, [0, 4, 7])]
    
    def get_roman(self, chord: str) -> str:
        """Get roman numeral representation"""
        return chord

def main():
    gen = ChordProgression(key="C")
    
    print("🎹 Chord Progression Generator\n")
    
    for genre in ["pop", "edm", "hip_hop", "jazz", "rock"]:
        progression = gen.generate(genre, length=8)
        print(f"{genre.upper():10}: {' - '.join(progression)}")

if __name__ == "__main__":
    main()
