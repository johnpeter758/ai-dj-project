#!/usr/bin/env python3
"""
Bass Line Generator
"""

import random

class BassGenerator:
    """Generate bass lines"""
    
    # Bass patterns by genre
    PATTERNS = {
        "house": [
            "1-1-1-1",     # Root on every beat
            "1-5-1-5",     # Root-fifth
            "1-1-5-5",     # Variation
            "1-3-5-7",     # Arpeggiated
        ],
        "techno": [
            "1-1-1-1",
            "1-1-2-2",
            "1-1-1-2",
            "1-0-1-0",     # Resting
        ],
        "trap": [
            "1---1---",     # Long notes
            "1-0-1-1",     # Offbeat
            "1-1-1-2",
            "1-3-5-3",      # Triplet feel
        ],
        "hip_hop": [
            "1-1-3-5",
            "1-3-1-3",
            "1-0-1-3",
            "1-5-1-5",
        ],
        "dubstep": [
            "1-0-0-0",     # Staccato
            "1-0-1-0",
            "1-2-1-2",
        ],
    }
    
    # Scale degrees
    SCALES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
    }
    
    def __init__(self, key: str = "C", scale: str = "minor"):
        self.key = key
        self.scale = scale
        self.root = self._get_root()
    
    def _get_root(self) -> int:
        """Get root note number"""
        notes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        return notes.get(self.key, 0)
    
    def generate(self, genre: str = "house", bars: int = 4) -> list:
        """Generate bass line"""
        patterns = self.PATTERNS.get(genre, self.PATTERNS["house"])
        pattern = random.choice(patterns)
        
        scale = self.SCALES.get(self.scale, [0, 2, 4, 5, 7, 9, 11])
        
        result = []
        for char in pattern:
            if char.isdigit():
                degree = int(char)
                if degree < len(scale):
                    note = self.root + scale[degree]
                    result.append(note)
                else:
                    result.append(self.root)
            else:
                result.append(None)  # Rest
        
        return result
    
    def to_notes(self, bass_line: list) -> list:
        """Convert to note events"""
        events = []
        for i, note in enumerate(bass_line):
            if note is not None:
                events.append({
                    "time": i * 0.5,
                    "note": 36 + note,  # Bass octave
                    "velocity": 100
                })
        return events

def main():
    gen = BassGenerator(key="C", scale="minor")
    
    print("🎸 Bass Line Generator\n")
    
    for genre in ["house", "techno", "trap", "hip_hop", "dubstep"]:
        bass = gen.generate(genre, bars=4)
        notes = gen.to_notes(bass)
        print(f"{genre.upper():10}: {bass}")

if __name__ == "__main__":
    main()
