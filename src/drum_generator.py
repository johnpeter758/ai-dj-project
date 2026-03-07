#!/usr/bin/env python3
"""
Drum Pattern Generator
"""

import random

class DrumPattern:
    """Generate drum patterns"""
    
    # Standard drum sounds
    KICK = "K"
    SNARE = "S"
    HIHAT = "H"
    OPEN_HAT = "O"
    CLAP = "C"
    
    PATTERNS = {
        "house": [
            "K...H...S.H...",
            "K.H.K.H.S.H...",
            "K...H...S.HHH",
        ],
        "techno": [
            "K...K...K...K...",
            "K.H.K.H.K.H.K.H",
            "K.O.K.O.K.O.K.O",
        ],
        "trap": [
            "K..H.K.S.K.H.K..S",
            "K.H.H.K.S.H.H.K.S",
            ".H..H.H..S..H.H..",
        ],
        "hip_hop": [
            "K....S..K....S..",
            "K..H..S.K..H..S.",
            "K.H..S..K.H..S..",
        ],
        "dubstep": [
            "K.......K.......",
            "K..O...K..O.....",
            ".H.H..S..H.H..S.",
        ],
    }
    
    def __init__(self, bpm: int = 120):
        self.bpm = bpm
    
    def generate(self, genre: str = "house", bars: int = 1) -> str:
        """Generate drum pattern"""
        patterns = self.PATTERNS.get(genre, self.PATTERNS["house"])
        pattern = random.choice(patterns)
        
        # Repeat for bars
        result = pattern * bars
        return result
    
    def add_swing(self, pattern: str, amount: float = 0.5) -> str:
        """Add swing to pattern"""
        # Simplified swing - just return for now
        return pattern
    
    def to_midi(self, pattern: str) -> list:
        """Convert pattern to MIDI-like events"""
        events = []
        for i, char in enumerate(pattern):
            if char == "K":
                events.append({"time": i * 0.25, "note": 36, "velocity": 100})  # Kick
            elif char == "S":
                events.append({"time": i * 0.25, "note": 38, "velocity": 90})   # Snare
            elif char == "H":
                events.append({"time": i * 0.25, "note": 42, "velocity": 70})   # Hi-hat
            elif char == "O":
                events.append({"time": i * 0.25, "note": 46, "velocity": 80})   # Open hat
            elif char == "C":
                events.append({"time": i * 0.25, "note": 39, "velocity": 95})   # Clap
        return events

def main():
    gen = DrumPattern(bpm=128)
    
    print("🎤 Drum Pattern Generator\n")
    
    for genre in ["house", "techno", "trap", "hip_hop", "dubstep"]:
        pattern = gen.generate(genre, bars=2)
        events = gen.to_midi(pattern)
        print(f"\n{genre.upper()}:")
        print(f"  Pattern: {pattern}")
        print(f"  Events: {len(events)}")

if __name__ == "__main__":
    main()
