#!/usr/bin/env python3
"""
Arrangement Generator - Creates song structures
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import random

class SectionType(Enum):
    INTRO = "intro"
    VERSE = "verse"
    PRECHORUS = "prechorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    DROP = "drop"
    HOOK = "hook"
    OUTRO = "outro"

@dataclass
class Section:
    type: SectionType
    bars: int
    energy_start: float
    energy_end: float

class ArrangementGenerator:
    """Generate song arrangements"""
    
    TEMPLATES = {
        "pop": [
            Section(SectionType.INTRO, 8, 0.2, 0.3),
            Section(SectionType.VERSE, 16, 0.3, 0.45),
            Section(SectionType.PRECHORUS, 8, 0.45, 0.75),
            Section(SectionType.CHORUS, 16, 0.8, 0.85),
            Section(SectionType.VERSE, 16, 0.35, 0.45),
            Section(SectionType.PRECHORUS, 8, 0.5, 0.8),
            Section(SectionType.CHORUS, 16, 0.85, 0.9),
            Section(SectionType.BRIDGE, 8, 0.4, 0.5),
            Section(SectionType.CHORUS, 16, 0.9, 0.85),
            Section(SectionType.OUTRO, 8, 0.4, 0.1),
        ],
        "edm": [
            Section(SectionType.INTRO, 8, 0.2, 0.4),
            Section(SectionType.PRECHORUS, 8, 0.5, 0.8),
            Section(SectionType.DROP, 16, 1.0, 1.0),
            Section(SectionType.BRIDGE, 8, 0.6, 0.85),
            Section(SectionType.DROP, 16, 1.0, 0.95),
            Section(SectionType.OUTRO, 8, 0.4, 0.1),
        ],
        "hip_hop": [
            Section(SectionType.INTRO, 4, 0.25, 0.3),
            Section(SectionType.VERSE, 16, 0.35, 0.5),
            Section(SectionType.HOOK, 8, 0.7, 0.75),
            Section(SectionType.VERSE, 16, 0.4, 0.55),
            Section(SectionType.HOOK, 8, 0.75, 0.8),
            Section(SectionType.OUTRO, 8, 0.3, 0.1),
        ],
    }
    
    def __init__(self, bpm: int = 120):
        self.bpm = bpm
        self.bars_per_minute = bpm / 4
    
    def generate(self, genre: str = "pop") -> List[dict]:
        """Generate arrangement"""
        template = self.TEMPLATES.get(genre, self.TEMPLATES["pop"])
        
        arrangement = []
        current_bar = 0
        
        for section in template:
            duration_sec = (section.bars / self.bars_per_minute) * 60
            
            arrangement.append({
                "section": section.type.value,
                "bars": section.bars,
                "start_bar": current_bar,
                "duration_sec": round(duration_sec, 1),
                "energy_start": section.energy_start,
                "energy_end": section.energy_end,
            })
            
            current_bar += section.bars
        
        return arrangement
    
    def to_json(self, arrangement: List[dict]) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(arrangement, indent=2)

if __name__ == "__main__":
    gen = ArrangementGenerator(bpm=128)
    
    for genre in ["pop", "edm", "hip_hop"]:
        print(f"\n=== {genre.upper()} ===")
        arr = gen.generate(genre)
        for section in arr:
            print(f"  {section['section']:12} | {section['bars']:3} bars | "
                  f"{section['duration_sec']:5}s | energy: {section['energy_start']:.1f}-{section['energy_end']:.1f}")
