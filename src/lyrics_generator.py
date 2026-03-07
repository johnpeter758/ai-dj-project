"""
Lyrics Generator for AI DJ
Generates lyrics based on mood, genre, and theme
"""

import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LyricsConfig:
    """Configuration for lyrics generation"""
    mood: str = "neutral"  # happy, sad, angry, energetic, calm
    genre: str = "pop"  # pop, rock, hip-hop, edm, rnb
    theme: str = "love"  # love, party, dreams, nightlife, energy
    verses: int = 2
    chorus_repeats: int = 2
    has_bridge: bool = True


class LyricsGenerator:
    """Generate lyrics for AI-generated music"""
    
    # Word banks by mood and theme
    MOOD_WORDS = {
        "happy": ["bright", "sunshine", "smile", "joy", "light", "beautiful", "amazing", "wonderful"],
        "sad": ["rain", "tears", "gone", "lost", "alone", "cold", "empty", "heartbreak"],
        "angry": ["fire", "rage", "fight", "storm", "burn", "scream", "break", "revenge"],
        "energetic": ["power", "fire", "rush", "wild", "crazy", "electric", "blast", "maximum"],
        "calm": ["peace", "soft", "quiet", "dream", "gentle", "slow", "breeze", "moonlight"]
    }
    
    GENRE_PATTERNS = {
        "pop": {"line_length": 8, "rhyme_scheme": "ABAB", "style": "catchy"},
        "rock": {"line_length": 10, "rhyme_scheme": "AABB", "style": "powerful"},
        "hip-hop": {"line_length": 12, "rhyme_scheme": "AABB", "style": "flow"},
        "edm": {"line_length": 8, "rhyme_scheme": "AAAA", "style": "drops"},
        "rnb": {"line_length": 10, "rhyme_scheme": "ABAB", "style": "smooth"}
    }
    
    THEMES = {
        "love": ["baby", "heart", "forever", "together", "love", "kiss", "touch", "feel"],
        "party": ["dance", "night", "party", "celebrate", "hands up", "move", "beat", "club"],
        "dreams": ["dream", "sky", "fly", "star", "wish", "hope", "believe", "free"],
        "nightlife": ["city", "lights", "street", "midnight", "driving", "neon", "dark", "glow"],
        "energy": ["power", "fire", "energy", "boost", "rise", "shine", "electric", "wild"]
    }
    
    CHORUS_HOOKS = {
        "love": ["Love is all we need", "Feel the love tonight", "You and me forever"],
        "party": ["Party all night long", "Let the music play", "Dance until we drop"],
        "dreams": ["Reach for the stars", "Never give up", "Believe in yourself"],
        "nightlife": ["In the city lights", "Midnight city vibes", "Neon dreams"],
        "energy": ["Feel the power", "Energy rising", "Turn it up now"]
    }
    
    def __init__(self, config: Optional[LyricsConfig] = None):
        self.config = config or LyricsConfig()
        
    def _get_words(self, category: str) -> List[str]:
        """Get word bank for category"""
        if category == "mood":
            return self.MOOD_WORDS.get(self.config.mood, self.MOOD_WORDS["happy"])
        elif category == "theme":
            return self.THEMES.get(self.config.theme, self.THEMES["love"])
        return []
    
    def _generate_line(self, words: List[str], syllables: int = 8) -> str:
        """Generate a single line with given syllables"""
        selected = random.sample(words, min(4, len(words)))
        line = " ".join(selected[:3])
        return line.title()
    
    def _generate_verse(self) -> List[str]:
        """Generate a verse"""
        pattern = self.GENRE_PATTERNS.get(self.config.genre, self.GENRE_PATTERNS["pop"])
        words = self._get_words("theme") + self._get_words("mood")
        lines = []
        
        for _ in range(pattern["line_length"]):
            lines.append(self._generate_line(words))
            
        return lines
    
    def _generate_chorus(self) -> List[str]:
        """Generate a chorus with hook"""
        hooks = self.CHORUS_HOOKS.get(self.config.theme, self.CHORUS_HOOKS["love"])
        hook = random.choice(hooks)
        
        words = self._get_words("theme")
        lines = [
            hook,
            self._generate_line(words),
            hook,
            self._generate_line(words)
        ]
        return lines
    
    def _generate_bridge(self) -> List[str]:
        """Generate a bridge section"""
        words = self._get_words("mood")
        lines = []
        
        for _ in range(4):
            lines.append(self._generate_line(words))
            
        return lines
    
    def generate(self) -> dict:
        """Generate complete lyrics"""
        lyrics = {
            "intro": [],
            "verses": [],
            "chorus": [],
            "bridge": [],
            "outro": []
        }
        
        # Generate verses
        for _ in range(self.config.verses):
            lyrics["verses"].append(self._generate_verse())
        
        # Generate chorus
        for _ in range(self.config.chorus_repeats):
            lyrics["chorus"].append(self._generate_chorus())
        
        # Generate bridge
        if self.config.has_bridge:
            lyrics["bridge"].append(self._generate_bridge())
        
        return lyrics
    
    def to_text(self, lyrics: Optional[dict] = None) -> str:
        """Convert lyrics to plain text"""
        if lyrics is None:
            lyrics = self.generate()
            
        output = []
        
        # Intro
        if lyrics.get("intro"):
            output.append("=== INTRO ===")
            for line in lyrics["intro"]:
                output.append(line)
            output.append("")
        
        # Verses
        for i, verse in enumerate(lyrics.get("verses", [])):
            output.append(f"=== VERSE {i+1} ===")
            for line in verse:
                output.append(line)
            output.append("")
        
        # Chorus
        for i, chorus in enumerate(lyrics.get("chorus", [])):
            output.append(f"=== CHORUS {i+1} ===")
            for line in chorus:
                output.append(line)
            output.append("")
        
        # Bridge
        if lyrics.get("bridge"):
            output.append("=== BRIDGE ===")
            for line in lyrics["bridge"][0]:
                output.append(line)
            output.append("")
        
        # Outro
        if lyrics.get("outro"):
            output.append("=== OUTRO ===")
            for line in lyrics["outro"]:
                output.append(line)
                
        return "\n".join(output)


if __name__ == "__main__":
    # Demo
    config = LyricsConfig(mood="energetic", genre="edm", theme="party", verses=2)
    generator = LyricsGenerator(config)
    lyrics = generator.generate()
    
    print(generator.to_text(lyrics))
    print("\n--- JSON ---")
    import json
    print(json.dumps(lyrics, indent=2))
