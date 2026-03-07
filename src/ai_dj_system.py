#!/usr/bin/env python3
"""
AI DJ System - Main Controller
"""

import os
import json
from datetime import datetime

class AIDJSystem:
    """Main AI DJ system"""
    
    def __init__(self):
        self.name = "Peter"
        self.version = "1.0"
        self.components = {}
        self.stats = {
            "songs_generated": 0,
            "fusions_created": 0,
            "research_docs": 17,
        }
    
    def status(self) -> dict:
        """Get system status"""
        return {
            "name": self.name,
            "version": self.version,
            "stats": self.stats,
            "components": list(self.components.keys()),
            "ready": True,
        }
    
    def generate(self, prompt: str, genre: str = "pop") -> dict:
        """Generate a song"""
        # This would integrate with ACE-Step
        result = {
            "prompt": prompt,
            "genre": genre,
            "status": "generated",
            "timestamp": datetime.now().isoformat(),
        }
        self.stats["songs_generated"] += 1
        return result
    
    def create_fusion(self, song1: str, song2: str) -> dict:
        """Create a fusion"""
        result = {
            "song1": song1,
            "song2": song2,
            "status": "created",
            "timestamp": datetime.now().isoformat(),
        }
        self.stats["fusions_created"] += 1
        return result
    
    def analyze(self, audio_file: str) -> dict:
        """Analyze audio"""
        # This would use librosa
        return {
            "file": audio_file,
            "bpm": 128,
            "key": "8A",
            "energy": 0.8,
            "duration": 180,
        }
    
    def get_research(self) -> list:
        """List available research"""
        research_dir = "/Users/johnpeter/ai-dj-project/src"
        docs = []
        for f in os.listdir(research_dir):
            if f.endswith(".md"):
                docs.append(f.replace(".md", ""))
        return docs
    
    def save_state(self):
        """Save system state"""
        state = {
            "name": self.name,
            "version": self.version,
            "stats": self.stats,
            "timestamp": datetime.now().isoformat(),
        }
        with open("system_state.json", "w") as f:
            json.dump(state, f, indent=2)
        return state

def main():
    dj = AIDJSystem()
    
    print("=" * 50)
    print(f"🎛️ {dj.name} - AI DJ System v{dj.version}")
    print("=" * 50)
    
    status = dj.status()
    print(f"\n📊 Status: {status}")
    
    print(f"\n📚 Research Available:")
    for doc in dj.get_research():
        print(f"  - {doc}")
    
    print("\n✅ System ready!")

if __name__ == "__main__":
    main()
