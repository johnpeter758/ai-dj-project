#!/usr/bin/env python3
"""
Stem Processor - Separate, process, and remix stems
"""

import subprocess
import os
from pathlib import Path

class StemProcessor:
    """Process audio stems"""
    
    def __init__(self, output_dir: str = "separated"):
        self.output_dir = output_dir
    
    def separate(self, input_file: str, stems: int = 4) -> dict:
        """Separate audio into stems"""
        if stems == 2:
            cmd = f"demucs --two-stems=vocals -o {self.output_dir} {input_file}"
        else:
            cmd = f"demucs -o {self.output_dir} {input_file}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Find output files
        basename = Path(input_file).stem
        stem_dir = Path(self.output_dir) / "htdemucs" / basename
        
        output = {}
        if stem_dir.exists():
            for stem_file in stem_dir.glob("*.wav"):
                output[stem_file.stem] = str(stem_file)
        
        return output
    
    def separate_vocals(self, input_file: str) -> dict:
        """Extract vocals (acappella)"""
        return self.separate(input_file, stems=2)
    
    def remix(self, stem_files: dict, pattern: str = "AABB") -> list:
        """Remix stems based on pattern"""
        # Pattern: A=stem1, B=stem2, etc.
        stems = list(stem_files.keys())
        remix = []
        
        for char in pattern:
            if char.isalpha() and len(stems) > 0:
                idx = (ord(char.upper()) - ord('A')) % len(stems)
                remix.append(stems[idx])
        
        return remix
    
    def mix_stems(self, stem_files: dict, volumes: dict = None) -> str:
        """Mix stems together"""
        # This would use librosa to mix
        pass

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: stem_processor.py <input.wav> [2|4]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    stems = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    processor = StemProcessor()
    result = processor.separate(input_file, stems)
    
    print(f"\nSeparated into {len(result)} stems:")
    for name, path in result.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    main()
