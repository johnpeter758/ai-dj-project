#!/usr/bin/env python3
"""
Stem Splitter - Audio source separation using Demucs
Separates audio into stems: vocals, drums, bass, and other
"""

import os
import subprocess
import shutil
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StemResult:
    """Result of stem separation"""
    vocals: Optional[str] = None
    drums: Optional[str] = None
    bass: Optional[str] = None
    other: Optional[str] = None
    instrumentals: Optional[str] = None  # Combined non-vocal stems
    
    def get_stems(self) -> Dict[str, str]:
        """Get all available stems as dict"""
        return {k: v for k, v in {
            'vocals': self.vocals,
            'drums': self.drums,
            'bass': self.bass,
            'other': self.other,
        }.items() if v is not None}
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'vocals': self.vocals,
            'drums': self.drums,
            'bass': self.bass,
            'other': self.other,
            'instrumentals': self.instrumentals,
        }


class StemSplitter:
    """
    Audio stem separator using Demucs.
    
    Supports:
    - 4-stem separation: vocals, drums, bass, other
    - 2-stem separation: vocals (acappella) or instrumentals
    - Caching of separated stems
    - Multiple model options
    """
    
    # Default stems produced by Demucs
    DEFAULT_STEMS = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(
        self,
        output_dir: str = "separated",
        cache_dir: str = ".cache/stems",
        model: str = "htdemucs_ft",
        device: str = "cuda"
    ):
        """
        Initialize stem splitter.
        
        Args:
            output_dir: Base directory for separated stems
            cache_dir: Cache directory for separated stems
            model: Demucs model to use (htdemucs_ft, htdemucs, mdx)
            device: Device to use (cuda, cpu)
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.device = device
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check Demucs is available
        self._check_demucs()
    
    def _check_demucs(self) -> bool:
        """Check if Demucs is installed"""
        try:
            result = subprocess.run(
                ['demucs', '--version'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            raise RuntimeError(
                "Demucs not found. Install with: pip install demucs"
            )
    
    def _get_cache_key(self, audio_path: str, stems: int = 4) -> str:
        """Generate cache key for audio file"""
        file_stat = os.stat(audio_path)
        key_data = f"{audio_path}_{file_stat.st_size}_{file_stat.st_mtime}_{stems}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_stems(self, cache_key: str) -> Optional[StemResult]:
        """Get cached stems if available"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                return StemResult(**data)
        return None
    
    def _save_cache(self, cache_key: str, result: StemResult) -> None:
        """Save stems to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(result.to_dict(), f)
    
    def split(
        self,
        audio_path: str,
        stems: int = 4,
        use_cache: bool = True,
        shifts: int = 5,
        segment: Optional[int] = None
    ) -> StemResult:
        """
        Separate audio into stems.
        
        Args:
            audio_path: Path to audio file
            stems: Number of stems (2 or 4)
            use_cache: Whether to use cached results
            shifts: Number of inference shifts for quality (0-10)
            segment: Segment length in seconds (for memory efficiency)
            
        Returns:
            StemResult with paths to separated stems
        """
        audio_path = Path(audio_path).resolve()
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check cache
        cache_key = self._get_cache_key(str(audio_path), stems)
        if use_cache:
            cached = self._get_cached_stems(cache_key)
            if cached:
                print(f"Using cached stems for {audio_path.name}")
                return cached
        
        # Build Demucs command
        cmd = self._build_command(audio_path, stems, shifts, segment)
        
        print(f"Separating {audio_path.name} into {stems} stems...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run Demucs
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.output_dir
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed: {result.stderr}")
        
        # Find output files
        stem_paths = self._find_stem_outputs(audio_path.stem)
        
        # Create StemResult
        stem_result = StemResult(
            vocals=stem_paths.get('vocals'),
            drums=stem_paths.get('drums'),
            bass=stem_paths.get('bass'),
            other=stem_paths.get('other'),
        )
        
        # Create instrumentals (all stems except vocals)
        if stems == 2 and 'vocals' in stem_paths:
            stem_result.instrumentals = self._create_instrumental(stem_paths)
        
        # Cache results
        if use_cache:
            self._save_cache(cache_key, stem_result)
        
        print(f"Successfully separated into {len(stem_result.get_stems())} stems")
        
        return stem_result
    
    def _build_command(
        self,
        audio_path: Path,
        stems: int,
        shifts: int,
        segment: Optional[int]
    ) -> List[str]:
        """Build Demucs command"""
        cmd = [
            'demucs',
            '-n', self.model,
            '-d', self.device,
            '--shifts', str(shifts),
            '-o', str(self.output_dir),
        ]
        
        if stems == 2:
            cmd.extend(['--two-stems', 'vocals'])
        
        if segment:
            cmd.extend(['--segment', str(segment)])
        
        cmd.append(str(audio_path))
        
        return cmd
    
    def _find_stem_outputs(self, basename: str) -> Dict[str, str]:
        """Find separated stem files"""
        stem_dir = self.output_dir / self.model / basename
        stems = {}
        
        if stem_dir.exists():
            for stem_file in stem_dir.glob("*.wav"):
                stems[stem_file.stem] = str(stem_file)
        
        return stems
    
    def _create_instrumental(self, stem_paths: Dict[str, str]) -> str:
        """Create instrumental by combining non-vocal stems"""
        # For 2-stem mode, Demucs creates vocals and no_vocals (instrumental)
        # The instrumental is saved as the original filename without vocals
        # We'll return the path to the 'no_vocals' or 'instrumental' file
        instrumental = stem_paths.get('no_vocals') or stem_paths.get('instrumental')
        return instrumental
    
    def get_vocals(self, audio_path: str) -> Optional[str]:
        """Extract just vocals (acappella)"""
        result = self.split(audio_path, stems=2)
        return result.vocals
    
    def get_instrumental(self, audio_path: str) -> Optional[str]:
        """Get instrumental version (no vocals)"""
        result = self.split(audio_path, stems=2)
        return result.instrumentals
    
    def clear_cache(self) -> int:
        """Clear stem cache"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
    
    def get_cache_size(self) -> int:
        """Get number of cached stems"""
        return len(list(self.cache_dir.glob("*.json")))


class BatchStemSplitter:
    """Batch process multiple audio files"""
    
    def __init__(self, splitter: Optional[StemSplitter] = None, **kwargs):
        self.splitter = splitter or StemSplitter(**kwargs)
        self.results: Dict[str, StemResult] = {}
    
    def process_directory(
        self,
        directory: str,
        pattern: str = "*.mp3",
        stems: int = 4
    ) -> Dict[str, StemResult]:
        """Process all audio files in a directory"""
        dir_path = Path(directory)
        results = {}
        
        for audio_file in dir_path.glob(pattern):
            print(f"\nProcessing: {audio_file.name}")
            try:
                result = self.splitter.split(str(audio_file), stems=stems)
                results[str(audio_file)] = result
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
        
        self.results = results
        return results
    
    def get_summary(self) -> dict:
        """Get summary of batch processing"""
        total = len(self.results)
        successful = sum(1 for r in self.results.values() if r.get_stems())
        
        return {
            "total_files": total,
            "successful": successful,
            "failed": total - successful,
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """CLI interface for stem splitting"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split audio into stems using Demucs"
    )
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument(
        "-s", "--stems",
        type=int,
        default=4,
        choices=[2, 4],
        help="Number of stems (2 or 4)"
    )
    parser.add_argument(
        "-o", "--output",
        default="separated",
        help="Output directory"
    )
    parser.add_argument(
        "-m", "--model",
        default="htdemucs_ft",
        choices=["htdemucs_ft", "htdemucs", "mdx"],
        help="Demucs model"
    )
    parser.add_argument(
        "-d", "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache and exit"
    )
    
    args = parser.parse_args()
    
    splitter = StemSplitter(
        output_dir=args.output,
        model=args.model,
        device=args.device
    )
    
    if args.clear_cache:
        count = splitter.clear_cache()
        print(f"Cleared {count} cached items")
        return
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Batch process directory
        batch = BatchStemSplitter(splitter)
        results = batch.process_directory(
            str(input_path),
            stems=args.stems
        )
        
        summary = batch.get_summary()
        print(f"\n{'='*40}")
        print(f"Batch Processing Complete")
        print(f"Total: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
    
    else:
        # Process single file
        result = splitter.split(
            str(input_path),
            stems=args.stems,
            use_cache=not args.no_cache
        )
        
        print(f"\nSeparated stems:")
        for name, path in result.get_stems().items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
