"""
Sample Manager for AI DJ Project
Manages audio samples with metadata, search, and auto-categorization.
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class Category(Enum):
    """Sample categories for auto-categorization."""
    KICK = "kick"
    SNARE = "snare"
    HIHAT = "hihat"
    PERCUSSION = "percussion"
    BASS = "bass"
    SYNTH = "synth"
    VOCAL = "vocal"
    FX = "fx"
    LOOP = "loop"
    ONE_SHOT = "one_shot"
    UNKNOWN = "unknown"


@dataclass
class Sample:
    """Represents an audio sample with metadata."""
    filename: str
    filepath: str
    category: str = Category.UNKNOWN.value
    bpm: Optional[float] = None
    key: Optional[str] = None
    duration_ms: Optional[int] = None
    tags: list = field(default_factory=list)
    hash: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class SampleManager:
    """Manages loading, searching, and categorizing audio samples."""
    
    SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.aiff', '.aif'}
    
    def __init__(self, samples_dir: str):
        self.samples_dir = Path(samples_dir)
        self.samples: dict[str, Sample] = {}
        self._metadata_file = self.samples_dir / ".sample_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load saved metadata from file."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                data = json.load(f)
                for filename, sample_data in data.items():
                    self.samples[filename] = Sample(**sample_data)
    
    def _save_metadata(self):
        """Persist metadata to file."""
        data = {name: sample.to_dict() for name, sample in self.samples.items()}
        with open(self._metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_hash(self, filepath: Path) -> str:
        """Compute file hash for deduplication."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    
    def _categorize_by_filename(self, filename: str) -> Category:
        """Auto-categorize based on filename keywords."""
        name_lower = filename.lower()
        
        keywords = {
            Category.KICK: ['kick', 'kd', 'bass drum', 'bd'],
            Category.SNARE: ['snare', 'sd', 'snap'],
            Category.HIHAT: ['hihat', 'hh', 'hat', 'hi-hat', 'high hat'],
            Category.PERCUSSION: ['perc', 'tom', 'conga', 'bongo', 'timbale'],
            Category.BASS: ['bass', 'sub', '808', 'loop'],
            Category.SYNTH: ['synth', 'lead', 'pad', 'arp', 'pluck'],
            Category.VOCAL: ['vocal', 'vox', 'voice', 'hook', 'adlib'],
            Category.FX: ['fx', 'effect', 'riser', 'impact', 'sweep', 'noise'],
            Category.LOOP: ['loop', 'break', 'breakbeat'],
            Category.ONE_SHOT: ['one-shot', 'oneshot', 'stab'],
        }
        
        for category, words in keywords.items():
            if any(word in name_lower for word in words):
                return category
        
        return Category.UNKNOWN
    
    def load_samples(self, recursive: bool = True) -> list[Sample]:
        """Load all samples from the samples directory."""
        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")
        
        loaded = []
        pattern = "**/*" if recursive else "*"
        
        for filepath in self.samples_dir.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                filename = filepath.name
                
                # Skip if already loaded and unchanged
                if filename in self.samples:
                    existing = self.samples[filename]
                    if existing.hash == self._compute_hash(filepath):
                        loaded.append(existing)
                        continue
                
                # Create new sample
                file_hash = self._compute_hash(filepath)
                category = self._categorize_by_filename(filename)
                
                sample = Sample(
                    filename=filename,
                    filepath=str(filepath),
                    category=category.value,
                    hash=file_hash,
                    tags=self._extract_tags(filename)
                )
                
                self.samples[filename] = sample
                loaded.append(sample)
        
        self._save_metadata()
        return loaded
    
    def _extract_tags(self, filename: str) -> list[str]:
        """Extract tags from filename."""
        name = Path(filename).stem
        # Split on common separators
        tags = set()
        for sep in ['_', '-', ' ', '.']:
            parts = name.split(sep)
            for part in parts:
                if len(part) > 2:
                    tags.add(part.lower())
        return list(tags)
    
    # --- Search Functions ---
    
    def search_by_category(self, category: str) -> list[Sample]:
        """Find samples by category."""
        cat = category.lower()
        return [s for s in self.samples.values() if s.category == cat]
    
    def search_by_bpm(self, bpm: float, tolerance: float = 5.0) -> list[Sample]:
        """Find samples by BPM (with tolerance)."""
        return [s for s in self.samples.values() 
                if s.bpm and abs(s.bpm - bpm) <= tolerance]
    
    def search_by_key(self, key: str) -> list[Sample]:
        """Find samples by musical key."""
        key_clean = key.upper().replace(' ', '').replace('#', '')
        return [s for s in self.samples.values() 
                if s.key and s.key.upper().replace('#', '') == key_clean]
    
    def search_by_tag(self, tag: str) -> list[Sample]:
        """Find samples by tag."""
        tag_lower = tag.lower()
        return [s for s in self.samples.values() 
                if tag_lower in s.tags]
    
    def search(self, query: str) -> list[Sample]:
        """Full-text search across filename, category, and tags."""
        query_lower = query.lower()
        results = []
        for sample in self.samples.values():
            if query_lower in sample.filename.lower():
                results.append(sample)
            elif query_lower in sample.category:
                results.append(sample)
            elif query_lower in sample.tags:
                results.append(sample)
        return results
    
    def get_category_counts(self) -> dict[str, int]:
        """Get count of samples per category."""
        counts = {}
        for sample in self.samples.values():
            counts[sample.category] = counts.get(sample.category, 0) + 1
        return counts
    
    def set_bpm(self, filename: str, bpm: float) -> bool:
        """Set BPM for a sample."""
        if filename in self.samples:
            self.samples[filename].bpm = bpm
            self._save_metadata()
            return True
        return False
    
    def set_key(self, filename: str, key: str) -> bool:
        """Set musical key for a sample."""
        if filename in self.samples:
            self.samples[filename].key = key.upper()
            self._save_metadata()
            return True
        return False
    
    def __len__(self):
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples.values())
    
    def list_all(self) -> list[Sample]:
        """Return all samples as a list."""
        return list(self.samples.values())
