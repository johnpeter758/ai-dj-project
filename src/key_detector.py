#!/usr/bin/env python3
"""
Key Detector - Musical Key Detection System
Detects musical key from audio files using chroma features and key profiles
"""

import numpy as np
import librosa
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import json


# Musical notes
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Camelot wheel mapping (key -> Camelot)
KEY_TO_CAMELOT = {
    ('C', 'major'): '8B', ('C', 'minor'): '5A',
    ('C#', 'major'): '3B', ('C#', 'minor'): '12A',
    ('D', 'major'): '10B', ('D', 'minor'): '7A',
    ('D#', 'major'): '5B', ('D#', 'minor'): '2A',
    ('E', 'major'): '12B', ('E', 'minor'): '9A',
    ('F', 'major'): '7B', ('F', 'minor'): '4A',
    ('F#', 'major'): '2B', ('F#', 'minor'): '11A',
    ('G', 'major'): '9B', ('G', 'minor'): '6A',
    ('G#', 'major'): '4B', ('G#', 'minor'): '1A',
    ('A', 'major'): '11B', ('A', 'minor'): '8A',
    ('A#', 'major'): '6B', ('A#', 'minor'): '3A',
    ('B', 'major'): '1B', ('B', 'minor'): '10A',
}

# Reverse Camelot mapping
CAMELOT_TO_KEY = {v: k for k, v in KEY_TO_CAMELOT.items()}

# Krumhansl-Schmuckler key profiles (tonal hierarchy)
# These profiles represent the relative prominence of each pitch class in major/minor keys
KROUHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Simple key profiles (simplified version)
SIMPLE_MAJOR = np.array([1.0, 0.5, 0.8, 0.4, 1.0, 0.6, 0.3, 1.0, 0.5, 0.8, 0.4, 0.6])
SIMPLE_MINOR = np.array([1.0, 0.5, 0.6, 0.8, 0.4, 1.0, 0.3, 0.8, 0.6, 0.5, 0.4, 0.6])


@dataclass
class KeyResult:
    """Result of key detection"""
    key: str           # Note name (C, C#, D, etc.)
    mode: str          # 'major' or 'minor'
    camelot: str       # Camelot notation (e.g., '8B', '5A')
    confidence: float  # Detection confidence (0-1)
    algorithm: str     # Algorithm used
    chroma_vector: Optional[np.ndarray] = None  # Raw chroma features


class KeyDetector:
    """
    Musical key detector using multiple algorithms.
    
    Supports:
    - Chroma-based key detection
    - Krumhansl-Schmuckler key profile correlation
    - Tonic/dominant analysis for mode detection
    - Camelot wheel notation for DJ compatibility
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize key detector.
        
        Args:
            sample_rate: Audio sample rate for analysis
            hop_length: Hop length for STFT/chroma computation
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return y, sr
    
    def compute_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute chroma features from audio"""
        # Use CQT-based chroma for better pitch resolution
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        return chroma
    
    def compute_chroma_stft(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute chroma features using STFT-based method"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        return chroma
    
    def key_profile_correlation(self, chroma: np.ndarray) -> Tuple[str, str, float]:
        """
        Detect key using Krumhansl-Schmuckler key profile correlation.
        
        Args:
            chroma: Chroma feature matrix (12 x frames)
            
        Returns:
            Tuple of (key, mode, confidence)
        """
        # Average chroma over time
        chroma_mean = np.mean(chroma, axis=1)
        
        # Normalize chroma
        chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-8)
        
        best_score = -np.inf
        best_key = 'C'
        best_mode = 'major'
        
        # Try all 24 keys (12 major + 12 minor)
        for root_idx in range(12):
            # Rotate chroma for this root note
            rotated = np.roll(chroma_mean, -root_idx)
            
            # Compare with major and minor profiles
            major_corr = np.corrcoef(rotated, KROUHANSL_MAJOR)[0, 1]
            minor_corr = np.corrcoef(rotated, KRUHANSL_MINOR)[0, 1]
            
            if major_corr > best_score:
                best_score = major_corr
                best_key = NOTES[root_idx]
                best_mode = 'major'
            
            if minor_corr > best_score:
                best_score = minor_corr
                best_key = NOTES[root_idx]
                best_mode = 'minor'
        
        # Convert correlation to confidence (0-1)
        confidence = max(0, min(1, (best_score + 1) / 2))
        
        return best_key, best_mode, confidence
    
    def chroma_peak_detection(self, chroma: np.ndarray) -> Tuple[str, str, float]:
        """
        Detect key based on chroma peak detection.
        
        Args:
            chroma: Chroma feature matrix
            
        Returns:
            Tuple of (key, mode, confidence)
        """
        chroma_mean = np.mean(chroma, axis=1)
        
        # Find dominant pitch class
        key_idx = np.argmax(chroma_mean)
        key = NOTES[key_idx]
        
        # Detect major/minor using harmonic relationships
        # Major: I, IV, V strong; Minor: i, iii, v strong
        major_indicators = [
            chroma_mean[0],   # I
            chroma_mean[4],   # IV (E)
            chroma_mean[7]   # V (G)
        ]
        minor_indicators = [
            chroma_mean[0],   # i
            chroma_mean[3],   # iii (D#)
            chroma_mean[7]   # v (G)
        ]
        
        major_sum = sum(major_indicators)
        minor_sum = sum(minor_indicators)
        
        mode = 'major' if major_sum > minor_sum else 'minor'
        
        # Confidence based on how dominant the tonic is
        confidence = float(chroma_mean[key_idx] / (np.sum(chroma_mean) + 1e-8))
        
        return key, mode, min(confidence, 1.0)
    
    def harmonic_change_detection(self, chroma: np.ndarray) -> Tuple[str, str, float]:
        """
        Detect key using harmonic change detection (key changes throughout track).
        
        Args:
            chroma: Chroma feature matrix
            
        Returns:
            Tuple of (key, mode, confidence)
        """
        # Compute key profile correlation at multiple points
        n_segments = min(10, chroma.shape[1])
        segment_size = chroma.shape[1] // n_segments
        
        all_keys = []
        all_modes = []
        
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size
            segment = chroma[:, start:end]
            
            key, mode, _ = self.key_profile_correlation(segment)
            all_keys.append(key)
            all_modes.append(mode)
        
        # Use most common key (mode)
        key_counts = {}
        mode_counts = {}
        
        for k in all_keys:
            key_counts[k] = key_counts.get(k, 0) + 1
        for m in all_modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        
        best_key = max(key_counts, key=key_counts.get)
        best_mode = max(mode_counts, key=mode_counts.get)
        
        # Confidence based on agreement
        confidence = key_counts[best_key] / len(all_keys)
        
        return best_key, best_mode, confidence
    
    def detect_key(self, y: Optional[np.ndarray] = None, sr: Optional[int] = None,
                   audio_path: Optional[str] = None,
                   algorithm: str = 'ensemble') -> KeyResult:
        """
        Detect musical key from audio.
        
        Args:
            y: Audio time series (optional if audio_path provided)
            sr: Sample rate (optional if audio_path provided)
            audio_path: Path to audio file
            algorithm: Detection method ('profile', 'peak', 'harmonic', 'ensemble')
            
        Returns:
            KeyResult object with key, mode, camelot, and confidence
        """
        # Load audio if path provided
        if audio_path:
            y, sr = self.load_audio(audio_path)
        
        # Compute chroma features
        chroma = self.compute_chroma(y, sr)
        
        if algorithm == 'profile':
            key, mode, confidence = self.key_profile_correlation(chroma)
        elif algorithm == 'peak':
            key, mode, confidence = self.chroma_peak_detection(chroma)
        elif algorithm == 'harmonic':
            key, mode, confidence = self.harmonic_change_detection(chroma)
        elif algorithm == 'ensemble':
            # Use multiple algorithms and combine results
            results = []
            
            # Profile-based detection
            k1, m1, c1 = self.key_profile_correlation(chroma)
            results.append((k1, m1, c1, 'profile'))
            
            # Peak detection
            k2, m2, c2 = self.chroma_peak_detection(chroma)
            results.append((k2, m2, c2, 'peak'))
            
            # Harmonic detection
            k3, m3, c3 = self.harmonic_change_detection(chroma)
            results.append((k3, m3, c3, 'harmonic'))
            
            # Weight by confidence and pick winner
            weighted_votes = {}
            for k, m, c, alg in results:
                vote_key = (k, m)
                weighted_votes[vote_key] = weighted_votes.get(vote_key, 0) + c
            
            best_vote = max(weighted_votes, key=weighted_votes.get)
            key, mode = best_vote
            confidence = weighted_votes[best_vote] / sum(weighted_votes.values())
            algorithm = 'ensemble'
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Get Camelot notation
        camelot = KEY_TO_CAMELOT.get((key, mode), '8B')
        
        return KeyResult(
            key=key,
            mode=mode,
            camelot=camelot,
            confidence=confidence,
            algorithm=algorithm,
            chroma_vector=np.mean(chroma, axis=1)
        )
    
    def get_compatible_keys(self, key: str, mode: str, 
                           include_relative: bool = True,
                           include_analogous: bool = True) -> List[str]:
        """
        Get harmonically compatible keys for mixing.
        
        Args:
            key: Root note
            mode: 'major' or 'minor'
            include_relative: Include relative major/minor
            include_analogous: Include analogous keys (±1 semitone)
            
        Returns:
            List of compatible Camelot codes
        """
        camelot = KEY_TO_CAMELOT.get((key, mode), '8B')
        root_num = int(camelot[0])
        letter = camelot[1]
        
        compatible = [camelot]  # Always compatible with self
        
        if include_relative:
            # Relative major/minor (same number, different letter)
            relative = f"{root_num}{'A' if letter == 'B' else 'B'}"
            compatible.append(relative)
        
        if include_analogous:
            # ±1 on the wheel
            plus_1 = f"{(root_num % 12) + 1}{letter}"
            minus_1 = f"{(root_num - 2) % 12 + 1}{letter}"
            compatible.extend([plus_1, minus_1])
        
        return compatible
    
    def analyze_file(self, audio_path: str, algorithm: str = 'ensemble') -> Dict:
        """
        Analyze audio file and return detailed key information.
        
        Args:
            audio_path: Path to audio file
            algorithm: Detection algorithm to use
            
        Returns:
            Dictionary with key detection results
        """
        result = self.detect_key(audio_path=audio_path, algorithm=algorithm)
        
        # Get compatible keys
        compatible = self.get_compatible_keys(result.key, result.mode)
        
        return {
            'file': str(audio_path),
            'key': result.key,
            'mode': result.mode,
            'camelot': result.camelot,
            'confidence': round(result.confidence, 3),
            'algorithm': result.algorithm,
            'compatible_keys': compatible,
            'notation': f"{result.key} {result.mode} ({result.camelot})"
        }


def detect_key_from_file(audio_path: str, algorithm: str = 'ensemble') -> Dict:
    """
    Convenience function to detect key from audio file.
    
    Args:
        audio_path: Path to audio file
        algorithm: Detection algorithm ('profile', 'peak', 'harmonic', 'ensemble')
        
    Returns:
        Dictionary with key detection results
    """
    detector = KeyDetector()
    return detector.analyze_file(audio_path, algorithm)


def get_camelot_circle() -> Dict[str, Dict]:
    """
    Get the full Camelot circle for reference.
    
    Returns:
        Dictionary mapping all Camelot codes to musical keys
    """
    circle = {}
    for (key, mode), camelot in KEY_TO_CAMELOT.items():
        circle[camelot] = {'key': key, 'mode': mode}
    return circle


# CLI interface
def main():
    """Command-line interface for key detection"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect musical key from audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  key_detector.py track.wav
  key_detector.py track.wav --algorithm profile
  key_detector.py track.wav --output json

Camelot System:
  Numbers 1-12 represent the key on the circle of fifths
  A = Minor, B = Major
        """
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--algorithm', '-a', 
                       choices=['profile', 'peak', 'harmonic', 'ensemble'],
                       default='ensemble',
                       help='Key detection algorithm (default: ensemble)')
    parser.add_argument('--output', '-o',
                       choices=['text', 'json', 'camelot'],
                       default='text',
                       help='Output format')
    parser.add_argument('--compatible', '-c',
                       action='store_true',
                       help='Show compatible keys for mixing')
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.audio_file).exists():
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        return 1
    
    # Detect key
    detector = KeyDetector()
    result = detector.analyze_file(args.audio_file, args.algorithm)
    
    # Output
    if args.output == 'json':
        print(json.dumps(result, indent=2))
    elif args.output == 'camelot':
        print(result['camelot'])
    else:
        print(f"File: {result['file']}")
        print(f"Key: {result['key']} {result['mode']}")
        print(f"Camelot: {result['camelot']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Algorithm: {result['algorithm']}")
        
        if args.compatible:
            print(f"Compatible Keys: {', '.join(result['compatible_keys'])}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
