#!/usr/bin/env python3
"""
VocalFusion - AI DJ Song Analysis Engine
Analyzes songs for key, tempo, energy, and other characteristics
"""

import sys
import json
import argparse
import librosa
import numpy as np
from pathlib import Path

# Camelot wheel mapping (key -> camelot)
KEY_TO_CAMELET = {
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

# Reverse mapping
CAMELOT_TO_KEY = {v: k for k, v in KEY_TO_CAMELET.items()}

def detect_key(y, sr):
    """Detect musical key using chroma features"""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Simple key detection based on chroma peaks
    chroma_mean = np.mean(chroma, axis=1)
    key_idx = np.argmax(chroma_mean)
    
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = keys[key_idx]
    
    # Detect major/minor using tonic/dominant relationship
    major_indicators = [chroma_mean[0], chroma_mean[4], chroma_mean[7]]  # I, IV, V
    minor_indicators = [chroma_mean[0], chroma_mean[3], chroma_mean[5]]  # i, iii, v
    
    mode = 'major' if sum(major_indicators) > sum(minor_indicators) else 'minor'
    
    camelot = KEY_TO_CAMELET.get((key, mode), '8B')
    
    return {
        'key': key,
        'mode': mode,
        'camelot': camelot
    }

def detect_bpm(y, sr):
    """Detect BPM using onset detection"""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)

def detect_energy(y):
    """Calculate energy (RMS)"""
    rms = librosa.feature.rms(y=y)
    return float(np.mean(rms))

def detect_valence(y, sr):
    """Estimate valence (mood) - simple implementation"""
    # Use spectral contrast as a proxy for valence
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    valence = float(np.mean(contrast))
    return valence

def detect_duration(y, sr):
    """Get duration in seconds"""
    return float(librosa.get_duration(y=y, sr=sr))

def detect_sections(y, sr):
    """Detect structural sections"""
    # Use beat-synchronous chroma for section detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.util.fix_frames(beats, x_max=chroma.shape[1])
    
    # Get beat-synchronous features
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.mean)
    
    # Find section boundaries (significant changes)
    delta = np.diff(chroma_sync, axis=1)
    boundary_scores = np.max(np.abs(delta), axis=0)
    
    # Top 3 boundaries
    if len(boundary_scores) > 3:
        boundary_idx = np.argsort(boundary_scores)[-3:]
    else:
        boundary_idx = range(len(boundary_scores))
    
    beat_times = librosa.frames_to_time(beats, sr=sr)
    sections = [{'time': float(beat_times[i]), 'strength': float(boundary_scores[i])} 
                for i in boundary_idx if i < len(beat_times)]
    
    return sections

def analyze_song(audio_path, song_id=None):
    """Main analysis function"""
    print(f"Loading: {audio_path}", file=sys.stderr)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Run all analyses
    result = {
        'song_id': song_id or Path(audio_path).stem,
        'file': str(audio_path),
        'duration': detect_duration(y, sr),
        'bpm': detect_bpm(y, sr),
        'key': detect_key(y, sr),
        'energy': detect_energy(y),
        'valence': detect_valence(y, sr),
        'sections': detect_sections(y, sr)
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='VocalFusion Song Analyzer')
    parser.add_argument('command', choices=['analyze'], help='Command to run')
    parser.add_argument('file', help='Audio file path')
    parser.add_argument('--name', help='Custom song ID')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        result = analyze_song(args.file, args.name)
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
