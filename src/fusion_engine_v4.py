#!/usr/bin/env python3
"""
AI DJ Fusion Engine v4.0 - IMPROVED
Based on analysis of what works best
"""

import numpy as np
import librosa
import soundfile as sf
import random
from scipy import signal

# Camelot wheel
CAMELOT = {
    '1A': ['1A', '2A', '12A', '1B', '6B'], '2A': ['2A', '1A', '3A', '2B', '7B'],
    '3A': ['3A', '2A', '4A', '3B', '8B'], '4A': ['4A', '3A', '5A', '4B', '9B'],
    '5A': ['5A', '4A', '6A', '5B', '10B'], '6A': ['6A', '5A', '7A', '6B', '11B'],
    '7A': ['7A', '6A', '8A', '7B', '12B'], '8A': ['8A', '7A', '9A', '8B', '1B'],
    '9A': ['9A', '8A', '10A', '9B', '2B'], '10A': ['10A', '9A', '11A', '10B', '3B'],
    '11A': ['11A', '10A', '12A', '11B', '4B'], '12A': ['12A', '11A', '1A', '12B', '5B'],
    '1B': ['1B', '2B', '12B', '1A', '8A'], '2B': ['2B', '3B', '1B', '2A', '9A'],
    '3B': ['3B', '4B', '2B', '3A', '10A'], '4B': ['4B', '5B', '3B', '4A', '11A'],
    '5B': ['5B', '6B', '4B', '5A', '12A'], '6B': ['6B', '7B', '5B', '6A', '1A'],
    '7B': ['7B', '8B', '6B', '7A', '2A'], '8B': ['8B', '9B', '7B', '8A', '3A'],
    '9B': ['9B', '10B', '8B', '9A', '4A'], '10B': ['10B', '11B', '9B', '10A', '5A'],
    '11B': ['11B', '12B', '10B', '11A', '6A'], '12B': ['12B', '1B', '11B', '12A', '7A'],
}

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    c = np.mean(chroma, axis=1)
    keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    k = keys[np.argmax(c)]
    major = c[0]+c[4]+c[7]
    minor = c[0]+c[2]+c[5]
    m = 'major' if major > minor else 'minor'
    return f"{np.argmax(c)+1}{'B' if m=='major' else 'A'}"

def detect_bpm(y, sr):
    t, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(t)

def get_energy(y):
    return float(np.mean(np.abs(y)))

def improved_fusion(s1_path, s2_path, name, style='optimal'):
    """Create improved fusion with analysis"""
    print(f"🎛️ {name}")
    
    # Load
    y1, sr = librosa.load(s1_path, sr=44100)
    y2, sr2 = librosa.load(s2_path, sr=44100)
    
    # Analyze
    key1, key2 = detect_key(y1, sr), detect_key(y2, sr)
    bpm1, bpm2 = detect_bpm(y1, sr), detect_bpm(y2, sr)
    e1, e2 = get_energy(y1), get_energy(y2)
    
    print(f"   {key1} @ {bpm1:.0f}bpm | {key2} @ {bpm2:.0f}bpm")
    
    # Compatibility
    compat = 1.0 if key1==key2 else (0.8 if key2 in CAMELOT.get(key1,[]) else 0.4)
    print(f"   Compatibility: {compat:.0%}")
    
    # Duration
    dur = min(len(y1), len(y2), 90*sr)
    y1, y2 = y1[:dur]*0.88, y2[:dur]*0.88
    
    # Style-based split
    if style == 'drop':
        split = int(dur * 0.65)
        cf = int(1.5 * sr)
    elif style == 'chill':
        split = int(dur * 0.35)
        cf = int(6 * sr)
    elif style == 'build':
        split = int(dur * 0.45)
        cf = int(5 * sr)
    else:  # optimal
        split = int(dur * 0.5)
        cf = int(3 * sr)
    
    # Create
    result = np.zeros(dur)
    result[:split] = y1[:split]
    
    fade = np.linspace(0, 1, min(cf, dur-split))
    result[split:split+len(fade)] = y1[split:split+len(fade)]*(1-fade) + y2[split:split+len(fade)]*fade
    result[split+len(fade):] = y2[split+len(fade):]
    
    # EQ: gentle high shelf on transition
    # Professional processing
    result = np.tanh(result * 1.3) / 1.3
    result = result / np.max(np.abs(result)) * 0.93
    
    # Save
    sf.write(f"music/{name}.wav", result, sr)
    print(f"   ✅ Saved: {name}.wav")
    
    return {'name': name, 'key1': key1, 'key2': key2, 'bpm1': bpm1, 'bpm2': bpm2, 
            'energy1': e1, 'energy2': e2, 'compatibility': compat, 'style': style}

if __name__ == '__main__':
    # Create improved fusions
    songs = [
        ("music/drake_in_my_feelings.wav", "music/travis_sicko_mode.wav"),
        ("music/marshmello_happier.wav", "music/edm_1.wav"),
    ]
    
    for i, (s1, s2) in enumerate(songs):
        improved_fusion(s1, s2, f"v4_improved_{i+1}", random.choice(['drop', 'chill', 'build', 'optimal']))
    
    print("\n✅ Improved fusions complete!")
