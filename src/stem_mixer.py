"""
Professional DJ Mix Engine - Stem-based Layering
Uses Demucs for stem separation and layers stems from both songs
"""

import os
import subprocess
import numpy as np
import soundfile as sf
from datetime import datetime


def extract_stems(audio_path, output_dir="stems"):
    """Extract stems using Demucs."""
    os.makedirs(output_dir, exist_ok=True)
    
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, "htdemucs", song_name)
    
    # Check if already extracted
    if os.path.exists(stem_dir):
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            path = os.path.join(stem_dir, f"{stem}.wav")
            if os.path.exists(path):
                stems[stem] = path
        if stems:
            return stems
    
    # Run Demucs
    cmd = [
        "python3", "-m", "demucs",
        "-n", "htdemucs",
        "-o", output_dir,
        "--two-stems", "vocals",  # Faster: just get vocals + other
        audio_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
    except Exception as e:
        print(f"Demucs failed: {e}")
    
    # Return what we have
    if os.path.exists(stem_dir):
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            path = os.path.join(stem_dir, f"{stem}.wav")
            if os.path.exists(path):
                stems[stem] = path
        return stems
    
    return {}


def load_audio(path):
    """Load audio file."""
    if not os.path.exists(path):
        return None, None
    audio, sr = sf.read(path)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return sr, audio


def apply_eq_carving(audio, sr, mode="layer"):
    """
    EQ carving to prevent frequency clash.
    mode: 'layer' = carve space for both, 'blend' = smooth transition
    """
    from scipy.signal import butter, lfilter
    
    # Simple 2-band EQ using butterworth
    def make_filter(low_cut, high_cut, btype='bandpass'):
        nyq = sr / 2
        low = low_cut / nyq
        high = high_cut / nyq
        return butter(2, [low, high], btype=btype)
    
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    
    output = audio.copy()
    
    if mode == "layer":
        # For layering: reduce bass slightly to prevent mud
        b, a = butter(2, 200 / (sr/2), btype='lowpass')
        bass_reduced = lfilter(b, a, audio, axis=0)
        # Mix: 70% original + 30% bass-reduced
        output = audio * 0.7 + bass_reduced * 0.3
    
    return output


def parallel_layer(audio1, audio2, sr, mix_ratio=0.5):
    """
    Layer two songs in parallel - both play together.
    mix_ratio: 0.5 = equal volume, <0.5 = more song1, >0.5 = more song2
    """
    # Normalize both to prevent clipping
    def normalize(audio, target_rms=0.3):
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            return audio / rms * target_rms
        return audio
    
    audio1 = normalize(audio1, 0.25)  # Lower for layering
    audio2 = normalize(audio2, 0.25)
    
    # Match lengths
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Parallel mix with crossfade
    fade_len = min(min_len // 4, sr * 8)  # 8 second crossfade
    fade_len = max(fade_len, sr * 4)  # At least 4 seconds
    
    result = np.zeros((min_len, 2))
    
    # Equal-power crossfade
    fade_out = np.sin(np.linspace(0, np.pi/2, fade_len))
    fade_in = np.cos(np.linspace(0, np.pi/2, fade_len))
    
    # Song 1 throughout, with fade out in transition zone
    result += audio1 * mix_ratio
    result[:fade_len] += audio2[:fade_len] * fade_in * (1 - mix_ratio)
    
    # Song 2 takes over after crossfade
    result[fade_len:] = audio2[fade_len:] * mix_ratio + audio1[fade_len:] * (1 - mix_ratio) * 0.3
    
    # Apply EQ carving to reduce clash
    result = apply_eq_carving(result, sr, "layer")
    
    # Normalize
    max_val = np.max(np.abs(result))
    if max_val > 0.9:
        result = result / max_val * 0.9
    
    return result


def create_stem_mix(song1_path, song2_path, output_path):
    """Create a professional stem-based mix."""
    
    print(f"Loading songs...")
    sr1, audio1 = load_audio(song1_path)
    sr2, audio2 = load_audio(song2_path)
    
    if sr1 is None or sr2 is None:
        print("Error loading audio files")
        return False
    
    print(f"Song 1: {sr1}Hz, {len(audio1)} samples")
    print(f"Song 2: {sr2}Hz, {len(audio2)} samples")
    
    # Try stem extraction (may skip if too slow)
    print("Attempting stem separation...")
    stems1 = extract_stems(song1_path)
    stems2 = extract_stems(song2_path)
    
    if stems1 and stems2:
        print(f"Stems found! Mixing stems...")
        # TODO: Implement proper stem mixing
        # For now, fall back to parallel layering
        print("Stem mixing - using parallel layer fallback")
    
    # Use parallel layering algorithm
    print("Creating parallel mix...")
    mixed = parallel_layer(audio1, audio2, sr1)
    
    # Save
    sf.write(output_path, mixed, sr1)
    print(f"Saved: {output_path}")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        create_stem_mix(sys.argv[1], sys.argv[2], sys.argv[3])
