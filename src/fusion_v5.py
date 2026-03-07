"""
Fusion Engine v5 - Professional Grade
Equal-power crossfade, warmth processing
"""

import numpy as np
import librosa
import soundfile as sf

def professional_fusion(song_a, song_b, name):
    """Create professional-grade fusion"""
    
    # Load
    y1, sr = librosa.load(song_a, sr=44100)
    y2, sr2 = librosa.load(song_b, sr=44100)
    
    # Duration
    dur = min(len(y1), len(y2), 120 * sr)
    y1, y2 = y1[:dur]*0.85, y2[:dur]*0.85
    
    # Professional split at 50%
    split = dur // 2
    crossfade = int(4 * sr)  # 4 second crossfade
    
    result = np.zeros(dur)
    
    # First half
    result[:split] = y1[:split]
    
    # Equal-power crossfade (smoother than linear)
    cf_len = min(crossfade, dur - split)
    fade = np.linspace(0, np.pi/2, cf_len)
    result[split:split+cf_len] = (
        y1[split:split+cf_len] * np.cos(fade) +
        y2[split:split+cf_len] * np.sin(fade)
    )
    
    # Second half
    if split + cf_len < dur:
        result[split+cf_len:] = y2[split+cf_len:]
    
    # Soft clipping for analog warmth
    result = np.tanh(result * 1.2) / 1.2
    
    # Limiter
    peak = np.max(np.abs(result))
    if peak > 0.95:
        result = result * (0.95 / peak)
    
    # Normalize
    result = result / np.max(np.abs(result)) * 0.91
    
    # Save
    sf.write(f"music/{name}.wav", result, sr)
    
    return name

if __name__ == "__main__":
    # Example
    professional_fusion(
        "music/drake_in_my_feelings.wav",
        "music/Travis Scott BUTTERFLY EFFECT.wav",
        "v5_professional_001"
    )
    print("✅ Professional fusion created!")
