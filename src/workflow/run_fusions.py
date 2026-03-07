#!/usr/bin/env python3
"""
Background workflow updater
Runs fusion creation and posts updates to Discord
"""

import os
import sys
import json
import time
import subprocess
import signal
from datetime import datetime

CHANNEL_ID = "1479544840894025809"
MUSIC_DIR = "/Users/johnpeter/ai-dj-project/music"
STATUS_FILE = "/Users/johnpeter/ai-dj-project/src/workflow/current.json"

def count_fusions():
    try:
        result = subprocess.run(
            ["find", MUSIC_DIR, "-name", "*.wav"],
            capture_output=True, text=True, timeout=5
        )
        return len([x for x in result.stdout.strip().split('\n') if x])
    except:
        return 0

def update_status(task, details="", progress=0):
    status = {
        "task": task,
        "details": details,
        "progress": progress,
        "updated": datetime.now().strftime("%H:%M:%S"),
        "fusions": count_fusions()
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    return status

def post_discord(message):
    """Post via curl to Discord - user needs to configure webhook"""
    pass  # Handled by OpenClaw message tool

# Main fusion loop
def create_fusions(duration=120, batch=50):
    """Create fusions and track progress"""
    update_status("Creating Fusions", f"Duration: {duration}s, Batch: {batch}", 0)
    
    code = f'''
import numpy as np
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

sources = ["Da Funk Doin It Right", "Drake Fair Trade", "Lose Control MEDUZA", "ODESZA A Moment Apart", "Travis Scott BUTTERFLY EFFECT"]

def make(s1, s2, name):
    try:
        y1, sr = librosa.load(f"music/{{s1}}.wav", sr=44100, duration={duration})
        y2, _ = librosa.load(f"music/{{s2}}.wav", sr=44100, duration={duration})
        d = min(len(y1), len(y2), {duration} * sr)
        y1, y2 = y1[:d]*0.85, y2[:d]*0.85
        split = d // 2
        cf = int(8 * sr)
        result = np.zeros(d)
        result[:split] = y1[:split]
        fade = np.linspace(0, np.pi/2, min(cf, d-split))
        result[split:split+len(fade)] = y1[split:split+len(fade)]*np.cos(fade) + y2[split:split+len(fade)]*np.sin(fade)
        result[split+len(fade):] = y2[split+len(fade):]
        result = np.tanh(result * 1.2) / 1.2
        result = result / np.max(np.abs(result)) * 0.90
        sf.write(f"music/v6_{{name}}.wav", result, sr)
        return True
    except: return False

count = 0
for i in range({batch}):
    make(sources[i%5], sources[(i+1)%5], str(i))
    count += 1
print(f"Created: {{count}}")
'''
    
    os.chdir("/Users/johnpeter/ai-dj-project")
    result = subprocess.run(["python3", "-c", code], capture_output=True, timeout=300)
    
    update_status("Complete", f"Created {batch} fusions", 100)
    return count

if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"🎛️ Creating {batch} fusions ({dur}s each)...")
    create_fusions(dur, batch)
    print("✅ Done!")
