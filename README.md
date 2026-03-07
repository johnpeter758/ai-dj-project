# 🎛️ Peter - AI DJ

Autonomous AI DJ that creates professional music fusions.

## Features

- **Smart Analysis** - Key (Camelot), BPM, energy detection
- **Harmonic Mixing** - Camelot wheel compatibility
- **Professional Quality** - Soft clipping, gain staging
- **Self-Learning** - Improves from your ratings

## Fusions

- **200** VIP fusions (self-assessed)
- **82** Pro fusions  
- **750** Standard fusions

## Quick Start

```bash
# Analyze a song
python src/fusion_engine_v3.py analyze <song.wav>

# Create fusion
python src/fusion_engine_v3.py fuse <song_a> <song_b>
```

## Dashboard

Run `python server.py` and open http://localhost:5000

## Tech Stack

- Python, librosa, numpy
- Camelot wheel for key compatibility
- Professional audio processing

## License

MIT
