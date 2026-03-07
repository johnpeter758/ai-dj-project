# Stem Generation System

## Best Models
- **Demucs v4 (htdemucs_ft)**: Best quality, 9.0 SDR
- **MDX-Net**: Fast, good quality
- **Spleeter**: Fastest, 2-stem

## Installation
```bash
pip install demucs
# Requires ffmpeg
```

## Commands
```bash
# 4-stem separation
demucs song.mp3

# Vocals only (acappella)
demucs --two-stems=vocals song.mp3

# Best quality
demucs -n htdemucs_ft --shifts 5 song.mp3

# CPU with less memory
demucs -d cpu --segment 10 song.mp3
```

## Stems Output
- vocals.wav
- drums.wav
- bass.wav
- other.wav

## Python Usage
```python
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
# Separate audio into stems
```

## Architecture
1. Pre-process (convert, resample 44100Hz)
2. Demucs separation
3. Post-process (normalize, export)
4. Output stems or remix
