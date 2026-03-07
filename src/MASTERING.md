# Auto-Mastering System

## LUFS Targets
| Platform | Target LUFS | True Peak |
|---------|-------------|-----------|
| Spotify | -14 | -1 dB |
| Apple Music | -16 | -1 dB |
| Tidal | -14 | -1.5 dB |

## Multi-Band Compressor
- 6 bands
- Threshold: -24 to -12 dB
- Ratio: 2:1 to 8:1
- Attack: 1-10ms
- Release: 50-200ms

## Stereo Widener
- Mid/side processing
- Boost highs in sides
- Keep sub mono

## Usage
```python
from auto_master import AutoMaster

master = AutoMaster(platform='spotify')
mastered, report = master.master(audio)
```
