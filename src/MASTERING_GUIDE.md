# Audio Quality Mastering Guide

## Mastering Chain
1. EQ (corrective)
2. Compression (subtle)
3. Saturation/harmonics
4. Limiting
5. Dithering

## Compression Settings

| Use | Ratio | Attack | Release | GR |
|-----|-------|--------|---------|-----|
| Vocal | 2:1-4:1 | 10-30ms | 100-200ms | 2-4dB |
| Bass | 4:1-8:1 | 1-5ms | 50-100ms | 3-6dB |
| Master | 1.5:1-3:1 | 5-10ms | 150-300ms | 1-3dB |
| Limiter | 10:1+ | 0.1-1ms | 50-100ms | 1-3dB |

## Loudness Targets
- Streaming: -14 LUFS
- Spotify: -14 LUFS
- Club EDM: -6 to -3 LUFS
- True Peak: -1dB

## Stereo Widening
- Mid/Side processing
- Haas effect
- Stereo imager plugins

## Soft Clipping (for warmth)
```python
result = np.tanh(audio * 1.3) / 1.3
```

## EQ Blending
- High-pass/low-pass to create space
- Match tonal character between tracks

## Key Tips
- Transparent > colored
- Subtle changes add up
- Reference commercial tracks
- Trust your ears
