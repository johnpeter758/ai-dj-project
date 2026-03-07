# Beat Matching

## Detection
- Onset detection (spectral flux)
- BPM estimation via autocorrelation
- Libraries: aubio, librosa, Essentia

## Time-Stretching
- **Phase Vocoder**: Polyphonic (librosa)
- **PSOLA**: Monophonic/vocals
- **SOLA**: Optimal crossfade

## BPM Matching
- Stretch factor = source_bpm / target_bpm
- Keep changes <20%

## Grid Alignment
- Build beat grid from detected beats
- Use adaptive grids for imperfect tracks

## Beat Sync Mixing
- Intro/outro blending
- Beat jumping
- Combine with key detection
