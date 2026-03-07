# Genre Diversity for AI Music

## Genre Classification
- MFCCs, chromagrams, spectral contrast
- CNN/RNN classifiers on Million Song Dataset
- Embedding-based classification

## Conditioning Methods
- **Embedding conditioning**: Genre → embedding → concat with text
- **Control tokens**: Tempo, key, instrumentation
- **Cross-attention**: Condition on genre vectors
- **LoRA/adapters**: Genre-specific modules

## Multi-Genre Models
| Model | Approach |
|-------|----------|
| MusicGen | Text conditioning |
| Jukebox | Hierarchical VQ-VAE + lyrics |
| Riffusion | Diffusion on spectrograms |
| ACE-Step | Multi-token conditioning |

## Genre Production Techniques

### EDM
- Synths, four-on-floor
- Sidechain, high compression
- Build-up → drop transitions

### Hip-Hop
- Trap hi-hats, boom-bap
- 808s, triplet flows
- Sample-based

### Pop
- Chord progressions
- Melodic hooks
- Vocal-forward

## Architecture
1. Transformer decoder (base)
2. Genre encoder → embedding
3. Style encoder (reference audio)
4. Multi-stream output
5. Classifier-free guidance
