# AI Music Generation Models

## Best Open Source Alternatives to Suno

### ACE-Step 1.5 ⭐
- Quality: Between Suno v4.5 and v5
- Speed: <2s on A100, <10s on RTX 3090
- Duration: 10s to 10min
- VRAM: 4GB minimum, 12-16GB recommended
- Features: LoRA training, reference audio, cover generation
- GitHub: https://github.com/ace-step/ACE-Step-1.5

### YuE ⭐
- Full-song generation (minutes with vocals)
- Apache 2.0 license
- In-context style transfer
- GitHub: https://github.com/multimodal-art-projection/YuE

### MusicGen (Meta)
- 300M - 3.3B params
- Text-to-music with melody conditioning
- GitHub: https://github.com/facebookresearch/audiocraft

## Best for Hip-Hop/EDM
- **ACE-Step 1.5**: Best overall, 1000+ instruments
- **YuE**: Great full songs with lyrics

## Local Deployment

| Model | Min GPU | Recommended |
|-------|---------|-------------|
| ACE-Step 1.5 | 4GB | 12-16GB |
| MusicGen | 16GB | 24GB+ |
| YuE | 24GB | 80GB+ |

## Fine-tuning
- ACE-Step: LoRA training with 16GB VRAM
- ~1 hour training on RTX 3090 with 8 songs
