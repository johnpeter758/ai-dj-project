# AI Vocal Generation

## Technologies
- WaveNet, Tacotron 2, FastSpeech, HiFi-GAN, Glow-TTS
- Encoder-decoder + neural vocoder

## Vocal Synthesis
- Concatenative (Vocaloid)
- Formant synthesis
- Neural waveform modeling

## Lyric Generation
- Use LLMs (ChatGPT, Claude) with genre prompts
- Suno/Udio have built-in lyrics

## Voice Cloning
- **ElevenLabs**: Instant (1-5 min samples)
- **15.ai**: 15-second samples
- **RVC**: Open-source (VITS-based)

## Open Source
- Bark, Coqui TTS, RVC
- So-Vits-SVC, DiffSinger
- OpenUtau

## Processing Chain
Pitch → EQ → Compression → Reverb → Delay → Harmony

## How Suno Does Vocals
- Uses Bark or proprietary
- GPT-style transformer
- 100+ voice presets
