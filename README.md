# AI DJ Project

An autonomous AI music system that generates, mixes, and masters music.

## Structure

```
ai-dj-project/
├── src/              # 122 Python modules
├── docs/             # Research & documentation
├── music/            # Music library
├── templates/        # Web dashboards
├── deploy/           # Docker & Kubernetes configs
├── ACE-Step-1.5/    # Music generation AI
└── logs/            # Application logs
```

## Quick Start

```bash
# Analyze one song to JSON
python3 ai_dj.py analyze path/to/song.wav --output out/song_dna.json

# Run the first end-to-end two-song prototype
python3 ai_dj.py prototype path/to/song_a.wav path/to/song_b.wav --output-dir runs/prototype-001
```

Prototype outputs:
- `song_a_dna.json`
- `song_b_dna.json`
- `compatibility_report.json`
- `arrangement_plan.json`

## Modules

- **Generators**: drums, bass, chords, melody, synth, vocals
- **Effects**: reverb, delay, chorus, flanger, compressor, EQ...
- **Analysis**: beat detection, key detection, LUFS metering
- **Infrastructure**: API, CLI, database, caching, workflow

## GitHub

https://github.com/johnpeter758/ai-dj-project
