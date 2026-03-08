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
# Start API server
python3 server.py

# Generate music
python3 ai_dj.py generate --genre house

# Run cleanup
python3 src/system_cleanup.py
```

## Modules

- **Generators**: drums, bass, chords, melody, synth, vocals
- **Effects**: reverb, delay, chorus, flanger, compressor, EQ...
- **Analysis**: beat detection, key detection, LUFS metering
- **Infrastructure**: API, CLI, database, caching, workflow

## GitHub

https://github.com/johnpeter758/ai-dj-project
