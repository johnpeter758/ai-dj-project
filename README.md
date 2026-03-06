# AI DJ Project - VocalFusion

Autonomous AI DJ that analyzes songs and creates professional mashups.

## Structure

```
ai-dj-project/
├── src/
│   └── vocalfusion.py    # Analysis engine
├── music/                 # Music library
├── logs/                  # Analysis logs
├── exports/               # Generated fusions
├── fusions/              # Saved mashups
└── README.md
```

## Setup

```bash
pip install librosa soundfile numpy
python src/vocalfusion.py analyze <song_file>
```

## Usage

```bash
# Analyze a song
python src/vocalfusion.py analyze music/track.wav --name "my-track"

# View analysis logs
cat logs/analyses.jsonl
```

## Obsidian Vault

Notes saved to: `/Users/johnpeter/obsidian-vault/`
