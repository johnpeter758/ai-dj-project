# AI DJ Project

VocalFusion's codebase for analyzing two parent songs, building a child arrangement plan, and rendering a first deterministic fusion output.

## Current state

This repo now supports a real first-pass end-to-end flow:
- analyze two songs into DNA artifacts
- compute a compatibility report
- build an arrangement plan
- render a deterministic fused output

It is **not** producer-grade yet. The main gap is still musical intelligence: stronger structure segmentation, better phrase/section ranking, and better planner decisions.

## Best entry points

- `ai_dj.py` — CLI for doctor / analyze / prototype / fusion
- `src/core/analysis/` — audio analysis and SongDNA generation
- `src/core/planner/` — compatibility + arrangement planning
- `src/core/render/` — deterministic render v1 stack
- `tests/` — targeted regression coverage for the current checkpoint
- `docs/README.md` — organized docs index

## Quick start

Use the project venv if you already have it:

```bash
/Users/johnpeter/venvs/vocalfusion-env/bin/python ai_dj.py doctor
```

Or set up a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 ai_dj.py doctor
```

## Main commands

### Analyze one track

```bash
python3 ai_dj.py analyze path/to/song.mp3 --output runs/checkpoint/song_dna.json
```

### Prototype on two songs

```bash
python3 ai_dj.py prototype path/to/song_a.mp3 path/to/song_b.mp3 --output-dir runs/prototype-001
```

Writes:
- `song_a_dna.json`
- `song_b_dna.json`
- `compatibility_report.json`
- `arrangement_plan.json`

### Render first-pass fusion

```bash
python3 ai_dj.py fusion path/to/song_a.mp3 path/to/song_b.mp3 --output runs/render-001
```

Writes:
- `child_raw.wav`
- `child_master.wav`
- `child_master.mp3` (if `ffmpeg` is installed)
- `render_manifest.json`

## Current checkpoint highlights

- deterministic render v1 stack exists and is tested
- prototype/fusion work on real MP3s in the project venv
- structure analysis now promotes phrase boundaries into coarse sections when novelty-only structure is sparse
- planner now chooses analyzed early/mid/late section candidates instead of fixed fake section placeholders

## What still needs work

- stronger section-role inference
- bar/phrase confidence scoring
- phrase-window ranking based on energy, repetition, and cross-parent compatibility
- better planner decisions for more intentional, professional-sounding outputs

## Repo shape

```text
ai-dj-project/
├── ai_dj.py
├── src/core/analysis/
├── src/core/planner/
├── src/core/render/
├── tests/
├── docs/
└── runs/  # local ignored artifacts
```

## Notes

- `runs/` is for local artifacts and is gitignored.
- `docs/` is organized into `render/`, `research/`, and `notes/`.
- This repo still contains older exploratory modules under `src/`; the current canonical path is `src/core/`.

## GitHub

https://github.com/johnpeter758/ai-dj-project
