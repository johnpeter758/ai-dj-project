# AI DJ Project

An autonomous AI music system focused on analyzing two parent songs, producing first-pass compatibility/planning artifacts, and rendering an early fusion prototype.

## Current first checkpoint

The most reliable user-runnable checkpoint today is:
1. verify dependencies
2. analyze one song successfully
3. generate prototype JSON artifacts for two songs
4. try the first-pass render once the local render path is stable

This repo is **not** at a polished product stage yet. The safest entry point is the CLI in `ai_dj.py`.

## Repository structure

```text
ai-dj-project/
├── ai_dj.py          # Main CLI for analysis / prototype / fusion
├── src/              # Core modules and legacy modules
├── docs/             # Research & render/planner notes
├── tests/            # Targeted regression tests
├── deploy/           # Docker and deployment configs
├── music/            # Local music library (if present)
└── runs/             # Recommended output location for local checkpoints
```

## Setup

Use Python 3.11+ if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Recommended verification right after install:

```bash
python3 ai_dj.py doctor
python3 -m pytest -q tests/test_prototype_cli.py tests/test_render_stack.py
```

Optional but recommended for render output:

- `ffmpeg` on your PATH for MP3 export during `fusion`

## Quick start

### 1) Check environment readiness

This now works even if the audio stack is only partially installed.

```bash
python3 ai_dj.py doctor
```

What to expect:
- exit code `0`: analysis dependencies are installed
- exit code `1`: one or more required Python packages are missing
- `test_ready`: whether `pytest` is importable in the current Python environment
- `ffmpeg` is reported separately because WAV render can still work without MP3 export

You can also save the report:

```bash
python3 ai_dj.py doctor --output runs/doctor.json
```

### 2) Analyze one song

```bash
python3 ai_dj.py analyze path/to/song.wav --output runs/checkpoint/song_dna.json
```

Expected output:
- `runs/checkpoint/song_dna.json`

### 3) Run the two-song prototype flow

```bash
python3 ai_dj.py prototype path/to/song_a.wav path/to/song_b.wav --output-dir runs/prototype-001
```

Expected outputs:
- `song_a_dna.json`
- `song_b_dna.json`
- `compatibility_report.json`
- `arrangement_plan.json`

### 4) Try the first-pass render

```bash
python3 ai_dj.py fusion path/to/song_a.wav path/to/song_b.wav --output runs/render-prototype
```

Expected outputs:
- `child_raw.wav`
- `child_master.wav`
- `child_master.mp3` if `ffmpeg` is installed
- `render_manifest.json`

## CLI reference

```bash
python3 ai_dj.py --help
```

Main commands:
- `doctor` — check local dependency readiness
- `analyze` — inspect one source track and write JSON
- `prototype` — generate the current two-song planning checkpoint
- `fusion` — render the current first-pass fused output
- `generate` — placeholder command for future generation work

## Notes on current behavior

- `prototype` and `fusion` now validate that the input song paths exist before starting work.
- `fusion --genre` and `fusion --bpm` are accepted for forward compatibility, but the current renderer does not apply them yet.
- If Python audio dependencies are missing, the CLI now prints a direct install hint instead of crashing with an import traceback.
- If `ffmpeg` is missing, render can still produce WAV outputs, but MP3 export will be skipped.

## Common failure cases

### `ModuleNotFoundError: librosa` or similar

Install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### `pytest: command not found`

Use the interpreter-aware form so you run tests in the same environment where dependencies were installed:

```bash
python3 -m pytest -q
```

### `Error: song_a not found` / `track not found`

Pass a real local audio file path. The CLI now checks this up front.

### No MP3 output after `fusion`

Install `ffmpeg` and re-run the command. The WAV artifacts should still be present.

## Current prototype scope

Today’s CLI is best treated as a **checkpoint harness** for:
- song analysis
- compatibility inspection
- stub arrangement planning
- first-pass deterministic rendering

It is not yet a full autonomous DJ/producer workflow.

## GitHub

https://github.com/johnpeter758/ai-dj-project
