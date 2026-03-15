# AI DJ Project

VocalFusion's codebase for turning two parent songs into one intentional child arrangement.

## The VocalFusion pyramid

GitHub should be read top-down as a hierarchy, not as a flat bag of modules.

```text
                    VocalFusion Brain
         (taste, product rules, architecture, memory)
                              |
                         Song Director
        (whole-song form, section program, macro energy arc)
                              |
                        Section Planner
   (phrase windows, role fit, chronology, compatibility, reuse)
                              |
                    Transition / Ownership Layer
     (handoffs, overlap policy, lead focus, seam legality)
                              |
                         Render Engine
     (deterministic resolve, stretch, schedule, export)
                              |
                     Analysis Foundation
   (tempo, bars, phrases, sections, energy, stems, SongDNA)

        Listen / Evaluator runs across the whole stack as the
        feedback spine: ranking, diagnosis, rejection, iteration.
```

## Current state

This repo now supports a real first-pass end-to-end flow:
- analyze two songs into DNA artifacts
- compute a compatibility report
- build an arrangement plan
- render a deterministic fused output
- score outputs with `listen` / `compare-listen`

It is **not** producer-grade yet. The main gap is still musical intelligence: stronger structure segmentation, richer section programs, better phrase/section ranking, cleaner transitions, and stronger quality judgment.

## Best entry points

- `ai_dj.py` — CLI for doctor / analyze / prototype / fusion / listen / compare-listen / benchmark-listen / listener-agent
- `src/core/analysis/` — analysis foundation and SongDNA generation
- `src/core/planner/` — song director + section planner
- `src/core/render/` — transition/ownership execution + deterministic render v1
- `src/core/evaluation/` — listen/evaluator feedback spine
- `tests/` — targeted regression coverage for the active core
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

### Gate multiple renders with the Listener agent

```bash
python3 ai_dj.py listener-agent runs/render_a runs/render_b runs/render_c --output runs/checkpoint/listener_agent.json
```

Purpose:
- reject non-songs before a human has to hear them
- shortlist only the strongest survivors for review
- preserve structured reasons/fixes for rejected outputs

### Run a fixed human-aligned listen benchmark gate

```bash
python3 scripts/listen_gate_benchmark.py scripts/listen_gate_benchmark.example.json --output runs/checkpoint/listen_gate_benchmark.json
```

Purpose:
- enforce labeled good / baseline / bad ordering
- fail fast when the listener starts preferring artistically worse outputs
- turn subjective quality expectations into a repeatable regression gate

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
