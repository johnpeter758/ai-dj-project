# AI DJ Project

VocalFusion's codebase for turning two parent songs into one intentional child arrangement.

## Product definition

For the canonical scope and permanent design rules, see [`docs/VOCALFUSION_PRODUCT_DEFINITION.md`](docs/VOCALFUSION_PRODUCT_DEFINITION.md).

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

GitHub Actions now runs the regression suite on Python 3.11, 3.12, and 3.13 with pip caching plus `libsndfile` / `ffmpeg` installed, so local test failures should generally reproduce in CI.

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

### Run the critic sprint benchmark harness across fixed fixture pairs

```bash
python3 scripts/critic_benchmark_harness.py scripts/critic_benchmark_harness.example.json \
  --output runs/checkpoint/critic_benchmark_harness.json \
  --markdown runs/checkpoint/critic_benchmark_harness.md \
  --spec-output-dir runs/checkpoint/critic_specs
```

Purpose:
- compare baseline vs adaptive vs critic outputs for the same fixture pair
- enforce the expected critic > adaptive > baseline ordering on every curated fixture
- persist per-fixture benchmark specs so regressions are debuggable instead of opaque
- roll fixture-level outcomes into one sprint scoreboard

### Gate generated vocals before expensive full renders

```bash
python3 scripts/vocal_quality_gate.py path/to/vocals.wav \
  --output runs/checkpoint/vocal_quality.json
```

Purpose:
- catch clipping, pitch instability, hiss, and thin/noisy vocal artifacts early
- produce an explicit quality score + verdict (`strong`, `usable`, `borderline`, `reject`)
- fail fast on bad generations before running full benchmark/render loops

### Build a reference-driven improvement brief for one fusion

```bash
python3 scripts/listen_feedback_loop.py runs/fusion_candidate runs/reference_a runs/reference_b --output runs/checkpoint/listen_feedback_brief.json
```

Purpose:
- compare one fusion directly against known-good references
- summarize the component gaps vs those references
- map the worst gaps to the next code targets in planner / render / evaluator

### Run a bounded closed-loop listener improvement cycle

```bash
python3 scripts/closed_loop_listener_runner.py song_a.mp3 song_b.mp3 runs/reference_a runs/reference_b \
  --output-root runs/closed_loop/demo \
  --max-iterations 3 \
  --quality-gate 85 \
  --change-command "python scripts/your_patch_step.py --context {change_context_json}" \
  --test-command "./.venv/bin/python -m pytest -q tests/test_closed_loop_listener_runner.py"
```

Purpose:
- render a candidate
- compare it against known-good references
- write a code-targeted improvement brief
- explicitly keep or reject each candidate based on listener progress vs the prior best
- write a structured change packet for external patch steps
- optionally call an external patch step
- rerun tests
- stop on plateau or quality-gate success while tracking the best iteration

Useful loop artifacts per iteration:
- `listen_feedback_brief.json` — ranked gaps vs references plus planner/render feedback routes
- `listener_assessment.json` — listener-agent decision/rank for the candidate
- `change_command_context.json` — structured fields for automation-friendly patch steps
- `change_request.md` — concise human-readable implementation brief for the same iteration
- `render/` — fusion output directory for that iteration

The saved `closed_loop_report.json` also records `candidate_keep_decision` per iteration and aggregate `candidate_decisions` counts so the loop is explicit about which candidates were kept as the new best versus rejected.

To inspect the available `{placeholder}` fields for `--change-command` and `--test-command`:

```bash
python3 scripts/closed_loop_listener_runner.py --print-template-fields
```

## Minimal 2-input -> 1-output entrypoint

Use this when you want the simplest possible contract (two songs in, one fused output out):

```bash
python scripts/two_input_one_output.py \
  /path/to/song_a.mp3 \
  /path/to/song_b.mp3 \
  --output-dir runs/simple_fuse_$(date +%Y%m%d_%H%M%S) \
  --arrangement-mode pro
```

It writes `two_input_one_output_report.json` in the output folder with command,
return code, artifact paths, and stdout/stderr tails.

## 24/7 advanced autopilot

### Run one autonomous mastering cycle (benchmark -> gate -> optional commit/push)

```bash
python3 scripts/autonomous_mastering_cycle.py \
  --min-song-likeness 80 \
  --min-score 70 \
  --min-delta 0.1 \
  --commit-on-promote \
  --push-on-promote
```

This writes a machine-readable cycle report to `runs/autopilot/last_mastering_cycle.json`.

### Run one bounded orchestrated cycle (cron-safe)

```bash
python3 scripts/autopilot_orchestrator.py \
  --command "python3 scripts/autonomous_mastering_cycle.py --min-song-likeness 80 --min-score 70 --min-delta 0.1 --commit-on-promote --push-on-promote" \
  --single-cycle \
  --state-path runs/autopilot/state.json
```

### Run continuous loop with checkpointing

```bash
python3 scripts/autopilot_orchestrator.py \
  --command "python3 scripts/autonomous_mastering_cycle.py --min-song-likeness 80 --min-score 70 --min-delta 0.1 --commit-on-promote --push-on-promote" \
  --sleep-seconds 90 \
  --state-path runs/autopilot/state.json
```

### Stop safely

```bash
touch AUTOPILOT_STOP
```

### Evaluate challenger promotion

```bash
python3 scripts/champion_gate.py \
  --benchmarks-dir runs \
  --champion-pointer runs/champion/current.json \
  --champion-history runs/champion/history.json \
  --min-song-likeness 80 \
  --min-score 70 \
  --min-delta 0.5
```

### Gateway mesh readiness

- Reliability template: `config/litellm.reliability.yaml`
- Ops runbook: `docs/gateway-mesh-readiness.md`

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

- `runs/`, `artifacts/`, `logs/`, and local server/process stdout/stderr captures are local-only outputs and are gitignored.
- `data/analyses/` can hold small checked-in fixture/reference JSON when it is useful for tests or reproducible examples.
- Local runtime state should not be committed: project DB files (`*.db`, `*.sqlite*`, `*.duckdb`) plus generated `data/mixes/`, `data/cache/`, `data/local/`, and `data/tmp/` content are treated as machine-local state.
- `docs/` is organized into `render/`, `research/`, and `notes/`.
- This repo still contains older exploratory modules under `src/`; the current canonical path is `src/core/`.

## GitHub

https://github.com/johnpeter758/ai-dj-project
