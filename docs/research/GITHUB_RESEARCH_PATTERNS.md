# GitHub Research: reusable patterns for VocalFusion / AI DJ architecture

## Why this matters
The local repo already points in the right direction in `docs/TARGET_ARCHITECTURE.md` and `docs/CORE_LAYOUT.md`: **analysis → planner → render → evaluation**. The strongest external repos reinforce that the winning move is not a giant flat surface of DJ/music modules, but a narrow, typed, artifact-driven core.

## Best external repos to borrow from

### 1) `mir-aidj/all-in-one`
- Repo: https://github.com/mir-aidj/all-in-one
- High-value files:
  - `src/allin1/analyze.py`
  - `src/allin1/typings.py`
  - `TRAINING.md`
- Reusable ideas:
  - A **single entrypoint** that runs a multi-stage analysis pipeline and returns one structured result object.
  - Strong use of **typed result dataclasses** (`AnalysisResult`, `Segment`) instead of loose dicts.
  - Saves intermediate byproducts only when useful, and cleans them up by default.
  - Treats long analysis as a **checkpointed batch job**: infer, then save result immediately.
  - Keeps research/training instructions separate from runtime analysis code.
- Best fit for VocalFusion:
  - Replace the current dict-heavy structure path with a typed `SongDNA` family that includes beats, downbeats, bar indices, phrase indices, and labeled sections.
  - Add a single canonical `analyze()` orchestration layer that owns demix, structure, and artifact persistence.

### 2) `facebookresearch/demucs`
- Repo: https://github.com/facebookresearch/demucs
- High-value files:
  - `demucs/api.py`
  - `demucs/separate.py`
- Reusable ideas:
  - Separation exposed as an **API object** (`Separator`), not just a subprocess call.
  - Explicit parameters for memory/perf tradeoffs: `segment`, `overlap`, `shifts`, `jobs`, `device`, callbacks.
  - Progress/callback hooks for long-running operations.
- Best fit for VocalFusion:
  - Current `src/core/analysis/stems.py` shells out to CLI only. Good for bootstrap, but weak for progress reporting, caching, retries, and segment tuning.
  - Wrap stem separation behind a provider interface like `StemSeparator`, even if Demucs remains the first backend.
  - Persist stem metadata beyond file paths: model, segment size, sample rate, stem confidence/notes, run duration.

### 3) `CPJKU/madmom`
- Repo: https://github.com/CPJKU/madmom
- High-value files:
  - `madmom/features/*`
  - `madmom/evaluation/*`
  - package layout documented in `README.rst`
- Reusable ideas:
  - Clean separation between **feature extraction**, **ML**, and **evaluation**.
  - Evaluation is a first-class package, not an afterthought.
  - Repo organization clearly distinguishes low-level audio, MIR features, models, utilities, and tests.
- Best fit for VocalFusion:
  - Mirror this split inside `src/core/analysis/`, `src/core/planner/`, and `src/core/evaluation/`.
  - Build planner evaluation as a sibling package, not mixed into rendering or dashboard code.
  - Introduce metric modules such as:
    - `section_alignment.py`
    - `energy_arc.py`
    - `transition_quality.py`
    - `fusion_coherence.py`

### 4) `MTG/essentia`
- Repo: https://github.com/MTG/essentia
- High-value areas:
  - `src/`
  - `utils/`
  - extractor-oriented docs in README
- Reusable ideas:
  - A large catalog of **small composable descriptor extractors**.
  - Strong bias toward **robust descriptors** and reusable executables.
  - Clear boundary between core algorithms and higher-level extraction workflows.
- Best fit for VocalFusion:
  - Keep the Song DNA schema compact, but make each field come from a composable extractor module.
  - Prefer “many stable descriptors” over one giant magical analyzer.
  - Good pattern for adding tonal/rhythmic/timbral descriptors without bloating planner logic.

### 5) `librosa/librosa`
- Repo: https://github.com/librosa/librosa
- High-value areas:
  - `librosa/segment.py`, `beat.py`, `feature/`, `sequence.py`
  - `tests/test_segment.py`, `tests/test_beat.py`, etc.
- Reusable ideas:
  - Mature decomposition into **small analysis functions** with heavy tests.
  - Tests organized by feature family, which keeps regressions easy to localize.
- Best fit for VocalFusion:
  - The current `tests/test_core_analysis.py` is a good seed, but far too thin.
  - Add feature-family tests for tempo, key, structure, stems, planner compatibility, and evaluation metrics.

## What the local repo is doing right
- `docs/TARGET_ARCHITECTURE.md` is the correct product frame.
- `docs/CORE_LAYOUT.md` correctly narrows active work to `analysis/`, `planner/`, `render/`, `evaluation/`.
- `src/core/analysis/analyzer.py` already acts like a canonical analysis entrypoint.
- `tests/test_core_analysis.py` proves the new core is beginning to be testable.

## Biggest current gaps vs strong repo patterns

### 1) Structure analysis is too weak for planner-first fusion
Current file: `src/core/analysis/structure.py`
- It estimates phrase boundaries from a fixed 4 bars × 4 beats heuristic.
- Section labels are generic (`section_0`, `section_1`, ...).
- No downbeat/bar-index artifact is emitted.

Why this matters:
A planner-first fusion engine needs **bar-accurate alignment**, phrase starts, section types, and confidence. `all-in-one` is the best immediate pattern here.

### 2) `SongDNA` needs richer typed musical timeline fields
Current file: `src/core/analysis/models.py`
- `SongDNA` is a single dataclass, but many subfields are still raw dicts.

Recommendation:
Break this into nested dataclasses/Pydantic-style models such as:
- `TempoAnalysis`
- `KeyAnalysis`
- `BeatGrid`
- `Section`
- `StructureAnalysis`
- `StemSet`
- `EnergyProfile`
- `SongDNA`

This matches the stronger pattern from `all-in-one/typings.py` and reduces schema drift.

### 3) Stem workflow is not yet production-minded
Current file: `src/core/analysis/stems.py`
- Uses subprocess CLI only.
- Returns paths, but not run metadata or chunking controls.

Recommendation:
Adopt a provider abstraction inspired by `demucs/api.py`:
- `StemSeparator` interface
- `DemucsSeparator` implementation
- config fields: model, device, shifts, overlap, segment, jobs
- return structured `StemArtifact` objects

### 4) No real planner implementation yet
Current local state:
- `src/core/planner/` exists but is empty

Recommendation:
Make planner the most explicit package in the repo:
- `compatibility.py`
- `section_alignment.py`
- `phrase_plan.py`
- `energy_arc.py`
- `timeline.py`
- `models.py`

The planner should consume only Song DNA artifacts and emit a deterministic **ChildArrangementPlan**.

### 5) Evaluation loop is underspecified
Current local state:
- `src/core/evaluation/` exists but is empty
- there is `src/quality_evaluator.py`, but the repo still feels split between old and new centers

Recommendation:
Borrow the “evaluation is its own subsystem” posture from `madmom/evaluation/`.
Create objective metrics before expanding render complexity.

## Recommended target artifact chain

### Analysis artifact
`SongDNA`
- source metadata
- tempo + confidence
- beat times
- downbeat times
- bar map
- phrase boundaries
- labeled sections
- energy curve
- optional stems
- analysis provenance

### Planner artifact
`ChildArrangementPlan`
- parent A/B references
- compatibility score breakdown
- section-to-section mapping
- bar timeline
- transition directives
- stem usage plan
- expected energy arc
- planner confidence / warnings

### Render artifact
`RenderManifest`
- exact source clips
- time-stretch / pitch-shift ops
- automation curves
- transition assets
- output stems / mix paths

### Evaluation artifact
`FusionScorecard`
- beat/bar alignment score
- harmonic compatibility score
- section plausibility score
- energy arc score
- transition cleanliness score
- stem bleed / artifact notes
- human ratings links

## Concrete repo-organization recommendation
Use the external pattern signal to finish the migration away from `src/` sprawl:

```text
src/
  core/
    analysis/
      analyzer.py
      models.py
      tempo.py
      key.py
      beatgrid.py
      structure.py
      energy.py
      stems.py
    planner/
      models.py
      compatibility.py
      section_alignment.py
      phrase_plan.py
      energy_arc.py
      timeline.py
    render/
      models.py
      scheduler.py
      transforms.py
      transitions.py
      export.py
    evaluation/
      metrics/
        alignment.py
        structure.py
        energy.py
        transitions.py
      scorecard.py
      compare.py
  legacy/
    ...demoted flat modules...
```

## Priority recommendations

### Highest priority
1. **Adopt `all-in-one`-style structure outputs**
   - beats, downbeats, beat positions, labeled segments
   - this is the fastest path to a real planner
2. **Define typed planner/evaluation artifacts now**
   - before writing render complexity
3. **Upgrade stem separation to a provider abstraction**
   - keep Demucs, improve orchestration

### Medium priority
4. Expand tests into feature families, following `librosa`/`madmom` style
5. Move old broad modules behind `legacy/` to make the active path obvious in under 5 minutes

### Lower priority
6. Pull in richer descriptors from Essentia-style modular extractors once planner contracts are stable

## Bottom line
The external repos strongly support the same conclusion as your local docs: **VocalFusion should be a planner-first system built on robust, typed analysis artifacts, not a large flat collection of music/DJ modules.**

If choosing only three immediate moves, do these:
1. strengthen structure analysis around beats/downbeats/segments,
2. define `ChildArrangementPlan` + `FusionScorecard`,
3. narrow the repo so `core/analysis → planner → render → evaluation` becomes the unmistakable center.
