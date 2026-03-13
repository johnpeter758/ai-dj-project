# Core Layout Proposal

## Canonical active core
A cleaner professional structure should make the VocalFusion pyramid obvious:

```text
src/
  core/
    analysis/      # analysis foundation
    planner/       # song director + section planner
    render/        # transition/ownership execution + render engine
    evaluation/    # listen/evaluator feedback spine
  legacy/
    archived experiments and demoted modules
```

## Meaning of each area

### `core/analysis/`
Foundation layer.
- audio loading
- tempo
- key
- structure
- energy
- stems
- song DNA schema

### `core/planner/`
Middle planning layers.
- compatibility scoring
- section-program logic
- phrase/section alignment
- energy arc planning
- child arrangement timeline
- chronology / reuse control

### `core/render/`
Execution layer.
- time/pitch alignment
- source resolution
- stem scheduling
- transitions
- ownership contracts
- deterministic export path

### `core/evaluation/`
Feedback spine.
- quality checks
- fusion coherence checks
- listen / compare-listen
- experiment comparison
- ranking / rejection logic

### `legacy/`
- old prototypes
- broad speculative modules
- historical experiments worth preserving but not actively extending

## Rule
The active path should become obvious to a new engineer in under five minutes, and the repo should read top-down as a hierarchy rather than as unrelated subsystems.
