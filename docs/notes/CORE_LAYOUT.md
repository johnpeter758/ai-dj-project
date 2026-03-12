# Core Layout Proposal

## Canonical active core
A cleaner professional structure should converge toward:

```text
src/
  core/
    analysis/
    planner/
    render/
    evaluation/
  legacy/
    archived experiments and demoted modules
```

## Meaning of each area

### `core/analysis/`
- audio loading
- tempo
- key
- structure
- energy
- stems
- song DNA schema

### `core/planner/`
- compatibility scoring
- phrase/section alignment
- energy arc planning
- child arrangement timeline

### `core/render/`
- time/pitch alignment
- stem scheduling
- transitions
- deterministic export path

### `core/evaluation/`
- quality checks
- fusion coherence checks
- experiment comparison
- human rating ingestion

### `legacy/`
- old prototypes
- broad speculative modules
- historical experiments worth preserving but not actively extending

## Rule
The active path should become obvious to a new engineer in under five minutes.
