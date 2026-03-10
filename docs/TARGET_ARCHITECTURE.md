# Target Architecture

## Product goal
Build VocalFusion into a producer-grade AI music system that:
- analyzes parent songs
- extracts reusable song DNA
- plans a child arrangement in bars/phrases/sections
- renders a coherent musical result
- evaluates output quality for iterative improvement

## Canonical subsystem layout

### 1. Analysis
Purpose: extract stable musical facts from source songs.

Owns:
- audio loading
- tempo and key detection
- structure and phrase estimation
- energy profiling
- optional stem separation
- canonical Song DNA artifact

### 2. Planner
Purpose: transform two parent song DNA objects into an intentional child arrangement plan.

Owns:
- compatibility scoring
- section alignment
- phrase-level planning
- energy arc planning
- transition planning
- timeline generation in bars

### 3. Render
Purpose: render the planned child arrangement deterministically.

Owns:
- time/pitch alignment
- stem scheduling
- transition application
- level balancing
- output export

### 4. Evaluation
Purpose: score outputs and guide improvement.

Owns:
- audio quality checks
- structural plausibility checks
- fusion coherence scoring
- human rating ingestion
- experiment comparison

## Architectural rules
- planner is the core differentiator
- raw overlay is not a finished fusion strategy
- prefer deterministic render paths over vague generation claims
- experiments should produce artifacts and notes
- broad speculative modules should not crowd the active core
