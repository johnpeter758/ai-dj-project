# Target Architecture

## Product goal
Build VocalFusion into a producer-grade AI music system that:
- analyzes parent songs
- extracts reusable song DNA
- plans a child arrangement in bars/phrases/sections
- renders a coherent musical result
- evaluates output quality for iterative improvement

## Canonical pyramid layout

VocalFusion should be understood as a hierarchy.

### 1. VocalFusion Brain
Purpose: own taste, product logic, architecture rules, and long-term memory.

Owns:
- what a “good child song” means
- replayability / intentionality criteria
- architecture decisions
- evaluation philosophy
- durable memory and operator rules

### 2. Song Director
Purpose: decide whole-song form before local sourcing.

Owns:
- child section program
- macro energy arc
- role sequence (intro / setup / build / payoff / reset / outro)
- high-level parent contribution strategy

### 3. Section Planner
Purpose: transform the song-level program into concrete source windows.

Owns:
- compatibility scoring
- phrase-window ranking
- role-fit scoring
- chronology / reuse constraints
- timeline generation in bars

### 4. Transition / Ownership Layer
Purpose: control how sections hand off and who owns listener focus.

Owns:
- cut / blend / lift / drop / swap behavior
- overlap policy
- low-end ownership
- foreground ownership
- lead-vocal ownership
- seam legality and crowding control

### 5. Render Engine
Purpose: render the planned child arrangement deterministically.

Owns:
- time/pitch alignment
- source resolution
- stem scheduling
- transition application
- level balancing
- output export

### 6. Analysis Foundation
Purpose: extract stable musical facts from source songs.

Owns:
- audio loading
- tempo and key detection
- structure and phrase estimation
- energy profiling
- optional stem separation
- canonical Song DNA artifact

### Cross-cutting: Listen / Evaluator
Purpose: act as the feedback spine across the pyramid.

Owns:
- output ranking and rejection
- structural plausibility checks
- transition / seam judgment
- coherence scoring
- mix / ownership sanity checks
- experiment comparison
- eventually planner feedback loops

## Architectural rules
- planner is the core differentiator
- raw overlay is not a finished fusion strategy
- prefer hierarchical planning over opaque end-to-end generation
- prefer deterministic render paths over vague generation claims
- evaluator should rank and explain, not only report
- experiments should produce artifacts and notes
- broad speculative modules should not crowd the active core
