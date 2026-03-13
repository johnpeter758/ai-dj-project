# VocalFusion Pyramid Architecture

VocalFusion should be understood as a hierarchical music system, not a flat collection of modules.

The architecture is intentionally top-down:
- the top layers decide *what kind of child song should exist*
- the middle layers decide *which musical material should fill that song*
- the lower layers decide *how to execute it legally and deterministically*
- the evaluator runs across the stack and reports whether the result is actually getting better

## Pyramid view

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

## Layer definitions

### 1. VocalFusion Brain
Purpose: own the system's taste, product direction, architecture rules, and durable memory.

This layer decides:
- what counts as a strong child song
- what tradeoffs are acceptable
- which system improvements matter most
- how evaluation should be interpreted
- which architectural rules are non-negotiable

Examples:
- planner-first architecture is core
- replayability matters more than novelty alone
- raw overlay is not a finished fusion strategy
- one canonical path is better than duplicate parallel implementations

## 2. Song Director
Purpose: design the child song at the whole-track level before local source selection begins.

This layer owns:
- child section program
- macro energy arc
- section-role sequence
- high-level parent contribution strategy

Typical roles:
- intro
- setup / verse
- build
- payoff / chorus
- reset / breakdown
- final payoff
- outro

This layer should answer:
- what is the child trying to feel like over time?
- where should energy rise, release, and return?
- how many major sections should the child contain?

## 3. Section Planner
Purpose: convert the song-level program into concrete windows from the parent songs.

This layer owns:
- compatibility scoring
- phrase-window ranking
- role-fit scoring
- chronology / reuse constraints
- timeline generation in bars

This is where VocalFusion chooses:
- which parent supplies each section
- which phrase window is legally and musically appropriate
- how to avoid rewinds, over-reuse, and weak role matches

## 4. Transition / Ownership Layer
Purpose: make section handoffs musical and keep listener focus clear.

This layer owns:
- cut / blend / lift / drop / swap behavior
- overlap policy
- low-end ownership
- foreground ownership
- lead-vocal ownership
- seam legality and crowding control

This layer should answer:
- how do sections connect without sounding pasted together?
- who owns the low end, foreground, and lead at each seam?
- when is overlap safe, and when should the render collapse to one owner?

## 5. Render Engine
Purpose: execute the chosen arrangement deterministically.

This layer owns:
- source resolution
- time/pitch alignment
- stem scheduling
- transition application
- export and manifest writing

This layer should stay:
- deterministic
- inspectable
- contract-driven
- easy to debug

The render engine is not the musical brain. It executes the plan.

## 6. Analysis Foundation
Purpose: extract stable musical facts from source songs.

This layer owns:
- audio loading
- tempo and key detection
- beat/downbeat and bar grid estimation
- structure and phrase estimation
- energy profiling
- optional stem separation
- canonical SongDNA artifacts

Everything above depends on this foundation being reliable.

## Cross-cutting spine: Listen / Evaluator
Purpose: evaluate outputs, compare candidates, reject weak results, and eventually feed quality signals back upward.

This layer owns:
- structure scoring
- groove scoring
- energy-arc scoring
- transition/seam scoring
- coherence scoring
- mix / ownership sanity checks
- compare-listen decision support
- benchmark-style regression gates

The evaluator is not a replacement for planning.
It is the feedback spine that helps the rest of the pyramid improve.

## Why this architecture matters

Without a pyramid:
- too much logic collapses into one opaque layer
- failures are hard to localize
- improvements become chaotic and hard to validate
- the system tends toward demos instead of product architecture

With a pyramid:
- each layer has a clear job
- failures are easier to diagnose
- evaluation can target specific layers
- the system remains modular, controllable, and extensible

## Practical repo mapping

```text
src/core/analysis/    -> Analysis Foundation
src/core/planner/     -> Song Director + Section Planner
src/core/render/      -> Transition / Ownership execution + Render Engine
src/core/evaluation/  -> Listen / Evaluator feedback spine
```

The "VocalFusion Brain" itself is expressed through:
- architecture docs
- memory
- evaluation philosophy
- project rules
- long-term planning notes

## Near-term implementation implications

Given the current repo state, the pyramid suggests this order:
1. strengthen the evaluator so it can rank good vs weak outputs more reliably
2. expand the macro planner beyond the current stubby section program
3. improve transition intent and ownership behavior at seams
4. use listen/compare-listen to benchmark whether the higher layers are genuinely improving

## Design rules
- Protect the planner-first architecture.
- Keep the evaluator explanatory, not just numeric.
- Prefer deterministic render behavior over hidden magic.
- Use bars, phrases, sections, and energy arcs as the canonical planning language.
- Treat the repository as a layered system, not a random toolkit.
