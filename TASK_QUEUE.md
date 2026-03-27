# VocalFusion Task Queue

Last updated: 2026-03-26 20:48 EDT (post musician-first revamp plan + no-intro-fallback rerun)
Owner: execution operator

## Current Task (active now)
1. **Phase 2 of revamp: enforce phrase-level continuity constraints (musician-first composition)**
   - Why: latest rerun `quality_push_after_no_intro_fallback_20260326_204820` removed intro-led fallback and recovered quality (`song_likeness=54.2`, `transition=61.2`), but still misses floor (`55.0`) and remains section-alternation-heavy.
   - Concrete change target:
     - add arrangement constraint limiting excessive owner switching and requiring at least one sustained integrated two-parent section in pro candidate generation.
   - Files likely touched:
     - `ai_dj.py`
     - `src/core/planner/*` and/or `src/core/render/*`
     - tests for shortlist/planner continuity behavior.

## Next Task (auto-start immediately after current)
1. **Run one pro benchmark and verify continuity metrics move in the right direction**
   - Command:
     - `python ai_dj.py fusion runs/quality_push_after_eaa9ee0_20260325_231707/relax_clip.mp3 runs/quality_push_after_eaa9ee0_20260325_231707/treasure_clip.mp3 --arrangement-mode pro --output runs/quality_push_after_<new_commit>_<timestamp>`
   - Success check:
     - best candidate keeps `song_likeness >= 54.2` while reducing medley behavior (`owner_switch_ratio` down and/or `integrated_two_parent_section_ratio` up).

## Blocked Tasks
1. **Promote a pro winner artifact automatically**
   - Blocker: latest run (`quality_push_after_no_intro_fallback_20260326_204820`) still has no floor-pass candidate (`song_likeness < 55.0`).
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
