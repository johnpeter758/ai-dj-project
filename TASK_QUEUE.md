# VocalFusion Task Queue

Last updated: 2026-03-26 12:03 EDT (post section-index fix + combo-priority correction)
Owner: execution operator

## Current Task (active now)
1. **Run one pro benchmark cycle after section-index fix + intro/outro combo penalty and compare against baseline run**
   - Command:
     - `python ai_dj.py fusion runs/quality_push_after_eaa9ee0_20260325_231707/relax_clip.mp3 runs/quality_push_after_eaa9ee0_20260325_231707/treasure_clip.mp3 --arrangement-mode pro --output runs/quality_push_after_<new_commit>_<timestamp>`
   - Success check:
     - best `song_likeness >= 54.3` (recover prior best) and then target `>= 55.0`

## Next Task (auto-start immediately after current)
1. **If still below 55, target mix-sanity bottleneck by reducing vocal competition during payoff seams**
   - Why: top candidate reports repeatedly show high `vocal_competition_risk` and no integrated two-parent sections.
   - Files likely touched:
     - `src/core/render/*` seam/overlap ownership policy
     - `tests/test_render_stack.py` or related render regression tests

## Blocked Tasks
1. **Promote a pro winner artifact automatically**
   - Blocker: no candidate meets hard floors yet (`song_likeness >= 55`, `groove >= 60`, `structure >= 58`, pass gate).
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
