# VocalFusion Task Queue

Last updated: 2026-03-26 12:02 EDT (post combo-priority patch/test)
Owner: execution operator

## Current Task (active now)
1. **Run one pro benchmark cycle and compare against previous run**
   - Command:
     - `python ai_dj.py fusion runs/quality_push_after_eaa9ee0_20260325_231707/relax_clip.mp3 runs/quality_push_after_eaa9ee0_20260325_231707/treasure_clip.mp3 --arrangement-mode pro --output runs/quality_push_after_<new_commit>_<timestamp>`
   - Success check:
     - `fusion_selection.json.selection_policy.floor_pass_count >= 1` OR
     - best `song_likeness >= 55.0` with gate `pass`

## Next Task (auto-start immediately after current)
1. **If still below floor, raise adaptive combo quality by preferring payoff+build dual combos before payoff+verse**
   - Why: adaptive combo currently trails baseline combo; next leverage is stronger mid→climax continuity.
   - Files likely touched:
     - `ai_dj.py` combo priority tuple
     - `tests/test_auto_shortlist_fusion.py` combo-order regression

## Blocked Tasks
1. **Promote a pro winner artifact automatically**
   - Blocker: no candidate meets hard floors yet (`song_likeness >= 55`, `groove >= 60`, `structure >= 58`, pass gate).
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
