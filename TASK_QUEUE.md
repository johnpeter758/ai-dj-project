# VocalFusion Task Queue

Last updated: 2026-03-26 12:24 EDT (post payoff+build combo-priority rerun)
Owner: execution operator

## Current Task (active now)
1. **Increase two-parent identity while preserving transition gains in 3-slot pro search**
   - Why: latest run recovered stronger metrics (`song_likeness=54.3`, `transition=61.0`, `overall=71.1`) but still blocks at `floor_pass_count=0`.
   - Concrete change:
     - require one core donor single (`verse/build/payoff/bridge`) in each mode before fallback to backbone-only/same-parent alternates.
   - Files likely touched:
     - `ai_dj.py` (`_build_auto_shortlist_variant_configs`)
     - `tests/test_auto_shortlist_fusion.py`

## Next Task (auto-start immediately after current)
1. **Run one pro benchmark and verify floor progress**
   - Command:
     - `python ai_dj.py fusion runs/quality_push_after_eaa9ee0_20260325_231707/relax_clip.mp3 runs/quality_push_after_eaa9ee0_20260325_231707/treasure_clip.mp3 --arrangement-mode pro --output runs/quality_push_after_<new_commit>_<timestamp>`
   - Success check:
     - `floor_pass_count >= 1` or best `song_likeness >= 55.0`

## Blocked Tasks
1. **Promote a pro winner artifact automatically**
   - Blocker: latest run (`quality_push_after_payoff_build_combo_20260326_1222`) still has `floor_pass_count=0`.
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
