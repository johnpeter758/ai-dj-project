# VocalFusion Task Queue

Last updated: 2026-03-26 23:33 EDT (adaptive dual-support variant patch validated)
Owner: execution operator

## Current Task (active now)
1. **Rerun pair2 benchmark after adaptive dual-support variant patch**
   - Why: dual-support candidate generation is now implemented/validated in shortlist config logic, but real-run impact on floor crossing is not measured yet.
   - Command:
     - `python ai_dj.py fusion runs/live_fuse_batch_fast_20260325_104746/clips/b.mp3 runs/live_fuse_batch_fast_20260325_104746/clips/c.mp3 --arrangement-mode pro --output runs/quality_push_pair2_bc_after_dual_support_<timestamp>`
   - Success check:
     - adaptive pass candidate reaches `song_likeness >= 55.0` while keeping gate `pass`.

## Next Task (auto-start immediately after current)
1. **Tune adaptive support intensity/placement only if rerun remains below floor**
   - Focus:
     - preserve reduced medley risk from counterparent supports,
     - raise song-likeness via support gain/section targeting tweaks without introducing vocal crowding.
   - Files likely touched:
     - `ai_dj.py`
     - `tests/test_auto_shortlist_fusion.py`

## Blocked Tasks
1. **Global promotion of support-overlay strategy as default winner path**
   - Blocker: pair2 still has not demonstrated a second stable `pass+floor` result under unchanged policy.
   - Unblock condition: pair2 rerun with dual-support path reaches `song_likeness >= 55.0` and floor-pass count > 0.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
