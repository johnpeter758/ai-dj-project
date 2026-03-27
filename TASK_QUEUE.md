# VocalFusion Task Queue

Last updated: 2026-03-26 22:29 EDT (research automation added + SOTA tracker generated)
Owner: execution operator

## Current Task (active now)
1. **Research-backed adaptive gate rescue (pair2): beat-locked support overlays + transition stability**
   - Why: pair2 still hard-fails with adaptive high-song-likeness candidates (`56.6`, `56.4`) rejected as medley-like.
   - Concrete change target:
     - apply SOTA-guided constraints from `docs/research/MUSIC_SOTA_TRACKER.md`:
       - downbeat/phrase hard alignment for adaptive supports,
       - owner-switch reduction under adaptive shortlist,
       - payoff seam vocal-collision suppression.
   - Files likely touched:
     - `ai_dj.py`
     - `src/core/render/resolver.py`
     - tests in `tests/test_auto_shortlist_fusion.py` and `tests/test_render_stack.py`.

## Next Task (auto-start immediately after current)
1. **Rerun pair2 benchmark after adaptive gate-rescue patch and compare reject->pass conversion**
   - Command:
     - `python ai_dj.py fusion runs/live_fuse_batch_fast_20260325_104746/clips/b.mp3 runs/live_fuse_batch_fast_20260325_104746/clips/c.mp3 --arrangement-mode pro --output runs/quality_push_pair2_bc_after_gate_rescue_<timestamp>`
   - Success check:
     - at least one adaptive candidate reaches `gate=pass` while keeping `song_likeness >= 55.0`.

## Blocked Tasks
1. **Global promotion of support-overlay strategy as default winner path**
   - Blocker: pair2 remains hard-fail (`floor_pass_count=0`) even after adaptive support reservation.
   - Unblock condition: second pair reaches pass+floor under unchanged policy.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
