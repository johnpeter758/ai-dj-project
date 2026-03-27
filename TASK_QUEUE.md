# VocalFusion Task Queue

Last updated: 2026-03-26 22:18 EDT (adaptive support fallback shipped + pair2 rerun)
Owner: execution operator

## Current Task (active now)
1. **Lift adaptive support-overlay quality from reject -> pass on pair2**
   - Why: fallback support synthesis now emits adaptive support variants, but latest run (`quality_push_pair2_bc_after_transition_patch_20260326_2215`) still hard-fails with adaptive support candidate stuck at `song_likeness=47.4`, `transition=53.0`, `gate=reject`.
   - Concrete change target:
     - tune support-overlay application for adaptive mode (owner/gain/mode defaults and seam behavior) so support candidates improve transition/coherence instead of collapsing to baseline-like scores,
     - rerun pair2 to validate adaptive support candidate quality trend.
   - Files likely touched:
     - `ai_dj.py` support variant construction/application
     - `src/core/render/resolver.py` / `src/core/render/renderer.py` support handling
     - tests in `tests/test_auto_shortlist_fusion.py` and `tests/test_render_stack.py`.

## Next Task (auto-start immediately after current)
1. **Rerun pair2 benchmark after adaptive support-tuning patch and compare adaptive gate_status + floor_pass_count**
   - Command:
     - `python ai_dj.py fusion runs/live_fuse_batch_fast_20260325_104746/clips/b.mp3 runs/live_fuse_batch_fast_20260325_104746/clips/c.mp3 --arrangement-mode pro --output runs/quality_push_pair2_bc_after_support_tuning_<timestamp>`
   - Success check:
     - adaptive support (or other adaptive integrated candidate) reaches `gate=pass` with `song_likeness >= 55.0`.

## Blocked Tasks
1. **Global promotion of support-overlay strategy as default winner path**
   - Blocker: pair2 remains hard-fail (`floor_pass_count=0`); adaptive support fallback now emits but still rejects.
   - Unblock condition: second pair reaches pass+floor under same policy.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
