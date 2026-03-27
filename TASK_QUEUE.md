# VocalFusion Task Queue

Last updated: 2026-03-26 22:38 EDT (adaptive counterparent support patch + pair2 rerun)
Owner: execution operator

## Current Task (active now)
1. **Lift adaptive pass candidate from 53.7 -> >=55.0 song_likeness on pair2**
   - Why: counterparent-support patch successfully converted adaptive `reject -> pass` (`support_01_payoff_A` now pass, `song_likeness=53.7`, `support_layer_section_ratio=0.2`), but still below hard floor.
   - Concrete change target:
     - increase adaptive integrated support strength without reintroducing medley risk:
       - add dual-support adaptive variant (`build+payoff` support overlays) when both donor-led,
       - keep support gains conservative to avoid vocal crowding.
   - Files likely touched:
     - `ai_dj.py`
     - `tests/test_auto_shortlist_fusion.py`

## Next Task (auto-start immediately after current)
1. **Rerun pair2 benchmark after dual-support adaptive variant patch**
   - Command:
     - `python ai_dj.py fusion runs/live_fuse_batch_fast_20260325_104746/clips/b.mp3 runs/live_fuse_batch_fast_20260325_104746/clips/c.mp3 --arrangement-mode pro --output runs/quality_push_pair2_bc_after_dual_support_<timestamp>`
   - Success check:
     - adaptive pass candidate reaches `song_likeness >= 55.0` while keeping gate `pass`.

## Blocked Tasks
1. **Global promotion of support-overlay strategy as default winner path**
   - Blocker: pair2 rerun `quality_push_pair2_bc_counterparent_support_20260326_223843` still hard-fails (`floor_pass_count=0`) despite adaptive reject->pass conversion.
   - Unblock condition: second pair reaches pass+floor under unchanged policy.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
