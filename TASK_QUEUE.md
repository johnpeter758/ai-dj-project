# VocalFusion Task Queue

Last updated: 2026-03-26 12:05 EDT (post f9ccdde pro benchmark)
Owner: execution operator

## Current Task (active now)
1. **Improve fusion identity by forcing one core donor (non-intro/outro) into 3-slot pro variants**
   - Why: latest run recovered best metrics but still blocks at `song_likeness 54.3` with `integrated_two_parent_section_ratio=0.0` and baseline parent balance `0.0`.
   - Concrete change:
     - in shortlist generation, require one donor alternate from core sections (`verse/build/payoff/bridge`) for combo/single selection before fallback to backbone-only swaps.
   - Files likely touched:
     - `ai_dj.py` (`_build_auto_shortlist_variant_configs` ranking constraints)
     - `tests/test_auto_shortlist_fusion.py`

## Next Task (auto-start immediately after current)
1. **Run one pro benchmark and verify donor-enforced variants improve song_likeness to >=55.0**
   - Command:
     - `python ai_dj.py fusion runs/quality_push_after_eaa9ee0_20260325_231707/relax_clip.mp3 runs/quality_push_after_eaa9ee0_20260325_231707/treasure_clip.mp3 --arrangement-mode pro --output runs/quality_push_after_<new_commit>_<timestamp>`
   - Success check:
     - `floor_pass_count >= 1` or best `song_likeness >= 55.0`

## Blocked Tasks
1. **Promote a pro winner artifact automatically**
   - Blocker: latest run (`quality_push_after_f9ccdde_20260326_122026`) still has `floor_pass_count=0` with best `song_likeness=54.3`.
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
