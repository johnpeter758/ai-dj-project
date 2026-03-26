# VocalFusion Task Queue

Last updated: 2026-03-26 16:25 EDT (post core-label normalization + pro rerun)
Owner: execution operator

## Current Task (active now)
1. **Force baseline combo to include at least one cross-parent core donor when pass candidates are all parent_balance=0.0**
   - Why: pro rerun `quality_push_after_core_label_norm_20260326_1620` still hard-failed (`floor_pass_count=0`, best `song_likeness=54.2`). Baseline winner path remains backbone-only (`combo_01_verse_payoff`, both swaps parent `A`) despite adaptive donor swaps existing.
   - Concrete change:
     - in `_build_auto_shortlist_variant_configs`, add baseline combo-construction guard so when no selected single contributes donor identity, the reserved combo slot prioritizes/filters to include at least one core donor swap (`verse/build/payoff/bridge`) if any safe candidate exists.
   - Files likely touched:
     - `ai_dj.py`
     - `tests/test_auto_shortlist_fusion.py`

## Next Task (auto-start immediately after current)
1. **Run one pro benchmark and verify floor progress**
   - Command:
     - `python ai_dj.py fusion runs/quality_push_after_eaa9ee0_20260325_231707/relax_clip.mp3 runs/quality_push_after_eaa9ee0_20260325_231707/treasure_clip.mp3 --arrangement-mode pro --output runs/quality_push_after_<new_commit>_<timestamp>`
   - Success check:
     - `floor_pass_count >= 1` or best `song_likeness >= 55.0`

## Blocked Tasks
1. **Promote a pro winner artifact automatically**
   - Blocker: latest run (`quality_push_after_core_label_norm_20260326_1620`) still has no floor-pass candidate (`song_likeness < 55.0`).
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
