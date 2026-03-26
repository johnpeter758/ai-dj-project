# VocalFusion Task Queue

Last updated: 2026-03-26 14:20 EDT (post core-donor-single benchmark)
Owner: execution operator

## Current Task (active now)
1. **Force donor-bearing combo construction when baseline shortlist has no core donor single**
   - Why: after shipping core-donor-single guarantee and rerunning pro, hard floor still failed (`quality_push_after_core_donor_single_20260326_1415`; best `song_likeness=54.2`, `transition=61.2`, `overall=71.1`). Baseline mode still emitted backbone-only variants (`swap_01_payoff_A`, `combo_01_verse_payoff`), so two-parent identity remains underpowered where winner-quality is currently strongest.
   - Concrete change:
     - in `_build_auto_shortlist_variant_configs`, when no eligible core donor single exists, bias the reserved combo slot toward at least one donor-bearing core section (verse/build/payoff/bridge) instead of allowing all-backbone combos.
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
   - Blocker: latest run (`quality_push_after_core_donor_single_20260326_1415`) still has no floor-pass candidate (`song_likeness < 55.0`).
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
