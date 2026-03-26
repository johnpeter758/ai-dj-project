# VocalFusion Task Queue

Last updated: 2026-03-26 18:23 EDT (post baseline donor-combo guard + pro rerun)
Owner: execution operator

## Current Task (active now)
1. **Diagnose why baseline still emits only parent-A combo candidates under safe combo gating**
   - Why: shipped baseline donor-combo filter in `_build_auto_shortlist_variant_configs`, but rerun `quality_push_after_baseline_donor_combo_20260326_1815` still produced baseline `combo_01_verse_payoff` with swaps `['A', 'A']` and `parent_balance=0.0`.
   - Concrete change target:
     - instrument/adjust combo construction so baseline can surface at least one safe cross-parent core donor combo candidate when donor opportunities exist but are currently filtered out by per-section representative or combo safety gating.
   - Files likely touched:
     - `ai_dj.py`
     - `tests/test_auto_shortlist_fusion.py`

## Next Task (auto-start immediately after current)
1. **Run one pro benchmark and verify floor progress after combo-candidate diagnosis patch**
   - Command:
     - `python ai_dj.py fusion runs/quality_push_after_eaa9ee0_20260325_231707/relax_clip.mp3 runs/quality_push_after_eaa9ee0_20260325_231707/treasure_clip.mp3 --arrangement-mode pro --output runs/quality_push_after_<new_commit>_<timestamp>`
   - Success check:
     - baseline candidate set includes at least one donor-bearing combo or single with `parent_balance > 0`, and `floor_pass_count >= 1` or best `song_likeness >= 55.0`

## Blocked Tasks
1. **Promote a pro winner artifact automatically**
   - Blocker: latest run (`quality_push_after_core_label_norm_20260326_1620`) still has no floor-pass candidate (`song_likeness < 55.0`).
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
