# VocalFusion Task Queue

Last updated: 2026-03-26 21:06 EDT (support-overlay relaxed run produced first floor-pass winner)
Owner: execution operator

## Current Task (active now)
1. **Promote and lock the first floor-pass winner as revamp checkpoint**
   - Why: run `quality_push_after_support_overlay_relaxed_20260326_210633` produced `winner_policy=pass+floor` with baseline `support_01_payoff_B` (`song_likeness=56.8`).
   - Concrete change target:
     - copy winner artifacts as milestone checkpoint,
     - add explicit regression coverage that baseline support-overlay candidate remains available when core donor swaps are unavailable,
     - preserve this path while continuing quality upgrades.
   - Files likely touched:
     - `docs/`
     - `tests/test_auto_shortlist_fusion.py`
     - optional `ai_dj.py` guardrail constants.

## Next Task (auto-start immediately after current)
1. **Run second benchmark (different pair) to check anti-medley + support-overlay generalization**
   - Command:
     - `python ai_dj.py fusion <pair2_track_a> <pair2_track_b> --arrangement-mode pro --output runs/quality_push_after_support_overlay_pair2_<timestamp>`
   - Success check:
     - at least one pass candidate with improved song-likeness trend and no intro-led back-and-forth winner.

## Blocked Tasks
1. **Default-enable support-overlay heuristics globally without pair validation**
   - Blocker: only one known pair has passed floor so far.
   - Unblock condition: second benchmark pair confirms non-regression.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
