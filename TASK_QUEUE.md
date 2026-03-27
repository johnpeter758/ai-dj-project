# VocalFusion Task Queue

Last updated: 2026-03-26 20:23 EDT (post donor-path combo fallback + pro rerun)
Owner: execution operator

## Current Task (active now)
1. **Raise donor identity without sacrificing song-likeness via intro-heavy fallback combos**
   - Why: rerun `quality_push_after_baseline_donor_fallback_20260326_202145` now surfaces baseline donor-bearing combo (`combo_01_intro_payoff`, `parent_balance=0.333`), confirming combo construction no longer collapses to all-`A`. But quality regressed (`song_likeness=52.7`, `overall=66.9`) and hard floor still fails.
   - Concrete change target:
     - keep donor-bearing fallback availability, but re-rank/guard baseline combo construction so intro-donor fallback is used only when it does not displace stronger core-structure candidates (or pair intro donor with less destructive companion section).
   - Files likely touched:
     - `ai_dj.py`
     - `tests/test_auto_shortlist_fusion.py`

## Next Task (auto-start immediately after current)
1. **Run one pro benchmark and verify floor progress after fallback-reranking patch**
   - Command:
     - `python ai_dj.py fusion runs/quality_push_after_eaa9ee0_20260325_231707/relax_clip.mp3 runs/quality_push_after_eaa9ee0_20260325_231707/treasure_clip.mp3 --arrangement-mode pro --output runs/quality_push_after_<new_commit>_<timestamp>`
   - Success check:
     - baseline set still includes at least one donor-bearing candidate (`parent_balance > 0`) while best `song_likeness` recovers to >= `54.2` and trends toward floor (`55.0`).

## Blocked Tasks
1. **Promote a pro winner artifact automatically**
   - Blocker: latest run (`quality_push_after_core_label_norm_20260326_1620`) still has no floor-pass candidate (`song_likeness < 55.0`).
   - Unblock condition: next run produces at least one floor-pass candidate.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
