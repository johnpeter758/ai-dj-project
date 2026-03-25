# Planner Quality Wave 04 — Opening-Lane Authenticity Upgrade

## What I changed
- Implemented a focused opening-authenticity upgrade in `src/core/planner/arrangement.py`.
- Tightened `_choose_with_major_section_balance_guard(...)` for `intro` selection so a non-preferred top intro that is only **locally plausible** but reads like a pseudo-opening will now lose to a close-scoring preferred-parent opening when that preferred lane is materially safer.
- The guard now explicitly compares:
  - `fake_intro_risk`
  - `opening_followthrough`
  - `opening_followthrough_identity_gap`
  - `opening_followthrough_lane_gap`
  - `section_identity`
  - `shape_intro_hotspot`
- If the chosen intro looks pseudo-opening-like and the preferred-parent alternate offers a safer authentic intro→verse backbone, the planner switches upstream instead of trusting the raw local rank.

## Why this matters
- This directly targets the baseline/decent-first failure mode where the child opens with donor or hotspot material that can pass local scoring but does **not** preserve the illusion of one real song.
- The new rule makes the planner favor the safer backbone opening lane when it actually exists, which is the highest-leverage way to improve intro→verse readability without touching render code.

## Tests added
- Added regression in `tests/test_core_planner.py`:
  - `test_choose_with_major_section_balance_guard_switches_to_preferred_intro_when_top_opening_is_only_locally_plausible_pseudo_opening`
- Existing intro/verse opening-lane guard tests continue to pass.

## Validation
- `.../python -m pytest -q tests/test_core_planner.py -k 'prefers_readable_preferred_parent_intro_when_close or locally_plausible_pseudo_opening or donor_intro_would_hand_off_into_late_pseudo_verse'` → `3 passed`
- `.../python -m pytest -q tests/test_core_planner.py` → `113 passed, 1 skipped`

## Durable result
- Baseline-path opening selection is now less willing to accept a pseudo-opening just because it is locally competitive.
- When a readable safer backbone opening lane exists, the planner is more likely to start like one song instead of a stitched handoff waiting to fail at `intro -> verse`.
