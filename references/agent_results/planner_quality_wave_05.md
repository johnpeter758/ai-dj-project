# Planner Quality Wave 05

Implemented a targeted hard-bridge legitimacy upgrade in `src/core/planner/arrangement.py`.

## What changed
- Extended `_bridge_reset_candidate_metrics(...)` with a new `bridge_release_gap` signal.
- This new gap explicitly measures whether a post-payoff bridge actually **de-emphasizes / releases** instead of staying hot and plateaued.
- The release metric favors lower normalized energy, less end-focus, more headroom, less plateau stability, and a downward slope.
- Tightened `_hard_bridge_reset_candidate_pool(...)` so late bridge candidates now need to satisfy:
  - convincing energy reset
  - convincing bridge identity
  - convincing release / de-emphasis
  - sufficiently late placement
- Added the new metric to planner diagnostics / score breakdowns as `bridge_release_gap`.

## Why this matters
Previously, the hard bridge filter could still admit pseudo-bridges that technically dropped a bit from the prior payoff but remained too sustained / plateau-like to function as a real reset. That produces children that feel stuck in the climax instead of earning the re-launch.

This change keeps the planner on a more musical path: if a safe alternate real release bridge exists, the fake plateau bridge gets filtered out.

## Tests
Updated `tests/test_core_planner.py` with a new regression:
- `test_bridge_selection_filters_plateaued_pseudo_reset_when_real_release_bridge_exists`

Also re-ran focused existing bridge-reset tests:
- `test_bridge_selection_prefers_real_reset_after_hot_payoff_over_staying_stuck_in_plateau`
- `test_bridge_selection_hard_shortlists_real_reset_when_clear_post_payoff_reset_exists`

Focused pytest result:
- `3 passed`
