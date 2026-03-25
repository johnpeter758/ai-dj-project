# Planner Quality Wave 06 — late-form outro continuity

Implemented a targeted planner/material-selection upgrade for post-payoff continuity so outro selection better preserves the one-song illusion.

## What changed
- Added explicit **outro release / closing-lane metrics** in `src/core/planner/arrangement.py` for outros that follow a `payoff` or `bridge`.
- New scoring now measures and penalizes:
  - insufficient energy release after payoff
  - weak outro/closing identity
  - weak closing-lane behavior
  - **still-climaxing risk** (fake outro that keeps sounding like the song is still trying to peak)
  - too-early late-form position
- Added a **hard outro shortlist pool** so fake still-climaxing outro windows get filtered out when a safer decaying/closing lane exists.
- Added a **late-outro release guard** in `_choose_with_major_section_balance_guard(...)` so if the top-ranked outro still behaves like post-payoff climax material, the planner can switch to a later/surer closing option within a bounded error delta.
- Exposed the new outro diagnostics in section score breakdowns.

## Tests added
- metric-level test that flags a fake still-climaxing outro after payoff
- hard-pool test that keeps a real closing lane and drops a fake continuing-climax lane
- guard test that switches from a fake top outro to a safer release option

## Validation
- Targeted outro tests: passed
- Full planner test file: `117 passed, 1 skipped`

## Musical intent
This specifically strengthens the late-form handoff after the payoff: if one candidate still sounds like “the chorus is still going” but another candidate actually decays/closes, the planner now prefers the closing lane rather than preserving false climax energy into the outro.
