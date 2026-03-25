# Planner Quality Wave 03

Implemented a targeted baseline-path material-selection upgrade focused on donor feature-block legitimacy.

## What changed
- Added a new baseline-only donor mini-arc validator/resolver in `src/core/planner/arrangement.py`.
- The baseline planner now inspects the chosen donor `build` + `payoff` pair and scores whether they form a believable contiguous donor mini-arc.
- The legitimacy check looks at:
  - timeline contiguity / rewind risk
  - excessive overlap or dead air between build and payoff
  - energy lift from build into payoff
  - payoff handoff safety (`seam_risk`, transition penalty)
  - weak payoff-delivery / payoff-hit / sustained-conviction signals
- If the initially locked donor pair is weak, the planner now:
  1. searches the donor shortlist for a safer legitimate donor build/payoff pair, or
  2. falls back to backbone-owned feature sections when no convincing donor mini-arc exists.
- Added structured diagnostics under `planning_diagnostics.baseline_donor_mini_arc` so the baseline path exposes whether the initial pair was accepted, replaced, or downgraded to backbone-only.

## Tests added / updated
- Added direct coverage for mini-arc illegitimacy metrics on a rewound, underpowered donor payoff.
- Added direct coverage for baseline fallback behavior when no legitimate donor pair exists.
- Full `tests/test_core_planner.py` passes after the change.

## Validation
Ran:

```bash
./.venv/bin/python -m pytest -q tests/test_core_planner.py
```

Result:
- `112 passed, 1 skipped`

## Practical effect
This makes the conservative baseline mode less likely to force a fake donor feature block just because the parent pair passes hard tempo/key admissibility. The planner now requires the donor build and payoff to behave like a real contiguous donor mini-arc; otherwise it prefers a safer donor pair or collapses back toward backbone-only structure instead of preserving a musically weak donor cameo.
