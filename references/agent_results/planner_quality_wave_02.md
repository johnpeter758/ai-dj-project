# Planner quality wave 02 — late-payoff legitimacy guard

## What I changed
- Added a new **late-payoff legitimacy guard** in `src/core/planner/arrangement.py` inside `_choose_with_major_section_balance_guard(...)`.
- Purpose: if the top-ranked `payoff` choice still looks like an **earlier / weaker pseudo-payoff** (`final_payoff_delivery` still high, weaker `payoff_hit`, weaker `payoff_sustained_conviction`, or meaningful early-position/start gaps), the chooser now scans later-ranked payoff options and **switches to a safer late sustained alternate** when one exists.
- The alternate must still be safe on the baseline path: low stretch risk, acceptable groove confidence, no large seam/transition regression, and a bounded planner-error delta.

## Why this is high leverage
- The planner already had payoff scoring and shortlist pressure, but the final chooser could still keep a locally top-ranked pseudo-payoff even when a more legitimate late sustained climax was sitting just behind it.
- This change is deliberately **material-changing**, not another soft weight tweak: it replaces weak early payoff landings when a convincing safe alternate exists.

## Tests added
- `test_choose_with_major_section_balance_guard_replaces_weak_early_payoff_when_safe_late_sustained_alternate_exists`
- `test_choose_with_major_section_balance_guard_keeps_top_payoff_when_late_alternate_is_not_safe_enough`

## Validation
- `python -m pytest -q tests/test_core_planner.py` -> `110 passed, 1 skipped`
- `python -m pytest -q tests/test_render_stack.py tests/test_listen_cli.py tests/test_compare_listen_benchmark.py` -> `90 passed`

## Practical effect
- Baseline-path payoff selection is now harder to fool with flashy but premature payoff-like windows.
- When a real late sustained payoff is available and safe, the chooser will now promote it instead of preserving a weaker earlier landing.
