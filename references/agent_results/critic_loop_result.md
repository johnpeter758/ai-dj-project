# Critic loop implementation result

## What I changed

Implemented the first explicit keep/reject layer for the bounded critic-driven auto-improve loop in `scripts/closed_loop_listener_runner.py`.

### Core behavior added
- Added `_candidate_keep_decision(...)` to turn each iteration into an explicit `keep` or `reject` outcome.
- Keep/reject now uses:
  - meaningful progress vs prior best
  - listener-agent decision (`reject` / `borderline` / `survivor`)
  - quality-gate status / reason
- Each iteration now records `candidate_keep_decision` in `closed_loop_report.json`.
- The loop-level report now records aggregate `candidate_decisions` counts (`kept_iterations`, `rejected_iterations`).
- `best_iteration` and `loop_summary` now preserve the keep/reject decision so downstream automation can tell what was accepted.

### Tests added/updated
Updated `tests/test_closed_loop_listener_runner.py` to cover:
- best iteration carrying a `keep` decision
- loop summary exposing aggregate keep/reject counts
- explicit scenario where iteration 1 is kept and iteration 2 is rejected

## Docs updated
- Updated `README.md` to describe the closed loop as explicitly keeping or rejecting candidates.
- Documented that `listener_assessment.json` and `closed_loop_report.json` now expose the candidate decision layer.

## Validation
Ran:

```bash
./.venv/bin/python -m pytest -q tests/test_closed_loop_listener_runner.py tests/test_listen_feedback_loop.py
```

Result:
- `38 passed`

## Notes for main agent
- I stayed within `scripts/`, `tests/`, and lightweight docs only.
- I did **not** modify planner/render core files.
- The repo already had strong listener/feedback scaffolding; this change makes the critic loop materially more explicit and automation-friendly without changing core fusion behavior.
