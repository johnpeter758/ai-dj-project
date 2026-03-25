# Onset / boundary renderer upgrade

Implemented a renderer-only cue-safe transition anchoring pass in `src/core/render/renderer.py`.

## What changed
- Added `_find_cue_safe_head_offset_samples(...)` to detect a stronger delayed attack/onset near the head of an incoming overlapped segment using short-window RMS + onset-strength analysis.
- Added `_cue_safe_transition_anchor(...)` to trim unstable pre-onset material from the front of the rendered segment and zero-pad the tail so the target section duration remains unchanged.
- Applied the anchor step inside `render_resolved_plan(...)` before gain / transition shaping / edge fades, so overlap joins land closer to a stable musical event without changing planner timing contracts.

## Why this helps
- Incoming sections with non-zero `fade_in_sec` no longer have to enter on arbitrary low-energy pre-attack material.
- The renderer now prefers a cue-safe local attack inside the allowed fade window, which should reduce smeared or weak-feeling section joins.

## Tests added
- `test_find_cue_safe_head_offset_samples_detects_delayed_attack_inside_fade_window`
- `test_cue_safe_transition_anchor_trims_pre_attack_head_and_preserves_length`

## Validation
- Focused new tests passed.
- Full render regression set passed:
  - `tests/test_render_stack.py`
  - `tests/test_render_spectral.py`
- Result: `46 passed`.
