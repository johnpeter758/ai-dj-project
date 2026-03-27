# VocalFusion Task Queue

Last updated: 2026-03-27 04:27 EDT (support release-tail shaping patch + render stack regression pass)
Owner: execution operator

## Current Task (active now)
1. **Continue adaptive dual-support render quality stabilization (post entry-shape patch)**
   - Why: pair2 remains `pass+floor`, and transition quality ticked up (`53.6 -> 53.8`) with stable song-likeness (`58.2`), but mix/transition are still below target polish.
   - Latest checkpoint:
     - patch: `src/core/render/renderer.py::_apply_support_entry_shape` now includes release-tail shaping for `section_support` (section-aware tail duck + filtered support tail low-pass contour).
     - regression: `tests/test_render_stack.py` full pass (`60 passed`) including new support-tail coverage.
     - last artifact run remains `runs/quality_push_pair2_support_entry_shape_20260327_0215`; next pass should re-render pair2 to validate audible tail cleanup.
   - Focus:
     - section-label-aware support entry/release shaping for build/payoff overlays,
     - raise transition clarity and mix sanity without reducing integrated two-parent identity.
   - Files likely touched:
     - `src/core/render/renderer.py`
     - `src/core/render/resolver.py`
     - `tests/test_render_stack.py`

## Next Task (auto-start immediately after current)
1. **Promote support-overlay strategy from fallback to primary pro-mode candidate path**
   - Why: floor-pass behavior is now reproducible across consecutive pair2 reruns.
   - Guardrails:
     - keep anti-medley penalties active,
     - require at least one integrated support candidate in both adaptive and baseline candidate sets,
     - preserve hard-floor gating.

## Blocked Tasks
1. **Wide regression sweep + artifact listening review across additional pairs**
   - Blocker: render-tuning pass not yet complete after pair2 dual-support floor-pass.
   - Unblock condition: transition/mix tuning patch lands with tests and no regressions on pair2.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
