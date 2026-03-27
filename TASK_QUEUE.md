# VocalFusion Task Queue

Last updated: 2026-03-27 11:06 EDT (support-overlay promoted to primary pro-mode candidate path + pair2 rerun)
Owner: execution operator

## Current Task (active now)
1. **Raise transition quality now that support overlays are first-class in both pro candidate sets**
   - Why: pair2 is floor-pass stable, but transition remains the bottleneck (`53.8`) despite repeated render-shaping gains.
   - Latest checkpoint:
     - patch: `ai_dj.py::_build_auto_shortlist_variant_configs` now reserves a support-overlay slot for both `arrangement_mode=adaptive` and `arrangement_mode=baseline` whenever support candidates exist (`batch_size >= 3`).
     - effect: support overlays are no longer baseline fallback-only; both mode candidate sets now keep an integrated-support path.
     - regression: `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_baseline_keeps_support_variant_even_with_core_donor_swaps` added.
     - validation:
       - `pytest -q tests/test_auto_shortlist_fusion.py -k "variant_configs"` → `15 passed, 4 deselected`.
       - `pytest -q tests/test_auto_shortlist_fusion.py tests/test_pro_fusion_quality.py tests/test_core_planner.py tests/test_render_stack.py` → `219 passed, 1 skipped`.
     - artifact rerun: `runs/quality_push_pair2_support_primary_baseline_20260327_110021`
       - policy: `pass+floor`, `promotion_blocked=false`
       - winner: adaptive `dual_section_support` (`support_01_payoff_build_A`)
       - winner metrics: `song_likeness=58.2`, `groove=64.3`, `structure=92.2`, `transition=53.8`, `overall=69.9`, `selection_score=73.571`
       - baseline set now includes integrated support variant (`support_01_payoff_B`, gate=pass).
   - Focus:
     - transition/mix polish on support-entry + release envelopes,
     - preserve floor-pass and anti-medley guardrails while pushing transition above current plateau.
   - Files likely touched:
     - `src/core/render/renderer.py`
     - `src/core/render/resolver.py`
     - `tests/test_render_stack.py`

## Next Task (auto-start immediately after current)
1. **Run wide multi-pair regression/listening sweep once transition lift patch lands**
   - Why: support strategy is now represented in both mode sets; next risk is overfitting pair2.
   - Guardrails:
     - keep hard-floor promotion gating unchanged,
     - capture per-pair deltas (`song_likeness`, `transition`, `overall`, winner policy),
     - block promotion if regressions reintroduce medley behavior.

## Blocked Tasks
1. **Ship-level confidence pass across additional pair catalog**
   - Blocker: transition bottleneck still plateaued at `53.8` on pair2.
   - Unblock condition: transition-focused render patch lands with tests and pair2 stays `pass+floor`.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
