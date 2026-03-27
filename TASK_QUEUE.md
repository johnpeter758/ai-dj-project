# VocalFusion Task Queue

Last updated: 2026-03-27 12:14 EDT (intelligent dual-support pairing + adaptive support policy)
Owner: execution operator

## Current Task (active now)
1. **Raise transition quality while preserving pass+floor stability**
   - Why: pair2 remains stable floor-pass, but transition score is still plateaued at `53.8`.
   - Latest checkpoint:
     - patch: support decisions now use adaptive `support_policy` (risk + context aware gain/mode) rather than rigid per-label defaults.
     - patch: adaptive dual-support pairing now uses pair scoring (payoff/build preference with hard risk caps, max/mean risk, error sum, section span, parent diversity) instead of blindly taking first two support candidates.
     - patch: removed rigid shortlist early-return that skipped support overlays when swap opportunities were empty.
     - regressions:
       - `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_support_policy_adapts_to_transition_risk`
       - `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_adaptive_dual_support_avoids_extreme_risk_payoff_pair`
     - validation:
       - `pytest -q tests/test_auto_shortlist_fusion.py -k "variant_configs or support_policy_adapts or avoids_extreme_risk_payoff_pair"` → `17 passed, 4 deselected`.
       - `pytest -q tests/test_auto_shortlist_fusion.py tests/test_pro_fusion_quality.py tests/test_core_planner.py tests/test_render_stack.py` → `221 passed, 1 skipped`.
     - artifact reruns:
       - `runs/quality_push_pair2_intelligent_support_policy_20260327_111418`
         - policy: `pass+floor`, `promotion_blocked=false`
         - winner metrics: `song_likeness=58.3`, `transition=53.8`, `overall=69.9`.
       - `runs/quality_push_pair2_intelligent_dual_pairing_20260327_121042`
         - policy: `pass+floor`, `promotion_blocked=false`
         - winner metrics: `song_likeness=58.3`, `transition=53.8`, `overall=69.9`, `selection_score=73.607` (selection-score lift vs prior 73.571).
   - Focus:
     - push transition above 53.8 by combining shortlist risk policy with render-time support envelope shaping,
     - keep anti-medley penalties and hard-floor gate untouched.
   - Files likely touched:
     - `src/core/render/renderer.py`
     - `src/core/render/resolver.py`
     - `tests/test_render_stack.py`

## Next Task (auto-start immediately after current)
1. **Transition-lift render patch (support envelope timing + handoff de-clutter) and pair2 rerun**
   - Why: shortlist intelligence improved selection confidence; audible transition remains bottleneck.
   - Guardrails:
     - preserve winner policy `pass+floor`,
     - do not regress song_likeness below 58.0,
     - keep integrated support as winning path.

## Blocked Tasks
1. **Wide multi-pair ship-confidence sweep**
   - Blocker: transition bottleneck still plateaued at `53.8`.
   - Unblock condition: transition-focused render patch lands and pair2 remains stable under floor gates.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
