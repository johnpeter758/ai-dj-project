# VocalFusion Task Queue

Last updated: 2026-03-27 16:33 EDT (planner-risk wiring into resolver support envelope + pair2 rerun)
Owner: execution operator

## Current Task (active now)
1. **Raise transition quality while preserving pass+floor stability**
   - Why: pair2 remains stable floor-pass, but transition score is still stuck in the `53.6-53.8` band.
   - Latest checkpoint:
     - patch: support decisions now use adaptive `support_policy` (risk + context aware gain/mode) rather than rigid per-label defaults.
     - patch: adaptive dual-support pairing now uses pair scoring (payoff/build preference with hard risk caps, max/mean risk, error sum, section span, parent diversity) instead of blindly taking first two support candidates.
     - patch: removed rigid shortlist early-return that skipped support overlays when swap opportunities were empty.
     - patch: resolver now applies handoff-aware support overlay profiles (gain + fade-in/out shaping) to reduce integrated-support crowding during `arrival_handoff`/`single_owner_handoff` transitions.
     - patch: support work orders now inherit normalized section `transition_mode` instead of support-role tokens, keeping planner intent intact through render.
     - patch: planner now persists support seam signals (`support_transition_risk`, `support_foreground_collision_risk`, `support_transition_viability`) into `PlannedSection` from both planner-generated support recipes and auto-shortlist support overlays.
     - patch: resolver support envelope shaping now prefers explicit planner risk/collision signals for handoff transitions (with gain-based fallback only when risk is absent).
     - regressions:
       - `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_support_policy_adapts_to_transition_risk`
       - `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_adaptive_dual_support_avoids_extreme_risk_payoff_pair`
       - `tests/test_auto_shortlist_fusion.py::test_apply_auto_shortlist_variant_applies_support_overlay_to_section_and_diagnostics`
       - `tests/test_render_stack.py::test_resolve_render_plan_handoff_support_profile_uses_gain_as_risk_proxy`
       - `tests/test_render_stack.py::test_resolve_render_plan_handoff_support_profile_prefers_explicit_planner_risk_signal`
     - validation:
       - `pytest -q tests/test_render_stack.py -k "handoff_support_profile"` → `3 passed`.
       - `pytest -q tests/test_auto_shortlist_fusion.py -k "applies_support_overlay_to_section_and_diagnostics"` → `1 passed`.
       - `pytest -q tests/test_render_stack.py tests/test_core_planner.py tests/test_auto_shortlist_fusion.py tests/test_pro_fusion_quality.py` → `225 passed, 1 skipped`.
     - artifact reruns:
       - `runs/quality_push_pair2_intelligent_support_policy_20260327_111418`
         - policy: `pass+floor`, `promotion_blocked=false`
         - winner metrics: `song_likeness=58.3`, `transition=53.8`, `overall=69.9`.
       - `runs/quality_push_pair2_intelligent_dual_pairing_20260327_121042`
         - policy: `pass+floor`, `promotion_blocked=false`
         - winner metrics: `song_likeness=58.3`, `transition=53.8`, `overall=69.9`, `selection_score=73.607` (selection-score lift vs prior 73.571).
       - `runs/quality_push_pair2_handoff_gain_adaptive_20260327_1415`
         - patch: resolver handoff support profile now uses support-gain risk buckets (higher-risk supports get extra duck + edge softening).
         - policy: `pass+floor`, `promotion_blocked=false`, winner still adaptive dual-support (`support_01_payoff_build_A`).
         - winner metrics: `song_likeness=58.5`, `transition=53.6`, `overall=70.0`, `selection_score=73.657`.
       - `runs/quality_push_pair2_planner_risk_wiring_20260327_1620`
         - patch: explicit planner support risk/collision is now wired into resolver support envelope shaping.
         - policy: `pass+floor`, `promotion_blocked=false`, winner still adaptive dual-support (`support_01_payoff_build_A`).
         - winner metrics: `song_likeness=58.5`, `transition=53.7`, `overall=70.0`, `selection_score=73.685`.
   - Focus:
     - push transition above 53.8 by combining shortlist risk policy with render-time support envelope shaping,
     - keep anti-medley penalties and hard-floor gate untouched.
   - Files likely touched:
     - `src/core/render/renderer.py`
     - `src/core/render/resolver.py`
     - `tests/test_render_stack.py`

## Next Task (auto-start immediately after current)
1. **Push transition over 53.8 via section-specific support envelope offsets (build vs payoff) and rerun pair2**
   - Why: explicit planner-risk wiring lifted selection score and recovered transition to 53.7, but the bottleneck remains below the 53.8 target.
   - Guardrails:
     - preserve winner policy `pass+floor`,
     - do not regress song_likeness below 58.0,
     - target transition `>53.8` while keeping integrated support as winning path.

## Blocked Tasks
1. **Wide multi-pair ship-confidence sweep**
   - Blocker: transition bottleneck still plateaued at `53.8`.
   - Unblock condition: transition-focused render patch lands and pair2 remains stable under floor gates.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
