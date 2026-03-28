# VocalFusion Task Queue

Last updated: 2026-03-28 00:24 EDT (planner viability calibration pass + pair2 rerun)
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
       - `./.venv/bin/python -m pytest -q tests/test_pro_fusion_quality.py tests/test_render_stack.py tests/test_core_planner.py` (nightly sanity rerun) → `205 passed, 1 skipped`.
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
      - `runs/quality_push_pair2_transition_intelligence_20260327_1815`
        - experiment: renderer-side dynamic transition-intelligence shaping (spectral-density-driven entry/tail filter/notch modulation).
        - outcome: no transition lift (`transition=53.6`), selection-score regression (`73.611` vs `73.685`), floor stability held (`pass+floor`, `song_likeness=58.4`, `overall=70.0`).
        - action: experiment rolled back (no code promoted) to preserve best-known branch state.
      - `runs/quality_push_pair2_label_handoff_offsets_20260327_2015`
        - experiment: resolver-side section-specific handoff support-envelope offsets (build vs payoff gain/fade shaping).
        - outcome: winner path and headline metrics unchanged vs best-known checkpoint (`pass+floor`, adaptive dual-support, `song_likeness=58.5`, `transition=53.7`, `overall=70.0`, `selection_score=73.685`).
        - action: experiment rolled back (no code promoted) to avoid adding complexity without measurable lift.
      - `runs/quality_push_pair2_payoff_viability_bucket_20260327_2215`
        - patch: resolver support profile now includes payoff-only low-viability handoff bucket (`risk+collision` high and `transition_viability` low) for extra duck/fade cleanup.
        - outcome: floor stability held (`pass+floor`) but winner metrics were unchanged (`song_likeness=58.5`, `transition=53.7`, `overall=70.0`, `selection_score=73.685`).
        - action: keep patch + tests; next step is planner-side viability calibration so payoff-specific bucket activates only on truly crowded handoffs in pair2.
      - `runs/quality_push_pair2_viability_calibration_20260328_0015`
        - patch: support policy calibration now records planner-aligned `transition_error`, calibrated `transition_viability` (health-style), and `foreground_collision_risk` for shortlist-generated support overlays.
        - outcome: floor stability held (`pass+floor`) and winner remained adaptive dual-support (`support_01_payoff_build_A`); winner support policy now shows crowded payoff/build handoffs (`collision≈0.61/0.60`, `transition_viability≈0.389/0.400`).
        - metrics: `song_likeness=58.5`, `transition=53.7`, `overall=70.1`, `selection_score=73.729` (selection/overall lift vs prior best `73.685`/`70.0`, transition unchanged).
        - action: keep patch; next step is render-side handoff envelope threshold tuning to convert calibrated planner signals into measurable transition lift.
   - Focus:
     - push transition above 53.8 by combining shortlist risk policy with render-time support envelope shaping,
     - keep anti-medley penalties and hard-floor gate untouched.
   - Files likely touched:
     - `src/core/render/renderer.py`
     - `src/core/render/resolver.py`
     - `tests/test_render_stack.py`

## Next Task (auto-start immediately after current)
1. **Push transition over 53.8 via render-side handoff envelope tuning using calibrated planner collision/viability signals**
   - Why: planner-side support policy calibration now feeds richer crowding/viability signals into winning pair2 support overlays, but transition remains at `53.7`.
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
