# VocalFusion Task Queue

Last updated: 2026-03-28 10:15 EDT (planner ownership-chain combo priority + pair2 rerun)
Owner: execution operator

## Current Task (active now)
1. **Break transition plateau via structure/planner levers while preserving pass+floor stability**
   - Why: resolver+renderer crowding tuning held floor-pass and selection stability, but pair2 transition remains stuck in the `53.6-53.8` band.
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
      - `runs/quality_push_pair2_handoff_crowding_tuning_20260328_0216`
        - patch: resolver support handoff shaping now computes `crowding_pressure = 0.45*risk + 0.35*collision + 0.20*(1-viability)` and applies extra duck/fade tightening when pressure is elevated, including a stronger build-specific bucket for very crowded handoffs.
        - regressions: added `tests/test_render_stack.py::test_resolve_render_plan_handoff_support_profile_tightens_high_crowding_build_handoffs`.
        - validation: `pytest -q tests/test_render_stack.py -k "handoff_support_profile"` → `5 passed`; `pytest -q tests/test_render_stack.py tests/test_core_planner.py tests/test_auto_shortlist_fusion.py tests/test_pro_fusion_quality.py` → `227 passed, 1 skipped`.
        - outcome: floor stability held (`pass+floor`) with unchanged winner path and headline metrics (`song_likeness=58.5`, `transition=53.7`, `overall=70.1`, `selection_score=73.729`).
        - action: keep patch/tests; next lever is renderer-side dynamic notch/HPF intensity keyed off the same planner crowding signals (resolver envelope-only tuning did not lift transition).
      - `runs/quality_push_pair2_renderer_crowding_notch_20260328_0415`
        - patch: renderer `section_support` entry/tail shaping is now crowding-conditioned for handoff transitions using resolved support gain as a risk proxy (dynamic handoff HPF start/end, notch bandwidth, notch depth, and tail LP end-frequency tightening for build/payoff).
        - regressions: added
          - `tests/test_render_stack.py::test_apply_support_entry_shape_build_entry_notches_high_crowding_handoff_more_than_low_crowding_handoff`
          - `tests/test_render_stack.py::test_apply_support_entry_shape_build_tail_notches_high_crowding_handoff_more_than_low_crowding_handoff`
        - validation:
          - `pytest -q tests/test_render_stack.py -k "support_entry_shape"` → `10 passed`.
          - `pytest -q tests/test_render_stack.py tests/test_core_planner.py tests/test_auto_shortlist_fusion.py tests/test_pro_fusion_quality.py` → `229 passed, 1 skipped`.
        - outcome: floor stability held (`pass+floor`) but winner headline metrics were unchanged (`song_likeness=58.5`, `transition=53.7`, `overall=70.1`, `selection_score=73.729`).
        - action: keep patch/tests for targeted handoff control; next step should shift leverage toward planner-level section structure/search (transition plateau persists despite resolver+renderer crowding tuning).
      - `runs/quality_push_pair2_handoff_weighted_dual_support_20260328_0615`
        - patch: planner shortlist dual-support pair ranking now consumes section `transition_mode` and prioritizes handoff-bearing build/payoff supports before lower-risk flow-only pairs.
        - implementation detail: `ai_dj.py::_build_auto_shortlist_variant_configs` now persists `transition_mode` in support candidate payloads and adds a handoff-pressure rank term (`max(policy risk/collision/low viability)` + section/mode boosts) in `_best_dual_support_pair`.
        - regressions: added `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_adaptive_dual_support_prioritizes_handoff_sections`.
        - validation:
          - `pytest -q tests/test_auto_shortlist_fusion.py -k "adaptive_dual_support_prioritizes_handoff_sections or adaptive_dual_support_avoids_extreme_risk_payoff_pair or support_policy_adapts_to_transition_risk"` → `3 passed`.
          - `pytest -q tests/test_render_stack.py tests/test_core_planner.py tests/test_auto_shortlist_fusion.py tests/test_pro_fusion_quality.py` → `230 passed, 1 skipped`.
        - outcome: floor stability held and winner path remained adaptive dual-support (`support_01_payoff_build_A`) with unchanged headline metrics (`song_likeness=58.5`, `transition=53.7`, `overall=70.1`, `selection_score=73.729`).
        - action: keep planner pairing patch/tests; next lever remains deeper structure search changes (section-window/ownership choices), not additional support-envelope tweaks.
      - `runs/quality_push_pair2_combo_handoff_priority_20260328_0815`
        - patch: planner combo (`dual_section_alternate`) ranking now propagates section `transition_mode` from opportunity collection and adds handoff-pressure priority (`seam_risk`, transition error, stretch pressure, section+mode boosts) so payoff/build handoff-bearing combos rank ahead of lower-pressure flow combos when budget is tight.
        - regressions: added `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_prefers_handoff_payoff_combo_over_lower_error_flow_combo`.
        - validation:
          - `pytest -q tests/test_auto_shortlist_fusion.py -k "prefers_handoff_payoff_combo_over_lower_error_flow_combo or prioritizes_payoff_combo_when_budget_is_one_combo_slot or prefers_payoff_build_combo_over_payoff_verse_when_single_combo_slot"` → `3 passed`.
          - `pytest -q tests/test_render_stack.py tests/test_core_planner.py tests/test_auto_shortlist_fusion.py tests/test_pro_fusion_quality.py` → `231 passed, 1 skipped`.
        - outcome: pair2 rerun held `pass+floor`; winner remained adaptive dual-support (`support_01_payoff_build_A`) with unchanged headline metrics (`song_likeness=58.5`, `transition=53.7`, `overall=70.1`, `selection_score=73.729`).
        - action: keep planner patch/tests for structure search quality; next leverage remains deeper section-window ownership proposal generation (not ranking-only adjustments).
      - `runs/quality_push_pair2_contiguous_handoff_combo_20260328_1015`
        - patch: planner combo ranking now adds contiguous ownership-chain preference for handoff-bearing combos (`adjacent section_index`, same `alternate_parent`, explicit handoff mode) so shortlist favors structurally coherent handoff chains over split/noncontiguous alternatives when errors are close.
        - regressions: added
          - `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_prefers_contiguous_handoff_combo_over_lower_error_noncontiguous_combo`
          - `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_prefers_contiguous_same_owner_handoff_combo_over_lower_error_split_combo`
        - validation:
          - `pytest -q tests/test_auto_shortlist_fusion.py -k "contiguous_handoff_combo or same_owner_handoff_combo or prefers_handoff_payoff_combo_over_lower_error_flow_combo or prioritizes_payoff_combo_when_budget_is_one_combo_slot or prefers_payoff_build_combo_over_payoff_verse_when_single_combo_slot"` → `4 passed`.
          - `pytest -q tests/test_render_stack.py tests/test_core_planner.py tests/test_auto_shortlist_fusion.py tests/test_pro_fusion_quality.py` → `232 passed, 1 skipped`.
        - outcome: pair2 rerun held `pass+floor`; winner remained adaptive dual-support (`support_01_payoff_build_A`) with unchanged headline metrics (`song_likeness=58.5`, `transition=53.7`, `overall=70.1`, `selection_score=73.729`).
        - action: keep contiguous ownership-chain priority patch/tests; next leverage remains deeper section-window ownership proposal generation because ranking-only refinements are stable but transition plateau persists.
      - `runs/quality_push_pair2_handoff_ownership_chain_20260328_1004`
        - patch: shortlist combo ranking now explicitly rewards contiguous same-owner handoff chains (adjacent sections with same alternate parent + handoff mode) so pro search favors cleaner ownership blocks over lower-error split ownership swaps.
        - implementation detail: `ai_dj.py::_build_auto_shortlist_variant_configs` adds `ownership_chain_combo` priority in `_combo_priority` (after handoff-bearing core pressure, before raw error) using `section_index` adjacency + shared `alternate_parent` + handoff transition modes.
        - regressions:
          - updated `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_prefers_contiguous_handoff_combo_over_lower_error_noncontiguous_combo`.
          - added `tests/test_auto_shortlist_fusion.py::test_build_auto_shortlist_variant_configs_prefers_contiguous_same_owner_handoff_combo_over_lower_error_split_combo`.
        - validation:
          - `pytest -q tests/test_auto_shortlist_fusion.py tests/test_core_planner.py` → `146 passed, 1 skipped`.
          - `pytest -q tests/test_auto_shortlist_fusion.py tests/test_core_planner.py tests/test_render_stack.py tests/test_pro_fusion_quality.py` → `232 passed, 1 skipped`.
        - outcome: pair2 rerun preserved winner policy `pass+floor` and support winner path; headline metrics unchanged vs `20260328_0415` (`song_likeness=58.5`, `transition=53.7`, `overall=70.1`, `selection_score=73.729`).
        - action: keep ownership-chain ranking guard (minimal/reversible), then target upstream proposal generation (explicit handoff-cluster candidate synthesis before ranking) since ranking-only changes still plateau transition.
   - Focus:
     - push transition above 53.8 by combining shortlist risk policy with render-time support envelope shaping,
     - keep anti-medley penalties and hard-floor gate untouched.
   - Files likely touched:
     - `ai_dj.py`
     - `tests/test_auto_shortlist_fusion.py`
     - `src/core/render/renderer.py`
     - `src/core/render/resolver.py`
     - `tests/test_render_stack.py`

## Next Task (auto-start immediately after current)
1. **Structure/planner pivot: improve section-level ownership/transition plan before render shaping**
   - Why: two consecutive crowding-focused render/resolver passes held quality gates but did not move transition; highest leverage is now upstream section planning.
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
