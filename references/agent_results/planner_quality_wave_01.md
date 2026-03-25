# Planner quality wave 01 — contiguous donor-cluster guard

## What I changed
Implemented a new **hard post-selection authenticity guard** in `src/core/planner/arrangement.py` to reduce stitched/garbage outputs caused by a **single donor cameo immediately snapping back to the majority parent**.

Concretely:
- Extended `_apply_section_level_authenticity_guard(...)` to detect major-section programs of the form:
  - `A, A, B, A`
  - or longer same-shape variants where a lone alternate-parent major section is immediately followed by a rebound to the original dominant parent.
- When that rebound section has a **safe alternate** from the cameo parent, the guard now switches it, turning the isolated cameo into a **contiguous donor cluster** instead of a one-off fake identity blip.
- Safety reuse stays conservative and aligned with existing planner gates:
  - stretch gate / stretch ratio
  - seam risk
  - transition viability
  - role prior
  - groove continuity / groove confidence

This is a planner-only material-selection change aimed at stronger **one-song illusion / macro continuity** without touching renderer code.

## Why this matters
The current stack already had a **soft `single_cameo_rebound_gap` penalty**, but memory and prior live evidence showed that soft weighting was not enough to reliably change chosen material. The new guard makes this failure mode **structural** when a musically safe fix exists.

Musical effect target:
- avoid `majority-majority-cameo-majority` programs that read like brief source-switch interruptions
- prefer a short, intentional donor feature cluster when the donor has already been introduced
- reduce medley feel and improve contiguous parent identity across late major sections

## Tests added
Updated `tests/test_core_planner.py` with new regressions:
1. `test_section_level_authenticity_guard_extends_single_major_cameo_into_contiguous_donor_cluster_when_safe`
   - verifies `A, A, B, A` major ownership becomes `A, A, B, B` when the bridge/payoff-side alternate is safe.
2. `test_section_level_authenticity_guard_does_not_force_cameo_extension_when_only_safe_alternate_is_too_seam_risky`
   - verifies the guard does **not** fire when the only cluster-extending alternate fails seam safety.

## Validation
Ran:
- `/Users/johnpeter/venvs/vocalfusion-env/bin/python3 -m pytest -q tests/test_core_planner.py -k 'section_level_authenticity_guard'`
  - `4 passed`
- `/Users/johnpeter/venvs/vocalfusion-env/bin/python3 -m pytest -q tests/test_core_planner.py`
  - `108 passed, 1 skipped`

## Practical implication for next mix
If the planner already commits to donor material for one major late section, it now has a hard bias toward making that donor appearance feel like a **real contiguous feature moment** rather than a single stitched cameo followed by an immediate reversion. That should help reduce fake two-parent identity and improve whole-song readability on failing pairs where donor appearance was previously too fragmentary to sound intentional.
