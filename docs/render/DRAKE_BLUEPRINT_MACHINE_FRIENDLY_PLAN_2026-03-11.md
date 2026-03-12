# Drake-led 5-section blueprint → machine-friendly arrangement plan

## What the current stack can use immediately
The current planner/resolver path accepts per-section fields:
- `label`
- `start_bar`
- `bar_count`
- `source_parent`
- `source_section_label`
- `target_energy`
- `transition_in`
- `transition_out`

That means the 5-section blueprint can be converted directly into a deterministic `sections[]` array for the existing resolver.

## Important current-schema limitations
1. **No explicit donor/secondary-parent field in `PlannedSection`.**
   - The blueprint's donor rules, hard ownership intent, and vocal mute policy are not representable in the current `ChildArrangementPlan` schema.
   - Those must live in planning notes, a companion manifest, or downstream render policy.

2. **No dedicated ownership-hints field.**
   - "Owner A with donor B filtered bed only" cannot be encoded structurally today.

3. **No vocal policy field.**
   - Rules like "A lead dominant / B vocals muted" are currently advisory only.

4. **No per-section donor timing windows.**
   - Constraints like "B donor only in first 4 bars of section 5" are not directly representable.

5. **Current structural analysis is too coarse for exact source section mapping.**
   - In the available DNA artifacts, both songs only expose `section_0`.
   - So `source_section_label` can only safely be `section_0` for now, with a warning that the resolver will effectively snap from full-song bounds.

## Section mapping from the blueprint

| # | Child bars | Label | Use now as `source_parent` | `source_section_label` strategy | `transition_in` | `transition_out` | `target_energy` | Ownership hint / policy note |
|---|---:|---|---|---|---|---|---:|---|
| 1 | 0-8 | Relax My Eyes intro wash | B | `section_0` | `null` | `lift` | 0.28 | B owns. A only as low-level accent/texture if available. |
| 2 | 8-16 | Drake entry | A | `section_0` | `swap` | `lift` | 0.52 | A owns. B should be filtered-only or percussion-light; no B lead vocal. |
| 3 | 16-24 | Drake consolidation | A | `section_0` | `lift` | `drop` | 0.66 | A owns. Tighten to near A-only with minor B texture/percussion support. |
| 4 | 24-32 | Relax My Eyes release | B | `section_0` | `drop` | `swap` | 0.72 | B owns. A lead muted; A only for short riser/tail support late in section. |
| 5 | 32-48 | Drake payoff / outro hold | A | `section_0` | `swap` | `null` | 0.84 | A owns. B only as brief front-loaded atmosphere/percussion; ideally gone by bar 40. |

## Suggested JSON example for the current planner schema
```json
{
  "analysis_version": "0.1.0",
  "planning_notes": [
    "Blueprint-derived deterministic plan: Drake-led 5-section structure.",
    "Ownership hints, donor restrictions, and vocal policy are documented externally because PlannedSection cannot encode them yet.",
    "Current song DNA only exposes section_0 for both parents, so all source_section_label values are coarse placeholders pending better section segmentation."
  ],
  "sections": [
    {
      "label": "relax_my_eyes_intro_wash",
      "start_bar": 0,
      "bar_count": 8,
      "source_parent": "B",
      "source_section_label": "section_0",
      "target_energy": 0.28,
      "transition_in": null,
      "transition_out": "lift"
    },
    {
      "label": "drake_entry",
      "start_bar": 8,
      "bar_count": 8,
      "source_parent": "A",
      "source_section_label": "section_0",
      "target_energy": 0.52,
      "transition_in": "swap",
      "transition_out": "lift"
    },
    {
      "label": "drake_consolidation",
      "start_bar": 16,
      "bar_count": 8,
      "source_parent": "A",
      "source_section_label": "section_0",
      "target_energy": 0.66,
      "transition_in": "lift",
      "transition_out": "drop"
    },
    {
      "label": "relax_my_eyes_release",
      "start_bar": 24,
      "bar_count": 8,
      "source_parent": "B",
      "source_section_label": "section_0",
      "target_energy": 0.72,
      "transition_in": "drop",
      "transition_out": "swap"
    },
    {
      "label": "drake_payoff_outro_hold",
      "start_bar": 32,
      "bar_count": 16,
      "source_parent": "A",
      "source_section_label": "section_0",
      "target_energy": 0.84,
      "transition_in": "swap",
      "transition_out": null
    }
  ]
}
```

## Companion ownership hints the renderer cannot yet consume directly
If you want to preserve the blueprint intent alongside the plan, attach something like this in docs or a sidecar file:

```json
{
  "section_hints": [
    {
      "label": "relax_my_eyes_intro_wash",
      "owner": "B",
      "donor_parent": "A",
      "donor_policy": "low_level_texture_only",
      "vocal_policy": "B texture/lead okay, A lead muted"
    },
    {
      "label": "drake_entry",
      "owner": "A",
      "donor_parent": "B",
      "donor_policy": "filtered_bed_or_light_percussion_only",
      "vocal_policy": "A lead dominant, B lead muted"
    },
    {
      "label": "drake_consolidation",
      "owner": "A",
      "donor_parent": "B",
      "donor_policy": "texture_or_percussion_only",
      "vocal_policy": "A lead dominant, B vocals muted"
    },
    {
      "label": "relax_my_eyes_release",
      "owner": "B",
      "donor_parent": "A",
      "donor_policy": "transition_support_only_late_section",
      "vocal_policy": "B lead allowed, A lead muted"
    },
    {
      "label": "drake_payoff_outro_hold",
      "owner": "A",
      "donor_parent": "B",
      "donor_policy": "first_4_bars_only_then_remove",
      "vocal_policy": "A lead dominant through bar 40, B vocals muted"
    }
  ]
}
```

## Recommended interpretation
- Treat the JSON `sections[]` block above as the **immediately usable plan**.
- Treat donor usage / vocal rules as **human-readable render constraints** until the planner schema grows fields for:
  - `donor_parent`
  - `donor_policy`
  - `vocal_policy`
  - `ownership_mode`
  - donor activation windows within a section

## Net result
This preserves the blueprint's intended macro-arc:
- B establishes color
- A controls the main narrative spine
- B gets one true spotlight release
- A closes with the dominant payoff

That is fully compatible with the current resolver shape, with the explicit caveat that source section selection is still structurally coarse (`section_0` only).