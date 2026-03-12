# Render v1 contract

_Date: 2026-03-11_
_Status: canonical implementation contract for the first deterministic render path_

## Purpose
Define the narrow contract between:

**planner -> resolver -> renderer -> export**

v1 optimizes for deterministic, testable, listenable output from current analysis quality. When rules conflict, prefer the safer, cleaner render over a more ambitious blend.

---

## 1) Canonical timeline truth

### Hard rule
**Planner bars are the canonical child timeline.**

The planner owns:
- section order
- `start_bar`
- `bar_count`
- `source_parent`
- coarse section/transition intent

The resolver must derive target timing from bars, not from source section seconds.

### Timing rule
For each planned section:
- `target_beats = bar_count * beats_per_bar`
- `target_start_beat = start_bar * beats_per_bar`
- `target_duration_sec = target_beats * 60 / anchor_bpm`
- `target_start_sec = target_start_beat * 60 / anchor_bpm`

`anchor_bpm` is the chosen section owner tempo in v1.

### Consequence
Source section duration never defines child duration. Source audio is conformed to the child timeline, not the other way around.

---

## 2) Resolver responsibilities

The resolver is a pure timing / selection layer. It must:
- validate the arrangement timeline
- resolve source-parent references
- build parent beat / phrase grids
- select a musically legal source window
- snap source bounds to safe musical boundaries
- compute exact target timing from planner bars
- compute explicit transform values (`stretch_ratio`, `semitone_shift`)
- emit explicit work orders and warnings/fallbacks

The resolver must **not** do:
- audio I/O
- DSP rendering
- time-stretch execution
- pitch-shift execution
- mix/master/export processing

### Source-window rule
In v1:
- `tempo.beat_times` are the hard snap grid
- `phrase_boundaries_seconds` are the extraction scaffold
- `source_section_label` is a soft hint when analysis is coarse

Snap policy:
- start snaps down / at safe boundary
- end snaps up / at safe boundary
- if snapping collapses the interval, expand to a minimum legal beat window

### Weak-section fallback rule
If the requested source section is missing, unresolved, generic/coarse (for example `section_0`), or spans most of the song, the resolver must **not** use the full-song span as the clip.

Instead it must:
1. convert the child request into target beat length
2. search phrase-safe candidate windows
3. prefer 4/8/16-bar legal windows
4. choose the safest deterministic candidate
5. if no safe window exists, downgrade usage instead of forcing a bad blend

---

## 3) Renderer responsibilities

The renderer is a pure execution engine over the resolved manifest. It must:
- validate manifest structure
- load source audio from exact paths in work orders
- extract the exact source window
- conform audio to exact target duration
- apply explicit gain / fade / transition settings
- place audio at exact target times
- sum deterministically
- write raw/master outputs and manifest

The renderer must **not** infer:
- section timing
- ownership rules
- fallback policy
- source window choice
- hidden transition semantics from planner labels alone

If the manifest is ambiguous, that is a resolver contract failure.

---

## 4) Ownership rules

### Global hard rules
At any moment:
1. **One parent owns the low end.**
2. **One parent owns the foreground.**
3. **One lead vocal owner by default.**

### Required resolved-section fields
Each resolved section must make these explicit:
- `foreground_owner`
- `background_owner`
- `low_end_owner`
- `vocal_policy`
- `allowed_overlap`
- `overlap_beats_max`
- `collapse_if_conflict`

### Required work-order intent
Each work order in an overlap-capable region must make explicit:
- `role`
- `foreground_state`
- `low_end_state`
- `vocal_state`
- `conflict_policy`

### Default v1 stance
- low end: single owner only
- lead vocal: single owner only
- donor usage: background-only unless clearly safe
- when ownership becomes ambiguous: collapse aggressively to single-source or owner + light background support

---

## 5) Fallback ladder

When the ideal render cannot be satisfied, v1 must degrade in this order:

1. **Choose a phrase-safe target-length window** inside the source parent
2. **Shorten donor usage** to a safer 4-bar or 8-bar region
3. **Reduce donor role** to filtered bed / percussion / tail / texture
4. **Convert blend to swap** with bounded overlap
5. **Collapse to single-source ownership** for the section or transition

Do **not** keep a full-song snapped window as the fallback for weak sections.

### Trigger conditions for downgrade
Examples:
- source section label missing or unresolved
- only coarse full-song-like section available
- no phrase-safe candidate window fits
- stretch ratio outside safe band for sustained blend
- low-end ownership conflict
- simultaneous competing lead material
- donor source missing / unusable

All downgrades must be recorded in manifest warnings/fallbacks.

---

## 6) Stretch / transition safety rules

### Stretch stance
- section-level deterministic stretching is acceptable in v1
- conservative blends should stay near unity
- aggressive full-mix blend under large stretch should be rejected or downgraded

Practical stance from current notes:
- safe-ish debug blend range: roughly within ±6%
- caution beyond that
- sustained blend should generally collapse before extreme stretch

### Transition stance
Use a tiny deterministic transition library only.

Safe overlap defaults:
- `swap`: 1-2 beats
- `blend`: 2-4 beats default
- longer overlap only if donor is clearly backgrounded and low-end ownership stays singular

Outside explicit overlap windows, only one primary section owner should be active.

---

## 7) Minimum synthetic test requirements

The minimum v1 synthetic suite is:

1. `test_resolve_render_plan_emits_contiguous_manifest`
   - validates section timing math, ordering, and minimum work-order coverage

2. `test_manifest_declares_single_low_end_and_foreground_owner_in_overlap`
   - validates ownership contract is explicit and non-ambiguous

3. `test_render_is_bitwise_or_samplewise_deterministic_for_same_inputs`
   - validates deterministic execution

4. `test_render_preserves_expected_duration_after_trim_pad_and_stretch`
   - validates exact sample-count duration conservation

5. `test_blend_transition_overlap_window_is_bounded_and_energy_sane`
   - validates bounded overlap and no gross double-summing spike

6. `test_renderer_falls_back_to_safe_single_source_when_requirements_fail`
   - validates recoverable failure behavior and fallback recording

### Fixture rule
Use only tiny synthetic fixtures or in-memory generated audio. No real-song dependence in unit tests.

---

## 8) Known v1 limitations

Current accepted limitations:
- source section labels may be structurally weak and often collapse to `section_0`
- beat grid is still derived from a global tempo/beat tracker, not a full bar-stable musical timeline
- phrase boundaries are more reliable than section labels for extraction
- harmonic policy is minimal; `semitone_shift` may remain `0.0` until key-aware rendering grows up
- ownership intent is richer in docs than in current planner schema, so some policy lives in resolved render metadata rather than planner-native fields
- renderer is deterministic but still simple: no advanced role-aware stem intelligence, no sophisticated vocal masking, no fully mature transition DSP

### Practical implication
v1 should prefer:
- clear ownership
- phrase-safe windows
- short explicit transitions
- conservative blends
- documented downgrades

over ambitious 50/50 fusion.

---

## 9) Bottom line

The render v1 contract is:
- **planner bars define the child timeline**
- **resolver turns plan + SongDNA into explicit, safe, deterministic work orders**
- **renderer executes those work orders without inference**
- **ownership stays singular where it matters**
- **fallbacks prefer clarity and safety over forced blending**
- **the path is validated by a small synthetic deterministic test suite**
