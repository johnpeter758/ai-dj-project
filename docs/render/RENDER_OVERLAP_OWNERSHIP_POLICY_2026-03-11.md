# Render overlap / ownership policy for the first deterministic renderer

_Date: 2026-03-11_

## Goal
Define the **simplest effective** policy that keeps the first renderer musical, deterministic, and easy to validate.

## Core rule
At any moment:
1. **One parent owns the low end.**
2. **One parent owns the foreground.**
3. Everything else is optional background support.

This should be treated as a hard render contract, not a soft hint.

---

## 1) Low-end ownership

### Hard rule
Only **one** parent may own **kick + bass/sub** at a time.

### Default behavior
- `low_end_owner = A` or `B`
- the non-owner must be one of:
  - fully muted in low frequencies
  - high-passed / low-cut enough to remove sub and kick dominance
  - excluded entirely

### Allowed cases
- owner drums + owner bass + donor harmonic/vocal background
- owner drums + donor vocal
- owner bass + donor percussion/harmonic tease

### Disallowed cases
- bass + bass overlap
- kick-dominant drums from both parents at once
- full-spectrum overlap where both parents imply low-end dominance

### Transition exception
A low-end handoff may exist only as a **very short explicit swap window**:
- `1–2 beats` max
- outgoing low end fades / filters down
- incoming low end fades / filters up
- work order must name the handoff explicitly

If a transition needs longer overlap than that, the low end should still remain single-owner while only mids/highs crossfade.

---

## 2) Foreground / background ownership

### Hard rule
Only **one** parent may own the foreground at a time.

### Foreground means
Whichever parent currently carries the main attention-driving element, usually one of:
- lead vocal
- dominant hook
- main drum+bass drive
- strongest melodic statement

### Background means
The other parent may contribute only support material such as:
- filtered harmonic bed
- light percussion
- texture / FX / tail
- short hook fragment at reduced level

### Default policy
- foreground owner plays full-range or near-full-range role set
- background owner is reduced by **role**, **level**, and often **spectrum**
- if the background starts to feel competitive, collapse to single-source

### Good default combinations
- foreground: `A vocal + A drums/bass`; background: `B harmonic texture`
- foreground: `A drums/bass + A hook`; background: `B filtered motif`
- foreground: `B vocal`; background: `A instrumental support`

### Bad combinations
- both parents pushing lead vocal
- both parents pushing dense hooks in the same register
- both parents effectively full-spectrum at equal level

---

## 3) When overlap is allowed

Overlap is allowed only when ownership remains clear.

### Safe overlap classes
1. **Bed crossfade**
   - mids/highs or harmonic bed overlap
   - low end stays single-owner
   - one foreground owner remains obvious

2. **Filtered tease-in**
   - incoming donor appears band-limited or low in level
   - used before payoff or handoff

3. **Role swap**
   - one role changes while another stays stable
   - example: drums swap while harmonic bed holds

4. **Vocal tail / echo carry**
   - outgoing vocal may spill briefly as a tail
   - incoming foreground must still become dominant quickly

### Safe overlap limits
- `swap`: `1–2 beats`
- `blend`: `2–4 beats` default, `8 beats` only for sparse/non-conflicting material
- long overlaps are acceptable only if donor is clearly backgrounded and low-end ownership stays singular

---

## 4) When overlap must collapse to single-source

Collapse to one source whenever any of these are true:
- both parents want bass/sub ownership
- both parents have lead vocals active
- both parents have dense midrange hooks competing
- required stretch / pitch shift makes one layer obviously degraded
- background stem separation is too artifact-heavy
- phrase boundary is unclear and overlap would sound indecisive

### Practical collapse rule
If the renderer cannot satisfy both:
- `one low_end_owner`
- `one foreground_owner`

then it must render the moment as:
- one active parent full-range, or
- one active parent full-range plus only clearly non-conflicting donor support

When in doubt, choose the cleaner single-source handoff over a muddy blend.

---

## 5) Recommended v1 policy by role

| Role | Overlap policy |
|---|---|
| Drums | Can overlap briefly, but only one kick-dominant drum source should lead |
| Bass/Sub | Single owner only |
| Harmonic bed | Can overlap if one is backgrounded or filtered |
| Lead vocal | Single owner only by default |
| FX / tails | Can overlap freely if level-controlled |

---

## 6) How this should appear in manifest / resolved plan

The current planner schema is too thin for deterministic overlap control. The resolved manifest should make ownership explicit per section and per transition.

### Add section-level fields
Each resolved section should include at least:

```json
{
  "section_index": 1,
  "label": "build",
  "foreground_owner": "B",
  "background_owner": "A",
  "low_end_owner": "B",
  "allowed_overlap": "background_only",
  "vocal_policy": "B_only",
  "overlap_beats_max": 4,
  "collapse_if_conflict": true
}
```

### Suggested enums
- `foreground_owner`: `A | B`
- `background_owner`: `A | B | none`
- `low_end_owner`: `A | B`
- `allowed_overlap`:
  - `none`
  - `background_only`
  - `role_swap_only`
  - `tail_only`
- `vocal_policy`:
  - `none`
  - `A_only`
  - `B_only`
  - `brief_overlap_allowed`

---

## 7) How this should appear in work orders

Do not make the renderer infer ownership from labels like `blend` or `swap`. Work orders should carry the rule directly.

### Required work-order concepts
Each audio work order involved in an overlap region should expose:
- `role`
- `parent_id`
- `target_start_sec`
- `target_duration_sec`
- `gain_db`
- `filter_preset` or equivalent spectral restriction
- `foreground_state`: `foreground | background`
- `low_end_state`: `owner | suppressed | none`
- `vocal_state`: `lead | suppressed | none`
- `conflict_policy`: `mute_other | duck_other | reject`

### Example transition work-order shape
```json
{
  "order_type": "transition_head",
  "parent_id": "B",
  "role": "drums",
  "foreground_state": "foreground",
  "low_end_state": "owner",
  "target_start_sec": 32.0,
  "target_duration_sec": 1.0,
  "gain_db": -1.5,
  "filter_preset": "full",
  "conflict_policy": "mute_other"
}
```

```json
{
  "order_type": "transition_tail",
  "parent_id": "A",
  "role": "harmonic",
  "foreground_state": "background",
  "low_end_state": "suppressed",
  "target_start_sec": 32.0,
  "target_duration_sec": 2.0,
  "gain_db": -8.0,
  "filter_preset": "highpass_bg",
  "conflict_policy": "duck_other"
}
```

---

## 8) Simple deterministic decision ladder

For every section / transition, resolve in this order:
1. Pick `foreground_owner`
2. Pick `low_end_owner`
3. Enforce `vocal_policy`
4. Allow only background-safe donor roles
5. If any conflict remains, collapse to single-source or single-owner + background texture

This keeps the renderer deterministic and testable.

---

## Bottom line
For v1, the renderer should behave like a disciplined arranger, not a democratic mixer:
- **one low-end owner**
- **one foreground owner**
- **background support only when clearly subordinate**
- **collapse aggressively to single-source when ownership gets ambiguous**

That is the simplest policy that will prevent most amateur-sounding mud while fitting the repo’s current resolver/work-order architecture.
