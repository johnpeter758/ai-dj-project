# Drake + Relax My Eyes: first improvement pass for deterministic render

_Date: 2026-03-11_

## What I inspected
- `runs/prototype_20260311_081144/arrangement_plan.json`
- `runs/prototype_20260311_081144/compatibility_report.json`
- `src/core/planner/arrangement.py`
- `src/core/render/resolver.py`
- `src/core/render/renderer.py`
- existing render docs, especially:
  - `docs/FIRST_LISTENABLE_DRAKE_RELAX_MY_EYES_BLUEPRINT_2026-03-11.md`
  - `docs/RENDER_OVERLAP_OWNERSHIP_POLICY_2026-03-11.md`
  - `docs/DETERMINISTIC_RENDERER_FAILURE_AUDIT_2026-03-11.md`

## Main finding
The **highest-probability first improvement** is **section selection**.

Right now the first render is still being driven by a stub plan:
- `intro = A`
- `build = B`
- `payoff = A`
- with placeholder `section_0` / `section_1` references

For this pair, that is the wrong first lever because the compatibility report already says:
- **large tempo gap** (~120.19 vs 132.51)
- **weak harmonic match**
- **large energy mismatch**

That means the renderer will sound bad fastest when it chooses the wrong section owner or wrong source window, even before any mix-rule sophistication.

## Prioritized patch order

### 1) Section selection / section ownership
**Do first.**

Why:
- The current planner is the biggest source of bad musical decisions.
- The existing Drake/Relax blueprint already points to the safest shape: **Drake-led, single-owner sections, one real Relax spotlight section**.
- If section ownership is wrong, later overlap/ducking/stretch tweaks only polish a weak arrangement.

What to tighten first:
- Replace the 3-section stub with the existing 5-section deterministic blueprint:
  - B intro bed
  - A arrival
  - A hybrid tension
  - B release spotlight
  - A payoff/outro
- Make owner choice explicit per section.
- Prefer phrase-safe 4/8-bar windows instead of broad `section_0` fallbacks.

Expected payoff:
- Biggest immediate jump in coherence.
- Output starts feeling arranged instead of arbitrarily alternated.

---

### 2) Transition shortening
**Do second.**

Why:
- Current transition defaults are too permissive for this pair.
- `blend` maps to **8 beats** and `swap`/`lift` to **4 beats**, which is risky when both sources are full mixes.
- For a weakly matched pair, long transition windows expose mud, vocal conflict, and groove mismatch.

What to tighten:
- Make this pair mostly **cut / drop / short swap**.
- Cap cross-source overlap to:
  - **1-2 beats** for low-end handoffs
  - **2-4 beats max** for sparse/non-vocal material
- Prefer “announce then swap” over “blend then hope.”

Expected payoff:
- Faster improvement in clarity than EQ tricks.
- Less amateur-sounding indecision at section boundaries.

---

### 3) Overlap reduction
**Do third.**

Why:
- The renderer currently sums section audio into one master buffer.
- Even with one base work order per section, fades at handoffs can still create accidental full-mix overlap.
- For this pair, full-spectrum overlap is usually worse than a cleaner handoff.

What to tighten:
- Default to **no full-mix overlap** unless explicitly allowed.
- Permit overlap only for background-safe material, not two full sections.
- If conflict is likely, collapse to single-source sooner.

Expected payoff:
- Cleaner transitions and less density buildup.
- More audible separation between the Drake-led spine and the donor moments.

---

### 4) Low-end ownership
**Do fourth, but enforce as a hard rule once overlap exists.**

Why:
- If transitions remain even slightly overlapped, low-end conflict becomes the fastest route to mud.
- However, once overlap is shortened/reduced, this is easier to control deterministically.

What to tighten:
- During any handoff, only one parent owns kick/sub.
- For this pair, treat low-end transfer as a **micro-event**, not an 8-beat blend.
- If no stem-safe suppression exists, pick one source and mute the other sooner.

Expected payoff:
- Drops feel more intentional.
- Less “smaller after the transition” effect.

---

### 5) Vocal-first balance
**Do fifth.**

Why:
- It matters a lot, but section ownership and shorter transitions already solve much of it indirectly.
- This pair should be **A-vocal-first** most of the time, with B getting one clean spotlight section.

What to tighten:
- Default rule: one lead vocal owner only.
- For this pair:
  - A owns the narrative spine
  - B vocal appears only in the dedicated release section
  - donor vocals elsewhere should be muted, tail-only, or absent

Expected payoff:
- Better intelligibility and identity.
- Render feels like a chosen arrangement, not two songs competing.

---

### 6) Stretch policy
**Do sixth, unless listening shows obvious warp artifacts immediately.**

Why:
- Stretch matters, but the larger early failure here is likely **bad arrangement choice**, not just DSP artifacts.
- Still, this pair has a meaningful tempo gap, so stretch policy should stay conservative.

What to tighten:
- Avoid long sustained full-mix blends across the tempo gap.
- Use stretching mainly on short sections or background-safe material.
- If a chosen donor window needs aggressive stretch, shorten or replace that section instead of forcing it.

Expected payoff:
- Fewer warped/cymbal-smeared moments.
- Better realism once the arrangement logic is cleaner.

## Recommended first patch order
1. **Section selection / section ownership**
2. **Transition shortening**
3. **Overlap reduction**
4. **Low-end ownership**
5. **Vocal-first balance**
6. **Stretch policy**

## Why this order is most likely to improve listenability quickly
This renderer does **not** yet have enough source-role intelligence to rescue a weak section plan with mix tricks. So the first fast win is to make the child timeline more intentional:
- pick better sections
- keep one owner at a time
- make transitions shorter
- only then refine conflict rules

## Smallest high-leverage implementation stance
For the next pass, do **not** aim for smarter blending.

Aim for this instead:
- **Drake-led 5-section deterministic plan**
- **short swaps, not long blends**
- **single-owner full-mix sections**
- **one real Relax My Eyes spotlight section**

That is the most likely first patch pass to make the render noticeably more listenable without expanding renderer complexity too early.
