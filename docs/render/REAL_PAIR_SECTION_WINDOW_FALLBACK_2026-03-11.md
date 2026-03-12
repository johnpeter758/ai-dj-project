# Real-pair resolver fallback for coarse `section_0` structure

_Date: 2026-03-11_

## Problem
Current real-pair renders often have only a single coarse source section (`section_0`). In `src/core/render/resolver.py`, that causes the resolver to treat nearly the whole song as the candidate window, then snap the whole span to phrase/beat bounds and stretch it to the target child duration.

That is unsafe because the chosen clip window is not actually chosen musically; it is just the full source bounds. The first rerender will sound bad fastest at the **window-selection** layer, not the DSP layer.

## Best fallback principle
When section labels are weak, the resolver should:

1. treat `source_section_label` as a **soft hint**, not a hard container
2. treat `tempo.beat_times` as the **hard snap grid**
3. treat `structure.phrase_boundaries_seconds` as the **main extraction scaffold**
4. choose a **target-length phrase window** inside the source song
5. prefer a clean hard swap over forcing a bad donor window

In short: **pick a musically legal 4/8/16-bar clip first, then stretch it**.

## Why the current behavior is risky
Current `_resolve_source_window()` effectively does this:
- resolve section label
- if found, use its raw start/end
- snap start/end to phrase boundaries or beats
- return the whole snapped range

With `section_0` covering almost the full song, the resolver returns a huge source span. Later, `stretch_ratio = source_duration / target_duration_sec` can become extreme, and the work order is built from a window that was never phrase-selected for the planned bar count.

That leaks weak analysis into rendering.

## Recommended fallback strategy

### 1) Detect weak-section mode
Treat the requested section as **weak** if any of these are true:
- label missing or unresolved
- section duration >= ~70% of song duration
- section duration > 64 bars at source tempo
- only one section exists
- section has generic placeholder label like `section_0` and there is no richer segmentation

When weak-section mode is active, do **not** use the full section bounds as the extracted clip.

### 2) Convert the child request into musical length
For each planned section:
- `target_beats = bar_count * beats_per_bar`
- default legal lengths should be multiples of 4 beats
- prefer windows of:
  - 16 beats = 4 bars for short bridges / safest minimum
  - 32 beats = 8 bars for most intros/builds/transitions
  - 64 beats = 16 bars for payoffs / longer holds

If the plan asks for an odd duration, still search on the beat grid, but prefer the nearest legal phrase-sized window first.

### 3) Build candidate windows from phrase boundaries
Use `phrase_boundaries_seconds` to generate candidate start points. For each candidate phrase boundary:
- snap candidate start to the nearest safe beat
- choose candidate end by advancing `target_beats` on the beat grid
- if exact target length is impossible near song end, back-shift to the latest phrase boundary that fits
- if still impossible, allow a shorter fallback only when the section is transition-only; otherwise warn and collapse to a hard swap

This is better than stretching an arbitrary large source span into the target section.

### 4) Score candidate windows instead of taking the first one
In weak-section mode, resolver should score a small set of candidate windows and choose the safest. A simple deterministic score is enough.

Suggested priority order:
1. **exact beat-count match** to `target_beats`
2. **starts on phrase boundary**
3. **beat index modulo 4 == 0** at clip start when possible
4. **stays inside hinted section bounds** if the weak section still provides a rough region
5. **not too close to song edges** unless doing intro/outro behavior
6. **minimal stretch severity** after mapping to target BPM

A practical scoring sketch:
- +4 exact target beat match
- +3 phrase-aligned start
- +2 phrase-aligned end
- +2 start beat index divisible by 4
- +1 contained within hinted region
- -3 if within last 1 phrase of song and not explicitly outro-like
- -5 if implied stretch exceeds safe band

No MIR sophistication is required for v1; just choose the best phrase-safe legal window.

### 5) Use section hints only to bias search region
If `source_section_label` resolves but is weak, convert it into a **search bias**, not a hard clip.

Example search policy:
- weak section near song start -> search first 35% of phrase boundaries
- weak section centered mid-song -> search middle 40%
- weak section near song end -> search last 35%
- full-song `section_0` -> search all phrase boundaries

This preserves intent without trusting bad segmentation.

## Resolver heuristics to implement

### Safe boundary rules
- start on beat, preferably phrase boundary
- end on beat, preferably phrase boundary
- snap starts **down/at** and ends **up/at** only when that preserves target beat count
- if nearest beat is > 250 ms away from intended phrase boundary, choose another phrase window instead of forcing the snap

### Stretch rules
After choosing the source window, compute stretch ratio. Use conservative gates:
- ideal: 0.97-1.03
- acceptable debug range: 0.94-1.06
- caution: 0.92-1.08
- beyond that: avoid sustained blend; downgrade to swap / accent-only donor

### Downgrade ladder
If no safe phrase window exists:
1. shorten donor usage to 4 bars
2. use donor only for transition accent / percussion / filtered texture
3. collapse to single-owner hard swap

Do **not** keep the original full-song snapped window as the fallback.

## Minimal algorithm
For each planned section:

1. Resolve source parent and compute `target_beats`.
2. Read beat grid and phrase boundaries.
3. Resolve section label.
4. If section is strong, search phrase-sized subwindows inside it.
5. If section is weak, search phrase-sized windows across the biased region.
6. Score candidates using exact length + phrase legality + stretch safety.
7. Choose best candidate.
8. If no candidate clears minimum safety score, downgrade transition policy instead of forcing blend.
9. Emit warnings/fallback notes describing why the downgrade happened.

## Smallest high-leverage code change
The best next implementation change is to replace `_resolve_source_window()` with a two-step resolver:

- `resolve_candidate_region(...)` -> returns search bounds / weak-vs-strong section status
- `select_phrase_safe_window(...)` -> returns the actual source clip window for the planned `bar_count`

That keeps the resolver pure and deterministic while fixing the most dangerous current failure mode.

## Bottom line
For the first real-pair rerender, the safest improvement is **not** better mix logic. It is making the resolver stop treating coarse `section_0` spans as actual clips.

Use `beat_times` as truth, phrase boundaries as the extraction scaffold, section labels as weak hints, and always choose a target-length phrase-safe window before stretch. That will produce materially cleaner musical boundaries immediately.