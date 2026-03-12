# First-pass strategy: section extraction + beat snapping for renderer

_Date: 2026-03-11_

## Goal
Use the current `SongDNA` artifacts to extract clips that are:
- phrase-safe
- beat-snapped
- duration-clamped
- resistant to ugly cuts
- conservative about tempo mismatch

This is a **renderer-side first pass**, not a final planner architecture.

---

## Current repo reality
Useful now:
- `tempo.beat_times` is present and looks dense/stable enough for coarse alignment.
- `tempo.bpm` is usable as a section-level stretch target.
- `structure.phrase_boundaries_seconds` exists and is more useful than raw second cuts.
- `arrangement_plan.json` already specifies `bar_count`, `source_parent`, `source_section_label`, and transition type.

Not reliable yet:
- `structure.sections` is currently often just one giant full-song section.
- `structure.section_boundaries_seconds` may be empty.
- no true downbeat/bar-grid confidence yet.

**Implication:** first-pass extraction should trust:
1. beat times first,
2. phrase boundaries second,
3. section boundaries only when they exist and are plausible.

---

## Recommended extraction hierarchy
For each planned child section in `arrangement_plan.sections`:

1. Resolve the source parent (`A` or `B`).
2. Resolve the requested source region:
   - prefer the labeled `structure.sections` entry if it exists,
   - otherwise fall back to the nearest phrase-window in the source song.
3. Snap start/end to beats.
4. Expand or contract to a phrase-safe duration based on planned `bar_count`.
5. Only then stretch donor audio to target tempo.

This ordering matters: **choose musically safe boundaries before time-stretching**.

---

## Core heuristics

### 1) Treat beats as the canonical snap grid
Use `tempo.beat_times` as the only hard timing grid in v1.

Renderer helpers should effectively support:
- nearest beat to time `t`
- first beat at/after `t`
- last beat at/before `t`
- beat index range between two times

### 2) Infer bars only as grouped beats
Temporary bar heuristic:
- assume `4/4`
- `1 bar = 4 beats`
- `1 phrase = 8 or 16 beats` depending on context

For the current repo, the safest practical default is:
- use **16 beats = 4 bars** as the minimum phrase unit
- prefer **32 beats = 8 bars** for intros/builds/payoffs

### 3) Prefer section containment, but do not trust sections blindly
If `structure.sections` contains a labeled source section:
- start with that time range
- but re-snap its boundaries to nearby beats
- and reject it as a strict clip container if it is absurdly long (for now, anything over ~48 bars without internal segmentation)

If the section is missing or full-song length:
- derive the clip from phrase boundaries and beat counts instead.

---

## Clip selection strategy

### A. If a plausible source section exists
A source section is plausible when:
- duration is at least **8 beats**
- and at most about **64 bars** for current first-pass logic

Then:
1. snap section start to the next beat at/after section start
2. snap section end to the previous beat at/before section end
3. compute available beat count
4. if available beats exceed planned beats, choose a phrase-aligned subwindow inside it
5. if available beats are shorter than planned beats, either:
   - shorten the child section, or
   - allow pad/loop only for intro/drum material, not vocals/full mix by default

### B. If no plausible source section exists
Use phrase-boundary fallback:
1. find the phrase boundary nearest the intended source time anchor
2. use a clip of planned length in beats, starting from that boundary
3. if that would run past song end, back-shift to the latest phrase boundary that still fits
4. final start/end must land on beats

### C. Source time anchor when none is given
If the plan only says `source_section_label` and current structure is weak:
- `intro` -> early song phrase window
- `build` -> middle phrase window
- `payoff` -> later high-energy phrase window if energy supports it
- otherwise default to the first phrase-aligned window inside the requested section/source

---

## Duration clamp rules
Use the child plan’s `bar_count` as the target musical duration.

### Target duration
- `target_beats = bar_count * 4`
- `target_seconds_anchor = target_beats * 60 / anchor_bpm`

### Clamp behavior
After beat snapping, allow only small deviations from planned size:
- preferred deviation: **0 beats**
- acceptable deviation: **±1 beat** only for cleanup around edges
- avoid deviations larger than **±2 beats**

If the candidate clip is off by more than 2 beats:
- choose a different snapped start/end pair,
- or choose a different phrase window,
- or fall back to a hard section swap instead of forcing the clip.

### Phrase-safe defaults by transition role
- `intro`: prefer **8 bars**
- `build`: prefer **8 bars**
- `payoff`: prefer **16 bars**
- `bridge/breakdown`: **4 or 8 bars**

Do not cut odd lengths like 5.5 bars unless explicitly doing a special effect.

---

## Start/end boundary rules to avoid ugly cuts

### Safe starts
Prefer starts at:
1. section boundary snapped to beat
2. phrase boundary snapped to beat
3. strong onset beat near boundary
4. otherwise nearest beat

### Safe ends
Prefer ends at:
1. phrase end
2. low-vocal-density beat or post-transient beat
3. beat immediately before a strong transition

### Hard no-go cuts
Avoid cutting:
- in the middle of a vocal syllable
- within **~150 ms** after a large transient if another signal enters immediately
- within the final **1 beat** before an obvious cadence unless doing a `drop`/`lift`
- inside a sustained tonal tail unless an `echo_out`/fade is applied

### Mandatory edge treatment
Even beat-snapped clips should get micro-smoothing:
- default fade-in: **5–20 ms**
- default fade-out: **10–30 ms**
- section handoff fade: **0.5 to 2 beats** depending on transition type

This prevents clicky edits without smearing the groove.

---

## Beat snapping rules

### Boundary snap tolerance
When snapping a requested boundary to a beat:
- if nearest beat is within **120 ms**, snap to it directly
- if within **250 ms**, snap only if it improves phrase/bar alignment
- if farther than **250 ms**, preserve phrase intent and choose the next safer beat-aligned phrase boundary instead

### Transition-length defaults
- `cut`: 0 to **0.5 beat** pre-tail, new clip starts on beat
- `swap`: **1 to 2 beats** overlap max
- `blend`: **4 beats** default, up to **8 beats** if tempo/key are close
- `lift`: **2 to 4 beats**
- `drop`: optional **0.25 to 0.5 beat** vacuum before impact

### Section alignment rule
If a planned section starts at child bar `N`, its source clip should also begin on a source beat index divisible by 4 when possible. Do not claim this is true downbeat alignment; it is just the safest current heuristic.

---

## Tempo mismatch policy
Time-stretch should be conservative and decided **after** clip extraction.

### Stretch ratio
Use:
- `stretch_ratio = source_bpm / target_bpm`

### Safe operating bands
- **0.97 to 1.03**: effectively safe for almost anything
- **0.94 to 1.06**: okay for most stems/full-mix debug renders
- **0.92 to 1.08**: only if source is clean and role is limited
- outside **0.92 to 1.08**: avoid dense blending
- outside **0.88 to 1.12**: do not do sustained blend; use hard swap, percussion-only donor, FX, or skip

### Material-specific policy
- drums/percussion: most tolerant
- harmonic bed: moderate tolerance
- bass: cautious beyond **±4%**
- vocals/full mix: cautious beyond **±3%**, avoid beyond **±6%** unless very short

### Renderer decision rule
If tempo gap is too large for a clean blend:
1. keep the anchor source un-stretched,
2. use donor only for a short transition accent, percussion layer, or FX tail,
3. otherwise do a hard phrase-safe section handoff.

---

## Minimal algorithm for v1

For each child section:

1. Read `bar_count`, `source_parent`, `source_section_label`, `transition_in`, `transition_out`.
2. Load source song beat times and structure data.
3. Compute `target_beats = bar_count * 4`.
4. Resolve candidate source range:
   - labeled section if plausible,
   - else phrase-window fallback.
5. Snap candidate start/end to beats.
6. Adjust to nearest phrase-safe window matching `target_beats`.
7. Reject clip if:
   - resulting duration differs by > 2 beats,
   - stretch ratio is outside allowed band for the content,
   - cut lands in an obviously unsafe boundary zone.
8. If rejected, downgrade in this order:
   - shorter transition-only donor use,
   - percussion/harmonic-only donor use,
   - hard swap instead of blend.
9. Apply micro fades and transition envelope.
10. Stretch donor to anchor tempo and trim/pad exactly to the child section duration.

---

## Best thresholds to start with

### Musical unit defaults
- beats per bar: **4**
- minimum usable clip: **8 beats**
- preferred phrase unit: **16 beats**
- preferred section unit: **32 or 64 beats** depending on plan

### Snap / edit thresholds
- direct beat snap window: **120 ms**
- conditional beat snap window: **250 ms**
- micro fade-in: **5–20 ms**
- micro fade-out: **10–30 ms**

### Transition defaults
- cut pre-tail: **0–0.5 beat**
- swap overlap: **1–2 beats**
- blend overlap: **4 beats** default, **8 beats** max
- lift ramp: **2–4 beats**
- drop vacuum: **0.25–0.5 beat**

### Tempo/stretch thresholds
- ideal: **±3%**
- acceptable: **±6%**
- caution: **±8%**
- beyond **±12%**: no sustained blend

---

## Recommended smallest high-leverage next change
Add a renderer-side utility layer that derives a **beat-snapped clip window** from current `SongDNA` using:
- beat times
- phrase boundaries
- optional section label
- planned bar count

That is the highest-value first pass because it keeps render logic musical without pretending the current structure analysis is already bar-accurate.

---

## Bottom line
The best first-pass renderer should **not** cut by raw seconds and should **not** trust current section labels as fully authoritative.

Instead:
- use `beat_times` as the hard snap grid,
- use phrase boundaries as the main extraction scaffold,
- use section labels only as soft containers,
- clamp clips to planned beat counts,
- refuse aggressive blends when tempo mismatch is too large,
- and prefer a clean hard swap over a warped ugly overlay.

That is the safest path to clips that sound intentional with the repo’s current analysis quality.
