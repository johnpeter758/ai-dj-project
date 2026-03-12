# Minimal high-value synthetic test suite for upcoming render implementation

_Date: 2026-03-11_

## Goal
Define the **smallest** test suite that gives strong confidence in the first deterministic render path without requiring full musical evaluation.

Focus areas requested:
- deterministic audio tests
- duration conservation
- transition overlap sanity
- manifest correctness
- render fallback behavior

---

## Recommendation: keep v1 to **6 tests total**

Use **tiny synthetic fixtures** only:
- mono or stereo sine/noise/click WAVs
- exact sample counts at a fixed sample rate (e.g. 44_100)
- a tiny hand-authored `SongDNA` + `ChildArrangementPlan`
- no real songs in unit tests

That keeps tests deterministic, fast, and easy to debug.

---

## Priority 0: contract / non-audio tests

### 1) `test_resolve_render_plan_emits_contiguous_manifest`
**Why:** highest-value seam. If resolver math is wrong, audio tests become noisy and harder to interpret.

**Synthetic setup**
- Parent A + B `SongDNA` with:
  - `tempo_bpm` known and different enough to matter (e.g. 120 and 128)
  - `duration_seconds` fixed
  - `structure.sections` with explicit `label`, `start`, `end`
- Plan with 3 sections:
  - intro: 8 bars from A
  - build: 8 bars from B
  - payoff: 16 bars from A

**Assert**
- section count matches plan
- work orders sorted by `target_start_sec`
- each section target duration == `bar_count * beats_per_bar * 60 / anchor_tempo_bpm`
- total render duration == max work-order/section end time
- no negative times, no negative durations
- one base work order per section at minimum

**Implementation note**
This should be a pure unit test against `resolver.py`, no audio I/O.

---

### 2) `test_manifest_declares_single_low_end_and_foreground_owner_in_overlap`
**Why:** this is the fastest way to prevent muddy amateur blends before listening tests exist.

**Synthetic setup**
- Plan section with `transition_out="blend"` into next section
- Resolved transition region with overlap enabled

**Assert**
- manifest/resolved section explicitly declares:
  - `low_end_owner`
  - `foreground_owner`
  - overlap policy / vocal policy
- never both parents as low-end owner simultaneously
- if overlap exists, non-owner work order is marked background/suppressed/filtered

**Implementation note**
Even if the exact field names change, test the rule, not just the schema spelling.

---

## Priority 1: deterministic audio execution tests

### 3) `test_render_is_bitwise_or_samplewise_deterministic_for_same_inputs`
**Why:** most important render-execution guarantee.

**Synthetic setup**
- Use short generated fixtures (e.g. 2 seconds each):
  - A = 440 Hz sine
  - B = 660 Hz sine
- Same manifest rendered twice into arrays or WAVs

**Assert**
- output sample count identical
- output arrays are exactly equal if pipeline is pure numpy/soundfile
- if exact equality is unrealistic due to ffmpeg/loudnorm stage, compare the pre-master render buffer instead

**Implementation note**
Test the deterministic core renderer before any optional final loudness pass.

---

### 4) `test_render_preserves_expected_duration_after_trim_pad_and_stretch`
**Why:** duration drift is one of the most likely first-pass failures.

**Synthetic setup**
- One 8-bar target section at known anchor BPM
- Source clip intentionally longer than target in one case, shorter in another
- Include one donor stretch case (e.g. 128 -> 120)

**Assert**
- rendered section sample count == exact expected target sample count
- full render sample count == exact expected timeline duration
- trim/pad/stretch never changes planned section boundaries

**Implementation note**
Write this as arithmetic-first assertions on sample counts, not fuzzy “close enough” wall-clock checks.

---

## Priority 2: transition behavior tests

### 5) `test_blend_transition_overlap_window_is_bounded_and_energy_sane`
**Why:** overlap logic is where renderers get messy quickly.

**Synthetic setup**
- Outgoing clip = constant-amplitude sine
- Incoming clip = different-frequency sine
- One transition with declared overlap of e.g. 4 beats

**Assert**
- overlap region duration in samples matches manifest/work-order policy
- outside overlap window, only one primary section is active
- transition does not create a gross amplitude spike
  - pragmatic check: peak in overlap <= reasonable bound such as 1.1x or 1.2x the louder isolated source before mastering
- low-end/foreground ownership metadata remains consistent across the overlap

**Implementation note**
This does not need perceptual evaluation; it only needs to catch accidental double-summing and unbounded overlaps.

---

## Priority 3: fallback behavior tests

### 6) `test_renderer_falls_back_to_safe_single_source_when_requirements_fail`
**Why:** fallback behavior is essential for robustness and should be explicit from day one.

**Test at least one of these failure modes in v1:**
- donor stretch ratio exceeds allowed blend threshold
- missing donor stem / missing source section label
- overlap ownership conflict cannot be satisfied

**Synthetic setup**
- Deliberately create an invalid blend scenario

**Assert**
- renderer does **not** crash for recoverable cases
- manifest/warnings record fallback reason
- output collapses to safe behavior:
  - single-source section render, or
  - section swap with no blend, or
  - donor muted/background-only
- final duration still matches plan

**Implementation note**
This test matters more than adding multiple fancy transition tests.

---

## Optional 7th test if you want one extra guardrail

### `test_export_manifest_matches_audio_artifact`
After render, assert:
- exported WAV exists
- manifest path references correct output
- manifest duration/sample count/sample rate match actual file metadata

This is especially useful if render and export are separate modules.

---

## Minimal fixture strategy

Create a tiny `tests/fixtures/render_synth/` set or generate fixtures in-memory during tests:
- `sine_a.wav` — 440 Hz
- `sine_b.wav` — 660 Hz
- `kick_click.wav` — sparse impulses for boundary checks
- optional `noise_pad.wav` — low-level filtered noise for background bed tests

Preferred approach: **generate in test code** so fixtures never drift and exact sample counts are obvious.

---

## Suggested implementation order

1. **Resolver tests first**
   - manifest contiguity
   - ownership/overlap contract
2. **Renderer duration + determinism tests**
3. **One bounded blend transition test**
4. **One fallback test**

That order matches the architecture:
- planner -> resolver -> renderer -> export

---

## Practical notes for implementation

- Test **pre-master** buffers for determinism; `ffmpeg loudnorm` can complicate exact equality.
- Prefer **sample-count assertions** over floating-point seconds.
- Keep synthetic audio very short so tests stay fast.
- Use one canonical sample rate in tests.
- Avoid real stem separation in unit tests; mock manifest inputs instead.
- If transition DSP is added later, preserve these tests as coarse regression guards rather than making them overly specific.

---

## Bottom line
If only the smallest high-value suite is wanted, implement these **first 6 tests**:

1. `test_resolve_render_plan_emits_contiguous_manifest`
2. `test_manifest_declares_single_low_end_and_foreground_owner_in_overlap`
3. `test_render_is_bitwise_or_samplewise_deterministic_for_same_inputs`
4. `test_render_preserves_expected_duration_after_trim_pad_and_stretch`
5. `test_blend_transition_overlap_window_is_bounded_and_energy_sane`
6. `test_renderer_falls_back_to_safe_single_source_when_requirements_fail`

That is the minimum suite that will catch most early render failures while staying fast, synthetic, and deterministic.