# First-pass render recommendations for VocalFusion

_Date: 2026-03-11_

## Goal
Produce the best-sounding **deterministic** first render pass possible from the repo’s current state:
- existing `SongDNA` gives global tempo/key/sections/energy
- `arrangement_plan.json` gives coarse section sequence
- implementation target is practical Python using `librosa`, `numpy`, `soundfile`, and `ffmpeg`
- no codegen fantasies, no black-box remixing

---

## Bottom line
For the first renderer, do **not** try to make a full producer-grade mashup engine yet. The highest-quality deterministic pass is:

1. **Pick one parent as rhythmic anchor** for each planned section.
2. **Time-stretch only the donor material** to the anchor BPM.
3. **Pitch-shift donor by the smallest musically legal interval** before mixing.
4. **Render mostly stem-role substitutions, not full-song overlays**:
   - drums from one song + musical/vocal content from the other
   - avoid bass+bass and vocal+vocal overlap by default
5. **Place all entries/exits on section boundaries now, and on bars as soon as bargrid exists.**
6. **Use short deterministic transition templates**: fade, filter-in/out, drum swap, impact+noise, vocal tail echo.
7. **Mix conservatively** with headroom, sidechain-lite ducking, and loudness normalization at the end.

That will sound much better than raw overlay while staying implementable.

---

## Best practical first-pass render architecture

## 1) Render around roles, not whole tracks
Given current project state, the renderer should think in these roles:
- `drums`
- `bass`
- `harmonic` (keys/chords/pads/guitar/synth musical bed)
- `vocal`
- `fx/misc`

If real stems exist, use them.
If stems do not exist, fall back to:
- full mix as `harmonic+misc`
- optional HPSS split for rough `percussive` vs `harmonic`

### Default layering rules
These should be hard-coded in v1:
- allow `drums + vocal`
- allow `drums + harmonic`
- allow `drums + bass`
- allow `vocal + harmonic` only if harmonic source is sparse
- **disallow `bass + bass` by default**
- **disallow `lead vocal + lead vocal` by default**
- if conflict exists, prefer one dominant source and mute/duck the other

This single decision will improve quality more than fancy effects.

---

## 2) Time alignment strategy

## Recommendation
Use **section-level anchor BPM** and stretch the donor audio to it with `librosa.effects.time_stretch` or phase-vocoder utilities.

### Practical rule
For each planned child section:
- the section’s `source_parent` is the anchor by default
- any borrowed donor content is stretched to the anchor BPM
- if stretch ratio is outside about **0.92 to 1.08**, avoid dense/full-range layering
- if outside **0.88 to 1.12**, use only short fills, FX, or do a hard section swap instead of blend

### Why
Current repo has only global BPM, not bar-stable BPM. So the safest first render is:
- **one stretch ratio per section**
- not beat-by-beat warping
- not elastic micro-alignment

This keeps the output deterministic and avoids warbly artifacts.

### Implementation note
In Python:
- load with `librosa.load(..., sr=target_sr, mono=False if possible)`
- convert to mono only for analysis; keep stereo for render
- stretch donor stem/wave by `rate = donor_bpm / anchor_bpm`
- trim or pad after stretch to exact target section duration

---

## 3) Beat/bar alignment in current repo state

## Immediate recommendation
Until a true downbeat/bar tracker exists, use this hierarchy:

1. current section boundaries from `structure.sections`
2. beat times from `tempo.beat_times`
3. quantize transition points to nearest beat
4. assume 4/4 and estimate bars from beat groups of 4 only as a temporary heuristic

## Important constraint
Do **not** claim bar-accurate transitions yet. The current analysis is not reliable enough for that.

## Best temporary method
For each section boundary or transition point:
- find nearest beat time
- define section duration from beat counts rather than raw seconds when possible
- when crossfading two tracks, align fade start/end to nearest detected beats

That alone will sound more intentional than second-based cuts.

---

## 4) Key / pitch handling

## Recommendation
Use the smallest signed semitone shift that makes the donor harmonically compatible with the anchor.

### First-pass legal moves
Prefer these transpositions only:
- 0 semitones
- ±1 semitone
- ±2 semitones
- ±5 semitones
- ±7 semitones
- relative major/minor if mode logic supports it later

### Practical default
If global keys disagree:
- compute pitch-class distance
- choose smallest interval into anchor tonic
- reject intervals with absolute shift `> 3` for vocals/full mixes unless there is no better choice
- allow up to `5` semitones for drums/percussive or non-pitched FX only

### Mixing rule
- **vocals:** keep shifts within ±2 semitones when possible
- **harmonic stem:** ±3 is still usable
- **bass:** shift carefully; if large shift required, prefer using only one bass source

### Implementation note
Use `librosa.effects.pitch_shift(y, sr, n_steps=...)` offline and cache the result per source+semitone.
Caching matters because the renderer will reuse the same transforms.

---

## 5) Section-aware transition templates

The first renderer should not improvise transitions. It should pick from a tiny deterministic library.

## Recommended v1 transition types

### A. `cut`
Use for:
- strong downbeat change
- replacing full section with another

Method:
- 1-beat pre-tail fade of outgoing source
- start incoming section at beat-aligned boundary

### B. `blend`
Use for:
- intro/build overlays
- drums from one song under melodic bed of the other

Method:
- 2 to 8 beat equal-power crossfade
- low-cut outgoing bass if incoming bass enters
- optional LPF opening on incoming source

### C. `swap`
Use for:
- drums switch
- bass switch
- harmonic switch

Method:
- keep one role constant while switching one conflicting role over 1 to 2 beats

### D. `lift`
Use for:
- end of intro or pre-drop

Method:
- high-pass outgoing mix over 1 to 4 beats
- add gain ramp + short riser/noise asset if available
- mute bass in final half-beat before drop

### E. `drop`
Use for:
- payoff entry

Method:
- brief pre-drop silence or filtered vacuum (0.25 to 0.5 beat)
- hard-enter drums+bass on next beat/downbeat

### F. `echo_out`
Use for vocals

Method:
- send final vocal phrase to deterministic delay tail in ffmpeg
- fade tail under incoming section

## Recommendation for current arrangement plan
Map existing planner labels directly:
- `lift` -> filtered pre-transition ramp
- `blend` -> 4-beat equal-power overlap
- `swap` -> role swap with bass protection
- `drop` -> short pre-drop vacuum + full entry

---

## 6) Stem strategy that will sound best first

## If Demucs stems exist
Prioritize using:
- drums from whichever song has stronger groove / cleaner transient content
- bass from only one parent at a time
- vocals from only one parent at a time
- harmonic bed from the other parent when it does not clash

### Best practical combinations
1. **Anchor drums + donor vocal**
2. **Anchor drums+bass + donor harmonic hook**
3. **Donor drums + anchor vocal**
4. **Hard section handoff** rather than dense simultaneous playback

## If stems do not exist
Use HPSS as fallback:
- `librosa.decompose.hpss` or equivalent
- treat percussive as rough drums bed
- treat harmonic as rough musical bed

Then:
- avoid stacking two harmonic beds unless one is low-passed and low in gain
- never stack two full-band mixtures at equal level

---

## 7) Gain staging and loudness management

This is where many mashups fail.

## First-pass deterministic mix rules
- render all internal buses with **at least -6 dB headroom**
- normalize stems before summing to sensible working peaks, not to 0 dBFS
- use **equal-power fades** instead of linear fades
- reserve final limiting for the end only

## Suggested role targets before summing
Relative to a normalized section anchor:
- drums: 0 dB reference
- bass: -2 to -4 dB
- harmonic bed: -4 to -8 dB
- lead vocal: -1 to -3 dB
- donor overlay element: start around -6 dB and raise only if clean

## Conflict management
Implement simple deterministic ducking:
- when vocal active, duck harmonic bed 1 to 2 dB around 1–4 kHz via static EQ or broadband gain
- when kick-heavy drums active, duck bass 1 to 3 dB with simple envelope sidechain-lite
- when donor hook enters, duck anchor harmonic bed 2 to 4 dB

### Minimal implementation path
You do not need a full compressor first.
You can do:
- RMS envelope extraction
- smoothing
- gain curve multiplication

That is enough for basic sidechain-like behavior.

## Final loudness
After full render:
- peak normalize to about **-1.0 dBTP equivalent ceiling**
- then loudness normalize with ffmpeg `loudnorm` to around **-14 LUFS integrated** for debug renders
- keep an optional louder club target later, but not in v1

Using debug-safe loudness first prevents the renderer from sounding crushed.

---

## 8) Simple-but-musical first-pass section recipe

This is the most useful v1 recipe.

## Intro
- one dominant source only
- optionally filtered donor texture in background
- no dual bass
- low energy, wide headroom

## Build
- anchor drums continue
- donor harmonic or vocal enters gradually
- 4-beat blend
- rising filter/opening energy

## Payoff / hook
- strongest drums from one source
- one bass only
- one lead vocal/hook only
- optional secondary harmonic layer low in level and high-passed

## Breakdown
- strip drums or bass
- feature vocal phrase or harmonic motif from other parent
- use echo tail into next section

## Outro
- reduce to one source
- long filtered decay or simple fade

This produces a musically plausible child track even with limited analysis.

---

## 9) Recommended deterministic render pipeline

## Offline pipeline
1. Load arrangement plan + both `SongDNA` files
2. Resolve section source audio/stems
3. Decide anchor source per child section
4. Compute target section BPM and key
5. Precompute/cache transformed donor assets:
   - time-stretched versions
   - pitch-shifted versions
   - optional HPSS fallback splits
6. For each child section:
   - pick active roles
   - trim/pad assets to target duration
   - apply role gains
   - apply transition envelopes
   - sum to section stereo bus
7. Concatenate section buses with overlap regions
8. Apply master cleanup:
   - gentle high-pass if needed
   - peak control / limiter
   - ffmpeg `loudnorm`
9. Export WAV first, MP3/AAC second

## Cache key suggestion
Cache transformed audio by:
`source_path + stem_name + target_bpm + semitone_shift + target_sr`

That will save a lot of repeated offline work.

---

## 10) Specific tools best suited to each task

## Python / librosa / numpy / soundfile
Use for:
- loading analysis artifacts
- stretch ratio calculation
- pitch shifting
- beat/section alignment
- HPSS fallback
- envelope generation
- deterministic gain curves
- WAV writing

## ffmpeg
Use for:
- final loudnorm pass
- simple delay/reverb/filter tails if easier than custom DSP
- format conversion
- optional concat/filtergraph finishing

## Recommended split
Do the composition in Python, then do final loudness/export in ffmpeg.
That is the least painful path.

---

## 11) What not to do in first pass

Avoid these for v1:
- no real-time timestretch engine
- no source separation invented from scratch
- no chord-level reharmonization
- no beat-synchronous micro-chopping everywhere
- no dual-lead-vocal chorus unless explicitly designed
- no dual full-spectrum drop overlays
- no master-bus heavy compression to hide arrangement problems

The first pass should win by **selection and clean transitions**, not complexity.

---

## 12) Concrete implementation recommendations for this repo

## Highest-value next render module behavior
The first `core/render` implementation should expose something like:
- plan loader
- transform cache
- role allocator
- transition renderer
- section mixer
- master exporter

## Minimal additional data worth deriving at render time
From existing artifacts, derive:
- nearest-beat-aligned section boundaries
- estimated bar counts from beat groups of 4
- pitch shift recommendation between parents
- energy-based gain presets for each section

## Default fallback decisions
If information is missing:
- no stems -> HPSS split or single-source section swap
- bad harmonic match -> use drums/percussion only from donor
- large tempo mismatch -> hard section switch instead of blend
- large energy mismatch -> add ramp section, not immediate overlay

---

## 13) Recommended first milestone

A good first render milestone for `ai-dj-project` is:

> Given two analyzed songs and an `arrangement_plan.json`, render a stereo WAV where each section is either (a) a clean source-parent section, or (b) a controlled role-based blend with deterministic time-stretch, pitch-shift, transition envelopes, and loudness normalization.

If that works reliably, the next major upgrade is obvious:
- add real downbeat/bar grid
- move from section-only to bar/phrase-safe transitions
- use true stems consistently

---

## Final recommendation
If you want the first pass to sound as good as possible **right now**, prioritize these three things in order:

1. **Role-based layering rules** with no bass+bass or vocal+vocal conflicts
2. **Section-level tempo/key adaptation** of donor material
3. **Deterministic transition templates + conservative gain/loudness control**

That is the shortest path from current project state to something that sounds intentional instead of like two songs being played at once.
