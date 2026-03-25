# Simple Blend Baseline Report

Date: 2026-03-19
Target: a **decent-first** two-song blend engine that prefers sane musical results over ambition.

## Bottom line

The best baseline is **not** a full "fusion" engine. Borrow a small stack:

1. **BPM/key analysis** from `tfriedel/trackanalyzer`
2. **Conservative downbeat/overlap transition logic** from `akriptonn/aidjsystem`
3. **Very limited section/entry heuristics** from the local reference `jpemslie/vocal-fusion-ai`

That gives a practical v0:
- choose a backbone song
- detect BPM/key for both
- time-stretch donor conservatively to backbone tempo
- only allow harmonic-compatible or small-shift pairs
- enter donor on a **4- or 8-bar boundary**
- use **one short, conservative overlap window**
- prefer **single contiguous donor cluster** or one intentional handoff, not constant switching

---

## Strongest code to borrow

## 1) `akriptonn/aidjsystem`
Repo: `https://github.com/akriptonn/aidjsystem`
Main file: `MusicPlayer/main.py`

### Why it matters
This is the clearest web-accessible example I found of a **simple DJ-style transition system** that actually encodes:
- tempo mismatch handling
- downbeat-aware entry
- a finite overlap window
- EQ-based transition cleanup

It is crude, but for a baseline blend engine that is a feature, not a bug.

### Exact pieces worth borrowing

#### A. Tempo-match before transition
File: `MusicPlayer/main.py`
Function: `change_audioseg_tempo(...)`

Useful idea:
- stretch the incoming song to the currently playing song's BPM before transition
- keep this as a **single global tempo ratio**, not clever microtiming

Why borrow:
- exactly aligned with a decent-first baseline
- deterministic
- low conceptual complexity

Caution:
- use your own renderer/timeline, not this UI/player implementation

#### B. Cache and detect beat/downbeat positions
File: `MusicPlayer/main.py`
Logic in `addFile(...)` around beat/downbeat extraction
Relevant lines observed:
- `madmom.features.beats.RNNBeatProcessor`
- `madmom.features.beats.BeatTrackingProcessor`
- `madmom.features.downbeats.DBNDownBeatTrackingProcessor`
- cached `bpm`, `beatpos`, `downbeat` in `storage.csv`

Useful idea:
- compute and cache:
  - BPM
  - beat positions
  - downbeat positions
- a baseline blend engine should operate on **downbeat/phrase-safe candidate entries**, not raw seconds

Why borrow:
- this is exactly the minimum timing metadata you need for sane blends

#### C. Conservative overlap scheduling from downbeat counts
File: `MusicPlayer/main.py`
Function: `callback(...)`
Related prep in `addFile(...)`

Useful logic:
- fade outgoing song toward its ending region
- start the incoming song at a predetermined downbeat-relative location
- transition is triggered at fixed offsets from the current song's later downbeats
- overlap is finite and predictable, not free-form

Most reusable concept:
- choose transition entry/exit relative to **known downbeats near phrase boundaries**
- encode transition templates like:
  - start donor 8 bars before backbone phrase end
  - low-end duck outgoing backbone first
  - bring donor in on bar 1 of a phrase
  - keep overlap short

Why borrow:
- this is the strongest directly useful baseline transition scaffold I found

### What to reject from this repo
- `getNextMusic(...)`: webcam/crowd-energy control is irrelevant
- full GUI/player architecture: not useful for your offline render pipeline
- exact hardcoded magic numbers (`18th from end`, `12th from end`, 10 s fades) should not be copied literally
- key handling here is weak; it displays key info but does not provide a robust harmonic planner

### Verdict
**Borrow the transition skeleton, not the product.**

---

## 2) `tfriedel/trackanalyzer`
Repo: `https://github.com/tfriedel/trackanalyzer`
README: `README.txt`
Relevant source tree:
- `TrackAnalyzer/KeyFinder.java`
- `TrackAnalyzer/KeyClassifier.java`
- `at/ofai/music/beatroot/BeatRoot.java`
- `at/ofai/music/worm/TempoInducer.java`

### Why it matters
This repo is not a blend engine, but it is a good **baseline analysis dependency** for one.

The README explicitly states:
- key analyzer is a port of **KeyFinder**
- BPM estimation is based on a modified **BeatRoot**
- intended for DJ/harmonic mixing workflows

### Exact pieces worth borrowing

#### A. Key detection path
Files:
- `TrackAnalyzer/KeyFinder.java`
- `TrackAnalyzer/KeyClassifier.java`

Useful idea:
- use a DJ-oriented key detector that is already framed for harmonic mixing
- map output into your own Camelot / semitone-shift compatibility layer

Why borrow:
- a decent-first baseline should reject or heavily penalize bad key pairs before arrangement gets fancy

#### B. BPM analysis path
Files:
- `at/ofai/music/beatroot/BeatRoot.java`
- `at/ofai/music/worm/TempoInducer.java`

Useful idea:
- use robust DJ-style tempo estimation as a first-class preflight signal

Why borrow:
- your baseline mode should win first by choosing pairs and shifts conservatively, not by heroically fixing bad pairs later

### What to reject from this repo
- using the full Java app directly as your blend engine
- any attempt to inherit its whole CLI/app structure

### Verdict
**Borrow analysis concepts / possibly port logic, not the app.**

---

## 3) `jpemslie/vocal-fusion-ai` (already local reference)
Local path: `/Users/johnpeter/Code/ai-dj-project/references/jpemslie_vocal_fusion_ai/fuser.py`

### Why it matters
This is the most relevant reference for your current codebase, but only in **very selective** slices. It contains some sane baseline heuristics buried inside a monolith.

### Exact pieces worth borrowing

#### A. BPM detection with consensus / octave correction
Function: `detect_bpm(...)`

Why useful:
- multi-estimate consensus
- explicit half/double correction
- better than single-call `librosa.beat_track` for baseline robustness

Baseline use:
- keep as preflight / analysis layer
- still apply a hard acceptable stretch cap afterward

#### B. Multi-window weighted key detection
Function: `detect_key(...)`

Why useful:
- looks at full track + temporal windows
- better than trusting intro-only harmonic material
- suitable for baseline compatibility screening

#### C. First-entry / phrase-safe placement heuristic
Function: `_beat_align(...)`

Why useful:
- explicitly snaps the first stretched vocal onset to the nearest **measure boundary**
- this is one of the few directly reusable phrase-safe entry ideas in the repo

For a baseline blend engine, adapt it into:
- detect donor entry onset
- snap to nearest 4- or 8-bar backbone boundary
- reject if no stable boundary match is available

#### D. Section scoring / assignment only as a weak prior
Functions:
- `_detect_all_sections(...)`
- `_score_section_pair(...)`
- `_arrange_sections(...)`

Why only partially useful:
- the important idea is **section-aware assignment** instead of random overlay
- for a baseline mode, the right simplification is:
  - backbone keeps macro structure
  - donor contributes only one or two safe windows
  - donor windows should be chosen from chorus/build/verse-like regions with compatible energy/density

Do **not** port the full assignment system as-is.

#### E. Sequential section placement without accumulation
Function: `_stitch_sections(...)`

Most useful idea:
- avoid sloppy additive overlaps everywhere
- place chosen donor windows **sequentially** into intended slots
- use short fades at boundaries

This matches the decent-first philosophy.

### What to reject from this repo

#### Hard reject
- `fuse(...)` as an architectural template
  - too monolithic
  - too many responsibilities
  - too much hidden coupling

- `_groove_quantize(...)`
  - explicitly disabled in the file
  - comment says the implementation is broken and click-prone

- the big auto-mix / DSP stack as baseline requirements
  - `_iterative_mix(...)`
  - `_produce_vocal_for_beat(...)`
  - room-IR / mastering / fancy carve systems

These are not baseline blend essentials; they are complexity multipliers.

#### Soft reject / use only later
- `_arrange_sections(...)` full Hungarian assignment approach
  - too ambitious for v0
  - risks medley behavior instead of one clear backbone song

### Verdict
**Mine small heuristics; reject the monolith.**

---

## Recommended baseline engine design from these findings

## Core contract
A decent-first blend mode should do only this:

1. **Pick a backbone** track
2. **Analyze** both tracks for BPM, key, beats, downbeats, coarse sections
3. **Reject bad pairs early**
   - large BPM mismatch after conservative stretch cap
   - bad key compatibility unless small shift fixes it
4. **Select one donor window**
   - preferably chorus/build/verse-like, not random
   - only if it can enter on a phrase-safe boundary
5. **Snap donor entry to backbone phrase/downbeat**
6. **Use one short overlap recipe**
   - e.g. 4-8 bars max
   - low-end ownership stays singular
   - outgoing layer ducked/carved, not both full-force
7. **Return to backbone or stay on donor**, but do not keep pinballing

## Concrete borrow plan

### Borrow directly / adapt closely
- From `aidjsystem`:
  - downbeat caching concept
  - pre-transition tempo matching
  - fixed transition templates driven by downbeat indices
  - finite overlap window

- From `trackanalyzer`:
  - DJ-oriented BPM/key analysis concepts
  - harmonic-mixing framing for preflight

- From `jpemslie`:
  - `detect_bpm(...)`
  - `detect_key(...)`
  - `_beat_align(...)`
  - only the simplest section-choice heuristics from `_detect_all_sections(...)` / `_score_section_pair(...)`

### Do not borrow directly
- any GUI/player code
- webcam/crowd logic
- full monolithic fusion orchestration
- disabled groove-quantization code
- heavy mastering and room-model code as prerequisites for baseline success

---

## Practical implementation guidance for VocalFusion

## Best v0 baseline policy

### Pair admissibility
- tempo stretch cap: ideally `<= 1.06x`, absolute max `<= 1.10x`
- key shift cap: prefer `<= 2 semitones`, absolute max `<= 4`
- if either fails, reject the blend or fall back to backbone-only

### Entry safety
- only start donor on a detected downbeat / bar boundary
- prefer 8-bar phrase starts
- if phrase confidence is weak, do not blend; do a simpler late swap or no-op

### Overlap policy
- default overlap: **4 bars**
- max overlap: **8 bars**
- one low-end owner at all times
- if vocals compete, donor should enter filtered/background first or wait

### Arrangement policy
- keep backbone chronology intact
- donor gets one contiguous feature block, not many re-entries
- baseline goal is **"sounds like one sane transition or feature blend"**, not "child-song synthesis"

---

## Final recommendation

If you want a **strong decent-first baseline**, the clearest path is:

- use `trackanalyzer`-style BPM/key screening,
- use `aidjsystem`-style downbeat-triggered transition scheduling,
- use only the smallest useful `jpemslie` heuristics for measure-safe entry and very coarse donor-window choice.

That combination is much more promising than copying any single repo wholesale.

## Short answer

### Worth borrowing most
1. `akriptonn/aidjsystem` → `MusicPlayer/main.py`
   - `change_audioseg_tempo(...)`
   - beat/downbeat extraction in `addFile(...)`
   - downbeat-relative transition triggering in `callback(...)`

2. `tfriedel/trackanalyzer`
   - `TrackAnalyzer/KeyFinder.java`
   - `TrackAnalyzer/KeyClassifier.java`
   - `at/ofai/music/beatroot/BeatRoot.java`

3. `jpemslie/vocal-fusion-ai` → `fuser.py`
   - `detect_bpm(...)`
   - `detect_key(...)`
   - `_beat_align(...)`
   - light ideas only from `_detect_all_sections(...)`, `_score_section_pair(...)`, `_stitch_sections(...)`

### Reject
- `aidjsystem` webcam / GUI / energy-selection product logic
- `jpemslie` monolithic `fuse(...)`
- `jpemslie` `_groove_quantize(...)`
- treating advanced DSP/mastering as baseline blend prerequisites
