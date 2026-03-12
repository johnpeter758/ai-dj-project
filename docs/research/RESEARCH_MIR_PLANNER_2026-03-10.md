# MIR + Planner Research Notes for VocalFusion

_Date: 2026-03-10_

## Why this matters
`ai-dj-project` already has the right top-level shape: **analysis -> planner -> render -> evaluation**. The current gap is not “more audio features”; it is **better bar/phrase/section certainty** so the planner can operate on musically stable units instead of raw seconds.

---

## 1) Practical MIR findings

### Tempo / beat tracking
- **librosa `beat_track`** is a solid baseline, but it is still a **dynamic-programming beat tracker** driven by onset strength and global tempo assumptions.
- Good for: quick BPM estimate, coarse beat grid, development baseline.
- Weak for VocalFusion use: half/double-time ambiguity, drift on sparse intros/outros, and no native downbeat/bar semantics.
- Relevant detail from docs: it estimates tempo from onset correlation, then picks beat peaks approximately consistent with tempo.

**Implication:** keep librosa as fallback, but do not let planner logic depend on raw librosa beats alone.

### Better rhythm / beat confidence
- **Essentia `RhythmExtractor2013`** returns:
  - bpm
  - beat ticks
  - confidence
  - bpm estimates distribution
  - beat intervals
- This is more useful than a single BPM number because planner quality depends on whether the beat grid is trustworthy.

**Implication:** store multiple rhythm hypotheses and confidence, not just one scalar BPM.

### Downbeat / bar tracking
- **madmom** provides a strong practical route for **joint beat + downbeat tracking**.
- Relevant doc detail: `RNNDownBeatProcessor` produces joint beat/downbeat activations, and `DBNDownBeatTrackingProcessor` applies a **dynamic Bayesian network** with `beats_per_bar`, tempo bounds, and transition control.
- This is exactly the missing layer between “tempo detected” and “usable bars/phrases for planning.”

**Implication:** for planner-first fusion, downbeat tracking is not optional. It is the anchor for bar indexing, phrase boundaries, and clean transition placement.

### Key detection
- Current repo key detection is a reasonable first pass: mean chroma + Krumhansl correlation + Camelot mapping.
- But for DJ/fusion work, plain 12-bin chroma averages are fragile when songs have:
  - strong percussion
  - modal ambiguity
  - section-dependent tonality
  - tuning offset
  - rap / sparse harmonic material
- **Essentia `KeyExtractor`** is stronger for production use because it operates on **HPCP** features and exposes:
  - key
  - scale
  - strength
  - profile families (`krumhansl`, `temperley`, `edma`, etc.)
  - tuning-aware handling

**Implication:** upgrade key estimation from “whole-song one-shot label” to **section-aware harmonic evidence** with confidence and alternate candidates.

### Structure segmentation
- The current `structure.py` uses tempogram novelty + peak-pick + fixed 4-bar phrase stepping.
- That is acceptable as a bootstrap, but it is not a robust song-structure model.
- **MSAF** exists specifically as a **Music Structure Analysis Framework** for segmentation research/prototyping and is a strong fit for evaluating multiple boundary/labeling methods without hand-rolling everything.

**Implication:** structure should be treated as a benchmarking problem: compare several segmentation methods and retain section boundaries only when they agree with beat/downbeat/bar evidence.

### Energy extraction
- Current RMS/beat-RMS is useful but incomplete.
- “Energy” for arrangement planning should separate at least:
  1. **loudness / intensity** (RMS, LUFS-ish proxies)
  2. **spectral brightness**
  3. **rhythmic density / onset density**
  4. **bass weight / low-frequency activity**
  5. **harmonic occupancy**
- Planner decisions need **relative energy over bars/phrases**, not just track-average RMS.

**Implication:** energy should become a multiband, bar-aligned feature sequence that the planner can shape into arcs.

---

## 2) What a planner-first fusion system should do

A usable fusion system should not ask: “are these two songs compatible overall?”
It should ask:

1. **Can I align their bar grids?**
2. **Which sections from A can functionally substitute for sections from B?**
3. **Where are the phrase-safe transition points?**
4. **Which bars carry vocals / bass / drums / hooks strongly enough to survive combination?**
5. **Can I design a child energy arc that feels intentional?**

That means the planner should operate on a representation closer to:

```text
Song
  -> bars
  -> phrases (4/8/16 bars)
  -> sections (intro / verse / hook / build / drop / bridge / outro or unlabeled structural states)
  -> per-bar feature vectors
  -> compatibility edges
```

This is more like **arrangement planning + constrained search** than classic mashuping.

---

## 3) Recommended analysis representation

For each song, build a canonical artifact that includes:

### Global
- duration
- primary BPM
- alternative BPM hypotheses
- primary key / mode
- alternative key hypotheses
- estimated time signature if possible
- confidence summary

### Beat / bar grid
- beat times
- downbeat times
- bar start/end times
- beat confidence
- downbeat confidence
- tempo stability / drift metrics
- half-time / double-time ambiguity flag

### Phrase / section structure
- phrase boundaries with confidence
- section boundaries with confidence
- section embeddings / feature summaries
- optional functional labels (`intro`, `verse`, `hook`, `drop`, etc.) only if confidence is high

### Per-bar features
- RMS / loudness proxy
- spectral centroid / brightness
- onset density
- low-band energy
- vocal activity probability
- drum activity probability
- bass activity probability
- harmonic stability / chord-change rate
- repetition score / self-similarity score

### Section summaries
- mean energy
- energy slope
- density score
- vocal dominance
- instrumental dominance
- tension/release proxy

---

## 4) Phrase and downbeat handling recommendations

### Non-negotiable rule
**Every planner action should snap to bars first, phrases second, seconds never.**

### Suggested strategy
1. Detect beats.
2. Detect downbeats / bar starts.
3. Build bar grid.
4. Aggregate features per bar.
5. Detect phrase boundaries from:
   - every 4/8 bars as prior
   - novelty peaks
   - downbeat-aligned energy changes
   - self-similarity boundary evidence
6. Only promote a phrase boundary when multiple signals agree.

### Why this is important
A lot of ugly mashups happen because transitions are placed at acoustically interesting points that are **not structurally legal**. For VocalFusion, phrase legality should outrank novelty.

### Practical heuristic
For each candidate boundary, score:

```text
boundary_score =
  0.35 * downbeat_strength
+ 0.25 * novelty_peak_strength
+ 0.20 * local_energy_change
+ 0.20 * self_similarity_boundary_strength
```

Then quantize accepted boundaries to the nearest strong downbeat/bar.

---

## 5) Compatibility scoring: what actually matters

A single overall compatibility score is useful for ranking candidates, but the planner really needs **factorized compatibility**.

### Recommended factors
- **Tempo compatibility**
  - exact / near-exact BPM after allowed stretch
  - penalty for extreme stretch
  - penalty for unstable tempo grid
- **Meter / bar compatibility**
  - same implied beats-per-bar
  - stable downbeat confidence
- **Harmonic compatibility**
  - same key / Camelot-neighbor / relative major-minor
  - transposition cost
  - ambiguity penalty when confidence is low
- **Structural compatibility**
  - intro↔intro, verse↔verse, hook↔hook, build↔drop patterns
  - section length match in bars
  - phrase periodicity match
- **Energy compatibility**
  - similar section energy level when layering
  - intentional energy contrast when replacing
- **Texture/stem compatibility**
  - avoid vocal-vocal conflict unless designed
  - avoid bass-bass masking
  - allow drum replacement and hook borrowing

### Example factorization

```text
global_pair_score =
  0.20 * tempo_score
+ 0.20 * harmonic_score
+ 0.20 * structure_score
+ 0.20 * energy_arc_score
+ 0.20 * stem_conflict_score
```

But for planning, use a **matrix**:

```text
compat(section_A_i, section_B_j)
compat(phrase_A_m, phrase_B_n)
compat(bar_A_x, bar_B_y)
```

This unlocks actual timeline search instead of picking songs by vibes.

---

## 6) Planning model that fits this repo best

For `ai-dj-project`, the strongest next architecture is:

### Stage A: Analysis emits planner-ready song DNA
Not just BPM/key/sections, but **bar-indexed evidence**.

### Stage B: Section-role inference
Even if labels are weak, infer coarse roles such as:
- low-energy intro
- groove verse
- pre-chorus/build
- hook/drop
- breakdown / bridge
- outro

This can be rule-based at first from bar-level energy, density, repetition, and vocal activity.

### Stage C: Compatibility graph
Build edges between candidate sections/phrases from song A and song B.
Each edge stores:
- harmonic/transposition cost
- tempo-stretch cost
- phrase-length alignment cost
- stem-conflict cost
- expected transition cleanliness

### Stage D: Child arrangement search
Search for a child timeline that satisfies:
- coherent macro arc
- phrase legality
- limited number of abrupt conflicts
- minimum hook payoff
- acceptable render cost

This can start as constrained dynamic programming or beam search, not ML.

### Stage E: Render validation loop
Before rendering full audio, run planner validation:
- are all transitions at bars/phrases?
- are energy jumps plausible?
- do vocals overlap illegally?
- is there excessive key shifting?
- is the child arc balanced?

---

## 7) Direct assessment of current `ai-dj-project` analysis code

### Good seeds already present
- `src/core/analysis/tempo.py`: reasonable baseline beat tracker using librosa.
- `src/core/analysis/key.py`: clear deterministic baseline with confidence + Camelot.
- `src/core/analysis/energy.py`: already bar-adjacent conceptually because it exposes beat-aligned energy.
- `src/core/analysis/structure.py`: explicit placeholder for phrase/section logic.

### Main limitations
- no downbeat/bar tracker
- phrase boundaries are hard-coded from 4/4 assumptions rather than inferred from evidence
- structure boundaries are not self-similarity-driven enough
- energy is too scalar for planning
- key is whole-song only and likely brittle on modern pop/rap/EDM hybrids
- planner package exists, but does not yet own compatibility graph / timeline search

---

## 8) Best next engineering steps

### Priority 1: Add bar-grid reliability
Implement or prototype:
- beat tracker baseline (keep existing)
- downbeat/bar tracker (madmom-style path preferred)
- bar object generation
- bar confidence metrics

**Why first:** without this, phrase and transition planning are not trustworthy.

### Priority 2: Upgrade structure from heuristic to evidence fusion
Add a structure pass that combines:
- self-similarity / novelty boundaries
- downbeat-aligned energy changes
- repetition cues
- fixed 4/8/16-bar priors

Output:
- `bar_segments`
- `phrase_segments`
- `section_segments`
- confidence for each boundary

### Priority 3: Replace single energy score with bar-level planner features
At minimum per bar:
- loudness proxy
- low-band energy
- brightness
- onset density
- vocal probability

Then derive phrase/section summaries.

### Priority 4: Make key estimation section-aware
Upgrade artifact to include:
- global key candidates
- section key candidates
- confidence per section
- suggested transposition cost between sections

This matters because many usable fusions are locally compatible even when whole-song key labels disagree.

### Priority 5: Build compatibility as graph data, not one score
Create planner inputs like:
- `section_pair_scores`
- `phrase_pair_scores`
- `transition_candidates`
- `illegal_overlap_flags`

That gives the planner something actionable.

### Priority 6: Start with a deterministic planner before any learning
First planner version should be rule/constrained-search based:
- choose anchor song
- borrow compatible sections from donor song
- enforce phrase-safe transitions
- target a desired energy arc
- keep vocal conflicts below threshold

This is the shortest path to hearing whether the architecture is musically right.

---

## 9) Suggested evaluation metrics

Use evaluation that matches planner goals:

### Analysis metrics
- BPM error
- downbeat F-measure / bar accuracy
- boundary hit rate for phrase/section boundaries
- key accuracy / Camelot-neighbor accuracy

### Planner metrics
- % transitions on bar starts
- % transitions on phrase boundaries
- average section compatibility score
- vocal conflict rate
- bass masking rate
- energy arc smoothness / intentionality score

### Human evaluation
Rate each child on:
- structural coherence
- transition cleanliness
- harmonic fit
- groove retention
- replayability
- “feels like a new child song”

---

## 10) Recommended implementation order inside this repo

1. `core/analysis/tempo.py`
   - keep current baseline
   - extend to multi-hypothesis + tempo stability metrics
2. `core/analysis/bargrid.py` (new)
   - downbeats, bars, confidence
3. `core/analysis/structure.py`
   - self-similarity + novelty + bar-aligned boundary fusion
4. `core/analysis/energy.py`
   - convert to per-bar multidimensional energy/features
5. `core/analysis/key.py`
   - section-aware candidates and stronger confidence model
6. `core/planner/compatibility.py` (new)
   - factorized pair scoring
7. `core/planner/timeline.py` (new)
   - deterministic phrase-level search
8. `core/evaluation/`
   - planner-specific validation metrics

---

## 11) Bottom line

The repo is already pointed in the correct direction. The highest-leverage move is **not** more generation or more UI; it is building a **reliable bar/phrase/section representation** and then a **planner that searches over those units**.

If I were setting the next milestone, it would be:

> **Milestone:** given two songs, emit a planner-ready JSON with beats, downbeats, bars, phrase boundaries, section boundaries, bar-level energy vectors, and factorized compatibility edges.

That milestone would turn VocalFusion from “audio analysis demo” into “actual arrangement system.”

---

## Reference anchors used
- `librosa.beat.beat_track` docs: dynamic-programming beat tracking from onset strength + tempo estimation.
- Essentia `RhythmExtractor2013` docs: BPM, ticks, confidence, estimate distribution, beat intervals.
- Essentia `KeyExtractor` docs: HPCP-based key/scale/strength estimation with multiple profile types and tuning-aware options.
- madmom downbeat docs: RNN beat/downbeat activations plus DBN downbeat tracking with `beats_per_bar` and tempo/transition controls.
- MSAF README/docs: framework for computational music structure analysis and method comparison.
