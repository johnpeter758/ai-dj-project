# Planner / Material-Selection Web Harvest Report

Date: 2026-03-19
Scope: public GitHub/web-accessible code relevant to VocalFusion planner/material selection: phrase segmentation, section selection, transition heuristics, mashup/DJ arrangement logic, compatibility scoring.

## Executive take

Best reusable value is **not** in full auto-mix monoliths. The strongest planner-port candidates are:

1. **MSAF / CBM + OLDA + Foote/SF** for **bar/phrase/section boundary proposals** and section-likelihood priors.
2. **mir-aidj/transition-analysis** for **beat-aligned cue extraction** and **alignment-derived transition windows**.
3. **djkr8** for small, clean, **constraint-friendly compatibility primitives** (Camelot distance, BPM half/double-time handling, transition-quality enums/scalars).

Most “AI automix” repos are useful only as heuristic inspiration. They are usually monolithic, DJ-crossfade oriented, and weak on child-song arrangement design.

---

## 1) Strong adopt candidate: MSAF (Music Structure Analysis Framework)

- Repo: https://github.com/urinieto/msaf
- Why it matters: mature modular segmentation framework with multiple interchangeable algorithms. Good fit for planner-first architecture because it separates:
  - feature extraction
  - boundary detection
  - hierarchical/flat segmentation
  - labeling hooks

### Exact files / functions worth mining

#### A. Structural novelty boundary proposals
- File: `msaf/msaf/algorithms/sf/segmenter.py`
- Functions:
  - `embedded_space(X, m, tau=1)`
  - `compute_ssm(X, metric="seuclidean")`
  - `compute_nc(X)`
  - `pick_peaks(nc, L=16, offset_denom=0.1)`
  - `Segmenter.processFlat()`
- Why adopt:
  - Clean novelty-curve boundary candidate generation.
  - Easy to port into a planner pre-pass that proposes **possible section-change bars**.
  - `pick_peaks(...)` is especially useful as a modular boundary-candidate generator, not just final segmentation.
- What to adopt:
  - novelty-curve based **boundary proposal scores**
  - adaptive threshold peak-picking
  - embedded recurrence features as one signal among many
- What not to adopt wholesale:
  - do not make this the only segmentation truth; use as a candidate source feeding planner scoring.

#### B. Bar-length prior via dynamic programming
- File: `msaf/msaf/algorithms/cbm/CBM_algorithm.py`
- Functions:
  - `compute_cbm(...)`
  - `compute_all_kernels(max_size, bands_number=None)`
  - `conrrelation_cost(...)`
  - `penalty_cost_from_arg(penalty_func, segment_length)`
  - `possible_segment_start(idx, min_size=1, max_size=None)`
- Why adopt:
  - This is the single best public code hit for **planner-relevant section sizing priors**.
  - `compute_cbm(...)` does dynamic programming over segment candidates using autosimilarity + size penalties.
  - `penalty_cost_from_arg(...)` explicitly encodes useful pop priors: prefer 8-bar segments, then 4-bar, then even lengths.
- Best direct port ideas for VocalFusion:
  - Replace or augment current section selection with **DP over bar windows** using:
    - local cohesion score
    - repetition score
    - compatibility score
    - seam risk
    - explicit size prior (8/16/32-bar preference)
  - Port the idea, not the exact API.
- Concrete planner adaptation:
  - New module idea: `src/core/planner/segment_dp.py`
  - Scoring term per candidate window `w=(start_bar,end_bar)`:
    - `window_score = self_similarity + motif_cohesion + groove_stability + role_fit - seam_risk - overcrowding_penalty - size_penalty`
  - Use CBM-style traceback to choose the best song-scale segmentation or donor-window decomposition.
- Caution:
  - Raw autosimilarity alone will overvalue repetition and under-model musical function. Must be fused with role priors (intro/verse/build/payoff/outro).

#### C. Learned / feature-rich segmentation for phrase-scale proposals
- File: `msaf/msaf/algorithms/olda/segmenter.py`
- Functions:
  - `features(file_struct, annot_beats=False, framesync=False)`
  - `gaussian_cost(X)`
  - `clustering_cost(X, boundaries)`
  - `get_k_segments(X, k)`
  - `get_segments(X, kmin=8, kmax=32)`
  - `get_num_segs(duration, MIN_SEG=10.0, MAX_SEG=45.0)`
  - `Segmenter.processFlat()` / `processHierarchical()`
- Why adopt:
  - Good template for **multi-feature section proposal generation**.
  - `features(...)` explicitly combines timbre, chroma, repetition, and time features; this is close to what VocalFusion wants before planner ranking.
  - `get_segments(...)` chooses segmentation count by cost rather than hard-coding one split.
- What to adopt:
  - the idea of generating phrase/section candidates from a **stacked feature representation**
  - duration-aware segment count priors
  - optional hierarchical outputs
- What to reject:
  - do not port MSAF’s full feature stack as-is if it drags in old infrastructure or frame semantics mismatched to current bar-aware planner.

### Adopt / reject summary for MSAF
- **Adopt:** yes, selectively and strongly.
- **Priority:** very high.
- **Port style:** copy ideas + small functions into modular planner utilities, not framework integration.

---

## 2) Strong adopt candidate: transition-analysis (real DJ transition reverse engineering)

- Repo: https://github.com/mir-aidj/transition-analysis
- Why it matters: this repo is one of the most concrete hits for **transition window discovery** using aligned beats/features from real mixes.

### Exact files / functions worth mining

#### A. Alignment-derived cue points from subsequence DTW
- File: `transition-analysis/scripts/alignment.py`
- Functions:
  - `alignment(mix_id, features=['chroma', 'mfcc'], key_invariant=True)`
  - `extract_feature(path, feature_names)`
- Why adopt:
  - `alignment(...)` uses **subsequence DTW** across beat-synchronous features and optionally circular chroma shifts for key invariance.
  - This is directly relevant to finding **cross-parent material alignments** and candidate transition anchors.
- Best direct port ideas:
  - Use beat-synchronous chroma + MFCC windows to align candidate donor windows between parent A/B.
  - Add a key-invariant variant by rotating chroma when comparing windows.
  - Use the resulting path cost as a **transition viability** or **material substitution** score.

#### B. Extracting stable cue-in / cue-out zones from a warp path
- File: `transition-analysis/lib/cue.py`
- Function:
  - `find_cue(wp, cue_in=False, num_diag=32)`
- Why adopt:
  - Elegant small function.
  - Finds earliest/latest sufficiently diagonal region in a DTW warp path.
  - This is highly portable for identifying **stable overlap entry/exit bars** instead of arbitrary seam points.
- Best direct port ideas:
  - Adapt to bar-domain path instead of beat-index path.
  - Use returned cue zones to constrain where lead/support handoffs may occur.
  - Add to planner diagnostics as `warp_diagonal_stability` or `alignment_stability`.

#### C. Beat-aligned transition reconstruction heuristics
- File: `transition-analysis/scripts/reproduce_mixing.py`
- Functions:
  - `estimate_eq_curve(...)`
  - `extract_curves(...)`
  - `cvxopt_eq3(...)`
  - `cvxopt_xfade(...)`
- Why adopt:
  - Not for full rendering pipeline, but useful conceptually:
    - align source windows to a target transition region
    - derive per-band/per-track fade curves
- For VocalFusion:
  - adopt only as a possible **transition-control representation** for render metadata, not as planner core logic.

### Adopt / reject summary for transition-analysis
- **Adopt:** yes, especially `alignment.py` + `lib/cue.py` ideas.
- **Priority:** high for transition scoring.
- **Reject:** full dependency-heavy reproduction pipeline unless later needed for benchmark analysis.

---

## 3) Strong adopt candidate: djkr8 (clean compatibility primitives + constrained sequencing)

- Repo: https://github.com/schoi80/djkr8
- Why it matters: not mashup generation, but the code is small, explicit, and planner-portable. Good source for clean compatibility functions.

### Exact files / functions worth mining

#### A. Harmonic compatibility / transition class scoring
- File: `djkr8/src/djkr8/camelot.py`
- Functions:
  - `get_transition_quality(key1, key2)`
  - `is_energy_boost(key1, key2)`
  - `parse_camelot_key(key)`
  - `get_hour_distance(hour1, hour2)`
  - `is_harmonic_compatible(key1, key2, level=...)`
  - `get_compatible_keys(key, level=...)`
- Why adopt:
  - Small, readable, deterministic.
  - Gives categorical + scalar compatibility outputs.
  - The distinction between `SMOOTH`, `ENERGY_BOOST`, and `VIOLATION` maps nicely to planner intent.
- Best direct port ideas:
  - add planner feature: `harmonic_transition_type`
  - use `quality_score` as a prior, not a final decision
  - allow role-dependent tolerance: stricter for exposed lead swaps, looser for filtered support layers

#### B. BPM compatibility with half/double-time support
- File: `djkr8/src/djkr8/bpm.py`
- Functions:
  - `bpm_compatible(...)`
  - `get_bpm_difference(...)`
- Why adopt:
  - Exactly the kind of small primitive worth porting.
  - Half/double-time logic is important for real-world candidate pools.
- Best direct port ideas:
  - use for shortlist admissibility
  - compute normalized `tempo_relation_class` (`direct`, `half`, `double`, `far`)

#### C. Constraint-based sequencing ideas
- File: `djkr8/src/djkr8/optimizer.py`
- Class / methods:
  - `PlaylistOptimizer.optimize(...)`
  - inner helper `is_energy_flow_valid(i, j)`
  - `_extract_result(...)`
  - `_reconstruct_path_with_dummy(...)`
- Why adopt:
  - The CP-SAT playlist model is overkill for direct song fusion, but the structure is useful:
    - hard constraints for admissibility
    - soft objective for transition quality
    - energy-flow bounds
- Best direct port ideas:
  - not the solver itself first
  - instead borrow the distinction between:
    - hard constraints: BPM/key/seam/stretch/collision
    - soft objective: transition quality + arc fit + donor usefulness
- Specific idea for VocalFusion:
  - build a compact DP/beam-search section chain scorer using djkr8-style hard/soft separation.

### Adopt / reject summary for djkr8
- **Adopt:** yes, selectively.
- **Priority:** medium-high.
- **Reject:** full playlist optimizer as product architecture; useful as a scoring-pattern reference, not core engine.

---

## 4) Partial heuristic source only: traktor-harmony

- Repo: https://github.com/0xf4b1/traktor-harmony

### Exact file / function
- File: `traktor-harmony/harmonize.py`
- Function:
  - `calculate_transition_score(first, second)`

### Assessment
- Value:
  - demonstrates very simple transition scoring = wheel key score minus quadratic BPM penalty.
  - useful as the most minimal baseline sanity formula.
- Weakness:
  - greedy playlist picking
  - no structure awareness
  - no section/phrase logic
  - no timbral/groove modeling
- Recommendation:
  - **Do not port directly**.
  - If needed, keep only as a mental baseline for “simple score should beat this.”

---

## 5) Heuristic inspiration but mostly reject: AutoMixSuite

- Repo: https://github.com/congnghetinhtu/AutoMixSuite
- Why inspected: contains phrase, downbeat, transition-style, and compatibility heuristics.
- Big problem: single huge monolithic `automix.py` with many cross-coupled heuristics. Bad fit for VocalFusion’s modular planner-first architecture.

### Exact functions worth reading, but not copying wholesale

#### A. Beat/meter/downbeat/phrase heuristics
- File: `AutoMixSuite/automix.py`
- Functions:
  - `_score_meter_hypothesis(...)`
  - `_detect_downbeats(...)`
  - `_detect_phrases(...)`
  - `_calculate_beat_strengths(...)`
  - `_calculate_beat_confidence(...)`
- Value:
  - explicit multi-feature downbeat/beat-confidence heuristics
  - phrase detection via downbeat grouping into 8/16/32-bar units
- What to adopt:
  - only the idea that beat/downbeat confidence should influence planner trust in bar boundaries
  - phrase candidates should be 8/16/32-bar aligned by default
- What to reject:
  - current implementation is heuristic-heavy, broad, and not obviously benchmarked for robust segmentation.

#### B. Structure-aware transition heuristics
- Functions:
  - `_find_vocal_transition_points(...)`
  - `_find_structure_transition(...)`
  - `_snap_to_beat(...)`
- Value:
  - useful reminder that transitions should map **outro→intro**, **outro→verse**, etc., not just nearest time points.
- What to adopt:
  - role-pair scoring table: `outro→intro`, `outro→verse`, `bridge→intro`, etc.
  - beat-snapping of seam candidates
- What to reject:
  - hardcoded “boring intro” skips and ad hoc branch tree as-is
  - should become planner score features, not imperative transition rules

#### C. Compatibility scoring
- Function:
  - `calculate_compatibility(track1, track2)`
- Value:
  - decent menu of score components: tempo, key, energy, spectral, timbral, rhythm
- What to adopt:
  - component decomposition only
- What to reject:
  - exact weights / genre-specific hacks / monolithic track dict assumptions

#### D. Transition-style taxonomy
- Function:
  - `_determine_transition_style(track1, track2)`
- Value:
  - nice concept: classify transition intent before rendering
- Adopt idea:
  - planner can tag seams as `smooth_blend`, `energy_punch`, `harmonic_layer`, `palate_cleanser`
- Reject implementation:
  - too tied to crossfade DJ behavior and monolith state.

### Adopt / reject summary for AutoMixSuite
- **Adopt:** conceptually only, in small rewritten modules.
- **Priority:** low-medium inspiration.
- **Reject:** direct integration or large code port.

---

## 6) Heuristic inspiration but mostly reject: SongMix

- Repo: https://github.com/congnghetinhtu/SongMix
- File: `SongMix/automix.py`
- Key signal: another monolithic automix implementation with similar vocal transition logic.
- Assessment:
  - mostly redundant with AutoMixSuite, but smaller.
  - still oriented toward DJ-like transitions, not child-song planning.
- Recommendation:
  - **Reject for direct port**.
  - Only useful if you want a second source confirming that public automix repos converge on:
    - vocal-aware seam selection
    - beat snapping
    - intro/outro heuristics

---

## What to implement in VocalFusion from this harvest

## A. New modular planner pieces worth building now

### 1. Boundary proposal layer
Create a small boundary proposal module inspired by MSAF SF/CBM/OLDA:

- Suggested file: `src/core/planner/boundary_proposals.py`
- Inputs:
  - bar-synchronous chroma/timbre/rhythm/groove features
  - repetition/autosimilarity matrices
- Outputs:
  - per-bar boundary probability / novelty score
  - candidate phrase windows (8/16/32 bars)
  - confidence score per proposal

Port ideas:
- `pick_peaks(...)` style adaptive peak extraction
- CBM-style segment size prior
- OLDA-style multi-feature stack

### 2. Window-chain DP scorer
Create a planner DP over section windows inspired by `compute_cbm(...)`:

- Suggested file: `src/core/planner/window_chain_dp.py`
- Core idea:
  - choose sequence of windows maximizing:
    - section readability
    - repetition coherence
    - donor usefulness
    - cross-parent compatibility
    - macro energy arc fit
  - while penalizing:
    - odd bar counts
    - seam risk
    - foreground collision
    - bad tempo/key relations

### 3. Transition alignment utility
Create a lean alignment scorer inspired by `transition-analysis/scripts/alignment.py` and `lib/cue.py`:

- Suggested file: `src/core/planner/transition_alignment.py`
- Outputs per candidate seam:
  - DTW/subsequence cost
  - cue-in / cue-out bars
  - diagonal stability length
  - key-invariant alignment variant

This would directly strengthen:
- seam viability
- donor-window insertion quality
- “handoff vs support bed” decisions

### 4. Compatibility primitives module
Create small deterministic primitives inspired by djkr8:

- Suggested file: `src/core/planner/compatibility_primitives.py`
- Functions to include:
  - `tempo_relation(...)`
  - `bpm_compatible(...)`
  - `key_transition_quality(...)`
  - `harmonic_compatibility_class(...)`

Use them everywhere as shared low-level signals instead of repeating one-off heuristics.

---

## Concrete adopt vs reject table

| Repo | Adopt? | Best files/functions | Why |
|---|---|---|---|
| `urinieto/msaf` | **Yes** | `sf/segmenter.py`, `cbm/CBM_algorithm.py`, `olda/segmenter.py` | Best source for modular phrase/section proposal logic and DP segment-size priors |
| `mir-aidj/transition-analysis` | **Yes** | `scripts/alignment.py`, `lib/cue.py` | Best source for alignment-derived cue extraction and beat-stable transition anchors |
| `schoi80/djkr8` | **Yes, selective** | `camelot.py`, `bpm.py`, `optimizer.py` | Clean compatibility primitives and hard/soft constraint separation |
| `0xf4b1/traktor-harmony` | No direct port | `harmonize.py::calculate_transition_score` | Too simple, but a useful baseline sanity formula |
| `congnghetinhtu/AutoMixSuite` | Mostly no | `automix.py` heuristics only | Some useful ideas, but monolithic and DJ-crossfade-centric |
| `congnghetinhtu/SongMix` | Mostly no | `automix.py` | Similar issues; weak fit for modular child-song planner |

---

## Highest-value immediate ports

If only 3 things get built from this harvest, do these:

1. **CBM-style bar-window DP with 8/16/32-bar priors**
   - source inspiration: `msaf/msaf/algorithms/cbm/CBM_algorithm.py`
   - expected payoff: stronger phrase/section readability and less arbitrary window sizing

2. **DTW/cue-based seam viability scorer**
   - source inspiration: `transition-analysis/scripts/alignment.py`, `transition-analysis/lib/cue.py`
   - expected payoff: better transition bars and donor insertions, fewer stitched seams

3. **Shared compatibility primitives (tempo/key/transition class)**
   - source inspiration: `djkr8/src/djkr8/camelot.py`, `djkr8/src/djkr8/bpm.py`
   - expected payoff: consistent shortlist/planner/render decisions and easier diagnostics

---

## Final recommendation

Do **not** import any public automix repo as a system. The strongest path is:

- port **MSAF-style modular segmentation ideas**,
- add **transition-analysis-style alignment cue extraction**,
- standardize **djkr8-style compatibility primitives**,
- and keep all of it under VocalFusion’s existing **planner-first, bar-aware, modular** architecture.

That combination is much more aligned with VocalFusion than monolithic “DJ automix” codebases.
