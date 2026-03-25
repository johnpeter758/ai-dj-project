# Render / DSP web harvest report

Date: 2026-03-19
Workspace target: `/Users/johnpeter/Code/ai-dj-project`

## Goal
Find small, selectively portable renderer / DSP ideas for VocalFusion rather than importing another monolithic mashup engine. Focus areas:
- spectral carve / ducking
- overlap cleanup
- mastering / limiting
- transition shaping
- vocal clarity
- onset-aligned dynamics
- related music-mix utilities

## Current local integration surface
Primary local hooks already present:
- `src/core/render/renderer.py`
  - `_apply_transition_sonics(...)`
  - `_prepare_role_layer(...)`
  - `_section_mix_cleanup(...)`
  - `_overlap_carve_settings(...)`
  - `render_resolved_plan(...)`
  - `_finalize_master(...)`
- `src/core/render/spectral.py`
  - `compute_vocal_presence_mask(...)`
  - `apply_spectral_carve(...)`
- `src/core/render/mastering.py`
  - `bpm_synced_glue_compress(...)`
  - `lufs_normalize(...)`
- `src/core/render/transitions.py`
  - `equal_power_fade_in(...)`
  - `equal_power_fade_out(...)`
- `src/core/analysis/energy.py`
- `src/core/analysis/musical_intelligence.py`

---

## Best selective extraction candidates

### 1) Onset-aligned cut / fade anchoring
**Repo:** `librosa/librosa`
- File: `librosa/onset.py`
- Functions:
  - `onset_strength(...)`
  - `onset_detect(...)`
  - `onset_backtrack(...)`
- Why it matters:
  - `onset_strength` gives a robust spectral-flux envelope.
  - `onset_backtrack` is especially relevant: detected peaks get rolled back to the nearest preceding energy minimum, which is exactly the kind of logic that can make section boundaries and fade starts land on cleaner micro-attacks instead of random frame cuts.
- What to port/use:
  - Do **not** reimplement the whole module; we already depend on librosa.
  - Add a thin local helper that uses these calls to snap planned transition boundaries and fade starts/ends to nearby onset-safe minima.
- Best target integration points:
  - New file: `src/core/render/onset_alignment.py`
  - Call from `render_resolved_plan(...)` before `_apply_edge_fades(...)`
  - Optional upstream use in resolver manifest generation for `source_start_sec` / `source_end_sec` micro-adjustments.
- Concrete feature idea:
  - `snap_transition_edges_to_onsets(audio, sr, proposed_edges_sec, search_ms=120)`
  - Use only when stretch ratio is already safe and a nearby minimum exists.
- Risk:
  - Low. Dependency already present.

### 2) Adaptive onset detector with whitening/compression before peak-pick
**Repo:** `aubio/aubio`
- File: `src/onset/onset.c`
- Functions:
  - `aubio_onset_do(...)`
  - `aubio_onset_set_awhitening(...)`
  - `aubio_onset_set_compression(...)`
- Why it matters:
  - Aubio’s onset path is a nice compact recipe:
    1. phase vocoder frame
    2. adaptive spectral whitening
    3. log-magnitude compression
    4. spectral descriptor
    5. peak picker
  - This is useful not to replace planner analysis, but to improve transient localization for transition placement and overlap entry shaping in dense material.
- What to port/use:
  - Port the **idea**, not the C directly.
  - Add an optional Python helper that applies lightweight band whitening + log compression to STFT magnitudes before computing onset envelopes for transition alignment.
- Best target integration points:
  - `src/core/render/onset_alignment.py`
  - `src/core/analysis/musical_intelligence.py` for a second onset flavor like `whitened_onset_strength`
- Concrete feature idea:
  - `compute_transition_onset_envelope(..., whiten=True, log_compress_lambda=...)`
  - Use for `lift`, `swap`, and `drop` entries only.
- Risk:
  - Medium-low. Good concept, but best to re-express in Python/Numpy rather than bind aubio C.
- License note:
  - Aubio is GPL; safest path is algorithmic inspiration, not direct code copy into a non-GPL repo.

### 3) Smoothed spectral gating for overlap cleanup tails / noisy donor support
**Repo:** `timsainb/noisereduce`
- Files:
  - `noisereduce/spectralgate/base.py`
  - `noisereduce/spectralgate/nonstationary.py`
- Functions / methods:
  - `_smoothing_filter(...)`
  - `SpectralGate._generate_mask_smoothing_filter(...)`
  - `SpectralGateNonStationary.spectral_gating_nonstationary(...)`
  - `get_time_smoothed_representation(...)`
- Why it matters:
  - Their non-stationary spectral gate is small and practical:
    - estimate a smoothed spectral baseline
    - compare instantaneous STFT magnitude to baseline
    - build a sigmoid mask
    - smooth the mask in time/frequency
    - reapply to spectrogram
  - This is directly useful for **very narrow** VocalFusion cases:
    - suppress hissy/noisy donor-support intros
    - reduce low-level clutter inside overlap windows
    - clean transition tails after filtered support
- What to port/use:
  - Port only a tiny mask-builder, not the full library wrapper/chunking/temp-file stack.
- Best target integration points:
  - New file: `src/core/render/cleanup.py`
  - Call inside `_section_mix_cleanup(...)` only for:
    - `filtered_support`
    - `filtered_counterlayer`
    - arrival overlap intros shorter than ~2 bars
- Concrete feature idea:
  - `apply_nonstationary_spectral_gate(segment, sr, band_focus='presence'|'full', strength=...)`
  - Use a presence-focused mask for donor vocal support, full-band light gate for noisy tails.
- Risk:
  - Medium. Easy to overdo and cause underwater artifacts if applied to full sections.
- License note:
  - Verify compatibility before direct code lift. Safe path is reimplementation of the mask idea.

### 4) Simple, useful limiter envelope logic for final master protection
**Repo:** `pedicino/MultiAudio`
- File: `effects/Limiter.cpp`
- Class / methods:
  - `Limiter::calculateCoeffs()`
  - `Limiter::process(...)`
  - attack / release smoothing around `targetGain`
- Why it matters:
  - This is a clean tiny limiter implementation:
    - derive target gain from threshold exceedance
    - smooth with separate attack/release coefficients
    - apply samplewise gain
  - More transparent than the current `_soft_limit(...)`-only safety stage.
- What to port/use:
  - Port the control-envelope logic in Python.
  - Keep it stereo-linked using max abs across channels.
- Best target integration points:
  - `src/core/render/mastering.py`
  - Replace or precede `_soft_limit(...)` in `_finalize_master(...)`
- Concrete feature idea:
  - `lookless_peak_limiter(audio, sr, threshold_db=-1.2, attack_ms=1.5, release_ms=70)`
  - Later upgrade to short lookahead if needed.
- Expected payoff:
  - Better peak containment on dense overlaps without immediately tanh-flattening the whole render.
- Risk:
  - Low.
- License note:
  - Inspect before direct copy; if uncertain, reimplement from the clearly visible algorithm.

### 5) Cheap de-esser band attenuation for vocal clarity during overlap
**Repo:** `pedicino/MultiAudio`
- File: `effects/DeEsser.cpp`
- Function:
  - `applyDeEsser(std::vector<double>& samples, int sampleRate, int startFreq, int endFreq, double reductionDB)`
- Why it matters:
  - The implementation is extremely simple: FFT frame, attenuate bins in a sibilant band, inverse FFT.
  - It is not a full smart de-esser, but it is enough to inspire a focused VocalFusion tool for the exact failure mode where both overlap layers are bright and sibilant.
- What to port/use:
  - Port the concept as a **conditional presence-band tamer**, not as a global mastering effect.
- Best target integration points:
  - `src/core/render/cleanup.py`
  - Trigger from `_section_mix_cleanup(...)` or overlap-carve path when:
    - `vocal_state in {'lead', 'support'}`
    - strong high-band occupancy in both incumbent and incoming layers
- Concrete feature idea:
  - `apply_sibilance_tamer(segment, sr, start_hz=4500, end_hz=9000, max_reduction_db=2.5)`
  - Apply only on overlap windows or short arrivals.
- Expected payoff:
  - Cleaner consonants during stacked vocal moments without broad dulling.
- Risk:
  - Medium. Needs masking logic; static full-section use will sound lifeless.

### 6) FFT-band energy gate for “support present or mute it” logic
**Repo:** `pedicino/MultiAudio`
- File: `effects/NoiseGate.cpp`
- Methods:
  - `calculateBandEnergies()`
  - `determineTargetGain(...)`
  - attack/release smoothed gate in `process(...)`
- Why it matters:
  - The useful idea here is not a classic vocal noise gate. It is the coarse log-spaced FFT-band-energy summary used to decide whether a signal is meaningfully present.
  - That can become a cheap validator for donor support inserts.
- What to port/use:
  - Port the band-energy summary as a utility to detect “this support layer has enough useful content to justify occupying the overlap”.
- Best target integration points:
  - New helper in `src/core/render/cleanup.py` or `src/core/render/metrics.py`
  - Use before placing very quiet filtered support in `render_resolved_plan(...)`
- Concrete feature idea:
  - `support_presence_score(segment, sr)`
  - If score is too low, shrink fade-in or drop the support work order entirely.
- Expected payoff:
  - Prevents cosmetic donor layers from cluttering transitions when they contribute little audible identity.
- Risk:
  - Low-medium.

### 7) Speech-oriented denoise / AGC / VAD primitives for optional vocal-island cleanup
**Repo:** `xiph/speexdsp`
- Files:
  - `libspeexdsp/preprocess.c`
  - `include/speex/speex_preprocess.h`
- Functions / API:
  - `speex_preprocess_state_init(...)`
  - `speex_preprocess_run(...)`
  - `speex_preprocess_ctl(...)`
- Why it matters:
  - SpeexDSP preprocess exposes a compact bundle of denoise / AGC / VAD / dereverb-ish controls aimed at speech.
  - For VocalFusion, the value is **not** whole-track processing. The value is optional cleanup of small vocal islands or spoken intros before insertion.
- What to port/use:
  - Prefer using a wrapper/binary if ever adopted, not a full source port.
  - Near-term: treat as an architecture reference for a future `vocal_cleanup_mode` that is only allowed on isolated voice support clips.
- Best target integration points:
  - New optional path in `src/core/render/cleanup.py`
  - Only for stem-isolated vocal clips, not mixed full-range sections.
- Concrete feature idea:
  - `optional_voice_preclean(segment, sr, mode='light_denoise')`
- Expected payoff:
  - Cleaner spoken/vocal donor snippets in intros/bridges.
- Risk:
  - Medium-high for music. Speech preprocessors can destroy singing texture if overused.
- License note:
  - BSD-style and practical, but still better as optional boundary tool rather than default mix stage.

### 8) Band-energy interpolation + pitch-coherent enhancement ideas for vocal-presence masks
**Repo:** `xiph/rnnoise`
- File: `src/denoise.c`
- Functions:
  - `compute_band_energy(...)`
  - `interp_band_gain(...)`
  - `rnn_pitch_filter(...)`
  - `rnnoise_process_frame(...)`
- Why it matters:
  - Two small ideas here are highly reusable even without the RNN:
    1. ERB-ish band energy accumulation instead of raw linear FFT-bin logic.
    2. Interpolating band gains back to FFT bins for smoother masks.
  - This is directly useful for better spectral carve masks.
- What to port/use:
  - Do **not** port the whole RNNoise denoiser.
  - Port the band-energy/interpolated-gain pattern into the local spectral carve utility.
- Best target integration points:
  - `src/core/render/spectral.py`
  - upgrade `compute_vocal_presence_mask(...)`
  - add optional `band_smoothed=True` path for `apply_spectral_carve(...)`
- Concrete feature idea:
  - Build the vocal mask in 24-32 perceptual bands, then interpolate back to FFT resolution.
  - This should avoid brittle bin-by-bin carving and reduce chirpy artifacts.
- Expected payoff:
  - Smoother, more musical ducking during vocal overlaps.
- Risk:
  - Low-medium.
- License note:
  - BSD-like; practical if reimplemented.

### 9) Better resampling kernel reference for duration-fit quality
**Repo:** `xiph/speexdsp`
- File: `libspeexdsp/resample.c`
- Core object:
  - `SpeexResamplerState_`
- Why it matters:
  - Your current `_fit_to_duration(...)` uses `librosa.effects.time_stretch`, which is fine at the macro level, but if you later separate small timing correction from larger tempo adaptation, SpeexDSP’s resampling approach is a good reference for high-quality short correction moves.
- What to port/use:
  - Mostly architecture reference unless a Python binding is already available.
- Best target integration points:
  - Future `src/core/render/timebase.py`
  - Use for micro duration correction after a main phase-vocoder stretch.
- Concrete feature idea:
  - large ratio change via current stretch path, then tiny end correction via high-quality resample instead of pad/truncate.
- Expected payoff:
  - Cleaner tails and less zippery duration forcing on phrase-trim windows.
- Risk:
  - Medium.

---

## Highest-value near-term implementation order

### Tier 1: should build now
1. **Onset-aligned transition snapping**
   - Source ideas: `librosa/onset.py`, `aubio/src/onset/onset.c`
   - Why first: directly attacks seam feel and timing believability.
   - Files:
     - add `src/core/render/onset_alignment.py`
     - patch `src/core/render/renderer.py`

2. **Envelope limiter before tanh soft clip**
   - Source ideas: `pedicino/MultiAudio/effects/Limiter.cpp`
   - Why first: easy win, low complexity, useful on every render.
   - Files:
     - patch `src/core/render/mastering.py`
     - patch `_finalize_master(...)` in `renderer.py`

3. **Band-smoothed spectral carve mask**
   - Source ideas: `xiph/rnnoise/src/denoise.c`
   - Why first: directly strengthens an already-existing subsystem instead of creating another processing branch.
   - Files:
     - patch `src/core/render/spectral.py`

### Tier 2: build after Tier 1 validates audibly
4. **Selective nonstationary gate for overlap intros / filtered support only**
   - Source ideas: `timsainb/noisereduce`
   - Files:
     - add `src/core/render/cleanup.py`
     - patch `_section_mix_cleanup(...)`

5. **Conditional sibilance tamer for stacked vocals**
   - Source ideas: `pedicino/MultiAudio/effects/DeEsser.cpp`
   - Files:
     - `src/core/render/cleanup.py`
     - maybe `src/core/render/spectral.py`

### Tier 3: optional / experimental
6. **Voice-island preclean mode** from SpeexDSP ideas
7. **Micro end-correction resampler path** from SpeexDSP resampler

---

## Concrete code extraction plan

### A. `src/core/render/onset_alignment.py` (new)
Implement:
- `compute_onset_envelope(audio, sr, whiten=False, compress=False)`
- `find_onset_safe_boundary(audio, sr, target_sec, search_ms=120)`
- `snap_fade_window_to_onset_minima(audio, sr, start_sec, end_sec)`

Port ideas from:
- `librosa.onset.onset_strength`
- `librosa.onset.onset_backtrack`
- aubio whitening/compression pre-peak flow

Wire into:
- `render_resolved_plan(...)` right after `_extract(...)` or right before `_apply_transition_sonics(...)`

### B. `src/core/render/mastering.py`
Add:
- `peak_limiter(audio, sr, threshold_db=-1.2, attack_ms=1.5, release_ms=70.0)`

Port ideas from:
- `pedicino/MultiAudio/effects/Limiter.cpp`

Wire into:
- `_finalize_master(...)` before `_soft_limit(...)`

### C. `src/core/render/spectral.py`
Upgrade:
- `compute_vocal_presence_mask(...)`
- `apply_spectral_carve(...)`

Port ideas from:
- `xiph/rnnoise/src/denoise.c`
  - band energy accumulation
  - band gain interpolation

Result:
- perceptual-band carve mask rather than raw STFT-bin ratio only

### D. `src/core/render/cleanup.py` (new)
Add narrow-scope utilities:
- `apply_nonstationary_spectral_gate(...)`
- `apply_sibilance_tamer(...)`
- `support_presence_score(...)`

Port ideas from:
- `timsainb/noisereduce`
- `pedicino/MultiAudio/effects/DeEsser.cpp`
- `pedicino/MultiAudio/effects/NoiseGate.cpp`

Wire into:
- `_section_mix_cleanup(...)`
- maybe pre-overlap path in `render_resolved_plan(...)`

---

## What not to import
- Do **not** import giant render engines or full monolithic mashup repos.
- Do **not** copy GPL aubio / Rubber Band code directly into core render code unless licensing is intentionally handled.
- Do **not** run speech denoisers on full mixed sections by default.
- Do **not** globally spectral-gate every overlap; keep it limited to noisy support / short transitional material.

---

## Recommended next patch sequence
1. Add onset-aware transition snapping helper.
2. Add limiter envelope in mastering.
3. Improve spectral carve mask with perceptual-band smoothing.
4. A/B those three changes on one fixed pair.
5. Only then test narrow spectral gate and conditional de-esser.

## Bottom line
The strongest selective ports are **not** giant repos. They are compact primitives:
- librosa / aubio for onset-safe transition timing
- MultiAudio limiter logic for better final protection
- RNNoise band interpolation ideas for smoother spectral carving
- noisereduce mask smoothing for narrow overlap cleanup
- simple de-esser / band-energy gating utilities for vocal clarity and support pruning

Those all fit the existing deterministic planner-first renderer and can land as small focused modules instead of architectural bloat.
