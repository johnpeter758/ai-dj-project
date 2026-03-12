# Deterministic renderer failure audit

_Date: 2026-03-11_

## Scope
Audit of likely failure modes for the **first deterministic renderer** in `ai-dj-project`, based on current analysis/planner code and render design docs. No code changes made.

## High-probability failure modes

| Area | Why the first pass will likely fail | Prevention heuristic |
|---|---|---|
| Beat snapping | Current timing truth is weak: `structure.sections` can collapse to one giant section, while beat timing comes from a single global tempo estimate plus detected beat frames. If the renderer cuts by raw section seconds or "nearest beat" symmetrically, attacks will smear and downbeats will drift. | Treat `tempo.beat_times` as the only hard snap grid in v1. Snap **starts down / ends up**, clamp to planned beat counts, and reject clips when nearest safe beat is too far from phrase intent. |
| Phrase truncation | Planner bars are canonical, but source sections are second-based and often not phrase-safe. The first implementation will tend to trim to target seconds after stretch, which can cut cadences, vocal syllables, or pickup notes. | Resolve a beat/phrase-safe source window **before** stretching. Prefer 4- or 8-bar windows; do not cut odd lengths unless intentional. If a clean window is unavailable, shorten donor use or hard-swap sections. |
| Low-end overlap | Docs already note this as the main mud risk. A naive "blend" implementation will let both parents contribute kick/bass/full mix at once, especially if stems are missing and fallback is full-mix or HPSS. | Make `low_end_owner` explicit per section/transition. One parent owns kick+sub; the other is high-passed, muted, or excluded. Limit low-end handoff windows to 1-2 beats. |
| Vocal masking | With only coarse section labels and no reliable vocal activity mask, the first renderer will easily stack lead vocal over lead vocal or dense vocal over harmonic mids. Even if levels are lower, intelligibility will collapse. | Default to **one lead vocal owner**. Any donor vocal should be tail-only, ad-lib-only, or fully muted. When vocals are active, duck competing harmonic mids 1-4 kHz by a small deterministic amount. |
| Stretch artifacts | Repo/docs point to section-level stretching with `librosa.effects.time_stretch`, but tempo gaps can be large (e.g. 120.19 vs 132.51 ≈ +10.3%). First-pass code will be tempted to stretch dense full-mix material too far. | Gate stretch by content type. Safe: drums/percussion. Cautious: harmonic beds. Avoid sustained vocal/full-mix blends past ~±6%; collapse to swap/transition-only use past ~±8%; no sustained blend past ~±12%. |
| Export/finalization | The repo has generic export helpers, but the deterministic renderer path does not yet appear to have a dedicated finalization contract. First implementation risk: clipping summed buses, wrong channel/bit-depth assumptions, loudnorm-only "fixes," or exporting lossy first. | Export WAV first, then derive MP3/AAC. Keep at least -6 dB internal headroom, use equal-power fades, and only loudness-normalize after summing. Validate sample rate, channel count, peak ceiling, and output file existence in a final render manifest. |
| Path / provenance mistakes | `SongDNA.source_path` is absolute/resolved during analysis, but planned/resolved render artifacts are not yet enforcing a strong provenance chain. First renderer may mix stale stems, wrong parent files, relative paths, or transformed caches without enough identity in the key. | Every work order should carry: parent id, exact source path, source window, target window, anchor BPM, semitone shift, transform cache key, and artifact version. Cache key should include `source_path + stem_name + target_bpm + semitone_shift + sample_rate`. |

## Specific repo signals behind the audit

- `src/core/analysis/structure.py`
  - infers phrase boundaries by grouping beats in fixed 4/4 chunks
  - often falls back to a single full-song section
  - makes phrase-safe extraction more reliable than section labels
- `src/core/analysis/tempo.py`
  - provides usable beat times, but still from one global beat tracker
  - not enough to claim true downbeat/bar accuracy
- `src/core/planner/arrangement.py`
  - current arrangement is a stub with placeholder section labels like `section_0`
  - first renderer will likely receive coarse, not musically rich, section references
- `docs/RENDER_RESOLVER_DESIGN_2026-03-11.md`
  - correctly says planner bars should be canonical and source bounds should be snapped before rendering
  - main risk is implementation drifting away from that contract
- `docs/RENDER_OVERLAP_OWNERSHIP_POLICY_2026-03-11.md`
  - already identifies one-low-end-owner / one-foreground-owner as the key anti-mud rule
  - failure happens if these remain docs-only and not explicit in work orders

## Failure patterns to expect in the very first render

1. **Beat-aligned but still wrong-feeling starts**
   - Cause: nearest-beat snapping without phrase intent
   - Symptom: vocals enter late/early relative to groove even though the edit is click-free

2. **Section endings that sound chopped**
   - Cause: target duration enforced after stretch without preserving cadence
   - Symptom: missing pickup to next phrase, clipped reverb tail, unfinished lyric line

3. **Drops that feel small or muddy**
   - Cause: both tracks still own bass/kick energy during overlap
   - Symptom: less punch after the "drop" than before it

4. **Hooks become unintelligible during blends**
   - Cause: simultaneous lead material in the same midrange band
   - Symptom: words/hook contour disappear even when meters look fine

5. **Warped donor sections**
   - Cause: stretching full mixes beyond safe tolerance just to satisfy the plan
   - Symptom: cymbal smear, chorus wobble, unstable bass sustain

6. **A render that technically exports but is operationally untrustworthy**
   - Cause: missing provenance / wrong transformed asset reused
   - Symptom: output does not match plan, or cannot be reproduced from artifacts

## Minimal prevention checklist for v1

- Use **bars from planner** as canonical child duration.
- Use **beat_times** as the hard snap grid; phrase boundaries as the extraction scaffold.
- Snap **start backward, end forward** where possible.
- Enforce **one low-end owner** and **one foreground owner** at all times.
- Default to **single lead vocal owner**.
- Reject aggressive full-mix blends when stretch ratio is outside safe bands.
- Render **WAV first**, then loudness normalize / transcode.
- Emit a **resolved manifest** with exact source paths, windows, transforms, and cache identities.

## Bottom line
The first deterministic renderer is most likely to fail not because of missing DSP sophistication, but because of **unsafe musical boundary selection and ambiguous ownership**. If v1 is strict about beat/phrase-safe extraction, single-owner low end, single-owner foreground, conservative stretch limits, and explicit provenance in work orders, it can avoid most of the obvious amateur-sounding failure modes.