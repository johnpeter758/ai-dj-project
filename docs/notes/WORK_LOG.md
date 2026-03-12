# Work Log

## 2026-03-10
- Connected the repo to the active local workspace.
- Confirmed GitHub auth is working for `johnpeter758`.
- Audited the repo at a high level and confirmed severe module sprawl.
- Identified the likely core files worth keeping directionally:
  - `src/vocalfusion.py`
  - `src/arrangement_generator.py`
  - `src/fusion_v5.py`
  - `src/quality_evaluator.py`
- Created initial repo-level triage, status, and target architecture notes.

## 2026-03-11
- Reviewed the current render stack and today’s design notes around resolver behavior, fallback logic, overlap ownership, transition policy, and synthetic test scope.
- Converged on a resolver-first v1 contract: planner bars stay canonical, beat times are the hard snap grid, phrase boundaries are the extraction scaffold, and coarse section labels are only soft hints.
- Identified the main real-pair failure mode as weak source-window selection from coarse `section_0` spans; safest next step is phrase-safe target-length window selection with explicit downgrade/fallback recording.
- Confirmed the first listenable Drake + Relax My Eyes direction should stay Drake-led with one clear Relax spotlight rather than a 50/50 blend.
- Defined the smallest high-value render test focus: contiguous manifest math, explicit ownership rules, deterministic output, exact duration conservation, bounded overlaps, and safe single-source fallback behavior.
- Commit-readiness audit: the current render-stack checkpoint is close to coherent, but local handoff is still blocked by unverified test execution in this shell (missing `pytest` / runtime deps like `librosa`) and by unresolved v1 semantics around phrase-safe source-window selection versus coarse full-section snapping.

## Next step
Iterate on the first deterministic renderer so the Drake + Relax My Eyes pair produces a listenable child output, starting with phrase-safe resolver window selection and explicit ownership/fallback enforcement before richer DSP or mixing logic.

## Candidate core modules inspected
- `src/beat_detector.py` — usable seed for tempo/beat extraction
- `src/key_detector.py` — stronger-than-average seed for key/camelot analysis
- `src/stem_splitter.py` — usable Demucs wrapper seed
- `src/audio_utils.py` — broad utility bulk; probably needs trimming or stricter ownership
- `src/auto_dj.py` — not part of the real professional core
- `src/genre_classifier.py` — possible secondary analysis helper
- `src/stem_mixer.py` — historical layering experiment, not a final architecture
- `src/system_cleanup.py` — should be demoted out of the product core

## Structural progress
- Added module classification and legacy demotion docs.
- Added a core layout proposal.
- Created a canonical `src/core/` scaffold with analysis/planner/render/evaluation subpackages.
- Created `src/legacy/` as the destination for future controlled demotions.
- Implemented the first active core code under `src/core/analysis/`:
  - loader
  - tempo detection
  - key detection
  - structure estimation
  - energy profiling
  - optional Demucs stems wrapper
  - SongDNA model
  - canonical analyzer entrypoint
- Added and updated focused tests for the new analysis core.
- Added planner foundation code under `src/core/planner/`:
  - typed planner models
  - factorized compatibility scoring
  - compatibility report artifact
  - stub arrangement-plan artifact
- Added focused planner tests.
