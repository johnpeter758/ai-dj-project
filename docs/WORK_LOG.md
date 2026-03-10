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

## Next step
Map the active `src/` modules into keep / archive / replace buckets and begin controlled cleanup.

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
