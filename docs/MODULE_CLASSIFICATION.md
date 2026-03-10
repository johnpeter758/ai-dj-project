# Module Classification

## Goal
Classify the current flat `src/` surface so we can reduce sprawl without losing the few useful seeds.

## Keep / modernize candidates
These are the most promising directionally aligned modules found so far:
- `src/vocalfusion.py`
- `src/arrangement_generator.py`
- `src/fusion_v5.py`
- `src/quality_evaluator.py`
- `src/beat_detector.py`
- `src/key_detector.py`
- `src/stem_splitter.py`
- `src/genre_classifier.py` (secondary analysis support only)

## Demote / archive candidates
These may contain ideas, but should not stay in the active product center:
- `src/auto_dj.py`
- `src/ai_dj_system.py`
- `src/orchestrator.py`
- `src/system_cleanup.py`
- most dashboard/social/cloud/mobile/workflow wrappers
- effect-specific one-file modules unless they support a real deterministic render path

## Rationale
The project wins from:
- better song analysis
- better arrangement planning
- better deterministic rendering
- better evaluation

It does not win from having a very large number of shallow modules.

## Immediate implementation rule
No new active subsystem should be added unless it clearly belongs to one of:
- analysis
- planner
- render
- evaluation
