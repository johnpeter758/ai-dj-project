# JPEMSLIE Vocal Fusion Import Plan

## Imported source snapshot
- Source repo vendored at `references/jpemslie_vocal_fusion_ai/`
- Purpose: mine product and DSP ideas without blindly replacing the current core architecture.

## Strong parts worth adopting
1. **Simple product contract**
   - `run.py` + `templates/index.html`
   - clear async fuse/status/share flow
   - better default UX than heavyweight multi-branch orchestration

2. **Reference DSP / mix heuristics**
   - `fuser.py`
   - useful source of vocal-chain, beat-sync, carve, and mastering ideas
   - should be mined selectively, not imported wholesale (monolith, high coupling, many hardcoded assumptions)

3. **Output grading / QA ideas**
   - `listen.py`, `qa_audio.py`, `compare.py`, `grade.py`
   - useful as benchmark inspiration and regression-fixture source

4. **API contract ideas**
   - `api.py`
   - useful if VocalFusion needs a cleaner external fuse API later

## Do NOT import directly into core
- giant monolithic `fuser.py` as a replacement for `src/core/*`
- repo-specific state paths (`vf_data/*`)
- ad hoc auth/job storage patterns without adaptation
- hardcoded DSP claims without regression validation

## Recommended merge strategy
1. keep current `src/core/analysis`, `src/core/planner`, `src/core/render`, `src/core/evaluation`
2. extract useful ideas from reference repo into:
   - `src/legacy_reference/jpemslie_notes/` or docs/tests
   - targeted utilities / tests / fixtures, not a second parallel engine
3. copy over the strongest product UX ideas first:
   - async job model
   - clearer progress/status/share contract
4. then mine DSP heuristics into isolated, testable modules
5. benchmark any adopted logic against current fixtures before promotion

## Immediate high-value files to inspect next
- `references/jpemslie_vocal_fusion_ai/run.py`
- `references/jpemslie_vocal_fusion_ai/fuser.py`
- `references/jpemslie_vocal_fusion_ai/listen.py`
- `references/jpemslie_vocal_fusion_ai/templates/index.html`
- `references/jpemslie_vocal_fusion_ai/tests/regression.py`
