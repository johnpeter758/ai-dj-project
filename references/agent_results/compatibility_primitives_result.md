# Compatibility primitives result

Implemented a focused compatibility-primitives cleanup for the decent-first / baseline path in `src/core/planner`.

## What changed
- Added reusable conservative compatibility helpers in `src/core/planner/compatibility.py`:
  - `tempo_ratio(...)`
  - `tonic_pitch_class(...)`
  - `key_semitone_distance(...)`
  - `baseline_hard_key_pass(...)`
  - `baseline_pair_admissibility(...)`
- Moved the baseline pair-admissibility logic out of the private inline implementation in `src/core/planner/arrangement.py` so the baseline gate now delegates to the shared helper instead of carrying duplicated embedded tempo/key logic.
- Kept the baseline rule conservative and explicit:
  - hard tempo cap via ratio (`<= 1.10` by default)
  - hard key pass via strong Camelot relation or close semitone distance / exact same-key fallback
  - structured rejection reasons for tempo and key failures

## Tests added
Updated `tests/test_core_planner.py` with explicit helper-level regressions that verify:
- a near pair passes the conservative tempo/key gate
- a clearly bad pair is rejected with explicit tempo and key reasons

## Validation
- `python -m pytest -q tests/test_core_planner.py -k 'compatibility_helpers or baseline_mode'` -> `4 passed`
- `python -m pytest -q tests/test_core_planner.py` -> `106 passed, 1 skipped`

## Notes
- Scope stayed inside planner compatibility/planner tests only.
- No server/render files were touched.
- The net effect is to make baseline pair rejection more explicit, reusable, and directly testable instead of leaving it buried in planner flow code.
