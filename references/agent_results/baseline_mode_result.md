# Baseline mode implementation result

Implemented a first decent-first backbone/donor baseline path centered on `build_stub_arrangement_plan(..., arrangement_mode="baseline")` and wired it into the CLI-facing `fusion`, `prototype`, and `auto-shortlist-fusion` paths as the new default arrangement mode.

## What changed
- `src/core/planner/arrangement.py`
  - Added hard baseline pair admissibility checks:
    - tempo hard cap at `1.10x` ratio
    - conservative key admissibility via harmony score / semitone distance
  - Added a dedicated baseline section program that:
    - keeps one backbone parent for `intro -> verse -> outro`
    - uses one contiguous donor feature block in `build -> payoff`
    - disables donor support layers in baseline mode
    - prefers phrase-safe, preferred-parent selections to preserve backbone chronology and a single donor block
  - Added diagnostics/notes for `arrangement_mode` and `baseline_admissibility`
  - Added backbone-only fallback when the pair is not baseline-admissible
- `ai_dj.py`
  - `fusion`, `prototype`, and `auto-shortlist-fusion` now accept `--arrangement-mode {baseline,adaptive}`
  - default arrangement mode is now `baseline` for those entrypoints
- `src/core/intelligence/recipe_builder.py`
  - added `arrangement_mode` handling so baseline sections emit baseline-specific recipe metadata (`policy_id=section_recipe_v1_baseline`) and baseline timbral anchors

## Tests added/updated
- `tests/test_core_planner.py`
  - baseline mode keeps one contiguous donor block and disables support layers
  - baseline mode falls back to backbone-only when BPM/key admissibility fails
- `tests/test_section_recipe_builder.py`
  - baseline mode marks recipe policy/anchor metadata

## Validation
- `python -m pytest -q tests/test_core_planner.py tests/test_section_recipe_builder.py`
  - `108 passed, 1 skipped`
- `python -m pytest -q tests/test_auto_shortlist_fusion.py tests/test_render_stack.py`
  - `46 passed`

## Important behavior notes
- Baseline mode is intentionally conservative: one backbone, one donor block, no support-layer clutter.
- If a pair fails hard tempo/key admissibility, the planner now stays backbone-only instead of forcing a bad blend.
- The older richer planner behavior is still available via `--arrangement-mode adaptive`.
