# VocalFusion Task Queue

Last updated: 2026-03-27 00:20 EDT (pair2 dual-support rerun reached pass+floor)
Owner: execution operator

## Current Task (active now)
1. **Stabilize adaptive dual-support render quality while preserving floor pass**
   - Why: pair2 now clears hard floor (`song_likeness=58.2`, winner `pass+floor`) but transition/mix quality is still moderate (`transition=53.6`, `mix_sanity=66.2`).
   - Focus:
     - render-side support intensity/shape tuning for payoff+build overlays,
     - improve transition clarity without losing integrated two-parent identity.
   - Files likely touched:
     - `src/core/render/renderer.py`
     - `src/core/render/resolver.py`
     - `tests/test_render_stack.py`

## Next Task (auto-start immediately after current)
1. **Promote support-overlay strategy from fallback to primary pro-mode candidate path**
   - Why: unblock condition is now met with a second floor-pass checkpoint on pair2.
   - Guardrails:
     - keep anti-medley penalties active,
     - require at least one integrated support candidate in both adaptive and baseline candidate sets,
     - preserve hard-floor gating.

## Blocked Tasks
1. **Wide regression sweep + artifact listening review across additional pairs**
   - Blocker: render-tuning pass not yet complete after pair2 dual-support floor-pass.
   - Unblock condition: transition/mix tuning patch lands with tests and no regressions on pair2.

## Queue Rules (enforced each cycle)
- Always keep `Current`, `Next`, and `Blocked` sections updated.
- When current task is done, move next -> current immediately.
- If blocked, document blocker and continue with next highest-value task.
- Never end a cycle without updating this file.
