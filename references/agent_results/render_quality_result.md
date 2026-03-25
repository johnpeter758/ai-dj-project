# Render quality improvement result

## Change implemented
Added a proper lookahead envelope limiter in `src/core/render/mastering.py` and inserted it into the final master chain in `src/core/render/renderer.py` **before** the soft clipper.

## Why this change
The renderer previously went from glue/bus compression straight into soft clipping. That leaves short peaks to be handled mostly by the clipper, which is a brittle way to control transients and can add avoidable harshness. The new limiter catches hot transient spikes first, then leaves the clipper to do only light final rounding.

## Concrete implementation
- Added `lookahead_envelope_limit(...)`:
  - stereo-linked peak envelope detection
  - short lookahead via max-filtered predicted envelope
  - fast attack / slower release gain smoothing
  - ceiling-based gain targeting
- Updated `_finalize_master(...)` to run:
  1. high-pass
  2. BPM-synced glue compression
  3. bus compression
  4. **lookahead envelope limiter**
  5. soft clip
  6. LUFS normalize
  7. final peak normalize
- Slightly reduced soft-clip drive (`1.08 -> 1.05`) now that peak control is handled more cleanly upstream.

## Test coverage added
- `tests/test_mastering.py`
  - verifies the limiter catches a short hot transient near the configured ceiling without crushing the sustained body
- `tests/test_render_stack.py`
  - verifies `_finalize_master(...)` smooths a hot transient while preserving a meaningful sustained signal after the spike

## Validation
Ran:

```bash
/Users/johnpeter/venvs/vocalfusion-env/bin/python -m pytest -q tests/test_mastering.py tests/test_render_stack.py
```

Result:
- `45 passed`

## Notes
This was kept tightly scoped to `src/core/render/*` and renderer-adjacent tests only, per request.
