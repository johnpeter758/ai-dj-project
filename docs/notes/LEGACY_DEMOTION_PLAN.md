# Legacy Demotion Plan

## Purpose
Document which legacy modules should be treated as historical references instead of active product code.

## First demotion wave
### Control shells / placeholders
- `src/ai_dj_system.py`
- `src/auto_dj.py`
- `src/orchestrator.py`

Reason:
- too broad
- too simulated
- not centered on the real product differentiator

### Personal-machine / ops clutter
- `src/system_cleanup.py`

Reason:
- unrelated to the music intelligence core
- mixes personal file cleanup with product logic
- should not be in the critical path

## Demotion approach
1. document the status first
2. avoid breaking imports accidentally
3. move truly non-core files out of the active path in controlled batches
4. preserve only what helps analysis, planning, rendering, or evaluation
