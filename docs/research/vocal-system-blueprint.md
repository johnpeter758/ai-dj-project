# Vocal System Blueprint (No AI-Slop Path)

## Goal
Build a vocal pipeline that produces **release-grade**, emotionally convincing vocals and rejects synthetic artifacts early.

## What creates “AI slop” (failure modes)
- unstable pitch drift / robotic vibrato
- over-sibilant highs (harsh "s" / "sh")
- thin body (no chest energy)
- hiss/noise floor from bad generation or stacking
- over-compressed, lifeless dynamics
- lyric/mouth timing not locked to groove

## Non-negotiable quality gates
1. **Vocal stem gate** (before mixing)
   - score with `scripts/vocal_quality_gate.py`
   - reject stems below threshold (default `<70`)
2. **Song-level listen gate** (after fusion/render)
   - `ai_dj.py listen` + listener-agent shortlist
3. **Human ear pass** on top candidates only

## Recommended architecture

### Stage A — Vocal Source Strategy
- Prefer legal high-quality stems / takes / toplines.
- If synthetic, generate multiple takes and keep only gate-approved takes.
- Keep identity/style inspiration broad; avoid artist-clone objectives.

### Stage B — Vocal Conditioning
- timing quantization to pocket (without killing feel)
- pitch correction with bounded naturalness (avoid hard autotune unless style requires)
- de-ess + resonance control
- breath/noise cleanup

### Stage C — Mix Integration
- sidechain vocal-safe ducking against dominant mids
- dynamic EQ against competing synth/pad bands
- section-aware vocal arrangement (drop/intensity lanes)

### Stage D — Objective + Subjective Gating
- objective: vocal_quality_score + listen overall/components
- subjective: human pass/fail tags (believable? catchy? fatiguing?)

## Immediate code added
- `scripts/vocal_quality_gate.py`
  - computes vocal quality metrics (pitch jitter, voiced ratio, sibilance/hiss/body, clipping)
  - outputs score 0–100 + verdict (`strong/usable/borderline/reject`)

## Next implementation steps
1. Add API endpoint `/api/rate-vocal` using `vocal_quality_gate.py`.
2. Require vocal gate pass before any candidate enters final fuse render.
3. Add per-section vocal arrangement rules in planner (where vocals lead vs support).
4. Extend critic loop to include vocal-specific penalties (sibilance, jitter, low-end masking).
5. Build a curated "great vocals" fixture pack and benchmark against it weekly.

## Success criteria
- >=80 average vocal-quality score on candidate stems
- >=90 chart-calibrated song score on top shortlisted outputs
- zero obvious robotic artifacts in blind A/B internal reviews
