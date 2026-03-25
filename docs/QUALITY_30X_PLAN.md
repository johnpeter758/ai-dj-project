# VocalFusion 30x Quality Plan

Date: 2026-03-25
Owner: Assistant (execution) + John

## Goal (explicit)
Make fusion output quality feel ~30x better to humans by shifting from "passes metrics" to "sounds like one intentional song".

## Current reality
- Engineering health is decent (tests green), but musical quality is inconsistent.
- Best recent automated score in current loop reached high 70s on clipped runs, but user bar is much higher than score.
- Main risk: over-optimizing evaluator numbers while humans still hear stitched medleys.

## 30x Definition (what success actually means)
A run is a true success only when all are true:
1. Human blind A/B preference wins against previous best and simple baseline.
2. Listener-agent gate = pass with margin (not borderline).
3. No obvious "track switch" seams in first listen.
4. Clear intro->build->payoff->resolution narrative.
5. Repeatable across multiple song pairs (not one lucky pair).

## Strategic changes (high leverage)

### 1) Optimize for human listening, not just scalar score
- Keep current evaluator, but add hard promotion policy:
  - Never promote candidate with listener gate != pass.
  - Require minimum song_likeness and groove floors before publish.
- Add a small human-review artifact packet per winner:
  - 30s highlights around top 3 seams.
  - Structured rationale: why this candidate won.

### 2) Expand candidate diversity in pro mode (already started)
- Pro mode now explores more planner variants and applies gate-aware selection.
- Next: add explicit diversity constraints in candidate generation:
  - Vary donor cluster placement.
  - Vary intro lane source windows.
  - Vary payoff strategy windows.

### 3) Hard structural guards before render
- Enforce opening readability and contiguous narrative constraints upstream.
- Guard against pseudo-payoff and pseudo-bridge sections even when they score locally.
- Require at least one credible late payoff candidate before allowing final program.

### 4) Transition sonics as first-class objective
- Add transition-local objective checks in selection:
  - vocal collision risk windowed over seam region
  - low-end ownership violations over seam
  - onset misalignment confidence
- Penalize candidates with seam artifacts even if overall score is high.

### 5) Two-stage winner selection
Stage A: shortlist by listener gate + structural constraints.
Stage B: rank shortlist by weighted quality + diversity + seam confidence.

## Execution roadmap

## Sprint 1 (immediate, 1-2 days)
- [x] Pro mode candidate expansion + gating-aware scoring (landed).
- [ ] Add hard winner floors in pro mode promotion logic:
      - gate must be pass
      - song_likeness >= threshold
      - groove >= threshold
- [ ] Emit seam-spotlight clips + diagnostics for top candidate.
- [ ] Add regression tests for pro-mode promotion constraints.

## Sprint 2 (2-4 days)
- [ ] Add diversity-aware variant generation (multi-swap + role-targeted variants).
- [ ] Add structural viability checks before render for intro/verse/payoff path.
- [ ] Add transition risk features into final selection payload and scoring.

## Sprint 3 (4-7 days)
- [ ] Build fixed benchmark suite of 10 pairings (with known hard cases).
- [ ] Run nightly benchmark and track pass-rate + human audit notes.
- [ ] Promote only if benchmark pass-rate and human A/B both improve.

## KPI dashboard (must move together)
- Human A/B win rate vs previous best
- Listener gate pass-rate across benchmark set
- Median song_likeness
- Median groove
- Seam-failure count per track
- % outputs rejected by human within 15s

## Non-negotiables
- No scoring-only improvements without listening-proof.
- No winner promotion if listener gate rejects.
- Every major quality claim requires artifact + benchmark evidence.
