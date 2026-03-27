# VocalFusion Musician-First Revamp Plan

Last updated: 2026-03-26 20:48 EDT

## Goal
Build fused songs that feel like **one intentional composition**, not two tracks taking turns.

## Musician Workflow We Are Adopting
1. **Composition intent first**
   - Define one backbone narrative (intro -> verse -> build -> payoff -> outro).
   - Every section decision must serve that narrative.
2. **Arrangement continuity**
   - Keep section ownership coherent over phrases; avoid frequent owner switching.
   - Donor inserts must reinforce backbone, not replace it abruptly.
3. **Performance seams**
   - Treat transitions as performance handoffs, not hard source swaps.
   - Reduce seam crowding (especially into payoff).
4. **Mix hierarchy**
   - One clear foreground at a time; support layers stay supportive.
5. **Listener gate reflects musical truth**
   - Penalize medley behavior (`back-and-forth`) directly in candidate ranking.

## Revamp Phases

### Phase 1 — Stop medley behavior (in progress)
- [x] Penalize medley-like scoring patterns in `_pro_fusion_selection_score`:
  - `full_mix_medley_risk`
  - low `integrated_two_parent_section_ratio`
  - high `max_parent_share`
  - high `owner_switch_ratio`
- [x] Tighten same-parent payoff overlap to reduce payoff seam crowding.
- [x] Baseline combo fallback now prefers core-shape combo (`verse/payoff`) over intro-donor fallback when core donor is unavailable.
- Success criteria:
  - top-selected candidate no longer defaults to intro-led back-and-forth combos.

### Phase 2 — Enforce phrase-continuity composition constraints
- [ ] Add hard arrangement constraints limiting owner switches per phrase block.
- [ ] Require at least one sustained cross-parent integrated section (not just section-level alternation).
- [ ] Add tests proving continuity constraints survive shortlist and render resolution.
- Success criteria:
  - `integrated_two_parent_section_ratio` rises above 0 for winning candidates.

### Phase 3 — Donor role realism and vocal collision control
- [ ] Add role-aware donor admission (drums/bass/vocal support eligibility by section type).
- [ ] Tighten vocal-policy ownership rules at high-risk seams.
- [ ] Add diagnostics so failed candidates report exact role-collision reasons.
- Success criteria:
  - lower vocal competition evidence in listen reports; higher song_likeness stability.

### Phase 4 — Promotion policy realignment
- [ ] Add explicit anti-medley floor in pass+floor eligibility (for promotion only).
- [ ] Require cohesion metrics to pass before artifact promotion.
- Success criteria:
  - first promoted candidate meets hard floor **and** anti-medley constraints.

## Execution Mode
- One concrete change per cycle.
- Every cycle must include:
  1) change, 2) tests, 3) one benchmark run, 4) metric delta capture.
