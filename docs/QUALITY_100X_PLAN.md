# VocalFusion 100x Quality Plan (Execution)

Date: 2026-03-25
Status: In progress (executing)

## Target
Move from "passing outputs" to consistently human-approved, song-like full mixes.

## Hard acceptance bar (all required)
1. Listener gate: `pass`
2. Song-likeness >= 65
3. Transition >= 65
4. Human quick-listen: no obvious stitched/medley feel in first 30s
5. Reproducible across multiple pairings

## Phase 1 — Reliability + selection hardening (DONE)
- [x] Expand pro-mode candidate search (adaptive + baseline + variants)
- [x] Make winner selection gate-aware (reject/review/pass impact)
- [x] Add pass-first winner policy
- [x] Add quality floors to promotion logic (song-likeness/groove/structure/readability)
- [x] Add seam diagnostics into `fusion_selection.json`
- [x] Fix planner crash paths for empty ranked-candidate scenarios

Shipped commits:
- `c6b04e3` — stronger pro candidate search + gating-aware selection
- `3f8455c` — pass-first winner policy + roadmap foundation
- `f710d83` — quality floors + seam diagnostics
- `0fdbb80` — planner empty-ranked-candidate crash fixes

## Phase 2 — Musical quality uplift (IN PROGRESS)
- [x] Add section-diverse variant generation (not only top-single swaps)
- [x] Add multi-swap safe variants to improve macro arc options
- [x] Bias winner ranking toward transition clarity + song-likeness under pass gate
- [x] Add targeted seam-risk penalty in final candidate sorting

Shipped commits (phase 2):
- `this checkpoint commit` — section-diverse shortlist generation + safe dual-section variants

## Phase 3 — Full-mix production loop
- [ ] Run standardized full-length benchmark batch on priority pairings
- [ ] Keep only pass + floor winners
- [ ] Promote best full mix artifact and attach listen report automatically
- [ ] Track regressions in daily quality log

## Current latest full mix artifact
- `/Users/johnpeter/Music/AI_DJ_Output/full_mix_20260325_230329_relax_x_treasure_ADAPTIVE_FIXED.mp3`
- Overall: `69.3` | Gate: `pass` | Song-likeness: `49.2` | Transition: `49.7`

## Immediate next execution block
1. Implement section-diverse + multi-swap variant generation.
2. Re-run pro full-length on Relax × Treasure.
3. Promote best candidate only if it beats current artifact on song-likeness + transition.
