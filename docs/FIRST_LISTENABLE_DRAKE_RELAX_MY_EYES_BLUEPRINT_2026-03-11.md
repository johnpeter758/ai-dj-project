# First listenable render blueprint: Drake + Relax My Eyes

_Date: 2026-03-11_

## Parents
- **A / Drake-side anchor:** `PARTYNEXTDOOR & DRAKE - SOMEBODY LOVES ME.mp3`
  - BPM: **120.19**
  - Key estimate: **12B / E major**
- **B / donor / alternate anchor:** `Relax My Eyes.mp3`
  - BPM: **132.51**
  - Key estimate: **5B / D# major**

## Render stance for v1
- Make the first pass **Drake-led and section-deterministic**.
- Because tempo gap is ~**+10.3%** and key match is weak, avoid long full-mix sustained blends.
- Use **hard ownership per section** with only short donor accents, filtered beds, or percussion where noted.
- Prefer **swap / drop / lift** transitions over dense 8-bar overlays.

## Deterministic section blueprint

| # | Bars | Label | Parent owner | Anchor BPM | Target energy | Transition in | Transition out | Vocal policy | Donor usage policy |
|---|---:|---|---|---:|---:|---|---|---|---|
| 1 | 0-8 | Intro bed | **B** | 132.51 | 0.22 | none | lift | **No lead vocals.** Allow only tiny Drake ad-lib tail in bars 7-8 if available; otherwise instrumental only. | Use **B full mix as base**. No A full-mix overlay. Optional A one-shot vocal pickup only in last 2 bars. |
| 2 | 8-16 | Drake arrival | **A** | 120.19 | 0.42 | lift | swap | **A lead vocal allowed.** B vocals muted. | Use **A full mix as base**. B limited to filtered texture/percussion in bars 8-12 only, then clear out by bar 14. |
| 3 | 16-24 | Hybrid tension build | **A** | 120.19 | 0.58 | swap | drop | **A lead remains primary.** No simultaneous B lead vocal. | Keep **A base**. Allow **B instrumental/percussive donor only**, high-passed or band-limited, max 4-bar phrases. No wide full-spectrum B overlay. |
| 4 | 24-32 | Relax My Eyes release | **B** | 132.51 | 0.72 | drop | swap | **B lead vocal allowed.** A lead muted; only A ad-lib or tail fragments if clean. | Use **B full mix as base** for a clean contrast section. A may donate short riser, impact, or low-level rhythmic support only in bars 30-32. |
| 5 | 32-48 | Drake payoff / outro hold | **A** | 120.19 | 0.84 → 0.68 | swap | none | **A lead vocal dominant** through bar 40, then reduce density for exit. B vocals fully muted. | Use **A full mix as base**. B donor only as intro/outro atmosphere or percussion glue in first 4 bars of the section; remove by bar 40 so ending resolves clearly to A. |

## Why this shape is the safest first render
1. **Only one owner at a time.** The section renderer can treat each section as one anchor parent with optional donor layers.
2. **Two explicit contrast moments.** Section 1 establishes B color; Section 4 gives B a true spotlight without forcing long mixed-vocal overlays.
3. **Drake gets the main narrative spine.** Sections 2, 3, and 5 make the output feel intentionally Drake-led instead of randomly alternating songs.
4. **Transitions stay realistic for current analysis quality.** With phrase boundaries available but weak section segmentation, short ownership swaps are safer than clever continuous morphing.

## Renderer-ready implementation notes
- Treat each section's **owner** as the `source_parent` / anchor parent.
- Derive target duration strictly from the listed bar spans.
- If donor material cannot be extracted cleanly at phrase-safe boundaries, **drop donor usage entirely** rather than forcing a messy overlay.
- For sections 2, 3, and 5, if A vocal extraction is messy, keep donor policy unchanged and simply reduce B layers further.
- For section 4, if B vocal entry is not clean, keep B as owner but use an instrumental-only release instead of mixed leads.

## Suggested section-level resolver config
- **Section 1:** owner B, donor A optional, low-density intro, no dense blend.
- **Section 2:** owner A, donor B filtered-only, transition length 2-4 beats.
- **Section 3:** owner A, donor B percussion/texture only, no donor lead vocal.
- **Section 4:** owner B, donor A transition support only, clean contrast section.
- **Section 5:** owner A, donor B only in first 4 bars, final 8 bars A-only if possible.

## Bottom line
For the first listenable render, do **not** chase equal 50/50 fusion. Make it a **Drake-led child arrangement with one real Relax My Eyes spotlight section and controlled donor coloration elsewhere**. That is the highest-probability path to something coherent with the repo's current deterministic renderer design.
