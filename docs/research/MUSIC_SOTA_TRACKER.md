# VocalFusion Music R&D Tracker

Generated: 2026-03-26 22:30:27 EDT

This report is generated automatically so implementation work stays anchored to current music-tech practice.

## 1) High-signal open-source repos (engineering benchmarks)

- **facebookresearch/audiocraft** (23126★, updated 2026-03-03)
  - What it is: Audiocraft is a library for audio processing and generation with deep learning. It features the state-of-the-art EnCodec audio compressor / tokenizer, along with MusicGen, a simple and controllable music generation LM with textual and melodic conditioning.
  - Why it matters: Reference implementation for modern controllable audio/music generation systems.
  - VocalFusion integration: Reuse conditioning interfaces and token-level control ideas for section-role constraints.
- **facebookresearch/demucs** (9895★, updated 2024-04-24)
  - What it is: Code for the paper Hybrid Spectrogram and Waveform Source Separation
  - Why it matters: Strong practical baseline for stem-quality separation and mashup-safe layer isolation.
  - VocalFusion integration: Use separated stems to build integrated support layers instead of section-level hard swaps.
- **spotify/basic-pitch** (4808★, updated 2025-11-13)
  - What it is: A lightweight yet powerful audio-to-MIDI converter with pitch bend detection
  - Why it matters: Fast melody/pitch extraction suitable for phrase compatibility checks.
  - VocalFusion integration: Penalize donor candidates with conflicting melodic contour during dense vocal windows.
- **openvpi/DiffSinger** (3084★, updated 2026-03-26)
  - What it is: An advanced singing voice synthesis system with high fidelity, expressiveness, controllability and flexibility based on DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
  - Why it matters: Mature open-source singing synthesis pipeline for melody/vocal conditioning ideas.
  - VocalFusion integration: Borrow phrase-conditioned vocal continuity priors for payoff transitions.
- **CPJKU/madmom** (1608★, updated 2026-03-20)
  - What it is: Python audio and music signal processing library
  - Why it matters: Robust beat/downbeat stack used in production-level MIR pipelines.
  - VocalFusion integration: Tighten section-entry/downbeat alignment before support-overlay admission.

## 2) Recent paper scan (OpenAlex title-filtered)

### Music Source Separation
- 2026 — Hybrid Transformers for Music Source Separation (International Journal for Research in Applied Science and Engineering Technology); cites=0; ref=https://doi.org/10.22214/ijraset.2026.78294
- 2026 — Music Source Restoration with Ensemble Separation and Targeted Reconstruction (ArXiv.org); cites=0; ref=https://openalex.org/W7139147523
- 2026 — Multi-Stage Music Source Restoration with BandSplit-RoFormer Separation and HiFi++ GAN (arXiv (Cornell University)); cites=0; ref=https://doi.org/10.48550/arxiv.2603.04032
- 2026 — How Far Can a U-Net Go? An Empirical Analysis of Music Source Separation Performance (Applied Sciences); cites=0; ref=https://doi.org/10.3390/app16052195
- 2026 — A Knowledge-Driven Approach to Music Segmentation, Music Source Separation and Cinematic Audio Source Separation (arXiv (Cornell University)); cites=0; ref=https://doi.org/10.48550/arxiv.2602.21476

### Music Structure Segmentation
- 2026 — EDMFormer: Genre-Specific Self-Supervised Learning for Music Structure Segmentation (arXiv (Cornell University)); cites=0; ref=https://doi.org/10.48550/arxiv.2603.08759
- 2026 — Music-Structure Segmentation in Balinese Gamelan (Tabuh Lelambatan) with SSM, Checkerboard Novelty, and HMM (SinkrOn); cites=0; ref=https://doi.org/10.33395/sinkron.v10i1.15494
- 2025 — Improving Phrase Segmentation in Symbolic Folk Music: A Hybrid Model with Local Context and Global Structure Awareness (Entropy); cites=1; ref=https://doi.org/10.3390/e27050460
- 2023 — Pitchclass2vec: Symbolic Music Structure Segmentation with Chord Embeddings (arXiv (Cornell University)); cites=2; ref=https://doi.org/10.48550/arxiv.2303.15306
- 2023 — Convolutive Block-Matching Segmentation Algorithm with Application to Music Structure Analysis; cites=1; ref=https://doi.org/10.1109/waspaa58266.2023.10248174

### Audio Mashup Generation
- 2025 — Music mashup generation and trend prediction using hybrid attention net (The Journal of the Acoustical Society of America); cites=0; ref=https://doi.org/10.1121/10.0037391
- 2024 — Graph Neural Network Guided Music Mashup Generation; cites=0; ref=https://doi.org/10.1109/bigdata62323.2024.10825542

### Vocal Mixture Disentanglement
- 2025 — Singing Voice Separation From Carnatic Music Mixtures Using a Regression-Guided Latent Diffusion Model (Zenodo (CERN European Organization for Nuclear Research)); cites=0; ref=https://doi.org/10.5281/zenodo.17706605
- 2024 — Facing the Music: Tackling Singing Voice Separation in Cinematic Audio Source Separation (arXiv (Cornell University)); cites=0; ref=https://doi.org/10.48550/arxiv.2408.03588
- 2023 — Exploiting Music Source Separation For Singing Voice Detection; cites=2; ref=https://doi.org/10.1109/mlsp55844.2023.10285863

## 3) Direct coding implications for current bottleneck

Current bottleneck: medley-like section alternation and adaptive gate reject despite occasional high song-likeness.

Highest-value research-aligned actions:
1. **Stem-aware integrated overlays by default for major sections**
   - Move beyond ownership swaps to simultaneous backbone+donor support layers.
2. **Phrase/downbeat hard alignment before adaptive admission**
   - Reject adaptive swaps/supports that violate downbeat confidence windows.
3. **Owner-switch minimization objective in shortlist ranking**
   - Penalize variants with high owner_switch_ratio unless integration ratio also rises.
4. **Vocal-collision guard at payoff seams**
   - Use melody/voicing confidence to suppress donor vocal content when backbone lead is active.

## 4) Next implementation checkpoint

- Implement adaptive transition-gate rescue with beat-locked support overlays and rerun pair2 benchmark.
- Keep this file regenerated each major cycle (`python scripts/generate_music_research_brief.py`).
