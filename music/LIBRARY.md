# AI DJ Music Library

## Overview
- **Total Songs:** 13
- **Purpose:** Training data for AI music fusion engine
- **Source:** YouTube audio downloads

## Song Library

| # | Filename | Artist | Genre | BPM | Key | Energy |
|---|----------|--------|-------|-----|-----|--------|
| 1 | drake_in_my_feelings | Drake | Hip-Hop | 185 | 2A | 0.150 |
| 2 | travis_sicko_mode | Travis Scott | Trap | 152 | 3A | 0.267 |
| 3 | travis_butterfly_effect | Travis Scott | Trap | 96 | 2A | - |
| 4 | drake_fair_trade | Drake | Hip-Hop | 172 | 4A | - |
| 5 | marshmello_happier | Marshmello | EDM | 99 | 6A | 0.124 |
| 6 | rick_roll | Rick Astley | Pop | 112 | 9B | 0.129 |
| 7 | daft_punk_doin_it_right | Daft Punk | Electronic | 89 | 9B | - |
| 8 | meduza_lose_control | MEDUZA | EDM | 123 | 1B | - |
| 9 | odesza_a_moment_apart | ODESZA | Electronic | 117 | 8B | - |
| 10 | edm_1 | Various | EDM | 129 | 6A | 0.296 |
| 11 | edm_2 | Various | EDM | 96 | 6A | 0.198 |
| 12 | edm_3 | Various | EDM | 123 | 8B | 0.250 |
| 13 | edm_4 | Various | EDM | 99 | 6A | 0.303 |

## Key Groups (for compatible mixing)

- **2A (C# minor):** Drake In My Feelings, Travis Butterfly Effect
- **3A (D minor):** Travis Sicko Mode
- **4A (D# minor):** Drake Fair Trade
- **6A (F minor):** Marshmello Happier, EDM 1,2,4
- **8B (G major):** ODESZA, EDM 3
- **9B (G# major):** Rick Astley, Daft Punk

## Compatible Pairs (Score > 0.6)

| Pair | Score |
|------|-------|
| Drake In My Feelings + Travis Butterfly Effect | 0.66 |
| Marshmello Happier + EDM 1 | 0.65 |

## Storage

- **Location:** `/ai-dj-project/music/`
- **Format:** WAV (22kHz, stereo)
- **Total Size:** ~400MB

## Notes

- Analyze songs using: `python src/fusion_engine_v2.py analyze <file>`
- Check compatibility: `python src/fusion_engine_v2.py compatibility <song_a> <song_b>`
- Create fusion: `python src/fusion_engine_v2.py fuse <song_a> <song_b>`

## Last Updated

2026-03-06
