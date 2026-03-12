from __future__ import annotations

import librosa
import numpy as np


def _safe_float_list(values) -> list[float]:
    return [float(v) for v in values]


def _phrase_boundaries_from_beats(beat_times: np.ndarray, beats_per_phrase: int) -> list[float]:
    if beat_times.size == 0:
        return [0.0]
    boundaries = [float(beat_times[i]) for i in range(0, len(beat_times), beats_per_phrase)]
    if not boundaries or boundaries[0] > 1e-6:
        boundaries = [0.0, *boundaries]
    return sorted(set(boundaries))


def estimate_structure(audio: np.ndarray, sample_rate: int, hop_length: int = 512) -> dict:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length)
    novelty = np.mean(tempogram, axis=0)
    novelty = novelty / (np.max(novelty) + 1e-8)

    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    beat_times = np.asarray(beat_times, dtype=float)

    beats_per_bar = 4
    bars_per_phrase = 4
    beats_per_phrase = beats_per_bar * bars_per_phrase
    phrase_boundaries = _phrase_boundaries_from_beats(beat_times, beats_per_phrase)

    boundary_frames = librosa.util.peak_pick(novelty, pre_max=8, post_max=8, pre_avg=16, post_avg=16, delta=0.10, wait=16)
    novelty_boundaries = _safe_float_list(librosa.frames_to_time(boundary_frames, sr=sample_rate, hop_length=hop_length))

    duration = float(len(audio) / sample_rate)
    candidate_boundaries = sorted({b for b in phrase_boundaries + novelty_boundaries if 0.0 < float(b) < duration})

    # Promote phrase boundaries into coarse structural sections so planner has musically legal chunks
    # even when novelty peaks are sparse or unreliable.
    section_boundaries = phrase_boundaries[1:]
    if not section_boundaries:
        section_boundaries = candidate_boundaries

    sections = []
    starts = [0.0] + [float(b) for b in section_boundaries if 0.0 < float(b) < duration]
    ends = starts[1:] + [duration]
    for idx, (start, end) in enumerate(zip(starts, ends)):
        if end - start < 1.0:
            continue
        sections.append({'start': float(start), 'end': float(end), 'label': f'section_{idx}'})

    if not sections:
        sections.append({'start': 0.0, 'end': duration, 'label': 'section_0'})

    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])

    return {
        'tempo_reference_bpm': tempo_bpm,
        'phrase_boundaries_seconds': phrase_boundaries,
        'section_boundaries_seconds': [float(s['start']) for s in sections[1:]],
        'novelty_boundaries_seconds': novelty_boundaries,
        'sections': sections,
    }
