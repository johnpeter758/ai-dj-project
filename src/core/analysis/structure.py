from __future__ import annotations

import librosa
import numpy as np


def estimate_structure(audio: np.ndarray, sample_rate: int, hop_length: int = 512) -> dict:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length)
    novelty = np.mean(tempogram, axis=0)
    novelty = novelty / (np.max(novelty) + 1e-8)

    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)

    phrase_boundaries = []
    beats_per_bar = 4
    bars_per_phrase = 4
    beats_per_phrase = beats_per_bar * bars_per_phrase
    for i in range(0, len(beat_times), beats_per_phrase):
        phrase_boundaries.append(float(beat_times[i]))

    boundary_frames = librosa.util.peak_pick(novelty, pre_max=8, post_max=8, pre_avg=16, post_avg=16, delta=0.10, wait=16)
    section_times = librosa.frames_to_time(boundary_frames, sr=sample_rate, hop_length=hop_length)

    sections = []
    duration = float(len(audio) / sample_rate)
    if len(section_times) == 0:
        sections.append({"start": 0.0, "end": duration, "label": "section_0"})
    else:
        starts = [0.0] + section_times.tolist()
        ends = section_times.tolist() + [duration]
        for idx, (start, end) in enumerate(zip(starts, ends)):
            if end > start:
                sections.append({"start": float(start), "end": float(end), "label": f"section_{idx}"})

    return {
        "tempo_reference_bpm": float(tempo),
        "phrase_boundaries_seconds": phrase_boundaries,
        "section_boundaries_seconds": section_times.tolist(),
        "sections": sections,
    }
