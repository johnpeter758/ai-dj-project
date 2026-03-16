from __future__ import annotations

import librosa
import numpy as np

from .tempo import beat_grid_metrics


def _safe_float_list(values) -> list[float]:
    return [float(v) for v in values]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 1e-6)
    return (values - low) / span


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, 0.0, 1.0))


def _phrase_boundaries_from_beats(beat_times: np.ndarray, beats_per_phrase: int) -> list[float]:
    if beat_times.size == 0:
        return [0.0]
    boundaries = [float(beat_times[i]) for i in range(0, len(beat_times), beats_per_phrase)]
    if not boundaries or boundaries[0] > 1e-6:
        boundaries = [0.0, *boundaries]
    return sorted(set(boundaries))


def _regularized_phrase_boundaries(duration: float, beats_per_phrase: int, tempo_bpm: float) -> list[float]:
    if duration <= 0.0:
        return [0.0]

    if tempo_bpm > 0.0:
        phrase_seconds = (60.0 / tempo_bpm) * beats_per_phrase
    else:
        phrase_seconds = 8.0
    phrase_seconds = float(np.clip(phrase_seconds, 4.0, 16.0))

    boundaries = [0.0]
    cursor = phrase_seconds
    while cursor < duration - 1e-6:
        boundaries.append(round(float(cursor), 3))
        cursor += phrase_seconds
    return sorted(set(boundaries))


def _select_phrase_boundaries(
    beat_times: np.ndarray,
    duration: float,
    beats_per_phrase: int,
    tempo_bpm: float,
    beat_grid_confidence: float,
) -> tuple[list[float], str]:
    beat_phrase_boundaries = _phrase_boundaries_from_beats(beat_times, beats_per_phrase)
    beat_span = max(0.0, float(beat_times[-1] - beat_times[0])) if beat_times.size >= 2 else 0.0
    weak_beat_grid = (
        beat_grid_confidence < 0.45
        or beat_times.size < max(8, beats_per_phrase // 2)
        or beat_span < max(duration * 0.45, 8.0)
    )
    if weak_beat_grid:
        return _regularized_phrase_boundaries(duration, beats_per_phrase, tempo_bpm), 'coarse_regularized_grid'
    return beat_phrase_boundaries, 'beat_phrase_grid'


def _peak_boundaries_from_novelty(
    novelty: np.ndarray,
    times: np.ndarray,
    duration: float,
    delta: float,
    wait: int,
) -> list[float]:
    novelty = _safe_normalize(np.asarray(novelty, dtype=float).reshape(-1))
    times = np.asarray(times, dtype=float).reshape(-1)
    if novelty.size == 0 or times.size == 0:
        return []

    pre_max = max(1, min(8, novelty.size // 12 or 1))
    post_max = pre_max
    pre_avg = max(pre_max + 1, min(16, novelty.size // 6 or 2))
    post_avg = pre_avg
    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=max(1, wait),
    )
    boundaries = [
        float(times[idx])
        for idx in peaks
        if 0 <= idx < times.size and 0.0 < float(times[idx]) < duration
    ]
    return sorted(set(boundaries))


def _checkerboard_kernel(size: int) -> np.ndarray:
    size = max(4, int(size))
    if size % 2:
        size += 1
    axis = np.linspace(-1.0, 1.0, size, endpoint=False) + (1.0 / size)
    signs = np.where(np.outer(axis, axis) >= 0.0, 1.0, -1.0)
    window = np.outer(np.hanning(size), np.hanning(size))
    kernel = signs * window
    norm = float(np.sum(np.abs(kernel)))
    return kernel / max(norm, 1e-6)


def _foote_novelty_curve(affinity: np.ndarray) -> np.ndarray:
    affinity = np.asarray(affinity, dtype=float)
    if affinity.ndim != 2 or affinity.shape[0] < 4:
        return np.zeros(int(affinity.shape[0]) if affinity.ndim == 2 else 0, dtype=float)

    max_even = affinity.shape[0] if affinity.shape[0] % 2 == 0 else affinity.shape[0] - 1
    kernel_size = int(np.clip((affinity.shape[0] // 4) * 2, 4, max(4, min(32, max_even))))
    if kernel_size > max_even:
        kernel_size = max_even
    if kernel_size < 4:
        return np.zeros(affinity.shape[0], dtype=float)

    kernel = _checkerboard_kernel(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(affinity, ((pad, pad), (pad, pad)), mode='edge')

    novelty = np.zeros(affinity.shape[0], dtype=float)
    for idx in range(novelty.size):
        window = padded[idx:idx + kernel_size, idx:idx + kernel_size]
        novelty[idx] = max(0.0, float(np.sum(window * kernel)))
    return _safe_normalize(novelty)


def _self_similarity_boundaries(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    beat_frames: np.ndarray,
    duration: float,
) -> list[float]:
    beat_frames = np.asarray(beat_frames, dtype=int).reshape(-1)
    if beat_frames.size < 6:
        return []

    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=hop_length)
    frame_count = chroma.shape[1]
    if frame_count < 8:
        return []

    beat_boundaries = np.unique(np.clip(np.concatenate(([0], beat_frames, [frame_count])), 0, frame_count))
    if beat_boundaries.size < 4:
        return []

    beat_chroma = librosa.util.sync(chroma, beat_boundaries, aggregate=np.median)
    if beat_chroma.shape[1] < 4:
        return []

    affinity = librosa.segment.recurrence_matrix(
        beat_chroma,
        mode='affinity',
        metric='cosine',
        sym=True,
        width=1,
    )
    novelty = _foote_novelty_curve(affinity)
    segment_times = librosa.frames_to_time(beat_boundaries[:-1], sr=sample_rate, hop_length=hop_length)
    return _peak_boundaries_from_novelty(
        novelty,
        segment_times,
        duration=duration,
        delta=0.08,
        wait=max(1, beat_chroma.shape[1] // 8),
    )


def _merge_boundary_evidence(
    duration: float,
    phrase_boundaries: list[float],
    tempogram_boundaries: list[float],
    self_similarity_boundaries: list[float],
    beat_grid_confidence: float,
    phrase_boundary_method: str,
) -> list[dict[str, float | list[str]]]:
    phrase_weight = 0.34 if phrase_boundary_method == 'beat_phrase_grid' else 0.16
    source_weights = {
        'phrase_grid': phrase_weight,
        'tempogram_novelty': 0.26,
        'self_similarity': 0.42,
    }

    items: list[dict[str, float | str]] = []
    for boundary in phrase_boundaries[1:]:
        if 0.0 < float(boundary) < duration:
            items.append({'time': float(boundary), 'source': 'phrase_grid'})
    for boundary in tempogram_boundaries:
        if 0.0 < float(boundary) < duration:
            items.append({'time': float(boundary), 'source': 'tempogram_novelty'})
    for boundary in self_similarity_boundaries:
        if 0.0 < float(boundary) < duration:
            items.append({'time': float(boundary), 'source': 'self_similarity'})

    if not items:
        return []

    tolerance = max(duration * 0.02, 0.35)
    items.sort(key=lambda item: float(item['time']))
    groups: list[list[dict[str, float | str]]] = []
    for item in items:
        if not groups or abs(float(item['time']) - float(groups[-1][-1]['time'])) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)

    merged: list[dict[str, float | list[str]]] = []
    for group in groups:
        weights = np.asarray([source_weights[str(item['source'])] for item in group], dtype=float)
        times = np.asarray([float(item['time']) for item in group], dtype=float)
        merged_time = float(np.average(times, weights=weights))
        sources = sorted({str(item['source']) for item in group})
        spread = float(np.average(np.abs(times - merged_time), weights=weights)) if times.size else 0.0
        source_strength = sum(max(source_weights[source] for item in group if str(item['source']) == source) for source in sources)
        proximity_bonus = 1.0 - min(1.0, spread / tolerance)
        confidence = _clamp01((0.70 * source_strength) + (0.30 * proximity_bonus))
        if 'phrase_grid' in sources:
            confidence *= 0.70 + (0.30 * beat_grid_confidence)
        merged.append({
            'time': round(merged_time, 3),
            'confidence': round(_clamp01(confidence), 3),
            'sources': sources,
        })

    return merged


def _boundary_support_at_time(
    boundary_confidences: list[dict[str, float | list[str]]],
    target_time: float,
    tolerance: float,
) -> float:
    best = 0.0
    for item in boundary_confidences:
        time = float(item['time'])
        delta = abs(time - float(target_time))
        if delta > tolerance:
            continue
        proximity = 1.0 - (delta / max(tolerance, 1e-6))
        confidence = float(item['confidence'])
        best = max(best, confidence * (0.60 + (0.40 * proximity)))
    return _clamp01(best)


def _snap_phrase_boundaries_to_evidence(
    phrase_boundaries: list[float],
    boundary_confidences: list[dict[str, float | list[str]]],
    duration: float,
) -> list[float]:
    if not phrase_boundaries:
        return [0.0]
    if not boundary_confidences:
        return sorted(set(float(boundary) for boundary in phrase_boundaries))

    tolerance = max(duration * 0.02, 0.60)
    snapped = [0.0]
    for boundary in phrase_boundaries[1:]:
        boundary = float(boundary)
        if not 0.0 < boundary < duration:
            continue
        nearby = [
            item for item in boundary_confidences
            if abs(float(item['time']) - boundary) <= tolerance
        ]
        if not nearby:
            snapped.append(round(boundary, 3))
            continue

        best = max(
            nearby,
            key=lambda item: float(item['confidence']) - 0.30 * (abs(float(item['time']) - boundary) / tolerance),
        )
        if float(best['confidence']) >= 0.35:
            snapped.append(round(float(best['time']), 3))
        else:
            snapped.append(round(boundary, 3))
    return sorted(set(snapped))


def _section_frame_slice(frame_times: np.ndarray, start: float, end: float) -> slice:
    start_idx = int(np.searchsorted(frame_times, float(start), side='left'))
    end_idx = int(np.searchsorted(frame_times, float(end), side='left'))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, frame_times.size)
    return slice(start_idx, end_idx)


def _compute_section_features(
    sections: list[dict[str, float | str]],
    frame_times: np.ndarray,
    rms: np.ndarray,
    flatness: np.ndarray,
    chroma: np.ndarray,
    mfcc: np.ndarray,
) -> list[dict[str, float | np.ndarray]]:
    features: list[dict[str, float | np.ndarray]] = []
    if frame_times.size == 0:
        return features

    for section in sections:
        start = float(section['start'])
        end = float(section['end'])
        span = _section_frame_slice(frame_times, start, end)

        sec_rms = np.asarray(rms[span], dtype=float)
        sec_flatness = np.asarray(flatness[span], dtype=float)
        sec_chroma = np.asarray(chroma[:, span], dtype=float)
        sec_mfcc = np.asarray(mfcc[:, span], dtype=float)

        chroma_mean = np.mean(sec_chroma, axis=1) if sec_chroma.size else np.zeros(chroma.shape[0], dtype=float)
        chroma_norm = chroma_mean / max(float(np.linalg.norm(chroma_mean)), 1e-6)
        mfcc_mean = np.mean(sec_mfcc, axis=1) if sec_mfcc.size else np.zeros(mfcc.shape[0], dtype=float)
        mfcc_centered = mfcc_mean - float(np.mean(mfcc_mean))
        mfcc_norm = mfcc_centered / max(float(np.linalg.norm(mfcc_centered)), 1e-6)

        features.append({
            'mean_rms': float(np.mean(sec_rms)) if sec_rms.size else 0.0,
            'mean_flatness': float(np.mean(sec_flatness)) if sec_flatness.size else 1.0,
            'chroma': chroma_norm,
            'mfcc': mfcc_norm,
        })
    return features


def _infer_section_role_hints(section_features: list[dict[str, float | np.ndarray]]) -> list[dict[str, float | str | bool]]:
    if not section_features:
        return []

    energy = _safe_normalize(np.asarray([float(item['mean_rms']) for item in section_features], dtype=float))
    timbre_clarity = _safe_normalize(
        np.asarray([1.0 - float(item['mean_flatness']) for item in section_features], dtype=float)
    )
    section_count = len(section_features)

    repetition_scores: list[float] = []
    repeat_counts: list[int] = []
    adjacency_penalties: list[float] = []

    for idx, item in enumerate(section_features):
        chroma_vec = np.asarray(item['chroma'], dtype=float)
        mfcc_vec = np.asarray(item['mfcc'], dtype=float)
        row: list[float] = []
        repeat_count = 0
        best_nonlocal = 0.0
        best_adjacent = 0.0

        for other_idx, other in enumerate(section_features):
            if idx == other_idx:
                continue
            other_chroma = np.asarray(other['chroma'], dtype=float)
            other_mfcc = np.asarray(other['mfcc'], dtype=float)
            chroma_sim = _cosine_similarity(chroma_vec, other_chroma)
            timbre_sim = _cosine_similarity(mfcc_vec, other_mfcc)
            similarity = float(np.clip((0.62 * chroma_sim) + (0.38 * timbre_sim), 0.0, 1.0))
            row.append(similarity)
            if similarity >= 0.88:
                repeat_count += 1
            if abs(idx - other_idx) <= 1:
                best_adjacent = max(best_adjacent, similarity)
            else:
                best_nonlocal = max(best_nonlocal, similarity)

        if best_nonlocal == 0.0 and row and section_count <= 2:
            best_nonlocal = max(row)
        repetition_scores.append(best_nonlocal)
        repeat_counts.append(repeat_count)
        adjacency_penalties.append(float(np.clip(best_adjacent - best_nonlocal, 0.0, 1.0)))

    repetition = _safe_normalize(np.asarray(repetition_scores, dtype=float))
    if max(repeat_counts, default=0) > 0:
        repeat_density = _safe_normalize(np.asarray(repeat_counts, dtype=float))
    else:
        repeat_density = np.zeros(section_count, dtype=float)

    hints: list[dict[str, float | str | bool]] = []
    chorus_scores: list[float] = []
    for idx in range(section_count):
        position = idx / max(section_count - 1, 1)
        late_bias = float(np.clip((position - 0.20) / 0.80, 0.0, 1.0))
        chorus_score = float(np.clip(
            0.38 * repetition[idx]
            + 0.24 * energy[idx]
            + 0.14 * timbre_clarity[idx]
            + 0.12 * repeat_density[idx]
            + 0.12 * late_bias
            - 0.12 * adjacency_penalties[idx],
            0.0,
            1.0,
        ))
        chorus_scores.append(chorus_score)
        hints.append({
            'repetition': round(float(repetition[idx]), 3),
            'energy': round(float(energy[idx]), 3),
            'timbre_clarity': round(float(timbre_clarity[idx]), 3),
            'repeat_density': round(float(repeat_density[idx]), 3),
            'adjacent_repeat_penalty': round(float(adjacency_penalties[idx]), 3),
            'chorus_likelihood': round(chorus_score, 3),
            'role_hint': 'section_like',
            'is_chorus_candidate': False,
        })

    top_score = max(chorus_scores, default=0.0)
    primary_idx = int(np.argmax(chorus_scores)) if chorus_scores else -1
    primary_profile = np.asarray(section_features[primary_idx]['chroma'], dtype=float) if primary_idx >= 0 else np.asarray([], dtype=float)

    for idx, hint in enumerate(hints):
        qualifies = (
            chorus_scores[idx] >= 0.58
            and repetition[idx] >= 0.45
            and energy[idx] >= 0.35
        )
        if primary_idx >= 0 and idx != primary_idx and primary_profile.size:
            profile = np.asarray(section_features[idx]['chroma'], dtype=float)
            profile_match = _cosine_similarity(primary_profile, profile)
            qualifies = qualifies or (
                top_score >= 0.60
                and chorus_scores[idx] >= max(0.50, top_score - 0.16)
                and profile_match >= 0.90
                and energy[idx] >= 0.35
            )

        if qualifies:
            hint['role_hint'] = 'chorus_like'
            hint['is_chorus_candidate'] = True
        elif idx == 0 and energy[idx] <= 0.35:
            hint['role_hint'] = 'intro_like'
        elif idx == section_count - 1 and energy[idx] <= 0.45:
            hint['role_hint'] = 'outro_like'
        elif repetition[idx] >= 0.35 and energy[idx] < max(0.55, top_score):
            hint['role_hint'] = 'verse_like'

    return hints


def _annotate_sections_with_role_hints(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    sections: list[dict[str, float | str]],
) -> tuple[list[dict[str, float | str | bool | dict[str, float]]], list[dict[str, float | str | bool | dict[str, float]]]]:
    if not sections:
        return [], []

    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=hop_length, n_mfcc=8)
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)

    section_features = _compute_section_features(sections, frame_times, rms, flatness, chroma, mfcc)
    role_hints = _infer_section_role_hints(section_features)

    annotated_sections: list[dict[str, float | str | bool | dict[str, float]]] = []
    chorus_candidates: list[dict[str, float | str | bool | dict[str, float]]] = []
    for section, hint in zip(sections, role_hints):
        role_scores = {
            'chorus_likelihood': float(hint['chorus_likelihood']),
            'repetition': float(hint['repetition']),
            'energy': float(hint['energy']),
            'timbre_clarity': float(hint['timbre_clarity']),
        }
        annotated = {
            **section,
            'role_hint': str(hint['role_hint']),
            'role_scores': role_scores,
            'is_chorus_candidate': bool(hint['is_chorus_candidate']),
        }
        annotated_sections.append(annotated)
        if annotated['is_chorus_candidate']:
            chorus_candidates.append(annotated)

    return annotated_sections, chorus_candidates


def estimate_structure(audio: np.ndarray, sample_rate: int, hop_length: int = 512) -> dict:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length)
    novelty = np.mean(tempogram, axis=0)
    novelty = novelty / (np.max(novelty) + 1e-8)

    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    beat_times = np.asarray(beat_times, dtype=float)
    beat_frames = np.asarray(beat_frames, dtype=int)

    duration = float(len(audio) / sample_rate)
    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
    beat_metrics = beat_grid_metrics(beat_times, duration=duration)

    beats_per_bar = 4
    bars_per_phrase = 4
    beats_per_phrase = beats_per_bar * bars_per_phrase
    phrase_boundaries, phrase_boundary_method = _select_phrase_boundaries(
        beat_times=beat_times,
        duration=duration,
        beats_per_phrase=beats_per_phrase,
        tempo_bpm=tempo_bpm,
        beat_grid_confidence=float(beat_metrics['confidence']),
    )

    frame_times = librosa.frames_to_time(np.arange(novelty.size), sr=sample_rate, hop_length=hop_length)
    tempogram_boundaries = _peak_boundaries_from_novelty(
        novelty,
        frame_times,
        duration=duration,
        delta=0.10,
        wait=16,
    )
    self_similarity_boundaries = _self_similarity_boundaries(
        audio,
        sample_rate,
        hop_length,
        beat_frames,
        duration,
    )

    boundary_confidences = _merge_boundary_evidence(
        duration=duration,
        phrase_boundaries=phrase_boundaries,
        tempogram_boundaries=tempogram_boundaries,
        self_similarity_boundaries=self_similarity_boundaries,
        beat_grid_confidence=float(beat_metrics['confidence']),
        phrase_boundary_method=phrase_boundary_method,
    )
    snapped_phrase_boundaries = _snap_phrase_boundaries_to_evidence(
        phrase_boundaries,
        boundary_confidences,
        duration=duration,
    )

    section_boundaries = [float(boundary) for boundary in snapped_phrase_boundaries[1:] if 0.0 < float(boundary) < duration]
    if not section_boundaries:
        section_boundaries = [
            float(item['time'])
            for item in boundary_confidences
            if 0.0 < float(item['time']) < duration
        ]

    boundary_tolerance = max(duration * 0.02, 0.35)
    sections: list[dict[str, float | str]] = []
    starts = [0.0] + section_boundaries
    ends = starts[1:] + [duration]
    for idx, (start, end) in enumerate(zip(starts, ends)):
        if end - start < 1.0:
            continue
        start_confidence = 1.0 if idx == 0 else _boundary_support_at_time(boundary_confidences, start, boundary_tolerance)
        end_confidence = 1.0 if idx == len(ends) - 1 else _boundary_support_at_time(boundary_confidences, end, boundary_tolerance)
        sections.append({
            'start': float(start),
            'end': float(end),
            'label': f'section_{idx}',
            'start_boundary_confidence': round(float(start_confidence), 3),
            'end_boundary_confidence': round(float(end_confidence), 3),
            'boundary_confidence': round(float((0.35 * start_confidence) + (0.65 * end_confidence)), 3),
        })

    if not sections:
        sections.append({
            'start': 0.0,
            'end': duration,
            'label': 'section_0',
            'start_boundary_confidence': 1.0,
            'end_boundary_confidence': 1.0,
            'boundary_confidence': 1.0,
        })

    sections, chorus_candidates = _annotate_sections_with_role_hints(audio, sample_rate, hop_length, sections)

    novelty_union = sorted({
        *[float(boundary) for boundary in tempogram_boundaries],
        *[float(boundary) for boundary in self_similarity_boundaries],
    })

    return {
        'tempo_reference_bpm': tempo_bpm,
        'beat_grid_confidence': round(float(beat_metrics['confidence']), 3),
        'phrase_boundary_method': phrase_boundary_method,
        'phrase_boundaries_seconds': snapped_phrase_boundaries,
        'section_boundaries_seconds': [float(s['start']) for s in sections[1:]],
        'novelty_boundaries_seconds': novelty_union,
        'tempogram_novelty_boundaries_seconds': tempogram_boundaries,
        'self_similarity_boundaries_seconds': self_similarity_boundaries,
        'boundary_confidences_seconds': boundary_confidences,
        'chorus_candidates_seconds': [
            {
                'start': float(section['start']),
                'end': float(section['end']),
                'label': str(section['label']),
                'role_hint': str(section['role_hint']),
                'chorus_likelihood': round(float(section['role_scores']['chorus_likelihood']), 3),
            }
            for section in chorus_candidates
        ],
        'sections': sections,
    }
