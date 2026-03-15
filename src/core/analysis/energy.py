from __future__ import annotations

import librosa
import numpy as np


ANALYSIS_VERSION = "0.3.0"


def _safe_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 1e-6)
    return (values - low) / span


def _aggregate_means(values: np.ndarray, boundaries: list[int]) -> np.ndarray:
    out: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk = values[start:end]
        out.append(float(np.mean(chunk)) if chunk.size else 0.0)
    return np.asarray(out, dtype=float)


def _frame_bar_boundaries(beat_frames: np.ndarray, total_frames: int, beats_per_bar: int = 4) -> list[int]:
    points = [0]
    for idx in range(0, len(beat_frames), beats_per_bar):
        points.append(int(beat_frames[idx]))
    if not points or points[0] != 0:
        points = [0, *points]
    points.append(int(total_frames))
    bounded = [min(max(0, p), total_frames) for p in points]
    return sorted(set(bounded))


def _window_seconds(times: np.ndarray, start_idx: int, end_idx: int) -> tuple[float, float]:
    if times.size == 0:
        return 0.0, 0.0
    start_idx = min(max(start_idx, 0), times.size - 1)
    end_idx = min(max(end_idx - 1, start_idx), times.size - 1)
    return float(times[start_idx]), float(times[end_idx])


def _phrase_window_payload(
    idx: int,
    bars_per_phrase: int,
    bar_count: int,
    bar_times: np.ndarray,
    score: float,
    **extra: float,
) -> dict[str, float | int]:
    start_bar = idx * bars_per_phrase
    end_bar = min((idx + 1) * bars_per_phrase, bar_count)
    start_sec, end_sec = _window_seconds(bar_times, start_bar, end_bar)
    payload: dict[str, float | int] = {
        "start_bar": int(start_bar),
        "end_bar": int(end_bar),
        "start": round(start_sec, 3),
        "end": round(end_sec, 3),
        "score": round(float(score), 3),
    }
    for key, value in extra.items():
        payload[key] = round(float(value), 3)
    return payload


def _derive_phrase_signals(bar_rms: np.ndarray, bar_onset: np.ndarray, bar_low: np.ndarray, bar_flatness: np.ndarray, bar_times: np.ndarray) -> dict:
    if bar_rms.size == 0:
        return {
            "hook_strength": 0.0,
            "hook_repetition": 0.0,
            "payoff_strength": 0.0,
            "energy_confidence": 0.0,
            "late_lift": 0.0,
            "climax_strength": 0.0,
            "plateau_strength": 0.0,
            "hook_spend": 0.0,
            "hook_windows": [],
            "payoff_windows": [],
            "climax_windows": [],
            "plateau_windows": [],
        }

    bars_per_phrase = 4
    phrase_count = max(1, int(np.ceil(bar_rms.size / bars_per_phrase)))
    phrase_vectors: list[np.ndarray] = []
    phrase_features: list[dict[str, float]] = []

    norm_rms = _safe_normalize(bar_rms)
    norm_onset = _safe_normalize(bar_onset)
    norm_low = _safe_normalize(bar_low)
    norm_flat = _safe_normalize(bar_flatness)

    for idx in range(phrase_count):
        start = idx * bars_per_phrase
        end = min((idx + 1) * bars_per_phrase, bar_rms.size)
        if start >= end:
            continue
        energy_slice = norm_rms[start:end]
        onset_slice = norm_onset[start:end]
        low_slice = norm_low[start:end]
        flat_slice = norm_flat[start:end]

        vector = np.asarray([
            float(np.mean(energy_slice)),
            float(np.mean(onset_slice)),
            float(np.mean(low_slice)),
            float(np.mean(flat_slice)),
        ])
        phrase_vectors.append(vector)

        local_rise = float(np.mean(np.maximum(0.0, np.diff(energy_slice)))) if (end - start) >= 2 else 0.0
        head = energy_slice[:max(1, len(energy_slice) // 2)]
        tail = energy_slice[-max(1, len(energy_slice) // 2):]
        head_energy = float(np.mean(head)) if head.size else 0.0
        tail_energy = float(np.mean(tail)) if tail.size else head_energy
        peak_energy = float(np.max(energy_slice)) if energy_slice.size else tail_energy
        plateau_range = float(np.max(tail) - np.min(tail)) if tail.size else 0.0
        plateau_stability = float(np.clip(1.0 - plateau_range, 0.0, 1.0))
        end_focus = float(np.clip(tail_energy / max(peak_energy, 1e-6), 0.0, 1.0))
        phrase_position = idx / max(phrase_count - 1, 1)
        late_bias = float(np.clip((phrase_position - 0.40) / 0.60, 0.0, 1.0))
        phrase_energy = float(np.mean(energy_slice))
        phrase_onset = float(np.mean(onset_slice))
        phrase_low = float(np.mean(low_slice))
        phrase_flat = float(np.mean(flat_slice))
        anti_flat = 1.0 - phrase_flat
        headroom = float(np.clip(1.0 - head_energy, 0.0, 1.0))
        lift = float(max(0.0, tail_energy - head_energy))

        plateau_strength = float(np.clip(
            0.48 * tail_energy
            + 0.26 * plateau_stability
            + 0.16 * phrase_low
            + 0.10 * anti_flat,
            0.0,
            1.0,
        ))
        payoff_strength = float(np.clip(
            0.22 * phrase_energy
            + 0.15 * phrase_onset
            + 0.10 * phrase_low
            + 0.10 * anti_flat
            + 0.12 * local_rise
            + 0.11 * tail_energy
            + 0.08 * lift
            + 0.06 * plateau_strength
            + 0.06 * end_focus
            + 0.10 * late_bias,
            0.0,
            1.0,
        ))
        climax_strength = float(np.clip(
            0.26 * payoff_strength
            + 0.22 * plateau_strength
            + 0.14 * phrase_energy
            + 0.12 * phrase_onset
            + 0.10 * phrase_low
            + 0.08 * lift
            + 0.08 * late_bias,
            0.0,
            1.0,
        ))

        phrase_features.append({
            "energy": phrase_energy,
            "onset": phrase_onset,
            "low": phrase_low,
            "flatness": phrase_flat,
            "rise": local_rise,
            "head_energy": head_energy,
            "tail_energy": tail_energy,
            "peak_energy": peak_energy,
            "plateau_stability": plateau_stability,
            "plateau_strength": plateau_strength,
            "end_focus": end_focus,
            "headroom": headroom,
            "lift": lift,
            "late_bias": late_bias,
            "payoff_strength": payoff_strength,
            "climax_strength": climax_strength,
        })

    repetition_scores: list[float] = []
    hook_windows: list[dict[str, float | int]] = []
    climax_windows: list[dict[str, float | int]] = []
    plateau_windows: list[dict[str, float | int]] = []
    payoff_windows: list[dict[str, float | int]] = []

    for idx, vector in enumerate(phrase_vectors):
        features = phrase_features[idx]
        similarities = []
        for other_idx, other in enumerate(phrase_vectors):
            if idx == other_idx:
                continue
            diff = float(np.mean(np.abs(vector - other)))
            similarities.append(max(0.0, 1.0 - diff))
        repetition = max(similarities, default=0.0)
        repetition_scores.append(repetition)
        hook_score = float(np.clip(
            0.54 * repetition
            + 0.20 * features["energy"]
            + 0.10 * features["onset"]
            + 0.10 * features["end_focus"]
            + 0.06 * features["plateau_strength"],
            0.0,
            1.0,
        ))
        if hook_score >= 0.58:
            hook_windows.append(_phrase_window_payload(
                idx,
                bars_per_phrase,
                bar_rms.size,
                bar_times,
                hook_score,
                repetition=repetition,
                energy=features["energy"],
            ))

        if features["plateau_strength"] >= 0.56:
            plateau_windows.append(_phrase_window_payload(
                idx,
                bars_per_phrase,
                bar_rms.size,
                bar_times,
                features["plateau_strength"],
                stability=features["plateau_stability"],
                tail_energy=features["tail_energy"],
            ))

        phrase_position = idx / max(phrase_count - 1, 1)
        payoff_threshold = 0.56 if phrase_position >= 0.60 else 0.66
        if features["payoff_strength"] >= payoff_threshold:
            payoff_windows.append(_phrase_window_payload(
                idx,
                bars_per_phrase,
                bar_rms.size,
                bar_times,
                features["payoff_strength"],
                late_bias=features["late_bias"],
                tail_energy=features["tail_energy"],
                plateau=features["plateau_strength"],
            ))

        climax_threshold = 0.58 if phrase_position >= 0.60 else 0.70
        if features["climax_strength"] >= climax_threshold:
            climax_windows.append(_phrase_window_payload(
                idx,
                bars_per_phrase,
                bar_rms.size,
                bar_times,
                features["climax_strength"],
                payoff=features["payoff_strength"],
                plateau=features["plateau_strength"],
            ))

    quartile = max(1, int(np.ceil(bar_rms.size / 4)))
    early_mean = float(np.mean(norm_rms[:quartile])) if norm_rms.size else 0.0
    late_mean = float(np.mean(norm_rms[-quartile:])) if norm_rms.size else 0.0
    late_lift = float(max(0.0, late_mean - early_mean))

    top_hook_windows = sorted(hook_windows, key=lambda item: float(item["score"]), reverse=True)
    late_payoff_strength = max(
        (features["payoff_strength"] for idx, features in enumerate(phrase_features) if idx / max(phrase_count - 1, 1) >= 0.60),
        default=0.0,
    )
    early_hook_strength = max(
        (float(item["score"]) for item in top_hook_windows if (float(item["start_bar"]) / max(bar_rms.size, 1)) < 0.50),
        default=0.0,
    )
    hook_spend = float(np.clip(early_hook_strength - late_payoff_strength, 0.0, 1.0))

    bar_deltas = np.abs(np.diff(norm_rms)) if norm_rms.size >= 2 else np.asarray([], dtype=float)
    confidence = float(np.clip(
        0.35
        + 0.20 * min(1.0, bar_rms.size / 24.0)
        + 0.15 * (1.0 - min(1.0, float(np.std(bar_deltas)) / 0.35 if bar_deltas.size else 0.0))
        + 0.12 * min(1.0, float(np.max(norm_rms) - np.min(norm_rms)))
        + 0.10 * max((features["climax_strength"] for features in phrase_features), default=0.0)
        + 0.08 * late_lift
        + 0.10 * (1.0 - hook_spend),
        0.0,
        1.0,
    ))

    return {
        "hook_strength": round(max((float(w["score"]) for w in hook_windows), default=0.0), 3),
        "hook_repetition": round(max(repetition_scores, default=0.0), 3),
        "payoff_strength": round(max((features["payoff_strength"] for features in phrase_features), default=0.0), 3),
        "energy_confidence": round(confidence, 3),
        "late_lift": round(late_lift, 3),
        "climax_strength": round(max((float(w["score"]) for w in climax_windows), default=0.0), 3),
        "plateau_strength": round(max((float(w["score"]) for w in plateau_windows), default=0.0), 3),
        "hook_spend": round(hook_spend, 3),
        "hook_windows": top_hook_windows[:6],
        "payoff_windows": sorted(payoff_windows, key=lambda item: float(item["score"]), reverse=True)[:6],
        "climax_windows": sorted(climax_windows, key=lambda item: float(item["score"]), reverse=True)[:6],
        "plateau_windows": sorted(plateau_windows, key=lambda item: float(item["score"]), reverse=True)[:6],
    }


def compute_energy_profile(audio: np.ndarray, sample_rate: int, hop_length: int = 512) -> dict:
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    stft = np.abs(librosa.stft(y=audio, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sample_rate)
    low_mask = freqs <= 200.0
    band_energy = np.sum(stft, axis=0) + 1e-8
    low_band_ratio = np.sum(stft[low_mask], axis=0) / band_energy

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)

    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=hop_length)
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)

    beat_rms = np.interp(beat_times, times, rms) if len(beat_times) else np.asarray([], dtype=float)
    beat_centroid = np.interp(beat_times, times, centroid) if len(beat_times) else np.asarray([], dtype=float)
    beat_rolloff = np.interp(beat_times, times, rolloff) if len(beat_times) else np.asarray([], dtype=float)
    beat_flatness = np.interp(beat_times, times, flatness) if len(beat_times) else np.asarray([], dtype=float)
    beat_onset = np.interp(beat_times, times, onset_env) if len(beat_times) else np.asarray([], dtype=float)
    beat_low = np.interp(beat_times, times, low_band_ratio) if len(beat_times) else np.asarray([], dtype=float)

    bar_boundaries = _frame_bar_boundaries(beat_frames, total_frames=len(rms), beats_per_bar=4)
    bar_rms = _aggregate_means(rms, bar_boundaries)
    bar_centroid = _aggregate_means(centroid, bar_boundaries)
    bar_rolloff = _aggregate_means(rolloff, bar_boundaries)
    bar_flatness = _aggregate_means(flatness, bar_boundaries)
    bar_onset = _aggregate_means(onset_env, bar_boundaries)
    bar_low = _aggregate_means(low_band_ratio, bar_boundaries)
    bar_times = librosa.frames_to_time(np.asarray(bar_boundaries[:-1]), sr=sample_rate, hop_length=hop_length) if len(bar_boundaries) >= 2 else np.asarray([], dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    return {
        "analysis_version": ANALYSIS_VERSION,
        "frame_times": times.tolist(),
        "frame_rms": rms.tolist(),
        "rms": rms.tolist(),
        "spectral_centroid": centroid.tolist(),
        "spectral_rolloff": rolloff.tolist(),
        "spectral_flatness": flatness.tolist(),
        "onset_density": onset_env.tolist(),
        "low_band_ratio": low_band_ratio.tolist(),
        "beat_times": beat_times.tolist(),
        "beat_rms": beat_rms.tolist(),
        "beat_spectral_centroid": beat_centroid.tolist(),
        "beat_spectral_rolloff": beat_rolloff.tolist(),
        "beat_spectral_flatness": beat_flatness.tolist(),
        "beat_onset_density": beat_onset.tolist(),
        "beat_low_band_ratio": beat_low.tolist(),
        "bar_times": bar_times.tolist(),
        "bar_rms": bar_rms.tolist(),
        "bar_spectral_centroid": bar_centroid.tolist(),
        "bar_spectral_rolloff": bar_rolloff.tolist(),
        "bar_spectral_flatness": bar_flatness.tolist(),
        "bar_onset_density": bar_onset.tolist(),
        "bar_low_band_ratio": bar_low.tolist(),
        "derived": derived,
        "summary": {
            "mean_rms": float(np.mean(rms)) if len(rms) else 0.0,
            "max_rms": float(np.max(rms)) if len(rms) else 0.0,
            "dynamic_range_rms": float(np.max(rms) - np.min(rms)) if len(rms) else 0.0,
            "mean_bar_rms": float(np.mean(bar_rms)) if len(bar_rms) else 0.0,
            "max_bar_rms": float(np.max(bar_rms)) if len(bar_rms) else 0.0,
            "bar_dynamic_range_rms": float(np.max(bar_rms) - np.min(bar_rms)) if len(bar_rms) else 0.0,
            "hook_strength": derived["hook_strength"],
            "payoff_strength": derived["payoff_strength"],
            "energy_confidence": derived["energy_confidence"],
            "late_lift": derived["late_lift"],
            "climax_strength": derived["climax_strength"],
            "plateau_strength": derived["plateau_strength"],
            "hook_spend": derived["hook_spend"],
        },
    }
