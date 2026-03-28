from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


class SongDNAError(ValueError):
    """Raised when Song DNA analysis cannot proceed."""


@dataclass(frozen=True)
class _AudioData:
    samples: np.ndarray
    sample_rate: int


def _load_audio(audio_path: str | Path) -> _AudioData:
    path = Path(audio_path)
    if not path.exists():
        raise SongDNAError(f"Audio file does not exist: {path}")

    try:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except wave.Error as exc:
        raise SongDNAError(
            f"Could not load audio '{path}'. Only uncompressed WAV is supported right now."
        ) from exc
    except OSError as exc:
        raise SongDNAError(f"Could not open audio file '{path}': {exc}") from exc

    if sample_rate <= 0:
        raise SongDNAError(f"Invalid sample rate in '{path}': {sample_rate}")
    if channels <= 0:
        raise SongDNAError(f"Invalid channel count in '{path}': {channels}")

    if sampwidth == 1:
        dtype = np.uint8
        scale = 128.0
        offset = 128.0
    elif sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
        offset = 0.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
        offset = 0.0
    else:
        raise SongDNAError(
            f"Unsupported WAV sample width ({sampwidth} bytes) in '{path}'."
        )

    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if sampwidth == 1:
        data = (data - offset) / scale
    else:
        data = data / scale

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if data.size == 0:
        raise SongDNAError(f"Audio file contains no samples: {path}")

    return _AudioData(samples=data, sample_rate=sample_rate)


def _frame_rms(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    n = len(samples)
    if n < frame_size:
        padded = np.pad(samples, (0, frame_size - n))
        return np.array([np.sqrt(np.mean(padded**2))], dtype=np.float64)

    rms = []
    for start in range(0, n - frame_size + 1, hop_size):
        frame = samples[start : start + frame_size]
        rms.append(np.sqrt(np.mean(frame**2)))
    return np.asarray(rms, dtype=np.float64)


def _estimate_tempo(rms: np.ndarray, hop_sec: float) -> tuple[float, float]:
    if rms.size < 8:
        return 120.0, 0.0

    onset = np.diff(rms, prepend=rms[0])
    onset = np.clip(onset, 0.0, None)
    if np.max(onset) > 0:
        onset = onset / np.max(onset)

    min_bpm, max_bpm = 60.0, 200.0
    min_lag = max(1, int(round((60.0 / max_bpm) / hop_sec)))
    max_lag = max(min_lag + 1, int(round((60.0 / min_bpm) / hop_sec)))

    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    if max_lag >= len(ac):
        max_lag = len(ac) - 1
    if max_lag <= min_lag:
        return 120.0, 0.0

    region = ac[min_lag : max_lag + 1]
    best_rel = int(np.argmax(region))
    best_lag = min_lag + best_rel

    bpm = 60.0 / (best_lag * hop_sec)
    confidence = float(region[best_rel] / (ac[0] + 1e-9))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return float(np.round(bpm, 3)), float(np.round(confidence, 3))


def _estimate_key(samples: np.ndarray, sample_rate: int) -> tuple[str, str, float]:
    frame_size = 4096
    hop = 2048
    if len(samples) < frame_size:
        padded = np.pad(samples, (0, frame_size - len(samples)))
        frames = padded[None, :]
    else:
        starts = range(0, len(samples) - frame_size + 1, hop)
        frames = np.stack([samples[s : s + frame_size] for s in starts], axis=0)

    window = np.hanning(frame_size)
    chroma = np.zeros(12, dtype=np.float64)

    for frame in frames:
        spec = np.abs(np.fft.rfft(frame * window))
        freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
        valid = freqs > 20.0
        freqs = freqs[valid]
        mags = spec[valid]
        if freqs.size == 0:
            continue
        midi = 69 + 12 * np.log2(freqs / 440.0)
        pcs = np.mod(np.round(midi).astype(int), 12)
        for pc, mag in zip(pcs, mags):
            chroma[pc] += mag

    if np.all(chroma == 0):
        return "C", "major", 0.0

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    scores = []
    for tonic in range(12):
        maj = np.corrcoef(chroma, np.roll(major_profile, tonic))[0, 1]
        minr = np.corrcoef(chroma, np.roll(minor_profile, tonic))[0, 1]
        scores.append((tonic, "major", np.nan_to_num(maj, nan=0.0)))
        scores.append((tonic, "minor", np.nan_to_num(minr, nan=0.0)))

    tonic, mode, score = max(scores, key=lambda x: x[2])
    tonic_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][tonic]

    score_vals = np.array([s[2] for s in scores], dtype=np.float64)
    spread = np.max(score_vals) - np.mean(score_vals)
    conf = float(np.clip(spread / (np.std(score_vals) + 1e-9) / 3.0, 0.0, 1.0))
    return tonic_name, mode, float(np.round(conf, 3))


def _density_maps(samples: np.ndarray, sample_rate: int, points: int = 64) -> dict[str, list[float]]:
    if points <= 0:
        points = 64
    segments = np.array_split(samples, points)

    maps = {
        "vocal": [],
        "drum": [],
        "bass": [],
        "melodic": [],
        "harmonic": [],
    }

    for seg in segments:
        if seg.size == 0:
            for k in maps:
                maps[k].append(0.0)
            continue

        nfft = int(2 ** np.ceil(np.log2(max(256, seg.size))))
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size), n=nfft))
        freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
        total = float(np.sum(spec) + 1e-9)

        def band(lo: float, hi: float) -> float:
            idx = (freqs >= lo) & (freqs < hi)
            return float(np.sum(spec[idx]) / total)

        bass = band(20, 250)
        drum = band(60, 500) + 0.5 * band(2000, 8000)
        vocal = band(300, 3400)
        harmonic = band(250, 4000)
        melodic = band(500, 5000)

        maps["vocal"].append(round(vocal, 6))
        maps["drum"].append(round(min(drum, 1.0), 6))
        maps["bass"].append(round(bass, 6))
        maps["melodic"].append(round(melodic, 6))
        maps["harmonic"].append(round(harmonic, 6))

    return maps


def analyze_song_dna(audio_path: str | Path) -> dict:
    """Analyze a WAV file and return a pragmatic, deterministic Song DNA structure."""
    audio = _load_audio(audio_path)
    samples = audio.samples
    sr = audio.sample_rate
    duration = float(np.round(len(samples) / sr, 6))

    frame_size = 2048
    hop_size = 512
    hop_sec = hop_size / sr

    rms = _frame_rms(samples, frame_size=frame_size, hop_size=hop_size)
    loudness_profile = np.round((20.0 * np.log10(rms + 1e-8)), 3)
    loudness_profile = np.clip(loudness_profile, -80.0, 0.0)
    energy_curve = np.round((rms / (np.max(rms) + 1e-9)), 6)

    tempo_bpm, tempo_conf = _estimate_tempo(rms, hop_sec)

    beat_period = 60.0 / max(tempo_bpm, 1e-6)
    beat_grid = np.arange(0.0, duration + 1e-9, beat_period)
    beat_grid = np.round(beat_grid, 3)
    bar_grid = np.round(beat_grid[::4], 3)
    downbeats = bar_grid.copy()

    tonic, mode, key_conf = _estimate_key(samples, sr)

    density = _density_maps(samples, sr, points=64)

    sections = []
    if len(bar_grid) == 0:
        sections.append(
            {
                "label": "section_1",
                "start": 0.0,
                "end": duration,
                "length_bars": 0,
                "metrics": {
                    "avg_energy": float(np.round(np.mean(energy_curve), 6)),
                    "avg_loudness_db": float(np.round(np.mean(loudness_profile), 3)),
                    "drum_density": float(np.round(np.mean(density["drum"]), 6)),
                    "bass_density": float(np.round(np.mean(density["bass"]), 6)),
                },
            }
        )
    else:
        bars_per_section = 8
        section_starts = list(range(0, len(bar_grid), bars_per_section))
        for i, start_idx in enumerate(section_starts):
            end_idx = min(start_idx + bars_per_section, len(bar_grid) - 1)
            start_t = float(bar_grid[start_idx])
            end_t = float(bar_grid[end_idx]) if end_idx > start_idx else duration
            if i == len(section_starts) - 1:
                end_t = duration
            length_bars = max(1, end_idx - start_idx)

            e0 = int((start_t / max(duration, 1e-9)) * len(energy_curve))
            e1 = int((end_t / max(duration, 1e-9)) * len(energy_curve))
            e1 = max(e1, e0 + 1)
            e_slice = energy_curve[e0:e1]
            l_slice = loudness_profile[e0:e1]

            d0 = int((start_t / max(duration, 1e-9)) * 64)
            d1 = int((end_t / max(duration, 1e-9)) * 64)
            d1 = max(d1, d0 + 1)

            sections.append(
                {
                    "label": f"section_{i + 1}",
                    "start": round(start_t, 3),
                    "end": round(end_t, 3),
                    "length_bars": int(length_bars),
                    "metrics": {
                        "avg_energy": float(np.round(np.mean(e_slice), 6)),
                        "avg_loudness_db": float(np.round(np.mean(l_slice), 3)),
                        "drum_density": float(np.round(np.mean(density["drum"][d0:d1]), 6)),
                        "bass_density": float(np.round(np.mean(density["bass"][d0:d1]), 6)),
                    },
                }
            )

    phrases = []
    phrase_beats = 4
    for i in range(0, len(beat_grid), phrase_beats):
        start = float(beat_grid[i])
        end = float(beat_grid[min(i + phrase_beats - 1, len(beat_grid) - 1)])
        phrases.append(
            {
                "start": round(start, 3),
                "end": round(min(end + beat_period, duration), 3),
                "length_beats": phrase_beats,
            }
        )

    hook_candidates = sorted(
        [
            {
                "time": round(float(i / max(len(energy_curve), 1) * duration), 3),
                "score": round(float(v), 6),
            }
            for i, v in enumerate(energy_curve)
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:3]

    transitions = []
    for i in range(len(sections) - 1):
        transitions.append(
            {
                "from_section": sections[i]["label"],
                "to_section": sections[i + 1]["label"],
                "time": sections[i]["end"],
            }
        )

    result = {
        "duration_seconds": duration,
        "tempo_bpm": tempo_bpm,
        "tempo_confidence": tempo_conf,
        "key_tonic": tonic,
        "key_mode": mode,
        "key_confidence": key_conf,
        "beat_grid": beat_grid.tolist(),
        "bar_grid": bar_grid.tolist(),
        "downbeats": downbeats.tolist(),
        "loudness_profile": loudness_profile.tolist(),
        "energy_curve": energy_curve.tolist(),
        "density_maps": density,
        "vocal_density_map": density["vocal"],
        "drum_density_map": density["drum"],
        "bass_density_map": density["bass"],
        "melodic_density_map": density["melodic"],
        "harmonic_density_map": density["harmonic"],
        "sections": sections,
        "phrases": phrases,
        "candidate_hooks": hook_candidates,
        "candidate_transitions": transitions,
    }
    return result
