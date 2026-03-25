#!/usr/bin/env python3
"""Vocal quality gate to reject AI-slop vocals early.

Usage:
  python scripts/vocal_quality_gate.py vocals.wav --output runs/checkpoint/vocal_quality.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return float(20.0 * np.log10(max(rms, 1e-9)))


def _clip_ratio(audio: np.ndarray, threshold: float = 0.995) -> float:
    return float(np.mean(np.abs(audio) >= threshold))


def _band_energy_ratio(spec_mag: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    band = np.sum(spec_mag[mask, :])
    total = np.sum(spec_mag)
    return float(band / max(total, 1e-9))


def _pitch_metrics(audio: np.ndarray, sr: int) -> dict[str, float]:
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )
    voiced = f0[~np.isnan(f0)]
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    if voiced.size == 0:
        return {
            "voiced_ratio": voiced_ratio,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pitch_jitter_ratio": 1.0,
        }

    diffs = np.abs(np.diff(voiced))
    jitter_ratio = float(np.median(diffs) / max(np.median(voiced), 1e-6)) if diffs.size else 0.0
    return {
        "voiced_ratio": voiced_ratio,
        "pitch_mean_hz": float(np.mean(voiced)),
        "pitch_std_hz": float(np.std(voiced)),
        "pitch_jitter_ratio": jitter_ratio,
    }


def _compute_metrics(path: Path) -> dict[str, Any]:
    audio, sr = librosa.load(path.as_posix(), sr=None, mono=True)
    if audio.size == 0:
        raise ValueError("empty audio")

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    n_fft = 2048
    hop = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    spec_mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    centroid = librosa.feature.spectral_centroid(S=spec_mag, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=spec_mag)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop)[0]

    pitch = _pitch_metrics(audio, sr)

    sibilance_ratio = _band_energy_ratio(spec_mag, freqs, 5000.0, 11000.0)
    body_ratio = _band_energy_ratio(spec_mag, freqs, 180.0, 2200.0)
    high_hiss_ratio = _band_energy_ratio(spec_mag, freqs, 11000.0, 20000.0)

    metrics = {
        "duration_seconds": float(audio.size / sr),
        "sample_rate": int(sr),
        "rms_db": _rms_db(audio),
        "clip_ratio": _clip_ratio(audio),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "zcr_mean": float(np.mean(zcr)),
        "sibilance_ratio": sibilance_ratio,
        "body_ratio": body_ratio,
        "high_hiss_ratio": high_hiss_ratio,
        **pitch,
    }
    return metrics


def _score(metrics: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 100.0

    clip_ratio = float(metrics.get("clip_ratio") or 0.0)
    if clip_ratio > 0.002:
        score -= min(25.0, (clip_ratio - 0.002) * 8000.0)
        reasons.append("clipping detected")

    jitter = float(metrics.get("pitch_jitter_ratio") or 0.0)
    if jitter > 0.055:
        score -= min(18.0, (jitter - 0.055) * 320.0)
        reasons.append("pitch instability")

    voiced_ratio = float(metrics.get("voiced_ratio") or 0.0)
    if voiced_ratio < 0.30:
        score -= min(16.0, (0.30 - voiced_ratio) * 60.0)
        reasons.append("weak voiced presence")

    sibilance = float(metrics.get("sibilance_ratio") or 0.0)
    if sibilance > 0.26:
        score -= min(14.0, (sibilance - 0.26) * 120.0)
        reasons.append("harsh sibilance")

    hiss = float(metrics.get("high_hiss_ratio") or 0.0)
    if hiss > 0.14:
        score -= min(12.0, (hiss - 0.14) * 150.0)
        reasons.append("high-band hiss")

    flatness = float(metrics.get("spectral_flatness_mean") or 0.0)
    if flatness > 0.22:
        score -= min(12.0, (flatness - 0.22) * 140.0)
        reasons.append("noisy/airy texture")

    body = float(metrics.get("body_ratio") or 0.0)
    if body < 0.34:
        score -= min(10.0, (0.34 - body) * 110.0)
        reasons.append("thin vocal body")

    final = float(max(0.0, min(100.0, score)))
    return round(final, 1), reasons


def run(path: Path) -> dict[str, Any]:
    metrics = _compute_metrics(path)
    quality_score, reasons = _score(metrics)

    if quality_score >= 85:
        verdict = "strong"
    elif quality_score >= 70:
        verdict = "usable"
    elif quality_score >= 55:
        verdict = "borderline"
    else:
        verdict = "reject"

    return {
        "input_path": str(path),
        "quality_score": quality_score,
        "verdict": verdict,
        "reasons": reasons,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate vocal quality to avoid AI-slop artifacts")
    parser.add_argument("input", help="Path to vocal audio file")
    parser.add_argument("--output", "-o", help="Optional output JSON path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    payload = run(input_path)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote vocal quality report: {out_path}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    print(f"Vocal quality score: {payload['quality_score']}")
    print(f"Verdict: {payload['verdict']}")
    if payload["reasons"]:
        print("Reasons:")
        for line in payload["reasons"]:
            print(f"- {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
