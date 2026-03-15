from pathlib import Path

import numpy as np

from src.core.analysis import analyzer
from src.core.analysis.energy import _derive_phrase_signals, compute_energy_profile


def test_analyze_audio_file_returns_song_dna(monkeypatch):
    monkeypatch.setattr(analyzer, "load_audio", lambda path: (np.zeros(44100), 44100))
    monkeypatch.setattr(analyzer, "duration_seconds", lambda audio, sr: 1.0)
    monkeypatch.setattr(analyzer, "detect_tempo", lambda audio, sr: {"bpm": 128.0, "confidence": 0.9, "beat_times": [], "method": "librosa"})
    monkeypatch.setattr(analyzer, "detect_key", lambda audio, sr: {"tonic": "A", "mode": "minor", "camelot": "8A", "confidence": 0.91, "method": "krumhansl_correlation", "chroma": []})
    monkeypatch.setattr(analyzer, "estimate_structure", lambda audio, sr: {"sections": [{"start": 0.0, "end": 1.0, "label": "section_0"}]})
    monkeypatch.setattr(analyzer, "compute_energy_profile", lambda audio, sr: {"summary": {"mean_rms": 0.1}})

    result = analyzer.analyze_audio_file(Path("song.wav")).to_dict()

    assert result["sample_rate"] == 44100
    assert result["tempo_bpm"] == 128.0
    assert result["key"]["camelot"] == "8A"
    assert result["structure"]["sections"][0]["label"] == "section_0"
    assert result["analysis_version"] == "0.1.0"


def test_compute_energy_profile_emits_bar_and_derived_signals():
    sr = 22050
    duration = 8.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    envelope = np.concatenate([
        np.linspace(0.15, 0.35, int(t.size * 0.25), endpoint=False),
        np.linspace(0.35, 0.85, int(t.size * 0.25), endpoint=False),
        np.linspace(0.20, 0.30, int(t.size * 0.25), endpoint=False),
        np.linspace(0.70, 0.95, t.size - int(t.size * 0.75), endpoint=False),
    ])
    audio = envelope * np.sin(2 * np.pi * 220 * t)

    profile = compute_energy_profile(audio.astype(float), sr)

    assert profile["bar_rms"]
    assert profile["bar_onset_density"]
    assert "derived" in profile
    assert profile["derived"]["energy_confidence"] >= 0.0
    assert "hook_strength" in profile["summary"]
    assert "payoff_strength" in profile["summary"]
    assert "late_lift" in profile["summary"]
    assert "climax_strength" in profile["summary"]
    assert "plateau_strength" in profile["summary"]
    assert "hook_spend" in profile["summary"]


def test_compute_energy_profile_prefers_late_sustained_payoff_window_over_early_spike():
    bar_rms = np.asarray([0.08, 0.10, 0.88, 0.24, 0.22, 0.26, 0.74, 0.82], dtype=float)
    bar_onset = np.asarray([0.12, 0.14, 0.78, 0.20, 0.22, 0.24, 0.68, 0.72], dtype=float)
    bar_low = np.asarray([0.18, 0.20, 0.40, 0.22, 0.24, 0.28, 0.62, 0.68], dtype=float)
    bar_flatness = np.asarray([0.48, 0.44, 0.52, 0.46, 0.44, 0.42, 0.24, 0.20], dtype=float)
    bar_times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    assert derived["payoff_windows"]
    top = max(derived["payoff_windows"], key=lambda item: item["score"])
    assert top["start_bar"] == 4
    assert top["end_bar"] == 8
    assert derived["climax_windows"]
    top_climax = max(derived["climax_windows"], key=lambda item: item["score"])
    assert top_climax["start_bar"] == 4
    assert top_climax["end_bar"] == 8
    assert derived["late_lift"] > 0.2
    assert derived["plateau_strength"] > 0.5


def test_compute_energy_profile_flags_early_hook_spend_when_late_payoff_never_arrives():
    bar_rms = np.asarray([
        0.30, 0.34, 0.32, 0.35,
        0.31, 0.35, 0.33, 0.36,
        0.20, 0.22, 0.24, 0.23,
        0.18, 0.20, 0.21, 0.19,
    ], dtype=float)
    bar_onset = np.asarray([
        0.58, 0.60, 0.56, 0.59,
        0.57, 0.61, 0.58, 0.60,
        0.22, 0.24, 0.25, 0.23,
        0.18, 0.20, 0.19, 0.18,
    ], dtype=float)
    bar_low = np.asarray([
        0.42, 0.44, 0.40, 0.43,
        0.41, 0.45, 0.42, 0.44,
        0.22, 0.24, 0.23, 0.22,
        0.18, 0.19, 0.20, 0.18,
    ], dtype=float)
    bar_flatness = np.asarray([
        0.24, 0.22, 0.25, 0.23,
        0.24, 0.22, 0.23, 0.22,
        0.40, 0.42, 0.41, 0.43,
        0.46, 0.48, 0.47, 0.49,
    ], dtype=float)
    bar_times = np.arange(bar_rms.size, dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    assert derived["hook_windows"]
    assert derived["hook_spend"] > 0.2
    assert derived["late_lift"] == 0.0
    assert derived["climax_strength"] < derived["hook_strength"]


def test_compute_energy_profile_detects_late_plateau_and_climax_windows():
    bar_rms = np.asarray([
        0.10, 0.12, 0.16, 0.20,
        0.24, 0.30, 0.40, 0.52,
        0.66, 0.72, 0.74, 0.75,
        0.73, 0.74, 0.75, 0.74,
    ], dtype=float)
    bar_onset = np.asarray([
        0.12, 0.14, 0.16, 0.18,
        0.24, 0.28, 0.36, 0.44,
        0.58, 0.62, 0.60, 0.58,
        0.54, 0.52, 0.50, 0.48,
    ], dtype=float)
    bar_low = np.asarray([
        0.18, 0.18, 0.20, 0.22,
        0.26, 0.30, 0.36, 0.44,
        0.56, 0.60, 0.62, 0.64,
        0.63, 0.64, 0.64, 0.63,
    ], dtype=float)
    bar_flatness = np.asarray([
        0.52, 0.50, 0.48, 0.46,
        0.40, 0.36, 0.30, 0.24,
        0.18, 0.16, 0.14, 0.14,
        0.15, 0.15, 0.14, 0.15,
    ], dtype=float)
    bar_times = np.arange(bar_rms.size, dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    assert derived["plateau_windows"]
    assert derived["climax_windows"]
    top_plateau = max(derived["plateau_windows"], key=lambda item: item["score"])
    top_climax = max(derived["climax_windows"], key=lambda item: item["score"])
    assert top_plateau["start_bar"] >= 8
    assert top_climax["start_bar"] >= 8
    assert derived["late_lift"] > 0.4
    assert derived["climax_strength"] >= derived["payoff_strength"] - 0.1
