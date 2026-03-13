from pathlib import Path

import numpy as np

from src.core.analysis import analyzer
from src.core.analysis.energy import compute_energy_profile


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
