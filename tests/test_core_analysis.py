from pathlib import Path

import numpy as np

from src.core.analysis import analyzer


def test_analyze_audio_file_returns_song_dna(monkeypatch):
    monkeypatch.setattr(analyzer, "load_audio", lambda path: (np.zeros(44100), 44100))
    monkeypatch.setattr(analyzer, "duration_seconds", lambda audio, sr: 1.0)
    monkeypatch.setattr(analyzer, "detect_tempo", lambda audio, sr: {"bpm": 128.0, "confidence": 0.9, "beat_times": [], "method": "librosa"})
    monkeypatch.setattr(analyzer, "detect_key", lambda audio, sr: {"tonic": "A", "mode": "minor", "camelot": "8A", "confidence": 0.91, "method": "krumhansl_correlation", "chroma": []})

    result = analyzer.analyze_audio_file(Path("song.wav")).to_dict()

    assert result["sample_rate"] == 44100
    assert result["tempo_bpm"] == 128.0
    assert result["key"]["camelot"] == "8A"
    assert result["analysis_version"] == "0.1.0"
