from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np
import pytest

from song_dna import SongDNAError, analyze_song_dna


def _write_test_wav(path: Path, sample_rate: int = 22050, seconds: float = 4.0) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    # Deterministic synthetic signal: bass-ish tone + melodic tone + periodic click pulse.
    tone = 0.35 * np.sin(2 * math.pi * 110 * t) + 0.2 * np.sin(2 * math.pi * 440 * t)

    bpm = 120
    beat_interval = int(sample_rate * 60 / bpm)
    clicks = np.zeros_like(tone)
    clicks[::beat_interval] = 0.7

    signal = np.clip(tone + clicks, -1.0, 1.0)
    int16 = (signal * 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


def test_song_dna_schema_and_determinism(tmp_path: Path) -> None:
    wav = tmp_path / "fixture.wav"
    _write_test_wav(wav)

    first = analyze_song_dna(wav)
    second = analyze_song_dna(wav)

    # Deterministic output check
    assert first == second

    # Required schema keys
    required = [
        "duration_seconds",
        "tempo_bpm",
        "tempo_confidence",
        "key_tonic",
        "key_mode",
        "key_confidence",
        "beat_grid",
        "bar_grid",
        "downbeats",
        "loudness_profile",
        "energy_curve",
        "vocal_density_map",
        "drum_density_map",
        "bass_density_map",
        "melodic_density_map",
        "harmonic_density_map",
        "sections",
        "phrases",
        "candidate_hooks",
        "candidate_transitions",
    ]
    for key in required:
        assert key in first

    assert first["duration_seconds"] > 0
    assert 40 <= first["tempo_bpm"] <= 220
    assert 0 <= first["tempo_confidence"] <= 1
    assert first["key_mode"] in {"major", "minor"}
    assert 0 <= first["key_confidence"] <= 1

    assert isinstance(first["beat_grid"], list)
    assert isinstance(first["sections"], list)
    assert len(first["sections"]) >= 1

    sec0 = first["sections"][0]
    assert {"label", "start", "end", "length_bars", "metrics"}.issubset(sec0.keys())
    assert {"avg_energy", "avg_loudness_db", "drum_density", "bass_density"}.issubset(
        sec0["metrics"].keys()
    )


@pytest.mark.parametrize("bad_path", ["/definitely/not/here.wav", "missing.wav"])
def test_song_dna_load_error_is_clear(bad_path: str) -> None:
    with pytest.raises(SongDNAError) as exc:
        analyze_song_dna(bad_path)

    msg = str(exc.value).lower()
    assert "audio" in msg
    assert "exist" in msg or "could not" in msg
