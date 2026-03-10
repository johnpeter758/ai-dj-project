from __future__ import annotations

from pathlib import Path

from .key import detect_key
from .loader import duration_seconds, load_audio
from .models import SongDNA
from .tempo import detect_tempo


def analyze_audio_file(audio_path: str | Path) -> SongDNA:
    source_path = Path(audio_path).expanduser().resolve()
    audio, sample_rate = load_audio(source_path)
    tempo = detect_tempo(audio, sample_rate)
    key = detect_key(audio, sample_rate)

    return SongDNA(
        source_path=str(source_path),
        sample_rate=sample_rate,
        duration_seconds=duration_seconds(audio, sample_rate),
        tempo_bpm=float(tempo["bpm"]),
        key=key,
        metadata={
            "schema_version": "0.1.0",
            "tempo": tempo,
        },
    )
