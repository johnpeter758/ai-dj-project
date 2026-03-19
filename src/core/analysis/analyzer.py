from __future__ import annotations

from pathlib import Path

from .energy import compute_energy_profile
from .key import detect_key
from .loader import duration_seconds, load_audio
from .models import SongDNA
from .musical_intelligence import analyze_musical_intelligence
from .stems import separate_stems
from .structure import estimate_structure
from .tempo import detect_tempo


def analyze_audio_file(audio_path: str | Path, stems_dir: str | Path | None = None) -> SongDNA:
    source_path = Path(audio_path).expanduser().resolve()
    audio, sample_rate = load_audio(source_path)
    tempo = detect_tempo(audio, sample_rate)
    key = detect_key(audio, sample_rate)

    stems = {"enabled": False, "files": {}}
    if stems_dir is not None:
        stems = {"enabled": True, **separate_stems(source_path, stems_dir)}

    return SongDNA(
        source_path=str(source_path),
        sample_rate=sample_rate,
        duration_seconds=duration_seconds(audio, sample_rate),
        tempo_bpm=float(tempo["bpm"]),
        key=key,
        structure=estimate_structure(audio, sample_rate),
        energy=compute_energy_profile(audio, sample_rate),
        stems=stems,
        musical_intelligence=analyze_musical_intelligence(audio, sample_rate),
        metadata={
            "schema_version": "0.1.0",
            "tempo": tempo,
        },
    )
