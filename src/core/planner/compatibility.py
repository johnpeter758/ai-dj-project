from __future__ import annotations

from math import fabs

from ..analysis.models import SongDNA
from .models import CompatibilityFactors, CompatibilityReport, ParentReference

CAMELOT_NEIGHBORS = {
    "1A": {"12A", "2A", "1B"}, "1B": {"12B", "2B", "1A"},
    "2A": {"1A", "3A", "2B"}, "2B": {"1B", "3B", "2A"},
    "3A": {"2A", "4A", "3B"}, "3B": {"2B", "4B", "3A"},
    "4A": {"3A", "5A", "4B"}, "4B": {"3B", "5B", "4A"},
    "5A": {"4A", "6A", "5B"}, "5B": {"4B", "6B", "5A"},
    "6A": {"5A", "7A", "6B"}, "6B": {"5B", "7B", "6A"},
    "7A": {"6A", "8A", "7B"}, "7B": {"6B", "8B", "7A"},
    "8A": {"7A", "9A", "8B"}, "8B": {"7B", "9B", "8A"},
    "9A": {"8A", "10A", "9B"}, "9B": {"8B", "10B", "9A"},
    "10A": {"9A", "11A", "10B"}, "10B": {"9B", "11B", "10A"},
    "11A": {"10A", "12A", "11B"}, "11B": {"10B", "12B", "11A"},
    "12A": {"11A", "1A", "12B"}, "12B": {"11B", "1B", "12A"},
}


def _tempo_score(a: SongDNA, b: SongDNA) -> tuple[float, str]:
    delta = fabs(a.tempo_bpm - b.tempo_bpm)
    if delta <= 2.0:
        return 1.0, f"close tempo match ({delta:.1f} BPM delta)"
    if delta <= 5.0:
        return 0.8, f"workable tempo match ({delta:.1f} BPM delta)"
    if delta <= 10.0:
        return 0.55, f"requires meaningful stretch ({delta:.1f} BPM delta)"
    return 0.2, f"large tempo gap ({delta:.1f} BPM delta)"


def _harmony_score(a: SongDNA, b: SongDNA) -> tuple[float, str]:
    ca = str(a.key.get("camelot", ""))
    cb = str(b.key.get("camelot", ""))
    if ca and cb and ca == cb:
        return 1.0, f"same Camelot key ({ca})"
    if ca and cb and cb in CAMELOT_NEIGHBORS.get(ca, set()):
        return 0.8, f"neighbor Camelot keys ({ca} vs {cb})"
    if a.key.get("tonic") == b.key.get("tonic"):
        return 0.65, "same tonic but different mode"
    return 0.35, f"harmonic gap ({ca or 'unknown'} vs {cb or 'unknown'})"


def _structure_score(a: SongDNA, b: SongDNA) -> tuple[float, str]:
    a_sections = len(a.structure.get("sections", []))
    b_sections = len(b.structure.get("sections", []))
    if a_sections == 0 or b_sections == 0:
        return 0.5, "structure incomplete in one or both songs"
    delta = abs(a_sections - b_sections)
    if delta == 0:
        return 0.9, "similar section count"
    if delta == 1:
        return 0.75, "near section-count match"
    if delta <= 3:
        return 0.55, "moderate structural mismatch"
    return 0.3, "strong structural mismatch"


def _energy_score(a: SongDNA, b: SongDNA) -> tuple[float, str]:
    a_mean = float(a.energy.get("summary", {}).get("mean_rms", 0.0))
    b_mean = float(b.energy.get("summary", {}).get("mean_rms", 0.0))
    delta = abs(a_mean - b_mean)
    if delta <= 0.03:
        return 0.9, "similar average energy"
    if delta <= 0.08:
        return 0.7, "workable energy relationship"
    return 0.4, "large energy mismatch"


def _stem_conflict_score(a: SongDNA, b: SongDNA) -> tuple[float, str]:
    a_has_stems = bool(a.stems.get("enabled"))
    b_has_stems = bool(b.stems.get("enabled"))
    if a_has_stems and b_has_stems:
        return 0.8, "both songs have stems available"
    if a_has_stems or b_has_stems:
        return 0.65, "only one song has stems available"
    return 0.5, "no stems available yet"


def build_compatibility_report(song_a: SongDNA, song_b: SongDNA) -> CompatibilityReport:
    tempo, tempo_note = _tempo_score(song_a, song_b)
    harmony, harmony_note = _harmony_score(song_a, song_b)
    structure, structure_note = _structure_score(song_a, song_b)
    energy, energy_note = _energy_score(song_a, song_b)
    stem_conflict, stem_note = _stem_conflict_score(song_a, song_b)

    factors = CompatibilityFactors(
        tempo=tempo,
        harmony=harmony,
        structure=structure,
        energy=energy,
        stem_conflict=stem_conflict,
        notes=[tempo_note, harmony_note, structure_note, energy_note, stem_note],
    )

    return CompatibilityReport(
        parent_a=ParentReference(
            source_path=song_a.source_path,
            tempo_bpm=song_a.tempo_bpm,
            key_tonic=str(song_a.key.get("tonic", "unknown")),
            key_mode=str(song_a.key.get("mode", "unknown")),
            duration_seconds=song_a.duration_seconds,
        ),
        parent_b=ParentReference(
            source_path=song_b.source_path,
            tempo_bpm=song_b.tempo_bpm,
            key_tonic=str(song_b.key.get("tonic", "unknown")),
            key_mode=str(song_b.key.get("mode", "unknown")),
            duration_seconds=song_b.duration_seconds,
        ),
        factors=factors,
    )
