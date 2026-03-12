from src.core.analysis.models import SongDNA
from src.core.planner import build_compatibility_report, build_stub_arrangement_plan


def make_song(path: str, tempo: float, tonic: str, mode: str, camelot: str, sections: int, mean_rms: float) -> SongDNA:
    return SongDNA(
        source_path=path,
        sample_rate=44100,
        duration_seconds=180.0,
        tempo_bpm=tempo,
        key={"tonic": tonic, "mode": mode, "camelot": camelot, "confidence": 0.9},
        structure={"sections": [{"label": f"section_{i}"} for i in range(sections)]},
        energy={"summary": {"mean_rms": mean_rms}},
    )


def test_build_compatibility_report_returns_factorized_scores():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 4, 0.11)
    b = make_song("b.wav", 129.0, "A", "minor", "8A", 4, 0.10)

    report = build_compatibility_report(a, b).to_dict()

    assert report["factors"]["tempo"] >= 0.8
    assert report["factors"]["harmony"] >= 0.8
    assert report["factors"]["structure"] >= 0.8
    assert report["factors"]["overall"] > 0.0


def test_build_stub_arrangement_plan_returns_sections():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 4, 0.11)
    b = make_song("b.wav", 130.0, "C", "major", "8B", 5, 0.14)
    a.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0]
    b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]
    a.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]
    a.energy["beat_rms"] = [0.10, 0.12, 0.16, 0.18, 0.35, 0.40, 0.70, 0.75]
    b.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0]
    b.energy["beat_rms"] = [0.08, 0.10, 0.30, 0.34, 0.48, 0.52, 0.58, 0.62, 0.66, 0.70]

    plan = build_stub_arrangement_plan(a, b).to_dict()

    assert len(plan["sections"]) == 3
    assert plan["sections"][0]["label"] == "intro"
    assert plan["sections"][0]["source_section_label"].startswith("phrase_")
    assert plan["sections"][1]["source_section_label"].startswith("phrase_")
    assert plan["sections"][2]["source_section_label"].startswith("phrase_")
    assert "compatibility" in plan
    assert any("Phrase-window planner" in note for note in plan["planning_notes"])


def test_build_stub_arrangement_plan_prefers_higher_energy_late_phrase_for_payoff():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 3, 0.11)
    b = make_song("b.wav", 130.0, "C", "major", "8B", 6, 0.14)
    b.duration_seconds = 48.0
    b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    b.energy["beat_times"] = [
        2.0, 6.0,
        10.0, 14.0,
        18.0, 22.0,
        26.0, 30.0,
        34.0, 38.0,
        42.0, 46.0,
    ]
    b.energy["beat_rms"] = [
        0.10, 0.12,
        0.18, 0.20,
        0.24, 0.28,
        0.32, 0.36,
        0.38, 0.40,
        0.88, 0.92,
    ]

    plan = build_stub_arrangement_plan(a, b).to_dict()
    payoff = next(section for section in plan["sections"] if section["label"] == "payoff")

    assert payoff["source_parent"] == "B"
    assert payoff["source_section_label"] == "phrase_2_6"
