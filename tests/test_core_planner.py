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

    plan = build_stub_arrangement_plan(a, b).to_dict()

    assert len(plan["sections"]) == 3
    assert plan["sections"][0]["label"] == "intro"
    assert "compatibility" in plan
