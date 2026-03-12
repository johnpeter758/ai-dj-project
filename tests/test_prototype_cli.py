import json
from pathlib import Path

import ai_dj


class DummySongDNA:
    def __init__(self, path: str):
        self.source_path = path
        self.sample_rate = 44100
        self.duration_seconds = 180.0
        self.tempo_bpm = 128.0
        self.key = {"tonic": "A", "mode": "minor", "camelot": "8A"}
        self.structure = {"sections": [{"label": "section_0"}]}
        self.energy = {"summary": {"mean_rms": 0.1}}
        self.stems = {"enabled": False, "files": {}}

    def to_dict(self):
        return {
            "source_path": self.source_path,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "tempo_bpm": self.tempo_bpm,
            "key": self.key,
            "structure": self.structure,
            "energy": self.energy,
            "stems": self.stems,
            "analysis_version": "0.1.0",
            "metadata": {"schema_version": "0.1.0"},
        }


def test_prototype_writes_expected_artifacts(monkeypatch, tmp_path: Path):
    song_a = tmp_path / "song_a.wav"
    song_b = tmp_path / "song_b.wav"
    song_a.write_bytes(b"fake-audio")
    song_b.write_bytes(b"fake-audio")

    monkeypatch.setattr(ai_dj, "analyze_audio_file", lambda path, stems_dir=None: DummySongDNA(str(path)))
    monkeypatch.setattr(ai_dj, "build_compatibility_report", lambda a, b: type("R", (), {"to_dict": lambda self: {"factors": {"overall": 0.8}}})())
    monkeypatch.setattr(ai_dj, "build_stub_arrangement_plan", lambda a, b: type("P", (), {"to_dict": lambda self: {"sections": [{"label": "intro"}]}})())

    outdir = tmp_path / "prototype"
    code = ai_dj.prototype(str(song_a), str(song_b), str(outdir))

    assert code == 0
    assert (outdir / "song_a_dna.json").exists()
    assert (outdir / "song_b_dna.json").exists()
    assert (outdir / "compatibility_report.json").exists()
    assert (outdir / "arrangement_plan.json").exists()


def test_doctor_json_output_includes_test_hint(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        ai_dj,
        "_dependency_status",
        lambda: {
            "librosa": {"ok": False, "kind": "python-module", "required_for": "analysis"},
            "numpy": {"ok": True, "kind": "python-module", "required_for": "analysis"},
            "soundfile": {"ok": True, "kind": "python-module", "required_for": "analysis"},
            "pytest": {"ok": True, "kind": "python-module", "required_for": "tests"},
            "ffmpeg": {"ok": False, "kind": "binary", "required_for": "mp3-export"},
        },
    )

    out = tmp_path / "doctor.json"
    code = ai_dj.doctor(str(out))
    payload = json.loads(out.read_text())

    assert code == 1
    assert payload["analysis_ready"] is False
    assert payload["test_ready"] is True
    assert payload["render_ready"] is False
    assert payload["checks"]["pytest"]["required_for"] == "tests"
