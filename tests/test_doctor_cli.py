import json
from pathlib import Path

import ai_dj


def test_doctor_reports_test_readiness_and_writes_json(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        ai_dj,
        "_dependency_status",
        lambda: {
            "librosa": {"ok": True, "kind": "python-module", "required_for": "analysis"},
            "numpy": {"ok": True, "kind": "python-module", "required_for": "analysis"},
            "soundfile": {"ok": True, "kind": "python-module", "required_for": "analysis"},
            "pytest": {"ok": False, "kind": "python-module", "required_for": "tests"},
            "ffmpeg": {"ok": True, "kind": "binary", "required_for": "mp3-export"},
        },
    )

    out = tmp_path / "doctor.json"
    code = ai_dj.doctor(str(out))
    payload = json.loads(out.read_text())

    assert code == 0
    assert payload["analysis_ready"] is True
    assert payload["test_ready"] is False
    assert payload["render_ready"] is True
    assert payload["test_hint"] == "python3 -m pytest -q"
    assert payload["python_executable"]
    assert payload["python_version"]
