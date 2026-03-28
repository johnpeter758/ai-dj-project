from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path


LISTEN_COMPONENT_KEYS = (
    "structure",
    "groove",
    "energy_arc",
    "transition",
    "coherence",
    "mix_sanity",
    "song_likeness",
)


def _write_demo_tone(path: Path, *, freq_hz: float, duration_sec: float = 2.0, sample_rate: int = 22050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            sample = int(12000 * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate)))
            wav.writeframes(struct.pack("<h", sample))


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def test_song_birth_lightweight_end_to_end_generates_artifacts_and_score_schema(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "demo_inputs"
    track_a = input_dir / "a.wav"
    track_b = input_dir / "b.wav"
    _write_demo_tone(track_a, freq_hz=220.0)
    _write_demo_tone(track_b, freq_hz=330.0)

    outdir = tmp_path / "song_birth_integration"
    fusion_cmd = [
        sys.executable,
        "ai_dj.py",
        "fusion",
        str(track_a),
        str(track_b),
        "--output",
        str(outdir),
        "--arrangement-mode",
        "baseline",
    ]
    fusion = _run(fusion_cmd, cwd=repo_root)
    assert fusion.returncode == 0, f"fusion failed\nstdout:\n{fusion.stdout}\nstderr:\n{fusion.stderr}"

    manifest_path = outdir / "render_manifest.json"
    master_wav_path = outdir / "child_master.wav"
    raw_wav_path = outdir / "child_raw.wav"
    assert manifest_path.exists(), "missing render manifest"
    assert master_wav_path.exists(), "missing master wav artifact"
    assert raw_wav_path.exists(), "missing raw wav artifact"

    listen_report_path = outdir / "listen_report.json"
    listen_cmd = [
        sys.executable,
        "ai_dj.py",
        "listen",
        str(master_wav_path),
        "--output",
        str(listen_report_path),
    ]
    listen = _run(listen_cmd, cwd=repo_root)
    assert listen.returncode == 0, f"listen failed\nstdout:\n{listen.stdout}\nstderr:\n{listen.stderr}"
    assert listen_report_path.exists(), "missing listen report artifact"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    assert outputs.get("master_wav"), "manifest missing outputs.master_wav"
    assert outputs.get("raw_wav"), "manifest missing outputs.raw_wav"

    report = json.loads(listen_report_path.read_text(encoding="utf-8"))
    assert "overall_score" in report
    assert isinstance(report["overall_score"], (int, float))
    for key in LISTEN_COMPONENT_KEYS:
        component = report.get(key)
        assert isinstance(component, dict), f"missing score component: {key}"
        assert isinstance(component.get("score"), (int, float)), f"missing numeric score for {key}"
        assert isinstance(component.get("summary"), str), f"missing summary for {key}"
