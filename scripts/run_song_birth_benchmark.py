#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any


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


def _resolve_default_pair2(repo_root: Path) -> tuple[Path, Path]:
    clips_dir = repo_root / "runs" / "live_fuse_batch_fast_20260325_104746" / "clips"
    return clips_dir / "b.mp3", clips_dir / "c.mp3"


def _validate_listen_schema(report: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if not isinstance(report.get("overall_score"), (int, float)):
        notes.append("listen report missing numeric overall_score")
    for key in LISTEN_COMPONENT_KEYS:
        component = report.get(key)
        if not isinstance(component, dict):
            notes.append(f"listen report missing component: {key}")
            continue
        if not isinstance(component.get("score"), (int, float)):
            notes.append(f"listen component {key} missing numeric score")
        if not isinstance(component.get("summary"), str):
            notes.append(f"listen component {key} missing summary")
    return notes


def _run_fusion_and_listen(repo_root: Path, track_a: Path, track_b: Path, outdir: Path) -> tuple[int, dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    fusion_cmd = [
        sys.executable,
        "ai_dj.py",
        "fusion",
        str(track_a),
        str(track_b),
        "--output",
        str(outdir),
        "--arrangement-mode",
        "pro",
    ]
    fusion = _run(fusion_cmd, cwd=repo_root)

    listen_report_path = outdir / "listen_report.json"
    master_wav = outdir / "child_master.wav"
    listen = None
    if fusion.returncode == 0 and master_wav.exists():
        listen_cmd = [
            sys.executable,
            "ai_dj.py",
            "listen",
            str(master_wav),
            "--output",
            str(listen_report_path),
        ]
        listen = _run(listen_cmd, cwd=repo_root)

    payload = {
        "fusion": {
            "returncode": fusion.returncode,
            "stdout_tail": fusion.stdout[-3000:],
            "stderr_tail": fusion.stderr[-3000:],
            "cmd": fusion_cmd,
        },
        "listen": {
            "returncode": (listen.returncode if listen else None),
            "stdout_tail": (listen.stdout[-3000:] if listen else ""),
            "stderr_tail": (listen.stderr[-3000:] if listen else ""),
        },
        "artifacts": {
            "outdir": str(outdir),
            "manifest": str(outdir / "render_manifest.json"),
            "selection": str(outdir / "fusion_selection.json"),
            "raw_wav": str(outdir / "child_raw.wav"),
            "master_wav": str(master_wav),
            "master_mp3": str(outdir / "child_master.mp3"),
            "listen_report": str(listen_report_path),
        },
    }
    rc = 0
    if fusion.returncode != 0:
        rc = fusion.returncode
    elif listen and listen.returncode != 0:
        rc = listen.returncode
    return rc, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Song Birth phase-12 benchmark pass and emit JSON summary.")
    parser.add_argument("--clip-a", help="Path to clip A (defaults to known pair2 path)")
    parser.add_argument("--clip-b", help="Path to clip B (defaults to known pair2 path)")
    parser.add_argument("--output-dir", help="Output run dir (default: runs/song_birth_phase12_<timestamp>)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (repo_root / "runs" / f"song_birth_phase12_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    default_a, default_b = _resolve_default_pair2(repo_root)
    requested_a = Path(args.clip_a).expanduser().resolve() if args.clip_a else default_a
    requested_b = Path(args.clip_b).expanduser().resolve() if args.clip_b else default_b

    mode = "real"
    fallback_reason = ""
    track_a = requested_a
    track_b = requested_b

    if not requested_a.exists() or not requested_b.exists():
        mode = "demo"
        missing = []
        if not requested_a.exists():
            missing.append(str(requested_a))
        if not requested_b.exists():
            missing.append(str(requested_b))
        fallback_reason = f"real clips missing; fallback to demo tones ({', '.join(missing)})"
        demo_dir = outdir / "demo_inputs"
        track_a = demo_dir / "a_demo.wav"
        track_b = demo_dir / "b_demo.wav"
        _write_demo_tone(track_a, freq_hz=220.0)
        _write_demo_tone(track_b, freq_hz=330.0)

    rc, run_payload = _run_fusion_and_listen(repo_root, track_a, track_b, outdir)

    notes: list[str] = []
    if fallback_reason:
        notes.append(f"FALLBACK: {fallback_reason}")

    artifacts = run_payload["artifacts"]
    for key in ("manifest", "raw_wav", "master_wav", "listen_report"):
        if not Path(artifacts[key]).exists():
            notes.append(f"missing artifact: {key} -> {artifacts[key]}")

    listen_report = {}
    listen_path = Path(artifacts["listen_report"])
    if listen_path.exists():
        try:
            listen_report = json.loads(listen_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            notes.append(f"listen report JSON decode failed: {exc}")
        else:
            notes.extend(_validate_listen_schema(listen_report))

    passed = rc == 0 and not notes
    summary = {
        "phase": "song_birth_phase12",
        "timestamp": ts,
        "mode": mode,
        "requested_inputs": {"clip_a": str(requested_a), "clip_b": str(requested_b)},
        "resolved_inputs": {"clip_a": str(track_a), "clip_b": str(track_b)},
        "passed": passed,
        "notes": notes,
        "scores": {
            "overall_score": listen_report.get("overall_score") if isinstance(listen_report, dict) else None,
            "transition_score": ((listen_report.get("transition") or {}).get("score") if isinstance(listen_report, dict) else None),
            "song_likeness_score": ((listen_report.get("song_likeness") or {}).get("score") if isinstance(listen_report, dict) else None),
            "gating_status": ((listen_report.get("gating") or {}).get("status") if isinstance(listen_report, dict) else None),
        },
        "artifacts": artifacts,
        "commands": {
            "fusion": run_payload["fusion"],
            "listen": run_payload["listen"],
        },
    }

    summary_path = outdir / "song_birth_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Song Birth benchmark summary: {summary_path}")
    print(f"Mode: {mode}")
    if fallback_reason:
        print(f"Fallback: {fallback_reason}")
    print(f"Passed: {passed}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
