#!/usr/bin/env python3
"""Minimal 2-input -> 1-output wrapper for VocalFusion.

This script intentionally keeps a tiny interface:
- Input A song path
- Input B song path
- Output directory

It shells out to the existing ai_dj.py fusion flow and emits a tiny JSON report
so automation can consume stable fields.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def build_fusion_command(
    *,
    repo_root: Path,
    song_a: Path,
    song_b: Path,
    output_dir: Path,
    arrangement_mode: str,
) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "ai_dj.py"),
        "fusion",
        str(song_a),
        str(song_b),
        "--output",
        str(output_dir),
        "--arrangement-mode",
        arrangement_mode,
    ]


def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), check=False, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run simple two-input VocalFusion")
    parser.add_argument("song_a", help="Path to first input song")
    parser.add_argument("song_b", help="Path to second input song")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--arrangement-mode", default="pro", choices=["pro", "safe", "adaptive", "baseline"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    song_a = Path(args.song_a).expanduser().resolve()
    song_b = Path(args.song_b).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not song_a.exists():
        print(f"error: song_a missing: {song_a}", file=sys.stderr)
        return 2
    if not song_b.exists():
        print(f"error: song_b missing: {song_b}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_fusion_command(
        repo_root=repo_root,
        song_a=song_a,
        song_b=song_b,
        output_dir=output_dir,
        arrangement_mode=args.arrangement_mode,
    )
    result = run_command(cmd)

    report = {
        "ok": result.returncode == 0,
        "command": cmd,
        "returncode": result.returncode,
        "artifacts": {
            "selection": str(output_dir / "fusion_selection.json"),
            "manifest": str(output_dir / "render_manifest.json"),
            "master_wav": str(output_dir / "child_master.wav"),
            "raw_wav": str(output_dir / "child_raw.wav"),
        },
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }

    report_path = output_dir / "two_input_one_output_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    if result.returncode != 0:
        print(json.dumps(report, indent=2), file=sys.stderr)
        return result.returncode

    print(json.dumps({"ok": True, "report": str(report_path), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
