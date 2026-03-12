#!/usr/bin/env python3
"""
AI DJ CLI Tool
Generate, analyze, and fuse music tracks with AI.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional


class CliError(RuntimeError):
    """User-facing CLI error."""


_REQUIRED_ANALYSIS_MODULES = ("librosa", "numpy", "soundfile")
_OPTIONAL_TEST_MODULES = ("pytest",)
_OPTIONAL_RENDER_BINARIES = ("ffmpeg",)

# Test hooks / backwards-compatible monkeypatch targets.
analyze_audio_file = None
build_compatibility_report = None
build_stub_arrangement_plan = None
resolve_render_plan = None
render_resolved_plan = None


def _write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_existing_audio_path(path_str: str, label: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise CliError(f"{label} not found: {path}")
    if not path.is_file():
        raise CliError(f"{label} is not a file: {path}")
    return str(path)


def _import_or_raise(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown dependency"
        if missing in _REQUIRED_ANALYSIS_MODULES:
            raise CliError(
                "Missing Python dependency: "
                f"{missing}. Install project requirements first, for example:\n"
                "  python3 -m pip install -r requirements.txt"
            ) from exc
        raise CliError(
            f"Unable to import runtime dependency '{missing}'. "
            "Install project requirements first with:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from exc


def _get_analyze_audio_file() -> Any:
    if analyze_audio_file is not None:
        return analyze_audio_file
    return _import_or_raise("src.core.analysis").analyze_audio_file


def _get_planner_functions() -> tuple[Any, Any]:
    if build_compatibility_report is not None and build_stub_arrangement_plan is not None:
        return build_compatibility_report, build_stub_arrangement_plan
    planner = _import_or_raise("src.core.planner")
    return planner.build_compatibility_report, planner.build_stub_arrangement_plan


def _get_render_functions() -> tuple[Any, Any]:
    if resolve_render_plan is not None and render_resolved_plan is not None:
        return resolve_render_plan, render_resolved_plan
    render = _import_or_raise("src.core.render")
    return render.resolve_render_plan, render.render_resolved_plan


def _dependency_status() -> dict[str, dict[str, str | bool]]:
    status: dict[str, dict[str, str | bool]] = {}
    for module_name in _REQUIRED_ANALYSIS_MODULES:
        try:
            importlib.import_module(module_name)
            status[module_name] = {"ok": True, "kind": "python-module", "required_for": "analysis"}
        except ModuleNotFoundError:
            status[module_name] = {"ok": False, "kind": "python-module", "required_for": "analysis"}

    for module_name in _OPTIONAL_TEST_MODULES:
        try:
            importlib.import_module(module_name)
            status[module_name] = {"ok": True, "kind": "python-module", "required_for": "tests"}
        except ModuleNotFoundError:
            status[module_name] = {"ok": False, "kind": "python-module", "required_for": "tests"}

    for binary_name in _OPTIONAL_RENDER_BINARIES:
        status[binary_name] = {
            "ok": shutil.which(binary_name) is not None,
            "kind": "binary",
            "required_for": "mp3-export",
        }
    return status


def generate(genre: Optional[str], bpm: Optional[int], key: Optional[str], output: Optional[str]) -> int:
    """Generate a new track."""
    print("Generating track...")
    print(f"  Genre: {genre or 'auto-detect'}")
    print(f"  BPM: {bpm or 'auto-detect'}")
    print(f"  Key: {key or 'auto-detect'}")
    print(f"  Output: {output or 'output.mp3'}")
    return 0


def analyze(track: str, detailed: bool, output: Optional[str]) -> int:
    """Analyze a track's properties."""
    analyze_audio = _get_analyze_audio_file()
    track_path = _resolve_existing_audio_path(track, "track")
    result = analyze_audio(track_path).to_dict()

    if output:
        _write_json(output, result)
        print(f"Wrote analysis JSON: {output}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))

    if detailed:
        print("Detailed analysis complete.")

    return 0


def fusion(
    track1: str,
    track2: str,
    genre: Optional[str],
    bpm: Optional[int],
    key: Optional[str],
    output: Optional[str],
) -> int:
    """Fuse two tracks together."""
    analyze_audio = _get_analyze_audio_file()
    _, build_arrangement_plan = _get_planner_functions()
    resolve_plan, render_plan = _get_render_functions()
    track1_path = _resolve_existing_audio_path(track1, "track1")
    track2_path = _resolve_existing_audio_path(track2, "track2")

    outdir = Path(output or "runs/render-prototype").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if genre:
        print(f"Note: --genre is accepted for future render control but is not yet applied ({genre}).")
    if bpm:
        print(f"Note: --bpm is accepted for future render control but is not yet applied ({bpm}).")
    if key:
        print(f"Note: --key is accepted for future render control but is not yet applied ({key}).")

    song_a = analyze_audio(track1_path)
    song_b = analyze_audio(track2_path)
    plan = build_arrangement_plan(song_a, song_b)
    manifest = resolve_plan(plan, song_a, song_b)
    result = render_plan(manifest, outdir)
    if genre or bpm or key:
        print("Note: v1 render currently ignores target genre/BPM/key overrides and uses analyzed parent timing.")
    print("Render outputs:")
    print(f"  raw wav: {result.raw_wav_path}")
    print(f"  master wav: {result.master_wav_path}")
    print(f"  master mp3: {result.master_mp3_path or 'not written (ffmpeg unavailable)'}")
    print(f"  manifest: {result.manifest_path}")
    return 0


def prototype(song_a: str, song_b: str, output_dir: str, stems_dir: Optional[str] = None) -> int:
    """Run the first end-to-end prototype workflow for two songs."""
    analyze_audio = _get_analyze_audio_file()
    build_compatibility, build_arrangement_plan = _get_planner_functions()
    song_a_path = _resolve_existing_audio_path(song_a, "song_a")
    song_b_path = _resolve_existing_audio_path(song_b, "song_b")

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    stems_root = Path(stems_dir).expanduser().resolve() if stems_dir else None
    stems_a = stems_root / "song_a" if stems_root else None
    stems_b = stems_root / "song_b" if stems_root else None

    song_a_obj = analyze_audio(song_a_path, stems_dir=stems_a)
    song_b_obj = analyze_audio(song_b_path, stems_dir=stems_b)

    song_a_dna = song_a_obj.to_dict()
    song_b_dna = song_b_obj.to_dict()
    compatibility = build_compatibility(song_a_obj, song_b_obj).to_dict()
    arrangement = build_arrangement_plan(song_a_obj, song_b_obj).to_dict()

    paths = {
        "song_a_dna": outdir / "song_a_dna.json",
        "song_b_dna": outdir / "song_b_dna.json",
        "compatibility": outdir / "compatibility_report.json",
        "arrangement": outdir / "arrangement_plan.json",
    }

    _write_json(paths["song_a_dna"], song_a_dna)
    _write_json(paths["song_b_dna"], song_b_dna)
    _write_json(paths["compatibility"], compatibility)
    _write_json(paths["arrangement"], arrangement)

    print("Prototype artifacts written:")
    for label, path in paths.items():
        print(f"  {label}: {path}")

    return 0


def doctor(output_json: Optional[str] = None) -> int:
    """Report whether the local environment is ready for prototype analysis/render steps."""
    status = _dependency_status()
    analysis_ready = all(bool(status[name]["ok"]) for name in _REQUIRED_ANALYSIS_MODULES)
    test_ready = all(bool(status[name]["ok"]) for name in _OPTIONAL_TEST_MODULES)
    render_ready = analysis_ready and all(bool(status[name]["ok"]) for name in _OPTIONAL_RENDER_BINARIES)

    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "analysis_ready": analysis_ready,
        "test_ready": test_ready,
        "render_ready": render_ready,
        "checks": status,
        "install_hint": "python3 -m pip install -r requirements.txt",
        "test_hint": "python3 -m pytest -q",
        "ffmpeg_hint": "Install ffmpeg to enable MP3 export for fusion renders.",
    }

    if output_json:
        _write_json(output_json, payload)
        print(f"Wrote doctor report: {output_json}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    if analysis_ready:
        print("Environment check: analysis path is ready.")
    else:
        print("Environment check: analysis path is NOT ready.")
    if test_ready:
        print("Environment check: test runner is ready (use: python3 -m pytest -q).")
    else:
        print("Environment check: pytest is not importable; install requirements before running tests.")
    if render_ready:
        print("Environment check: render path is ready.")
    else:
        print("Environment check: render path is missing at least one dependency.")

    return 0 if analysis_ready else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ai-dj",
        description="AI DJ - Generate, analyze, and plan song fusion artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Suggested first checkpoint:
  python3 ai_dj.py doctor
  python3 ai_dj.py analyze song.wav --output runs/checkpoint/song_dna.json
  python3 ai_dj.py prototype song_a.wav song_b.wav --output-dir runs/prototype-001
  python3 ai_dj.py fusion song_a.wav song_b.wav --output runs/render-prototype
''',
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    gen_parser = subparsers.add_parser("generate", help="Generate a new track")
    gen_parser.add_argument("--genre", "-g", help="Music genre")
    gen_parser.add_argument("--bpm", "-b", type=int, help="Beats per minute")
    gen_parser.add_argument("--key", "-k", help="Musical key (e.g., C minor)")
    gen_parser.add_argument("--output", "-o", help="Output file path")

    ana_parser = subparsers.add_parser("analyze", help="Analyze a track")
    ana_parser.add_argument("track", help="Path to track file")
    ana_parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed analysis")
    ana_parser.add_argument("--output", "-o", help="Path to output JSON")

    fus_parser = subparsers.add_parser("fusion", help="Render a first-pass fused audio prototype")
    fus_parser.add_argument("track1", help="Path to first track")
    fus_parser.add_argument("track2", help="Path to second track")
    fus_parser.add_argument("--genre", "-g", help="Target genre for fusion (accepted but not yet applied)")
    fus_parser.add_argument("--bpm", "-b", type=int, help="Target BPM (accepted but not yet applied)")
    fus_parser.add_argument("--key", "-k", help="Target musical key")
    fus_parser.add_argument("--output", "-o", help="Output directory for render artifacts")

    proto_parser = subparsers.add_parser("prototype", help="Generate first-pass two-song prototype artifacts")
    proto_parser.add_argument("song_a", help="Path to parent song A")
    proto_parser.add_argument("song_b", help="Path to parent song B")
    proto_parser.add_argument("--output-dir", "-o", required=True, help="Directory to write prototype artifacts")
    proto_parser.add_argument("--stems-dir", help="Optional directory for stem outputs")

    doctor_parser = subparsers.add_parser("doctor", help="Check whether local dependencies are installed")
    doctor_parser.add_argument("--output", "-o", help="Optional path to write dependency report JSON")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "generate":
            return generate(args.genre, args.bpm, args.key, args.output)
        if args.command == "analyze":
            return analyze(args.track, args.detailed, args.output)
        if args.command == "fusion":
            return fusion(args.track1, args.track2, args.genre, args.bpm, args.key, args.output)
        if args.command == "prototype":
            return prototype(args.song_a, args.song_b, args.output_dir, args.stems_dir)
        if args.command == "doctor":
            return doctor(args.output)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
