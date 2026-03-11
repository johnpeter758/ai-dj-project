#!/usr/bin/env python3
"""
AI DJ CLI Tool
Generate, analyze, and fuse music tracks with AI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.core.analysis import analyze_audio_file
from src.core.planner import build_compatibility_report, build_stub_arrangement_plan


def _write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    result = analyze_audio_file(track).to_dict()

    if output:
        _write_json(output, result)
        print(f"Wrote analysis JSON: {output}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))

    if detailed:
        print("Detailed analysis complete.")

    return 0


def fusion(track1: str, track2: str, genre: Optional[str], bpm: Optional[int], output: Optional[str]) -> int:
    """Fuse two tracks together."""
    print("Fusion/render is not implemented yet.")
    print("Use the prototype command to generate analysis and planning artifacts.")
    print(f"  Track 1: {track1}")
    print(f"  Track 2: {track2}")
    print(f"  Genre: {genre or 'blend'}")
    print(f"  BPM: {bpm or 'auto-blend'}")
    print(f"  Output: {output or 'fusion.mp3'}")
    return 0


def prototype(song_a: str, song_b: str, output_dir: str, stems_dir: Optional[str] = None) -> int:
    """Run the first end-to-end prototype workflow for two songs."""
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    stems_a = Path(stems_dir) / "song_a" if stems_dir else None
    stems_b = Path(stems_dir) / "song_b" if stems_dir else None

    song_a_dna = analyze_audio_file(song_a, stems_dir=stems_a).to_dict()
    song_b_dna = analyze_audio_file(song_b, stems_dir=stems_b).to_dict()

    # Rehydrate via analyzer output for planner inputs
    song_a_obj = analyze_audio_file(song_a, stems_dir=stems_a)
    song_b_obj = analyze_audio_file(song_b, stems_dir=stems_b)

    compatibility = build_compatibility_report(song_a_obj, song_b_obj).to_dict()
    arrangement = build_stub_arrangement_plan(song_a_obj, song_b_obj).to_dict()

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


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ai-dj",
        description="AI DJ - Generate, analyze, and plan song fusion artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  ai-dj analyze my_track.mp3 --output out/song_dna.json
  ai-dj prototype song_a.mp3 song_b.mp3 --output-dir runs/prototype-001
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

    fus_parser = subparsers.add_parser("fusion", help="Legacy fusion placeholder")
    fus_parser.add_argument("track1", help="Path to first track")
    fus_parser.add_argument("track2", help="Path to second track")
    fus_parser.add_argument("--genre", "-g", help="Target genre for fusion")
    fus_parser.add_argument("--bpm", "-b", type=int, help="Target BPM")
    fus_parser.add_argument("--key", "-k", help="Target musical key")
    fus_parser.add_argument("--output", "-o", help="Output file path")

    proto_parser = subparsers.add_parser("prototype", help="Generate first-pass two-song prototype artifacts")
    proto_parser.add_argument("song_a", help="Path to parent song A")
    proto_parser.add_argument("song_b", help="Path to parent song B")
    proto_parser.add_argument("--output-dir", "-o", required=True, help="Directory to write prototype artifacts")
    proto_parser.add_argument("--stems-dir", help="Optional directory for stem outputs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "generate":
        return generate(args.genre, args.bpm, args.key, args.output)
    if args.command == "analyze":
        return analyze(args.track, args.detailed, args.output)
    if args.command == "fusion":
        return fusion(args.track1, args.track2, args.genre, args.bpm, args.output)
    if args.command == "prototype":
        return prototype(args.song_a, args.song_b, args.output_dir, args.stems_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
