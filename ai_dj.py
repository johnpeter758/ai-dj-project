#!/usr/bin/env python3
"""
AI DJ CLI Tool
Generate, analyze, and fuse music tracks with AI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

LISTEN_COMPONENT_KEYS = ("structure", "groove", "energy_arc", "transition", "coherence", "mix_sanity")


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
evaluate_song = None


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


def _get_evaluate_song() -> Any:
    if evaluate_song is not None:
        return evaluate_song
    return _import_or_raise("src.core.evaluation").evaluate_song


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


def _is_listen_report_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and "overall_score" in payload and all(key in payload for key in LISTEN_COMPONENT_KEYS)


def _is_render_manifest_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and "outputs" in payload and isinstance(payload.get("outputs"), dict)


def _pick_render_audio_path(manifest: dict[str, Any], manifest_path: Path) -> str:
    outputs = manifest.get("outputs") or {}
    for key in ("master_wav", "master_mp3", "raw_wav"):
        candidate = outputs.get(key)
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        if path.exists() and path.is_file():
            return str(path)
    raise CliError(f"render manifest does not point to an existing render output: {manifest_path}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"invalid JSON file: {path}") from exc


def _short_label(path_str: Optional[str], fallback: str) -> str:
    if not path_str:
        return fallback
    return Path(path_str).name or fallback


def _stable_compare_output_path(left_input: str, right_input: str) -> Path:
    left_name = Path(left_input).expanduser().name or "left"
    right_name = Path(right_input).expanduser().name or "right"
    slug_left = ''.join(ch if ch.isalnum() else '_' for ch in left_name.lower()).strip('_') or 'left'
    slug_right = ''.join(ch if ch.isalnum() else '_' for ch in right_name.lower()).strip('_') or 'right'
    digest = hashlib.sha1(f"{Path(left_input).expanduser().resolve()}||{Path(right_input).expanduser().resolve()}".encode("utf-8")).hexdigest()[:10]
    return (Path("runs") / "compare_listen" / f"{slug_left}__vs__{slug_right}__{digest}.json").resolve()


def _resolve_output_path(output: Optional[str], default_path: Optional[Path] = None, default_filename: Optional[str] = None) -> Optional[Path]:
    if output is None:
        return default_path.resolve() if default_path else None

    output_path = Path(output).expanduser()
    if output_path.exists() and output_path.is_dir():
        if not default_filename:
            raise CliError(f"output path is a directory but no default filename is available: {output_path}")
        return (output_path / default_filename).resolve()
    if output.endswith(("/", "\\")):
        if not default_filename:
            raise CliError(f"output path looks like a directory but no default filename is available: {output_path}")
        return (output_path / default_filename).resolve()
    return output_path.resolve()


def _metric_label(key: str) -> str:
    return key.replace("_", " ")


def _decision_confidence(overall_delta: float, decisive_component_count: int) -> str:
    magnitude = abs(overall_delta)
    if magnitude < 1e-6:
        return "tie"
    if magnitude >= 12.0 or decisive_component_count >= 3:
        return "clear"
    if magnitude >= 5.0 or decisive_component_count >= 2:
        return "leaning"
    return "narrow"


def _component_reason_block(
    key: str,
    winner_side: str,
    delta: float,
    left_report: dict[str, Any],
    right_report: dict[str, Any],
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    winner_report = left_report if winner_side == "left" else right_report
    loser_report = right_report if winner_side == "left" else left_report
    winner_label = left_label if winner_side == "left" else right_label
    loser_label = right_label if winner_side == "left" else left_label
    winner_component = winner_report.get(key) or {}
    loser_component = loser_report.get(key) or {}
    return {
        "component": key,
        "label": _metric_label(key),
        "winner": winner_side,
        "winner_label": winner_label,
        "loser_label": loser_label,
        "delta": round(abs(delta), 1),
        "winner_summary": winner_component.get("summary"),
        "loser_summary": loser_component.get("summary"),
        "winner_evidence": list((winner_component.get("evidence") or [])[:2]),
        "loser_fixes": list((loser_component.get("fixes") or [])[:2]),
    }


def _build_comparison_decision(
    left_report: dict[str, Any],
    right_report: dict[str, Any],
    component_deltas: dict[str, float],
    overall_delta: float,
    overall_winner: str,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    decisive = [
        _component_reason_block(key, "left" if delta > 0 else "right", delta, left_report, right_report, left_label, right_label)
        for key, delta in sorted(component_deltas.items(), key=lambda item: abs(item[1]), reverse=True)
        if abs(delta) >= 0.5
    ]
    deciding_components = decisive[:3]

    if overall_winner == "tie":
        why = [
            f"Tie: {left_label} and {right_label} land on the same overall listen score.",
        ]
        if deciding_components:
            why.append(
                "Largest tradeoff: "
                + "; ".join(
                    f"{item['winner_label']} leads on {item['label']} (+{item['delta']:.1f})"
                    for item in deciding_components[:2]
                )
                + "."
            )
        if left_report.get("verdict") != right_report.get("verdict"):
            why.append(
                f"Verdicts still differ: {left_label}={left_report.get('verdict')} vs {right_label}={right_report.get('verdict')}."
            )
        return {
            "winner": "tie",
            "winner_label": "tie",
            "loser_label": None,
            "confidence": "tie",
            "deciding_components": deciding_components,
            "why": why,
            "winner_reasons": [],
            "loser_fixes": [],
        }

    winner_report = left_report if overall_winner == "left" else right_report
    loser_report = right_report if overall_winner == "left" else left_report
    winner_label = left_label if overall_winner == "left" else right_label
    loser_label = right_label if overall_winner == "left" else left_label
    confidence = _decision_confidence(overall_delta, len(deciding_components))

    why = [
        f"{winner_label} wins overall by {abs(overall_delta):.1f} listen points over {loser_label}.",
    ]
    if deciding_components:
        why.extend(
            f"Deciding edge: {item['winner_label']} is stronger on {item['label']} (+{item['delta']:.1f}) — {item['winner_summary']}"
            for item in deciding_components
        )
    if winner_report.get("verdict") != loser_report.get("verdict"):
        why.append(
            f"Verdict shift: {winner_label} is rated {winner_report.get('verdict')} while {loser_label} is rated {loser_report.get('verdict')}."
        )

    return {
        "winner": overall_winner,
        "winner_label": winner_label,
        "loser_label": loser_label,
        "confidence": confidence,
        "deciding_components": deciding_components,
        "why": why,
        "winner_reasons": list((winner_report.get("top_reasons") or [])[:4]),
        "loser_fixes": list((loser_report.get("top_fixes") or [])[:4]),
    }


def _summarize_comparison(left: dict[str, Any], right: dict[str, Any], deltas: dict[str, Any], left_label: str, right_label: str) -> list[str]:
    lines = []
    overall_delta = float(deltas["overall_score_delta"])
    if abs(overall_delta) < 1e-6:
        lines.append(f"Overall: tie — {left_label} and {right_label} land on the same listen score.")
    elif overall_delta > 0:
        lines.append(f"Overall: {left_label} wins by {overall_delta:.1f} listen points over {right_label}.")
    else:
        lines.append(f"Overall: {right_label} wins by {abs(overall_delta):.1f} listen points over {left_label}.")

    ranked = sorted(
        ((key, abs(float(deltas["component_score_deltas"][key]))) for key in LISTEN_COMPONENT_KEYS),
        key=lambda item: item[1],
        reverse=True,
    )
    for key, magnitude in ranked[:3]:
        delta = float(deltas["component_score_deltas"][key])
        if magnitude < 0.1:
            continue
        winner = left_label if delta > 0 else right_label
        loser = right_label if delta > 0 else left_label
        lines.append(f"Edge: {winner} is stronger on {_metric_label(key)} by {abs(delta):.1f} points vs {loser}.")

    if left.get("verdict") != right.get("verdict"):
        lines.append(f"Verdict shift: {left_label}={left.get('verdict')} while {right_label}={right.get('verdict')}.")
    return lines


def _resolve_compare_input(input_path: str) -> dict[str, Any]:
    analyze_audio = _get_analyze_audio_file()
    evaluate = _get_evaluate_song()

    raw_input = str(Path(input_path).expanduser())
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise CliError(f"compare input not found: {path}")

    render_manifest_path: Path | None = None
    report_origin = "audio"
    analyzed_path = path

    if path.is_dir():
        manifest_candidate = path / "render_manifest.json"
        if not manifest_candidate.exists():
            raise CliError(f"directory does not contain render_manifest.json: {path}")
        render_manifest_path = manifest_candidate.resolve()
        manifest = _load_json(render_manifest_path)
        analyzed_path = Path(_pick_render_audio_path(manifest, render_manifest_path))
        report_origin = "render_output"
    elif path.suffix.lower() == ".json":
        payload = _load_json(path)
        if _is_listen_report_payload(payload):
            return {
                "input_path": str(path),
                "input_label": _short_label(raw_input, path.name or "listen_report"),
                "report_origin": "listen_report",
                "resolved_audio_path": payload.get("source_path"),
                "render_manifest_path": None,
                "report": payload,
            }
        if _is_render_manifest_payload(payload):
            render_manifest_path = path
            analyzed_path = Path(_pick_render_audio_path(payload, path))
            report_origin = "render_output"
        else:
            raise CliError(f"JSON input is neither a listen report nor a render manifest: {path}")
    elif not path.is_file():
        raise CliError(f"compare input is not a supported file: {path}")

    song = analyze_audio(str(analyzed_path))
    report = evaluate(song).to_dict()
    return {
        "input_path": str(path),
        "input_label": _short_label(raw_input, path.name or report_origin),
        "report_origin": report_origin,
        "resolved_audio_path": str(analyzed_path),
        "render_manifest_path": str(render_manifest_path) if render_manifest_path else None,
        "report": report,
    }


def _build_listen_comparison(left_input: str, right_input: str) -> dict[str, Any]:
    left = _resolve_compare_input(left_input)
    right = _resolve_compare_input(right_input)
    left_report = left["report"]
    right_report = right["report"]

    component_deltas = {
        key: round(float(left_report[key]["score"]) - float(right_report[key]["score"]), 1)
        for key in LISTEN_COMPONENT_KEYS
    }
    overall_delta = round(float(left_report["overall_score"]) - float(right_report["overall_score"]), 1)

    if overall_delta > 0:
        overall_winner = "left"
    elif overall_delta < 0:
        overall_winner = "right"
    else:
        overall_winner = "tie"

    component_winners = {
        key: ("left" if delta > 0 else "right" if delta < 0 else "tie")
        for key, delta in component_deltas.items()
    }

    left_label = left.get("input_label") or "left"
    right_label = right.get("input_label") or "right"
    comparison = {
        "schema_version": "0.1.0",
        "left": left,
        "right": right,
        "deltas": {
            "overall_score_delta": overall_delta,
            "component_score_deltas": component_deltas,
        },
        "winner": {
            "overall": overall_winner,
            "components": component_winners,
        },
        "decision": _build_comparison_decision(
            left_report,
            right_report,
            component_deltas,
            overall_delta,
            overall_winner,
            left_label,
            right_label,
        ),
        "summary": _summarize_comparison(
            left_report,
            right_report,
            {
                "overall_score_delta": overall_delta,
                "component_score_deltas": component_deltas,
            },
            left_label,
            right_label,
        ),
    }
    return comparison


def listen(track: str, output: Optional[str]) -> int:
    analyze_audio = _get_analyze_audio_file()
    evaluate = _get_evaluate_song()
    track_path = _resolve_existing_audio_path(track, "track")
    song = analyze_audio(track_path)
    report = evaluate(song).to_dict()

    resolved_output = _resolve_output_path(output)
    if resolved_output:
        _write_json(resolved_output, report)
        print(f"Wrote listen report: {resolved_output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    print(f"Track: {Path(track_path).name}")
    print(f"Overall score: {report['overall_score']}")
    print(f"Verdict: {report['verdict']}")
    for key in LISTEN_COMPONENT_KEYS:
        part = report[key]
        print(f"- {key}: {part['score']} — {part['summary']}")
    return 0


def compare_listen(left: str, right: str, output: Optional[str]) -> int:
    comparison = _build_listen_comparison(left, right)
    resolved_output = _resolve_output_path(
        output,
        default_path=_stable_compare_output_path(left, right),
        default_filename="listen_compare.json",
    )
    if resolved_output:
        _write_json(resolved_output, comparison)
        print(f"Wrote listen comparison: {resolved_output}")

    if output is None:
        print(json.dumps(comparison, indent=2, sort_keys=True))

    left_label = comparison['left'].get('input_label', 'left')
    right_label = comparison['right'].get('input_label', 'right')
    print(f"Compare: {left_label} vs {right_label}")
    print(f"Overall winner: {comparison['winner']['overall']}")
    print(f"Overall score delta ({left_label} - {right_label}): {comparison['deltas']['overall_score_delta']:+.1f}")
    for key in LISTEN_COMPONENT_KEYS:
        delta = comparison['deltas']['component_score_deltas'][key]
        winner = comparison['winner']['components'][key]
        winner_label = left_label if winner == 'left' else right_label if winner == 'right' else 'tie'
        print(f"- {_metric_label(key)}: {winner_label} ({delta:+.1f})")
    decision = comparison.get('decision') or {}
    if decision.get('confidence'):
        print(f"Decision confidence: {decision['confidence']}")
    for line in decision.get('why') or []:
        print(f"  Why: {line}")
    for line in comparison['summary']:
        print(f"  {line}")
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
  python3 ai_dj.py listen song.wav --output runs/checkpoint/listen_report.json
  python3 ai_dj.py compare-listen left.json right.json --output runs/checkpoint/listen_compare.json
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

    listen_parser = subparsers.add_parser("listen", help="Evaluate how musically strong/coherent a track appears")
    listen_parser.add_argument("track", help="Path to track file")
    listen_parser.add_argument("--output", "-o", help="Path to output JSON")

    compare_parser = subparsers.add_parser("compare-listen", help="Compare two listen reports, audio files, or rendered outputs")
    compare_parser.add_argument("left", help="Left input: listen JSON, audio file, render manifest JSON, or render output directory")
    compare_parser.add_argument("right", help="Right input: listen JSON, audio file, render manifest JSON, or render output directory")
    compare_parser.add_argument("--output", "-o", help="Path to output comparison JSON")

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
        if args.command == "listen":
            return listen(args.track, args.output)
        if args.command == "compare-listen":
            return compare_listen(args.left, args.right, args.output)
        if args.command == "doctor":
            return doctor(args.output)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
