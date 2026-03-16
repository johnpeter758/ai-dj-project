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
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

LISTEN_COMPONENT_KEYS = ("structure", "groove", "energy_arc", "transition", "coherence", "mix_sanity", "song_likeness")


LISTENER_AGENT_COMPONENT_WEIGHTS = {
    "overall_score": 0.32,
    "song_likeness": 0.22,
    "groove": 0.14,
    "energy_arc": 0.14,
    "transition": 0.08,
    "coherence": 0.06,
    "mix_sanity": 0.04,
}

LISTENER_AGENT_CRITICAL_RANK_TARGETS = {
    "song_likeness": 70.0,
    "groove": 68.0,
    "energy_arc": 68.0,
    "transition": 68.0,
}

LISTENER_AGENT_RANK_PENALTY_SCALE = 0.35

LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS = {
    "overall_score": 55.0,
    "song_likeness": 45.0,
    "groove": 45.0,
    "energy_arc": 45.0,
    "transition": 45.0,
}

LISTENER_AGENT_SURVIVOR_MINIMUMS = {
    "overall_score": 70.0,
    "song_likeness": 60.0,
    "groove": 55.0,
    "energy_arc": 55.0,
}


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


def _stable_case_id(path_str: str) -> str:
    resolved = str(Path(path_str).expanduser().resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]


def _path_tail_label(path_str: str, depth: int) -> str:
    path = Path(path_str).expanduser().resolve()
    parts = list(path.parts)
    if parts and parts[0] == path.anchor:
        parts = parts[1:]
    if not parts:
        return path.name or str(path)
    depth = max(1, min(int(depth), len(parts)))
    return "/".join(parts[-depth:])


def _assign_display_labels(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = [dict(item) for item in items]
    if not labeled:
        return labeled

    depths = [1] * len(labeled)
    max_depths = []
    for item in labeled:
        path = Path(str(item.get("input_path") or "")).expanduser().resolve()
        parts = list(path.parts)
        if parts and parts[0] == path.anchor:
            parts = parts[1:]
        max_depths.append(max(1, len(parts)))

    labels = [str(item.get("input_label") or item.get("input_path") or f"case_{index}") for index, item in enumerate(labeled)]
    while True:
        labels = [_path_tail_label(str(item.get("input_path") or ""), depth) for item, depth in zip(labeled, depths)]
        collisions: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            collisions.setdefault(label, []).append(index)
        duplicate_groups = [indexes for indexes in collisions.values() if len(indexes) > 1]
        if not duplicate_groups:
            break

        advanced = False
        for indexes in duplicate_groups:
            for index in indexes:
                if depths[index] < max_depths[index]:
                    depths[index] += 1
                    advanced = True
        if not advanced:
            labels = [
                f"{label}#{str(item.get('case_id') or '')[:6]}" if len(collisions.get(label, [])) > 1 else label
                for item, label in zip(labeled, labels)
            ]
            break

    for item, label in zip(labeled, labels):
        item["display_label"] = label
    return labeled


def _report_component_snapshot(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for key in LISTEN_COMPONENT_KEYS:
        component = report.get(key) or {}
        snapshot[key] = {
            "score": float(component.get("score") or 0.0),
            "summary": component.get("summary"),
        }
    return snapshot


def _report_strengths_and_weaknesses(report: dict[str, Any], limit: int = 2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = [
        {
            "component": key,
            "label": _metric_label(key),
            "score": float((report.get(key) or {}).get("score") or 0.0),
            "summary": (report.get(key) or {}).get("summary"),
        }
        for key in LISTEN_COMPONENT_KEYS
    ]
    strengths = sorted(ranked, key=lambda item: (-item["score"], item["component"]))[:limit]
    weaknesses = sorted(ranked, key=lambda item: (item["score"], item["component"]))[:limit]
    return strengths, weaknesses


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


def _flatten_numeric_details(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            metrics.update(_flatten_numeric_details(value, metric_key))
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[metric_key] = float(value)
    return metrics


def _component_numeric_details(component: dict[str, Any]) -> dict[str, float]:
    details = component.get("details") or {}
    return _flatten_numeric_details(details)


def _component_profile_match(left_component: dict[str, Any], right_component: dict[str, Any]) -> dict[str, Any]:
    left_metrics = _component_numeric_details(left_component)
    right_metrics = _component_numeric_details(right_component)
    shared = sorted(set(left_metrics) & set(right_metrics))
    if not shared:
        return {
            "similarity": None,
            "shared_metric_count": 0,
            "largest_gaps": [],
        }

    gaps = []
    for key in shared:
        left_value = float(left_metrics[key])
        right_value = float(right_metrics[key])
        scale = max(abs(left_value), abs(right_value), 1.0)
        normalized_gap = min(1.0, abs(left_value - right_value) / scale)
        gaps.append({
            "metric": key,
            "left": round(left_value, 4),
            "right": round(right_value, 4),
            "delta": round(left_value - right_value, 4),
            "normalized_gap": round(normalized_gap, 4),
        })

    mean_gap = sum(float(item["normalized_gap"]) for item in gaps) / len(gaps)
    ranked_gaps = sorted(gaps, key=lambda item: (-float(item["normalized_gap"]), item["metric"]))
    return {
        "similarity": round(max(0.0, 1.0 - mean_gap), 4),
        "shared_metric_count": len(shared),
        "largest_gaps": ranked_gaps[:5],
    }


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
                "case_id": _stable_case_id(str(path)),
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
        "case_id": _stable_case_id(str(path)),
        "report_origin": report_origin,
        "resolved_audio_path": str(analyzed_path),
        "render_manifest_path": str(render_manifest_path) if render_manifest_path else None,
        "report": report,
    }


def _build_listen_comparison_from_resolved(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
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

    left_label = left.get("display_label") or left.get("input_label") or "left"
    right_label = right.get("display_label") or right.get("input_label") or "right"
    ranked_component_swings = [
        {
            "component": key,
            "label": _metric_label(key),
            "delta": round(abs(float(delta)), 1),
            "winner": ("left" if delta > 0 else "right" if delta < 0 else "tie"),
        }
        for key, delta in sorted(component_deltas.items(), key=lambda item: abs(item[1]), reverse=True)
    ]
    comparison = {
        "schema_version": "0.2.0",
        "comparison_id": f"{left.get('case_id')}__vs__{right.get('case_id')}",
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
        "diagnostics": {
            "ranked_component_swings": ranked_component_swings,
            "groove_profile_match": _component_profile_match(
                left_report.get("groove") or {},
                right_report.get("groove") or {},
            ),
            "energy_profile_match": _component_profile_match(
                left_report.get("energy_arc") or {},
                right_report.get("energy_arc") or {},
            ),
            "left_profile": {
                "case_id": left.get("case_id"),
                "display_label": left_label,
                "strengths": _report_strengths_and_weaknesses(left_report)[0],
                "weaknesses": _report_strengths_and_weaknesses(left_report)[1],
                "components": _report_component_snapshot(left_report),
            },
            "right_profile": {
                "case_id": right.get("case_id"),
                "display_label": right_label,
                "strengths": _report_strengths_and_weaknesses(right_report)[0],
                "weaknesses": _report_strengths_and_weaknesses(right_report)[1],
                "components": _report_component_snapshot(right_report),
            },
        },
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


def _build_listen_comparison(left_input: str, right_input: str) -> dict[str, Any]:
    left, right = _assign_display_labels([
        _resolve_compare_input(left_input),
        _resolve_compare_input(right_input),
    ])
    return _build_listen_comparison_from_resolved(left, right)


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

    left_label = comparison['left'].get('display_label') or comparison['left'].get('input_label', 'left')
    right_label = comparison['right'].get('display_label') or comparison['right'].get('input_label', 'right')
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


def _build_listen_benchmark(inputs: list[str]) -> dict[str, Any]:
    if len(inputs) < 2:
        raise CliError("listen benchmark requires at least two inputs")

    resolved = _assign_display_labels([_resolve_compare_input(path) for path in inputs])
    scoreboard = {
        str(item.get('display_label') or item['input_label']): {
            'label': str(item.get('display_label') or item['input_label']),
            'case_id': item['case_id'],
            'input_path': item['input_path'],
            'report_origin': item['report_origin'],
            'overall_score': float(item['report']['overall_score']),
            'verdict': item['report'].get('verdict'),
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'net_score_delta': 0.0,
            'pairwise': {},
            'strengths': _report_strengths_and_weaknesses(item['report'])[0],
            'weaknesses': _report_strengths_and_weaknesses(item['report'])[1],
        }
        for item in resolved
    }
    comparisons: list[dict[str, Any]] = []

    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            comparison = _build_listen_comparison_from_resolved(left, right)
            left_label = str(comparison['left'].get('display_label') or comparison['left']['input_label'])
            right_label = str(comparison['right'].get('display_label') or comparison['right']['input_label'])
            delta = float(comparison['deltas']['overall_score_delta'])
            winner = comparison['winner']['overall']

            scoreboard[left_label]['net_score_delta'] = round(float(scoreboard[left_label]['net_score_delta']) + delta, 1)
            scoreboard[right_label]['net_score_delta'] = round(float(scoreboard[right_label]['net_score_delta']) - delta, 1)
            scoreboard[left_label]['pairwise'][right_label] = {
                'winner': winner,
                'overall_score_delta': delta,
            }
            mirrored_winner = 'right' if winner == 'left' else 'left' if winner == 'right' else 'tie'
            scoreboard[right_label]['pairwise'][left_label] = {
                'winner': mirrored_winner,
                'overall_score_delta': round(-delta, 1),
            }

            if winner == 'left':
                scoreboard[left_label]['wins'] += 1
                scoreboard[right_label]['losses'] += 1
            elif winner == 'right':
                scoreboard[right_label]['wins'] += 1
                scoreboard[left_label]['losses'] += 1
            else:
                scoreboard[left_label]['ties'] += 1
                scoreboard[right_label]['ties'] += 1

            deciding_components = list((comparison.get('decision') or {}).get('deciding_components') or [])
            biggest_swing = deciding_components[0] if deciding_components else None
            comparisons.append(
                {
                    'comparison_id': comparison.get('comparison_id'),
                    'left': left_label,
                    'right': right_label,
                    'left_case_id': left.get('case_id'),
                    'right_case_id': right.get('case_id'),
                    'winner': winner,
                    'overall_score_delta': delta,
                    'decision': comparison.get('decision', {}),
                    'diagnostics': {
                        'biggest_swing': biggest_swing,
                        'winner_reasons': list((comparison.get('decision') or {}).get('winner_reasons') or []),
                        'loser_fixes': list((comparison.get('decision') or {}).get('loser_fixes') or []),
                        'ranked_component_swings': list((comparison.get('diagnostics') or {}).get('ranked_component_swings') or [])[:3],
                    },
                }
            )

    ranking = sorted(
        scoreboard.values(),
        key=lambda row: (-int(row['wins']), -float(row['net_score_delta']), -float(row['overall_score']), row['label']),
    )

    return {
        'schema_version': '0.2.0',
        'inputs': [item['input_path'] for item in resolved],
        'case_index': [
            {
                'label': str(item.get('display_label') or item['input_label']),
                'input_label': item['input_label'],
                'case_id': item['case_id'],
                'input_path': item['input_path'],
                'report_origin': item['report_origin'],
            }
            for item in resolved
        ],
        'ranking': ranking,
        'comparisons': comparisons,
        'winner': ranking[0]['label'] if ranking else None,
    }


def benchmark_listen(inputs: list[str], output: Optional[str]) -> int:
    benchmark = _build_listen_benchmark(inputs)
    resolved_output = _resolve_output_path(output, default_filename='listen_benchmark.json')
    if resolved_output:
        _write_json(resolved_output, benchmark)
        print(f"Wrote listen benchmark: {resolved_output}")

    if output is None:
        print(json.dumps(benchmark, indent=2, sort_keys=True))

    print(f"Benchmark winner: {benchmark['winner']}")
    for row in benchmark['ranking']:
        print(
            f"- {row['label']}: wins={row['wins']} ties={row['ties']} losses={row['losses']} "
            f"net_delta={row['net_score_delta']:+.1f} overall={row['overall_score']:.1f}"
        )
    return 0



def _stable_listener_agent_output_path(inputs: list[str]) -> Path:
    resolved = sorted(str(Path(item).expanduser().resolve()) for item in inputs)
    digest = hashlib.sha1("||".join(resolved).encode("utf-8")).hexdigest()[:10]
    return (Path("runs") / "listener_agent" / f"listener_agent__{digest}.json").resolve()


def _stable_closed_loop_output_root(song_a: str, song_b: str, references: list[str]) -> Path:
    resolved = [
        str(Path(song_a).expanduser().resolve()),
        str(Path(song_b).expanduser().resolve()),
        *sorted(str(Path(item).expanduser().resolve()) for item in references),
    ]
    digest = hashlib.sha1("||".join(resolved).encode("utf-8")).hexdigest()[:10]
    song_a_slug = ''.join(ch if ch.isalnum() else '_' for ch in Path(song_a).expanduser().stem.lower()).strip('_') or 'song_a'
    song_b_slug = ''.join(ch if ch.isalnum() else '_' for ch in Path(song_b).expanduser().stem.lower()).strip('_') or 'song_b'
    return (Path("runs") / "closed_loop" / f"{song_a_slug}__{song_b_slug}__{digest}").resolve()


def _listener_component_score(report: dict[str, Any], key: str) -> float:
    if key == "overall_score":
        return float(report.get("overall_score") or 0.0)
    return float((report.get(key) or {}).get("score") or 0.0)


def _listener_song_likeness_metrics(report: dict[str, Any]) -> dict[str, Optional[float]]:
    metrics = ((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {}

    def _metric(name: str) -> Optional[float]:
        value = metrics.get(name)
        if value is None:
            return None
        return float(value)

    return {
        "backbone_continuity": _metric("backbone_continuity"),
        "recognizable_section_ratio": _metric("recognizable_section_ratio"),
        "boundary_recovery": _metric("boundary_recovery"),
        "role_plausibility": _metric("role_plausibility"),
        "planner_audio_climax_conviction": _metric("planner_audio_climax_conviction"),
        "composite_song_risk": _metric("composite_song_risk"),
        "background_only_identity_gap": _metric("background_only_identity_gap"),
        "owner_switch_ratio": _metric("owner_switch_ratio"),
    }


def _append_unique_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _listener_rank_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    weighted_components = {
        key: round(LISTENER_AGENT_COMPONENT_WEIGHTS[key] * _listener_component_score(report, key), 2)
        for key in LISTENER_AGENT_COMPONENT_WEIGHTS
    }
    weighted_rank = sum(weighted_components.values())

    bottlenecks: list[dict[str, Any]] = []
    imbalance_penalty = 0.0
    for key, target in LISTENER_AGENT_CRITICAL_RANK_TARGETS.items():
        score = _listener_component_score(report, key)
        gap = max(target - score, 0.0)
        penalty = round(gap * LISTENER_AGENT_RANK_PENALTY_SCALE, 2)
        if penalty > 0.0:
            bottlenecks.append(
                {
                    "component": key,
                    "score": round(score, 1),
                    "target": round(target, 1),
                    "gap": round(gap, 1),
                    "penalty": penalty,
                }
            )
            imbalance_penalty += penalty

    weakest_critical = min(
        (
            {
                "component": key,
                "score": round(_listener_component_score(report, key), 1),
            }
            for key in LISTENER_AGENT_CRITICAL_RANK_TARGETS
        ),
        key=lambda item: (item["score"], item["component"]),
    )

    return {
        "weighted_rank": round(weighted_rank, 1),
        "imbalance_penalty": round(imbalance_penalty, 1),
        "critical_floor": weakest_critical["score"],
        "weakest_critical_component": weakest_critical["component"],
        "bottlenecks": sorted(bottlenecks, key=lambda item: (-item["penalty"], item["component"])),
        "final_rank": round(max(weighted_rank - imbalance_penalty, 0.0), 1),
        "weighted_components": weighted_components,
    }


def _listener_acceptance_checks(report: dict[str, Any]) -> dict[str, Any]:
    component_floors = {}
    for key, minimum in LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS.items():
        actual = _listener_component_score(report, key)
        component_floors[key] = {
            "minimum": round(minimum, 1),
            "actual": round(actual, 1),
            "passed": actual >= minimum,
        }

    survivor_minimums = {}
    for key, minimum in LISTENER_AGENT_SURVIVOR_MINIMUMS.items():
        actual = _listener_component_score(report, key)
        survivor_minimums[key] = {
            "minimum": round(minimum, 1),
            "actual": round(actual, 1),
            "passed": actual >= minimum,
        }

    verdict = str(report.get("verdict") or "unknown")
    gate_status = str((report.get("gating") or {}).get("status") or "unknown")
    return {
        "hard_reject_component_floors": component_floors,
        "survivor_minimums": survivor_minimums,
        "verdict_gate": {
            "allowed_values": ["promising"],
            "actual": verdict,
            "passed": verdict not in {"mixed", "weak", "poor"},
        },
        "listen_gate": {
            "required_status": "pass",
            "actual": gate_status,
            "passed": gate_status == "pass",
        },
    }


def _listener_agent_case_assessment(item: dict[str, Any]) -> dict[str, Any]:
    report = item["report"]
    gate_status = str((report.get("gating") or {}).get("status") or "unknown")
    overall = _listener_component_score(report, "overall_score")
    song_likeness = _listener_component_score(report, "song_likeness")
    groove = _listener_component_score(report, "groove")
    energy_arc = _listener_component_score(report, "energy_arc")
    transition = _listener_component_score(report, "transition")
    coherence = _listener_component_score(report, "coherence")
    mix_sanity = _listener_component_score(report, "mix_sanity")
    verdict = str(report.get("verdict") or "unknown")
    song_likeness_metrics = _listener_song_likeness_metrics(report)

    hard_fail_reasons: list[str] = []
    if gate_status == "reject":
        _append_unique_reason(hard_fail_reasons, "hard listen gate rejected the track")
    if verdict in {"weak", "poor"}:
        _append_unique_reason(hard_fail_reasons, "listener verdict says the output is weak or poor")
    if overall < LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS["overall_score"]:
        _append_unique_reason(hard_fail_reasons, "overall musical quality is still too low")
    if song_likeness < LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS["song_likeness"]:
        _append_unique_reason(hard_fail_reasons, "does not sound like one real song")
    if groove < LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS["groove"]:
        _append_unique_reason(hard_fail_reasons, "groove is too unstable to trust")
    if energy_arc < LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS["energy_arc"]:
        _append_unique_reason(hard_fail_reasons, "section arc / payoff shape is too weak")
    if transition < LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS["transition"]:
        _append_unique_reason(hard_fail_reasons, "transitions still read like track switching")

    backbone_continuity = song_likeness_metrics["backbone_continuity"]
    recognizable_section_ratio = song_likeness_metrics["recognizable_section_ratio"]
    boundary_recovery = song_likeness_metrics["boundary_recovery"]
    role_plausibility = song_likeness_metrics["role_plausibility"]
    composite_song_risk = song_likeness_metrics["composite_song_risk"]
    background_only_identity_gap = song_likeness_metrics["background_only_identity_gap"]
    owner_switch_ratio = song_likeness_metrics["owner_switch_ratio"]

    if song_likeness >= 45.0 and backbone_continuity is not None and backbone_continuity < 0.42:
        _append_unique_reason(hard_fail_reasons, "whole-song backbone continuity is too weak")
    if (
        song_likeness >= 45.0
        and recognizable_section_ratio is not None
        and boundary_recovery is not None
        and recognizable_section_ratio < 0.40
        and boundary_recovery < 0.38
    ):
        _append_unique_reason(hard_fail_reasons, "section readability is too weak to trust as one arrangement")
    if (
        song_likeness >= 45.0
        and role_plausibility is not None
        and boundary_recovery is not None
        and role_plausibility < 0.40
        and boundary_recovery < 0.42
    ):
        _append_unique_reason(hard_fail_reasons, "section roles are not plausible enough to read as one song")
    if composite_song_risk is not None and composite_song_risk > 0.50:
        _append_unique_reason(hard_fail_reasons, "composite detector says the arrangement still reads like multiple pasted songs")
    if background_only_identity_gap is not None and background_only_identity_gap > 0.45:
        _append_unique_reason(hard_fail_reasons, "fusion identity is mostly background-only glue")
    if transition < 55.0 and owner_switch_ratio is not None and owner_switch_ratio > 0.78:
        _append_unique_reason(hard_fail_reasons, "too many section-owner flips still read like track switching")

    acceptance_checks = _listener_acceptance_checks(report)

    if hard_fail_reasons:
        decision = "reject"
    elif (
        verdict in {"mixed"}
        or overall < LISTENER_AGENT_SURVIVOR_MINIMUMS["overall_score"]
        or song_likeness < LISTENER_AGENT_SURVIVOR_MINIMUMS["song_likeness"]
        or groove < LISTENER_AGENT_SURVIVOR_MINIMUMS["groove"]
        or energy_arc < LISTENER_AGENT_SURVIVOR_MINIMUMS["energy_arc"]
    ):
        decision = "borderline"
    else:
        decision = "survivor"

    rank_diagnostics = _listener_rank_diagnostics(report)
    listener_rank = rank_diagnostics["final_rank"]

    strengths, weaknesses = _report_strengths_and_weaknesses(report, limit=3)
    return {
        "label": item.get("input_label"),
        "case_id": item.get("case_id"),
        "input_path": item.get("input_path"),
        "report_origin": item.get("report_origin"),
        "resolved_audio_path": item.get("resolved_audio_path"),
        "render_manifest_path": item.get("render_manifest_path"),
        "overall_score": round(overall, 1),
        "verdict": verdict,
        "gate_status": gate_status,
        "decision": decision,
        "listener_rank": listener_rank,
        "acceptance_checks": acceptance_checks,
        "hard_fail_reasons": hard_fail_reasons,
        "top_reasons": list((report.get("top_reasons") or [])[:4]),
        "top_fixes": list((report.get("top_fixes") or [])[:4]),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "rank_diagnostics": rank_diagnostics,
        "hard_reject_signals": {
            key: round(value, 3)
            for key, value in song_likeness_metrics.items()
            if value is not None
        },
        "component_scores": {
            key: round(_listener_component_score(report, key), 1)
            for key in LISTENER_AGENT_COMPONENT_WEIGHTS
            if key != "overall_score"
        },
    }


def _build_listener_agent_report(inputs: list[str], shortlist: int = 3) -> dict[str, Any]:
    if not inputs:
        raise CliError("listener-agent requires at least one input")

    resolved = [_resolve_compare_input(path) for path in inputs]
    assessments = [_listener_agent_case_assessment(item) for item in resolved]
    label_counts: dict[str, int] = {}
    for row in assessments:
        label = str(row.get("label") or "")
        label_counts[label] = label_counts.get(label, 0) + 1
    for row in assessments:
        label = str(row.get("label") or "")
        if label_counts.get(label, 0) > 1:
            input_path = Path(str(row.get("input_path") or "")).expanduser()
            parent = input_path.parent.name or "case"
            row["label"] = f"{parent}/{input_path.name}"
    assessments_sorted = sorted(
        assessments,
        key=lambda row: (
            {"survivor": 0, "borderline": 1, "reject": 2}.get(row["decision"], 3),
            -float(row["listener_rank"]),
            row["label"] or "",
        ),
    )
    survivors = [row for row in assessments_sorted if row["decision"] == "survivor"]
    borderlines = [row for row in assessments_sorted if row["decision"] == "borderline"]
    rejected = [row for row in assessments_sorted if row["decision"] == "reject"]
    recommended = survivors[: max(shortlist, 0)]

    summary: list[str] = []
    if recommended:
        summary.append(
            f"Listener agent kept {len(recommended)} of {len(assessments_sorted)} candidates for human review."
        )
    else:
        summary.append("Listener agent found no outputs good enough for human review.")
    if rejected:
        summary.append(
            "Hard rejects: "
            + "; ".join(
                f"{row['label']} ({'; '.join(row['hard_fail_reasons'][:2])})"
                for row in rejected[:3]
            )
            + "."
        )
    if borderlines and not recommended:
        summary.append(
            "Borderline-only pool: "
            + "; ".join(f"{row['label']} ({row['overall_score']:.1f})" for row in borderlines[:3])
            + "."
        )

    survivor_inputs = [row["input_path"] for row in survivors]
    survivor_benchmark = _build_listen_benchmark(survivor_inputs) if len(survivor_inputs) >= 2 else None

    return {
        "schema_version": "0.1.0",
        "listener_agent": {
            "purpose": "Reject non-song outputs and only recommend promising survivors for human listening.",
            "ranking_policy": {
                "weighted_components": dict(LISTENER_AGENT_COMPONENT_WEIGHTS),
                "critical_rank_targets": dict(LISTENER_AGENT_CRITICAL_RANK_TARGETS),
                "imbalance_penalty_scale": LISTENER_AGENT_RANK_PENALTY_SCALE,
            },
            "acceptance_criteria": {
                "hard_reject_component_floors": dict(LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS),
                "survivor_minimums": dict(LISTENER_AGENT_SURVIVOR_MINIMUMS),
                "required_listener_verdict_for_survivor": ["promising"],
                "required_listen_gate_status_for_survivor": "pass",
            },
            "hard_reject_rules": [
                "hard listen gate rejected the track",
                "does not sound like one real song",
                "whole-song backbone continuity is too weak",
                "section readability is too weak to trust as one arrangement",
                "section roles are not plausible enough to read as one song",
                "fusion identity is mostly background-only glue",
                "groove is too unstable to trust",
                "section arc / payoff shape is too weak",
                "transitions still read like track switching",
                "too many section-owner flips still read like track switching",
            ],
        },
        "inputs": inputs,
        "summary": summary,
        "recommended_for_human_review": recommended,
        "survivors": survivors,
        "borderline": borderlines,
        "rejected": rejected,
        "counts": {
            "total": len(assessments_sorted),
            "survivors": len(survivors),
            "borderline": len(borderlines),
            "rejected": len(rejected),
        },
        "survivor_benchmark": survivor_benchmark,
    }


def listener_agent(inputs: list[str], output: Optional[str], shortlist: int = 3) -> int:
    report = _build_listener_agent_report(inputs, shortlist=shortlist)
    resolved_output = _resolve_output_path(
        output,
        default_path=_stable_listener_agent_output_path(inputs),
        default_filename="listener_agent.json",
    )
    if resolved_output:
        _write_json(resolved_output, report)
        print(f"Wrote listener-agent report: {resolved_output}")

    if output is None:
        print(json.dumps(report, indent=2, sort_keys=True))

    print(report["summary"][0])
    for row in report["recommended_for_human_review"]:
        print(
            f"- KEEP {row['label']}: rank={row['listener_rank']:.1f} overall={row['overall_score']:.1f} verdict={row['verdict']}"
        )
    for row in report["rejected"][:5]:
        reason = "; ".join(row["hard_fail_reasons"][:2]) or "listener rejected the output"
        print(f"- REJECT {row['label']}: {reason}")
    return 0


def closed_loop(
    song_a: str,
    song_b: str,
    references: list[str],
    output_root: Optional[str],
    max_iterations: int = 3,
    quality_gate: float = 85.0,
    plateau_limit: int = 2,
    min_improvement: float = 0.5,
    change_command: Optional[str] = None,
    test_command: Optional[str] = None,
    target_score: float = 99.0,
) -> int:
    song_a_path = _resolve_existing_audio_path(song_a, "song_a")
    song_b_path = _resolve_existing_audio_path(song_b, "song_b")
    if not references:
        raise CliError("closed-loop requires at least one reference input")

    try:
        from scripts.closed_loop_listener_runner import LoopError, run_closed_loop
    except ModuleNotFoundError as exc:
        raise CliError(f"Unable to import closed-loop runner: {exc}") from exc

    resolved_output_root = _resolve_output_path(
        output_root,
        default_path=_stable_closed_loop_output_root(song_a_path, song_b_path, references),
        default_filename="closed_loop_report.json",
    )
    assert resolved_output_root is not None
    output_root_dir = resolved_output_root.parent if resolved_output_root.suffix.lower() == ".json" else resolved_output_root

    try:
        report = run_closed_loop(
            song_a=song_a_path,
            song_b=song_b_path,
            references=references,
            output_root=str(output_root_dir),
            max_iterations=max_iterations,
            quality_gate=quality_gate,
            plateau_limit=plateau_limit,
            min_improvement=min_improvement,
            change_command=change_command,
            test_command=test_command,
            target_score=target_score,
        )
    except LoopError as exc:
        raise CliError(str(exc)) from exc

    report_path = output_root_dir / "closed_loop_report.json"
    print(f"Wrote closed-loop report: {report_path}")
    if report.get("best_iteration"):
        best = report["best_iteration"]
        print(
            f"Best iteration {best['iteration']} scored {float(best['candidate_overall_score']):.1f} ({best['candidate_verdict']})."
        )
    if report.get("stop_reason"):
        print(f"Closed loop stopped: {report['stop_reason']}")
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
  python3 ai_dj.py benchmark-listen runs/fusion_a runs/fusion_b runs/fusion_c --output runs/checkpoint/listen_benchmark.json
  python3 ai_dj.py listener-agent runs/fusion_a runs/fusion_b runs/fusion_c --output runs/checkpoint/listener_agent.json
  python3 ai_dj.py closed-loop song_a.wav song_b.wav ref_a.wav ref_b.wav --output runs/closed_loop/demo --max-iterations 2
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

    benchmark_parser = subparsers.add_parser("benchmark-listen", help="Round-robin benchmark multiple listen reports, audio files, or rendered outputs")
    benchmark_parser.add_argument("inputs", nargs="+", help="Two or more inputs: listen JSON, audio file, render manifest JSON, or render output directory")
    benchmark_parser.add_argument("--output", "-o", help="Path to output benchmark JSON")

    listener_agent_parser = subparsers.add_parser("listener-agent", help="Gate multiple outputs so only promising survivors reach human listening")
    listener_agent_parser.add_argument("inputs", nargs="+", help="One or more inputs: listen JSON, audio file, render manifest JSON, or render output directory")
    listener_agent_parser.add_argument("--shortlist", type=int, default=3, help="Maximum number of survivors to recommend for human review")
    listener_agent_parser.add_argument("--output", "-o", help="Path to output listener-agent JSON")

    closed_loop_parser = subparsers.add_parser("closed-loop", help="Run a bounded listener-driven improvement loop for one fusion pair")
    closed_loop_parser.add_argument("song_a", help="Path to parent song A")
    closed_loop_parser.add_argument("song_b", help="Path to parent song B")
    closed_loop_parser.add_argument("references", nargs="+", help="One or more good reference inputs")
    closed_loop_parser.add_argument("--output", "-o", help="Directory (or JSON path inside a directory) for closed-loop artifacts/report")
    closed_loop_parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of loop iterations")
    closed_loop_parser.add_argument("--quality-gate", type=float, default=85.0, help="Stop once the candidate clears this overall score")
    closed_loop_parser.add_argument("--plateau-limit", type=int, default=2, help="Stop after this many non-improving iterations")
    closed_loop_parser.add_argument("--min-improvement", type=float, default=0.5, help="Minimum score gain required to reset plateau detection")
    closed_loop_parser.add_argument("--target-score", type=float, default=99.0, help="Long-term aspirational target score for the feedback brief")
    closed_loop_parser.add_argument("--change-command", help="Optional direct command template used to change code between iterations")
    closed_loop_parser.add_argument("--test-command", help="Optional direct command template used to validate changes between iterations")

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
        if args.command == "benchmark-listen":
            return benchmark_listen(args.inputs, args.output)
        if args.command == "listener-agent":
            return listener_agent(args.inputs, args.output, args.shortlist)
        if args.command == "closed-loop":
            return closed_loop(
                args.song_a,
                args.song_b,
                args.references,
                args.output,
                args.max_iterations,
                args.quality_gate,
                args.plateau_limit,
                args.min_improvement,
                args.change_command,
                args.test_command,
                args.target_score,
            )
        if args.command == "doctor":
            return doctor(args.output)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
