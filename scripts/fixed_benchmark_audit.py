#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

DEFAULT_CASES = {
    "simple_fuse_145930": {
        "run_dir": RUNS / "simple_fuse_20260312_145930",
        "listen_path": RUNS / "validation_20260315_fixed_benchmark_audit" / "simple_fuse_listen.json",
    },
    "backbone_guard_realpass": {
        "run_dir": RUNS / "validation_20260314_backbone_guard_realpass",
    },
    "shared_extended_gate": {
        "run_dir": RUNS / "validation_20260315_shared_extended_gate",
    },
}

SECTION_KEYS = (
    "source_parent",
    "source_section_label",
    "start_bar",
    "bar_count",
    "transition_in",
    "transition_mode",
    "foreground_owner",
    "background_owner",
    "low_end_owner",
    "owner_mode",
    "vocal_policy",
    "allowed_overlap",
    "overlap_beats_max",
)

LISTEN_KEYS = (
    "overall_score",
    "verdict",
    "coherence",
    "energy_arc",
    "groove",
    "mix_sanity",
    "song_likeness",
    "structure",
    "transition",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest(run_dir: Path) -> Path:
    candidates = [
        run_dir / "render_manifest.json",
        run_dir / "fusion_rerun" / "render_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No render manifest found under {run_dir}")


def _resolve_listen(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "listen.json",
        run_dir / "listen_report.json",
        run_dir / "listen_fusion_rerun.json",
        run_dir / "fusion_rerun_listen.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _section_signature(section: dict[str, Any]) -> dict[str, Any]:
    source = section.get("source") or {}
    target = section.get("target") or {}
    return {
        "label": section.get("label"),
        "source_parent": section.get("source_parent"),
        "source_section_label": source.get("source_section_label"),
        "start_bar": target.get("start_bar"),
        "bar_count": target.get("bar_count"),
        "transition_in": section.get("transition_in"),
        "transition_mode": section.get("transition_mode"),
        "foreground_owner": section.get("foreground_owner"),
        "background_owner": section.get("background_owner"),
        "low_end_owner": section.get("low_end_owner"),
        "owner_mode": section.get("owner_mode"),
        "vocal_policy": section.get("vocal_policy"),
        "allowed_overlap": section.get("allowed_overlap"),
        "overlap_beats_max": section.get("overlap_beats_max"),
    }


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    sections = [_section_signature(section) for section in manifest.get("sections", [])]
    by_label: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        by_label.setdefault(str(section["label"]), []).append(section)
    return {
        "program": [f"{section['label']}({section['bar_count']})" for section in sections],
        "program_labels": [section["label"] for section in sections],
        "intro": by_label.get("intro", [None])[0],
        "first_payoff": by_label.get("payoff", [None])[0],
        "last_payoff": by_label.get("payoff", [None, None])[-1] if by_label.get("payoff") else None,
        "owner_counts": {
            "foreground": _count_field(sections, "foreground_owner"),
            "low_end": _count_field(sections, "low_end_owner"),
            "source_parent": _count_field(sections, "source_parent"),
        },
        "transition_modes": _count_field(sections, "transition_mode"),
        "sections": sections,
    }


def _count_field(sections: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for section in sections:
        value = section.get(key)
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _listen_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    payload: dict[str, Any] = {
        "overall_score": report.get("overall_score"),
        "verdict": report.get("verdict"),
    }
    for key in LISTEN_KEYS[2:]:
        component = report.get(key) or {}
        payload[key] = component.get("score")
    energy = (((report.get("energy_arc") or {}).get("details") or {}).get("aggregate_metrics") or {})
    payload["late_lift"] = energy.get("late_lift")
    payload["payoff_strength"] = energy.get("payoff_strength")
    payload["peak_position"] = energy.get("peak_position")
    payload["top_payoff_windows"] = (((report.get("energy_arc") or {}).get("details") or {}).get("top_payoff_windows") or [])[:3]
    mix = (((report.get("mix_sanity") or {}).get("details") or {}).get("manifest_metrics") or {})
    payload["true_two_parent_major_section_ratio"] = (((mix.get("aggregate_metrics") or {}).get("true_two_parent_major_section_ratio")))
    payload["seam_risk_ratio"] = (((mix.get("aggregate_metrics") or {}).get("seam_risk_ratio")))
    song_like = (((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {})
    payload["owner_switch_ratio"] = song_like.get("owner_switch_ratio")
    payload["climax_conviction"] = song_like.get("climax_conviction")
    return payload


def _compare_section(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    if left is None or right is None:
        return {"left": left, "right": right, "changed_keys": ["missing"]}
    changed = [key for key in SECTION_KEYS if left.get(key) != right.get(key)]
    return {
        "left": left,
        "right": right,
        "changed_keys": changed,
    }


def _compare_lists(left: list[str], right: list[str]) -> dict[str, Any]:
    return {
        "left": left,
        "right": right,
        "changed": left != right,
    }


def _compare_listen(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
    if left is None or right is None:
        return None
    deltas: dict[str, Any] = {}
    for key, left_value in left.items():
        right_value = right.get(key)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            deltas[key] = round(float(left_value) - float(right_value), 3)
        elif left_value != right_value:
            deltas[key] = {"left": left_value, "right": right_value}
    return deltas


def _load_case(label: str, run_dir: Path, listen_path: Path | None = None) -> dict[str, Any]:
    manifest_path = _resolve_manifest(run_dir)
    manifest = _load_json(manifest_path)
    listen_path = listen_path if (listen_path and listen_path.exists()) else _resolve_listen(run_dir)
    listen = _load_json(listen_path) if listen_path else None
    return {
        "label": label,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "listen_path": str(listen_path) if listen_path else None,
        "manifest_summary": _manifest_summary(manifest),
        "listen_summary": _listen_summary(listen),
    }


def _pairwise(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_manifest = left["manifest_summary"]
    right_manifest = right["manifest_summary"]
    return {
        "left": left["label"],
        "right": right["label"],
        "program_change": _compare_lists(left_manifest["program"], right_manifest["program"]),
        "intro_change": _compare_section(left_manifest["intro"], right_manifest["intro"]),
        "first_payoff_change": _compare_section(left_manifest["first_payoff"], right_manifest["first_payoff"]),
        "last_payoff_change": _compare_section(left_manifest["last_payoff"], right_manifest["last_payoff"]),
        "owner_count_change": {
            "left": left_manifest["owner_counts"],
            "right": right_manifest["owner_counts"],
            "changed": left_manifest["owner_counts"] != right_manifest["owner_counts"],
        },
        "transition_mode_change": {
            "left": left_manifest["transition_modes"],
            "right": right_manifest["transition_modes"],
            "changed": left_manifest["transition_modes"] != right_manifest["transition_modes"],
        },
        "listen_delta": _compare_listen(left.get("listen_summary"), right.get("listen_summary")),
    }


def _markdown_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Fixed Benchmark Audit",
        "",
        f"Benchmark cases: {', '.join(case['label'] for case in audit['cases'])}",
        "",
    ]
    for case in audit["cases"]:
        manifest = case["manifest_summary"]
        listen = case.get("listen_summary") or {}
        lines += [
            f"## {case['label']}",
            f"- run_dir: `{case['run_dir']}`",
            f"- program: `{' -> '.join(manifest['program'])}`",
            f"- intro: `{manifest['intro']['source_parent']}:{manifest['intro']['source_section_label']}` start_bar={manifest['intro']['start_bar']} bars={manifest['intro']['bar_count']}",
            f"- first payoff: `{manifest['first_payoff']['source_parent']}:{manifest['first_payoff']['source_section_label']}` start_bar={manifest['first_payoff']['start_bar']} bars={manifest['first_payoff']['bar_count']}",
        ]
        if manifest.get("last_payoff"):
            lines.append(
                f"- last payoff: `{manifest['last_payoff']['source_parent']}:{manifest['last_payoff']['source_section_label']}` start_bar={manifest['last_payoff']['start_bar']} bars={manifest['last_payoff']['bar_count']}"
            )
        if listen:
            lines += [
                f"- listen: overall={listen.get('overall_score')} verdict={listen.get('verdict')} structure={listen.get('structure')} transition={listen.get('transition')} song_likeness={listen.get('song_likeness')} energy_arc={listen.get('energy_arc')}",
                f"- energy details: payoff_strength={listen.get('payoff_strength')} late_lift={listen.get('late_lift')} peak_position={listen.get('peak_position')}",
            ]
        lines.append("")

    lines.append("## Pairwise material changes")
    lines.append("")
    for pair in audit["pairwise"]:
        lines.append(f"### {pair['left']} vs {pair['right']}")
        lines.append(f"- program changed: `{pair['program_change']['changed']}`")
        lines.append(f"- intro changed keys: `{pair['intro_change']['changed_keys']}`")
        lines.append(f"- first payoff changed keys: `{pair['first_payoff_change']['changed_keys']}`")
        lines.append(f"- last payoff changed keys: `{pair['last_payoff_change']['changed_keys']}`")
        lines.append(f"- ownership/program counts changed: `{pair['owner_count_change']['changed']}`")
        lines.append(f"- transition-mode counts changed: `{pair['transition_mode_change']['changed']}`")
        listen_delta = pair.get("listen_delta")
        if listen_delta:
            focus_keys = [
                "overall_score",
                "structure",
                "transition",
                "song_likeness",
                "energy_arc",
                "mix_sanity",
                "payoff_strength",
                "late_lift",
                "true_two_parent_major_section_ratio",
                "owner_switch_ratio",
                "climax_conviction",
            ]
            focus = {key: listen_delta[key] for key in focus_keys if key in listen_delta}
            lines.append(f"- key listen deltas: `{json.dumps(focus, sort_keys=True)}`")
        else:
            lines.append("- key listen deltas: `unavailable`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fixed benchmark runs for material manifest/listen changes.")
    parser.add_argument("--case", action="append", default=[], help="label=path to a run dir or render dir; can be repeated")
    parser.add_argument("--output", help="Path to JSON output")
    parser.add_argument("--markdown", help="Path to markdown output")
    args = parser.parse_args()

    cases = {label: dict(value) for label, value in DEFAULT_CASES.items()}
    for item in args.case:
        if "=" not in item:
            raise SystemExit(f"Invalid --case '{item}'. Expected label=path")
        label, raw_path = item.split("=", 1)
        cases[label] = {"run_dir": Path(raw_path).expanduser().resolve()}

    loaded = [
        _load_case(label, Path(config["run_dir"]).expanduser().resolve(), Path(config["listen_path"]).expanduser().resolve() if config.get("listen_path") else None)
        for label, config in cases.items()
    ]
    pairwise = []
    for index, left in enumerate(loaded):
        for right in loaded[index + 1 :]:
            pairwise.append(_pairwise(left, right))

    audit = {
        "schema_version": "0.1.0",
        "cases": loaded,
        "pairwise": pairwise,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else None
    markdown_path = Path(args.markdown).expanduser().resolve() if args.markdown else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote audit JSON: {output_path}")
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown_report(audit), encoding="utf-8")
        print(f"Wrote audit markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
