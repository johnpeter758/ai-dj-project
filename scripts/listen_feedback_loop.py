#!/usr/bin/env python3
"""Build a listener-driven improvement brief for a fusion candidate vs good references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai_dj


INTERVENTION_LIBRARY: dict[str, dict[str, Any]] = {
    "song_likeness": {
        "problem": "The render does not yet read like one intentional song.",
        "files": ["src/core/planner/arrangement.py", "src/core/evaluation/listen.py"],
        "actions": [
            "Simplify the section program when the second payoff is not truly earned.",
            "Keep one readable backbone lane instead of stitched strong chunks.",
            "Tighten listener-agent hard rejects for obvious non-song continuity failures.",
        ],
    },
    "groove": {
        "problem": "The groove pocket is unstable or collapses during parent switching.",
        "files": ["src/core/planner/arrangement.py", "src/core/render"],
        "actions": [
            "Prefer source windows with steadier groove support and fewer handoff shocks.",
            "Reduce transitions that interrupt kick/bass continuity or beat-grid feel.",
            "Shorten or remove same-parent overlaps that blur rhythmic focus.",
        ],
    },
    "energy_arc": {
        "problem": "The child song peaks too early or fails to land a convincing late climax.",
        "files": ["src/core/analysis/energy.py", "src/core/planner/arrangement.py"],
        "actions": [
            "Bias payoff selection toward late sustained plateau windows, not early spikes.",
            "Downgrade extended forms when payoff #2 is not clearly stronger than payoff #1.",
            "Penalize early hook-spend and front-loaded climax material.",
        ],
    },
    "transition": {
        "problem": "Transitions still read like track switching instead of arrangement moves.",
        "files": ["src/core/render", "src/core/planner/arrangement.py"],
        "actions": [
            "Reduce abrupt ownership changes unless they are genuine structural turns.",
            "Use cleaner arrival/departure overlap policies around section boundaries.",
            "Prefer source handoffs with lower seam-risk and stronger continuity support.",
        ],
    },
    "coherence": {
        "problem": "The result loses internal consistency across bars and phrases.",
        "files": ["src/core/planner/arrangement.py", "src/core/evaluation/listen.py"],
        "actions": [
            "Penalize chronology rewinds and weak source-lane continuity harder.",
            "Reduce exact-window or near-window reuse that makes the timeline feel random.",
            "Strengthen planner continuity scoring across adjacent sections.",
        ],
    },
    "mix_sanity": {
        "problem": "The blend is crowded, amateur, or ownership-conflicted.",
        "files": ["src/core/render", "src/core/evaluation/listen.py"],
        "actions": [
            "Enforce singular low-end and foreground ownership in dense sections.",
            "Shorten overlap density and remove unnecessary simultaneous full-spectrum layers.",
            "Reject sections with lead-vocal competition or overcompressed wall-of-sound behavior.",
        ],
    },
    "structure": {
        "problem": "The section program is weak or not clearly readable.",
        "files": ["src/core/planner/arrangement.py", "src/core/evaluation/listen.py"],
        "actions": [
            "Strengthen section-identity scoring for intro/verse/build/payoff/bridge/outro.",
            "Use simpler programs when the material cannot support a longer arc.",
            "Reward boundary recovery and readable section contrast in both planning and evaluation.",
        ],
    },
}


def _load_report(input_path: str) -> dict[str, Any]:
    return ai_dj._resolve_compare_input(input_path)


def _component_gap_summary(candidate_path: str, reference_paths: list[str]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    deltas_by_component: dict[str, list[float]] = {key: [] for key in ai_dj.LISTEN_COMPONENT_KEYS}
    overall_deltas: list[float] = []

    for reference in reference_paths:
        comparison = ai_dj._build_listen_comparison(candidate_path, reference)
        comparisons.append(comparison)
        overall_deltas.append(float((comparison.get("deltas") or {}).get("overall_score_delta") or 0.0))
        component_deltas = (comparison.get("deltas") or {}).get("component_score_deltas") or {}
        for key in ai_dj.LISTEN_COMPONENT_KEYS:
            deltas_by_component[key].append(float(component_deltas.get(key) or 0.0))

    summary = {
        "overall_vs_references": round(sum(overall_deltas) / max(len(overall_deltas), 1), 1),
    }
    for key, values in deltas_by_component.items():
        summary[key] = round(sum(values) / max(len(values), 1), 1)
    return summary, comparisons


def _rank_interventions(gap_summary: dict[str, float], *, limit: int = 4) -> list[dict[str, Any]]:
    ranked = []
    for key in ai_dj.LISTEN_COMPONENT_KEYS:
        delta = float(gap_summary.get(key, 0.0))
        if delta >= 0.0:
            continue
        library = INTERVENTION_LIBRARY.get(key)
        if not library:
            continue
        ranked.append(
            {
                "component": key,
                "gap_vs_references": round(delta, 1),
                "priority": round(abs(delta), 1),
                "problem": library["problem"],
                "code_targets": list(library["files"]),
                "actions": list(library["actions"]),
            }
        )
    ranked.sort(key=lambda item: (-float(item["priority"]), item["component"]))
    return ranked[:limit]


def _top_reference_strengths(reference_items: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    scored: dict[str, list[float]] = {key: [] for key in ai_dj.LISTEN_COMPONENT_KEYS}
    for item in reference_items:
        report = item["report"]
        for key in ai_dj.LISTEN_COMPONENT_KEYS:
            value = float(report.get(key, {}).get("score") or 0.0)
            scored[key].append(value)
    ranked = []
    for key, values in scored.items():
        if not values:
            continue
        ranked.append({
            "component": key,
            "avg_reference_score": round(sum(values) / len(values), 1),
            "label": key.replace("_", " "),
        })
    ranked.sort(key=lambda item: (-float(item["avg_reference_score"]), item["component"]))
    return ranked[:limit]


def build_feedback_brief(candidate: str, references: list[str], target_score: float = 99.0) -> dict[str, Any]:
    if not references:
        raise ValueError("at least one reference is required")

    candidate_item = _load_report(candidate)
    reference_items = [_load_report(path) for path in references]
    candidate_report = candidate_item["report"]
    gap_summary, comparisons = _component_gap_summary(candidate, references)
    ranked_interventions = _rank_interventions(gap_summary)

    return {
        "schema_version": "0.1.0",
        "goal": {
            "target_listener_score": float(target_score),
            "current_overall_score": float(candidate_report.get("overall_score") or 0.0),
            "gap_to_target": round(float(target_score) - float(candidate_report.get("overall_score") or 0.0), 1),
        },
        "candidate": {
            "label": candidate_item.get("input_label"),
            "case_id": candidate_item.get("case_id"),
            "input_path": candidate_item.get("input_path"),
            "report_origin": candidate_item.get("report_origin"),
            "resolved_audio_path": candidate_item.get("resolved_audio_path"),
            "overall_score": candidate_report.get("overall_score"),
            "verdict": candidate_report.get("verdict"),
            "top_reasons": list((candidate_report.get("top_reasons") or [])[:5]),
            "top_fixes": list((candidate_report.get("top_fixes") or [])[:5]),
        },
        "references": [
            {
                "label": item.get("input_label"),
                "case_id": item.get("case_id"),
                "input_path": item.get("input_path"),
                "overall_score": item["report"].get("overall_score"),
                "verdict": item["report"].get("verdict"),
            }
            for item in reference_items
        ],
        "reference_strengths": _top_reference_strengths(reference_items),
        "gap_summary": gap_summary,
        "ranked_interventions": ranked_interventions,
        "next_code_targets": sorted({target for item in ranked_interventions for target in item["code_targets"]}),
        "pairwise_comparisons": comparisons,
        "automation_loop": [
            "Compare candidate against references and compute component gaps.",
            "Patch the highest-priority code targets tied to the worst gaps.",
            "Rerender the fusion and rerun listener-agent + reference benchmark.",
            "Accept only changes that materially improve score and survivor status.",
            "Repeat until the candidate clearly beats the configured regression gates.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a reference-driven listener improvement brief for one fusion candidate.")
    parser.add_argument("candidate", help="Candidate input: listen JSON, audio file, render manifest JSON, or render output directory")
    parser.add_argument("references", nargs="+", help="One or more good reference inputs")
    parser.add_argument("--target-score", type=float, default=99.0, help="Desired long-term listener score target")
    parser.add_argument("--output", "-o", help="Optional path to write improvement brief JSON")
    args = parser.parse_args()

    brief = build_feedback_brief(args.candidate, args.references, target_score=args.target_score)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(brief, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote listener feedback brief: {output_path}")
    else:
        print(json.dumps(brief, indent=2, sort_keys=True))

    print(f"Candidate: {brief['candidate']['label']}")
    print(f"Current overall: {brief['goal']['current_overall_score']}")
    print(f"Gap to target: {brief['goal']['gap_to_target']}")
    for item in brief["ranked_interventions"][:3]:
        print(f"- {item['component']}: {item['gap_vs_references']} vs refs -> {item['problem']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
