#!/usr/bin/env python3
"""Build a listener-driven improvement brief for a fusion candidate vs good references."""

from __future__ import annotations

import argparse
import json
import math
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai_dj
from scripts.reference_input_normalizer import normalize_reference_inputs


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


PLANNER_FEEDBACK_RULES: list[dict[str, Any]] = [
    {
        "id": "backbone_continuity",
        "component": "song_likeness",
        "problem": "The child arrangement is losing its backbone lane and reading like stitched chunks instead of one song.",
        "code_targets": ["src/core/planner/arrangement.py", "src/core/evaluation/listen.py"],
        "actions": [
            "Strengthen backbone continuity guards so structural sections stay on one readable parent lane.",
            "Penalize donor carryover and chronology rewinds that break the child-song narrative.",
            "Expose backbone continuity diagnostics directly in the closed-loop brief so fixes are not inferred from prose alone.",
        ],
        "keywords": [
            "backbone continuity",
            "one song",
            "stitched",
            "donor carryover",
            "readable backbone",
            "track switching",
        ],
        "metric_path": ["song_likeness", "details", "aggregate_metrics", "backbone_continuity"],
        "metric_max": 0.58,
    },
    {
        "id": "late_payoff_mapping",
        "component": "energy_arc",
        "problem": "The planner is spending payoff material too early or failing to earn a strong late climax.",
        "code_targets": ["src/core/planner/arrangement.py", "src/core/analysis/energy.py"],
        "actions": [
            "Bias payoff selection toward late sustained plateau windows instead of early spikes.",
            "Downgrade extended forms when reset and relaunch evidence is weak.",
            "Keep explicit diagnostics for hook-spend and second-payoff conviction so reranking is inspectable.",
        ],
        "keywords": [
            "hook too early",
            "spend its hook too early",
            "late payoff",
            "front-loaded",
            "peaking too early",
            "build a real payoff",
        ],
    },
    {
        "id": "section_readability",
        "component": "structure",
        "problem": "The section map is too coarse, chopped, or generic to support confident planner choices.",
        "code_targets": ["src/core/planner/arrangement.py", "src/core/evaluation/listen.py"],
        "actions": [
            "Route low-confidence section maps into simpler child programs instead of forcing long-form structure.",
            "Strengthen role scoring for intro, verse, build, payoff, bridge, and outro against phrase/bar evidence.",
            "Preserve section-readability diagnostics so the loop can tell whether the failure is sparse coverage or over-chopped segmentation.",
        ],
        "keywords": [
            "section coverage",
            "dominant mega-section",
            "too chopped",
            "coarse whole-song windows",
            "structural certainty",
            "readable section",
        ],
    },
    {
        "id": "ownership_switching",
        "component": "transition",
        "problem": "Parent ownership is flipping often enough to read like track switching rather than intentional arrangement control.",
        "code_targets": ["src/core/planner/arrangement.py", "src/core/render", "src/core/evaluation/listen.py"],
        "actions": [
            "Keep a steadier backbone parent through adjacent sections and reserve swaps for major structural turns.",
            "Prefer handoffs with lower seam risk instead of alternating-parent ping-pong sequences.",
            "Expose explicit switch-risk diagnostics so planner reranking can target ownership stability instead of generic transition penalties.",
        ],
        "keywords": [
            "track switching",
            "switch detector",
            "ownership is flipping",
            "obvious switches",
            "owner switch",
            "alternating",
        ],
        "metric_path": ["transition", "details", "aggregate_metrics", "manifest_switch_detector_risk"],
        "metric_min": 0.45,
    },
    {
        "id": "low_end_ownership",
        "component": "mix_sanity",
        "problem": "Low-end or foreground ownership is unresolved across seams, creating clutter and weak arrivals.",
        "code_targets": ["src/core/planner/arrangement.py", "src/core/render", "src/core/evaluation/listen.py"],
        "actions": [
            "Keep one low-end owner across seam windows unless the structural swap is unavoidable.",
            "Reduce overlap density and singularize foreground ownership through swaps and payoffs.",
            "Feed manifest ownership diagnostics back into planner section selection so unstable seams are avoided earlier.",
        ],
        "keywords": [
            "low-end ownership",
            "lead-vocal ownership",
            "competing lead",
            "overlap",
            "crowding",
            "foreground owner",
        ],
        "metric_path": ["mix_sanity", "details", "manifest_metrics", "aggregate_metrics", "low_end_owner_stability_risk"],
        "metric_min": 0.35,
    },
    {
        "id": "groove_handoff_stability",
        "component": "groove",
        "problem": "Groove support is collapsing across adjacent windows or seam handoffs.",
        "code_targets": ["src/core/planner/arrangement.py", "src/core/render", "src/core/evaluation/listen.py"],
        "actions": [
            "Prefer windows with steadier groove support and fewer adjacent-bar shocks.",
            "Reduce seam policies that interrupt kick, bass, or downbeat continuity.",
            "Expose groove-handoff diagnostics in the closed-loop brief so continuity failures are not hidden behind one aggregate groove score.",
        ],
        "keywords": [
            "stabilize groove",
            "pocket collapses",
            "groove pocket",
            "downbeat alignment",
            "kick/bass continuity",
            "adjacent bars",
        ],
    },
]


RENDER_FEEDBACK_RULES: list[dict[str, Any]] = [
    {
        "id": "seam_handoff_envelope",
        "component": "transition",
        "problem": "Render-layer seam envelopes are causing audible energy, timbre, or density cliffs at section handoffs.",
        "code_targets": ["src/core/render/renderer.py", "src/core/render/transitions.py"],
        "actions": [
            "Retune fade/filter envelopes by transition intent so swaps and lifts do not share one generic handoff shape.",
            "Reduce seam-local spectral shock with render-stage filtering or gain shaping before the overlap collapses.",
            "Expose render-facing seam diagnostics from listener metrics so transition fixes are tied to specific audio handoff behaviors.",
        ],
        "keywords": [
            "smooth energy handoffs",
            "spectral shock",
            "brightness and timbre swaps",
            "rhythmic-density changes",
            "intent mismatch",
            "cliff-like",
        ],
        "metric_path": ["transition", "details", "aggregate_metrics", "avg_edge_cliff_risk"],
        "metric_min": 0.28,
    },
    {
        "id": "overlap_density_control",
        "component": "mix_sanity",
        "problem": "Render overlap policy is letting too much simultaneous full-spectrum material survive through seam windows.",
        "code_targets": ["src/core/render/renderer.py", "src/core/render/resolver.py"],
        "actions": [
            "Shorten or thin overlap entries/exits when listener metrics show sustained crowding rather than impact.",
            "Map overlap-risk feedback into stricter donor-support attenuation and cleanup during dense sections.",
            "Keep explicit render diagnostics for crowding bursts so overlap fixes are not hidden inside one mix score.",
        ],
        "keywords": [
            "reduce full-spectrum overlap",
            "trim overlap entries/exits",
            "crowding stays active",
            "wall-of-sound",
            "too many simultaneous elements",
            "thin overlaps",
        ],
        "metric_path": ["mix_sanity", "details", "ownership_clutter_metrics", "crowding_burst_risk"],
        "metric_min": 0.35,
    },
    {
        "id": "foreground_vocal_singularity",
        "component": "mix_sanity",
        "problem": "Render handoffs are not singularizing lead/foreground focus, so seams keep sounding competitively layered.",
        "code_targets": ["src/core/render/renderer.py", "src/core/render/resolver.py"],
        "actions": [
            "Apply stronger arrival cleanup when foreground ownership changes so the new lead actually owns the seam.",
            "Reduce donor presence earlier in the overlap when listener feedback reports competing lead or spotlight overload.",
            "Preserve a render-facing ownership-control map so vocal/foreground fixes stay inspectable across loop iterations.",
        ],
        "keywords": [
            "competing lead",
            "lead-vocal ownership",
            "foreground owner",
            "listener focus",
            "spotlight overload",
            "vocal competition",
        ],
        "metric_path": ["transition", "details", "aggregate_metrics", "avg_vocal_competition_risk"],
        "metric_min": 0.32,
    },
    {
        "id": "low_end_seam_control",
        "component": "mix_sanity",
        "problem": "Render-stage seam control is not protecting one clear low-end owner through overlaps and arrivals.",
        "code_targets": ["src/core/render/renderer.py", "src/core/render/resolver.py"],
        "actions": [
            "Increase low-end cleanup and donor suppression at seam entry when low-end ownership instability is flagged.",
            "Map low-end ownership risk into tighter overlap caps for vulnerable handoffs before mastering tries to hide the conflict.",
            "Expose low-end seam-control diagnostics so closed-loop fixes can separate arrangement ownership from render leakage.",
        ],
        "keywords": [
            "low-end ownership",
            "one clear low-end owner",
            "kick/sub overlap",
            "stacked kick/bass",
            "low-end owner",
            "kick/sub anchor",
        ],
        "metric_path": ["mix_sanity", "details", "manifest_metrics", "aggregate_metrics", "low_end_owner_stability_risk"],
        "metric_min": 0.45,
    },
]


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


def _reference_weighted_quality_diagnostics(candidate_report: dict[str, Any], reference_items: list[dict[str, Any]]) -> dict[str, Any]:
    reference_component_avgs: dict[str, float] = {}
    weight_basis_total = 0.0
    for component in ai_dj.LISTEN_COMPONENT_KEYS:
        values = [
            float((item.get("report") or {}).get(component, {}).get("score") or 0.0)
            for item in reference_items
        ]
        avg_reference_score = sum(values) / max(len(values), 1)
        reference_component_avgs[component] = avg_reference_score
        weight_basis_total += max(avg_reference_score, 0.0)

    if weight_basis_total <= 0.0:
        weight_basis_total = float(len(ai_dj.LISTEN_COMPONENT_KEYS) or 1)

    candidate_weighted_score = 0.0
    reference_weighted_score = 0.0
    component_breakdown: list[dict[str, Any]] = []

    for component in ai_dj.LISTEN_COMPONENT_KEYS:
        candidate_score = float((candidate_report.get(component) or {}).get("score") or 0.0)
        avg_reference_score = float(reference_component_avgs.get(component) or 0.0)
        weight_basis = max(avg_reference_score, 0.0)
        if weight_basis_total > 0.0 and weight_basis > 0.0:
            weight = weight_basis / weight_basis_total
        else:
            weight = 1.0 / float(len(ai_dj.LISTEN_COMPONENT_KEYS) or 1)
        weighted_candidate = candidate_score * weight
        weighted_reference = avg_reference_score * weight
        weighted_gap = weighted_candidate - weighted_reference
        candidate_weighted_score += weighted_candidate
        reference_weighted_score += weighted_reference
        component_breakdown.append({
            "component": component,
            "weight": round(weight, 4),
            "candidate_score": round(candidate_score, 1),
            "avg_reference_score": round(avg_reference_score, 1),
            "gap_vs_references": round(candidate_score - avg_reference_score, 1),
            "weighted_candidate_contribution": round(weighted_candidate, 2),
            "weighted_reference_contribution": round(weighted_reference, 2),
            "weighted_gap_contribution": round(weighted_gap, 2),
        })

    top_blockers = [item for item in component_breakdown if float(item["gap_vs_references"]) < 0.0]
    top_blockers.sort(key=lambda item: (float(item["weighted_gap_contribution"]), item["component"]))

    return {
        "candidate_weighted_score": round(candidate_weighted_score, 1),
        "reference_weighted_score": round(reference_weighted_score, 1),
        "weighted_gap_vs_references": round(candidate_weighted_score - reference_weighted_score, 1),
        "component_breakdown": component_breakdown,
        "top_blockers": top_blockers[:3],
    }


def _flatten_numeric_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            metrics.update(_flatten_numeric_metrics(value, metric_key))
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[metric_key] = float(value)
    return metrics


def _reference_feature_summary(reference_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not reference_items:
        return []

    minimum_support = max(1, math.ceil(len(reference_items) * 0.6))
    feature_rows: list[dict[str, Any]] = []

    for component in ai_dj.LISTEN_COMPONENT_KEYS:
        component_scores: list[float] = []
        metric_values: dict[str, list[float]] = {}
        for item in reference_items:
            report = item["report"]
            part = report.get(component) or {}
            score = part.get("score")
            if isinstance(score, (int, float)) and math.isfinite(float(score)):
                component_scores.append(float(score))
            details = part.get("details") or {}
            for metric_name, metric_value in _flatten_numeric_metrics(details).items():
                metric_values.setdefault(metric_name, []).append(metric_value)

        if not component_scores and not metric_values:
            continue

        stable_metrics = []
        for metric_name, values in metric_values.items():
            if len(values) < minimum_support:
                continue
            stable_metrics.append({
                "metric": metric_name,
                "support": len(values),
                "avg": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "spread": round(max(values) - min(values), 3),
            })

        stable_metrics.sort(key=lambda item: (item["spread"], -item["support"], item["metric"]))
        feature_rows.append({
            "component": component,
            "reference_score_floor": round(min(component_scores), 1) if component_scores else None,
            "reference_score_avg": round(sum(component_scores) / len(component_scores), 1) if component_scores else None,
            "reference_score_ceiling": round(max(component_scores), 1) if component_scores else None,
            "reference_count": len(component_scores),
            "stable_metrics": stable_metrics[:5],
        })

    feature_rows.sort(key=lambda item: (-(item["reference_score_avg"] or 0.0), item["component"]))
    return feature_rows


def _reference_profile_similarity(comparisons: list[dict[str, Any]], diagnostic_key: str) -> dict[str, Any]:
    similarities: list[float] = []
    strongest_mismatches: list[dict[str, Any]] = []

    for comparison in comparisons:
        diagnostics = comparison.get("diagnostics") or {}
        profile_match = diagnostics.get(diagnostic_key) or {}
        similarity = profile_match.get("similarity")
        if isinstance(similarity, (int, float)) and math.isfinite(float(similarity)):
            similarities.append(float(similarity))
        for mismatch in profile_match.get("largest_gaps") or []:
            if not isinstance(mismatch, dict):
                continue
            strongest_mismatches.append({
                "reference_label": ((comparison.get("right") or {}).get("display_label") or (comparison.get("right") or {}).get("input_label") or "reference"),
                **mismatch,
            })

    strongest_mismatches.sort(key=lambda item: (-float(item.get("normalized_gap") or 0.0), str(item.get("metric") or "")))
    return {
        "avg_similarity": round(sum(similarities) / len(similarities), 4) if similarities else None,
        "comparison_count": len(comparisons),
        "strongest_mismatches": strongest_mismatches[:5],
    }



def _reference_groove_similarity(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    return _reference_profile_similarity(comparisons, "groove_profile_match")



def _reference_dynamic_contour_similarity(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    return _reference_profile_similarity(comparisons, "energy_profile_match")


def _get_nested(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_climax_position(report: dict[str, Any]) -> float | None:
    try:
        raw = (((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {}).get("climax_section_relative_center")
        if raw is None:
            raw = (((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {}).get("climax_position")
        if raw is None:
            raw = (((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {}).get("planner_audio_climax_position")
        if raw is None:
            return None
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return _clamp01(value)


def _climax_reference_alignment(candidate_report: dict[str, Any], reference_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidate_position = _extract_climax_position(candidate_report)
    if candidate_position is None:
        return None

    comparisons: list[dict[str, Any]] = []
    for item in reference_items:
        reference_position = _extract_climax_position(item.get("report") or {})
        if reference_position is None:
            continue
        abs_delta = abs(candidate_position - reference_position)
        comparisons.append(
            {
                "reference_label": item.get("input_label"),
                "reference_case_id": item.get("case_id"),
                "reference_climax_position": round(reference_position, 3),
                "absolute_delta": round(abs_delta, 3),
                "position_similarity": round(_clamp01(1.0 - (abs_delta / 0.35)), 3),
            }
        )

    if not comparisons:
        return None

    mean_abs_delta = sum(float(item["absolute_delta"]) for item in comparisons) / len(comparisons)
    mean_similarity = sum(float(item["position_similarity"]) for item in comparisons) / len(comparisons)
    avg_reference_position = sum(float(item["reference_climax_position"]) for item in comparisons) / len(comparisons)
    return {
        "candidate_climax_position": round(candidate_position, 3),
        "avg_reference_climax_position": round(avg_reference_position, 3),
        "mean_absolute_delta": round(mean_abs_delta, 3),
        "mean_position_similarity": round(mean_similarity, 3),
        "per_reference": comparisons,
    }


def _resolve_manifest_path(item: dict[str, Any]) -> Path | None:
    manifest_path = item.get("render_manifest_path")
    if manifest_path:
        path = Path(str(manifest_path)).expanduser()
        if path.exists():
            return path

    candidate_paths: list[Path] = []
    for key in ("input_path", "resolved_audio_path"):
        raw_value = item.get(key)
        if not raw_value:
            continue
        try:
            base_path = Path(str(raw_value)).expanduser()
        except (TypeError, ValueError):
            continue
        if base_path.is_dir():
            candidate_paths.append(base_path / "render_manifest.json")
            continue
        candidate_paths.append(base_path.with_name("render_manifest.json"))
        candidate_paths.append(base_path.parent / "render_manifest.json")

    seen: set[str] = set()
    for path in candidate_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path
    return None


def _load_manifest_summary(item: dict[str, Any]) -> dict[str, Any] | None:
    path = _resolve_manifest_path(item)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sections = payload.get("sections") or []
    if not isinstance(sections, list) or not sections:
        return None

    labels: list[str] = []
    role_centers: dict[str, list[float]] = {}
    total_bars = 0.0
    running_bar = 0.0
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        label = str(section.get("label") or f"section_{index}")
        try:
            bar_count = max(1.0, float(section.get("bar_count") or 0.0))
        except (TypeError, ValueError):
            bar_count = 1.0
        start_bar_raw = section.get("start_bar")
        try:
            start_bar = float(start_bar_raw) if start_bar_raw is not None else running_bar
        except (TypeError, ValueError):
            start_bar = running_bar
        center = start_bar + (bar_count / 2.0)
        labels.append(label)
        role_centers.setdefault(label, []).append(center)
        total_bars = max(total_bars, start_bar + bar_count)
        running_bar = max(running_bar, start_bar + bar_count)

    if not labels:
        return None

    normalized_centers = {
        label: [round(_clamp01(center / max(total_bars, 1.0)), 3) for center in centers]
        for label, centers in role_centers.items()
    }
    return {
        "program": [str(label) for label in labels],
        "program_signature": " -> ".join(str(label) for label in labels),
        "section_count": len(labels),
        "total_bars": round(total_bars, 3),
        "role_centers": normalized_centers,
    }


def _role_center_similarity(candidate_summary: dict[str, Any], reference_summary: dict[str, Any]) -> float | None:
    candidate_centers = candidate_summary.get("role_centers") or {}
    reference_centers = reference_summary.get("role_centers") or {}
    shared_roles = sorted(set(candidate_centers) & set(reference_centers))
    if not shared_roles:
        return None

    similarities: list[float] = []
    for role in shared_roles:
        candidate_values = list(candidate_centers.get(role) or [])
        reference_values = list(reference_centers.get(role) or [])
        for candidate_value, reference_value in zip(candidate_values, reference_values):
            similarities.append(_clamp01(1.0 - (abs(float(candidate_value) - float(reference_value)) / 0.3)))
    if not similarities:
        return None
    return round(sum(similarities) / len(similarities), 3)


def _section_program_reference_alignment(candidate_item: dict[str, Any], reference_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidate_summary = _load_manifest_summary(candidate_item)
    if candidate_summary is None:
        return None

    comparisons: list[dict[str, Any]] = []
    candidate_program = list(candidate_summary["program"])
    for item in reference_items:
        reference_summary = _load_manifest_summary(item)
        if reference_summary is None:
            continue
        reference_program = list(reference_summary["program"])
        sequence_similarity = round(SequenceMatcher(a=candidate_program, b=reference_program).ratio(), 3)
        length_similarity = round(
            _clamp01(1.0 - (abs(len(candidate_program) - len(reference_program)) / max(len(candidate_program), len(reference_program), 1))),
            3,
        )
        role_center_similarity = _role_center_similarity(candidate_summary, reference_summary)
        weighted_similarity = (
            0.65 * sequence_similarity
            + 0.20 * length_similarity
            + 0.15 * (role_center_similarity if role_center_similarity is not None else sequence_similarity)
        )
        comparisons.append(
            {
                "reference_label": item.get("input_label"),
                "reference_case_id": item.get("case_id"),
                "reference_program_signature": reference_summary["program_signature"],
                "reference_section_count": reference_summary["section_count"],
                "sequence_similarity": sequence_similarity,
                "length_similarity": length_similarity,
                "role_center_similarity": role_center_similarity,
                "program_similarity": round(weighted_similarity, 3),
            }
        )

    if not comparisons:
        return None

    return {
        "candidate_program_signature": candidate_summary["program_signature"],
        "candidate_section_count": candidate_summary["section_count"],
        "avg_reference_section_count": round(sum(float(item["reference_section_count"]) for item in comparisons) / len(comparisons), 3),
        "mean_sequence_similarity": round(sum(float(item["sequence_similarity"]) for item in comparisons) / len(comparisons), 3),
        "mean_program_similarity": round(sum(float(item["program_similarity"]) for item in comparisons) / len(comparisons), 3),
        "mean_role_center_similarity": round(sum(float(item["role_center_similarity"] if item["role_center_similarity"] is not None else item["sequence_similarity"]) for item in comparisons) / len(comparisons), 3),
        "per_reference": comparisons,
    }


def _feedback_texts(candidate_report: dict[str, Any], component: str) -> list[str]:
    part = candidate_report.get(component) or {}
    texts: list[str] = []
    for value in [part.get("summary"), *(part.get("fixes") or []), *(part.get("evidence") or [])]:
        text = str(value or "").strip()
        if text:
            texts.append(text)
    if component in {"transition", "mix_sanity"}:
        details = part.get("details") or {}
        for value in details.get("transition_diagnostics") or []:
            text = str(value or "").strip()
            if text:
                texts.append(text)
    return texts


def _rule_feedback_map(
    candidate_report: dict[str, Any],
    gap_summary: dict[str, float],
    rules: list[dict[str, Any]],
    *,
    target_key: str,
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    seen: set[str] = set()
    top_fix_texts = [str(item).strip() for item in (candidate_report.get("top_fixes") or []) if str(item).strip()]

    for rule in rules:
        component = str(rule["component"])
        texts = _feedback_texts(candidate_report, component) + top_fix_texts
        matched_texts = [text for text in texts if any(keyword in text.lower() for keyword in rule["keywords"])]
        metric_value = None
        metric_triggered = False
        metric_path = rule.get("metric_path")
        if metric_path:
            raw_metric = _get_nested(candidate_report, list(metric_path))
            if raw_metric is not None:
                try:
                    metric_value = float(raw_metric)
                except (TypeError, ValueError):
                    metric_value = None
        if metric_value is not None and rule.get("metric_min") is not None and metric_value >= float(rule["metric_min"]):
            metric_triggered = True
        if metric_value is not None and rule.get("metric_max") is not None and metric_value <= float(rule["metric_max"]):
            metric_triggered = True

        if not matched_texts and not metric_triggered:
            continue

        if rule["id"] in seen:
            continue
        seen.add(rule["id"])

        gap = float(gap_summary.get(component, 0.0))
        evidence = matched_texts[:3]
        if metric_triggered and metric_value is not None:
            evidence.append(f"metric trigger: {'.'.join(metric_path)}={metric_value:.3f}")

        confidence = 0.45
        if matched_texts:
            confidence += min(0.30, 0.10 * len(matched_texts))
        if metric_triggered:
            confidence += 0.20
        if gap < 0.0:
            confidence += min(0.15, abs(gap) / 100.0)

        mapped.append(
            {
                "failure_mode": rule["id"],
                "component": component,
                "problem": rule["problem"],
                "gap_vs_references": round(gap, 1),
                "confidence": round(min(0.99, confidence), 2),
                target_key: list(rule["code_targets"]),
                "actions": list(rule["actions"]),
                "matched_feedback": evidence,
            }
        )

    mapped.sort(key=lambda item: (-float(item["confidence"]), item["component"], item["failure_mode"]))
    return mapped



def _planner_feedback_map(candidate_report: dict[str, Any], gap_summary: dict[str, float]) -> list[dict[str, Any]]:
    return _rule_feedback_map(candidate_report, gap_summary, PLANNER_FEEDBACK_RULES, target_key="planner_code_targets")



def _render_feedback_map(candidate_report: dict[str, Any], gap_summary: dict[str, float]) -> list[dict[str, Any]]:
    return _rule_feedback_map(candidate_report, gap_summary, RENDER_FEEDBACK_RULES, target_key="render_code_targets")



def _prioritized_execution_plan(
    ranked_interventions: list[dict[str, Any]],
    planner_feedback_map: list[dict[str, Any]],
    render_feedback_map: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []

    for intervention in ranked_interventions:
        component = str(intervention.get("component") or "")
        problem = str(intervention.get("problem") or "")
        gap = float(intervention.get("gap_vs_references") or 0.0)
        priority = float(intervention.get("priority") or abs(gap))
        code_targets = [str(item).strip() for item in (intervention.get("code_targets") or []) if str(item).strip()]
        actions = [str(item).strip() for item in (intervention.get("actions") or []) if str(item).strip()]

        matching_planner = [
            item for item in planner_feedback_map
            if str(item.get("component") or "") == component
        ]
        matching_render = [
            item for item in render_feedback_map
            if str(item.get("component") or "") == component
        ]

        planner_modes = [str(item.get("failure_mode") or "") for item in matching_planner if str(item.get("failure_mode") or "").strip()]
        render_modes = [str(item.get("failure_mode") or "") for item in matching_render if str(item.get("failure_mode") or "").strip()]
        planner_targets = sorted({
            str(target).strip()
            for item in matching_planner
            for target in (item.get("planner_code_targets") or [])
            if str(target).strip()
        })
        render_targets = sorted({
            str(target).strip()
            for item in matching_render
            for target in (item.get("render_code_targets") or [])
            if str(target).strip()
        })
        merged_targets = sorted(set(code_targets) | set(planner_targets) | set(render_targets))

        focus_area = "cross_cutting"
        if matching_planner and not matching_render:
            focus_area = "planner"
        elif matching_render and not matching_planner:
            focus_area = "render"
        elif matching_planner and matching_render:
            focus_area = "planner_and_render"

        plan.append(
            {
                "priority_rank": len(plan) + 1,
                "component": component,
                "focus_area": focus_area,
                "gap_vs_references": round(gap, 1),
                "priority": round(priority, 1),
                "problem": problem,
                "recommended_code_targets": merged_targets,
                "primary_actions": actions[:3],
                "planner_failure_modes": planner_modes,
                "render_failure_modes": render_modes,
                "why_now": {
                    "planner_signal_count": len(matching_planner),
                    "render_signal_count": len(matching_render),
                    "planner_targets": planner_targets,
                    "render_targets": render_targets,
                },
            }
        )

    return plan


def build_feedback_brief(candidate: str, references: list[str], target_score: float = 99.0) -> dict[str, Any]:
    normalized_references = normalize_reference_inputs(references)

    candidate_item = _load_report(candidate)
    reference_items = [_load_report(path) for path in normalized_references]
    candidate_report = candidate_item["report"]
    gap_summary, comparisons = _component_gap_summary(candidate, normalized_references)
    ranked_interventions = _rank_interventions(gap_summary)
    planner_feedback_map = _planner_feedback_map(candidate_report, gap_summary)
    render_feedback_map = _render_feedback_map(candidate_report, gap_summary)
    climax_reference_alignment = _climax_reference_alignment(candidate_report, reference_items)
    section_program_alignment = _section_program_reference_alignment(candidate_item, reference_items)
    dynamic_contour_similarity = _reference_dynamic_contour_similarity(comparisons)
    reference_weighted_quality = _reference_weighted_quality_diagnostics(candidate_report, reference_items)

    if climax_reference_alignment and float(climax_reference_alignment.get("mean_position_similarity") or 0.0) < 0.6:
        planner_feedback_map = [
            {
                "failure_mode": "climax_position_alignment",
                "component": "energy_arc",
                "problem": "The candidate's strongest section lands in a materially different part of the song than the references' climax region.",
                "gap_vs_references": round(float(gap_summary.get("energy_arc", 0.0)), 1),
                "confidence": round(min(0.99, 0.55 + (0.6 - float(climax_reference_alignment.get("mean_position_similarity") or 0.0))), 2),
                "planner_code_targets": ["src/core/planner/arrangement.py", "src/core/evaluation/listen.py"],
                "actions": [
                    "Expose late-payoff position targets explicitly during section ranking instead of inferring them from one blended energy score.",
                    "Penalize candidate programs whose strongest payoff lands far from the reference climax region unless the alternative is clearly more convincing.",
                    "Keep climax-position diagnostics in the loop output so reference mismatch is inspectable across iterations.",
                ],
                "matched_feedback": [
                    f"candidate climax position {float(climax_reference_alignment['candidate_climax_position']):.3f}",
                    f"avg reference climax position {float(climax_reference_alignment['avg_reference_climax_position']):.3f}",
                    f"mean position similarity {float(climax_reference_alignment['mean_position_similarity']):.3f}",
                ],
            },
            *planner_feedback_map,
        ]

    if section_program_alignment and float(section_program_alignment.get("mean_program_similarity") or 0.0) < 0.67:
        planner_feedback_map = [
            {
                "failure_mode": "section_program_reference_alignment",
                "component": "structure",
                "problem": "The candidate's section program diverges materially from the reference family, so the child arc likely is not landing a believable macro form.",
                "gap_vs_references": round(float(gap_summary.get("structure", 0.0)), 1),
                "confidence": round(min(0.99, 0.52 + (0.67 - float(section_program_alignment.get("mean_program_similarity") or 0.0))), 2),
                "planner_code_targets": ["src/core/planner/arrangement.py", "src/core/evaluation/listen.py"],
                "actions": [
                    "Expose section-program family diagnostics explicitly during closed-loop comparison instead of inferring macro-form mismatch from one structure score.",
                    "Favor candidate programs whose role order and relative placement stay close to proven reference forms unless local material clearly argues otherwise.",
                    "Keep program-similarity evidence inspectable per reference so planner changes can be validated against actual macro-form movement.",
                ],
                "matched_feedback": [
                    f"candidate program {section_program_alignment['candidate_program_signature']}",
                    f"mean program similarity {float(section_program_alignment['mean_program_similarity']):.3f}",
                    f"mean sequence similarity {float(section_program_alignment['mean_sequence_similarity']):.3f}",
                ],
            },
            *planner_feedback_map,
        ]

    if float(dynamic_contour_similarity.get("avg_similarity") or 1.0) < 0.7:
        planner_feedback_map = [
            {
                "failure_mode": "dynamic_contour_alignment",
                "component": "energy_arc",
                "problem": "The candidate's macro-dynamic contour diverges materially from the reference set's energy-shape profile.",
                "gap_vs_references": round(float(gap_summary.get("energy_arc", 0.0)), 1),
                "confidence": round(min(0.99, 0.55 + (0.7 - float(dynamic_contour_similarity.get("avg_similarity") or 0.0))), 2),
                "planner_code_targets": ["src/core/analysis/energy.py", "src/core/planner/arrangement.py", "src/core/evaluation/listen.py"],
                "actions": [
                    "Emit reusable dynamic-contour metrics from analysis so planner and loop diagnostics share one contract instead of local heuristics.",
                    "Use reference contour similarity during payoff/build reranking to reject candidates with flat or mis-timed macro-dynamics.",
                    "Keep top contour mismatches inspectable in the feedback brief so loop iterations can target the exact energy-shape failure mode.",
                ],
                "matched_feedback": [
                    f"avg dynamic contour similarity {float(dynamic_contour_similarity['avg_similarity']):.3f}",
                    *[
                        f"{item.get('reference_label', 'reference')}: {item.get('metric')} gap {float(item.get('normalized_gap') or 0.0):.3f}"
                        for item in (dynamic_contour_similarity.get("strongest_mismatches") or [])[:3]
                    ],
                ],
            },
            *planner_feedback_map,
        ]

    prioritized_execution_plan = _prioritized_execution_plan(
        ranked_interventions,
        planner_feedback_map,
        render_feedback_map,
    )

    return {
        "schema_version": "0.5.0",
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
        "reference_feature_summary": _reference_feature_summary(reference_items),
        "reference_groove_similarity": _reference_groove_similarity(comparisons),
        "reference_dynamic_contour_similarity": dynamic_contour_similarity,
        "reference_alignment": {
            "climax_position": climax_reference_alignment,
            "section_program": section_program_alignment,
            "dynamic_contour": dynamic_contour_similarity,
        },
        "quality_gate_diagnostics": {
            "reference_weighted": reference_weighted_quality,
        },
        "gap_summary": gap_summary,
        "ranked_interventions": ranked_interventions,
        "planner_feedback_map": planner_feedback_map,
        "render_feedback_map": render_feedback_map,
        "prioritized_execution_plan": prioritized_execution_plan,
        "next_code_targets": sorted({
            target
            for item in ranked_interventions
            for target in item["code_targets"]
        } | {
            target
            for item in planner_feedback_map
            for target in item["planner_code_targets"]
        } | {
            target
            for item in render_feedback_map
            for target in item["render_code_targets"]
        }),
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
