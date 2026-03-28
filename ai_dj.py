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
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from src.feedback_learning import build_feedback_learning_summary, write_feedback_learning_summary

LISTEN_COMPONENT_KEYS = ("structure", "groove", "energy_arc", "transition", "coherence", "mix_sanity", "song_likeness")

# VocalFusion permanent design rules (product definition guardrails):
# 1) Keep only musically necessary elements active at any moment.
# 2) Every section should serve a clear purpose (setup/build/tension/release/swap/peak/outro).
VOCALFUSION_PERMANENT_DESIGN_RULES = (
    "Only musically necessary elements should be active at any point in time.",
    "Every section must have a purpose: setup, build, tension, release, swap, peak, or outro.",
)


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

AUTO_SHORTLIST_SCHEMA_VERSION = "0.1.0"
AUTO_SHORTLIST_DEFAULT_BATCH_SIZE = 6
AUTO_SHORTLIST_DEFAULT_SHORTLIST = 2
AUTO_SHORTLIST_MAX_BATCH_SIZE = 12
AUTO_SHORTLIST_SECTION_PRIORITY = {
    "payoff": 0,
    "build": 1,
    "bridge": 2,
    "verse": 3,
    "outro": 4,
    "intro": 5,
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


def _render_fusion_plan_candidate(
    song_a: Any,
    song_b: Any,
    plan: Any,
    outdir: str | Path,
    *,
    variant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolve_plan, render_plan = _get_render_functions()
    outdir_path = Path(outdir).expanduser().resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    plan_payload = plan.to_dict()
    if variant_config is not None:
        plan_payload.setdefault("planning_diagnostics", {})["variant"] = variant_config
    _write_json(outdir_path / "arrangement_plan.json", plan_payload)

    manifest = resolve_plan(plan, song_a, song_b)
    result = render_plan(manifest, outdir_path)
    if variant_config is not None:
        _write_json(outdir_path / "candidate_variant.json", variant_config)
    return {
        "outdir": str(outdir_path),
        "variant_config": variant_config or {"variant_id": "baseline", "strategy": "baseline"},
        "arrangement_plan_path": str(outdir_path / "arrangement_plan.json"),
        "render_manifest_path": str(result.manifest_path),
        "raw_wav_path": str(result.raw_wav_path),
        "master_wav_path": str(result.master_wav_path),
        "master_mp3_path": str(result.master_mp3_path) if result.master_mp3_path else None,
    }



def _opportunity_priority(label: str) -> int:
    return AUTO_SHORTLIST_SECTION_PRIORITY.get(str(label or ""), 99)



def _variant_swap_score(opportunity: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _opportunity_priority(str(opportunity.get("section_label") or "")),
        float(opportunity.get("error_delta", 999.0) or 999.0),
        0 if opportunity.get("alternate_parent") != opportunity.get("selected_parent") else 1,
        int(opportunity.get("selection_rank") or 999),
        int(opportunity.get("section_index") or 0),
    )



def _collect_plan_variant_opportunities(plan: Any) -> list[dict[str, Any]]:
    diagnostics = getattr(plan, "planning_diagnostics", {}) or {}
    selected_sections = list(diagnostics.get("selected_sections") or [])
    backbone_parent = ((diagnostics.get("backbone_plan") or {}).get("backbone_parent") or "A")
    opportunities: list[dict[str, Any]] = []
    for index, section_diag in enumerate(selected_sections):
        selected_parent = str(section_diag.get("selected_parent") or "")
        selected_label = str(section_diag.get("selected_window_label") or "")
        shortlist = list(section_diag.get("candidate_shortlist") or [])
        selected_rank = int(section_diag.get("selection_rank") or 1)
        seen: set[tuple[str, str]] = {(selected_parent, selected_label)} if selected_parent and selected_label else set()
        alternates: list[dict[str, Any]] = []
        cross_parent = section_diag.get("cross_parent_best_alternate")
        if isinstance(cross_parent, dict):
            alternates.append(cross_parent)
        alternates.extend(item for item in shortlist if isinstance(item, dict) and not bool(item.get("selected")))
        for alt in alternates:
            alternate_parent = str(alt.get("parent_id") or "")
            alternate_label = str(alt.get("window_label") or "")
            identity = (alternate_parent, alternate_label)
            if not alternate_parent or not alternate_label or identity in seen:
                continue
            seen.add(identity)
            score_breakdown = dict(alt.get("score_breakdown") or {})
            stretch_ratio = float(score_breakdown.get("stretch_ratio", 1.0) or 1.0)
            stretch_gate = float(score_breakdown.get("stretch_gate", 0.0) or 0.0)
            seam_risk = float(score_breakdown.get("seam_risk", 0.0) or 0.0)
            transition_viability = float(score_breakdown.get("transition_viability", 0.0) or 0.0)
            if float(alt.get("error_delta_vs_selected", 99.0) or 99.0) > 1.15:
                continue
            if stretch_gate > 0.60 or stretch_ratio > 1.18 or seam_risk > 0.82 or transition_viability > 0.88:
                continue
            opportunities.append(
                {
                    "section_index": index,
                    "section_label": section_diag.get("label"),
                    "transition_mode": section_diag.get("transition_mode"),
                    "selected_parent": selected_parent,
                    "selected_window_label": selected_label,
                    "alternate_parent": alternate_parent,
                    "alternate_window_label": alternate_label,
                    "alternate_role": alt.get("role"),
                    "window_origin": alt.get("window_origin"),
                    "window_seconds": alt.get("window_seconds"),
                    "selection_rank": int(alt.get("rank") or selected_rank),
                    "selected_rank": selected_rank,
                    "error_delta": float(alt.get("error_delta_vs_selected", 0.0) or 0.0),
                    "planner_error": float(alt.get("planner_error", 0.0) or 0.0),
                    "score_breakdown": score_breakdown,
                    "backbone_parent": backbone_parent,
                    "kind": "cross_parent_alternate" if alternate_parent != selected_parent else "shortlist_alternate",
                }
            )
    opportunities.sort(key=_variant_swap_score)
    return opportunities



def _build_auto_shortlist_variant_configs(plan: Any, batch_size: int, *, variant_mode: str = "safe") -> list[dict[str, Any]]:
    opportunities = _collect_plan_variant_opportunities(plan)
    max_variants = max(1, int(batch_size or 1))
    configs: list[dict[str, Any]] = [
        {
            "variant_id": "baseline",
            "label": "baseline",
            "strategy": "baseline",
            "variant_mode": variant_mode,
            "swaps": [],
        }
    ]
    if max_variants <= 1:
        return configs[:max_variants]

    def _section_index_of(op: dict[str, Any], *, default: int = -1) -> int:
        raw = op.get("section_index", default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _op_identity(op: dict[str, Any]) -> tuple[int, str, str]:
        return (
            _section_index_of(op, default=-1),
            str(op.get("alternate_parent") or ""),
            str(op.get("alternate_window_label") or ""),
        )

    used_identities: set[tuple[int, str, str]] = set()
    singles: list[dict[str, Any]] = []

    def _has_safe_dual_combo(ops: list[dict[str, Any]]) -> bool:
        best_by_section: dict[int, dict[str, Any]] = {}
        for item in ops:
            sec_idx = _section_index_of(item, default=-1)
            if sec_idx < 0 or sec_idx in best_by_section:
                continue
            best_by_section[sec_idx] = item
        section_ops = [best_by_section[idx] for idx in sorted(best_by_section)]
        for left_idx in range(len(section_ops)):
            for right_idx in range(left_idx + 1, len(section_ops)):
                left = section_ops[left_idx]
                right = section_ops[right_idx]
                combo_error = float(left.get("error_delta", 0.0) or 0.0) + float(right.get("error_delta", 0.0) or 0.0)
                if combo_error <= 1.65:
                    return True
        return False

    core_labels = {"verse", "build", "payoff", "bridge"}

    def _normalize_section_label(value: Any) -> str:
        label = str(value or "").strip().lower().replace("-", "_")
        if " " in label:
            label = label.split()[0]
        if "_" in label:
            label = label.split("_", 1)[0]
        return label

    def _is_core_donor_op(op: dict[str, Any]) -> bool:
        section_label = _normalize_section_label(op.get("section_label"))
        alternate_role = str(op.get("alternate_role") or "").strip().lower()
        alternate_parent = str(op.get("alternate_parent") or "")
        return (
            section_label in core_labels
            and (
                alternate_role == "donor"
                or (alternate_parent and alternate_parent != str(op.get("backbone_parent") or ""))
            )
        )

    def _support_recipe_for_section(
        section_label: str,
        *,
        arrangement_mode_local: str,
        error_delta: float,
        seam_risk: float,
        transition_viability: float,
        stretch_ratio: float,
        selected_parent: str,
        backbone_parent: str,
        source_kind: str,
    ) -> dict[str, Any]:
        normalized_label = _normalize_section_label(section_label)
        base_gain = -9.5
        if normalized_label == "payoff":
            base_gain = -11.0
        elif normalized_label == "build":
            base_gain = -10.5
        elif normalized_label == "bridge":
            base_gain = -10.0

        stretch_pressure = min(1.0, abs(float(stretch_ratio or 1.0) - 1.0) * 3.0)
        transition_error = max(0.0, min(1.0, float(transition_viability or 0.0)))
        risk = max(0.0, min(1.0, max(float(seam_risk or 0.0), transition_error, stretch_pressure)))

        transition_health = 1.0 - transition_error
        crowding_pressure = max(float(seam_risk or 0.0), risk, min(1.0, float(error_delta or 0.0) / 2.0), stretch_pressure)
        viability_penalty = 0.0
        if normalized_label == "payoff":
            viability_penalty += 0.22 * max(0.0, crowding_pressure - 0.50)
        elif normalized_label == "build":
            viability_penalty += 0.12 * max(0.0, crowding_pressure - 0.56)
        if selected_parent and backbone_parent and selected_parent != backbone_parent:
            viability_penalty += 0.08
        calibrated_transition_viability = max(0.0, min(1.0, transition_health - viability_penalty))

        collision_risk = max(
            float(seam_risk or 0.0),
            min(1.0, 1.0 - calibrated_transition_viability),
            min(1.0, 0.55 * risk + 0.35 * float(seam_risk or 0.0)),
        )
        if normalized_label == "payoff" and risk >= 0.58:
            collision_risk = min(1.0, collision_risk + 0.08)

        gain = float(base_gain)
        if arrangement_mode_local == "adaptive":
            gain -= 0.25
        if normalized_label in {"build", "payoff"}:
            gain -= 0.35
        if source_kind in {"support_overlay_counterparent", "support_overlay_fallback"}:
            gain -= 0.20

        if risk >= 0.75:
            gain -= 1.20
        elif risk >= 0.55:
            gain -= 0.70
        elif risk <= 0.30 and error_delta <= 0.35:
            gain += 0.35

        if error_delta >= 1.80:
            gain -= 0.50
        elif error_delta <= 0.20:
            gain += 0.20

        if selected_parent and backbone_parent and selected_parent != backbone_parent:
            gain -= 0.35

        gain = round(max(-14.0, min(-8.0, gain)), 2)

        mode = "filtered_counterlayer"
        if normalized_label in {"build", "verse"} and risk <= 0.55 and error_delta <= 1.20:
            mode = "foreground_counterlayer"
        if risk >= 0.70:
            mode = "filtered_counterlayer"
        if selected_parent and backbone_parent and selected_parent != backbone_parent:
            mode = "filtered_counterlayer"

        policy = {
            "risk": round(float(risk), 3),
            "foreground_collision_risk": round(float(collision_risk), 3),
            "transition_viability": round(float(calibrated_transition_viability), 3),
            "transition_error": round(float(transition_error), 3),
            "base_gain_db": float(base_gain),
            "arrangement_mode": arrangement_mode_local or "unknown",
            "source_kind": source_kind,
            "error_delta": round(float(error_delta or 0.0), 3),
        }

        return {
            "support_gain_db": gain,
            "support_mode": mode,
            "support_policy": policy,
        }

    def _collect_core_support_candidates() -> list[dict[str, Any]]:
        diagnostics = getattr(plan, "planning_diagnostics", {}) or {}
        selected_sections = list(diagnostics.get("selected_sections") or [])
        backbone_parent = str(((diagnostics.get("backbone_plan") or {}).get("backbone_parent") or "A"))
        arrangement_mode_local = str((diagnostics.get("arrangement_mode") or "")).strip().lower()
        raw_candidates: list[dict[str, Any]] = []

        for index, section_diag in enumerate(selected_sections):
            section_label_raw = section_diag.get("label")
            section_label = _normalize_section_label(section_label_raw)
            if section_label not in core_labels:
                continue
            selected_parent = str(section_diag.get("selected_parent") or "")
            selected_window_label = str(section_diag.get("selected_window_label") or "")
            seen: set[tuple[str, str]] = {(selected_parent, selected_window_label)} if selected_parent and selected_window_label else set()

            alternates: list[dict[str, Any]] = []
            cross_parent = section_diag.get("cross_parent_best_alternate")
            if isinstance(cross_parent, dict):
                alternates.append(cross_parent)
            alternates.extend(
                item
                for item in list(section_diag.get("candidate_shortlist") or [])
                if isinstance(item, dict) and not bool(item.get("selected"))
            )

            for alt in alternates:
                support_parent = str(alt.get("parent_id") or "")
                support_label = str(alt.get("window_label") or "")
                identity = (support_parent, support_label)
                if not support_parent or not support_label or identity in seen:
                    continue
                seen.add(identity)
                if support_parent == selected_parent:
                    continue

                score_breakdown = dict(alt.get("score_breakdown") or {})
                stretch_ratio = float(score_breakdown.get("stretch_ratio", 1.0) or 1.0)
                stretch_gate = float(score_breakdown.get("stretch_gate", 0.0) or 0.0)
                seam_risk = float(score_breakdown.get("seam_risk", 0.0) or 0.0)
                transition_viability = float(score_breakdown.get("transition_viability", 0.0) or 0.0)
                error_delta = float(alt.get("error_delta_vs_selected", 99.0) or 99.0)
                if error_delta > 9.5:
                    continue
                if stretch_gate > 0.85 or stretch_ratio > 1.30 or seam_risk > 0.92 or transition_viability > 0.95:
                    continue

                recipe = _support_recipe_for_section(
                    section_label_raw,
                    arrangement_mode_local=arrangement_mode_local,
                    error_delta=error_delta,
                    seam_risk=seam_risk,
                    transition_viability=transition_viability,
                    stretch_ratio=stretch_ratio,
                    selected_parent=selected_parent,
                    backbone_parent=backbone_parent,
                    source_kind="support_overlay",
                )

                raw_candidates.append(
                    {
                        "section_index": index,
                        "section_label": section_label_raw,
                        "transition_mode": section_diag.get("transition_mode"),
                        "support_parent": support_parent,
                        "support_section_label": support_label,
                        "support_gain_db": recipe["support_gain_db"],
                        "support_mode": recipe["support_mode"],
                        "support_policy": recipe["support_policy"],
                        "selection_rank": int(alt.get("rank") or section_diag.get("selection_rank") or 1),
                        "error_delta": error_delta,
                        "planner_error": float(alt.get("planner_error", 0.0) or 0.0),
                        "backbone_parent": backbone_parent,
                        "kind": "support_overlay",
                    }
                )

        best_by_section: dict[int, dict[str, Any]] = {}
        for candidate in raw_candidates:
            sec_idx = _section_index_of(candidate, default=-1)
            if sec_idx < 0:
                continue
            existing = best_by_section.get(sec_idx)
            candidate_rank = (
                float(candidate.get("error_delta", 99.0) or 99.0),
                int(candidate.get("selection_rank") or 999),
            )
            if existing is None:
                best_by_section[sec_idx] = candidate
                continue
            existing_rank = (
                float(existing.get("error_delta", 99.0) or 99.0),
                int(existing.get("selection_rank") or 999),
            )
            if candidate_rank < existing_rank:
                best_by_section[sec_idx] = candidate

        # Fallback stage 1: if explicit alternates are missing, synthesize
        # counter-parent supports on donor-led core sections so adaptive variants
        # can become integrated two-parent arrangements instead of A/B handoffs.
        if not best_by_section:
            for index, section_diag in enumerate(selected_sections):
                section_label_raw = section_diag.get("label")
                section_label = _normalize_section_label(section_label_raw)
                if section_label not in core_labels:
                    continue
                selected_parent = str(section_diag.get("selected_parent") or "").strip()
                selected_window_label = str(section_diag.get("selected_window_label") or "").strip()
                if not selected_parent or not selected_window_label:
                    continue
                if not backbone_parent or selected_parent == backbone_parent:
                    continue

                synthesized_error_delta = 0.85
                recipe = _support_recipe_for_section(
                    section_label_raw,
                    arrangement_mode_local=arrangement_mode_local,
                    error_delta=synthesized_error_delta,
                    seam_risk=0.55,
                    transition_viability=0.52,
                    stretch_ratio=1.0,
                    selected_parent=selected_parent,
                    backbone_parent=backbone_parent,
                    source_kind="support_overlay_counterparent",
                )
                synthesized = {
                    "section_index": index,
                    "section_label": section_label_raw,
                    "transition_mode": section_diag.get("transition_mode"),
                    "support_parent": backbone_parent,
                    "support_section_label": selected_window_label,
                    "support_gain_db": recipe["support_gain_db"],
                    "support_mode": recipe["support_mode"],
                    "support_policy": recipe["support_policy"],
                    "selection_rank": int(section_diag.get("selection_rank") or 999),
                    "error_delta": synthesized_error_delta,
                    "planner_error": 0.0,
                    "backbone_parent": backbone_parent,
                    "kind": "support_overlay_counterparent",
                }
                best_by_section[index] = synthesized

        # Fallback stage 2: if still empty, synthesize supports from safe donor swap opportunities.
        if not best_by_section:
            for op in opportunities:
                if not _is_core_donor_op(op):
                    continue
                sec_idx = _section_index_of(op, default=-1)
                if sec_idx < 0:
                    continue
                alt_parent = str(op.get("alternate_parent") or "")
                selected_parent = str(op.get("selected_parent") or "")
                backbone_parent = str(op.get("backbone_parent") or "")
                if not alt_parent or alt_parent == backbone_parent or (selected_parent and alt_parent == selected_parent):
                    continue
                section_label = str(op.get("section_label") or "")
                error_delta = float(op.get("error_delta", 99.0) or 99.0)
                if error_delta > 1.35:
                    continue
                score_breakdown = dict(op.get("score_breakdown") or {})
                recipe = _support_recipe_for_section(
                    section_label,
                    arrangement_mode_local=arrangement_mode_local,
                    error_delta=error_delta,
                    seam_risk=float(score_breakdown.get("seam_risk", 0.45) or 0.45),
                    transition_viability=float(score_breakdown.get("transition_viability", 0.45) or 0.45),
                    stretch_ratio=float(score_breakdown.get("stretch_ratio", 1.0) or 1.0),
                    selected_parent=selected_parent,
                    backbone_parent=backbone_parent,
                    source_kind="support_overlay_fallback",
                )
                synthesized = {
                    "section_index": sec_idx,
                    "section_label": section_label,
                    "transition_mode": op.get("transition_mode"),
                    "support_parent": alt_parent,
                    "support_section_label": str(op.get("alternate_window_label") or ""),
                    "support_gain_db": recipe["support_gain_db"],
                    "support_mode": recipe["support_mode"],
                    "support_policy": recipe["support_policy"],
                    "selection_rank": int(op.get("selection_rank") or 999),
                    "error_delta": error_delta,
                    "planner_error": float(op.get("planner_error", 0.0) or 0.0),
                    "backbone_parent": backbone_parent,
                    "kind": "support_overlay_fallback",
                }
                existing = best_by_section.get(sec_idx)
                if existing is None:
                    best_by_section[sec_idx] = synthesized
                    continue
                existing_rank = (
                    float(existing.get("error_delta", 99.0) or 99.0),
                    int(existing.get("selection_rank") or 999),
                )
                synthesized_rank = (
                    float(synthesized.get("error_delta", 99.0) or 99.0),
                    int(synthesized.get("selection_rank") or 999),
                )
                if synthesized_rank < existing_rank:
                    best_by_section[sec_idx] = synthesized

        section_priority = {"payoff": 0, "build": 1, "verse": 2, "bridge": 3}
        ordered = sorted(
            best_by_section.values(),
            key=lambda item: (
                section_priority.get(_normalize_section_label(item.get("section_label")), 9),
                float(((item.get("support_policy") or {}).get("risk", 0.0) or 0.0)),
                float(item.get("error_delta", 99.0) or 99.0),
                int(item.get("selection_rank") or 999),
            ),
        )
        return ordered

    support_candidates = _collect_core_support_candidates()
    reserve_combo_slot = max_variants >= 3 and _has_safe_dual_combo(opportunities)
    has_core_donor_swap = any(_is_core_donor_op(op) for op in opportunities)
    arrangement_mode = str(((getattr(plan, "planning_diagnostics", {}) or {}).get("arrangement_mode") or "")).strip().lower()
    reserve_support_slot = False
    if arrangement_mode in {"baseline", "adaptive"}:
        # Promote support overlays from fallback behavior into the primary pro-mode search path.
        # Guardrail: always keep at least one integrated support candidate in both baseline and adaptive sets.
        reserve_support_slot = max_variants >= 3 and bool(support_candidates)

    max_single_slots = max_variants - len(configs) - (1 if reserve_combo_slot else 0) - (1 if reserve_support_slot else 0)
    max_single_slots = max(0, max_single_slots)

    # First pass: section-diverse singles (one strong alternate per section).
    seen_sections: set[int] = set()

    # Guarantee one core donor single (verse/build/payoff/bridge) when available.
    if max_single_slots > 0:
        for op in opportunities:
            if not _is_core_donor_op(op):
                continue
            sec_idx = _section_index_of(op, default=-1)
            identity = _op_identity(op)
            singles.append(op)
            seen_sections.add(sec_idx)
            used_identities.add(identity)
            break

    for op in opportunities:
        if len(singles) >= max_single_slots:
            break
        sec_idx = _section_index_of(op, default=-1)
        identity = _op_identity(op)
        if sec_idx in seen_sections or identity in used_identities:
            continue
        singles.append(op)
        seen_sections.add(sec_idx)
        used_identities.add(identity)

    # Second pass: fill remaining slots with next-best singles.
    if len(singles) < max_single_slots:
        for op in opportunities:
            identity = _op_identity(op)
            if identity in used_identities:
                continue
            singles.append(op)
            used_identities.add(identity)
            if len(singles) >= max_single_slots:
                break

    for idx, opportunity in enumerate(singles, start=1):
        section_label = str(opportunity.get("section_label") or f"section_{opportunity.get('section_index', idx)}")
        alt_parent = str(opportunity.get("alternate_parent") or "X")
        alt_label = str(opportunity.get("alternate_window_label") or f"window_{idx}")
        variant_id = f"swap_{idx:02d}_{section_label}_{alt_parent}"
        configs.append(
            {
                "variant_id": variant_id,
                "label": f"{section_label} -> {alt_parent}:{alt_label}",
                "strategy": "single_section_alternate",
                "variant_mode": variant_mode,
                "swaps": [opportunity],
            }
        )
        if len(configs) >= max_variants:
            return configs[:max_variants]

    # Third pass: when room exists, add safe two-swap variants to explore macro shape.
    by_section: dict[int, dict[str, Any]] = {}
    core_donor_single_selected = any(_is_core_donor_op(op) for op in singles)
    prefer_core_donor_combo = not core_donor_single_selected
    enforce_baseline_core_donor_combo = arrangement_mode == "baseline"

    def _combo_section_rank(op: dict[str, Any]) -> tuple[int, float]:
        # If no donor-bearing core single exists, bias combo construction toward
        # donor-bearing core section swaps so baseline doesn't collapse to backbone-only shape changes.
        donor_bias = 0 if (prefer_core_donor_combo and _is_core_donor_op(op)) else 1
        try:
            error_delta = float(op.get("error_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            error_delta = 0.0
        return donor_bias, error_delta

    donor_by_section: dict[int, dict[str, Any]] = {}
    opportunities_by_section: dict[int, list[dict[str, Any]]] = {}
    for op in opportunities:
        sec_idx = _section_index_of(op, default=-1)
        if sec_idx < 0:
            continue
        opportunities_by_section.setdefault(sec_idx, []).append(op)
        existing = by_section.get(sec_idx)
        if existing is None or _combo_section_rank(op) < _combo_section_rank(existing):
            by_section[sec_idx] = op
        if _is_core_donor_op(op):
            donor_existing = donor_by_section.get(sec_idx)
            if donor_existing is None or _combo_section_rank(op) < _combo_section_rank(donor_existing):
                donor_by_section[sec_idx] = op

    def _is_handoff_mode(op: dict[str, Any]) -> bool:
        mode = str(op.get("transition_mode") or "").strip().lower().replace("-", "_")
        return mode in {"arrival_handoff", "single_owner_handoff"}

    def _chain_candidate_rank(op: dict[str, Any]) -> tuple[float, int, float, int]:
        return (
            float(op.get("error_delta", 99.0) or 99.0),
            int(op.get("selection_rank") or 999),
            float(op.get("planner_error", 99.0) or 99.0),
            _section_index_of(op, default=999),
        )

    section_candidates: dict[int, list[dict[str, Any]]] = {}
    for sec_idx in sorted(by_section):
        primary = by_section[sec_idx]
        choices = [primary]

        donor_choice = donor_by_section.get(sec_idx)
        if donor_choice is not None and _op_identity(donor_choice) != _op_identity(primary):
            choices.append(donor_choice)

        # Proposal synthesis: if the primary choice for a handoff section cannot form
        # a contiguous same-owner chain, include the best nearby handoff alternate
        # that can. Ranking logic then decides if this chain candidate wins.
        if _is_handoff_mode(primary):
            adjacent_handoff_parents: set[str] = set()
            for neighbor_idx in (sec_idx - 1, sec_idx + 1):
                for neighbor in opportunities_by_section.get(neighbor_idx, []):
                    if not _is_handoff_mode(neighbor):
                        continue
                    parent = str(neighbor.get("alternate_parent") or "")
                    if parent:
                        adjacent_handoff_parents.add(parent)

            if adjacent_handoff_parents:
                chain_candidates = [
                    item
                    for item in opportunities_by_section.get(sec_idx, [])
                    if _is_handoff_mode(item)
                    and str(item.get("alternate_parent") or "") in adjacent_handoff_parents
                    and _op_identity(item) not in {_op_identity(choice) for choice in choices}
                ]
                if chain_candidates:
                    chain_choice = min(chain_candidates, key=_chain_candidate_rank)
                    choices.append(chain_choice)

        section_candidates[sec_idx] = choices

    combo_candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    seen_combo_pairs: set[tuple[tuple[int, str, str], tuple[int, str, str]]] = set()
    ordered_sections = sorted(section_candidates)
    for left_idx in range(len(ordered_sections)):
        for right_idx in range(left_idx + 1, len(ordered_sections)):
            left_section = ordered_sections[left_idx]
            right_section = ordered_sections[right_idx]
            left_choices = section_candidates[left_section]
            right_choices = section_candidates[right_section]
            for left in left_choices:
                for right in right_choices:
                    if _section_index_of(left, default=-1) == _section_index_of(right, default=-1):
                        continue
                    left_identity = _op_identity(left)
                    right_identity = _op_identity(right)
                    pair_identity = tuple(sorted((left_identity, right_identity)))
                    if pair_identity in seen_combo_pairs:
                        continue
                    seen_combo_pairs.add(pair_identity)
                    combo_error = float(left.get("error_delta", 0.0) or 0.0) + float(right.get("error_delta", 0.0) or 0.0)
                    if combo_error > 1.65:
                        continue
                    combo_candidates.append((left, right, combo_error))

    def _combo_has_core_donor(item: tuple[dict[str, Any], dict[str, Any], float]) -> bool:
        left, right, _ = item
        return _is_core_donor_op(left) or _is_core_donor_op(right)

    def _combo_has_any_donor(item: tuple[dict[str, Any], dict[str, Any], float]) -> bool:
        left, right, _ = item
        return (
            str(left.get("alternate_parent") or "") != str(left.get("backbone_parent") or "")
            or str(right.get("alternate_parent") or "") != str(right.get("backbone_parent") or "")
        )

    def _combo_has_intro_or_outro(item: tuple[dict[str, Any], dict[str, Any], float]) -> bool:
        left, right, _ = item
        left_label = _normalize_section_label(left.get("section_label"))
        right_label = _normalize_section_label(right.get("section_label"))
        return left_label in {"intro", "outro"} or right_label in {"intro", "outro"}

    def _op_transition_mode(op: dict[str, Any]) -> str:
        return str(op.get("transition_mode") or "").strip().lower().replace("-", "_")

    def _op_handoff_pressure(op: dict[str, Any]) -> float:
        score_breakdown = dict(op.get("score_breakdown") or {})
        seam_risk = float(score_breakdown.get("seam_risk", 0.0) or 0.0)
        transition_error = float(score_breakdown.get("transition_viability", 0.0) or 0.0)
        stretch_ratio = float(score_breakdown.get("stretch_ratio", 1.0) or 1.0)
        stretch_pressure = min(1.0, abs(stretch_ratio - 1.0) * 3.0)
        base_pressure = max(seam_risk, transition_error, stretch_pressure)

        section_label = _normalize_section_label(op.get("section_label"))
        if section_label in {"build", "payoff"}:
            base_pressure = min(1.0, base_pressure + 0.06)
        if _op_transition_mode(op) in {"arrival_handoff", "single_owner_handoff"}:
            base_pressure = min(1.0, base_pressure + 0.12)
        return base_pressure

    def _combo_priority(item: tuple[dict[str, Any], dict[str, Any], float]) -> tuple[int, int, int, int, int, float, int, float, int, int]:
        left, right, combo_error = item
        left_label = _normalize_section_label(left.get("section_label"))
        right_label = _normalize_section_label(right.get("section_label"))
        has_payoff = left_label == "payoff" or right_label == "payoff"
        has_build = left_label == "build" or right_label == "build"
        has_payoff_build = has_payoff and has_build
        has_core_donor = _combo_has_core_donor(item)
        left_mode = _op_transition_mode(left)
        right_mode = _op_transition_mode(right)
        left_handoff = left_mode in {"arrival_handoff", "single_owner_handoff"}
        right_handoff = right_mode in {"arrival_handoff", "single_owner_handoff"}
        has_any_handoff = left_handoff or right_handoff
        has_handoff_build_or_payoff = (
            (left_handoff and left_label in {"build", "payoff"})
            or (right_handoff and right_label in {"build", "payoff"})
        )
        handoff_pressure_sum = _op_handoff_pressure(left) + _op_handoff_pressure(right)
        intro_outro_penalty = int(left_label in {"intro", "outro"}) + int(right_label in {"intro", "outro"})
        left_idx = int(left.get("section_index", 0) or 0)
        right_idx = int(right.get("section_index", 0) or 0)
        section_gap = abs(left_idx - right_idx)
        contiguous_sections = section_gap == 1
        same_alt_parent = (
            str(left.get("alternate_parent") or "")
            and str(left.get("alternate_parent") or "") == str(right.get("alternate_parent") or "")
        )
        ownership_chain_combo = contiguous_sections and same_alt_parent and has_any_handoff
        return (
            0 if (prefer_core_donor_combo and has_core_donor) else 1,
            0 if has_handoff_build_or_payoff else 1,
            0 if ownership_chain_combo else 1,
            0 if has_payoff_build else 1,
            0 if has_payoff else 1,
            -round(handoff_pressure_sum, 4),
            intro_outro_penalty,
            combo_error,
            0 if has_build else 1,
            left_idx + right_idx,
        )

    if enforce_baseline_core_donor_combo:
        core_donor_combo_candidates = [item for item in combo_candidates if _combo_has_core_donor(item)]
        if core_donor_combo_candidates:
            combo_candidates = core_donor_combo_candidates
        else:
            core_shape_combo_candidates = [item for item in combo_candidates if not _combo_has_intro_or_outro(item)]
            if core_shape_combo_candidates:
                combo_candidates = core_shape_combo_candidates
            else:
                any_donor_combo_candidates = [item for item in combo_candidates if _combo_has_any_donor(item)]
                if any_donor_combo_candidates:
                    combo_candidates = any_donor_combo_candidates

    combo_candidates.sort(key=_combo_priority)

    combo_slot_cap = max_variants - (1 if reserve_support_slot else 0)
    combo_slot_cap = max(1, combo_slot_cap)
    combo_idx = 1
    for left, right, _combo_error in combo_candidates:
        if len(configs) >= combo_slot_cap:
            break
        combo_variant_id = f"combo_{combo_idx:02d}_{left.get('section_label')}_{right.get('section_label')}"
        configs.append(
            {
                "variant_id": combo_variant_id,
                "label": f"combo {left.get('section_label')} + {right.get('section_label')}",
                "strategy": "dual_section_alternate",
                "variant_mode": variant_mode,
                "swaps": [left, right],
            }
        )
        combo_idx += 1

    if reserve_support_slot and support_candidates and len(configs) < max_variants:
        def _support_section_name(item: dict[str, Any]) -> str:
            return _normalize_section_label(item.get("section_label"))

        def _support_risk(item: dict[str, Any]) -> float:
            policy = dict(item.get("support_policy") or {})
            return float(policy.get("risk", 0.0) or 0.0)

        def _support_transition_mode(item: dict[str, Any]) -> str:
            return str(item.get("transition_mode") or "").strip().lower().replace("-", "_")

        def _support_handoff_pressure(item: dict[str, Any]) -> float:
            policy = dict(item.get("support_policy") or {})
            mode = _support_transition_mode(item)
            base_pressure = max(
                float(policy.get("risk", 0.0) or 0.0),
                float(policy.get("foreground_collision_risk", 0.0) or 0.0),
                min(1.0, 1.0 - float(policy.get("transition_viability", 0.0) or 0.0)),
            )
            label = _support_section_name(item)
            if label in {"build", "payoff"}:
                base_pressure = min(1.0, base_pressure + 0.07)
            if mode in {"arrival_handoff", "single_owner_handoff"}:
                base_pressure = min(1.0, base_pressure + 0.12)
            return base_pressure

        def _best_dual_support_pair(items: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
            if len(items) < 2:
                return None
            best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
            best_rank: tuple[Any, ...] | None = None

            for left_idx in range(len(items)):
                for right_idx in range(left_idx + 1, len(items)):
                    left = items[left_idx]
                    right = items[right_idx]
                    if _section_index_of(left, default=-1) == _section_index_of(right, default=-1):
                        continue

                    left_label = _support_section_name(left)
                    right_label = _support_section_name(right)
                    has_payoff = left_label == "payoff" or right_label == "payoff"
                    has_build = left_label == "build" or right_label == "build"

                    left_risk = _support_risk(left)
                    right_risk = _support_risk(right)
                    max_risk = max(left_risk, right_risk)
                    mean_risk = (left_risk + right_risk) / 2.0

                    left_mode = _support_transition_mode(left)
                    right_mode = _support_transition_mode(right)
                    left_handoff = left_mode in {"arrival_handoff", "single_owner_handoff"}
                    right_handoff = right_mode in {"arrival_handoff", "single_owner_handoff"}
                    has_any_handoff = left_handoff or right_handoff
                    has_handoff_build_or_payoff = (
                        (left_handoff and left_label in {"build", "payoff"})
                        or (right_handoff and right_label in {"build", "payoff"})
                    )
                    handoff_pressure_sum = _support_handoff_pressure(left) + _support_handoff_pressure(right)

                    has_payoff_build = has_payoff and has_build and max_risk <= 0.85
                    payoff_preferred = has_payoff and max_risk <= 0.92

                    left_section_idx = _section_index_of(left, default=0)
                    right_section_idx = _section_index_of(right, default=0)
                    section_span = abs(left_section_idx - right_section_idx)
                    contiguous_pair = section_span == 1
                    same_transition_mode = bool(left_mode and left_mode == right_mode)
                    chain_mismatch_penalty = 0
                    if contiguous_pair and has_payoff_build:
                        if left_handoff != right_handoff:
                            chain_mismatch_penalty = 3 if max_risk >= 0.50 else 2
                        elif not same_transition_mode:
                            chain_mismatch_penalty = 1

                    error_sum = float(left.get("error_delta", 99.0) or 99.0) + float(right.get("error_delta", 99.0) or 99.0)
                    same_parent_penalty = 0 if str(left.get("support_parent") or "") != str(right.get("support_parent") or "") else 1

                    rank = (
                        chain_mismatch_penalty,
                        0 if has_handoff_build_or_payoff else 1,
                        0 if has_any_handoff else 1,
                        0 if has_payoff_build else 1,
                        0 if payoff_preferred else 1,
                        -round(handoff_pressure_sum, 4),
                        round(max_risk, 4),
                        round(mean_risk, 4),
                        round(error_sum, 4),
                        same_parent_penalty,
                        -section_span,
                    )
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = (left, right)

            return best_pair

        if arrangement_mode == "adaptive" and len(support_candidates) >= 2:
            chosen_pair = _best_dual_support_pair(support_candidates)
            if chosen_pair is not None:
                primary, secondary = chosen_pair
                primary_label = str(primary.get("section_label") or "section")
                secondary_label = str(secondary.get("section_label") or "section")
                support_parent = str(primary.get("support_parent") or secondary.get("support_parent") or "X")
                configs.append(
                    {
                        "variant_id": f"support_01_{primary_label}_{secondary_label}_{support_parent}",
                        "label": f"{primary_label}+{secondary_label} integrated support",
                        "strategy": "dual_section_support",
                        "variant_mode": variant_mode,
                        "swaps": [],
                        "supports": [primary, secondary],
                    }
                )
        if len(configs) < max_variants and (arrangement_mode != "adaptive" or len(configs) == 1):
            support_idx = 1
            for support in support_candidates:
                if len(configs) >= max_variants:
                    break
                section_label = str(support.get("section_label") or f"section_{support.get('section_index', support_idx)}")
                support_parent = str(support.get("support_parent") or "X")
                support_label = str(support.get("support_section_label") or f"window_{support_idx}")
                variant_id = f"support_{support_idx:02d}_{section_label}_{support_parent}"
                configs.append(
                    {
                        "variant_id": variant_id,
                        "label": f"{section_label} + {support_parent}:{support_label} support",
                        "strategy": "single_section_support",
                        "variant_mode": variant_mode,
                        "swaps": [],
                        "supports": [support],
                    }
                )
                support_idx += 1

    if len(configs) < max_variants:
        for op in opportunities:
            identity = _op_identity(op)
            if identity in used_identities:
                continue
            section_label = str(op.get("section_label") or f"section_{op.get('section_index', len(configs))}")
            alt_parent = str(op.get("alternate_parent") or "X")
            alt_label = str(op.get("alternate_window_label") or f"window_{len(configs)}")
            variant_id = f"swap_{len(configs):02d}_{section_label}_{alt_parent}"
            configs.append(
                {
                    "variant_id": variant_id,
                    "label": f"{section_label} -> {alt_parent}:{alt_label}",
                    "strategy": "single_section_alternate",
                    "variant_mode": variant_mode,
                    "swaps": [op],
                }
            )
            used_identities.add(identity)
            if len(configs) >= max_variants:
                break

    return configs[:max_variants]



def _apply_auto_shortlist_variant(plan: Any, variant_config: dict[str, Any] | None) -> Any:
    cloned = deepcopy(plan)
    swaps = list((variant_config or {}).get("swaps") or [])
    supports = list((variant_config or {}).get("supports") or [])
    if not variant_config or (not swaps and not supports):
        diagnostics = getattr(cloned, "planning_diagnostics", {}) or {}
        diagnostics["variant"] = variant_config or {"variant_id": "baseline", "strategy": "baseline"}
        cloned.planning_diagnostics = diagnostics
        return cloned

    diagnostics = getattr(cloned, "planning_diagnostics", {}) or {}
    selected_sections = list(diagnostics.get("selected_sections") or [])
    overrides: list[dict[str, Any]] = []
    support_overrides: list[dict[str, Any]] = []
    for swap in swaps:
        raw_section_index = swap.get("section_index", -1)
        try:
            section_index = int(raw_section_index)
        except (TypeError, ValueError):
            section_index = -1
        if section_index < 0 or section_index >= len(cloned.sections):
            continue
        section = cloned.sections[section_index]
        section.source_parent = str(swap.get("alternate_parent") or section.source_parent)
        section.source_section_label = str(swap.get("alternate_window_label") or section.source_section_label)
        if 0 <= section_index < len(selected_sections):
            diag = dict(selected_sections[section_index])
            diag["selected_parent"] = section.source_parent
            diag["selected_role"] = "backbone" if section.source_parent == swap.get("backbone_parent") else "donor"
            diag["selected_window_label"] = section.source_section_label
            diag["selected_window_origin"] = swap.get("window_origin")
            diag["selected_window_seconds"] = swap.get("window_seconds")
            diag["planner_error"] = round(float(swap.get("planner_error", diag.get("planner_error", 0.0)) or 0.0), 3)
            diag["selection_rank"] = int(swap.get("selection_rank") or diag.get("selection_rank") or 1)
            diag["selected_by_guard"] = True
            shortlist = []
            for row in list(diag.get("candidate_shortlist") or []):
                row_copy = dict(row)
                row_copy["selected"] = bool(
                    row_copy.get("parent_id") == section.source_parent
                    and row_copy.get("window_label") == section.source_section_label
                )
                shortlist.append(row_copy)
            diag["candidate_shortlist"] = shortlist
            selected_sections[section_index] = diag
        overrides.append(
            {
                "section_index": section_index,
                "section_label": section.label,
                "source_parent": section.source_parent,
                "source_section_label": section.source_section_label,
                "strategy": variant_config.get("strategy"),
                "kind": swap.get("kind"),
                "planner_error": swap.get("planner_error"),
                "error_delta": swap.get("error_delta"),
            }
        )
    for support in supports:
        raw_section_index = support.get("section_index", -1)
        try:
            section_index = int(raw_section_index)
        except (TypeError, ValueError):
            section_index = -1
        if section_index < 0 or section_index >= len(cloned.sections):
            continue

        section = cloned.sections[section_index]
        support_parent = str(support.get("support_parent") or "").strip()
        support_section_label = str(support.get("support_section_label") or "").strip()
        if not support_parent or not support_section_label:
            continue

        section.support_parent = support_parent
        section.support_section_label = support_section_label
        try:
            section.support_gain_db = float(support.get("support_gain_db", -10.0) or -10.0)
        except (TypeError, ValueError):
            section.support_gain_db = -10.0
        section.support_mode = str(support.get("support_mode") or "filtered_counterlayer")

        support_policy = dict(support.get("support_policy") or {})
        try:
            section.support_transition_risk = float(support_policy.get("risk")) if support_policy.get("risk") is not None else None
        except (TypeError, ValueError):
            section.support_transition_risk = None
        try:
            section.support_foreground_collision_risk = (
                float(support_policy.get("foreground_collision_risk"))
                if support_policy.get("foreground_collision_risk") is not None
                else None
            )
        except (TypeError, ValueError):
            section.support_foreground_collision_risk = None
        try:
            section.support_transition_viability = (
                float(support_policy.get("transition_viability"))
                if support_policy.get("transition_viability") is not None
                else None
            )
        except (TypeError, ValueError):
            section.support_transition_viability = None

        if 0 <= section_index < len(selected_sections):
            diag = dict(selected_sections[section_index])
            diag["support_recipe"] = {
                "parent_id": section.support_parent,
                "window_label": section.support_section_label,
                "gain_db": round(float(section.support_gain_db or -10.0), 3),
                "mode": section.support_mode,
                "policy": support_policy,
            }
            selected_sections[section_index] = diag

        support_overrides.append(
            {
                "section_index": section_index,
                "section_label": section.label,
                "support_parent": section.support_parent,
                "support_section_label": section.support_section_label,
                "support_gain_db": round(float(section.support_gain_db or -10.0), 3),
                "support_mode": section.support_mode,
                "support_policy": support_policy,
                "strategy": variant_config.get("strategy"),
                "kind": support.get("kind"),
                "planner_error": support.get("planner_error"),
                "error_delta": support.get("error_delta"),
            }
        )

    diagnostics["selected_sections"] = selected_sections
    diagnostics["variant"] = {
        "variant_id": variant_config.get("variant_id"),
        "label": variant_config.get("label"),
        "strategy": variant_config.get("strategy"),
        "variant_mode": variant_config.get("variant_mode"),
        "overrides": overrides,
        "support_overrides": support_overrides,
    }
    cloned.planning_diagnostics = diagnostics
    note_parts: list[str] = [f"Auto-shortlist variant applied: {variant_config.get('label') or variant_config.get('variant_id')}." ]
    if support_overrides:
        note_parts.append(
            "Integrated support overlays: "
            + ", ".join(
                f"{item['section_label']}<-{item['support_parent']}:{item['support_section_label']}"
                for item in support_overrides
            )
            + "."
        )
    cloned.planning_notes = list(getattr(cloned, "planning_notes", []) or []) + note_parts
    return cloned



def _render_fusion_candidate(
    song_a: Any,
    song_b: Any,
    base_plan: Any,
    outdir: str | Path,
    *,
    candidate_id: str,
    variant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant = variant_config or {"variant_id": candidate_id, "label": candidate_id, "strategy": "baseline", "swaps": [], "supports": []}
    plan = _apply_auto_shortlist_variant(base_plan, variant)
    result = _render_fusion_plan_candidate(song_a, song_b, plan, outdir, variant_config=variant)
    result["candidate_id"] = candidate_id
    result["variant_config"] = variant
    result["render_strategy"] = variant.get("strategy", "baseline")
    return result


def _parent_balance_from_plan(plan_path: str | None) -> float:
    if not plan_path:
        return 0.0
    try:
        payload = _load_json(Path(plan_path))
    except Exception:
        return 0.0
    diagnostics = (payload or {}).get("planning_diagnostics") or {}
    sections = list(diagnostics.get("selected_sections") or [])
    if not sections:
        return 0.0
    counts = {"A": 0, "B": 0}
    for section in sections:
        parent = str((section or {}).get("selected_parent") or "").strip().upper()
        if parent in counts:
            counts[parent] += 1
    a, b = counts["A"], counts["B"]
    if a <= 0 or b <= 0:
        return 0.0
    return float(min(a, b) / max(a, b))


def _pro_fusion_selection_score(report: dict[str, Any], parent_balance: float) -> float:
    overall = float(report.get("overall_score") or 0.0)
    song_likeness = float((report.get("song_likeness") or {}).get("score") or 0.0)
    groove = float((report.get("groove") or {}).get("score") or 0.0)
    transition = float((report.get("transition") or {}).get("score") or 0.0)
    mix_sanity = float((report.get("mix_sanity") or {}).get("score") or 0.0)
    gating_status = str((report.get("gating") or {}).get("status") or "").strip().lower()

    song_metrics = (((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {})
    medley_risk = _clamp01(_safe_metric(song_metrics.get("full_mix_medley_risk"), default=0.0))
    integrated_ratio = _clamp01(_safe_metric(song_metrics.get("integrated_two_parent_section_ratio"), default=0.0))
    max_parent_share = _clamp01(_safe_metric(song_metrics.get("max_parent_share"), default=1.0))
    owner_switch_ratio = _clamp01(_safe_metric(song_metrics.get("owner_switch_ratio"), default=0.0))

    score = (
        0.34 * overall
        + 0.24 * song_likeness
        + 0.10 * groove
        + 0.22 * transition
        + 0.10 * mix_sanity
    )
    score += 10.0 * float(max(0.0, min(1.0, parent_balance)))
    if parent_balance <= 0.0:
        score -= 8.0

    # Guardrail: de-prioritize medley-like outputs that feel like back-and-forth track switching.
    integration_target = 0.35
    integration_gap = max(0.0, integration_target - integrated_ratio)
    medley_penalty = (
        18.0 * medley_risk
        + 12.0 * integration_gap
        + 8.0 * max(0.0, max_parent_share - 0.80)
        + 10.0 * owner_switch_ratio
    )
    score -= medley_penalty

    if gating_status == "reject":
        score -= 12.0
    elif gating_status == "review":
        score -= 3.0
    elif gating_status == "pass":
        # Under pass gate, bias toward listener-perceived song flow quality.
        score += 0.12 * song_likeness + 0.16 * transition

    score -= _seam_risk_penalty(report)
    return round(float(score), 3)


def _extract_transition_seam_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    transition = report.get("transition") or {}
    details = transition.get("details") or {}
    aggregate = details.get("aggregate_metrics") or {}
    snapshot: dict[str, Any] = {
        "transition_score": float(transition.get("score") or 0.0),
    }
    for key in (
        "mean_seam_risk",
        "max_seam_risk",
        "mean_energy_jump",
        "mean_spectral_jump",
        "mean_onset_jump",
        "avg_overlap_beats",
    ):
        if key in aggregate:
            snapshot[key] = aggregate.get(key)
    return snapshot


def _safe_metric(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _seam_risk_penalty(report: dict[str, Any]) -> float:
    transition_metrics = (((report.get("transition") or {}).get("details") or {}).get("aggregate_metrics") or {})
    mean_seam_risk = _clamp01(_safe_metric(transition_metrics.get("mean_seam_risk"), default=0.0))
    max_seam_risk = _clamp01(_safe_metric(transition_metrics.get("max_seam_risk"), default=mean_seam_risk))
    energy_jump = _clamp01(_safe_metric(transition_metrics.get("mean_energy_jump"), default=0.0))

    # Penalize unstable seams heavily so "track-switch" feeling loses ranking priority.
    return float(12.0 * mean_seam_risk + 10.0 * max_seam_risk + 3.0 * energy_jump)


def _winner_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    report = candidate.get("listen_report") or {}
    transition_metrics = (((report.get("transition") or {}).get("details") or {}).get("aggregate_metrics") or {})
    selection_score = _safe_metric(candidate.get("selection_score"), default=0.0)
    structure = _safe_metric((report.get("structure") or {}).get("score"), default=0.0)
    transition = _safe_metric((report.get("transition") or {}).get("score"), default=0.0)
    song_likeness = _safe_metric((report.get("song_likeness") or {}).get("score"), default=0.0)
    groove = _safe_metric((report.get("groove") or {}).get("score"), default=0.0)
    overall = _safe_metric(report.get("overall_score"), default=0.0)
    mean_seam_risk = _safe_metric(transition_metrics.get("mean_seam_risk"), default=1.0)
    adjusted_selection = selection_score - _seam_risk_penalty(report)
    return (
        round(structure, 6),
        round(transition, 6),
        round(song_likeness, 6),
        round(adjusted_selection, 6),
        round(groove, 6),
        round(overall, 6),
        round(-mean_seam_risk, 6),
    )


def _song_likeness_aggregate_metric(report: dict[str, Any], key: str) -> float | None:
    metrics = (((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {})
    if key not in metrics:
        return None
    value = _safe_metric(metrics.get(key), default=float("nan"))
    if not math.isfinite(value):
        return None
    return float(value)


def _candidate_meets_quality_floor(
    report: dict[str, Any],
    *,
    min_song_likeness: float,
    min_groove: float,
    min_structure: float,
    min_boundary_recovery: float,
    min_role_plausibility: float,
) -> bool:
    gate = str((report.get("gating") or {}).get("status") or "").strip().lower()
    song_likeness = float((report.get("song_likeness") or {}).get("score") or 0.0)
    groove = float((report.get("groove") or {}).get("score") or 0.0)
    structure = float((report.get("structure") or {}).get("score") or 0.0)
    boundary_recovery = _song_likeness_aggregate_metric(report, "boundary_recovery")
    role_plausibility = _song_likeness_aggregate_metric(report, "role_plausibility")

    structure_readability_ok = True
    if boundary_recovery is not None and boundary_recovery < float(min_boundary_recovery):
        structure_readability_ok = False
    if role_plausibility is not None and role_plausibility < float(min_role_plausibility):
        structure_readability_ok = False

    return (
        gate == "pass"
        and song_likeness >= float(min_song_likeness)
        and groove >= float(min_groove)
        and structure >= float(min_structure)
        and structure_readability_ok
    )


def _select_pro_fusion_winner(
    candidates: list[dict[str, Any]],
    *,
    min_song_likeness: float,
    min_groove: float,
    min_structure: float,
    min_boundary_recovery: float,
    min_role_plausibility: float,
) -> tuple[dict[str, Any] | None, str, dict[str, int]]:
    pass_candidates = [
        item
        for item in candidates
        if str(((item.get("listen_report") or {}).get("gating") or {}).get("status") or "").strip().lower() == "pass"
    ]
    floor_pass_candidates = [
        item
        for item in pass_candidates
        if _candidate_meets_quality_floor(
            item.get("listen_report") or {},
            min_song_likeness=min_song_likeness,
            min_groove=min_groove,
            min_structure=min_structure,
            min_boundary_recovery=min_boundary_recovery,
            min_role_plausibility=min_role_plausibility,
        )
    ]
    review_candidates = [
        item
        for item in candidates
        if str(((item.get("listen_report") or {}).get("gating") or {}).get("status") or "").strip().lower() == "review"
    ]

    counts = {
        "candidate_count": len(candidates),
        "pass_count": len(pass_candidates),
        "floor_pass_count": len(floor_pass_candidates),
        "review_count": len(review_candidates),
    }

    if floor_pass_candidates:
        winner = max(floor_pass_candidates, key=_winner_sort_key)
        return winner, "pass+floor", counts

    if pass_candidates:
        return None, "hard-fail:pass-below-floor", counts
    if review_candidates:
        return None, "hard-fail:no-pass", counts
    return None, "hard-fail:all-reject", counts


def _copy_if_exists(src: str | None, dst: Path) -> None:
    if not src:
        return
    src_path = Path(src)
    if not src_path.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)



def fusion(
    track1: str,
    track2: str,
    genre: Optional[str],
    bpm: Optional[int],
    key: Optional[str],
    output: Optional[str],
    arrangement_mode: str = "baseline",
) -> int:
    """Fuse two tracks together."""
    analyze_audio = _get_analyze_audio_file()
    evaluate_song = _get_evaluate_song()
    _, build_arrangement_plan = _get_planner_functions()
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

    if arrangement_mode == "pro":
        candidate_modes = ("adaptive", "baseline")
        per_mode_batch = 3
        candidates: list[dict[str, Any]] = []

        for mode in candidate_modes:
            plan = build_arrangement_plan(song_a, song_b, arrangement_mode=mode)
            variant_configs = _build_auto_shortlist_variant_configs(plan, per_mode_batch, variant_mode="safe")
            for idx, variant in enumerate(variant_configs, start=1):
                variant_id = str(variant.get("variant_id") or f"{mode}_{idx:02d}")
                candidate_dir = outdir / f"candidate_{mode}_{idx:02d}_{variant_id}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                result = _render_fusion_candidate(
                    song_a,
                    song_b,
                    plan,
                    candidate_dir,
                    candidate_id=f"{mode}_{idx:02d}",
                    variant_config=variant,
                )
                listened = evaluate_song(analyze_audio(result["master_wav_path"]))
                report = listened.to_dict()
                report_path = candidate_dir / "listen_report.json"
                _write_json(report_path, report)
                parent_balance = _parent_balance_from_plan(result.get("arrangement_plan_path"))
                selection_score = _pro_fusion_selection_score(report, parent_balance)
                candidates.append(
                    {
                        "mode": mode,
                        "variant": variant,
                        "result": result,
                        "listen_report": report,
                        "listen_report_path": str(report_path),
                        "parent_balance": round(parent_balance, 3),
                        "selection_score": selection_score,
                    }
                )

        min_song_likeness = 55.0
        min_groove = 60.0
        min_structure = 58.0
        min_boundary_recovery = 0.45
        min_role_plausibility = 0.48

        winner, selection_policy, selection_counts = _select_pro_fusion_winner(
            candidates,
            min_song_likeness=min_song_likeness,
            min_groove=min_groove,
            min_structure=min_structure,
            min_boundary_recovery=min_boundary_recovery,
            min_role_plausibility=min_role_plausibility,
        )

        selection_payload = {
            "mode": "pro",
            "selection_policy": {
                "winner_policy": selection_policy,
                "min_song_likeness": min_song_likeness,
                "min_groove": min_groove,
                "min_structure": min_structure,
                "min_boundary_recovery": min_boundary_recovery,
                "min_role_plausibility": min_role_plausibility,
                **selection_counts,
            },
            "winner": None,
            "promotion_blocked": winner is None,
            "candidates": [
                {
                    "arrangement_mode": item["mode"],
                    "variant": item.get("variant") or {},
                    "selection_score": item["selection_score"],
                    "overall_score": float((item["listen_report"] or {}).get("overall_score") or 0.0),
                    "song_likeness": float(((item["listen_report"] or {}).get("song_likeness") or {}).get("score") or 0.0),
                    "groove": float(((item["listen_report"] or {}).get("groove") or {}).get("score") or 0.0),
                    "transition": float(((item["listen_report"] or {}).get("transition") or {}).get("score") or 0.0),
                    "mix_sanity": float(((item["listen_report"] or {}).get("mix_sanity") or {}).get("score") or 0.0),
                    "structure": float(((item["listen_report"] or {}).get("structure") or {}).get("score") or 0.0),
                    "boundary_recovery": _song_likeness_aggregate_metric(item["listen_report"] or {}, "boundary_recovery"),
                    "role_plausibility": _song_likeness_aggregate_metric(item["listen_report"] or {}, "role_plausibility"),
                    "gating_status": str(((item["listen_report"] or {}).get("gating") or {}).get("status") or ""),
                    "seam_snapshot": _extract_transition_seam_snapshot(item["listen_report"] or {}),
                    "parent_balance": item["parent_balance"],
                    "listen_report_path": item["listen_report_path"],
                    "candidate_dir": str(Path(item["result"].get("master_wav_path") or "").parent),
                }
                for item in candidates
            ],
            "promoted_outputs": {
                "raw_wav": str(outdir / "child_raw.wav"),
                "master_wav": str(outdir / "child_master.wav"),
                "master_mp3": str(outdir / "child_master.mp3"),
                "manifest": str(outdir / "render_manifest.json"),
                "arrangement_plan": str(outdir / "arrangement_plan.json"),
            },
        }

        if winner is not None:
            best = winner["result"]
            _copy_if_exists(best.get("raw_wav_path"), outdir / "child_raw.wav")
            _copy_if_exists(best.get("master_wav_path"), outdir / "child_master.wav")
            _copy_if_exists(best.get("master_mp3_path"), outdir / "child_master.mp3")
            _copy_if_exists(best.get("render_manifest_path"), outdir / "render_manifest.json")
            _copy_if_exists(best.get("arrangement_plan_path"), outdir / "arrangement_plan.json")
            selection_payload["winner"] = {
                "arrangement_mode": winner["mode"],
                "variant": winner.get("variant") or {},
                "selection_score": winner["selection_score"],
                "overall_score": float((winner["listen_report"] or {}).get("overall_score") or 0.0),
                "song_likeness": float(((winner["listen_report"] or {}).get("song_likeness") or {}).get("score") or 0.0),
                "groove": float(((winner["listen_report"] or {}).get("groove") or {}).get("score") or 0.0),
                "transition": float(((winner["listen_report"] or {}).get("transition") or {}).get("score") or 0.0),
                "mix_sanity": float(((winner["listen_report"] or {}).get("mix_sanity") or {}).get("score") or 0.0),
                "structure": float(((winner["listen_report"] or {}).get("structure") or {}).get("score") or 0.0),
                "boundary_recovery": _song_likeness_aggregate_metric(winner["listen_report"] or {}, "boundary_recovery"),
                "role_plausibility": _song_likeness_aggregate_metric(winner["listen_report"] or {}, "role_plausibility"),
                "gating_status": str(((winner["listen_report"] or {}).get("gating") or {}).get("status") or ""),
                "seam_snapshot": _extract_transition_seam_snapshot(winner["listen_report"] or {}),
                "parent_balance": winner["parent_balance"],
                "candidate_dir": str(Path(best.get("master_wav_path") or "").parent),
            }
        selection_path = outdir / "fusion_selection.json"
        _write_json(selection_path, selection_payload)

        if genre or bpm or key:
            print("Note: v1 render currently ignores target genre/BPM/key overrides and uses analyzed parent timing.")
        print("Render outputs (pro mode):")
        print(f"  candidates evaluated: {len(candidates)}")
        print(f"  winner policy: {selection_policy}")
        if winner is None:
            print("  promotion blocked: no candidate met hard pass+quality floors")
            print(
                "  required floors: "
                f"song_likeness >= {min_song_likeness:.1f}, "
                f"groove >= {min_groove:.1f}, "
                f"structure >= {min_structure:.1f}, "
                f"boundary_recovery >= {min_boundary_recovery:.2f} (if present), "
                f"role_plausibility >= {min_role_plausibility:.2f} (if present), "
                "gate=pass"
            )
            print(f"  selection report: {selection_path}")
            return 2

        print(f"  winner mode: {winner['mode']}")
        print(f"  winner selection score: {winner['selection_score']}")
        print(f"  winner overall score: {(winner['listen_report'] or {}).get('overall_score')}")
        print(f"  winner parent balance: {winner['parent_balance']}")
        print(f"  raw wav: {outdir / 'child_raw.wav'}")
        print(f"  master wav: {outdir / 'child_master.wav'}")
        print(f"  master mp3: {(outdir / 'child_master.mp3') if (outdir / 'child_master.mp3').exists() else 'not written (ffmpeg unavailable)'}")
        print(f"  manifest: {outdir / 'render_manifest.json'}")
        print(f"  selection report: {selection_path}")
        return 0

    plan = build_arrangement_plan(song_a, song_b, arrangement_mode=arrangement_mode)
    result = _render_fusion_plan_candidate(song_a, song_b, plan, outdir)
    if genre or bpm or key:
        print("Note: v1 render currently ignores target genre/BPM/key overrides and uses analyzed parent timing.")
    print("Render outputs:")
    print(f"  raw wav: {result['raw_wav_path']}")
    print(f"  master wav: {result['master_wav_path']}")
    print(f"  master mp3: {result['master_mp3_path'] or 'not written (ffmpeg unavailable)'}")
    print(f"  manifest: {result['render_manifest_path']}")
    return 0


def prototype(song_a: str, song_b: str, output_dir: str, stems_dir: Optional[str] = None, arrangement_mode: str = "baseline") -> int:
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
    arrangement = build_arrangement_plan(song_a_obj, song_b_obj, arrangement_mode=arrangement_mode).to_dict()

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


def _split_labeled_input(raw: str) -> tuple[str | None, str]:
    text = str(raw).strip()
    for delimiter in ("=", "::"):
        if delimiter not in text:
            continue
        label, candidate = text.split(delimiter, 1)
        label = label.strip()
        candidate = candidate.strip()
        if not label or not candidate:
            continue
        if candidate.startswith(("/", "./", "../", "~/")):
            return label, candidate
        if len(candidate) >= 3 and candidate[1] == ":" and candidate[2] in {"/", "\\"}:
            return label, candidate
    return None, text


def _short_label(path_str: Optional[str], fallback: str) -> str:
    if not path_str:
        return fallback
    return Path(path_str).name or fallback


def _stable_case_id(path_str: str, *, explicit_label: str | None = None) -> str:
    resolved = str(Path(path_str).expanduser().resolve())
    identity = f"{explicit_label}::{resolved}" if explicit_label else resolved
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]


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

    explicit_labels = [str(item.get("explicit_case_label") or "").strip() for item in labeled]
    unlabeled_indexes = [index for index, value in enumerate(explicit_labels) if not value]

    resolved_labels: list[str | None] = [value or None for value in explicit_labels]
    if unlabeled_indexes:
        depths = [1] * len(unlabeled_indexes)
        max_depths = []
        for index in unlabeled_indexes:
            item = labeled[index]
            path = Path(str(item.get("input_path") or "")).expanduser().resolve()
            parts = list(path.parts)
            if parts and parts[0] == path.anchor:
                parts = parts[1:]
            max_depths.append(max(1, len(parts)))

        while True:
            labels = [_path_tail_label(str(labeled[index].get("input_path") or ""), depth) for index, depth in zip(unlabeled_indexes, depths)]
            collisions: dict[str, list[int]] = {}
            for local_index, label in enumerate(labels):
                collisions.setdefault(label, []).append(local_index)
            duplicate_groups = [indexes for indexes in collisions.values() if len(indexes) > 1]
            if not duplicate_groups:
                for local_index, label in enumerate(labels):
                    resolved_labels[unlabeled_indexes[local_index]] = label
                break

            advanced = False
            for indexes in duplicate_groups:
                for local_index in indexes:
                    if depths[local_index] < max_depths[local_index]:
                        depths[local_index] += 1
                        advanced = True
            if not advanced:
                for local_index, label in enumerate(labels):
                    index = unlabeled_indexes[local_index]
                    resolved_labels[index] = f"{label}#{str(labeled[index].get('case_id') or '')[:6]}" if len(collisions.get(label, [])) > 1 else label
                break

    final_collisions: dict[str, list[int]] = {}
    for index, label in enumerate(resolved_labels):
        final_collisions.setdefault(str(label or ""), []).append(index)
    for label, indexes in final_collisions.items():
        if len(indexes) < 2:
            continue
        for index in indexes:
            resolved_labels[index] = f"{label}#{str(labeled[index].get('case_id') or '')[:6]}"

    for item, label in zip(labeled, resolved_labels):
        item["display_label"] = str(label or item.get("input_label") or item.get("input_path") or "case")
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

    explicit_case_label, raw_candidate = _split_labeled_input(input_path)
    raw_input = str(Path(raw_candidate).expanduser())
    path = Path(raw_candidate).expanduser().resolve()
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
        listen_candidates = sorted(path.glob("*listen*.json"), key=lambda candidate: candidate.stat().st_mtime, reverse=True)
        for listen_candidate in listen_candidates:
            payload = _load_json(listen_candidate)
            if _is_listen_report_payload(payload):
                return {
                    "input_path": str(path),
                    "input_label": explicit_case_label or _short_label(raw_input, path.name or "render_output"),
                    "explicit_case_label": explicit_case_label,
                    "case_id": _stable_case_id(str(path), explicit_label=explicit_case_label),
                    "report_origin": "render_output",
                    "resolved_audio_path": payload.get("source_path") or _pick_render_audio_path(manifest, render_manifest_path),
                    "render_manifest_path": str(render_manifest_path),
                    "report": payload,
                }
        analyzed_path = Path(_pick_render_audio_path(manifest, render_manifest_path))
        report_origin = "render_output"
    elif path.suffix.lower() == ".json":
        payload = _load_json(path)
        if _is_listen_report_payload(payload):
            return {
                "input_path": str(path),
                "input_label": explicit_case_label or _short_label(raw_input, path.name or "listen_report"),
                "explicit_case_label": explicit_case_label,
                "case_id": _stable_case_id(str(path), explicit_label=explicit_case_label),
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
        "input_label": explicit_case_label or _short_label(raw_input, path.name or report_origin),
        "explicit_case_label": explicit_case_label,
        "case_id": _stable_case_id(str(path), explicit_label=explicit_case_label),
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


def listen(track: str, output: Optional[str], score_only: bool = False) -> int:
    analyze_audio = _get_analyze_audio_file()
    evaluate = _get_evaluate_song()
    track_path = _resolve_existing_audio_path(track, "track")
    song = analyze_audio(track_path)
    report = evaluate(song).to_dict()

    resolved_output = _resolve_output_path(output)
    if resolved_output:
        _write_json(resolved_output, report)
        print(f"Wrote listen report: {resolved_output}")
    elif not score_only:
        print(json.dumps(report, indent=2, sort_keys=True))

    if score_only:
        print(f"{float(report['overall_score']):.1f}")
        return 0

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


def _slugify_stem(path_str: str, fallback: str) -> str:
    stem = Path(path_str).expanduser().stem.lower()
    slug = ''.join(ch if ch.isalnum() else '_' for ch in stem).strip('_')
    return slug or fallback



def _stable_auto_shortlist_output_root(song_a: str, song_b: str, *, batch_size: int, shortlist: int, variant_mode: str) -> Path:
    resolved = [
        str(Path(song_a).expanduser().resolve()),
        str(Path(song_b).expanduser().resolve()),
        str(int(batch_size)),
        str(int(shortlist)),
        str(variant_mode or "safe"),
    ]
    digest = hashlib.sha1("||".join(resolved).encode("utf-8")).hexdigest()[:10]
    song_a_slug = _slugify_stem(song_a, "song_a")
    song_b_slug = _slugify_stem(song_b, "song_b")
    return (Path("runs") / "auto_shortlist" / f"{song_a_slug}__{song_b_slug}__{digest}").resolve()


def _listener_policy_snapshot() -> dict[str, Any]:
    return {
        "policy_version": AUTO_SHORTLIST_SCHEMA_VERSION,
        "weighted_components": dict(LISTENER_AGENT_COMPONENT_WEIGHTS),
        "critical_rank_targets": dict(LISTENER_AGENT_CRITICAL_RANK_TARGETS),
        "imbalance_penalty_scale": LISTENER_AGENT_RANK_PENALTY_SCALE,
        "hard_reject_component_floors": dict(LISTENER_AGENT_HARD_REJECT_COMPONENT_FLOORS),
        "survivor_minimums": dict(LISTENER_AGENT_SURVIVOR_MINIMUMS),
    }


@lru_cache(maxsize=1)
def _feedback_learning_snapshot() -> dict[str, Any]:
    feedback_root = (Path(__file__).resolve().parent / "data" / "human_feedback").resolve()
    snapshot_path = feedback_root / "learning_snapshot.json"
    try:
        if snapshot_path.exists():
            return json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        payload = build_feedback_learning_summary(feedback_root, limit=5000)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
    except Exception:
        return {
            "schema_version": "0.1.0",
            "summary": {"render_event_count": 0, "pairwise_event_count": 0},
            "derived_priors": {
                "medley_rejection_pressure": 0.0,
                "groove_rejection_pressure": 0.0,
                "transition_rejection_pressure": 0.0,
                "payoff_upgrade_pressure": 0.0,
                "backbone_reward_pressure": 0.0,
            },
            "timestamped_moments": [],
        }



def _vault_memory_dir() -> Path | None:
    env = os.environ.get("VOCALFUSION_VAULT") or os.environ.get("VF_VAULT_PATH")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser().resolve())
    repo_root = Path(__file__).resolve().parent
    if len(repo_root.parents) >= 2:
        candidates.append((repo_root.parents[1] / "VocalFusionVault").resolve())
    for candidate in candidates:
        memory_dir = candidate / "memory"
        if memory_dir.exists() and memory_dir.is_dir():
            return memory_dir
    return None



def _append_auto_shortlist_memory_log(report: dict[str, Any]) -> Path | None:
    memory_dir = _vault_memory_dir()
    if memory_dir is None:
        return None
    memory_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone()
    path = memory_dir / f"{stamp.strftime('%Y-%m-%d')}.md"
    if not path.exists():
        path.write_text(f"# {stamp.strftime('%Y-%m-%d')}\n\n", encoding="utf-8")

    job = dict(report.get("job") or {})
    counts = (((report.get("listener_agent_report") or {}).get("counts") or {}))
    summary = list(report.get("summary") or [])
    lines = [
        "",
        f"## Fusion run — {stamp.isoformat(timespec='seconds')}",
        f"- Inputs: `{Path(str(job.get('song_a') or '')).name}` + `{Path(str(job.get('song_b') or '')).name}`",
        f"- Output root: `{job.get('output_root')}`",
        f"- Candidate batch: {job.get('batch_size')} | shortlist target: {job.get('shortlist')} | variant mode: `{job.get('variant_mode')}`",
        f"- Listen/gate counts: survivors={counts.get('survivors', 0)}, borderline={counts.get('borderline', 0)}, rejected={counts.get('rejected', 0)}",
        "- Listen basis: overall score + component scores (structure, groove, energy arc, transition, coherence, mix sanity, song-likeness) + gate verdict + hard/soft fail reasons.",
    ]
    if summary:
        lines.append("- Summary:")
        lines.extend(f"  - {item}" for item in summary)
    lines.append("- Candidate results:")
    for row in report.get("candidates") or []:
        component_scores = dict(row.get("component_scores") or {})
        component_blob = ", ".join(f"{key}={value}" for key, value in sorted(component_scores.items())) or "no component scores"
        reasons = "; ".join((row.get("hard_fail_reasons") or [])[:2] or (row.get("top_reasons") or [])[:2]) or "n/a"
        lines.append(
            f"  - {row.get('candidate_id')}: decision={row.get('decision')} overall={row.get('overall_score')} rank={row.get('listener_rank')} verdict={row.get('verdict')} | components: {component_blob} | reasons: {reasons}"
        )
    pruning = dict(report.get("pruning") or {})
    if pruning:
        lines.append(
            f"- Pruning: enabled={pruning.get('enabled')} deleted={pruning.get('deleted_candidate_count', 0)} kept={pruning.get('kept_candidate_ids', [])}"
        )
    path.open("a", encoding="utf-8").write("\n".join(lines) + "\n")
    return path



def _apply_feedback_learning_bias(result: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    snapshot = _feedback_learning_snapshot()
    priors = dict(snapshot.get("derived_priors") or {})
    song_metrics = (((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {})
    groove_score = float(((report.get("groove") or {}).get("score", 0.0)) or 0.0)
    transition_score = float(((report.get("transition") or {}).get("score", 0.0)) or 0.0)
    energy_arc_score = float(((report.get("energy_arc") or {}).get("score", 0.0)) or 0.0)

    medley_pressure = float(priors.get("medley_rejection_pressure", 0.0) or 0.0)
    groove_pressure = float(priors.get("groove_rejection_pressure", 0.0) or 0.0)
    transition_pressure = float(priors.get("transition_rejection_pressure", 0.0) or 0.0)
    payoff_pressure = float(priors.get("payoff_upgrade_pressure", 0.0) or 0.0)
    backbone_pressure = float(priors.get("backbone_reward_pressure", 0.0) or 0.0)

    medley_risk = float(song_metrics.get("full_mix_medley_risk", 0.0) or 0.0)
    backbone_continuity = float(song_metrics.get("backbone_continuity", 0.5) or 0.5)
    groove_gap = max(0.0, min(1.0, (68.0 - groove_score) / 22.0))
    transition_gap = max(0.0, min(1.0, (68.0 - transition_score) / 22.0))
    payoff_gap = max(0.0, min(1.0, (66.0 - energy_arc_score) / 24.0))

    penalty = 8.0 * medley_pressure * medley_risk + 6.5 * groove_pressure * groove_gap + 5.0 * transition_pressure * transition_gap + 4.5 * payoff_pressure * payoff_gap
    bonus = 5.0 * backbone_pressure * backbone_continuity
    adjusted_rank = round(float(result.get("listener_rank", 0.0)) + bonus - penalty, 1)

    result["base_listener_rank"] = result.get("listener_rank")
    result["listener_rank"] = adjusted_rank
    result["feedback_learning"] = {
        "derived_priors": priors,
        "penalty": round(penalty, 3),
        "bonus": round(bonus, 3),
        "adjusted_listener_rank": adjusted_rank,
        "signals": {
            "medley_risk": round(medley_risk, 3),
            "backbone_continuity": round(backbone_continuity, 3),
            "groove_gap": round(groove_gap, 3),
            "transition_gap": round(transition_gap, 3),
            "payoff_gap": round(payoff_gap, 3),
        },
    }
    return result



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
    result = {
        "label": item.get("input_label"),
        "case_id": item.get("case_id"),
        "input_path": item.get("input_path"),
        "report_origin": item.get("report_origin"),
        "resolved_audio_path": item.get("resolved_audio_path"),
        "render_manifest_path": item.get("render_manifest_path"),
        "overall_score": round(overall, 1),
        "verdict": verdict,
        "gate_status": gate_status,
        "raw_gating_status": gate_status,
        "decision": decision,
        "gate_lane": decision,
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
        "metadata": dict(item.get("metadata") or {}),
    }
    return _apply_feedback_learning_bias(result, report)


def _build_listener_agent_report_from_resolved(
    resolved: list[dict[str, Any]],
    *,
    inputs: list[str],
    shortlist: int = 3,
) -> dict[str, Any]:
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
            "policy_version": _listener_policy_snapshot()["policy_version"],
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



def _build_listener_agent_report(inputs: list[str], shortlist: int = 3) -> dict[str, Any]:
    if not inputs:
        raise CliError("listener-agent requires at least one input")
    resolved = [_resolve_compare_input(path) for path in inputs]
    return _build_listener_agent_report_from_resolved(resolved, inputs=inputs, shortlist=shortlist)


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



def _evaluate_auto_shortlist_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    run_dir = str(candidate["outdir"])
    resolved = _resolve_compare_input(run_dir)
    listen_report_path = Path(run_dir) / "listen_report.json"
    _write_json(listen_report_path, resolved["report"])
    resolved["metadata"] = {
        "candidate_id": candidate.get("candidate_id"),
        "variant_config": candidate.get("variant_config") or {},
        "run_dir": run_dir,
        "arrangement_plan_path": candidate.get("arrangement_plan_path"),
        "listen_report_path": str(listen_report_path),
    }
    assessment = _listener_agent_case_assessment(resolved)
    candidate.update(
        {
            "listen_report_path": str(listen_report_path),
            "listen_report": resolved["report"],
            "assessment": assessment,
        }
    )
    return candidate



def _build_auto_shortlist_report(
    *,
    song_a: str,
    song_b: str,
    output_root: Path,
    batch_size: int,
    shortlist: int,
    variant_mode: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved_inputs = []
    for candidate in candidates:
        listen_path = str(candidate.get("listen_report_path") or candidate["outdir"])
        resolved = _resolve_compare_input(f"{candidate['candidate_id']}={listen_path}")
        resolved["metadata"] = {
            "candidate_id": candidate.get("candidate_id"),
            "variant_config": candidate.get("variant_config") or {},
            "run_dir": candidate.get("outdir"),
            "arrangement_plan_path": candidate.get("arrangement_plan_path"),
            "listen_report_path": candidate.get("listen_report_path"),
        }
        resolved_inputs.append(resolved)

    listener_report = _build_listener_agent_report_from_resolved(
        resolved_inputs,
        inputs=[str(candidate.get("outdir")) for candidate in candidates],
        shortlist=shortlist,
    )
    survivors = list(listener_report.get("survivors") or [])
    borderlines = list(listener_report.get("borderline") or [])
    pairwise_pool = survivors or borderlines[: max(shortlist, 0)]
    pairwise_inputs = [f"{row['label']}={row['metadata'].get('listen_report_path') or row['input_path']}" for row in pairwise_pool]
    pairwise_benchmark = _build_listen_benchmark(pairwise_inputs) if len(pairwise_inputs) >= 2 else None

    ranking_index: dict[str, dict[str, Any]] = {}
    if pairwise_benchmark:
        ranking_index = {str(row["label"]): row for row in pairwise_benchmark.get("ranking") or []}

    candidate_index = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    recommended_shortlist: list[dict[str, Any]] = []
    if survivors:
        ranked_survivors = list(survivors)
        if ranking_index:
            ranked_survivors.sort(
                key=lambda row: (
                    -int((ranking_index.get(str(row["label"])) or {}).get("wins", 0)),
                    -float((ranking_index.get(str(row["label"])) or {}).get("net_score_delta", 0.0)),
                    -float(row.get("listener_rank", 0.0)),
                    row.get("label") or "",
                )
            )
        for rank, row in enumerate(ranked_survivors[: max(shortlist, 0)], start=1):
            meta = dict(row.get("metadata") or {})
            recommended_shortlist.append(
                {
                    "rank": rank,
                    "candidate_id": meta.get("candidate_id") or row.get("label"),
                    "label": row.get("label"),
                    "decision": row.get("decision"),
                    "listener_rank": row.get("listener_rank"),
                    "overall_score": row.get("overall_score"),
                    "verdict": row.get("verdict"),
                    "run_dir": meta.get("run_dir") or row.get("input_path"),
                    "audio_path": row.get("resolved_audio_path"),
                    "listen_report_path": meta.get("listen_report_path") or row.get("input_path"),
                    "variant_config": meta.get("variant_config") or {},
                    "pairwise": ranking_index.get(str(row.get("label"))),
                    "top_reasons": row.get("top_reasons") or [],
                    "top_fixes": row.get("top_fixes") or [],
                }
            )

    closest_misses: list[dict[str, Any]] = []
    if not recommended_shortlist and borderlines:
        for row in borderlines[: max(shortlist, 0)]:
            meta = dict(row.get("metadata") or {})
            closest_misses.append(
                {
                    "candidate_id": meta.get("candidate_id") or row.get("label"),
                    "label": row.get("label"),
                    "decision": row.get("decision"),
                    "listener_rank": row.get("listener_rank"),
                    "overall_score": row.get("overall_score"),
                    "verdict": row.get("verdict"),
                    "run_dir": meta.get("run_dir") or row.get("input_path"),
                    "audio_path": row.get("resolved_audio_path"),
                    "listen_report_path": meta.get("listen_report_path") or row.get("input_path"),
                    "variant_config": meta.get("variant_config") or {},
                    "top_reasons": row.get("top_reasons") or [],
                    "top_fixes": row.get("top_fixes") or [],
                }
            )

    summary = [
        f"Generated {len(candidates)} candidate renders for automatic shortlist evaluation.",
        f"Listener gate result: {listener_report['counts']['survivors']} survivors, {listener_report['counts']['borderline']} borderline, {listener_report['counts']['rejected']} rejected.",
    ]
    if recommended_shortlist:
        summary.append(
            f"Shortlisted {len(recommended_shortlist)} survivor(s) for human review; top winner is {recommended_shortlist[0]['candidate_id']}."
        )
    else:
        summary.append("No candidates survived the automatic gate.")
    if pairwise_benchmark:
        summary.append(
            f"Pairwise ranking winner in review pool: {pairwise_benchmark.get('winner')} across {len(pairwise_benchmark.get('comparisons') or [])} comparisons."
        )

    candidate_rows = []
    for candidate in candidates:
        assessment = dict(candidate.get("assessment") or {})
        candidate_rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "run_dir": candidate.get("outdir"),
                "render_manifest_path": candidate.get("render_manifest_path"),
                "arrangement_plan_path": candidate.get("arrangement_plan_path"),
                "audio_path": candidate.get("master_mp3_path") or candidate.get("master_wav_path"),
                "raw_audio_path": candidate.get("master_wav_path"),
                "listen_report_path": candidate.get("listen_report_path"),
                "variant_config": candidate.get("variant_config") or {},
                "decision": assessment.get("decision"),
                "gate_status": assessment.get("gate_status"),
                "listener_rank": assessment.get("listener_rank"),
                "base_listener_rank": assessment.get("base_listener_rank"),
                "overall_score": assessment.get("overall_score"),
                "verdict": assessment.get("verdict"),
                "hard_fail_reasons": assessment.get("hard_fail_reasons") or [],
                "top_reasons": assessment.get("top_reasons") or [],
                "top_fixes": assessment.get("top_fixes") or [],
                "component_scores": assessment.get("component_scores") or {},
                "feedback_learning": assessment.get("feedback_learning") or {},
            }
        )

    return {
        "schema_version": AUTO_SHORTLIST_SCHEMA_VERSION,
        "job": {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "song_a": song_a,
            "song_b": song_b,
            "output_root": str(output_root),
            "batch_size": int(batch_size),
            "shortlist": int(shortlist),
            "selection_policy": "survivor_then_pairwise_v1",
            "pairwise_pool_policy": "survivors_only_else_borderline_fallback",
            "variant_mode": variant_mode,
            "gate_version": AUTO_SHORTLIST_SCHEMA_VERSION,
            "policy_version": _listener_policy_snapshot()["policy_version"],
        },
        "policy_snapshot": _listener_policy_snapshot(),
        "feedback_learning": _feedback_learning_snapshot(),
        "listener_agent_report": listener_report,
        "candidates": candidate_rows,
        "gated_groups": {
            "survivors": [str(row.get("metadata", {}).get("candidate_id") or row.get("label")) for row in survivors],
            "borderline": [str(row.get("metadata", {}).get("candidate_id") or row.get("label")) for row in borderlines],
            "rejected": [str(row.get("metadata", {}).get("candidate_id") or row.get("label")) for row in listener_report.get("rejected") or []],
        },
        "pairwise_pool": {
            "candidate_ids": [str(row.get("metadata", {}).get("candidate_id") or row.get("label")) for row in pairwise_pool],
            "benchmark": pairwise_benchmark,
            "winner": pairwise_benchmark.get("winner") if pairwise_benchmark else (recommended_shortlist[0]["candidate_id"] if recommended_shortlist else None),
        },
        "recommended_shortlist": recommended_shortlist,
        "closest_misses": closest_misses,
        "pruning": {
            "enabled": False,
            "kept_candidate_ids": [row["candidate_id"] for row in recommended_shortlist],
            "deleted_candidate_ids": [],
            "deleted_candidate_count": 0,
        },
        "summary": summary,
    }



def _apply_auto_shortlist_pruning(
    output_root: Path,
    report: dict[str, Any],
    *,
    delete_non_survivors: bool,
) -> dict[str, Any]:
    pruning = dict(report.get("pruning") or {})
    pruning["enabled"] = bool(delete_non_survivors)
    if not delete_non_survivors:
        report["pruning"] = pruning
        return report

    keep_ids = {str(row.get("candidate_id")) for row in report.get("recommended_shortlist") or [] if row.get("candidate_id")}
    deleted_ids: list[str] = []
    candidate_rows = list(report.get("candidates") or [])
    for row in candidate_rows:
        candidate_id = str(row.get("candidate_id") or "")
        run_dir_value = row.get("run_dir")
        if not candidate_id or candidate_id in keep_ids or not run_dir_value:
            continue
        run_dir = Path(str(run_dir_value)).expanduser().resolve()
        try:
            run_dir.relative_to(output_root.resolve())
        except ValueError:
            continue
        if run_dir.exists() and run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
        deleted_ids.append(candidate_id)
        row["artifacts_pruned"] = True
        row["audio_path"] = None
        row["raw_audio_path"] = None
        row["listen_report_path"] = None
        row["render_manifest_path"] = None
        row["arrangement_plan_path"] = None

    for collection_key in ("closest_misses",):
        for row in list(report.get(collection_key) or []):
            if str(row.get("candidate_id") or "") in deleted_ids:
                row["artifacts_pruned"] = True
                row["audio_path"] = None
                row["listen_report_path"] = None

    pruning.update(
        {
            "kept_candidate_ids": sorted(keep_ids),
            "deleted_candidate_ids": deleted_ids,
            "deleted_candidate_count": len(deleted_ids),
        }
    )
    report["pruning"] = pruning
    if deleted_ids:
        report.setdefault("summary", []).append(
            f"Pruned {len(deleted_ids)} non-survivor candidate run(s) after gating so only shortlist survivors remain on disk."
        )
    return report



def auto_shortlist_fusion(
    track1: str,
    track2: str,
    output_root: Optional[str],
    *,
    batch_size: int = AUTO_SHORTLIST_DEFAULT_BATCH_SIZE,
    shortlist: int = AUTO_SHORTLIST_DEFAULT_SHORTLIST,
    variant_mode: str = "safe",
    delete_non_survivors: bool = True,
    arrangement_mode: str = "baseline",
) -> int:
    analyze_audio = _get_analyze_audio_file()
    _, build_arrangement_plan = _get_planner_functions()
    track1_path = _resolve_existing_audio_path(track1, "track1")
    track2_path = _resolve_existing_audio_path(track2, "track2")
    batch_size = max(1, min(int(batch_size), AUTO_SHORTLIST_MAX_BATCH_SIZE))
    shortlist = max(1, int(shortlist))
    resolved_output_root = _resolve_output_path(
        output_root,
        default_path=_stable_auto_shortlist_output_root(track1_path, track2_path, batch_size=batch_size, shortlist=shortlist, variant_mode=variant_mode),
        default_filename="auto_shortlist_report.json",
    )
    output_root_path = resolved_output_root.parent if resolved_output_root.suffix else Path(resolved_output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    song_a = analyze_audio(track1_path)
    song_b = analyze_audio(track2_path)
    base_plan = build_arrangement_plan(song_a, song_b, arrangement_mode=arrangement_mode)
    variant_configs = _build_auto_shortlist_variant_configs(base_plan, batch_size, variant_mode=variant_mode)

    candidates: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for index, variant in enumerate(variant_configs, start=1):
        candidate_id = f"candidate_{index:03d}"
        signature = json.dumps(variant.get("swaps") or [], sort_keys=True)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        run_dir = output_root_path / candidate_id
        candidate = _render_fusion_candidate(song_a, song_b, base_plan, run_dir, candidate_id=candidate_id, variant_config=variant)
        candidates.append(_evaluate_auto_shortlist_candidate(candidate))

    report = _build_auto_shortlist_report(
        song_a=track1_path,
        song_b=track2_path,
        output_root=output_root_path,
        batch_size=batch_size,
        shortlist=shortlist,
        variant_mode=variant_mode,
        candidates=candidates,
    )
    report = _apply_auto_shortlist_pruning(output_root_path, report, delete_non_survivors=delete_non_survivors)
    report_path = output_root_path / "auto_shortlist_report.json"
    _write_json(report_path, report)
    memory_path = _append_auto_shortlist_memory_log(report)
    print(f"Wrote auto-shortlist report: {report_path}")
    if memory_path is not None:
        print(f"Appended fusion memory log: {memory_path}")
    if report["recommended_shortlist"]:
        winner = report["recommended_shortlist"][0]
        print(f"Top survivor: {winner['candidate_id']} (overall={winner['overall_score']:.1f}, rank={winner['listener_rank']:.1f})")
    else:
        print("No survivors cleared the automatic gate.")
    for line in report.get("summary") or []:
        print(f"- {line}")
    return 0



def distill_feedback_learning(output: Optional[str] = None) -> int:
    feedback_root = (Path(__file__).resolve().parent / "data" / "human_feedback").resolve()
    output_path = Path(output).expanduser().resolve() if output else (feedback_root / "learning_snapshot.json")
    payload = write_feedback_learning_summary(feedback_root, output_path, limit=5000)
    _feedback_learning_snapshot.cache_clear()
    print(f"Wrote feedback learning snapshot: {output_path}")
    print(json.dumps(payload.get("derived_priors") or {}, indent=2, sort_keys=True))
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
    change_dispatch: Optional[dict[str, Any]] = None,
    test_dispatch: Optional[dict[str, Any]] = None,
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
            change_dispatch=change_dispatch,
            test_dispatch=test_dispatch,
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
  python3 ai_dj.py auto-shortlist-fusion song_a.wav song_b.wav --output runs/auto_shortlist/demo --batch-size 4 --shortlist 2
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
    listen_parser.add_argument("--score-only", action="store_true", help="Print only the overall score (0-100) for quick scripting")

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

    auto_shortlist_parser = subparsers.add_parser("auto-shortlist-fusion", help="Render several candidate fusions, gate them automatically, and only keep survivors for human listening")
    auto_shortlist_parser.add_argument("track1", help="Path to first track")
    auto_shortlist_parser.add_argument("track2", help="Path to second track")
    auto_shortlist_parser.add_argument("--output", "-o", help="Output directory (or report path inside one) for shortlist artifacts")
    auto_shortlist_parser.add_argument("--batch-size", type=int, default=AUTO_SHORTLIST_DEFAULT_BATCH_SIZE, help="How many candidate variants to generate")
    auto_shortlist_parser.add_argument("--shortlist", type=int, default=AUTO_SHORTLIST_DEFAULT_SHORTLIST, help="Maximum number of survivors to surface for human review")
    auto_shortlist_parser.add_argument("--variant-mode", default="safe", help="Variant generation mode (currently: safe)")
    auto_shortlist_parser.add_argument("--arrangement-mode", default="baseline", choices=["baseline", "adaptive"], help="Arrangement planning mode")
    auto_shortlist_parser.add_argument("--keep-non-survivors", action="store_true", help="Do not delete rejected/non-shortlisted candidate run folders after gating")

    feedback_learning_parser = subparsers.add_parser("distill-feedback-learning", help="Distill stored human feedback into a stable learning snapshot used by shortlist ranking")
    feedback_learning_parser.add_argument("--output", "-o", help="Output JSON path for the distilled learning snapshot")

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
    closed_loop_parser.add_argument("--change-dispatch", help="Optional JSON dispatch spec for the change step")
    closed_loop_parser.add_argument("--test-dispatch", help="Optional JSON dispatch spec for the test step")

    fus_parser = subparsers.add_parser("fusion", help="Render a first-pass fused audio prototype")
    fus_parser.add_argument("track1", help="Path to first track")
    fus_parser.add_argument("track2", help="Path to second track")
    fus_parser.add_argument("--genre", "-g", help="Target genre for fusion (accepted but not yet applied)")
    fus_parser.add_argument("--bpm", "-b", type=int, help="Target BPM (accepted but not yet applied)")
    fus_parser.add_argument("--key", "-k", help="Target musical key")
    fus_parser.add_argument("--output", "-o", help="Output directory for render artifacts")
    fus_parser.add_argument("--arrangement-mode", default="baseline", choices=["baseline", "adaptive", "pro"], help="Arrangement planning mode (pro runs adaptive+baseline and promotes best)")

    proto_parser = subparsers.add_parser("prototype", help="Generate first-pass two-song prototype artifacts")
    proto_parser.add_argument("song_a", help="Path to parent song A")
    proto_parser.add_argument("song_b", help="Path to parent song B")
    proto_parser.add_argument("--output-dir", "-o", required=True, help="Directory to write prototype artifacts")
    proto_parser.add_argument("--stems-dir", help="Optional directory for stem outputs")
    proto_parser.add_argument("--arrangement-mode", default="baseline", choices=["baseline", "adaptive"], help="Arrangement planning mode")

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
            return fusion(args.track1, args.track2, args.genre, args.bpm, args.key, args.output, args.arrangement_mode)
        if args.command == "prototype":
            return prototype(args.song_a, args.song_b, args.output_dir, args.stems_dir, args.arrangement_mode)
        if args.command == "listen":
            return listen(args.track, args.output, args.score_only)
        if args.command == "compare-listen":
            return compare_listen(args.left, args.right, args.output)
        if args.command == "benchmark-listen":
            return benchmark_listen(args.inputs, args.output)
        if args.command == "listener-agent":
            return listener_agent(args.inputs, args.output, args.shortlist)
        if args.command == "auto-shortlist-fusion":
            return auto_shortlist_fusion(
                args.track1,
                args.track2,
                args.output,
                batch_size=args.batch_size,
                shortlist=args.shortlist,
                variant_mode=args.variant_mode,
                delete_non_survivors=not bool(args.keep_non_survivors),
                arrangement_mode=args.arrangement_mode,
            )
        if args.command == "distill-feedback-learning":
            return distill_feedback_learning(args.output)
        if args.command == "closed-loop":
            change_dispatch = None
            test_dispatch = None
            if args.change_dispatch or args.test_dispatch:
                try:
                    from scripts.closed_loop_listener_runner import _read_dispatch_spec
                except ModuleNotFoundError as exc:
                    raise CliError(f"Unable to import closed-loop runner: {exc}") from exc
                if args.change_dispatch:
                    change_dispatch = _read_dispatch_spec(args.change_dispatch, label="change")
                if args.test_dispatch:
                    test_dispatch = _read_dispatch_spec(args.test_dispatch, label="test")
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
                change_dispatch,
                test_dispatch,
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
