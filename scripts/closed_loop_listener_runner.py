#!/usr/bin/env python3
"""Run a bounded listener-driven improvement loop for one fusion pair.

This runner does not pretend to guarantee chart-grade output. It orchestrates a real
closed loop around the current VocalFusion stack:

1. render a fusion candidate
2. evaluate it against good references
3. write a structured improvement brief
4. optionally call an external code-change command
5. optionally run tests
6. keep track of best iteration and stop on plateau or gate success

The code-change step is intentionally externalized via `--change-command` so the
loop can be used with different patching strategies (manual patch scripts, coding
agents, ACP harnesses, etc.) without hard-wiring one editing runtime into the repo.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai_dj
from scripts.listen_feedback_loop import build_feedback_brief
from scripts.reference_input_normalizer import normalize_reference_inputs

PYTHON = ROOT / ".venv" / "bin" / "python"
CLOSED_LOOP_SCHEMA_VERSION = "0.2.0"
CHANGE_COMMAND_CONTEXT_SCHEMA_VERSION = "0.1.0"
DISPATCH_SPEC_SCHEMA_VERSION = "0.1.0"
COMMAND_TEMPLATE_FIELDS: list[tuple[str, str]] = [
    ("iteration", "1-indexed loop iteration number."),
    ("iteration_dir", "Directory for this iteration's artifacts."),
    ("render_dir", "Render output directory for this iteration."),
    ("candidate_input", "Candidate input passed back into listen/compare helpers (normally the render dir)."),
    ("feedback_json", "Path to the structured listen feedback brief JSON for this iteration."),
    ("change_context_json", "Path to the structured change-command context JSON written before the change command runs."),
    ("change_request_md", "Path to the human-readable Markdown change request written before the change command runs."),
    ("output_root", "Closed-loop output root containing all iteration folders."),
    ("best_score", "Best listener overall score seen so far."),
    ("quality_gate", "Configured closed-loop quality gate."),
    ("quality_gate_status_json", "Structured JSON string describing raw and reference-weighted gate status for the current candidate."),
    ("song_a", "Resolved parent-song A input path."),
    ("song_b", "Resolved parent-song B input path."),
    ("candidate_score", "Current iteration candidate overall listener score."),
    ("candidate_verdict", "Current iteration candidate verdict from the listener report."),
    ("gap_to_target", "Current gap between candidate score and target listener score."),
    ("top_component", "Component for the top-ranked intervention."),
    ("top_problem", "Problem statement for the top-ranked intervention."),
    ("top_actions", "Flattened action text for the top-ranked intervention."),
    ("top_code_targets", "Comma-separated code targets for the top-ranked intervention."),
]


class LoopError(RuntimeError):
    """Closed-loop runner failure."""


def _read_dispatch_spec(path: str | Path, *, label: str) -> dict[str, Any]:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise LoopError(f"{label} dispatch spec not found: {spec_path}")
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoopError(f"{label} dispatch spec must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LoopError(f"{label} dispatch spec must be a JSON object")
    payload = dict(payload)
    payload.setdefault("spec_path", str(spec_path))
    return payload


def _hydrate_dispatch_spec(spec: dict[str, Any] | None, *, label: str) -> dict[str, Any] | None:
    if spec is None:
        return None
    template = str(spec.get("command") or spec.get("template") or "").strip()
    if not template:
        raise LoopError(f"{label} dispatch spec must include command or template")
    timeout_raw = spec.get("timeout")
    timeout = 3600
    if timeout_raw is not None:
        try:
            timeout = int(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise LoopError(f"{label} dispatch spec timeout must be an integer") from exc
        if timeout < 1:
            raise LoopError(f"{label} dispatch spec timeout must be at least 1 second")
    hydrated = {
        "schema_version": str(spec.get("schema_version") or DISPATCH_SPEC_SCHEMA_VERSION),
        "command": template,
        "timeout": timeout,
    }
    if spec.get("spec_path"):
        hydrated["spec_path"] = str(spec.get("spec_path"))
    return hydrated


def _command_template_fields_text() -> str:
    lines = [
        "Closed-loop command template fields",
        "",
        "Use these placeholders in --change-command or --test-command:",
    ]
    for field, description in COMMAND_TEMPLATE_FIELDS:
        lines.append(f"- {{{field}}}: {description}")
    lines.extend([
        "",
        "Examples:",
        '  python3 scripts/closed_loop_listener_runner.py song_a.mp3 song_b.mp3 runs/reference_a runs/reference_b \\\n    --output-root runs/closed_loop/demo \\\n    --change-command "python scripts/your_patch_step.py --context {change_context_json}" \\\n    --test-command "./.venv/bin/python -m pytest -q tests/test_closed_loop_listener_runner.py"',
        "",
        "Tip: prefer {change_context_json} or {change_request_md} over ad-hoc parsing when driving an external patch step.",
    ])
    return "\n".join(lines)


def _python_executable() -> str:
    return str(PYTHON if PYTHON.exists() else sys.executable)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_iteration(song_a: str, song_b: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _python_executable(),
        str(ROOT / "ai_dj.py"),
        "fusion",
        song_a,
        song_b,
        "--output",
        str(output_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise LoopError(f"fusion failed for {output_dir}: {proc.stderr or proc.stdout}")
    return {
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output_dir": str(output_dir),
    }


def _validate_command_template(command_template: str, *, label: str) -> None:
    if not str(command_template).strip():
        raise LoopError(f"{label} command template must not be empty")
    if "\n" in command_template or "\r" in command_template:
        raise LoopError(f"{label} command template must be single-line")
    if "$(" in command_template:
        raise LoopError(f"{label} command template cannot use command substitution")
    if "`" in command_template:
        raise LoopError(f"{label} command template cannot use backtick shell execution")
    for snippet in ("&&", "||", ";", "|", ">", "<"):
        if snippet in command_template:
            raise LoopError(
                f"{label} command template contains unsupported shell operator {snippet!r}; pass a direct command instead"
            )
    try:
        tokens = shlex.split(command_template)
    except ValueError as exc:
        raise LoopError(f"{label} command template is not valid shell-style quoting: {exc}") from exc
    if not tokens:
        raise LoopError(f"{label} command template must contain a command")


def _format_command_template_tokens(command_template: str, context: dict[str, Any], *, label: str) -> list[str]:
    _validate_command_template(command_template, label=label)
    tokens = shlex.split(command_template)
    try:
        return [token.format(**context) for token in tokens]
    except KeyError as exc:
        missing = str(exc).strip("\"")
        raise LoopError(f"{label} command template references unknown field: {missing}") from exc


def _run_command_template(command_template: str, context: dict[str, Any], *, timeout: int = 3600, label: str) -> dict[str, Any]:
    argv = _format_command_template_tokens(command_template, context, label=label)
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    return {
        "command": argv,
        "command_text": shlex.join(argv),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _candidate_item(candidate_input: str) -> dict[str, Any]:
    return ai_dj._resolve_compare_input(candidate_input)


def _candidate_report(candidate_input: str) -> dict[str, Any]:
    return _candidate_item(candidate_input)["report"]


def _candidate_listener_assessment(candidate_input: str) -> dict[str, Any]:
    try:
        return ai_dj._listener_agent_case_assessment(_candidate_item(candidate_input))
    except ai_dj.CliError:
        report = _candidate_report(candidate_input)
        candidate_path = Path(candidate_input).expanduser().resolve()
        fallback_item = {
            "input_path": str(candidate_path),
            "input_label": candidate_path.name or "candidate",
            "case_id": ai_dj._stable_case_id(str(candidate_path)),
            "report_origin": "closed_loop_candidate_report",
            "resolved_audio_path": report.get("source_path"),
            "render_manifest_path": None,
            "report": report,
        }
        return ai_dj._listener_agent_case_assessment(fallback_item)


def _require_feedback_brief_shape(feedback_brief: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(feedback_brief, dict):
        raise LoopError("feedback brief must be a dict")
    schema_version = str(feedback_brief.get("schema_version") or "").strip()
    if not schema_version:
        raise LoopError("feedback brief must declare schema_version")

    goal = feedback_brief.get("goal")
    if not isinstance(goal, dict):
        raise LoopError("feedback brief missing goal dict")
    for required_key in ("target_listener_score", "current_overall_score", "gap_to_target"):
        if required_key not in goal:
            raise LoopError(f"feedback brief goal missing {required_key}")
    for numeric_key in ("target_listener_score", "current_overall_score", "gap_to_target"):
        try:
            float(goal.get(numeric_key))
        except (TypeError, ValueError) as exc:
            raise LoopError(f"feedback brief goal {numeric_key} must be numeric") from exc

    ranked = feedback_brief.get("ranked_interventions")
    if ranked is None:
        raise LoopError("feedback brief missing ranked_interventions")
    if not isinstance(ranked, list):
        raise LoopError("feedback brief ranked_interventions must be a list")

    normalized = dict(feedback_brief)
    normalized["ranked_interventions"] = [item for item in ranked if isinstance(item, dict)]
    for list_key in ("next_code_targets", "planner_feedback_map", "render_feedback_map", "prioritized_execution_plan"):
        value = normalized.get(list_key)
        if value is None:
            normalized[list_key] = []
        elif not isinstance(value, list):
            raise LoopError(f"feedback brief {list_key} must be a list")
    return normalized



def _artifact_record(*, kind: str, path: Path | str, schema_version: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    record: dict[str, Any] = {
        "kind": str(kind),
        "path": resolved,
        "exists": Path(resolved).exists(),
    }
    if schema_version:
        record["schema_version"] = str(schema_version)
    if metadata:
        record["metadata"] = dict(metadata)
    return record



def _build_iteration_artifacts(
    *,
    iteration_dir: Path,
    render_dir: Path,
    feedback_path: Path,
    feedback_brief: dict[str, Any],
    listener_assessment_path: Path | None = None,
    listener_assessment: dict[str, Any] | None = None,
    change_context_path: Path | None = None,
    change_request_path: Path | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "render_output": _artifact_record(
            kind="render_output_dir",
            path=render_dir,
            metadata={"iteration_dir": str(iteration_dir.resolve())},
        ),
        "listen_feedback_brief": _artifact_record(
            kind="listen_feedback_brief",
            path=feedback_path,
            schema_version=str(feedback_brief.get("schema_version") or ""),
            metadata={
                "ranked_intervention_count": len(feedback_brief.get("ranked_interventions") or []),
                "planner_feedback_count": len(feedback_brief.get("planner_feedback_map") or []),
                "render_feedback_count": len(feedback_brief.get("render_feedback_map") or []),
                "prioritized_execution_count": len(feedback_brief.get("prioritized_execution_plan") or []),
            },
        ),
    }
    if listener_assessment_path is not None and listener_assessment is not None:
        artifacts["listener_assessment"] = _artifact_record(
            kind="listener_agent_case_assessment",
            path=listener_assessment_path,
            schema_version="0.1.0",
            metadata={
                "decision": str(listener_assessment.get("decision") or "unknown"),
                "listener_rank": float(listener_assessment.get("listener_rank") or 0.0),
            },
        )
    if change_context_path is not None:
        artifacts["change_context"] = _artifact_record(
            kind="change_command_context",
            path=change_context_path,
            schema_version=CHANGE_COMMAND_CONTEXT_SCHEMA_VERSION,
        )
    if change_request_path is not None:
        artifacts["change_request"] = _artifact_record(
            kind="change_request_markdown",
            path=change_request_path,
        )
    return artifacts


def _top_intervention_summary(item: dict[str, Any] | None) -> dict[str, Any]:
    item = item or {}
    actions = [str(action).strip() for action in (item.get("actions") or []) if str(action).strip()]
    code_targets = [str(target).strip() for target in (item.get("code_targets") or []) if str(target).strip()]
    return {
        "component": str(item.get("component") or ""),
        "gap_vs_references": float(item.get("gap_vs_references") or 0.0),
        "problem": str(item.get("problem") or ""),
        "actions": actions,
        "actions_text": " ".join(actions),
        "code_targets": code_targets,
        "code_targets_csv": ",".join(code_targets),
    }


def _quality_gate_status(*, overall: float, quality_gate: float, feedback_brief: dict[str, Any]) -> dict[str, Any]:
    gate = float(quality_gate)
    passes_overall_gate = float(overall) >= gate
    reference_weighted_score: float | None = None
    passes_reference_weighted_gate = True
    blocking_components: list[str] = []

    diagnostics = (feedback_brief.get("quality_gate_diagnostics") or {}).get("reference_weighted") or {}
    if diagnostics:
        score_value = diagnostics.get("candidate_weighted_score")
        if score_value is not None:
            try:
                reference_weighted_score = float(score_value)
            except (TypeError, ValueError):
                reference_weighted_score = None
        if reference_weighted_score is not None:
            passes_reference_weighted_gate = reference_weighted_score >= gate
            blocking_components = [
                str(item.get("component") or "")
                for item in (diagnostics.get("top_blockers") or [])
                if str(item.get("component") or "").strip()
            ]

    if passes_overall_gate and passes_reference_weighted_gate:
        status = "pass"
        reason = "meets_overall_and_reference_weighted_gate"
    elif passes_overall_gate and not passes_reference_weighted_gate:
        status = "fail"
        reason = "reference_weighted_score_below_gate"
    else:
        status = "fail"
        reason = "overall_score_below_gate"

    return {
        "configured_gate": gate,
        "candidate_overall_score": round(float(overall), 1),
        "candidate_reference_weighted_score": None if reference_weighted_score is None else round(reference_weighted_score, 1),
        "passes_overall_gate": passes_overall_gate,
        "passes_reference_weighted_gate": passes_reference_weighted_gate,
        "status": status,
        "reason": reason,
        "blocking_components": blocking_components[:3],
    }


def _listener_decision_rank(decision: str) -> int:
    order = {
        "reject": 0,
        "borderline": 1,
        "survivor": 2,
    }
    return order.get(str(decision or "").strip().lower(), -1)


def _progress_signal(*, overall: float, listener_assessment: dict[str, Any], quality_gate_status: dict[str, Any]) -> tuple[int, float, float, float]:
    decision_rank = _listener_decision_rank(str(listener_assessment.get("decision") or "unknown"))
    reference_weighted = quality_gate_status.get("candidate_reference_weighted_score")
    if reference_weighted is None:
        reference_weighted = float(overall)
    return (
        decision_rank,
        float(reference_weighted),
        float(listener_assessment.get("listener_rank") or 0.0),
        float(overall),
    )


def _is_meaningful_progress(
    *,
    overall: float,
    listener_assessment: dict[str, Any],
    quality_gate_status: dict[str, Any],
    best_progress_signal: tuple[int, float, float, float] | None,
    min_improvement: float,
) -> tuple[bool, tuple[int, float, float, float]]:
    current = _progress_signal(
        overall=overall,
        listener_assessment=listener_assessment,
        quality_gate_status=quality_gate_status,
    )
    if best_progress_signal is None:
        return True, current

    best_decision, best_weighted, best_rank, best_overall = best_progress_signal
    decision, weighted, listener_rank, current_overall = current
    threshold = float(min_improvement)

    if decision > best_decision:
        return True, current
    if decision < best_decision:
        return False, best_progress_signal
    if weighted >= best_weighted + threshold:
        return True, current
    if listener_rank >= best_rank + threshold:
        return True, current
    if current_overall >= best_overall + threshold:
        return True, current
    return False, best_progress_signal


def _candidate_keep_decision(
    *,
    iteration: int,
    overall: float,
    listener_assessment: dict[str, Any],
    quality_gate_status: dict[str, Any],
    improved_vs_best_before: bool,
    best_iteration_index: int | None,
) -> dict[str, Any]:
    listener_decision = str(listener_assessment.get("decision") or "unknown")
    decision = "reject"
    reasons: list[str] = []

    if improved_vs_best_before:
        decision = "keep"
        if best_iteration_index is None:
            reasons.append("first_candidate_establishes_baseline")
        else:
            reasons.append("meaningful_progress_vs_previous_best")
    else:
        reasons.append("no_meaningful_progress_vs_previous_best")

    if quality_gate_status.get("status") == "pass":
        reasons.append("meets_quality_gate")
    else:
        reasons.append(str(quality_gate_status.get("reason") or "quality_gate_not_met"))

    if listener_decision == "survivor":
        reasons.append("listener_agent_survivor")
    elif listener_decision == "borderline":
        reasons.append("listener_agent_borderline")
    elif listener_decision == "reject":
        reasons.append("listener_agent_reject")

    return {
        "decision": decision,
        "is_best_so_far": bool(improved_vs_best_before),
        "best_iteration_before": None if best_iteration_index is None else int(best_iteration_index),
        "reasons": reasons,
        "candidate_listener_decision": listener_decision,
        "candidate_overall_score": round(float(overall), 1),
    }


def _build_iteration_summary(iteration_record: dict[str, Any]) -> dict[str, Any]:
    top_intervention = _top_intervention_summary((iteration_record.get("top_interventions") or [None])[0])
    change_result = iteration_record.get("change_command") or {}
    test_result = iteration_record.get("test_command") or {}
    listener_assessment = iteration_record.get("listener_assessment") or {}
    return {
        "iteration": int(iteration_record.get("iteration") or 0),
        "candidate_overall_score": float(iteration_record.get("candidate_overall_score") or 0.0),
        "candidate_verdict": str(iteration_record.get("candidate_verdict") or "unknown"),
        "candidate_listener_decision": str(listener_assessment.get("decision") or "unknown"),
        "candidate_listener_rank": float(listener_assessment.get("listener_rank") or 0.0),
        "candidate_keep_decision": dict(iteration_record.get("candidate_keep_decision") or {}),
        "gap_to_target": float(iteration_record.get("gap_to_target") or 0.0),
        "improved_vs_best_before": bool(iteration_record.get("improved_vs_best_before")),
        "plateau_count": int(iteration_record.get("plateau_count") or 0),
        "quality_gate_status": dict(iteration_record.get("quality_gate_status") or {}),
        "render_dir": str((iteration_record.get("render") or {}).get("output_dir") or iteration_record.get("candidate_input") or ""),
        "top_intervention": top_intervention,
        "change_command_status": None if not change_result else {
            "returncode": int(change_result.get("returncode") or 0),
            "command_text": str(change_result.get("command_text") or ""),
        },
        "test_command_status": None if not test_result else {
            "returncode": int(test_result.get("returncode") or 0),
            "command_text": str(test_result.get("command_text") or ""),
        },
    }


def _build_loop_summary(loop_report: dict[str, Any]) -> dict[str, Any]:
    iterations = list(loop_report.get("iterations") or [])
    best = loop_report.get("best_iteration") or {}
    scores = [float(item.get("candidate_overall_score") or 0.0) for item in iterations]
    improvement = 0.0
    if scores:
        improvement = round(max(scores) - scores[0], 1)
    best_iteration_id = int(best.get("iteration") or 0)
    best_record = next((item for item in iterations if int(item.get("iteration") or 0) == best_iteration_id), None)
    best_top_intervention = _top_intervention_summary(((best_record or {}).get("top_interventions") or [None])[0])
    keep_count = sum(
        1 for item in iterations
        if str((item.get("candidate_keep_decision") or {}).get("decision") or "") == "keep"
    )
    return {
        "total_iterations": len(iterations),
        "best_iteration": {
            "iteration": best_iteration_id,
            "candidate_overall_score": float(best.get("candidate_overall_score") or 0.0),
            "candidate_verdict": str(best.get("candidate_verdict") or "unknown"),
            "candidate_listener_decision": str(best.get("candidate_listener_decision") or "unknown"),
            "candidate_keep_decision": dict(best.get("candidate_keep_decision") or {}),
            "candidate_input": str(best.get("candidate_input") or ""),
        } if best else None,
        "score_trajectory": scores,
        "net_improvement": improvement,
        "candidate_decisions": {
            "kept_iterations": keep_count,
            "rejected_iterations": max(0, len(iterations) - keep_count),
        },
        "stop_reason": str(loop_report.get("stop_reason") or ""),
        "best_top_intervention": best_top_intervention,
        "iteration_summaries": [_build_iteration_summary(item) for item in iterations],
    }


def _build_change_command_context(
    *,
    iteration: int,
    iteration_dir: Path,
    render_dir: Path,
    root: Path,
    song_a: str,
    song_b: str,
    candidate_path: str,
    feedback_path: Path,
    feedback_brief: dict[str, Any],
    candidate_report: dict[str, Any],
    best_score: float,
    quality_gate: float,
    quality_gate_status: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    top_intervention = _top_intervention_summary((feedback_brief.get("ranked_interventions") or [None])[0])
    report = {
        "schema_version": CHANGE_COMMAND_CONTEXT_SCHEMA_VERSION,
        "iteration": int(iteration),
        "iteration_dir": str(iteration_dir),
        "render_dir": str(render_dir),
        "candidate_input": candidate_path,
        "feedback_json": str(feedback_path),
        "song_a": str(song_a),
        "song_b": str(song_b),
        "candidate": {
            "overall_score": float(candidate_report.get("overall_score") or 0.0),
            "verdict": str(candidate_report.get("verdict") or "unknown"),
            "top_reasons": list((candidate_report.get("top_reasons") or [])[:5]),
            "top_fixes": list((candidate_report.get("top_fixes") or [])[:5]),
        },
        "goal": dict(feedback_brief.get("goal") or {}),
        "next_code_targets": list(feedback_brief.get("next_code_targets") or []),
        "top_interventions": list((feedback_brief.get("ranked_interventions") or [])[:3]),
        "prioritized_execution_plan": list((feedback_brief.get("prioritized_execution_plan") or [])[:5]),
        "planner_feedback_map": list((feedback_brief.get("planner_feedback_map") or [])[:5]),
        "top_intervention": top_intervention,
        "best_score_so_far": float(best_score),
        "quality_gate": float(quality_gate),
        "quality_gate_status": dict(quality_gate_status or {}),
    }
    change_context_path = iteration_dir / "change_command_context.json"
    change_request_path = iteration_dir / "change_request.md"
    _write_json(change_context_path, report)

    headline = top_intervention["component"] or "overall_quality"
    bullets = []
    for action in top_intervention["actions"][:3]:
        bullets.append(f"- {action}")
    if not bullets:
        bullets.append("- Review the feedback brief and make the smallest high-leverage change.")
    targets = top_intervention["code_targets"] or list(feedback_brief.get("next_code_targets") or [])
    bullet_lines = "\n".join(bullets)
    target_lines = "\n".join(f"- `{path}`" for path in targets[:6]) or "- _No explicit code targets provided._"
    planner_map = list((feedback_brief.get("planner_feedback_map") or [])[:3])
    planner_lines = "\n".join(
        f"- `{item.get('failure_mode')}` ({item.get('component')}): confidence {float(item.get('confidence') or 0.0):.2f}; targets {', '.join(item.get('planner_code_targets') or []) or 'n/a'}"
        for item in planner_map
    ) or "- _No explicit planner feedback routes identified._"
    request_text = (
        f"# Closed-loop change request\n\n"
        f"Iteration: {iteration}\n"
        f"Candidate overall score: {report['candidate']['overall_score']:.1f}\n"
        f"Candidate verdict: {report['candidate']['verdict']}\n"
        f"Gap to target: {float((feedback_brief.get('goal') or {}).get('gap_to_target') or 0.0):.1f}\n"
        f"Best score so far: {float(best_score):.1f}\n"
        f"Quality gate: {float(quality_gate):.1f}\n"
        f"Quality gate status: {quality_gate_status.get('reason', 'unknown')}\n\n"
        f"## Primary intervention\n\n"
        f"Component: `{headline}`\n\n"
        f"Problem: {top_intervention['problem'] or 'See feedback brief for details.'}\n\n"
        f"Suggested actions:\n"
        f"{bullet_lines}\n\n"
        f"Code targets:\n{target_lines}\n\n"
        f"Planner feedback routes:\n{planner_lines}\n\n"
        f"Artifacts:\n"
        f"- Feedback brief JSON: `{feedback_path}`\n"
        f"- Structured change context JSON: `{change_context_path}`\n"
        f"- Render output: `{render_dir}`\n"
    )
    _write_text(change_request_path, request_text)

    context = {
        "iteration": iteration,
        "iteration_dir": str(iteration_dir),
        "render_dir": str(render_dir),
        "candidate_input": candidate_path,
        "feedback_json": str(feedback_path),
        "change_context_json": str(change_context_path),
        "change_request_md": str(change_request_path),
        "output_root": str(root),
        "best_score": float(best_score),
        "quality_gate": float(quality_gate),
        "quality_gate_status_json": json.dumps(quality_gate_status, indent=2, sort_keys=True),
        "song_a": str(song_a),
        "song_b": str(song_b),
        "candidate_score": report["candidate"]["overall_score"],
        "candidate_verdict": report["candidate"]["verdict"],
        "gap_to_target": float((feedback_brief.get("goal") or {}).get("gap_to_target") or 0.0),
        "top_component": top_intervention["component"],
        "top_problem": top_intervention["problem"],
        "top_actions": top_intervention["actions_text"],
        "top_code_targets": top_intervention["code_targets_csv"],
    }
    return context, change_context_path, change_request_path


def run_closed_loop(
    *,
    song_a: str,
    song_b: str,
    references: list[str],
    output_root: str,
    max_iterations: int = 3,
    quality_gate: float = 85.0,
    plateau_limit: int = 2,
    min_improvement: float = 0.5,
    change_command: str | None = None,
    test_command: str | None = None,
    change_dispatch: dict[str, Any] | None = None,
    test_dispatch: dict[str, Any] | None = None,
    target_score: float = 99.0,
) -> dict[str, Any]:
    if max_iterations < 1:
        raise LoopError("max_iterations must be at least 1")
    if plateau_limit < 1:
        raise LoopError("plateau_limit must be at least 1")
    if change_command and change_dispatch:
        raise LoopError("Specify either change_command or change_dispatch, not both")
    if test_command and test_dispatch:
        raise LoopError("Specify either test_command or test_dispatch, not both")

    hydrated_change_dispatch = _hydrate_dispatch_spec(change_dispatch, label="change")
    hydrated_test_dispatch = _hydrate_dispatch_spec(test_dispatch, label="test")
    if hydrated_change_dispatch and not change_command:
        change_command = str(hydrated_change_dispatch["command"])
    if hydrated_test_dispatch and not test_command:
        test_command = str(hydrated_test_dispatch["command"])

    try:
        references = normalize_reference_inputs(references)
    except ValueError as exc:
        raise LoopError(str(exc)) from exc

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    loop_report: dict[str, Any] = {
        "schema_version": CLOSED_LOOP_SCHEMA_VERSION,
        "song_a": str(Path(song_a).expanduser().resolve()),
        "song_b": str(Path(song_b).expanduser().resolve()),
        "references": [str(Path(ref).expanduser().resolve()) for ref in references],
        "config": {
            "max_iterations": int(max_iterations),
            "quality_gate": float(quality_gate),
            "plateau_limit": int(plateau_limit),
            "min_improvement": float(min_improvement),
            "target_score": float(target_score),
            "change_command": change_command,
            "test_command": test_command,
            "change_dispatch": hydrated_change_dispatch,
            "test_dispatch": hydrated_test_dispatch,
        },
        "iterations": [],
        "best_iteration": None,
        "stop_reason": None,
        "summary": [],
    }

    best_score: float | None = None
    best_iteration_index: int | None = None
    best_progress_signal: tuple[int, float, float, float] | None = None
    plateau_count = 0

    for iteration_index in range(1, max_iterations + 1):
        iteration_dir = root / f"iter_{iteration_index:02d}"
        render_dir = iteration_dir / "render"
        render_meta = render_iteration(song_a, song_b, render_dir)

        candidate_path = str(render_dir)
        candidate_report = _candidate_report(candidate_path)
        listener_assessment = _candidate_listener_assessment(candidate_path)
        feedback_brief = _require_feedback_brief_shape(
            build_feedback_brief(candidate_path, references, target_score=target_score)
        )
        feedback_path = iteration_dir / "listen_feedback_brief.json"
        listener_assessment_path = iteration_dir / "listener_assessment.json"
        _write_json(feedback_path, feedback_brief)
        _write_json(listener_assessment_path, listener_assessment)

        overall = float(candidate_report.get("overall_score") or 0.0)
        verdict = str(candidate_report.get("verdict") or "unknown")
        listener_decision = str(listener_assessment.get("decision") or "unknown")
        listener_rank = float(listener_assessment.get("listener_rank") or 0.0)
        quality_gate_status = _quality_gate_status(
            overall=overall,
            quality_gate=quality_gate,
            feedback_brief=feedback_brief,
        )
        prior_best_iteration_index = best_iteration_index
        improved, best_progress_signal = _is_meaningful_progress(
            overall=overall,
            listener_assessment=listener_assessment,
            quality_gate_status=quality_gate_status,
            best_progress_signal=best_progress_signal,
            min_improvement=min_improvement,
        )
        if improved:
            best_score = overall if best_score is None else max(best_score, overall)
            best_iteration_index = iteration_index
            plateau_count = 0
        else:
            plateau_count += 1

        candidate_keep_decision = _candidate_keep_decision(
            iteration=iteration_index,
            overall=overall,
            listener_assessment=listener_assessment,
            quality_gate_status=quality_gate_status,
            improved_vs_best_before=improved,
            best_iteration_index=prior_best_iteration_index,
        )

        iteration_record: dict[str, Any] = {
            "iteration": iteration_index,
            "render": render_meta,
            "candidate_input": candidate_path,
            "candidate_overall_score": overall,
            "candidate_verdict": verdict,
            "candidate_listener_decision": listener_decision,
            "candidate_listener_rank": listener_rank,
            "candidate_keep_decision": candidate_keep_decision,
            "quality_gate_status": quality_gate_status,
            "listener_assessment": listener_assessment,
            "listener_assessment_path": str(listener_assessment_path),
            "feedback_brief_path": str(feedback_path),
            "gap_to_target": feedback_brief["goal"]["gap_to_target"],
            "top_interventions": feedback_brief["ranked_interventions"][:3],
            "improved_vs_best_before": bool(improved),
            "plateau_count": plateau_count,
            "artifacts": _build_iteration_artifacts(
                iteration_dir=iteration_dir,
                render_dir=render_dir,
                feedback_path=feedback_path,
                feedback_brief=feedback_brief,
                listener_assessment_path=listener_assessment_path,
                listener_assessment=listener_assessment,
            ),
        }

        stop_reason = None
        if quality_gate_status["status"] == "pass" and listener_decision == "survivor":
            stop_reason = f"quality_gate_reached:{quality_gate:.1f}"
        elif plateau_count >= plateau_limit:
            stop_reason = f"plateau:{plateau_count}"
        elif iteration_index >= max_iterations:
            stop_reason = f"max_iterations:{max_iterations}"
        elif not change_command:
            stop_reason = "no_change_command_configured"

        if not stop_reason and change_command:
            context, change_context_path, change_request_path = _build_change_command_context(
                iteration=iteration_index,
                iteration_dir=iteration_dir,
                render_dir=render_dir,
                root=root,
                song_a=song_a,
                song_b=song_b,
                candidate_path=candidate_path,
                feedback_path=feedback_path,
                feedback_brief=feedback_brief,
                candidate_report=candidate_report,
                best_score=best_score if best_score is not None else overall,
                quality_gate=quality_gate,
                quality_gate_status=quality_gate_status,
            )
            iteration_record["change_context_json"] = str(change_context_path)
            iteration_record["change_request_md"] = str(change_request_path)
            iteration_record["artifacts"] = _build_iteration_artifacts(
                iteration_dir=iteration_dir,
                render_dir=render_dir,
                feedback_path=feedback_path,
                feedback_brief=feedback_brief,
                listener_assessment_path=listener_assessment_path,
                listener_assessment=listener_assessment,
                change_context_path=change_context_path,
                change_request_path=change_request_path,
            )
            change_result = _run_command_template(
                change_command,
                context,
                timeout=int((hydrated_change_dispatch or {}).get("timeout") or 3600),
                label="change",
            )
            iteration_record["change_command"] = change_result
            if int(change_result["returncode"]) != 0:
                stop_reason = f"change_command_failed:{iteration_index}"

            if not stop_reason and test_command:
                test_result = _run_command_template(
                    test_command,
                    context,
                    timeout=int((hydrated_test_dispatch or {}).get("timeout") or 3600),
                    label="test",
                )
                iteration_record["test_command"] = test_result
                if int(test_result["returncode"]) != 0:
                    stop_reason = f"test_command_failed:{iteration_index}"

        loop_report["iterations"].append(iteration_record)

        if stop_reason:
            loop_report["stop_reason"] = stop_reason
            break

    if best_iteration_index is not None:
        best = next(item for item in loop_report["iterations"] if item["iteration"] == best_iteration_index)
        loop_report["best_iteration"] = {
            "iteration": best_iteration_index,
            "candidate_overall_score": best["candidate_overall_score"],
            "candidate_verdict": best["candidate_verdict"],
            "candidate_listener_decision": best.get("candidate_listener_decision", "unknown"),
            "candidate_keep_decision": dict(best.get("candidate_keep_decision") or {}),
            "quality_gate_status": dict(best.get("quality_gate_status") or {}),
            "candidate_input": best["candidate_input"],
            "feedback_brief_path": best["feedback_brief_path"],
            "listener_assessment_path": best.get("listener_assessment_path"),
        }

    keep_count = sum(
        1 for item in loop_report["iterations"]
        if str((item.get("candidate_keep_decision") or {}).get("decision") or "") == "keep"
    )
    reject_count = max(0, len(loop_report["iterations"]) - keep_count)
    loop_report["candidate_decisions"] = {
        "kept_iterations": keep_count,
        "rejected_iterations": reject_count,
    }

    if loop_report["best_iteration"]:
        loop_report["summary"].append(
            f"Best iteration was {loop_report['best_iteration']['iteration']} with overall score {loop_report['best_iteration']['candidate_overall_score']:.1f}."
        )
    if loop_report["stop_reason"]:
        loop_report["summary"].append(f"Loop stopped because {loop_report['stop_reason']}.")

    loop_report["artifact_schema"] = {
        "iteration_artifacts": {
            "render_output": {"kind": "render_output_dir", "required": True},
            "listen_feedback_brief": {"kind": "listen_feedback_brief", "required": True, "schema_version": "from feedback brief artifact"},
            "listener_assessment": {"kind": "listener_agent_case_assessment", "required": True, "schema_version": "0.1.0"},
            "change_context": {"kind": "change_command_context", "required": False, "schema_version": CHANGE_COMMAND_CONTEXT_SCHEMA_VERSION},
            "change_request": {"kind": "change_request_markdown", "required": False},
        }
    }
    loop_report["loop_summary"] = _build_loop_summary(loop_report)
    if loop_report["loop_summary"].get("net_improvement"):
        loop_report["summary"].append(
            f"Net improvement across the run was {float(loop_report['loop_summary']['net_improvement']):.1f} points."
        )
    best_top_intervention = (loop_report["loop_summary"].get("best_top_intervention") or {}).get("component")
    if best_top_intervention:
        loop_report["summary"].append(
            f"Top unresolved intervention at the best iteration was {best_top_intervention}."
        )

    _write_json(root / "closed_loop_report.json", loop_report)
    return loop_report


def main() -> int:
    if "--print-template-fields" in sys.argv[1:]:
        print(_command_template_fields_text())
        return 0

    parser = argparse.ArgumentParser(
        description="Run a bounded listener-driven improvement loop for one fusion pair.",
        epilog="Use --print-template-fields to inspect the available placeholders for --change-command and --test-command.",
    )
    parser.add_argument("song_a", help="Path to parent song A")
    parser.add_argument("song_b", help="Path to parent song B")
    parser.add_argument("references", nargs="+", help="One or more good reference inputs")
    parser.add_argument("--output-root", "-o", required=True, help="Directory to write iteration outputs")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of loop iterations")
    parser.add_argument("--quality-gate", type=float, default=85.0, help="Stop once the candidate clears this overall score and, when available, the reference-weighted component gate")
    parser.add_argument("--plateau-limit", type=int, default=2, help="Stop after this many non-improving iterations")
    parser.add_argument("--min-improvement", type=float, default=0.5, help="Minimum score gain required to reset plateau detection")
    parser.add_argument("--target-score", type=float, default=99.0, help="Long-term aspirational target score for the feedback brief")
    parser.add_argument("--change-command", help="Optional command template used to change code between iterations")
    parser.add_argument("--test-command", help="Optional command template used to validate changes between iterations")
    parser.add_argument("--change-dispatch", help="Optional JSON dispatch spec for the change step")
    parser.add_argument("--test-dispatch", help="Optional JSON dispatch spec for the test step")
    args = parser.parse_args()

    change_dispatch = _read_dispatch_spec(args.change_dispatch, label="change") if args.change_dispatch else None
    test_dispatch = _read_dispatch_spec(args.test_dispatch, label="test") if args.test_dispatch else None

    report = run_closed_loop(
        song_a=args.song_a,
        song_b=args.song_b,
        references=args.references,
        output_root=args.output_root,
        max_iterations=args.max_iterations,
        quality_gate=args.quality_gate,
        plateau_limit=args.plateau_limit,
        min_improvement=args.min_improvement,
        change_command=args.change_command,
        test_command=args.test_command,
        change_dispatch=change_dispatch,
        test_dispatch=test_dispatch,
        target_score=args.target_score,
    )
    print(json.dumps({
        "best_iteration": report.get("best_iteration"),
        "stop_reason": report.get("stop_reason"),
        "summary": report.get("summary", []),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
