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

PYTHON = ROOT / ".venv" / "bin" / "python"


class LoopError(RuntimeError):
    """Closed-loop runner failure."""


def _python_executable() -> str:
    return str(PYTHON if PYTHON.exists() else sys.executable)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _run_shell_template(command_template: str, context: dict[str, Any], *, timeout: int = 3600) -> dict[str, Any]:
    rendered = command_template.format(**context)
    proc = subprocess.run(rendered, cwd=str(ROOT), shell=True, capture_output=True, text=True, timeout=timeout)
    return {
        "command": rendered,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _candidate_report(candidate_input: str) -> dict[str, Any]:
    return ai_dj._resolve_compare_input(candidate_input)["report"]


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
    target_score: float = 99.0,
) -> dict[str, Any]:
    if max_iterations < 1:
        raise LoopError("max_iterations must be at least 1")
    if plateau_limit < 1:
        raise LoopError("plateau_limit must be at least 1")
    if not references:
        raise LoopError("at least one reference is required")

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    loop_report: dict[str, Any] = {
        "schema_version": "0.1.0",
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
        },
        "iterations": [],
        "best_iteration": None,
        "stop_reason": None,
        "summary": [],
    }

    best_score: float | None = None
    best_iteration_index: int | None = None
    plateau_count = 0

    for iteration_index in range(1, max_iterations + 1):
        iteration_dir = root / f"iter_{iteration_index:02d}"
        render_dir = iteration_dir / "render"
        render_meta = render_iteration(song_a, song_b, render_dir)

        candidate_path = str(render_dir)
        candidate_report = _candidate_report(candidate_path)
        feedback_brief = build_feedback_brief(candidate_path, references, target_score=target_score)
        feedback_path = iteration_dir / "listen_feedback_brief.json"
        _write_json(feedback_path, feedback_brief)

        overall = float(candidate_report.get("overall_score") or 0.0)
        verdict = str(candidate_report.get("verdict") or "unknown")
        improved = best_score is None or overall >= (best_score + float(min_improvement))
        if improved:
            best_score = overall
            best_iteration_index = iteration_index
            plateau_count = 0
        else:
            plateau_count += 1

        iteration_record: dict[str, Any] = {
            "iteration": iteration_index,
            "render": render_meta,
            "candidate_input": candidate_path,
            "candidate_overall_score": overall,
            "candidate_verdict": verdict,
            "feedback_brief_path": str(feedback_path),
            "gap_to_target": feedback_brief["goal"]["gap_to_target"],
            "top_interventions": feedback_brief["ranked_interventions"][:3],
            "improved_vs_best_before": bool(improved),
            "plateau_count": plateau_count,
        }

        stop_reason = None
        if overall >= float(quality_gate):
            stop_reason = f"quality_gate_reached:{quality_gate:.1f}"
        elif plateau_count >= plateau_limit:
            stop_reason = f"plateau:{plateau_count}"
        elif iteration_index >= max_iterations:
            stop_reason = f"max_iterations:{max_iterations}"
        elif not change_command:
            stop_reason = "no_change_command_configured"

        if not stop_reason and change_command:
            context = {
                "iteration": iteration_index,
                "iteration_dir": str(iteration_dir),
                "render_dir": str(render_dir),
                "candidate_input": candidate_path,
                "feedback_json": str(feedback_path),
                "output_root": str(root),
                "best_score": best_score if best_score is not None else overall,
                "quality_gate": quality_gate,
                "song_a": str(song_a),
                "song_b": str(song_b),
            }
            change_result = _run_shell_template(change_command, context)
            iteration_record["change_command"] = change_result
            if int(change_result["returncode"]) != 0:
                stop_reason = f"change_command_failed:{iteration_index}"

            if not stop_reason and test_command:
                test_result = _run_shell_template(test_command, context)
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
            "candidate_input": best["candidate_input"],
            "feedback_brief_path": best["feedback_brief_path"],
        }

    if loop_report["best_iteration"]:
        loop_report["summary"].append(
            f"Best iteration was {loop_report['best_iteration']['iteration']} with overall score {loop_report['best_iteration']['candidate_overall_score']:.1f}."
        )
    if loop_report["stop_reason"]:
        loop_report["summary"].append(f"Loop stopped because {loop_report['stop_reason']}.")

    _write_json(root / "closed_loop_report.json", loop_report)
    return loop_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded listener-driven improvement loop for one fusion pair.")
    parser.add_argument("song_a", help="Path to parent song A")
    parser.add_argument("song_b", help="Path to parent song B")
    parser.add_argument("references", nargs="+", help="One or more good reference inputs")
    parser.add_argument("--output-root", "-o", required=True, help="Directory to write iteration outputs")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of loop iterations")
    parser.add_argument("--quality-gate", type=float, default=85.0, help="Stop once the candidate clears this overall score")
    parser.add_argument("--plateau-limit", type=int, default=2, help="Stop after this many non-improving iterations")
    parser.add_argument("--min-improvement", type=float, default=0.5, help="Minimum score gain required to reset plateau detection")
    parser.add_argument("--target-score", type=float, default=99.0, help="Long-term aspirational target score for the feedback brief")
    parser.add_argument("--change-command", help="Optional shell command template used to change code between iterations")
    parser.add_argument("--test-command", help="Optional shell command template used to validate changes between iterations")
    args = parser.parse_args()

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
