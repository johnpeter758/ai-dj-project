#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.champion_gate import gate


DEFAULT_STOP_SWITCH = Path("AUTOPILOT_STOP")
DEFAULT_REPORT_PATH = Path("runs/autopilot/last_mastering_cycle.json")
DEFAULT_CHAMPION_POINTER = Path("runs/champion/current.json")
DEFAULT_CHAMPION_HISTORY = Path("runs/champion/history.json")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _tail(text: str, lines: int = 30) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


def find_latest_song_birth_summary(runs_dir: Path) -> Path:
    candidates = sorted(
        runs_dir.glob("song_birth_phase12_*/song_birth_benchmark_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No song_birth benchmark summaries under {runs_dir}")
    return candidates[-1]


def _git_has_changes(repo_root: Path) -> bool:
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    return bool(status.stdout.strip())


def _git_branch_tracking(repo_root: Path) -> str:
    out = _run(["git", "status", "--short", "--branch"], cwd=repo_root)
    first = out.stdout.splitlines()[0] if out.stdout else ""
    return first.strip()


def _commit_and_push(repo_root: Path, message: str) -> dict[str, Any]:
    if not _git_has_changes(repo_root):
        return {"committed": False, "pushed": False, "reason": "no_changes"}

    add = _run(["git", "add", "-A"], cwd=repo_root)
    if add.returncode != 0:
        return {
            "committed": False,
            "pushed": False,
            "reason": "git_add_failed",
            "stderr_tail": _tail(add.stderr),
        }

    commit = _run(["git", "commit", "-m", message], cwd=repo_root)
    if commit.returncode != 0:
        return {
            "committed": False,
            "pushed": False,
            "reason": "git_commit_failed",
            "stdout_tail": _tail(commit.stdout),
            "stderr_tail": _tail(commit.stderr),
        }

    push = _run(["git", "push", "origin", "main"], cwd=repo_root)
    return {
        "committed": True,
        "pushed": push.returncode == 0,
        "commit_stdout_tail": _tail(commit.stdout),
        "push_stdout_tail": _tail(push.stdout),
        "push_stderr_tail": _tail(push.stderr),
        "push_returncode": push.returncode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one autonomous mastering benchmark/gate cycle")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stop-switch", type=Path, default=DEFAULT_STOP_SWITCH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--champion-pointer", type=Path, default=DEFAULT_CHAMPION_POINTER)
    parser.add_argument("--champion-history", type=Path, default=DEFAULT_CHAMPION_HISTORY)
    parser.add_argument("--min-song-likeness", type=float, default=80.0)
    parser.add_argument("--min-score", type=float, default=70.0)
    parser.add_argument("--min-delta", type=float, default=0.1)
    parser.add_argument("--commit-on-promote", action="store_true")
    parser.add_argument("--push-on-promote", action="store_true")
    parser.add_argument("--require-clean-start", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    stop_switch = (repo_root / args.stop_switch).resolve() if not args.stop_switch.is_absolute() else args.stop_switch
    report_path = (repo_root / args.report_path).resolve() if not args.report_path.is_absolute() else args.report_path
    champion_pointer = (repo_root / args.champion_pointer).resolve() if not args.champion_pointer.is_absolute() else args.champion_pointer
    champion_history = (repo_root / args.champion_history).resolve() if not args.champion_history.is_absolute() else args.champion_history
    runs_dir = repo_root / "runs"

    report: dict[str, Any] = {
        "timestamp": utc_now(),
        "repo_root": str(repo_root),
        "branch": _git_branch_tracking(repo_root),
        "stop_switch": str(stop_switch),
        "status": "starting",
    }

    if stop_switch.exists():
        report.update({"status": "paused", "reason": "stop_switch_present", "ok": True})
        _safe_write_json(report_path, report)
        print(json.dumps(report, sort_keys=True))
        return 0

    if args.require_clean_start and _git_has_changes(repo_root):
        report.update({"status": "blocked", "reason": "dirty_worktree", "ok": False})
        _safe_write_json(report_path, report)
        print(json.dumps(report, sort_keys=True))
        return 2

    benchmark_cmd = [sys.executable, "scripts/run_song_birth_benchmark.py"]
    benchmark = _run(benchmark_cmd, cwd=repo_root)
    report["benchmark"] = {
        "returncode": benchmark.returncode,
        "stdout_tail": _tail(benchmark.stdout),
        "stderr_tail": _tail(benchmark.stderr),
    }
    if benchmark.returncode != 0:
        report.update({"status": "failed", "reason": "benchmark_failed", "ok": False})
        _safe_write_json(report_path, report)
        print(json.dumps(report, sort_keys=True))
        return benchmark.returncode

    benchmark_summary = find_latest_song_birth_summary(runs_dir)
    gate_out = gate(
        benchmark_path=benchmark_summary,
        benchmarks_dir=runs_dir,
        champion_pointer_path=champion_pointer,
        champion_history_path=champion_history,
        min_song_likeness=args.min_song_likeness,
        min_score=args.min_score,
        min_delta=args.min_delta,
        dry_run=args.dry_run,
    )

    report["gate"] = gate_out
    report["benchmark_summary"] = str(benchmark_summary)

    git_result = {"committed": False, "pushed": False, "reason": "not_requested"}
    if gate_out.get("promoted") and args.commit_on_promote and not args.dry_run:
        commit_message = (
            f"autopilot(mastering): promote {gate_out.get('challenger_id')} "
            f"(delta={gate_out.get('score_delta')})"
        )
        if args.push_on_promote:
            git_result = _commit_and_push(repo_root, commit_message)
        else:
            if _git_has_changes(repo_root):
                add = _run(["git", "add", "-A"], cwd=repo_root)
                commit = _run(["git", "commit", "-m", commit_message], cwd=repo_root)
                git_result = {
                    "committed": add.returncode == 0 and commit.returncode == 0,
                    "pushed": False,
                    "commit_stdout_tail": _tail(commit.stdout),
                    "commit_stderr_tail": _tail(commit.stderr),
                }
            else:
                git_result = {"committed": False, "pushed": False, "reason": "no_changes"}

    report["git"] = git_result
    report.update(
        {
            "status": "ok",
            "ok": True,
            "promoted": bool(gate_out.get("promoted")),
            "next_step": (
                "continue_hypothesis_loop" if gate_out.get("promoted") else "try_next_non_overlapping_strategy"
            ),
        }
    )

    _safe_write_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
