from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_STATE_PATH = Path("runs/autopilot/state.json")
DEFAULT_STOP_SWITCH = Path("AUTOPILOT_STOP")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


@dataclass
class RunResult:
    ok: bool
    returncode: int
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str


def run_cycle(command: str, timeout_seconds: int | None = None) -> RunResult:
    start = time.monotonic()
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration = round(time.monotonic() - start, 3)
    stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-15:])
    stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-15:])
    return RunResult(
        ok=completed.returncode == 0,
        returncode=completed.returncode,
        duration_seconds=duration,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


class LockError(RuntimeError):
    pass


class FileLock:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise LockError(f"autopilot already running (lock exists: {self.lock_path})") from exc
        os.write(self.fd, f"pid={os.getpid()} started_at={utc_now()}\n".encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _default_state(command: str) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "idle",
        "run_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "last_started_at": None,
        "last_finished_at": None,
        "last_error": None,
        "checkpoint": {
            "last_completed_cycle": 0,
            "last_successful_command": None,
            "last_successful_at": None,
        },
        "next_step": {
            "action": "run_phase12_cycle",
            "command": command,
            "reason": "normal_loop",
        },
    }


def _with_defaults(state: dict[str, Any], command: str) -> dict[str, Any]:
    base = _default_state(command)
    merged = {**base, **state}
    merged["checkpoint"] = {**base["checkpoint"], **state.get("checkpoint", {})}
    merged["next_step"] = {**base["next_step"], **state.get("next_step", {})}
    if not merged["next_step"].get("command"):
        merged["next_step"]["command"] = command
    return merged


def orchestrate(
    *,
    command: str,
    state_path: Path = DEFAULT_STATE_PATH,
    stop_switch: Path = DEFAULT_STOP_SWITCH,
    single_cycle: bool = False,
    max_cycles: int = 0,
    sleep_seconds: int = 60,
    timeout_seconds: int | None = None,
) -> int:
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with FileLock(lock_path):
        state = _with_defaults(_safe_read_json(state_path), command)

        cycles_run = 0
        while True:
            if stop_switch.exists():
                state["status"] = "stopped"
                state["last_finished_at"] = utc_now()
                state["next_step"] = {
                    "action": "remove_stop_switch",
                    "path": str(stop_switch),
                    "reason": "operator_stop_requested",
                    "command": command,
                }
                _safe_write_json(state_path, state)
                return 0

            state["status"] = "running"
            state["last_started_at"] = utc_now()
            state["run_count"] = int(state.get("run_count", 0)) + 1
            _safe_write_json(state_path, state)

            result = run_cycle(command, timeout_seconds=timeout_seconds)
            state["last_finished_at"] = utc_now()

            if result.ok:
                state["status"] = "ok"
                state["success_count"] = int(state.get("success_count", 0)) + 1
                state["checkpoint"]["last_completed_cycle"] = int(
                    state["checkpoint"].get("last_completed_cycle", 0)
                ) + 1
                state["checkpoint"]["last_successful_command"] = command
                state["checkpoint"]["last_successful_at"] = state["last_finished_at"]
                state["last_error"] = None
                state["next_step"] = {
                    "action": "run_phase12_cycle",
                    "command": command,
                    "reason": "normal_loop",
                    "delay_seconds": sleep_seconds,
                }
            else:
                state["status"] = "failed"
                state["failure_count"] = int(state.get("failure_count", 0)) + 1
                state["last_error"] = {
                    "at": state["last_finished_at"],
                    "returncode": result.returncode,
                    "duration_seconds": result.duration_seconds,
                    "stdout_tail": result.stdout_tail,
                    "stderr_tail": result.stderr_tail,
                }
                state["next_step"] = {
                    "action": "inspect_failure_and_retry",
                    "command": command,
                    "reason": "last_cycle_failed",
                    "suggested_commands": [
                        "python scripts/champion_gate.py --dry-run",
                        "tail -n 200 runs/autopilot/state.json",
                    ],
                }

            _safe_write_json(state_path, state)
            cycles_run += 1

            if single_cycle:
                return 0 if result.ok else 1
            if max_cycles > 0 and cycles_run >= max_cycles:
                return 0
            time.sleep(max(0, sleep_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Durable phase-12 autopilot orchestrator")
    parser.add_argument("--command", required=True, help="Phase-12 command to execute each cycle")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--stop-switch", type=Path, default=DEFAULT_STOP_SWITCH)
    parser.add_argument("--single-cycle", action="store_true", help="Run exactly one cycle and exit")
    parser.add_argument("--max-cycles", type=int, default=0, help="Bound total cycles in this invocation")
    parser.add_argument("--sleep-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return orchestrate(
            command=args.command,
            state_path=args.state_path,
            stop_switch=args.stop_switch,
            single_cycle=args.single_cycle,
            max_cycles=args.max_cycles,
            sleep_seconds=args.sleep_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except LockError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
