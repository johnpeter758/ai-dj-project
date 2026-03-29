from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts.autopilot_orchestrator import LockError, orchestrate


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_single_cycle_success_persists_checkpoint(tmp_path: Path) -> None:
    state_path = tmp_path / "runs/autopilot/state.json"
    stop_switch = tmp_path / "AUTOPILOT_STOP"

    code = orchestrate(
        command=f"{sys.executable} -c \"print('ok')\"",
        state_path=state_path,
        stop_switch=stop_switch,
        single_cycle=True,
    )

    assert code == 0
    state = _read_json(state_path)
    assert state["status"] == "ok"
    assert state["run_count"] == 1
    assert state["success_count"] == 1
    assert state["checkpoint"]["last_completed_cycle"] == 1
    assert state["last_error"] is None


def test_single_cycle_failure_records_details(tmp_path: Path) -> None:
    state_path = tmp_path / "runs/autopilot/state.json"

    code = orchestrate(
        command=f"{sys.executable} -c \"import sys; print('boom'); sys.exit(7)\"",
        state_path=state_path,
        stop_switch=tmp_path / "AUTOPILOT_STOP",
        single_cycle=True,
    )

    assert code == 1
    state = _read_json(state_path)
    assert state["status"] == "failed"
    assert state["failure_count"] == 1
    assert state["last_error"]["returncode"] == 7
    assert "inspect_failure_and_retry" == state["next_step"]["action"]


def test_stop_switch_short_circuits_without_run(tmp_path: Path) -> None:
    state_path = tmp_path / "runs/autopilot/state.json"
    stop_switch = tmp_path / "AUTOPILOT_STOP"
    stop_switch.write_text("stop", encoding="utf-8")

    code = orchestrate(
        command="python -c \"print('should not run')\"",
        state_path=state_path,
        stop_switch=stop_switch,
        single_cycle=True,
    )

    assert code == 0
    state = _read_json(state_path)
    assert state["status"] == "stopped"
    assert state["run_count"] == 0
    assert state["next_step"]["action"] == "remove_stop_switch"


def test_lock_prevents_overlap(tmp_path: Path) -> None:
    state_path = tmp_path / "runs/autopilot/state.json"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("held", encoding="utf-8")

    try:
        orchestrate(
            command=f"{sys.executable} -c \"print('x')\"",
            state_path=state_path,
            stop_switch=tmp_path / "AUTOPILOT_STOP",
            single_cycle=True,
        )
        assert False, "expected LockError"
    except LockError:
        pass
