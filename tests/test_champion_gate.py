from __future__ import annotations

import json
from pathlib import Path

from scripts.champion_gate import gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_promotes_when_guardrails_and_delta_pass(tmp_path: Path) -> None:
    benchmark = tmp_path / "runs/phase-12/benchmark_summary_001.json"
    champion_pointer = tmp_path / "runs/champion/current.json"
    champion_history = tmp_path / "runs/champion/history.json"

    _write_json(
        champion_pointer,
        {
            "id": "champion-old",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 0.80,
            "score": 0.83,
        },
    )
    _write_json(
        benchmark,
        {
            "run_id": "challenger-1",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 0.84,
            "score": 0.87,
        },
    )

    out = gate(
        benchmark_path=benchmark,
        benchmarks_dir=tmp_path / "runs/phase-12",
        champion_pointer_path=champion_pointer,
        champion_history_path=champion_history,
        min_song_likeness=0.72,
        min_score=0.80,
        min_delta=0.01,
    )

    assert out["promoted"] is True
    assert _read_json(champion_pointer)["id"] == "challenger-1"
    history = _read_json(champion_history)
    assert len(history) == 1
    assert history[0]["decision"]["reason"] == "promoted"


def test_blocks_on_guardrail_failure(tmp_path: Path) -> None:
    benchmark = tmp_path / "runs/phase-12/benchmark_summary_002.json"
    champion_pointer = tmp_path / "runs/champion/current.json"
    champion_history = tmp_path / "runs/champion/history.json"

    _write_json(
        champion_pointer,
        {
            "id": "champion-ok",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 0.88,
            "score": 0.90,
        },
    )
    _write_json(
        benchmark,
        {
            "run_id": "challenger-low-like",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 0.50,
            "score": 0.98,
        },
    )

    out = gate(
        benchmark_path=benchmark,
        benchmarks_dir=tmp_path / "runs/phase-12",
        champion_pointer_path=champion_pointer,
        champion_history_path=champion_history,
        min_song_likeness=0.72,
        min_score=0.80,
        min_delta=0.01,
    )

    assert out["promoted"] is False
    assert out["reason"] == "guardrails_failed"
    assert _read_json(champion_pointer)["id"] == "champion-ok"


def test_blocks_on_delta_threshold(tmp_path: Path) -> None:
    benchmark = tmp_path / "runs/phase-12/benchmark_summary_003.json"
    champion_pointer = tmp_path / "runs/champion/current.json"
    champion_history = tmp_path / "runs/champion/history.json"

    _write_json(
        champion_pointer,
        {
            "id": "champion-tight",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 0.85,
            "score": 0.88,
        },
    )
    _write_json(
        benchmark,
        {
            "run_id": "challenger-tight",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 0.86,
            "score": 0.885,
        },
    )

    out = gate(
        benchmark_path=benchmark,
        benchmarks_dir=tmp_path / "runs/phase-12",
        champion_pointer_path=champion_pointer,
        champion_history_path=champion_history,
        min_song_likeness=0.72,
        min_score=0.80,
        min_delta=0.01,
    )

    assert out["promoted"] is False
    assert out["reason"] == "delta_below_threshold"


def test_dry_run_does_not_persist(tmp_path: Path) -> None:
    benchmark = tmp_path / "runs/phase-12/benchmark_summary_004.json"
    champion_pointer = tmp_path / "runs/champion/current.json"
    champion_history = tmp_path / "runs/champion/history.json"

    _write_json(
        benchmark,
        {
            "run_id": "challenger-dry",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 0.95,
            "score": 0.99,
        },
    )

    out = gate(
        benchmark_path=benchmark,
        benchmarks_dir=tmp_path / "runs/phase-12",
        champion_pointer_path=champion_pointer,
        champion_history_path=champion_history,
        min_song_likeness=0.72,
        min_score=0.80,
        min_delta=0.01,
        dry_run=True,
    )

    assert out["promoted"] is False
    assert champion_pointer.exists() is False
    assert champion_history.exists() is False


def test_supports_song_birth_summary_shape_and_autoscales_thresholds(tmp_path: Path) -> None:
    benchmark = tmp_path / "runs/song_birth_phase12_20260330_101010/song_birth_benchmark_summary.json"
    champion_pointer = tmp_path / "runs/champion/current.json"
    champion_history = tmp_path / "runs/champion/history.json"

    _write_json(
        champion_pointer,
        {
            "id": "champion-old",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 80.0,
            "score": 73.0,
        },
    )
    _write_json(
        benchmark,
        {
            "timestamp": "20260330_101010",
            "passed": True,
            "scores": {
                "overall_score": 74.5,
                "song_likeness_score": 80.7,
                "gating_status": "pass",
            },
        },
    )

    out = gate(
        benchmark_path=benchmark,
        benchmarks_dir=tmp_path / "runs",
        champion_pointer_path=champion_pointer,
        champion_history_path=champion_history,
        min_song_likeness=0.72,
        min_score=0.70,
        min_delta=0.01,
    )

    assert out["promoted"] is True
    assert out["reason"] == "promoted"
    assert abs(out["effective_thresholds"]["min_delta"] - 1.0) < 1e-9
    assert _read_json(champion_pointer)["id"] == "20260330_101010"


def test_song_birth_absolute_delta_is_not_overscaled(tmp_path: Path) -> None:
    benchmark = tmp_path / "runs/song_birth_phase12_20260330_111111/song_birth_benchmark_summary.json"
    champion_pointer = tmp_path / "runs/champion/current.json"
    champion_history = tmp_path / "runs/champion/history.json"

    _write_json(
        champion_pointer,
        {
            "id": "champion-old",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 80.0,
            "score": 74.4,
        },
    )
    _write_json(
        benchmark,
        {
            "timestamp": "20260330_111111",
            "passed": True,
            "scores": {
                "overall_score": 74.55,
                "song_likeness_score": 80.8,
                "gating_status": "pass",
            },
        },
    )

    out = gate(
        benchmark_path=benchmark,
        benchmarks_dir=tmp_path / "runs",
        champion_pointer_path=champion_pointer,
        champion_history_path=champion_history,
        min_song_likeness=80.0,
        min_score=70.0,
        min_delta=0.1,
    )

    assert abs(out["effective_thresholds"]["min_delta"] - 0.1) < 1e-9
    assert out["promoted"] is True


def test_song_birth_gating_status_fail_blocks_promotion(tmp_path: Path) -> None:
    benchmark = tmp_path / "runs/song_birth_phase12_20260330_121212/song_birth_benchmark_summary.json"
    champion_pointer = tmp_path / "runs/champion/current.json"
    champion_history = tmp_path / "runs/champion/history.json"

    _write_json(
        champion_pointer,
        {
            "id": "champion-old",
            "overall_pass": True,
            "gating_pass": True,
            "song_likeness": 80.0,
            "score": 73.0,
        },
    )
    _write_json(
        benchmark,
        {
            "timestamp": "20260330_121212",
            "passed": True,
            "scores": {
                "overall_score": 80.0,
                "song_likeness_score": 83.0,
                "gating_status": "fail",
            },
        },
    )

    out = gate(
        benchmark_path=benchmark,
        benchmarks_dir=tmp_path / "runs",
        champion_pointer_path=champion_pointer,
        champion_history_path=champion_history,
        min_song_likeness=80.0,
        min_score=70.0,
        min_delta=0.1,
    )

    assert out["promoted"] is False
    assert out["reason"] == "guardrails_failed"
    assert _read_json(champion_pointer)["id"] == "champion-old"
