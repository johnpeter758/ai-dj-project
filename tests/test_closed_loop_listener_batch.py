from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import closed_loop_listener_runner as loop


def _feedback_brief(*, current_overall_score: float, weighted_score: float) -> dict:
    return {
        "schema_version": "0.1.0",
        "goal": {
            "target_listener_score": 99.0,
            "current_overall_score": current_overall_score,
            "gap_to_target": round(99.0 - current_overall_score, 1),
        },
        "ranked_interventions": [
            {
                "component": "song_likeness",
                "gap_vs_references": 12.0,
                "problem": "Arrangement still feels stitched together.",
                "actions": ["Reduce ownership switching across core sections."],
                "code_targets": ["ai_dj.py"],
            }
        ],
        "next_code_targets": ["ai_dj.py"],
        "planner_feedback_map": [],
        "render_feedback_map": [],
        "prioritized_execution_plan": [],
        "quality_gate_diagnostics": {
            "reference_weighted": {
                "candidate_weighted_score": weighted_score,
                "top_blockers": [],
            }
        },
    }


def _write_auto_shortlist_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_closed_loop_batch_mode_selects_listener_survivor(monkeypatch, tmp_path: Path) -> None:
    def fake_render_iteration(*_args, **_kwargs):
        pytest.fail("single-candidate render path should not run when batch_size > 1")

    def fake_auto_shortlist_fusion(song_a: str, song_b: str, output_root: str, **kwargs) -> int:
        assert song_a == "song_a.wav"
        assert song_b == "song_b.wav"
        assert kwargs["batch_size"] == 3
        assert kwargs["shortlist"] == 1
        outdir = Path(output_root)
        candidate_001 = outdir / "candidate_001"
        candidate_002 = outdir / "candidate_002"
        candidate_001.mkdir(parents=True, exist_ok=True)
        candidate_002.mkdir(parents=True, exist_ok=True)
        _write_auto_shortlist_report(
            outdir / "auto_shortlist_report.json",
            {
                "schema_version": "0.2.0",
                "listener_agent_report": {"counts": {"survivors": 1, "borderline": 0, "rejected": 1}},
                "recommended_shortlist": [
                    {
                        "candidate_id": "candidate_002",
                        "run_dir": str(candidate_002),
                        "overall_score": 82.0,
                        "listener_rank": 81.0,
                        "decision": "survivor",
                        "verdict": "promising",
                    }
                ],
                "closest_misses": [],
                "candidates": [
                    {
                        "candidate_id": "candidate_001",
                        "run_dir": str(candidate_001),
                        "overall_score": 54.0,
                        "listener_rank": 42.0,
                        "decision": "reject",
                        "verdict": "weak",
                    },
                    {
                        "candidate_id": "candidate_002",
                        "run_dir": str(candidate_002),
                        "overall_score": 82.0,
                        "listener_rank": 81.0,
                        "decision": "survivor",
                        "verdict": "promising",
                    },
                ],
            },
        )
        return 0

    monkeypatch.setattr(loop, "render_iteration", fake_render_iteration)
    monkeypatch.setattr(loop.ai_dj, "auto_shortlist_fusion", fake_auto_shortlist_fusion)
    monkeypatch.setattr(
        loop,
        "_candidate_report",
        lambda candidate_input: {"overall_score": 82.0, "verdict": "promising"} if candidate_input.endswith("candidate_002") else {"overall_score": 54.0, "verdict": "weak"},
    )
    monkeypatch.setattr(
        loop,
        "_candidate_listener_assessment",
        lambda candidate_input: {"decision": "survivor", "listener_rank": 81.0} if candidate_input.endswith("candidate_002") else {"decision": "reject", "listener_rank": 42.0},
    )
    monkeypatch.setattr(
        loop,
        "build_feedback_brief",
        lambda candidate_input, references, target_score: _feedback_brief(current_overall_score=82.0, weighted_score=82.0),
    )

    report = loop.run_closed_loop(
        song_a="song_a.wav",
        song_b="song_b.wav",
        references=["ref.wav"],
        output_root=str(tmp_path / "loop_batch"),
        batch_size=3,
        shortlist=1,
        max_iterations=2,
        quality_gate=80.0,
    )

    assert report["stop_reason"] == "quality_gate_reached:80.0"
    assert report["best_iteration"]["candidate_input"].endswith("candidate_002")
    iteration = report["iterations"][0]
    assert iteration["render"]["selection_mode"] == "listener_batch"
    assert iteration["render"]["selection_lane"] == "recommended_shortlist"
    assert iteration["candidate_input"].endswith("candidate_002")
    assert iteration["artifacts"]["auto_shortlist_report"]["exists"] is True


def test_closed_loop_batch_mode_falls_back_to_closest_miss(monkeypatch, tmp_path: Path) -> None:
    def fake_render_iteration(*_args, **_kwargs):
        pytest.fail("single-candidate render path should not run when batch_size > 1")

    def fake_auto_shortlist_fusion(song_a: str, song_b: str, output_root: str, **kwargs) -> int:
        assert kwargs["batch_size"] == 2
        outdir = Path(output_root)
        candidate_001 = outdir / "candidate_001"
        candidate_002 = outdir / "candidate_002"
        candidate_001.mkdir(parents=True, exist_ok=True)
        candidate_002.mkdir(parents=True, exist_ok=True)
        _write_auto_shortlist_report(
            outdir / "auto_shortlist_report.json",
            {
                "schema_version": "0.2.0",
                "listener_agent_report": {"counts": {"survivors": 0, "borderline": 1, "rejected": 1}},
                "recommended_shortlist": [],
                "closest_misses": [
                    {
                        "candidate_id": "candidate_002",
                        "run_dir": str(candidate_002),
                        "overall_score": 67.0,
                        "listener_rank": 63.0,
                        "decision": "borderline",
                        "verdict": "mixed",
                    }
                ],
                "candidates": [
                    {
                        "candidate_id": "candidate_001",
                        "run_dir": str(candidate_001),
                        "overall_score": 46.0,
                        "listener_rank": 35.0,
                        "decision": "reject",
                        "verdict": "poor",
                    },
                    {
                        "candidate_id": "candidate_002",
                        "run_dir": str(candidate_002),
                        "overall_score": 67.0,
                        "listener_rank": 63.0,
                        "decision": "borderline",
                        "verdict": "mixed",
                    },
                ],
            },
        )
        return 0

    monkeypatch.setattr(loop, "render_iteration", fake_render_iteration)
    monkeypatch.setattr(loop.ai_dj, "auto_shortlist_fusion", fake_auto_shortlist_fusion)
    monkeypatch.setattr(
        loop,
        "_candidate_report",
        lambda candidate_input: {"overall_score": 67.0, "verdict": "mixed"} if candidate_input.endswith("candidate_002") else {"overall_score": 46.0, "verdict": "poor"},
    )
    monkeypatch.setattr(
        loop,
        "_candidate_listener_assessment",
        lambda candidate_input: {"decision": "borderline", "listener_rank": 63.0} if candidate_input.endswith("candidate_002") else {"decision": "reject", "listener_rank": 35.0},
    )
    monkeypatch.setattr(
        loop,
        "build_feedback_brief",
        lambda candidate_input, references, target_score: _feedback_brief(current_overall_score=67.0, weighted_score=67.0),
    )

    report = loop.run_closed_loop(
        song_a="song_a.wav",
        song_b="song_b.wav",
        references=["ref.wav"],
        output_root=str(tmp_path / "loop_closest_miss"),
        batch_size=2,
        shortlist=1,
        max_iterations=1,
        quality_gate=85.0,
    )

    assert report["stop_reason"] == "max_iterations:1"
    iteration = report["iterations"][0]
    assert iteration["render"]["selection_mode"] == "listener_batch"
    assert iteration["render"]["selection_lane"] == "closest_miss"
    assert iteration["candidate_input"].endswith("candidate_002")
    assert report["best_iteration"]["candidate_input"].endswith("candidate_002")
