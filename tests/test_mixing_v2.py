from __future__ import annotations

import json
from pathlib import Path

from arrangement_plan import validate_arrangement_plan
from mixing_v2 import (
    ARTIFACT_ARRANGEMENT,
    ARTIFACT_SCORE,
    ARTIFACT_SUMMARY,
    ARTIFACT_TRANSITIONS,
    main,
    orchestrate_mix,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_demo_orchestration_is_deterministic(tmp_path: Path) -> None:
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    result1 = orchestrate_mix("demo_a", "demo_b", output_dir=str(out1), demo=True)
    result2 = orchestrate_mix("demo_a", "demo_b", output_dir=str(out2), demo=True)

    plan1 = _load_json(out1 / ARTIFACT_ARRANGEMENT)
    plan2 = _load_json(out2 / ARTIFACT_ARRANGEMENT)
    transitions1 = _load_json(out1 / ARTIFACT_TRANSITIONS)
    transitions2 = _load_json(out2 / ARTIFACT_TRANSITIONS)
    score1 = _load_json(out1 / ARTIFACT_SCORE)
    score2 = _load_json(out2 / ARTIFACT_SCORE)

    assert plan1 == plan2
    assert transitions1 == transitions2
    assert score1 == score2
    assert result1["score"] == result2["score"]


def test_demo_outputs_validate_and_artifacts_written(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo_artifacts"
    orchestrate_mix("demo_a", "demo_b", output_dir=str(out_dir), demo=True)

    arrangement_path = out_dir / ARTIFACT_ARRANGEMENT
    transition_path = out_dir / ARTIFACT_TRANSITIONS
    score_path = out_dir / ARTIFACT_SCORE
    summary_path = out_dir / ARTIFACT_SUMMARY

    assert arrangement_path.exists()
    assert transition_path.exists()
    assert score_path.exists()
    assert summary_path.exists()

    arrangement = _load_json(arrangement_path)
    ok, errors = validate_arrangement_plan(arrangement)
    assert ok is True
    assert errors == []

    transitions = _load_json(transition_path)
    assert isinstance(transitions, list)
    assert transitions
    assert all("instruction" in row for row in transitions)

    score = _load_json(score_path)
    assert "score" in score
    assert "overall_score" in score["score"]

    summary = summary_path.read_text(encoding="utf-8")
    assert "VocalFusion Mixing v2 Orchestration Summary" in summary


def test_cli_demo_mode_no_crash(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli_demo"
    exit_code = main(["demo", "--output", str(output_dir)])
    assert exit_code == 0

    for artifact in (ARTIFACT_ARRANGEMENT, ARTIFACT_TRANSITIONS, ARTIFACT_SCORE, ARTIFACT_SUMMARY):
        assert (output_dir / artifact).exists()
