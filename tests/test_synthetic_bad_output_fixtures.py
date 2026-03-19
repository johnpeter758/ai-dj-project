from __future__ import annotations

import json
from pathlib import Path

import ai_dj
from scripts.listen_feedback_loop import build_feedback_brief
from tests.synthetic_listener_fixtures import (
    near_threshold_false_positive_review_fixture,
    not_one_song_composite_fixture,
    owner_switch_ping_pong_fixture,
    payoff_conviction_gap_fixture,
    section_role_confusion_fixture,
    unstable_groove_review_fixture,
)


def test_synthetic_owner_switch_ping_pong_fixture_hard_rejects_listener_agent(tmp_path: Path):
    case_path = tmp_path / "owner_switch_ping_pong.json"
    out_path = tmp_path / "listener_agent.json"
    case_path.write_text(json.dumps(owner_switch_ping_pong_fixture()), encoding="utf-8")

    rc = ai_dj.listener_agent([str(case_path)], str(out_path), shortlist=1)
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    rejected = payload["rejected"][0]
    assert rejected["decision"] == "reject"
    assert "hard listen gate rejected the track" in rejected["hard_fail_reasons"]
    assert "listener verdict says the output is weak or poor" in rejected["hard_fail_reasons"]
    assert "transitions still read like track switching" in rejected["hard_fail_reasons"]
    assert "too many section-owner flips still read like track switching" in rejected["hard_fail_reasons"]
    assert rejected["gate_status"] == "reject"
    assert rejected["hard_reject_signals"]["owner_switch_ratio"] == 0.91


def test_synthetic_not_one_song_composite_fixture_hard_rejects_listener_agent(tmp_path: Path):
    case_path = tmp_path / "not_one_song_composite.json"
    out_path = tmp_path / "listener_agent.json"
    case_path.write_text(json.dumps(not_one_song_composite_fixture()), encoding="utf-8")

    rc = ai_dj.listener_agent([str(case_path)], str(out_path), shortlist=1)
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    rejected = payload["rejected"][0]
    assert rejected["decision"] == "reject"
    assert "whole-song backbone continuity is too weak" in rejected["hard_fail_reasons"]
    assert rejected["hard_reject_signals"]["composite_song_risk"] == 0.69


def test_synthetic_unstable_groove_review_fixture_stays_out_of_human_shortlist(tmp_path: Path):
    case_path = tmp_path / "unstable_groove_review.json"
    out_path = tmp_path / "listener_agent.json"
    case_path.write_text(json.dumps(unstable_groove_review_fixture()), encoding="utf-8")

    rc = ai_dj.listener_agent([str(case_path)], str(out_path), shortlist=1)
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    borderline = payload["borderline"][0]
    assert payload["recommended_for_human_review"] == []
    assert borderline["decision"] == "borderline"
    assert borderline["gate_status"] == "review"
    assert borderline["acceptance_checks"]["listen_gate"]["passed"] is False
    assert borderline["acceptance_checks"]["survivor_minimums"]["groove"]["actual"] == 50.0
    assert borderline["acceptance_checks"]["survivor_minimums"]["groove"]["passed"] is False



def test_synthetic_near_threshold_review_fixture_does_not_false_positive_hard_reject(tmp_path: Path):
    case_path = tmp_path / "near_threshold_review.json"
    out_path = tmp_path / "listener_agent.json"
    case_path.write_text(json.dumps(near_threshold_false_positive_review_fixture()), encoding="utf-8")

    rc = ai_dj.listener_agent([str(case_path)], str(out_path), shortlist=1)
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["recommended_for_human_review"] == []
    assert payload["rejected"] == []
    borderline = payload["borderline"][0]
    assert borderline["decision"] == "borderline"
    assert borderline["gate_status"] == "review"
    assert borderline["acceptance_checks"]["listen_gate"]["passed"] is False
    assert borderline["hard_fail_reasons"] == []
    assert borderline["hard_reject_signals"]["backbone_continuity"] == 0.42
    assert borderline["hard_reject_signals"]["recognizable_section_ratio"] == 0.4
    assert borderline["hard_reject_signals"]["boundary_recovery"] == 0.38
    assert borderline["hard_reject_signals"]["role_plausibility"] == 0.4
    assert borderline["hard_reject_signals"]["background_only_identity_gap"] == 0.45
    assert borderline["hard_reject_signals"]["owner_switch_ratio"] == 0.78
    assert borderline["hard_reject_signals"]["composite_song_risk"] == 0.5



def test_synthetic_payoff_conviction_fixture_maps_to_energy_arc_feedback(tmp_path: Path):
    candidate = tmp_path / "payoff_conviction_gap.json"
    reference = tmp_path / "reference.json"
    candidate.write_text(json.dumps(payoff_conviction_gap_fixture()), encoding="utf-8")
    reference.write_text(
        json.dumps(
            {
                **payoff_conviction_gap_fixture(),
                "overall_score": 90.0,
                "verdict": "promising",
                "gate_status": "pass",
                "energy_arc": {
                    "score": 92.0,
                    "summary": "late payoff lands",
                    "evidence": [],
                    "fixes": [],
                    "details": {},
                },
                "song_likeness": {
                    "score": 88.0,
                    "summary": "song_likeness",
                    "evidence": [],
                    "fixes": [],
                    "details": {
                        "aggregate_metrics": {
                            "backbone_continuity": 0.80,
                            "readable_section_ratio": 0.78,
                            "recognizable_section_ratio": 0.77,
                            "boundary_recovery": 0.75,
                            "role_plausibility": 0.76,
                            "planner_audio_climax_conviction": 0.81,
                            "climax_conviction": 0.83,
                            "background_only_identity_gap": 0.07,
                            "owner_switch_ratio": 0.24,
                        }
                    },
                },
                "gating": {"status": "pass", "raw_overall_score": 90.0},
            }
        ),
        encoding="utf-8",
    )

    brief = build_feedback_brief(str(candidate), [str(reference)])

    failure_modes = {item["failure_mode"] for item in brief["planner_feedback_map"]}
    assert "late_payoff_mapping" in failure_modes
    payoff = next(item for item in brief["planner_feedback_map"] if item["failure_mode"] == "late_payoff_mapping")
    assert payoff["component"] == "energy_arc"
    assert any("build a real payoff" in text.lower() or "late payoff" in text.lower() for text in payoff["matched_feedback"])



def test_synthetic_section_role_confusion_fixture_maps_to_structure_feedback(tmp_path: Path):
    candidate = tmp_path / "section_role_confusion.json"
    reference = tmp_path / "reference.json"
    candidate.write_text(json.dumps(section_role_confusion_fixture()), encoding="utf-8")
    reference.write_text(
        json.dumps(
            {
                **section_role_confusion_fixture(),
                "overall_score": 91.0,
                "verdict": "promising",
                "structure": {"score": 90.0, "summary": "structure", "evidence": [], "fixes": [], "details": {}},
                "song_likeness": {
                    "score": 89.0,
                    "summary": "song_likeness",
                    "evidence": [],
                    "fixes": [],
                    "details": {
                        "aggregate_metrics": {
                            "backbone_continuity": 0.82,
                            "recognizable_section_ratio": 0.79,
                            "boundary_recovery": 0.76,
                            "role_plausibility": 0.78,
                            "planner_audio_climax_conviction": 0.74,
                            "background_only_identity_gap": 0.07,
                            "owner_switch_ratio": 0.24,
                        }
                    },
                },
                "gating": {"status": "pass", "raw_overall_score": 91.0},
            }
        ),
        encoding="utf-8",
    )

    brief = build_feedback_brief(str(candidate), [str(reference)])

    failure_modes = {item["failure_mode"] for item in brief["planner_feedback_map"]}
    assert "section_readability" in failure_modes
    assert "backbone_continuity" in failure_modes
    readability = next(item for item in brief["planner_feedback_map"] if item["failure_mode"] == "section_readability")
    assert readability["component"] == "structure"
    assert any("coarse whole-song windows" in text.lower() for text in readability["matched_feedback"])
