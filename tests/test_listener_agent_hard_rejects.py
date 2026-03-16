from __future__ import annotations

import json
from pathlib import Path

import ai_dj


def _report(
    overall: float,
    *,
    verdict: str = "promising",
    gate_status: str = "pass",
    song_likeness: float = 72.0,
    groove: float = 72.0,
    energy_arc: float = 72.0,
    transition: float = 72.0,
    coherence: float = 72.0,
    mix_sanity: float = 72.0,
    song_metrics: dict | None = None,
) -> dict:
    aggregate_metrics = {
        "backbone_continuity": 0.72,
        "readable_section_ratio": 0.73,
        "recognizable_section_ratio": 0.71,
        "boundary_recovery": 0.68,
        "role_plausibility": 0.67,
        "planner_audio_climax_conviction": 0.66,
        "background_only_identity_gap": 0.08,
        "owner_switch_ratio": 0.32,
    }
    if song_metrics:
        aggregate_metrics.update(song_metrics)
    return {
        "source_path": "case.wav",
        "duration_seconds": 60.0,
        "overall_score": overall,
        "structure": {"score": 74.0, "summary": "readable", "evidence": [], "fixes": [], "details": {}},
        "groove": {"score": groove, "summary": "steady", "evidence": [], "fixes": [], "details": {}},
        "energy_arc": {"score": energy_arc, "summary": "lands", "evidence": [], "fixes": [], "details": {}},
        "transition": {"score": transition, "summary": "musical", "evidence": [], "fixes": [], "details": {}},
        "coherence": {"score": coherence, "summary": "coherent", "evidence": [], "fixes": [], "details": {}},
        "mix_sanity": {"score": mix_sanity, "summary": "clear", "evidence": [], "fixes": [], "details": {}},
        "song_likeness": {
            "score": song_likeness,
            "summary": "mostly song-like",
            "evidence": [],
            "fixes": [],
            "details": {"aggregate_metrics": aggregate_metrics},
        },
        "verdict": verdict,
        "top_reasons": ["Candidate has some promise."],
        "top_fixes": [],
        "gating": {"status": gate_status, "raw_overall_score": overall},
        "analysis_version": "0.5.0",
    }


def test_listener_agent_hard_rejects_structural_non_song_even_when_topline_scores_are_borderline_ok(tmp_path: Path):
    case_path = tmp_path / "structural_fake_survivor.json"
    out_path = tmp_path / "listener_agent.json"
    case_path.write_text(
        json.dumps(
            _report(
                74.0,
                song_likeness=62.0,
                groove=70.0,
                energy_arc=68.0,
                transition=66.0,
                coherence=70.0,
                mix_sanity=69.0,
                song_metrics={
                    "backbone_continuity": 0.36,
                    "readable_section_ratio": 0.31,
                    "recognizable_section_ratio": 0.34,
                    "boundary_recovery": 0.32,
                    "role_plausibility": 0.35,
                },
            )
        ),
        encoding="utf-8",
    )

    rc = ai_dj.listener_agent([str(case_path)], str(out_path), shortlist=1)
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    rejected = payload["rejected"][0]
    assert rejected["label"] == "structural_fake_survivor.json"
    assert rejected["decision"] == "reject"
    assert rejected["acceptance_checks"]["survivor_minimums"]["song_likeness"]["passed"] is True
    assert rejected["acceptance_checks"]["listen_gate"]["passed"] is True
    assert "whole-song backbone continuity is too weak" in rejected["hard_fail_reasons"]
    assert "section readability is too weak to trust as one arrangement" in rejected["hard_fail_reasons"]
    assert rejected["hard_reject_signals"]["backbone_continuity"] == 0.36
    assert payload["recommended_for_human_review"] == []


def test_listener_agent_hard_rejects_background_only_identity_collapse(tmp_path: Path):
    case_path = tmp_path / "background_only_glue.json"
    out_path = tmp_path / "listener_agent.json"
    case_path.write_text(
        json.dumps(
            _report(
                76.0,
                song_likeness=64.0,
                groove=74.0,
                energy_arc=73.0,
                transition=70.0,
                coherence=71.0,
                mix_sanity=72.0,
                song_metrics={
                    "background_only_identity_gap": 0.58,
                    "owner_switch_ratio": 0.44,
                },
            )
        ),
        encoding="utf-8",
    )

    rc = ai_dj.listener_agent([str(case_path)], str(out_path), shortlist=1)
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    rejected = payload["rejected"][0]
    assert rejected["decision"] == "reject"
    assert "fusion identity is mostly background-only glue" in rejected["hard_fail_reasons"]
    assert rejected["hard_reject_signals"]["background_only_identity_gap"] == 0.58
