from __future__ import annotations

from ai_dj_timeline import build_timeline_plan


SONG_A_DNA = {
    "duration_seconds": 200.0,
    "tempo_bpm": 124.0,
    "key_tonic": "A",
    "sections": [
        {"start": 0.0, "end": 32.0},
        {"start": 32.0, "end": 64.0},
        {"start": 64.0, "end": 96.0},
        {"start": 96.0, "end": 128.0},
        {"start": 128.0, "end": 160.0},
        {"start": 160.0, "end": 200.0},
    ],
    "vocal_density_map": [0.7] * 16,
}

SONG_B_DNA = {
    "duration_seconds": 180.0,
    "tempo_bpm": 126.0,
    "key_tonic": "C#",
    "sections": [
        {"start": 0.0, "end": 30.0},
        {"start": 30.0, "end": 60.0},
        {"start": 60.0, "end": 90.0},
        {"start": 90.0, "end": 120.0},
        {"start": 120.0, "end": 150.0},
        {"start": 150.0, "end": 180.0},
    ],
    "vocal_density_map": [0.4] * 16,
}


def test_build_timeline_plan_is_deterministic() -> None:
    first = build_timeline_plan(SONG_A_DNA, SONG_B_DNA, project_id="vf-test")
    second = build_timeline_plan(SONG_A_DNA, SONG_B_DNA, project_id="vf-test")
    assert first == second


def test_build_timeline_plan_enforces_hard_rules() -> None:
    plan = build_timeline_plan(SONG_A_DNA, SONG_B_DNA)

    assert plan["hard_rules"]["no_full_song_overlay"] is True
    assert plan["hard_rules"]["one_primary_focus_per_section"] is True
    assert plan["hard_rules"]["explicit_purpose_per_section"] is True
    assert plan["hard_rules"]["boundary_transition_intent_required"] is True

    sections = plan["sections"]
    assert len(sections) >= 4

    for section in sections:
        assert isinstance(section["primary_focus"], str) and section["primary_focus"].strip()
        assert isinstance(section["purpose"], str) and section["purpose"].strip()
        assert isinstance(section["reason_for_existence"], str) and section["reason_for_existence"].strip()
        assert isinstance(section["dominant_parent_balance"], str) and section["dominant_parent_balance"].strip()
        assert "No full-song overlay" in section["overlay_guard"]
        assert isinstance(section["exit_transition_intent"], str) and section["exit_transition_intent"].strip()
        assert section["end"] > section["start"]

    # no full-song overlay at section level: each section should be a subset of full duration
    total_duration = plan["target"]["duration_seconds"]
    assert all((s["end"] - s["start"]) < total_duration for s in sections)

    # transition boundaries include explicit intent
    transitions = plan["transitions"]
    assert len(transitions) == len(sections) - 1
    for t in transitions:
        assert isinstance(t["intent"], str) and t["intent"].strip()
        assert t["boundary_time"] > 0


def test_balance_is_reported() -> None:
    plan = build_timeline_plan(SONG_A_DNA, SONG_B_DNA)
    balance = plan["dominant_parent_balance"]
    assert set(balance.keys()) == {"A", "B"}
    assert 0.0 <= balance["A"] <= 1.0
    assert 0.0 <= balance["B"] <= 1.0
    assert round(balance["A"] + balance["B"], 6) == 1.0
