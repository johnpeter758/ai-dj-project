from __future__ import annotations

from arrangement_plan import default_plan_skeleton, validate_arrangement_plan


def _valid_section(idx: int, start: float, end: float) -> dict:
    return {
        "section_id": f"section_{idx}",
        "start": start,
        "end": end,
        "purpose": "develop arrangement arc",
        "primary_focus": "vocal",
        "active_stems": ["drums_a", "vocal_b"],
        "dominant_parent_balance": "A:70/B:30",
        "reason_for_existence": "Maintain continuity while introducing contrast.",
    }


def test_default_plan_skeleton_covers_phase_5_fields() -> None:
    plan = default_plan_skeleton(
        project_id="phase5-check",
        parent_a_id="track-A",
        parent_b_id="track-B",
        bpm_target=125.0,
        key_target="D minor",
        duration_seconds=192.0,
        child_sections=[_valid_section(1, 0.0, 48.0)],
    )

    for key in (
        "project_id",
        "parents",
        "target",
        "duration_seconds",
        "child_sections",
        "transitions",
        "automation_fx_instructions",
        "muting_schedule",
        "refinement_notes",
    ):
        assert key in plan

    assert plan["project_id"] == "phase5-check"
    assert plan["parents"]["song_a"]["id"] == "track-A"
    assert plan["parents"]["song_b"]["id"] == "track-B"
    assert plan["target"]["bpm"] == 125.0
    assert plan["target"]["key"] == "D minor"


def test_validate_arrangement_plan_accepts_valid_payload() -> None:
    plan = default_plan_skeleton(
        child_sections=[_valid_section(1, 0.0, 60.0), _valid_section(2, 60.0, 120.0)]
    )
    plan["transitions"] = [
        {"from_section": "section_1", "to_section": "section_2", "intent": "energy lift"}
    ]
    plan["automation_fx_instructions"] = [{"time": 58.0, "action": "filter_sweep"}]
    plan["muting_schedule"] = [{"section": "section_1", "mute": ["bass_b"]}]
    plan["refinement_notes"] = ["Tighten pre-drop transition by 1 bar."]

    ok, errors = validate_arrangement_plan(plan)
    assert ok is True
    assert errors == []


def test_validate_arrangement_plan_flags_schema_issues() -> None:
    bad_plan = {
        "project_id": "x",
        "parents": {"song_a": {"id": "a"}},
        "target": {"bpm": 0, "key": ""},
        "duration_seconds": -1,
        "child_sections": [
            {
                "section_id": "section_1",
                "start": 30,
                "end": 20,
                "purpose": "",
                "primary_focus": "",
                "active_stems": [],
                "dominant_parent_balance": "",
                "reason_for_existence": "",
            }
        ],
        "transitions": [{"from_section": "section_1", "to_section": "", "intent": ""}],
        "automation_fx_instructions": {},
        "muting_schedule": [],
        "refinement_notes": [],
    }

    ok, errors = validate_arrangement_plan(bad_plan)
    assert ok is False
    assert errors
    assert any("parents missing: song_b" in e for e in errors)
    assert any("target.bpm" in e for e in errors)
    assert any("duration_seconds" in e for e in errors)
    assert any("end must be greater than start" in e for e in errors)
    assert any("automation_fx_instructions must be a list" in e for e in errors)
