from __future__ import annotations

from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = (
    "project_id",
    "parents",
    "target",
    "duration_seconds",
    "child_sections",
    "transitions",
    "automation_fx_instructions",
    "muting_schedule",
    "refinement_notes",
)


def default_plan_skeleton(
    *,
    project_id: str = "song-birth",
    parent_a_id: str = "song_a",
    parent_b_id: str = "song_b",
    bpm_target: float = 120.0,
    key_target: str = "C major",
    duration_seconds: float = 180.0,
    child_sections: list[dict[str, Any]] | None = None,
) -> dict:
    """Return a JSON-friendly arrangement plan scaffold.

    Covers the phase-5 checklist fields:
    project id, parent refs, bpm/key target, duration, child sections,
    active stems, transitions, automation/fx instructions, muting schedule,
    and refinement notes.
    """

    sections = child_sections if child_sections is not None else []

    return {
        "project_id": project_id,
        "parents": {
            "song_a": {"id": parent_a_id},
            "song_b": {"id": parent_b_id},
        },
        "target": {
            "bpm": float(bpm_target),
            "key": key_target,
        },
        "duration_seconds": float(duration_seconds),
        "child_sections": sections,
        "transitions": [],
        "automation_fx_instructions": [],
        "muting_schedule": [],
        "refinement_notes": [],
    }


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_arrangement_plan(plan: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not isinstance(plan, dict):
        return False, ["plan must be a dictionary"]

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in plan:
            errors.append(f"missing top-level field: {field}")

    parents = plan.get("parents")
    if not isinstance(parents, dict):
        errors.append("parents must be a dictionary")
    else:
        for pid in ("song_a", "song_b"):
            if pid not in parents:
                errors.append(f"parents missing: {pid}")

    target = plan.get("target")
    if not isinstance(target, dict):
        errors.append("target must be a dictionary")
    else:
        bpm = target.get("bpm")
        key = target.get("key")
        if not isinstance(bpm, (int, float)) or bpm <= 0:
            errors.append("target.bpm must be a positive number")
        if not _is_nonempty_str(key):
            errors.append("target.key must be a non-empty string")

    duration = plan.get("duration_seconds")
    if not isinstance(duration, (int, float)) or duration <= 0:
        errors.append("duration_seconds must be a positive number")

    sections = plan.get("child_sections")
    if not isinstance(sections, list) or len(sections) == 0:
        errors.append("child_sections must be a non-empty list")
    else:
        for idx, section in enumerate(sections):
            path = f"child_sections[{idx}]"
            if not isinstance(section, dict):
                errors.append(f"{path} must be a dictionary")
                continue

            required_section_fields = (
                "section_id",
                "start",
                "end",
                "purpose",
                "primary_focus",
                "active_stems",
                "dominant_parent_balance",
                "reason_for_existence",
            )
            for f in required_section_fields:
                if f not in section:
                    errors.append(f"{path} missing field: {f}")

            start = section.get("start")
            end = section.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                errors.append(f"{path} start/end must be numeric")
            elif end <= start:
                errors.append(f"{path} end must be greater than start")

            if not _is_nonempty_str(section.get("purpose")):
                errors.append(f"{path} purpose must be non-empty")
            if not _is_nonempty_str(section.get("primary_focus")):
                errors.append(f"{path} primary_focus must be non-empty")
            if not _is_nonempty_str(section.get("reason_for_existence")):
                errors.append(f"{path} reason_for_existence must be non-empty")

            active_stems = section.get("active_stems")
            if not isinstance(active_stems, list) or len(active_stems) == 0:
                errors.append(f"{path} active_stems must be a non-empty list")

            balance = section.get("dominant_parent_balance")
            if not _is_nonempty_str(balance):
                errors.append(f"{path} dominant_parent_balance must be non-empty")

    for list_field in ("transitions", "automation_fx_instructions", "muting_schedule", "refinement_notes"):
        if list_field in plan and not isinstance(plan[list_field], list):
            errors.append(f"{list_field} must be a list")

    transitions = plan.get("transitions", [])
    if isinstance(transitions, list):
        for idx, t in enumerate(transitions):
            path = f"transitions[{idx}]"
            if not isinstance(t, dict):
                errors.append(f"{path} must be a dictionary")
                continue
            for field in ("from_section", "to_section", "intent"):
                if not _is_nonempty_str(t.get(field)):
                    errors.append(f"{path}.{field} must be a non-empty string")

    return (len(errors) == 0), errors
