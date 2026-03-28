"""Deterministic transition-planning helpers for VocalFusion.

This module intentionally stays at planner level (instructions/metadata),
not heavy sample-domain DSP.
"""

from __future__ import annotations

from typing import Any

TRANSITION_TYPES: tuple[str, ...] = (
    "hard_cut",
    "filtered_handoff",
    "drum_fill",
    "riser_drop",
    "vocal_tail",
    "bass_dropout_slam",
    "sweep",
    "echo_exit",
    "stutter",
    "tension_mute",
    "kick_only_bridge",
    "acapella_spotlight",
)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _stem_role(stem: dict[str, Any]) -> str:
    role = str(stem.get("role", "")).lower()
    if role:
        return role
    name = str(stem.get("name", "")).lower()
    if "vocal" in name:
        return "vocal"
    if "bass" in name or "sub" in name:
        return "bass"
    if "kick" in name or "drum" in name or "perc" in name:
        return "drums"
    return "music"


def choose_transition_type(context: dict) -> str:
    """Pick a transition type deterministically from section/context metadata."""
    preferred = context.get("preferred_transition")
    if isinstance(preferred, str) and preferred in TRANSITION_TYPES:
        return preferred

    focus = str(context.get("focus", "")).lower()
    from_has_vocal = bool(context.get("from_has_vocal") or focus == "vocal")
    to_has_vocal = bool(context.get("to_has_vocal") or focus == "vocal")
    energy_delta = float(context.get("energy_delta", 0.0))
    bpm_delta = abs(float(context.get("bpm_delta", 0.0)))
    bars = int(context.get("bars", 8))
    dramatic = bool(context.get("dramatic", False))
    end_of_phrase = bool(context.get("end_of_phrase", True))

    if to_has_vocal and not from_has_vocal:
        return "acapella_spotlight"
    if from_has_vocal and not to_has_vocal:
        return "vocal_tail"

    if bpm_delta >= 8:
        return "hard_cut"
    if energy_delta >= 0.35:
        return "riser_drop" if dramatic else "drum_fill"
    if energy_delta <= -0.35:
        return "bass_dropout_slam" if dramatic else "tension_mute"

    if bars <= 2:
        return "stutter" if dramatic else "hard_cut"
    if bars <= 4:
        return "echo_exit" if end_of_phrase else "sweep"

    if bool(context.get("needs_rhythm_bridge", False)):
        return "kick_only_bridge"
    if bool(context.get("needs_filter_glue", True)):
        return "filtered_handoff"
    return "sweep"


def build_transition_instruction(from_section: dict, to_section: dict) -> dict:
    """Build deterministic instruction payload consumed by higher-level renderers."""
    from_energy = float(from_section.get("energy", 0.5))
    to_energy = float(to_section.get("energy", 0.5))
    bars = _clamp(int(to_section.get("transition_bars", from_section.get("transition_bars", 8))), 1, 16)
    context = {
        "from_has_vocal": bool(from_section.get("has_vocal", False)),
        "to_has_vocal": bool(to_section.get("has_vocal", False)),
        "energy_delta": to_energy - from_energy,
        "bpm_delta": float(to_section.get("bpm", 0.0)) - float(from_section.get("bpm", 0.0)),
        "bars": bars,
        "dramatic": bool(to_section.get("dramatic", False) or from_section.get("dramatic", False)),
        "focus": to_section.get("focus", from_section.get("focus", "")),
        "needs_filter_glue": abs(to_energy - from_energy) < 0.25,
        "needs_rhythm_bridge": bool(to_section.get("needs_rhythm_bridge", False)),
        "preferred_transition": to_section.get("preferred_transition") or from_section.get("preferred_transition"),
        "end_of_phrase": bool(from_section.get("end_of_phrase", True)),
    }
    transition_type = choose_transition_type(context)

    actions: list[dict[str, Any]] = []
    if transition_type in {"filtered_handoff", "sweep"}:
        actions.append({"op": "filter", "target": "music_bus", "curve": "hp_up_then_release", "bars": bars})
    if transition_type in {"drum_fill", "kick_only_bridge"}:
        actions.append({"op": "rhythm_focus", "target": "drums", "bars": min(2, bars)})
    if transition_type == "riser_drop":
        actions.append({"op": "fx", "target": "transition", "effect": "riser", "length_bars": bars})
    if transition_type == "vocal_tail":
        actions.append({"op": "fx", "target": "outgoing_vocal", "effect": "delay_tail", "beats": 2})
    if transition_type == "echo_exit":
        actions.append({"op": "fx", "target": "outgoing_master", "effect": "echo_freeze", "beats": 1})
    if transition_type == "bass_dropout_slam":
        actions.append({"op": "mute", "target": "bass", "for_beats": 2})
    if transition_type == "tension_mute":
        actions.append({"op": "mute", "target": "music", "for_beats": 1})
    if transition_type == "stutter":
        actions.append({"op": "chop", "target": "master", "rate": "1/8", "beats": 1})
    if transition_type == "acapella_spotlight":
        actions.append({"op": "duck", "target": "music", "db": -6, "bars": min(2, bars)})

    return {
        "type": transition_type,
        "bars": bars,
        "from_section": {"name": from_section.get("name"), "energy": from_energy},
        "to_section": {"name": to_section.get("name"), "energy": to_energy},
        "actions": actions,
    }


def pre_transition_cleanup(active_stems: list[dict], context: dict) -> list[dict]:
    """Trim dense arrangements before a transition (anti-mud prep)."""
    transition_type = choose_transition_type(context)
    cleaned: list[dict[str, Any]] = [dict(stem) for stem in active_stems]

    tonal_indices: list[int] = []
    bass_indices: list[int] = []
    for idx, stem in enumerate(cleaned):
        role = _stem_role(stem)
        stem.setdefault("state", "active")
        if role in {"music", "vocal"}:
            tonal_indices.append(idx)
        if role == "bass":
            bass_indices.append(idx)

    # Keep arrangement readable: max 3 tonal layers pre-transition.
    if len(tonal_indices) > 3:
        overflow = tonal_indices[3:]
        for idx in overflow:
            cleaned[idx]["state"] = "ducked"
            cleaned[idx]["gain_db"] = min(cleaned[idx].get("gain_db", 0), -8)

    # One bass at a time; keep highest-priority bass stem.
    if len(bass_indices) > 1:
        scored = sorted(
            bass_indices,
            key=lambda i: (
                -float(cleaned[i].get("priority", 0)),
                str(cleaned[i].get("name", "")),
            ),
        )
        for idx in scored[1:]:
            cleaned[idx]["state"] = "muted"
            cleaned[idx]["gain_db"] = -96

    if transition_type in {"bass_dropout_slam", "tension_mute", "acapella_spotlight"}:
        for stem in cleaned:
            role = _stem_role(stem)
            if transition_type == "bass_dropout_slam" and role == "bass":
                stem["state"] = "muted"
                stem["gain_db"] = -96
            if transition_type == "tension_mute" and role == "music":
                stem["state"] = "muted"
                stem["gain_db"] = -96
            if transition_type == "acapella_spotlight" and role == "music":
                stem["state"] = "ducked"
                stem["gain_db"] = min(stem.get("gain_db", 0), -10)

    return cleaned


def post_transition_stabilization(active_stems: list[dict], context: dict) -> list[dict]:
    """Bring arrangement back to stable groove after transition."""
    stabilized: list[dict[str, Any]] = [dict(stem) for stem in active_stems]
    focus = str(context.get("focus", "")).lower()

    # Ensure drum + bass backbone recovers if present.
    for stem in stabilized:
        role = _stem_role(stem)
        if role in {"drums", "bass"} and stem.get("state") == "muted":
            stem["state"] = "active"
            stem["gain_db"] = -2 if role == "bass" else -1

    # Keep one primary focus, ducking non-focus tonal stems slightly.
    if focus:
        focused = False
        for stem in stabilized:
            role = _stem_role(stem)
            name = str(stem.get("name", "")).lower()
            is_focus = focus in role or focus in name
            if is_focus and not focused:
                stem["state"] = "active"
                stem["gain_db"] = max(float(stem.get("gain_db", 0)), 0)
                stem["primary_focus"] = True
                focused = True
            elif role in {"music", "vocal"} and not is_focus:
                stem["state"] = "ducked"
                stem["gain_db"] = min(float(stem.get("gain_db", 0)), -4)
                stem["primary_focus"] = False

    return stabilized
