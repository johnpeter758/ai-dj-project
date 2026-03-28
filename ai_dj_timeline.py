from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _section_span(section: dict[str, Any], fallback_start: float, fallback_end: float) -> tuple[float, float]:
    start = _safe_float(section.get("start"), fallback_start)
    end = _safe_float(section.get("end"), fallback_end)
    if end <= start:
        end = start + max(4.0, fallback_end - fallback_start)
    return round(start, 3), round(end, 3)


def _transition_intent(index: int, total: int) -> str:
    if index == 0:
        return "cold-open intro to establish the child identity"
    if index == total - 2:
        return "energy convergence to set up the final statement"
    if index == total - 1:
        return "resolved outro with clean release"
    return "phrase-safe handoff with contrast and continuity"


def _purpose(index: int, total: int, dominant_parent: str) -> str:
    if index == 0:
        return f"establish core motif from parent {dominant_parent}"
    if index == total - 1:
        return "deliver closure and make ending unmistakable"
    if index == total // 2:
        return "peak identity moment of the child arrangement"
    return "develop tension and keep narrative movement"


def _focus(song_a: dict[str, Any], song_b: dict[str, Any], dominant_parent: str) -> str:
    a_vocal = sum(song_a.get("vocal_density_map", [])[:8])
    b_vocal = sum(song_b.get("vocal_density_map", [])[:8])
    if dominant_parent == "A":
        return "vocal" if a_vocal >= b_vocal else "rhythm"
    return "vocal" if b_vocal >= a_vocal else "rhythm"


def build_timeline_plan(song_a_dna: dict, song_b_dna: dict, *, project_id: str = "song-birth") -> dict:
    """Build a deterministic section-level timeline plan for Song Birth arrangement.

    Hard rules enforced in output:
    - no full-song overlay
    - one primary focus per section
    - every section has an explicit purpose
    - section boundaries include meaningful transition intent
    """

    duration_a = _safe_float(song_a_dna.get("duration_seconds"), 180.0)
    duration_b = _safe_float(song_b_dna.get("duration_seconds"), 180.0)
    target_duration = round(min(duration_a, duration_b), 3)

    sections_a = song_a_dna.get("sections") or []
    sections_b = song_b_dna.get("sections") or []

    candidate_count = min(max(len(sections_a), 1), max(len(sections_b), 1))
    section_count = max(4, min(8, candidate_count))

    span = target_duration / section_count
    timeline_sections: list[dict[str, Any]] = []

    dominant_counts = {"A": 0, "B": 0}

    for idx in range(section_count):
        dominant_parent = "A" if idx % 2 == 0 else "B"
        support_parent = "B" if dominant_parent == "A" else "A"
        dominant_counts[dominant_parent] += 1

        fallback_start = idx * span
        fallback_end = (idx + 1) * span if idx < section_count - 1 else target_duration

        source_section = (sections_a if dominant_parent == "A" else sections_b)
        source = source_section[idx] if idx < len(source_section) else {}
        src_start, src_end = _section_span(source, fallback_start, fallback_end)

        start = round(fallback_start, 3)
        end = round(fallback_end, 3)

        section = {
            "section_id": f"section_{idx + 1}",
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "primary_parent": dominant_parent,
            "support_parent": support_parent,
            "primary_focus": _focus(song_a_dna, song_b_dna, dominant_parent),
            "purpose": _purpose(idx, section_count, dominant_parent),
            "reason_for_existence": (
                f"Carry the arrangement narrative with parent {dominant_parent} leading "
                f"while parent {support_parent} adds selective contrast."
            ),
            "dominant_parent_balance": "A:70/B:30" if dominant_parent == "A" else "A:30/B:70",
            "source_reference": {
                "parent": dominant_parent,
                "source_start": src_start,
                "source_end": src_end,
            },
            "layering_rule": "primary-plus-support-only",
            "overlay_guard": "No full-song overlay; only section-local selective layering.",
            "entry_transition_intent": _transition_intent(max(0, idx - 1), section_count),
            "exit_transition_intent": _transition_intent(idx, section_count),
        }
        timeline_sections.append(section)

    transitions = []
    for idx in range(len(timeline_sections) - 1):
        current = timeline_sections[idx]
        nxt = timeline_sections[idx + 1]
        transitions.append(
            {
                "from_section": current["section_id"],
                "to_section": nxt["section_id"],
                "boundary_time": current["end"],
                "intent": current["exit_transition_intent"],
                "method": "phrase-aligned EQ/fader handoff",
            }
        )

    total = dominant_counts["A"] + dominant_counts["B"]
    summary_balance = {
        "A": round(dominant_counts["A"] / total, 3),
        "B": round(dominant_counts["B"] / total, 3),
    }

    return {
        "project_id": project_id,
        "strategy": "deterministic section handoff",
        "hard_rules": {
            "no_full_song_overlay": True,
            "one_primary_focus_per_section": True,
            "explicit_purpose_per_section": True,
            "boundary_transition_intent_required": True,
        },
        "target": {
            "duration_seconds": target_duration,
            "bpm": round((_safe_float(song_a_dna.get("tempo_bpm"), 120.0) + _safe_float(song_b_dna.get("tempo_bpm"), 120.0)) / 2, 3),
            "key": f"{song_a_dna.get('key_tonic', 'C')}/{song_b_dna.get('key_tonic', 'C')}",
        },
        "dominant_parent_balance": summary_balance,
        "sections": timeline_sections,
        "transitions": transitions,
    }
