from __future__ import annotations

from copy import deepcopy


def propose_refinement_passes(mix_report: dict, max_iterations: int = 3) -> list[dict]:
    """Propose deterministic refinement passes from the weakest scored dimensions."""
    if max_iterations <= 0:
        return []

    scores = mix_report.get("category_scores", {}) if isinstance(mix_report, dict) else {}
    if not isinstance(scores, dict):
        scores = {}

    defaults = {
        "coherence": 70.0,
        "groove": 70.0,
        "vocal_clarity": 70.0,
        "energy_arc": 70.0,
        "transition_quality": 70.0,
        "payoff_strength": 70.0,
        "originality": 70.0,
        "listenability": 70.0,
        "replay_value": 70.0,
    }
    merged = {**defaults, **{k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}}

    rules = {
        "coherence": ["tighten harmonic pivots", "reduce abrupt arrangement switches"],
        "groove": ["nudge drum microtiming to pocket", "align bass transients with kick"],
        "vocal_clarity": ["cut 250-450Hz masking beds", "raise vocal presence in chorus"],
        "energy_arc": ["increase mid-song lift", "create clearer final payoff ramp"],
        "transition_quality": ["add riser/fill before section boundaries", "smooth handoff EQ between sections"],
        "payoff_strength": ["thicken chorus stack", "delay full-drop by 4 bars for contrast"],
        "originality": ["introduce one signature motif variation", "replace generic fill with custom ear-candy"],
        "listenability": ["trim harsh layers in dense sections", "reduce repetitive loop exposure"],
        "replay_value": ["add subtle second-pass detail", "strengthen memorable return motif"],
    }

    threshold = 78.0
    ranked = sorted(merged.items(), key=lambda kv: (kv[1], kv[0]))

    passes: list[dict] = []
    for dim, value in ranked:
        if value >= threshold:
            continue
        gap = round(threshold - value, 3)
        passes.append(
            {
                "pass_id": len(passes) + 1,
                "focus": dim,
                "reason": f"{dim} below target by {gap:.1f} points",
                "actions": list(rules.get(dim, ["perform targeted polish pass"])),
                "expected_gain": min(8.0, round(gap * 0.35 + 2.0, 3)),
            }
        )
        if len(passes) >= max_iterations:
            break

    return passes


def apply_refinement_plan(arrangement_plan: dict, refinement_passes: list[dict]) -> dict:
    """Apply refinement notes into arrangement plan without mutating input."""
    updated = deepcopy(arrangement_plan) if isinstance(arrangement_plan, dict) else {}
    passes = refinement_passes if isinstance(refinement_passes, list) else []

    updated.setdefault("refinement_history", [])
    updated.setdefault("mix_notes", [])

    for item in passes:
        focus = item.get("focus", "general")
        actions = item.get("actions", [])
        if not isinstance(actions, list):
            actions = [str(actions)]

        updated["refinement_history"].append(
            {
                "pass_id": item.get("pass_id", len(updated["refinement_history"]) + 1),
                "focus": focus,
                "expected_gain": item.get("expected_gain", 0.0),
            }
        )

        for action in actions:
            updated["mix_notes"].append(f"[{focus}] {action}")

    updated["refinement_pass_count"] = len(updated["refinement_history"])
    return updated
