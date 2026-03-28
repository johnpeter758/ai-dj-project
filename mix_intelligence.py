from __future__ import annotations

from typing import Any


_WEIGHTS = {
    "coherence": 0.14,
    "groove": 0.11,
    "vocal_clarity": 0.10,
    "energy_arc": 0.13,
    "transition_quality": 0.12,
    "payoff_strength": 0.12,
    "originality": 0.08,
    "listenability": 0.10,
    "replay_value": 0.10,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _metric(report: dict[str, Any], key: str, default: float = 0.5) -> float:
    if key in report:
        return _clamp01(report[key])
    metrics = report.get("metrics", {})
    if isinstance(metrics, dict) and key in metrics:
        return _clamp01(metrics[key])
    return _clamp01(default)


def score_mix(report: dict) -> dict:
    """Score mix quality using deterministic human-taste heuristics."""
    tempo_stability = _metric(report, "tempo_stability")
    harmonic_consistency = _metric(report, "harmonic_consistency")
    groove_lock = _metric(report, "groove_lock")
    swing_balance = _metric(report, "swing_balance")
    vocal_presence = _metric(report, "vocal_presence")
    masking_control = _metric(report, "masking_control")
    energy_shape = _metric(report, "energy_shape")
    energy_variation = _metric(report, "energy_variation")
    transition_smoothness = _metric(report, "transition_smoothness")
    transition_tension_release = _metric(report, "transition_tension_release")
    hook_impact = _metric(report, "hook_impact")
    climax_delivery = _metric(report, "climax_delivery")
    novelty = _metric(report, "novelty")
    stylistic_identity = _metric(report, "stylistic_identity")
    fatigue_risk = _metric(report, "fatigue_risk", default=0.2)
    relisten_intent = _metric(report, "relisten_intent")

    category = {
        "coherence": (tempo_stability * 0.45 + harmonic_consistency * 0.55),
        "groove": (groove_lock * 0.7 + swing_balance * 0.3),
        "vocal_clarity": (vocal_presence * 0.5 + masking_control * 0.5),
        "energy_arc": (energy_shape * 0.65 + energy_variation * 0.35),
        "transition_quality": (transition_smoothness * 0.6 + transition_tension_release * 0.4),
        "payoff_strength": (hook_impact * 0.55 + climax_delivery * 0.45),
        "originality": (novelty * 0.6 + stylistic_identity * 0.4),
        "listenability": (1.0 - fatigue_risk) * 0.45 + coherence_guard(tempo_stability, groove_lock) * 0.55,
        "replay_value": (relisten_intent * 0.65 + hook_impact * 0.35),
    }

    category_scores = {k: round(_clamp01(v) * 100.0, 3) for k, v in category.items()}

    weighted_total = 0.0
    for key, weight in _WEIGHTS.items():
        weighted_total += category_scores[key] * weight

    penalties = {
        "flat_energy_penalty": round(max(0.0, 62.0 - category_scores["energy_arc"]) * 0.2, 3),
        "muddy_vocal_penalty": round(max(0.0, 60.0 - category_scores["vocal_clarity"]) * 0.15, 3),
        "weak_transition_penalty": round(max(0.0, 60.0 - category_scores["transition_quality"]) * 0.1, 3),
    }

    total_penalty = round(sum(penalties.values()), 3)
    overall = round(max(0.0, min(100.0, weighted_total - total_penalty)), 3)

    return {
        "category_scores": category_scores,
        "weights": dict(_WEIGHTS),
        "penalties": penalties,
        "total_penalty": total_penalty,
        "overall_score": overall,
        "grade": _grade(overall),
    }


def coherence_guard(tempo_stability: float, groove_lock: float) -> float:
    return _clamp01((tempo_stability * 0.5) + (groove_lock * 0.5))


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def summarize_mix_score(score: dict) -> str:
    overall = float(score.get("overall_score", 0.0))
    grade = score.get("grade", _grade(overall))
    categories = score.get("category_scores", {})

    if categories:
        strongest = max(categories, key=categories.get)
        weakest = min(categories, key=categories.get)
        return (
            f"Mix score {overall:.1f}/100 ({grade}). "
            f"Strongest: {strongest.replace('_', ' ')} ({categories[strongest]:.1f}). "
            f"Needs work: {weakest.replace('_', ' ')} ({categories[weakest]:.1f})."
        )

    return f"Mix score {overall:.1f}/100 ({grade})."
