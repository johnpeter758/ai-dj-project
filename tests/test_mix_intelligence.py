from mix_intelligence import score_mix, summarize_mix_score


def test_score_mix_outputs_expected_schema_and_is_deterministic() -> None:
    report = {
        "tempo_stability": 0.84,
        "harmonic_consistency": 0.8,
        "groove_lock": 0.78,
        "swing_balance": 0.72,
        "vocal_presence": 0.67,
        "masking_control": 0.63,
        "energy_shape": 0.81,
        "energy_variation": 0.7,
        "transition_smoothness": 0.75,
        "transition_tension_release": 0.71,
        "hook_impact": 0.77,
        "climax_delivery": 0.73,
        "novelty": 0.66,
        "stylistic_identity": 0.74,
        "fatigue_risk": 0.22,
        "relisten_intent": 0.79,
    }

    first = score_mix(report)
    second = score_mix(report)

    assert first == second
    assert set(first) == {
        "category_scores",
        "weights",
        "penalties",
        "total_penalty",
        "overall_score",
        "grade",
    }
    assert len(first["category_scores"]) == 9
    assert 0.0 <= first["overall_score"] <= 100.0
    assert first["grade"] in {"A", "B", "C", "D", "F"}


def test_summarize_mix_score_mentions_strength_and_weakness() -> None:
    score = score_mix({"tempo_stability": 0.9, "groove_lock": 0.8, "fatigue_risk": 0.1})
    summary = summarize_mix_score(score)

    assert "Mix score" in summary
    assert "Strongest:" in summary
    assert "Needs work:" in summary
