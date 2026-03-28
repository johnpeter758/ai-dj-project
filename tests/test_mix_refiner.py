from mix_refiner import apply_refinement_plan, propose_refinement_passes


def test_propose_refinement_passes_picks_lowest_categories() -> None:
    mix_report = {
        "category_scores": {
            "coherence": 82.0,
            "groove": 61.0,
            "vocal_clarity": 58.0,
            "energy_arc": 75.0,
            "transition_quality": 64.0,
            "payoff_strength": 79.0,
            "originality": 83.0,
            "listenability": 80.0,
            "replay_value": 77.0,
        }
    }

    passes = propose_refinement_passes(mix_report, max_iterations=3)

    assert len(passes) == 3
    assert [p["focus"] for p in passes] == ["vocal_clarity", "groove", "transition_quality"]
    assert all("actions" in p and isinstance(p["actions"], list) for p in passes)


def test_apply_refinement_plan_is_non_mutating_and_deterministic() -> None:
    arrangement = {"sections": [{"name": "intro"}], "mix_notes": []}
    passes = [
        {"pass_id": 1, "focus": "energy_arc", "actions": ["increase lift"], "expected_gain": 3.5},
        {"pass_id": 2, "focus": "vocal_clarity", "actions": ["cut masking"], "expected_gain": 4.0},
    ]

    out1 = apply_refinement_plan(arrangement, passes)
    out2 = apply_refinement_plan(arrangement, passes)

    assert out1 == out2
    assert out1 is not arrangement
    assert arrangement["mix_notes"] == []
    assert out1["refinement_pass_count"] == 2
    assert len(out1["refinement_history"]) == 2
    assert "[energy_arc] increase lift" in out1["mix_notes"]
