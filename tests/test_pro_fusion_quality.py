import ai_dj


def test_candidate_meets_quality_floor_requires_pass_and_thresholds():
    report = {
        "gating": {"status": "pass"},
        "song_likeness": {"score": 56.0},
        "groove": {"score": 61.0},
        "structure": {"score": 60.0},
    }
    assert ai_dj._candidate_meets_quality_floor(
        report,
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    ) is True


def test_candidate_meets_quality_floor_rejects_review_even_if_scores_high():
    report = {
        "gating": {"status": "review"},
        "song_likeness": {"score": 90.0},
        "groove": {"score": 90.0},
        "structure": {"score": 90.0},
    }
    assert ai_dj._candidate_meets_quality_floor(
        report,
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    ) is False


def test_candidate_meets_quality_floor_rejects_low_structure():
    report = {
        "gating": {"status": "pass"},
        "song_likeness": {"score": 80.0},
        "groove": {"score": 80.0},
        "structure": {"score": 52.0},
    }
    assert ai_dj._candidate_meets_quality_floor(
        report,
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    ) is False


def test_candidate_meets_quality_floor_rejects_low_boundary_recovery_when_present():
    report = {
        "gating": {"status": "pass"},
        "song_likeness": {
            "score": 82.0,
            "details": {"aggregate_metrics": {"boundary_recovery": 0.31, "role_plausibility": 0.62}},
        },
        "groove": {"score": 80.0},
        "structure": {"score": 78.0},
    }
    assert ai_dj._candidate_meets_quality_floor(
        report,
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    ) is False


def test_candidate_meets_quality_floor_does_not_require_readability_metrics_when_missing():
    report = {
        "gating": {"status": "pass"},
        "song_likeness": {"score": 75.0},
        "groove": {"score": 65.0},
        "structure": {"score": 61.0},
    }
    assert ai_dj._candidate_meets_quality_floor(
        report,
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    ) is True


def test_extract_transition_seam_snapshot_is_resilient():
    report = {
        "transition": {
            "score": 72.5,
            "details": {
                "aggregate_metrics": {
                    "mean_seam_risk": 0.21,
                    "avg_overlap_beats": 1.4,
                }
            },
        }
    }
    snap = ai_dj._extract_transition_seam_snapshot(report)
    assert snap["transition_score"] == 72.5
    assert snap["mean_seam_risk"] == 0.21
    assert snap["avg_overlap_beats"] == 1.4


def test_select_pro_fusion_winner_requires_pass_plus_quality_floor():
    candidates = [
        {
            "selection_score": 90.0,
            "listen_report": {
                "gating": {"status": "pass"},
                "song_likeness": {"score": 54.0},
                "groove": {"score": 70.0},
                "structure": {"score": 65.0},
            },
        },
        {
            "selection_score": 88.0,
            "listen_report": {
                "gating": {"status": "review"},
                "song_likeness": {"score": 90.0},
                "groove": {"score": 90.0},
                "structure": {"score": 92.0},
            },
        },
    ]

    winner, policy, counts = ai_dj._select_pro_fusion_winner(
        candidates,
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    )

    assert winner is None
    assert policy == "hard-fail:pass-below-floor"
    assert counts["candidate_count"] == 2
    assert counts["pass_count"] == 1
    assert counts["floor_pass_count"] == 0


def test_select_pro_fusion_winner_picks_top_floor_pass_candidate():
    low_score_floor_pass = {
        "selection_score": 78.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "song_likeness": {"score": 60.0},
            "groove": {"score": 62.0},
            "structure": {"score": 60.0},
        },
    }
    high_score_floor_pass = {
        "selection_score": 85.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "song_likeness": {"score": 66.0},
            "groove": {"score": 71.0},
            "structure": {"score": 64.0},
        },
    }

    winner, policy, counts = ai_dj._select_pro_fusion_winner(
        [low_score_floor_pass, high_score_floor_pass],
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    )

    assert winner is high_score_floor_pass
    assert policy == "pass+floor"
    assert counts["floor_pass_count"] == 2


def test_select_pro_fusion_winner_breaks_ties_toward_stronger_structure():
    lower_structure = {
        "selection_score": 90.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 80.0,
            "song_likeness": {"score": 70.0},
            "groove": {"score": 70.0},
            "structure": {"score": 60.0},
        },
    }
    higher_structure = {
        "selection_score": 90.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 79.0,
            "song_likeness": {"score": 69.0},
            "groove": {"score": 69.0},
            "structure": {"score": 74.0},
        },
    }

    winner, policy, counts = ai_dj._select_pro_fusion_winner(
        [lower_structure, higher_structure],
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    )

    assert winner is higher_structure
    assert policy == "pass+floor"
    assert counts["floor_pass_count"] == 2


def test_select_pro_fusion_winner_treats_nan_selection_score_as_zero():
    nan_score = {
        "selection_score": float("nan"),
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 70.0,
            "song_likeness": {"score": 60.0},
            "groove": {"score": 60.0},
            "structure": {"score": 60.0},
            "transition": {"score": 60.0},
        },
    }
    finite_score = {
        "selection_score": 1.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 70.0,
            "song_likeness": {"score": 60.0},
            "groove": {"score": 60.0},
            "structure": {"score": 60.0},
            "transition": {"score": 60.0},
        },
    }

    winner, policy, _counts = ai_dj._select_pro_fusion_winner(
        [nan_score, finite_score],
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    )

    assert winner is finite_score
    assert policy == "pass+floor"


def test_select_pro_fusion_winner_biases_transition_clarity_under_pass_gate():
    lower_transition = {
        "selection_score": 92.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 85.0,
            "song_likeness": {"score": 74.0},
            "groove": {"score": 72.0},
            "structure": {"score": 70.0},
            "transition": {"score": 62.0},
        },
    }
    higher_transition = {
        "selection_score": 91.5,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 84.0,
            "song_likeness": {"score": 78.0},
            "groove": {"score": 71.0},
            "structure": {"score": 70.0},
            "transition": {"score": 74.0},
        },
    }

    winner, policy, _counts = ai_dj._select_pro_fusion_winner(
        [lower_transition, higher_transition],
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    )

    assert winner is higher_transition
    assert policy == "pass+floor"


def test_select_pro_fusion_winner_penalizes_high_seam_risk_in_final_sort():
    high_risk = {
        "selection_score": 96.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 85.0,
            "song_likeness": {"score": 75.0},
            "groove": {"score": 72.0},
            "structure": {"score": 72.0},
            "transition": {
                "score": 72.0,
                "details": {"aggregate_metrics": {"mean_seam_risk": 0.9, "max_seam_risk": 0.95, "mean_energy_jump": 0.8}},
            },
        },
    }
    low_risk = {
        "selection_score": 93.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 85.0,
            "song_likeness": {"score": 75.0},
            "groove": {"score": 72.0},
            "structure": {"score": 72.0},
            "transition": {
                "score": 72.0,
                "details": {"aggregate_metrics": {"mean_seam_risk": 0.15, "max_seam_risk": 0.2, "mean_energy_jump": 0.15}},
            },
        },
    }

    winner, policy, _counts = ai_dj._select_pro_fusion_winner(
        [high_risk, low_risk],
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    )

    assert winner is low_risk
    assert policy == "pass+floor"


def test_select_pro_fusion_winner_promotes_real_transition_lift_within_structure_guardrails():
    baseline_winner = {
        "selection_score": 94.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 85.0,
            "song_likeness": {"score": 76.0},
            "groove": {"score": 72.0},
            "structure": {"score": 73.0},
            "transition": {"score": 70.0},
        },
    }
    transition_lift = {
        "selection_score": 92.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 84.0,
            "song_likeness": {"score": 75.4},
            "groove": {"score": 71.0},
            "structure": {"score": 71.6},
            "transition": {"score": 72.1},
        },
    }

    winner, policy, _counts = ai_dj._select_pro_fusion_winner(
        [baseline_winner, transition_lift],
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    )

    assert winner is transition_lift
    assert policy == "pass+floor"


def test_select_pro_fusion_winner_does_not_promote_transition_lift_when_structure_drop_is_too_large():
    baseline_winner = {
        "selection_score": 94.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 85.0,
            "song_likeness": {"score": 76.0},
            "groove": {"score": 72.0},
            "structure": {"score": 73.0},
            "transition": {"score": 70.0},
        },
    }
    too_costly_transition_lift = {
        "selection_score": 93.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "overall_score": 84.0,
            "song_likeness": {"score": 75.9},
            "groove": {"score": 71.0},
            "structure": {"score": 70.7},
            "transition": {"score": 72.3},
        },
    }

    winner, policy, _counts = ai_dj._select_pro_fusion_winner(
        [baseline_winner, too_costly_transition_lift],
        min_song_likeness=55.0,
        min_groove=60.0,
        min_structure=58.0,
        min_boundary_recovery=0.45,
        min_role_plausibility=0.48,
    )

    assert winner is baseline_winner
    assert policy == "pass+floor"


def test_pro_fusion_selection_score_penalizes_medley_like_outputs_with_low_integration():
    base = {
        "gating": {"status": "pass"},
        "overall_score": 78.0,
        "song_likeness": {"score": 72.0, "details": {"aggregate_metrics": {}}},
        "groove": {"score": 74.0},
        "transition": {"score": 73.0, "details": {"aggregate_metrics": {"mean_seam_risk": 0.2, "max_seam_risk": 0.25, "mean_energy_jump": 0.2}}},
        "mix_sanity": {"score": 70.0},
    }
    fused = {
        **base,
        "song_likeness": {
            "score": 72.0,
            "details": {
                "aggregate_metrics": {
                    "full_mix_medley_risk": 0.08,
                    "integrated_two_parent_section_ratio": 0.42,
                    "max_parent_share": 0.62,
                    "owner_switch_ratio": 0.18,
                }
            },
        },
    }
    medley_like = {
        **base,
        "song_likeness": {
            "score": 72.0,
            "details": {
                "aggregate_metrics": {
                    "full_mix_medley_risk": 0.86,
                    "integrated_two_parent_section_ratio": 0.02,
                    "max_parent_share": 0.98,
                    "owner_switch_ratio": 0.82,
                }
            },
        },
    }

    fused_score = ai_dj._pro_fusion_selection_score(fused, parent_balance=0.5)
    medley_score = ai_dj._pro_fusion_selection_score(medley_like, parent_balance=0.5)

    assert fused_score > medley_score



def test_pro_fusion_selection_score_penalizes_back_and_forth_owner_switching():
    base = {
        "gating": {"status": "pass"},
        "overall_score": 78.0,
        "song_likeness": {
            "score": 72.0,
            "details": {
                "aggregate_metrics": {
                    "full_mix_medley_risk": 0.45,
                    "integrated_two_parent_section_ratio": 0.20,
                    "max_parent_share": 0.78,
                }
            },
        },
        "groove": {"score": 74.0},
        "transition": {"score": 73.0, "details": {"aggregate_metrics": {"mean_seam_risk": 0.2, "max_seam_risk": 0.25, "mean_energy_jump": 0.2}}},
        "mix_sanity": {"score": 70.0},
    }
    stable = {
        **base,
        "song_likeness": {
            "score": 72.0,
            "details": {
                "aggregate_metrics": {
                    **base["song_likeness"]["details"]["aggregate_metrics"],
                    "owner_switch_ratio": 0.05,
                }
            },
        },
    }
    switchy = {
        **base,
        "song_likeness": {
            "score": 72.0,
            "details": {
                "aggregate_metrics": {
                    **base["song_likeness"]["details"]["aggregate_metrics"],
                    "owner_switch_ratio": 0.65,
                }
            },
        },
    }

    stable_score = ai_dj._pro_fusion_selection_score(stable, parent_balance=0.5)
    switchy_score = ai_dj._pro_fusion_selection_score(switchy, parent_balance=0.5)

    assert stable_score > switchy_score
