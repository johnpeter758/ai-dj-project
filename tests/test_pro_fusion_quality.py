import ai_dj


def test_candidate_meets_quality_floor_requires_pass_and_thresholds():
    report = {
        "gating": {"status": "pass"},
        "song_likeness": {"score": 56.0},
        "groove": {"score": 61.0},
    }
    assert ai_dj._candidate_meets_quality_floor(report, min_song_likeness=55.0, min_groove=60.0) is True


def test_candidate_meets_quality_floor_rejects_review_even_if_scores_high():
    report = {
        "gating": {"status": "review"},
        "song_likeness": {"score": 90.0},
        "groove": {"score": 90.0},
    }
    assert ai_dj._candidate_meets_quality_floor(report, min_song_likeness=55.0, min_groove=60.0) is False


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
            },
        },
        {
            "selection_score": 88.0,
            "listen_report": {
                "gating": {"status": "review"},
                "song_likeness": {"score": 90.0},
                "groove": {"score": 90.0},
            },
        },
    ]

    winner, policy, counts = ai_dj._select_pro_fusion_winner(
        candidates,
        min_song_likeness=55.0,
        min_groove=60.0,
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
        },
    }
    high_score_floor_pass = {
        "selection_score": 85.0,
        "listen_report": {
            "gating": {"status": "pass"},
            "song_likeness": {"score": 66.0},
            "groove": {"score": 71.0},
        },
    }

    winner, policy, counts = ai_dj._select_pro_fusion_winner(
        [low_score_floor_pass, high_score_floor_pass],
        min_song_likeness=55.0,
        min_groove=60.0,
    )

    assert winner is high_score_floor_pass
    assert policy == "pass+floor"
    assert counts["floor_pass_count"] == 2
