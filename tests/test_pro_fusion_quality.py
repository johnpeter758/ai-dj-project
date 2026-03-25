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
