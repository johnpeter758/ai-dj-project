from __future__ import annotations

import json
from pathlib import Path

from scripts.build_listen_gate_spec import build_spec
from scripts.listen_gate_benchmark import run_harness
from tests.synthetic_listener_fixtures import (
    near_threshold_false_positive_review_fixture,
    not_one_song_composite_fixture,
)
from tests.test_synthetic_good_output_fixtures import _synthetic_report


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_false_negative_survivor_regression_pack_preserves_survivor_review_reject_lanes(tmp_path: Path):
    survivor = _write_json(
        tmp_path / "survivor_good.json",
        _synthetic_report(
            86.0,
            source_path="survivor_good.wav",
            structure=84.0,
            groove=81.0,
            energy_arc=85.0,
            transition=80.0,
            coherence=82.0,
            mix_sanity=79.0,
            song_likeness=82.0,
        ),
    )
    review = _write_json(
        tmp_path / "near_threshold_review.json",
        near_threshold_false_positive_review_fixture(),
    )
    reject = _write_json(
        tmp_path / "not_one_song_composite.json",
        not_one_song_composite_fixture(),
    )

    spec = build_spec(
        good_cases_raw=[f"survivor_good={survivor}"],
        review_cases_raw=[f"near_threshold_review={review}"],
        cases_raw=[f"not_one_song_composite={reject}"],
        expected_order_raw="survivor_good,near_threshold_review,not_one_song_composite",
        gating_expectations=["not_one_song_composite:gating_status=pass"],
        overall_at_least=["survivor_good:overall_score_at_least=80"],
        overall_at_most=["not_one_song_composite:overall_score_at_most=70"],
        component_at_least=["survivor_good:song_likeness=80", "survivor_good:energy_arc=80"],
        component_at_most=["not_one_song_composite:transition=60"],
        metric_at_least=[
            "survivor_good:song_likeness.details.aggregate_metrics.backbone_continuity=0.70",
            "survivor_good:song_likeness.details.aggregate_metrics.recognizable_section_ratio=0.70",
        ],
        metric_at_most=[
            "near_threshold_review:song_likeness.details.aggregate_metrics.background_only_identity_gap=0.45",
            "near_threshold_review:song_likeness.details.aggregate_metrics.owner_switch_ratio=0.78",
            "near_threshold_review:song_likeness.details.aggregate_metrics.composite_song_risk=0.50",
            "not_one_song_composite:song_likeness.details.aggregate_metrics.backbone_continuity=0.35",
            "not_one_song_composite:song_likeness.details.aggregate_metrics.composite_song_risk=0.70",
        ],
        better_than_raw=[
            "survivor_good>near_threshold_review:overall=10:component=song_likeness=20,transition=20",
            "near_threshold_review>not_one_song_composite:overall=4:component=song_likeness=0,transition=-5",
        ],
    )

    spec_path = tmp_path / "survivor_regression_pack.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    report = run_harness(str(spec_path))

    assert report["passed"] is True
    assert report["expected_order"] == [
        "survivor_good",
        "near_threshold_review",
        "not_one_song_composite",
    ]
    ranked = [row["label"] for row in sorted(report["cases"], key=lambda row: int(row["benchmark_rank"]))]
    assert ranked == ["survivor_good", "near_threshold_review", "not_one_song_composite"]

    cases = {row["label"]: row for row in report["cases"]}
    assert cases["survivor_good"]["report"]["gating"]["status"] == "pass"
    assert cases["near_threshold_review"]["report"]["gating"]["status"] == "review"
    assert cases["not_one_song_composite"]["report"]["gating"]["status"] == "pass"
    assert report["failures"] == {
        "case_failures": {},
        "pairwise_failures": [],
        "order_failures": [],
    }
