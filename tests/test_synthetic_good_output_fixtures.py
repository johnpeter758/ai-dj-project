from __future__ import annotations

import json
from pathlib import Path

import ai_dj
from scripts.build_listen_gate_spec import build_spec
from scripts.listen_gate_benchmark import run_harness


def _synthetic_report(
    overall: float,
    *,
    source_path: str,
    verdict: str = "promising",
    gate_status: str = "pass",
    structure: float = 80.0,
    groove: float = 80.0,
    energy_arc: float = 80.0,
    transition: float = 80.0,
    coherence: float = 80.0,
    mix_sanity: float = 80.0,
    song_likeness: float = 80.0,
    aggregate_metrics: dict | None = None,
) -> dict:
    song_metrics = {
        "backbone_continuity": 0.76,
        "readable_section_ratio": 0.77,
        "recognizable_section_ratio": 0.74,
        "boundary_recovery": 0.72,
        "role_plausibility": 0.71,
        "planner_audio_climax_conviction": 0.73,
        "climax_conviction": 0.76,
        "background_only_identity_gap": 0.10,
        "owner_switch_ratio": 0.26,
    }
    if aggregate_metrics:
        song_metrics.update(aggregate_metrics)
    return {
        "source_path": source_path,
        "duration_seconds": 78.0,
        "overall_score": overall,
        "structure": {"score": structure, "summary": "readable", "evidence": [], "fixes": [], "details": {}},
        "groove": {"score": groove, "summary": "stable pocket", "evidence": [], "fixes": [], "details": {}},
        "energy_arc": {"score": energy_arc, "summary": "lands late", "evidence": [], "fixes": [], "details": {}},
        "transition": {"score": transition, "summary": "musical handoffs", "evidence": [], "fixes": [], "details": {}},
        "coherence": {"score": coherence, "summary": "coherent", "evidence": [], "fixes": [], "details": {}},
        "mix_sanity": {"score": mix_sanity, "summary": "clear", "evidence": [], "fixes": [], "details": {}},
        "song_likeness": {
            "score": song_likeness,
            "summary": "reads like one song",
            "evidence": [],
            "fixes": [],
            "details": {"aggregate_metrics": song_metrics},
        },
        "verdict": verdict,
        "top_reasons": ["Synthetic fixture for deterministic quality-gate coverage."],
        "top_fixes": [],
        "gating": {"status": gate_status, "raw_overall_score": overall},
        "analysis_version": "0.5.0",
    }


def _write_report(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_listener_agent_keeps_only_synthetic_good_outputs_for_human_review(tmp_path: Path):
    elite = _write_report(
        tmp_path / "elite_good.json",
        _synthetic_report(
            90.0,
            source_path="elite_good.wav",
            structure=88.0,
            groove=86.0,
            energy_arc=91.0,
            transition=87.0,
            coherence=89.0,
            mix_sanity=85.0,
            song_likeness=88.0,
        ),
    )
    solid = _write_report(
        tmp_path / "solid_good.json",
        _synthetic_report(
            84.0,
            source_path="solid_good.wav",
            structure=82.0,
            groove=80.0,
            energy_arc=83.0,
            transition=79.0,
            coherence=82.0,
            mix_sanity=80.0,
            song_likeness=81.0,
        ),
    )
    fake = _write_report(
        tmp_path / "fake_good.json",
        _synthetic_report(
            78.0,
            source_path="fake_good.wav",
            structure=76.0,
            groove=77.0,
            energy_arc=75.0,
            transition=73.0,
            coherence=76.0,
            mix_sanity=74.0,
            song_likeness=63.0,
            aggregate_metrics={
                "backbone_continuity": 0.38,
                "readable_section_ratio": 0.32,
                "recognizable_section_ratio": 0.35,
                "boundary_recovery": 0.33,
                "role_plausibility": 0.34,
            },
        ),
    )

    out_path = tmp_path / "listener_agent.json"
    rc = ai_dj.listener_agent([str(elite), str(solid), str(fake)], str(out_path), shortlist=2)
    assert rc == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["counts"] == {"total": 3, "survivors": 2, "borderline": 0, "rejected": 1}
    assert [row["label"] for row in payload["recommended_for_human_review"]] == [
        "elite_good.json",
        "solid_good.json",
    ]
    assert all(row["decision"] == "survivor" for row in payload["recommended_for_human_review"])
    assert payload["recommended_for_human_review"][0]["listener_rank"] > payload["recommended_for_human_review"][1]["listener_rank"]
    assert payload["rejected"][0]["label"] == "fake_good.json"
    assert "whole-song backbone continuity is too weak" in payload["rejected"][0]["hard_fail_reasons"]


def test_synthetic_good_fixture_spec_passes_fixed_benchmark_harness(tmp_path: Path):
    elite = _write_report(
        tmp_path / "elite_good.json",
        _synthetic_report(
            91.0,
            source_path="elite_good.wav",
            structure=89.0,
            groove=87.0,
            energy_arc=92.0,
            transition=88.0,
            coherence=90.0,
            mix_sanity=86.0,
            song_likeness=89.0,
        ),
    )
    solid = _write_report(
        tmp_path / "solid_good.json",
        _synthetic_report(
            83.0,
            source_path="solid_good.wav",
            structure=82.0,
            groove=79.0,
            energy_arc=81.0,
            transition=78.0,
            coherence=81.0,
            mix_sanity=79.0,
            song_likeness=80.0,
        ),
    )
    bad = _write_report(
        tmp_path / "bad_case.json",
        _synthetic_report(
            58.0,
            source_path="bad_case.wav",
            verdict="weak",
            gate_status="reject",
            structure=63.0,
            groove=55.0,
            energy_arc=50.0,
            transition=48.0,
            coherence=57.0,
            mix_sanity=56.0,
            song_likeness=46.0,
            aggregate_metrics={
                "backbone_continuity": 0.49,
                "readable_section_ratio": 0.43,
                "recognizable_section_ratio": 0.46,
                "boundary_recovery": 0.44,
                "role_plausibility": 0.45,
                "planner_audio_climax_conviction": 0.42,
                "climax_conviction": 0.40,
            },
        ),
    )

    spec = build_spec(
        good_cases_raw=[f"elite_good={elite}"],
        cases_raw=[f"solid_good={solid}"],
        bad_cases_raw=[f"bad_case={bad}"],
        expected_order_raw="elite_good,solid_good,bad_case",
        overall_at_least=["elite_good:overall_score_at_least=88", "solid_good:overall_score_at_least=80"],
        overall_at_most=["bad_case:overall_score_at_most=65"],
        component_at_least=[
            "elite_good:energy_arc=90",
            "elite_good:song_likeness=87",
            "solid_good:transition=76",
        ],
        component_at_most=["bad_case:transition=50", "bad_case:song_likeness=50"],
        metric_at_least=[
            "elite_good:song_likeness.details.aggregate_metrics.readable_section_ratio=0.75",
            "solid_good:song_likeness.details.aggregate_metrics.readable_section_ratio=0.75",
            "elite_good:song_likeness.details.aggregate_metrics.planner_audio_climax_conviction=0.72",
            "solid_good:song_likeness.details.aggregate_metrics.planner_audio_climax_conviction=0.72",
            "elite_good:song_likeness.details.aggregate_metrics.climax_conviction=0.75",
            "solid_good:song_likeness.details.aggregate_metrics.climax_conviction=0.75",
        ],
        metric_at_most=[
            "bad_case:song_likeness.details.aggregate_metrics.readable_section_ratio=0.50",
            "bad_case:song_likeness.details.aggregate_metrics.planner_audio_climax_conviction=0.45",
            "bad_case:song_likeness.details.aggregate_metrics.climax_conviction=0.45",
        ],
        better_than_raw=[
            "elite_good>solid_good:overall=5:component=energy_arc=8,song_likeness=8",
            "solid_good>bad_case:overall=20:component=transition=20,song_likeness=25",
        ],
    )
    spec_path = tmp_path / "synthetic_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    report = run_harness(str(spec_path))
    assert report["passed"] is True
    assert report["expected_order"] == ["elite_good", "solid_good", "bad_case"]
    assert [row["label"] for row in sorted(report["cases"], key=lambda row: int(row["benchmark_rank"]))] == [
        "elite_good",
        "solid_good",
        "bad_case",
    ]
    assert report["failures"] == {
        "case_failures": {},
        "pairwise_failures": [],
        "order_failures": [],
    }
    assert any(line.startswith("#1 elite_good") for line in report["summary"])
