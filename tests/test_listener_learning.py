from __future__ import annotations

import json
from pathlib import Path

import ai_dj
from src.feedback_learning import build_feedback_learning_summary
from src.listener_learning import build_listener_learning_models, score_report_with_listener_learning


def _listen_report(
    *,
    source_path: str,
    overall: float,
    structure: float,
    groove: float,
    energy_arc: float,
    transition: float,
    coherence: float,
    mix_sanity: float,
    song_likeness: float,
    gate_status: str,
    backbone_continuity: float,
    composite_song_risk: float,
    crowding_burst_risk: float,
) -> dict:
    return {
        "source_path": source_path,
        "overall_score": overall,
        "verdict": "promising" if overall >= 75 else "weak",
        "structure": {"score": structure, "summary": "ok"},
        "groove": {"score": groove, "summary": "ok"},
        "energy_arc": {"score": energy_arc, "summary": "ok"},
        "transition": {
            "score": transition,
            "summary": "ok",
            "details": {
                "aggregate_metrics": {
                    "avg_edge_cliff_risk": 0.15 if overall >= 75 else 0.62,
                    "avg_vocal_competition_risk": 0.12 if overall >= 75 else 0.44,
                    "manifest_switch_detector_risk": 0.14 if overall >= 75 else 0.71,
                }
            },
        },
        "coherence": {"score": coherence, "summary": "ok"},
        "mix_sanity": {
            "score": mix_sanity,
            "summary": "ok",
            "details": {
                "ownership_clutter_metrics": {"crowding_burst_risk": crowding_burst_risk},
                "manifest_metrics": {
                    "aggregate_metrics": {
                        "low_end_owner_stability_risk": 0.12 if overall >= 75 else 0.58,
                    }
                },
            },
        },
        "song_likeness": {
            "score": song_likeness,
            "summary": "ok",
            "details": {
                "aggregate_metrics": {
                    "backbone_continuity": backbone_continuity,
                    "recognizable_section_ratio": 0.78 if overall >= 75 else 0.32,
                    "boundary_recovery": 0.74 if overall >= 75 else 0.28,
                    "role_plausibility": 0.76 if overall >= 75 else 0.31,
                    "composite_song_risk": composite_song_risk,
                    "background_only_identity_gap": 0.14 if overall >= 75 else 0.57,
                    "owner_switch_ratio": 0.24 if overall >= 75 else 0.81,
                }
            },
        },
        "gating": {"status": gate_status},
    }


def _write_report(run_dir: Path, report: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "listen_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_listener_learning_trains_render_and_pairwise_models(tmp_path: Path) -> None:
    strong_a = tmp_path / "strong_a"
    strong_b = tmp_path / "strong_b"
    weak_a = tmp_path / "weak_a"
    weak_b = tmp_path / "weak_b"
    _write_report(strong_a, _listen_report(source_path="strong_a.wav", overall=84.0, structure=82.0, groove=80.0, energy_arc=81.0, transition=83.0, coherence=79.0, mix_sanity=78.0, song_likeness=85.0, gate_status="pass", backbone_continuity=0.82, composite_song_risk=0.12, crowding_burst_risk=0.11))
    _write_report(strong_b, _listen_report(source_path="strong_b.wav", overall=80.0, structure=79.0, groove=78.0, energy_arc=77.0, transition=80.0, coherence=76.0, mix_sanity=75.0, song_likeness=81.0, gate_status="pass", backbone_continuity=0.77, composite_song_risk=0.16, crowding_burst_risk=0.15))
    _write_report(weak_a, _listen_report(source_path="weak_a.wav", overall=48.0, structure=50.0, groove=46.0, energy_arc=45.0, transition=42.0, coherence=47.0, mix_sanity=44.0, song_likeness=38.0, gate_status="reject", backbone_continuity=0.29, composite_song_risk=0.74, crowding_burst_risk=0.66))
    _write_report(weak_b, _listen_report(source_path="weak_b.wav", overall=54.0, structure=56.0, groove=52.0, energy_arc=50.0, transition=47.0, coherence=51.0, mix_sanity=49.0, song_likeness=44.0, gate_status="review", backbone_continuity=0.38, composite_song_risk=0.58, crowding_burst_risk=0.49))

    events = [
        {"type": "render", "overall_label": "promising", "run_dir": str(strong_a)},
        {"type": "render", "overall_label": "favorite", "run_dir": str(strong_b)},
        {"type": "render", "overall_label": "reject", "run_dir": str(weak_a)},
        {"type": "render", "overall_label": "reject", "run_dir": str(weak_b)},
        {"type": "pairwise", "left_run_dir": str(strong_a), "right_run_dir": str(weak_a), "winner": "left"},
        {"type": "pairwise", "left_run_dir": str(strong_b), "right_run_dir": str(weak_b), "winner": "left"},
        {"type": "pairwise", "left_run_dir": str(weak_a), "right_run_dir": str(strong_a), "winner": "right"},
    ]

    models = build_listener_learning_models(events)
    assert models["render_acceptor"] is not None
    assert models["pairwise_preference"] is not None

    strong_score = score_report_with_listener_learning(json.loads((strong_a / "listen_report.json").read_text(encoding="utf-8")), models)
    weak_score = score_report_with_listener_learning(json.loads((weak_a / "listen_report.json").read_text(encoding="utf-8")), models)
    assert strong_score["acceptance_probability"] > weak_score["acceptance_probability"]
    assert strong_score["preference_score"] > weak_score["preference_score"]


def test_feedback_learning_summary_exposes_listener_learning(tmp_path: Path) -> None:
    feedback_root = tmp_path / "data" / "human_feedback"
    strong = tmp_path / "runs" / "strong"
    weak = tmp_path / "runs" / "weak"
    _write_report(strong, _listen_report(source_path="strong.wav", overall=83.0, structure=81.0, groove=79.0, energy_arc=80.0, transition=82.0, coherence=78.0, mix_sanity=77.0, song_likeness=84.0, gate_status="pass", backbone_continuity=0.8, composite_song_risk=0.14, crowding_burst_risk=0.1))
    _write_report(weak, _listen_report(source_path="weak.wav", overall=47.0, structure=48.0, groove=45.0, energy_arc=43.0, transition=41.0, coherence=44.0, mix_sanity=42.0, song_likeness=35.0, gate_status="reject", backbone_continuity=0.26, composite_song_risk=0.79, crowding_burst_risk=0.63))
    feedback_root.mkdir(parents=True, exist_ok=True)
    (feedback_root / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "render", "overall_label": "promising", "run_dir": str(strong), "tags": ["good backbone"]}),
                json.dumps({"type": "render", "overall_label": "reject", "run_dir": str(weak), "tags": ["not one song"]}),
                json.dumps({"type": "render", "overall_label": "promising", "run_dir": str(strong), "tags": ["favorite"]}),
                json.dumps({"type": "render", "overall_label": "reject", "run_dir": str(weak), "tags": ["bad groove"]}),
                json.dumps({"type": "pairwise", "left_run_dir": str(strong), "right_run_dir": str(weak), "winner": "left"}),
                json.dumps({"type": "pairwise", "left_run_dir": str(weak), "right_run_dir": str(strong), "winner": "right"}),
            ]
        ) + "\n",
        encoding="utf-8",
    )

    summary = build_feedback_learning_summary(feedback_root, limit=5000)
    assert summary["listener_learning"]["render_acceptor"]["available"] is True
    assert summary["listener_learning"]["pairwise_preference"]["available"] is True


def test_apply_feedback_learning_bias_uses_learned_scores(monkeypatch) -> None:
    report = _listen_report(
        source_path="candidate.wav",
        overall=79.0,
        structure=78.0,
        groove=77.0,
        energy_arc=76.0,
        transition=78.0,
        coherence=75.0,
        mix_sanity=74.0,
        song_likeness=80.0,
        gate_status="pass",
        backbone_continuity=0.76,
        composite_song_risk=0.18,
        crowding_burst_risk=0.14,
    )
    monkeypatch.setattr(
        ai_dj,
        "_feedback_learning_snapshot",
        lambda: {
            "derived_priors": {
                "medley_rejection_pressure": 0.0,
                "groove_rejection_pressure": 0.0,
                "transition_rejection_pressure": 0.0,
                "payoff_upgrade_pressure": 0.0,
                "backbone_reward_pressure": 0.0,
            },
            "listener_learning": {
                "render_acceptor": {"available": True},
                "pairwise_preference": {"available": True},
            },
        },
    )
    monkeypatch.setattr(ai_dj, "_listener_learning_models", lambda: {"dummy": True})
    monkeypatch.setattr(
        ai_dj,
        "score_report_with_listener_learning",
        lambda report, models: {
            "available": True,
            "acceptance_probability": 0.9,
            "preference_probability": 0.8,
        },
    )

    result = ai_dj._apply_feedback_learning_bias({"listener_rank": 70.0}, report)
    assert result["listener_rank"] > 70.0
    assert result["feedback_learning"]["learned_acceptance_delta"] > 0.0
    assert result["feedback_learning"]["learned_preference_delta"] > 0.0
