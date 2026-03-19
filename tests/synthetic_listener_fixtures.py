from __future__ import annotations

from typing import Any


def listen_report(
    overall: float,
    *,
    verdict: str = "promising",
    gate_status: str = "pass",
    structure: float = 72.0,
    groove: float = 72.0,
    energy_arc: float = 72.0,
    transition: float = 72.0,
    coherence: float = 72.0,
    mix_sanity: float = 72.0,
    song_likeness: float = 72.0,
    song_metrics: dict[str, float] | None = None,
    component_fixes: dict[str, list[str]] | None = None,
    top_fixes: list[str] | None = None,
    top_reasons: list[str] | None = None,
) -> dict[str, Any]:
    aggregate_metrics: dict[str, float] = {
        "backbone_continuity": 0.72,
        "readable_section_ratio": 0.73,
        "recognizable_section_ratio": 0.71,
        "boundary_recovery": 0.68,
        "role_plausibility": 0.67,
        "planner_audio_climax_conviction": 0.66,
        "climax_conviction": 0.68,
        "background_only_identity_gap": 0.08,
        "owner_switch_ratio": 0.32,
    }
    if song_metrics:
        aggregate_metrics.update(song_metrics)

    fixes = component_fixes or {}
    return {
        "source_path": "synthetic_case.wav",
        "duration_seconds": 60.0,
        "overall_score": overall,
        "structure": {"score": structure, "summary": "structure", "evidence": [], "fixes": fixes.get("structure", []), "details": {}},
        "groove": {"score": groove, "summary": "groove", "evidence": [], "fixes": fixes.get("groove", []), "details": {}},
        "energy_arc": {"score": energy_arc, "summary": "energy_arc", "evidence": [], "fixes": fixes.get("energy_arc", []), "details": {}},
        "transition": {"score": transition, "summary": "transition", "evidence": [], "fixes": fixes.get("transition", []), "details": {}},
        "coherence": {"score": coherence, "summary": "coherence", "evidence": [], "fixes": fixes.get("coherence", []), "details": {}},
        "mix_sanity": {"score": mix_sanity, "summary": "mix_sanity", "evidence": [], "fixes": fixes.get("mix_sanity", []), "details": {}},
        "song_likeness": {
            "score": song_likeness,
            "summary": "song_likeness",
            "evidence": [],
            "fixes": fixes.get("song_likeness", []),
            "details": {"aggregate_metrics": aggregate_metrics},
        },
        "verdict": verdict,
        "top_reasons": top_reasons or ["Synthetic fixture"],
        "top_fixes": top_fixes or [],
        "gating": {"status": gate_status, "raw_overall_score": overall},
        "analysis_version": "0.5.0",
    }


def owner_switch_ping_pong_fixture() -> dict[str, Any]:
    return listen_report(
        49.0,
        verdict="weak",
        gate_status="reject",
        song_likeness=61.0,
        groove=66.0,
        energy_arc=63.0,
        transition=44.0,
        coherence=64.0,
        mix_sanity=65.0,
        song_metrics={
            "owner_switch_ratio": 0.91,
            "backbone_continuity": 0.55,
        },
        component_fixes={
            "transition": [
                "Manifest ownership is flipping often enough to read like track switching.",
            ],
        },
        top_fixes=["Keep one backbone owner through adjacent sections instead of ping-pong handoffs."],
        top_reasons=["Synthetic bad output: parent ownership keeps ping-ponging."],
    )


def section_role_confusion_fixture() -> dict[str, Any]:
    return listen_report(
        63.0,
        verdict="mixed",
        gate_status="pass",
        structure=47.0,
        song_likeness=58.0,
        groove=61.0,
        energy_arc=59.0,
        transition=62.0,
        coherence=57.0,
        mix_sanity=66.0,
        song_metrics={
            "readable_section_ratio": 0.30,
            "recognizable_section_ratio": 0.33,
            "boundary_recovery": 0.31,
            "role_plausibility": 0.34,
            "backbone_continuity": 0.49,
        },
        component_fixes={
            "structure": [
                "Increase structural certainty so the planner is not forced into coarse whole-song windows.",
            ],
            "song_likeness": [
                "The arrangement does not maintain plausible section roles and keeps feeling like stitched chunks.",
            ],
        },
        top_fixes=["Sharpen readable section identity before trying another long-form program."],
        top_reasons=["Synthetic bad output: section identities are too blurry to read as one song."],
    )


def payoff_conviction_gap_fixture() -> dict[str, Any]:
    return listen_report(
        62.0,
        verdict="mixed",
        gate_status="review",
        structure=71.0,
        groove=64.0,
        energy_arc=48.0,
        transition=63.0,
        coherence=66.0,
        mix_sanity=68.0,
        song_likeness=60.0,
        song_metrics={
            "readable_section_ratio": 0.64,
            "recognizable_section_ratio": 0.62,
            "boundary_recovery": 0.59,
            "role_plausibility": 0.58,
            "planner_audio_climax_conviction": 0.34,
            "climax_conviction": 0.36,
            "backbone_continuity": 0.63,
        },
        component_fixes={
            "energy_arc": [
                "Build a real payoff instead of front-loading the hook.",
                "Keep late payoff conviction explicit in diagnostics so reranking can prefer sustained climaxes.",
            ],
        },
        top_fixes=["Bias section selection toward a stronger late payoff instead of spending the hook too early."],
        top_reasons=["Synthetic bad output: the child arc rises on paper but never lands a convincing late payoff."],
    )


def not_one_song_composite_fixture() -> dict[str, Any]:
    return listen_report(
        67.0,
        verdict="promising",
        gate_status="pass",
        structure=72.0,
        groove=70.0,
        energy_arc=68.0,
        transition=58.0,
        coherence=63.0,
        mix_sanity=64.0,
        song_likeness=61.0,
        song_metrics={
            "backbone_continuity": 0.34,
            "readable_section_ratio": 0.32,
            "recognizable_section_ratio": 0.36,
            "boundary_recovery": 0.33,
            "role_plausibility": 0.35,
            "planner_audio_climax_conviction": 0.38,
            "climax_conviction": 0.39,
            "background_only_identity_gap": 0.41,
            "owner_switch_ratio": 0.74,
            "composite_song_risk": 0.69,
        },
        component_fixes={
            "song_likeness": [
                "Hard reject stitched-composite arrangements that still read like multiple pasted songs instead of one continuous child record.",
            ],
        },
        top_fixes=["Reject outputs that still read like a stitched composite instead of one child song."],
        top_reasons=["Synthetic bad output: the arrangement behaves like multiple pasted songs rather than one record."],
    )


def unstable_groove_review_fixture() -> dict[str, Any]:
    return listen_report(
        71.0,
        verdict="promising",
        gate_status="review",
        structure=80.0,
        groove=50.0,
        energy_arc=74.0,
        transition=69.0,
        coherence=58.0,
        mix_sanity=70.0,
        song_likeness=72.0,
        component_fixes={
            "groove": [
                "Tighten bar-to-bar rhythmic continuity; the beat grid exists but the pocket collapses abruptly across adjacent windows.",
            ],
        },
        top_fixes=["Stabilize groove before human review; the pocket is too loose to trust even though the macro arc is readable."],
        top_reasons=["Synthetic borderline output: the arrangement mostly reads, but groove stability is still below the listener gate."],
    )


def near_threshold_false_positive_review_fixture() -> dict[str, Any]:
    return listen_report(
        72.0,
        verdict="mixed",
        gate_status="review",
        structure=76.0,
        groove=60.0,
        energy_arc=63.0,
        transition=54.0,
        coherence=61.0,
        mix_sanity=63.0,
        song_likeness=61.0,
        song_metrics={
            "backbone_continuity": 0.42,
            "readable_section_ratio": 0.40,
            "recognizable_section_ratio": 0.40,
            "boundary_recovery": 0.38,
            "role_plausibility": 0.40,
            "planner_audio_climax_conviction": 0.52,
            "climax_conviction": 0.51,
            "background_only_identity_gap": 0.45,
            "owner_switch_ratio": 0.78,
            "composite_song_risk": 0.50,
        },
        component_fixes={
            "groove": [
                "Keep groove confidence above the listener floor before promoting to humans.",
            ],
            "transition": [
                "Tighten seams a bit more before calling the arrangement human-review ready.",
            ],
        },
        top_fixes=["Hold this in review: it is imperfect, but it should not be hard-rejected as a fake non-song."],
        top_reasons=["Synthetic regression fixture: near-threshold listener metrics should stay review-only, not flip into a false hard reject."],
    )
