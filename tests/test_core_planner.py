from collections import Counter

import pytest

from src.core.analysis.models import SongDNA
from src.core.planner import build_compatibility_report, build_stub_arrangement_plan
from src.core.planner.arrangement import _SectionSpec, _WindowSelection, _apply_section_level_authenticity_guard, _backbone_selection_guard_reason, _build_section_program, _choose_backbone_parent, _choose_with_major_section_balance_guard, _enumerate_section_choices, _infer_transition_mode, _phrase_window_candidates, _pick_candidate, _planner_listen_feedback, _selection_backbone_continuity_penalty, _selection_opening_continuity_penalty


def make_song(path: str, tempo: float, tonic: str, mode: str, camelot: str, sections: int, mean_rms: float) -> SongDNA:
    return SongDNA(
        source_path=path,
        sample_rate=44100,
        duration_seconds=180.0,
        tempo_bpm=tempo,
        key={"tonic": tonic, "mode": mode, "camelot": camelot, "confidence": 0.9},
        structure={"sections": [{"label": f"section_{i}"} for i in range(sections)]},
        energy={"summary": {"mean_rms": mean_rms, "mean_bar_rms": mean_rms}, "derived": {"energy_confidence": 0.7}},
    )


def test_build_compatibility_report_returns_factorized_scores():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 4, 0.11)
    b = make_song("b.wav", 129.0, "A", "minor", "8A", 4, 0.10)

    report = build_compatibility_report(a, b).to_dict()

    assert report["factors"]["tempo"] >= 0.8
    assert report["factors"]["harmony"] >= 0.8
    assert report["factors"]["structure"] >= 0.8
    assert report["factors"]["overall"] > 0.0


def test_build_stub_arrangement_plan_returns_sections():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 4, 0.11)
    b = make_song("b.wav", 130.0, "C", "major", "8B", 5, 0.14)
    a.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0]
    b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]
    a.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]
    a.energy["beat_rms"] = [0.10, 0.12, 0.16, 0.18, 0.35, 0.40, 0.70, 0.75]
    b.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0]
    b.energy["beat_rms"] = [0.08, 0.10, 0.30, 0.34, 0.48, 0.52, 0.58, 0.62, 0.66, 0.70]

    plan = build_stub_arrangement_plan(a, b).to_dict()

    assert len(plan["sections"]) == 5
    assert [section["label"] for section in plan["sections"]] == ["intro", "verse", "build", "payoff", "outro"]
    assert all(section["source_section_label"].startswith("phrase_") for section in plan["sections"])
    assert "compatibility" in plan
    assert "planning_diagnostics" in plan
    assert plan["planning_diagnostics"]["planner_evaluator_bridge"] == "listen-aligned planner diagnostics"
    assert set(plan["planning_diagnostics"]["parent_listen_feedback"].keys()) == {"A", "B"}
    assert len(plan["planning_diagnostics"]["selected_sections"]) == len(plan["sections"])
    payoff_diag = next(item for item in plan["planning_diagnostics"]["selected_sections"] if item["label"] == "payoff")
    assert "evaluator_alignment" in payoff_diag
    assert "seam_risk" in payoff_diag["evaluator_alignment"]
    assert "transition_readiness" in payoff_diag["evaluator_alignment"]
    assert payoff_diag["transition_mode"] in {"same_parent_flow", "single_owner_handoff", "crossfade_support", "arrival_handoff", None}
    assert any("boundary confidence" in note for note in plan["planning_notes"])
    assert any("capacity-aware" in note for note in plan["planning_notes"])


def test_choose_backbone_parent_prefers_parent_with_stronger_groove_coherence_and_capacity():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.18)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 5, 0.18)

    a.duration_seconds = 64.0
    b.duration_seconds = 48.0
    a.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
    b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    a.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
    b.metadata["tempo"] = {"beat_times": [0.0, 0.52, 0.97, 1.57, 2.01, 2.63, 3.05, 3.71]}
    a.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.8, "hook_strength": 0.62, "hook_repetition": 0.56}
    b.energy["derived"] = {"energy_confidence": 0.7, "payoff_strength": 0.5, "hook_strength": 0.32, "hook_repetition": 0.28}
    a.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
    b.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.28, 0.34, 0.42, 0.48, 0.58, 0.64, 0.74, 0.80, 0.88, 0.92, 0.50, 0.40]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.30, 0.34, 0.42, 0.48, 0.56, 0.60, 0.68, 0.72]

    backbone = _choose_backbone_parent(a, b)

    assert backbone.backbone_parent == "A"
    assert backbone.donor_parent == "B"
    assert backbone.backbone_score > backbone.donor_score
    assert any("groove=" in reason for reason in backbone.backbone_reasons)


def test_build_section_program_scales_from_compact_to_extended_shapes():
    compact_a = make_song("compact_a.wav", 128.0, "A", "minor", "8A", 3, 0.11)
    compact_b = make_song("compact_b.wav", 128.0, "A", "minor", "8A", 3, 0.11)
    compact_a.duration_seconds = compact_b.duration_seconds = 32.0
    compact_a.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0]
    compact_b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0]

    standard_a = make_song("standard_a.wav", 128.0, "A", "minor", "8A", 5, 0.11)
    standard_b = make_song("standard_b.wav", 128.0, "A", "minor", "8A", 5, 0.11)
    standard_a.duration_seconds = standard_b.duration_seconds = 48.0
    standard_a.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]
    standard_b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]

    extended_a = make_song("extended_a.wav", 128.0, "A", "minor", "8A", 7, 0.11)
    extended_b = make_song("extended_b.wav", 128.0, "A", "minor", "8A", 7, 0.11)
    extended_a.duration_seconds = extended_b.duration_seconds = 64.0
    extended_a.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
    extended_b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
    extended_a.energy["beat_times"] = extended_b.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
    extended_a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.18, 0.26, 0.30, 0.42, 0.46, 0.74, 0.78, 0.30, 0.26, 0.82, 0.86, 0.90, 0.94]
    extended_b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.30, 0.34, 0.48, 0.52, 0.76, 0.80, 0.34, 0.30, 0.84, 0.88, 0.92, 0.96]

    compact = _build_section_program(compact_a, compact_b)
    standard = _build_section_program(standard_a, standard_b)
    extended = _build_section_program(extended_a, extended_b)

    assert [spec.label for spec in compact] == ["intro", "build", "payoff"]
    assert [spec.label for spec in standard] == ["intro", "verse", "build", "payoff", "outro"]
    assert [spec.label for spec in extended] == ["intro", "verse", "build", "payoff", "bridge", "payoff", "outro"]


def test_build_section_program_avoids_forcing_bridge_and_second_payoff_without_reset_relaunch_support():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.18)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.18, 0.24, 0.26, 0.36, 0.38, 0.52, 0.54, 0.58, 0.60, 0.62, 0.64, 0.42, 0.38]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.28, 0.30, 0.40, 0.42, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.46, 0.42]

    program = _build_section_program(a, b)

    assert [spec.label for spec in program] == ["intro", "verse", "build", "payoff", "outro"]


def test_build_section_program_requires_shared_reset_relaunch_support_for_extended_shape():
    strong = make_song("strong.wav", 128.0, "A", "minor", "8A", 7, 0.18)
    weak = make_song("weak.wav", 128.0, "A", "minor", "8A", 7, 0.18)

    for song in (strong, weak):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]

    strong.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.18, 0.26, 0.30, 0.42, 0.46, 0.74, 0.78, 0.30, 0.26, 0.82, 0.86, 0.90, 0.94]
    weak.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.28, 0.30, 0.40, 0.42, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.46, 0.42]

    program = _build_section_program(strong, weak)

    assert [spec.label for spec in program] == ["intro", "verse", "build", "payoff", "outro"]


def test_build_section_program_downgrades_extended_form_when_second_payoff_does_not_beat_first():
    a = make_song("flat_family_a.wav", 128.0, "A", "minor", "8A", 7, 0.18)
    b = make_song("flat_family_b.wav", 128.0, "A", "minor", "8A", 7, 0.18)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.18, 0.28, 0.32, 0.46, 0.50, 0.88, 0.92, 0.28, 0.24, 0.68, 0.70, 0.68, 0.66]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.30, 0.34, 0.48, 0.52, 0.86, 0.90, 0.30, 0.26, 0.66, 0.68, 0.66, 0.64]

    program = _build_section_program(a, b)

    assert [spec.label for spec in program] == ["intro", "verse", "build", "payoff", "outro"]


def test_build_stub_arrangement_plan_exposes_backbone_vs_donor_diagnostics_and_keeps_structural_sections_on_backbone():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.18)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.20)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]

    a.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
    b.metadata["tempo"] = {"beat_times": [0.0, 0.52, 0.97, 1.57, 2.01, 2.63, 3.05, 3.71]}
    a.energy["derived"] = {"energy_confidence": 0.92, "payoff_strength": 0.84, "hook_strength": 0.68, "hook_repetition": 0.56}
    b.energy["derived"] = {"energy_confidence": 0.60, "payoff_strength": 0.74, "hook_strength": 0.62, "hook_repetition": 0.52}

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.28, 0.34, 0.42, 0.48, 0.56, 0.62, 0.72, 0.78, 0.86, 0.92]
    b.energy["beat_rms"] = [0.09, 0.11, 0.18, 0.22, 0.32, 0.38, 0.50, 0.56, 0.64, 0.70, 0.82, 0.88, 0.90, 0.94]

    plan = build_stub_arrangement_plan(a, b).to_dict()

    backbone = plan["planning_diagnostics"]["backbone_plan"]
    assert backbone["backbone_parent"] == "A"
    assert backbone["donor_parent"] == "B"
    assert backbone["section_usage"]["A"] >= backbone["section_usage"]["B"]

    structural_sections = [section for section in plan["sections"] if section["label"] in {"intro", "verse", "bridge", "outro"}]
    assert structural_sections
    assert all(section["source_parent"] == "A" for section in structural_sections)
    assert any(item["selected_role"] == "donor" for item in plan["planning_diagnostics"]["selected_sections"])
    assert any("backbone-first child-song architecture" in note for note in plan["planning_notes"])


def test_backbone_selection_guard_requires_structural_backbone_and_build_donor():
    spec_verse = _SectionSpec(label="verse", start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference="A")
    spec_build = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference="B")

    assert _backbone_selection_guard_reason(spec_verse, "B", [], backbone_parent="A", donor_parent="B") == "structural_backbone_only"
    assert _backbone_selection_guard_reason(spec_verse, "A", [], backbone_parent="A", donor_parent="B") is None
    assert _backbone_selection_guard_reason(spec_build, "A", [], backbone_parent="A", donor_parent="B") == "build_donor_only"
    assert _backbone_selection_guard_reason(spec_build, "B", [], backbone_parent="A", donor_parent="B") is None



def test_backbone_selection_guard_blocks_donor_reentry_after_backbone_reclaim():
    payoff_spec = _SectionSpec(label="payoff", start_bar=48, bar_count=16, target_energy=0.90, source_parent_preference=None)
    prior = [
        _WindowSelection(parent_id="B", song=make_song("b1.wav", 128.0, "A", "minor", "8A", 7, 0.2), candidate=None, blended_error=0.0, section_label="build", score_breakdown={}),
        _WindowSelection(parent_id="B", song=make_song("b2.wav", 128.0, "A", "minor", "8A", 7, 0.2), candidate=None, blended_error=0.0, section_label="payoff", score_breakdown={}),
        _WindowSelection(parent_id="A", song=make_song("a1.wav", 128.0, "A", "minor", "8A", 7, 0.2), candidate=None, blended_error=0.0, section_label="bridge", score_breakdown={}),
    ]

    assert _backbone_selection_guard_reason(payoff_spec, "B", prior, backbone_parent="A", donor_parent="B") == "donor_reentry_after_backbone"
    assert _backbone_selection_guard_reason(payoff_spec, "A", prior, backbone_parent="A", donor_parent="B") is None


def test_build_stub_arrangement_plan_prefers_higher_energy_late_phrase_for_payoff():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 3, 0.11)
    b = make_song("b.wav", 130.0, "C", "major", "8B", 6, 0.14)
    b.duration_seconds = 48.0
    b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    b.energy["beat_times"] = [
        2.0, 6.0,
        10.0, 14.0,
        18.0, 22.0,
        26.0, 30.0,
        34.0, 38.0,
        42.0, 46.0,
    ]
    b.energy["beat_rms"] = [
        0.10, 0.12,
        0.18, 0.20,
        0.24, 0.28,
        0.32, 0.36,
        0.38, 0.40,
        0.88, 0.92,
    ]

    plan = build_stub_arrangement_plan(a, b).to_dict()
    payoff = next(section for section in plan["sections"] if section["label"] == "payoff")

    assert payoff["source_parent"] == "B"
    assert payoff["source_section_label"] in {"phrase_1_5", "phrase_2_6"}


def test_role_priors_prefer_intro_like_window_for_intro_role():
    song = make_song("a.wav", 128.0, "A", "minor", "8A", 5, 0.20)
    song.duration_seconds = 40.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]
    song.structure["novelty_boundaries_seconds"] = [7.5, 23.5]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0]
    song.energy["beat_rms"] = [0.10, 0.12, 0.28, 0.30, 0.42, 0.46, 0.72, 0.76, 0.22, 0.18]

    candidate = _pick_candidate(song, target_position="early", bar_count=8, target_energy=0.22, role="intro")

    assert candidate.label == "phrase_0_2"


def test_role_priors_prefer_novel_late_drop_for_bridge_role():
    song = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    song.duration_seconds = 48.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.structure["novelty_boundaries_seconds"] = [31.5, 39.5]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    song.energy["beat_rms"] = [
        0.18, 0.20,
        0.55, 0.58,
        0.54, 0.57,
        0.86, 0.88,
        0.34, 0.30,
        0.78, 0.82,
    ]

    candidate = _pick_candidate(song, target_position="late", bar_count=8, target_energy=0.50, role="bridge")

    assert candidate.label == "phrase_3_5"


def test_build_role_prior_prefers_rising_window_over_already_flat_payoff_plateau():
    song = make_song("build_bias.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    song.duration_seconds = 48.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    song.energy["beat_rms"] = [
        0.08, 0.10,
        0.62, 0.64,
        0.68, 0.70,
        0.34, 0.38,
        0.60, 0.72,
        0.78, 0.82,
    ]

    candidate = _pick_candidate(song, target_position="mid", bar_count=8, target_energy=0.55, role="build")

    assert candidate.label == "phrase_3_5"


def test_build_role_prior_prefers_consistent_ramp_over_flashy_late_spike():
    song = make_song("build_consistency.wav", 128.0, "A", "minor", "8A", 7, 0.20)
    song.duration_seconds = 56.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
    song.energy["beat_rms"] = [
        0.08, 0.10,
        0.18, 0.22,
        0.82, 0.24,
        0.26, 0.32,
        0.28, 0.34,
        0.42, 0.48,
        0.58, 0.64,
    ]

    candidate = _pick_candidate(song, target_position="mid", bar_count=8, target_energy=0.55, role="build")

    assert candidate.label == "phrase_4_6"


def test_infer_transition_mode_prefers_single_owner_handoff_for_cross_parent_build():
    song_a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    song_b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    previous = _WindowSelection(
        parent_id="A",
        song=song_a,
        candidate=_pick_candidate(song_a, target_position="mid", bar_count=8, target_energy=0.42, role="verse"),
        blended_error=0.10,
        score_breakdown={},
        section_label="verse",
    )
    chosen = _WindowSelection(
        parent_id="B",
        song=song_b,
        candidate=_pick_candidate(song_b, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.12,
        score_breakdown={},
        section_label="build",
    )
    spec = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference="B", transition_in="blend", transition_out="swap")

    assert _infer_transition_mode(spec, chosen, previous, "verse") == "single_owner_handoff"


def test_build_plan_uses_sequential_transition_viability_to_keep_energy_rising_into_payoff():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.18)
    b = make_song("b.wav", 129.0, "A", "minor", "8A", 6, 0.18)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]

    a.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    a.energy["beat_rms"] = [0.08, 0.10, 0.12, 0.16, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46]

    b.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    b.energy["beat_rms"] = [0.20, 0.24, 0.34, 0.38, 0.50, 0.54, 0.62, 0.66, 0.70, 0.74, 0.92, 0.96]

    plan = build_stub_arrangement_plan(a, b).to_dict()
    build = next(section for section in plan["sections"] if section["label"] == "build")
    payoff = next(section for section in plan["sections"] if section["label"] == "payoff")

    assert build["source_parent"] == "B"
    assert payoff["source_parent"] == "B"
    assert payoff["source_section_label"] == "phrase_2_6"


def test_build_plan_prefers_steady_intro_and_still_avoids_energy_backslide_in_build():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]

    a.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    a.energy["beat_rms"] = [0.08, 0.10, 0.12, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46]

    b.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    b.energy["beat_rms"] = [
        0.24, 0.28,
        0.68, 0.72,
        0.18, 0.22,
        0.24, 0.28,
        0.78, 0.82,
        0.90, 0.94,
    ]

    plan = build_stub_arrangement_plan(a, b).to_dict()
    intro = next(section for section in plan["sections"] if section["label"] == "intro")
    build = next(section for section in plan["sections"] if section["label"] == "build")
    payoff = next(section for section in plan["sections"] if section["label"] == "payoff")

    assert intro["source_parent"] == "A"
    assert intro["source_section_label"] == "phrase_0_2"
    assert build["source_section_label"] == "phrase_3_5"
    assert payoff["source_section_label"] == "phrase_2_6"


def test_build_plan_allows_blend_transition_to_choose_stronger_upward_lift_when_seam_is_safe():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
        song.energy["rms"] = song.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.26, 0.28, 0.34, 0.36, 0.40, 0.42, 0.46, 0.48]
        song.energy["spectral_centroid"] = [1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020]
        song.energy["spectral_rolloff"] = [3400, 3420, 3440, 3460, 3480, 3500, 3520, 3540, 3560, 3580, 3600, 3620]
        song.energy["onset_density"] = [0.18, 0.18, 0.20, 0.20, 0.22, 0.22, 0.24, 0.24, 0.26, 0.26, 0.28, 0.28]
        song.energy["low_band_ratio"] = [0.28, 0.28, 0.30, 0.30, 0.32, 0.32, 0.34, 0.34, 0.36, 0.36, 0.38, 0.38]
        song.energy["spectral_flatness"] = [0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.15, 0.15, 0.16, 0.16, 0.17, 0.17]

    # Anchor the previous section in A, then give B two plausible build choices:
    # phrase_1_3 is flatter/safer, phrase_2_4 gives a stronger lift while staying seam-safe.
    a.energy["beat_rms"] = a.energy["rms"] = [0.08, 0.10, 0.12, 0.14, 0.18, 0.20, 0.22, 0.24, 0.28, 0.30, 0.34, 0.36]
    b.energy["beat_rms"] = b.energy["rms"] = [0.10, 0.12, 0.20, 0.22, 0.28, 0.30, 0.46, 0.48, 0.50, 0.52, 0.62, 0.64]

    previous = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="early", bar_count=8, target_energy=0.25, role="intro"),
        blended_error=0.0,
        score_breakdown={},
    )
    spec = _SectionSpec(label="build", start_bar=8, bar_count=8, target_energy=0.55, source_parent_preference="B", transition_in="blend", transition_out="swap")
    ranked = _enumerate_section_choices(spec, a, b, previous)

    best = ranked[0]
    flatter_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_1_3")
    stronger_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_4")

    assert stronger_b.blended_error < flatter_b.blended_error
    assert best.parent_id == "B"
    assert best.candidate.label == "phrase_2_4"


def test_payoff_drop_transition_prefers_stronger_safe_landing_over_flatter_option():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
        song.energy["rms"] = song.energy["beat_rms"] = [0.18, 0.20, 0.24, 0.26, 0.30, 0.34, 0.40, 0.44, 0.50, 0.54, 0.60, 0.64]
        song.energy["spectral_centroid"] = [1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020]
        song.energy["spectral_rolloff"] = [3400, 3420, 3440, 3460, 3480, 3500, 3520, 3540, 3560, 3580, 3600, 3620]
        song.energy["onset_density"] = [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]
        song.energy["low_band_ratio"] = [0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39]
        song.energy["spectral_flatness"] = [0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.15, 0.15, 0.16, 0.16, 0.17, 0.17]

    a.energy["beat_rms"] = a.energy["rms"] = [0.10, 0.12, 0.18, 0.22, 0.28, 0.34, 0.38, 0.44, 0.48, 0.54, 0.58, 0.62]
    b.energy["beat_rms"] = b.energy["rms"] = [0.16, 0.18, 0.24, 0.28, 0.34, 0.38, 0.52, 0.56, 0.64, 0.68, 0.70, 0.74]

    previous = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )
    spec = _SectionSpec(label="payoff", start_bar=16, bar_count=8, target_energy=0.86, source_parent_preference=None, transition_in="drop")
    ranked = _enumerate_section_choices(spec, a, b, previous)

    flatter_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_4")
    stronger_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_3_5")

    assert stronger_b.score_breakdown["transition_impact"] < flatter_b.score_breakdown["transition_impact"]
    assert stronger_b.blended_error < flatter_b.blended_error
    assert ranked[0].parent_id == "B"
    assert ranked[0].candidate.label in {"phrase_3_5", "phrase_4_6"}


def test_extended_stub_arrangement_plan_includes_bridge_and_second_payoff_when_capacity_is_high():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.18)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.18, 0.26, 0.30, 0.42, 0.46, 0.74, 0.78, 0.30, 0.26, 0.82, 0.86, 0.90, 0.94]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.30, 0.34, 0.48, 0.52, 0.76, 0.80, 0.34, 0.30, 0.84, 0.88, 0.92, 0.96]

    plan = build_stub_arrangement_plan(a, b).to_dict()
    labels = [section["label"] for section in plan["sections"]]

    assert labels == ["intro", "verse", "build", "payoff", "bridge", "payoff", "outro"]
    assert plan["sections"][4]["transition_in"] == "swap"
    assert plan["sections"][5]["label"] == "payoff"
    assert plan["sections"][5]["start_bar"] == 48


def test_phrase_window_candidates_add_phrase_trim_alternates_for_overlong_windows():
    song = make_song("trimmed.wav", 132.0, "A", "minor", "8A", 4, 0.20)
    song.duration_seconds = 42.5
    song.structure["phrase_boundaries_seconds"] = [0.0, 10.625, 21.25, 31.875, 42.5]
    song.structure["section_boundaries_seconds"] = [10.625, 21.25, 31.875]
    song.energy["beat_times"] = [float(i) for i in range(1, 43)]
    song.energy["beat_rms"] = [0.20 + (0.004 * i) for i in range(42)]

    candidates = _enumerate_section_choices(
        _SectionSpec(label="verse", start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference="A", transition_in="blend", transition_out="lift"),
        song,
        song,
        previous=None,
        prior_selections=[],
    )

    trimmed = [item for item in candidates if item.candidate.origin == "phrase_trim"]
    assert trimmed
    assert any(item.candidate.label.startswith("phrase_0_2_trim_") for item in trimmed)
    assert any(0.82 <= item.candidate.duration / ((60.0 / song.tempo_bpm) * 32.0) <= 1.18 for item in trimmed)



def test_enumerate_section_choices_uses_backbone_tempo_for_stretch_instead_of_candidate_parent_tempo():
    a = make_song("a.wav", 120.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 90.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
        song.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.30, 0.34, 0.42, 0.46, 0.56, 0.60, 0.68, 0.72]

    spec = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference="B", transition_in="blend", transition_out="swap")
    ranked = _enumerate_section_choices(spec, a, b, previous=None, backbone_parent="A", donor_parent="B")

    donor = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_4")
    assert donor.score_breakdown["target_duration_seconds"] == pytest.approx(16.0, rel=1e-3)
    assert donor.score_breakdown["stretch_ratio"] == pytest.approx(1.0, rel=1e-3)


def test_enumerate_section_choices_blocks_donor_reentry_after_backbone_returns():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.18)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.22)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]

    a.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.28, 0.34, 0.40, 0.46, 0.56, 0.62, 0.68, 0.74, 0.76, 0.80]
    b.energy["beat_rms"] = [0.08, 0.10, 0.24, 0.30, 0.54, 0.60, 0.78, 0.84, 0.74, 0.78, 0.64, 0.68, 0.70, 0.74]

    prior = [
        _WindowSelection(parent_id="A", song=a, candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.42, role="verse"), blended_error=0.32, score_breakdown={}, section_label="verse"),
        _WindowSelection(parent_id="B", song=b, candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.58, role="build"), blended_error=0.28, score_breakdown={}, section_label="build"),
        _WindowSelection(parent_id="B", song=b, candidate=_pick_candidate(b, target_position="late", bar_count=16, target_energy=0.86, role="payoff"), blended_error=0.24, score_breakdown={}, section_label="payoff"),
        _WindowSelection(parent_id="A", song=a, candidate=_pick_candidate(a, target_position="late", bar_count=8, target_energy=0.52, role="bridge"), blended_error=0.30, score_breakdown={}, section_label="bridge"),
    ]

    spec = _SectionSpec(label="payoff", start_bar=48, bar_count=16, target_energy=0.90, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous=prior[-1], prior_selections=prior, backbone_parent="A", donor_parent="B")

    assert all(item.parent_id != "B" for item in ranked)
    assert ranked
    assert ranked[0].parent_id == "A"


def test_phrase_trim_candidate_reduces_stretch_pressure_for_overlong_phrase_window():
    song = make_song("stretchy.wav", 132.0, "A", "minor", "8A", 5, 0.22)
    song.duration_seconds = 42.5
    song.structure["phrase_boundaries_seconds"] = [0.0, 10.625, 21.25, 31.875, 42.5]
    song.structure["section_boundaries_seconds"] = [10.625, 21.25, 31.875]
    song.energy["beat_times"] = [float(i) for i in range(1, 43)]
    song.energy["beat_rms"] = [0.18 + (0.004 * i) for i in range(42)]

    ranked = _enumerate_section_choices(
        _SectionSpec(label="verse", start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference="A", transition_in="blend", transition_out="lift"),
        song,
        song,
        previous=None,
        prior_selections=[],
    )

    base = next(item for item in ranked if item.candidate.label == "phrase_0_2")
    trimmed = next(item for item in ranked if item.candidate.label.startswith("phrase_0_2_trim_"))

    assert base.score_breakdown["stretch_ratio"] > 1.25
    assert trimmed.score_breakdown["stretch_ratio"] < base.score_breakdown["stretch_ratio"]
    assert trimmed.score_breakdown["stretch_penalty"] < base.score_breakdown["stretch_penalty"]



def test_build_plan_can_override_parent_preference_when_other_parent_has_much_better_payoff_window():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]

    a.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    a.energy["beat_rms"] = [0.08, 0.10, 0.18, 0.22, 0.24, 0.28, 0.30, 0.34, 0.36, 0.40, 0.42, 0.46]

    b.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.22, 0.24, 0.28, 0.30, 0.36, 0.40, 0.94, 0.98]

    plan = build_stub_arrangement_plan(a, b).to_dict()
    payoff = next(section for section in plan["sections"] if section["label"] == "payoff")

    assert payoff["source_parent"] == "B"
    assert payoff["source_section_label"] == "phrase_2_6"


def test_build_plan_penalizes_high_seam_risk_even_when_target_energy_matches():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]

    a.energy["beat_rms"] = [0.18, 0.20, 0.22, 0.24, 0.28, 0.30, 0.38, 0.42, 0.52, 0.56, 0.74, 0.78]
    a.energy["rms"] = a.energy["beat_rms"]
    a.energy["spectral_centroid"] = [1700, 1750, 1800, 1850, 1900, 1950, 2050, 2100, 2200, 2250, 2350, 2400]
    a.energy["spectral_rolloff"] = [3200, 3300, 3400, 3500, 3600, 3700, 3900, 4000, 4200, 4300, 4500, 4600]
    a.energy["onset_density"] = [0.18, 0.19, 0.20, 0.21, 0.24, 0.25, 0.28, 0.30, 0.34, 0.35, 0.38, 0.40]
    a.energy["low_band_ratio"] = [0.30, 0.31, 0.31, 0.32, 0.34, 0.35, 0.36, 0.37, 0.39, 0.40, 0.41, 0.42]
    a.energy["spectral_flatness"] = [0.11, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.15, 0.15, 0.16, 0.16]

    b.energy["beat_rms"] = [0.10, 0.12, 0.16, 0.18, 0.54, 0.56, 0.56, 0.58, 0.86, 0.88, 0.90, 0.92]
    b.energy["rms"] = b.energy["beat_rms"]
    b.energy["spectral_centroid"] = [1400, 1450, 1550, 1600, 6400, 6500, 6550, 6600, 6600, 6650, 6700, 6750]
    b.energy["spectral_rolloff"] = [2800, 2900, 3000, 3100, 11800, 11900, 12000, 12100, 12100, 12200, 12300, 12400]
    b.energy["onset_density"] = [0.14, 0.15, 0.18, 0.19, 0.72, 0.74, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84]
    b.energy["low_band_ratio"] = [0.22, 0.24, 0.26, 0.28, 0.82, 0.84, 0.84, 0.86, 0.86, 0.87, 0.88, 0.89]
    b.energy["spectral_flatness"] = [0.09, 0.10, 0.11, 0.11, 0.36, 0.37, 0.37, 0.38, 0.39, 0.39, 0.40, 0.40]

    intro_previous = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="early", bar_count=8, target_energy=0.25, role="intro"),
        blended_error=0.0,
        score_breakdown={},
    )
    spec = _SectionSpec(label="build", start_bar=8, bar_count=8, target_energy=0.55, source_parent_preference="B", transition_in="blend", transition_out="swap")
    ranked = _enumerate_section_choices(spec, a, b, intro_previous)

    a_build = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_2_4")
    b_build = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_4")

    assert b_build.score_breakdown["seam_risk"] > 0.5
    assert b_build.score_breakdown["seam_risk"] > a_build.score_breakdown["seam_risk"]
    assert b_build.score_breakdown["seam_spectral_jump"] > a_build.score_breakdown["seam_spectral_jump"]
    assert b_build.score_breakdown["seam_onset_jump"] > a_build.score_breakdown["seam_onset_jump"]

    plan = build_stub_arrangement_plan(a, b).to_dict()
    notes_blob = "\n".join(plan["planning_notes"])
    assert "seam_risk" in notes_blob


def test_payoff_role_prior_uses_derived_payoff_signal_when_available():
    song = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    song.duration_seconds = 48.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
    song.energy["bar_times"] = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0]
    song.energy["bar_rms"] = [0.12, 0.14, 0.18, 0.22, 0.26, 0.30, 0.36, 0.40, 0.48, 0.52, 0.70, 0.74]
    song.energy["derived"] = {
        "energy_confidence": 0.95,
        "payoff_windows": [{"start": 16.0, "end": 48.0, "score": 0.98}],
        "hook_windows": [{"start": 16.0, "end": 48.0, "score": 0.72}],
    }

    candidate = _pick_candidate(song, target_position="late", bar_count=16, target_energy=0.85, role="payoff")

    assert candidate.label == "phrase_2_6"


def test_enumerate_section_choices_penalizes_reusing_same_parent_window_from_history():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]

    a.energy["beat_rms"] = [0.12, 0.14, 0.22, 0.24, 0.38, 0.40, 0.54, 0.56, 0.66, 0.68, 0.78, 0.80]
    b.energy["beat_rms"] = [0.10, 0.12, 0.20, 0.22, 0.40, 0.42, 0.58, 0.60, 0.62, 0.64, 0.70, 0.72]

    previous = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.55, role="build"),
        blended_error=0.0,
        score_breakdown={},
    )
    spec = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference="B", transition_in="swap")

    ranked_without_history = _enumerate_section_choices(spec, a, b, previous)
    ranked_with_history = _enumerate_section_choices(spec, a, b, previous, prior_selections=[previous])

    best_without_history = ranked_without_history[0]
    best_with_history = ranked_with_history[0]
    repeated_b_window = next(item for item in ranked_with_history if item.parent_id == "B" and item.candidate.label == previous.candidate.label)

    assert best_without_history.parent_id == "B"
    assert best_without_history.candidate.label == previous.candidate.label
    assert repeated_b_window.score_breakdown["selection_reuse"] > 0.9
    assert best_with_history.score_breakdown["selection_reuse"] < repeated_b_window.score_breakdown["selection_reuse"]
    assert (best_with_history.parent_id, best_with_history.candidate.label) != (previous.parent_id, previous.candidate.label)


def test_payoff_role_prior_prefers_sustained_late_plateau_over_spiky_peak():
    song = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.20)
    song.duration_seconds = 56.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
    song.energy["beat_rms"] = [
        0.10, 0.12,
        0.16, 0.18,
        0.42, 0.96,
        0.40, 0.38,
        0.52, 0.56,
        0.78, 0.82,
        0.86, 0.90,
    ]

    candidate = _pick_candidate(song, target_position="late", bar_count=16, target_energy=0.85, role="payoff")

    assert candidate.label == "phrase_3_7"


def test_enumerate_section_choices_penalizes_one_parent_dominating_when_other_parent_can_cover_the_role():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.20)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.75, "hook_strength": 0.6, "hook_repetition": 0.5}

    # A is slightly stronger overall, but B is still a musically plausible build source.
    a.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.34, 0.38, 0.54, 0.58, 0.66, 0.70, 0.78, 0.82, 0.88, 0.92]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.30, 0.34, 0.48, 0.52, 0.60, 0.64, 0.72, 0.76, 0.82, 0.86]

    prior_intro = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="early", bar_count=8, target_energy=0.25, role="intro"),
        blended_error=0.0,
        score_breakdown={},
        section_label="intro",
    )
    prior_verse = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.42, role="verse"),
        blended_error=0.0,
        score_breakdown={},
        section_label="verse",
    )

    spec = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference="B", transition_in="blend", transition_out="swap")
    ranked = _enumerate_section_choices(spec, a, b, previous=prior_verse, prior_selections=[prior_intro, prior_verse])

    best = ranked[0]
    best_a = next(item for item in ranked if item.parent_id == "A")
    best_b = next(item for item in ranked if item.parent_id == "B")

    assert best.parent_id == "B"
    assert best_a.score_breakdown["fusion_balance"] > 0.0
    assert best_a.score_breakdown["fusion_same_parent_run_bias"] > 0.0
    assert best_a.score_breakdown["fusion_preferred_parent_miss"] > 0.0
    assert best_a.score_breakdown["fusion_major_section_lockout"] > 0.0
    assert best_a.blended_error > best_b.blended_error


def test_build_stub_arrangement_plan_breaks_single_parent_major_section_collapse_when_other_parent_is_plausible():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.22)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.21)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.76, "hook_strength": 0.62, "hook_repetition": 0.55}

    # A stays a little stronger overall, but B has credible verse/build/payoff material.
    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.26, 0.30, 0.38, 0.44, 0.56, 0.62, 0.74, 0.80, 0.88, 0.92, 0.48, 0.38]
    b.energy["beat_rms"] = [0.09, 0.11, 0.18, 0.22, 0.28, 0.34, 0.42, 0.48, 0.52, 0.58, 0.70, 0.76, 0.84, 0.88, 0.44, 0.34]

    plan = build_stub_arrangement_plan(a, b).to_dict()

    major_sections = [section for section in plan["sections"] if section["label"] in {"verse", "build", "payoff", "bridge"}]
    major_parents = {section["source_parent"] for section in major_sections}
    build = next(section for section in plan["sections"] if section["label"] == "build")

    assert len(major_sections) >= 3
    assert major_parents == {"A", "B"}
    assert build["source_parent"] == "B"


def test_build_stub_arrangement_plan_pushes_for_more_than_token_second_parent_presence_when_late_major_choice_is_plausible():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.22)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.21)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.78, "hook_strength": 0.64, "hook_repetition": 0.55}

    # A still owns the stronger intro/verse contour, but B has credible build and late payoff material.
    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.30, 0.34, 0.42, 0.48, 0.56, 0.62, 0.78, 0.84, 0.94, 0.98]
    b.energy["beat_rms"] = [0.09, 0.11, 0.18, 0.22, 0.28, 0.34, 0.44, 0.50, 0.60, 0.66, 0.74, 0.80, 0.90, 0.94]

    plan = build_stub_arrangement_plan(a, b).to_dict()

    major_sections = [section for section in plan["sections"] if section["label"] in {"verse", "build", "payoff", "bridge"}]
    parent_counts = Counter(section["source_parent"] for section in plan["sections"])
    major_parent_counts = Counter(section["source_parent"] for section in major_sections)

    assert parent_counts["A"] >= 1
    assert parent_counts["B"] >= 2
    assert major_parent_counts["B"] >= 2


def test_section_level_authenticity_guard_breaks_full_section_monopoly_when_other_parent_has_safe_late_major_option():
    intro = _SectionSpec(label="intro", start_bar=0, bar_count=8, target_energy=0.24, source_parent_preference="A")
    verse = _SectionSpec(label="verse", start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference="A", transition_in="blend")
    build = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference="B", transition_in="blend", transition_out="swap")
    payoff = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")

    dominant_song = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.21)
    alternate_song = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.22)
    dominant_candidate = _pick_candidate(dominant_song, target_position="late", bar_count=16, target_energy=0.86, role="payoff")
    alternate_candidate = _pick_candidate(alternate_song, target_position="late", bar_count=16, target_energy=0.86, role="payoff")

    chosen = [
        _WindowSelection(parent_id="B", song=dominant_song, candidate=dominant_candidate, blended_error=0.40, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.20, "transition_viability": 0.18, "role_prior": 0.15, "groove_continuity": 0.0, "listen_groove_confidence": 0.82}, section_label="intro"),
        _WindowSelection(parent_id="B", song=dominant_song, candidate=dominant_candidate, blended_error=0.52, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.24, "transition_viability": 0.20, "role_prior": 0.18, "groove_continuity": 0.0, "listen_groove_confidence": 0.82}, section_label="verse"),
        _WindowSelection(parent_id="B", song=dominant_song, candidate=dominant_candidate, blended_error=0.60, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.26, "transition_viability": 0.22, "role_prior": 0.20, "groove_continuity": 0.0, "listen_groove_confidence": 0.82}, section_label="build"),
        _WindowSelection(parent_id="B", song=dominant_song, candidate=dominant_candidate, blended_error=0.72, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.28, "transition_viability": 0.24, "role_prior": 0.22, "groove_continuity": 0.0, "listen_groove_confidence": 0.82}, section_label="payoff"),
    ]
    ranked_choices = [
        [chosen[0]],
        [chosen[1]],
        [chosen[2], _WindowSelection(parent_id="A", song=alternate_song, candidate=alternate_candidate, blended_error=0.84, score_breakdown={"stretch_ratio": 1.02, "stretch_gate": 0.0, "seam_risk": 0.30, "transition_viability": 0.28, "role_prior": 0.24, "groove_continuity": 0.0, "listen_groove_confidence": 0.80}, section_label="build")],
        [chosen[3], _WindowSelection(parent_id="A", song=alternate_song, candidate=alternate_candidate, blended_error=0.90, score_breakdown={"stretch_ratio": 1.01, "stretch_gate": 0.0, "seam_risk": 0.32, "transition_viability": 0.26, "role_prior": 0.25, "groove_continuity": 0.0, "listen_groove_confidence": 0.79}, section_label="payoff")],
    ]

    updated, notes = _apply_section_level_authenticity_guard(
        [intro, verse, build, payoff],
        chosen,
        ranked_choices,
    )

    assert {selection.parent_id for selection in updated} == {"A", "B"}
    assert updated[-1].parent_id == "A"
    assert any("section-level authenticity guard" in note for note in notes)


def test_enumerate_section_choices_penalizes_token_non_major_second_parent_presence_when_major_identity_is_still_one_sided():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.22)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.21)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.78, "hook_strength": 0.64, "hook_repetition": 0.55}

    # B has owned the major story so far, while A has only shown up in a non-major intro slot.
    # A is still a plausible payoff source, so continuing with B should pay an explicit
    # major-identity cost instead of letting token section-level presence count as true fusion.
    a.energy["beat_rms"] = [0.09, 0.11, 0.18, 0.22, 0.30, 0.34, 0.42, 0.48, 0.58, 0.64, 0.78, 0.84, 0.90, 0.95]
    b.energy["beat_rms"] = [0.10, 0.12, 0.20, 0.24, 0.32, 0.36, 0.46, 0.52, 0.62, 0.68, 0.82, 0.88, 0.94, 0.98]

    prior_intro = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="early", bar_count=8, target_energy=0.24, role="intro"),
        blended_error=0.0,
        score_breakdown={},
        section_label="intro",
    )
    prior_verse = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.42, role="verse"),
        blended_error=0.0,
        score_breakdown={},
        section_label="verse",
    )
    prior_build = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )

    spec = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous=prior_build, prior_selections=[prior_intro, prior_verse, prior_build])

    a_payoff = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_3_7")
    b_payoff = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_3_7")

    assert b_payoff.score_breakdown["fusion_major_identity_gap"] > 0.0
    assert b_payoff.score_breakdown["fusion_weighted_identity_presence_gap"] > 0.0
    assert b_payoff.score_breakdown["fusion_balance"] > a_payoff.score_breakdown["fusion_balance"]
    assert a_payoff.blended_error < b_payoff.blended_error
    assert ranked[0].parent_id == "A"


def test_enumerate_section_choices_penalizes_snapping_back_to_majority_parent_after_single_major_cameo():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.22)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.21)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.80, "hook_strength": 0.66, "hook_repetition": 0.58}

    # Prior story is A,A,A,B on major sections. Returning immediately to A for the next major
    # section should pay a specific cameo-rebound penalty instead of treating B's one major slot
    # as sufficient identity coverage.
    a.energy["beat_rms"] = [0.08, 0.10, 0.18, 0.22, 0.30, 0.34, 0.44, 0.50, 0.60, 0.66, 0.76, 0.82, 0.90, 0.94, 0.54, 0.42]
    b.energy["beat_rms"] = [0.09, 0.11, 0.17, 0.21, 0.28, 0.32, 0.42, 0.48, 0.58, 0.64, 0.72, 0.78, 0.86, 0.90, 0.50, 0.40]

    prior_intro = _WindowSelection(parent_id="A", song=a, candidate=_pick_candidate(a, target_position="early", bar_count=8, target_energy=0.24, role="intro"), blended_error=0.0, score_breakdown={}, section_label="intro")
    prior_verse = _WindowSelection(parent_id="A", song=a, candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.42, role="verse"), blended_error=0.0, score_breakdown={}, section_label="verse")
    prior_build = _WindowSelection(parent_id="A", song=a, candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.58, role="build"), blended_error=0.0, score_breakdown={}, section_label="build")
    prior_payoff = _WindowSelection(parent_id="B", song=b, candidate=_pick_candidate(b, target_position="late", bar_count=16, target_energy=0.86, role="payoff"), blended_error=0.0, score_breakdown={}, section_label="payoff")

    spec = _SectionSpec(label="bridge", start_bar=40, bar_count=8, target_energy=0.52, source_parent_preference="A", transition_in="swap", transition_out="lift")
    ranked = _enumerate_section_choices(spec, a, b, previous=prior_payoff, prior_selections=[prior_intro, prior_verse, prior_build, prior_payoff])

    best_a = next(item for item in ranked if item.parent_id == "A")
    best_b = next(item for item in ranked if item.parent_id == "B")

    assert best_a.score_breakdown["fusion_single_cameo_rebound_gap"] > 0.0
    assert best_a.score_breakdown["fusion_balance"] > best_b.score_breakdown["fusion_balance"]
    assert ranked[0].parent_id == "B"



def test_section_level_authenticity_guard_breaks_major_section_monopoly_even_when_intro_outro_already_use_both_parents():
    intro = _SectionSpec(label="intro", start_bar=0, bar_count=8, target_energy=0.24, source_parent_preference="A")
    verse = _SectionSpec(label="verse", start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference="A", transition_in="blend")
    build = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference="B", transition_in="blend", transition_out="swap")
    payoff = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")
    outro = _SectionSpec(label="outro", start_bar=40, bar_count=8, target_energy=0.34, source_parent_preference="A", transition_in="blend")

    song_a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.22)
    song_b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.21)
    a_candidate = _pick_candidate(song_a, target_position="late", bar_count=16, target_energy=0.86, role="payoff")
    b_candidate = _pick_candidate(song_b, target_position="late", bar_count=16, target_energy=0.86, role="payoff")

    chosen = [
        _WindowSelection(parent_id="A", song=song_a, candidate=a_candidate, blended_error=0.40, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.20, "transition_viability": 0.18, "role_prior": 0.14, "groove_continuity": 0.0, "listen_groove_confidence": 0.82}, section_label="intro"),
        _WindowSelection(parent_id="B", song=song_b, candidate=b_candidate, blended_error=0.52, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.24, "transition_viability": 0.20, "role_prior": 0.18, "groove_continuity": 0.0, "listen_groove_confidence": 0.82}, section_label="verse"),
        _WindowSelection(parent_id="B", song=song_b, candidate=b_candidate, blended_error=0.60, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.26, "transition_viability": 0.22, "role_prior": 0.20, "groove_continuity": 0.0, "listen_groove_confidence": 0.82}, section_label="build"),
        _WindowSelection(parent_id="B", song=song_b, candidate=b_candidate, blended_error=0.72, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.28, "transition_viability": 0.24, "role_prior": 0.22, "groove_continuity": 0.0, "listen_groove_confidence": 0.82}, section_label="payoff"),
        _WindowSelection(parent_id="A", song=song_a, candidate=a_candidate, blended_error=0.44, score_breakdown={"stretch_ratio": 1.0, "stretch_gate": 0.0, "seam_risk": 0.22, "transition_viability": 0.18, "role_prior": 0.16, "groove_continuity": 0.0, "listen_groove_confidence": 0.83}, section_label="outro"),
    ]
    ranked_choices = [
        [chosen[0]],
        [chosen[1], _WindowSelection(parent_id="A", song=song_a, candidate=a_candidate, blended_error=0.80, score_breakdown={"stretch_ratio": 1.01, "stretch_gate": 0.0, "seam_risk": 0.29, "transition_viability": 0.26, "role_prior": 0.24, "groove_continuity": 0.0, "listen_groove_confidence": 0.79}, section_label="verse")],
        [chosen[2], _WindowSelection(parent_id="A", song=song_a, candidate=a_candidate, blended_error=0.86, score_breakdown={"stretch_ratio": 1.02, "stretch_gate": 0.0, "seam_risk": 0.30, "transition_viability": 0.28, "role_prior": 0.24, "groove_continuity": 0.0, "listen_groove_confidence": 0.80}, section_label="build")],
        [chosen[3], _WindowSelection(parent_id="A", song=song_a, candidate=a_candidate, blended_error=0.90, score_breakdown={"stretch_ratio": 1.01, "stretch_gate": 0.0, "seam_risk": 0.32, "transition_viability": 0.26, "role_prior": 0.25, "groove_continuity": 0.0, "listen_groove_confidence": 0.79}, section_label="payoff")],
        [chosen[4]],
    ]

    updated, notes = _apply_section_level_authenticity_guard(
        [intro, verse, build, payoff, outro],
        chosen,
        ranked_choices,
    )

    major_updated = [selection for selection in updated if selection.section_label in {"verse", "build", "payoff", "bridge"}]
    assert {selection.parent_id for selection in updated} == {"A", "B"}
    assert {selection.parent_id for selection in major_updated} == {"A", "B"}
    assert updated[3].parent_id == "A"
    assert any("major-section collapse" in note for note in notes)


def test_major_section_balance_guard_switches_to_other_parent_before_full_major_monopoly():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 5, 0.23)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 5, 0.24)

    prior_major_1 = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.42, role="verse"),
        blended_error=0.0,
        score_breakdown={},
        section_label="verse",
    )
    prior_major_2 = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )

    chosen = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="late", bar_count=16, target_energy=0.84, role="payoff"),
        blended_error=1.10,
        score_breakdown={
            "stretch_gate": 0.0,
            "stretch_ratio": 1.02,
            "seam_risk": 0.28,
            "transition_viability": 0.22,
            "role_prior": 0.18,
            "listen_groove_confidence": 0.83,
            "groove_continuity": 0.0,
        },
        section_label="payoff",
    )
    alternate = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="late", bar_count=16, target_energy=0.82, role="payoff"),
        blended_error=1.48,
        score_breakdown={
            "stretch_gate": 0.0,
            "stretch_ratio": 1.04,
            "seam_risk": 0.34,
            "transition_viability": 0.30,
            "role_prior": 0.24,
            "listen_groove_confidence": 0.79,
            "groove_continuity": 0.0,
        },
        section_label="payoff",
    )

    picked, note = _choose_with_major_section_balance_guard(
        _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend"),
        [chosen, alternate],
        [prior_major_1, prior_major_2],
    )

    assert picked.parent_id == "A"
    assert note is not None
    assert "major-section monopoly" in note
    assert "stretch 1.04" in note
    assert "groove 0.79" in note



def test_major_section_balance_guard_does_not_force_second_parent_when_alternate_is_not_groove_safe():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 5, 0.23)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 5, 0.24)

    prior_major_1 = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.42, role="verse"),
        blended_error=0.0,
        score_breakdown={},
        section_label="verse",
    )
    prior_major_2 = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )

    chosen = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="late", bar_count=16, target_energy=0.84, role="payoff"),
        blended_error=1.10,
        score_breakdown={
            "stretch_gate": 0.0,
            "stretch_ratio": 1.02,
            "seam_risk": 0.28,
            "transition_viability": 0.22,
            "role_prior": 0.18,
            "listen_groove_confidence": 0.84,
            "groove_continuity": 0.0,
        },
        section_label="payoff",
    )
    groove_unsafe_alternate = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="late", bar_count=16, target_energy=0.82, role="payoff"),
        blended_error=1.30,
        score_breakdown={
            "stretch_gate": 0.0,
            "stretch_ratio": 1.05,
            "seam_risk": 0.32,
            "transition_viability": 0.28,
            "role_prior": 0.22,
            "listen_groove_confidence": 0.49,
            "groove_continuity": 0.0,
        },
        section_label="payoff",
    )

    picked, note = _choose_with_major_section_balance_guard(
        _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend"),
        [chosen, groove_unsafe_alternate],
        [prior_major_1, prior_major_2],
    )

    assert picked.parent_id == "B"
    assert note is None



def test_build_stub_arrangement_plan_marks_cross_parent_late_payoff_handoff_as_explicit_mode():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.24)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.22)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.80, "hook_strength": 0.66, "hook_repetition": 0.56}

    a.energy["beat_rms"] = [0.08, 0.10, 0.18, 0.22, 0.34, 0.38, 0.48, 0.54, 0.64, 0.70, 0.84, 0.88, 0.96, 1.00]
    b.energy["beat_rms"] = [0.09, 0.11, 0.16, 0.20, 0.30, 0.34, 0.42, 0.48, 0.60, 0.66, 0.80, 0.86, 0.94, 0.98]

    plan = build_stub_arrangement_plan(a, b).to_dict()

    sections = plan["sections"]
    for idx in range(1, len(sections)):
        prev = sections[idx - 1]
        cur = sections[idx]
        if prev["label"] == "payoff" and cur["label"] in {"bridge", "outro"} and prev["source_parent"] != cur["source_parent"]:
            assert cur["transition_mode"] == "arrival_handoff"
            break
    else:
        pytest.skip("fixture did not produce a cross-parent late payoff handoff")



def test_build_stub_arrangement_plan_keeps_late_major_sections_on_backbone_after_donor_cluster():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.24)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.22)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.80, "hook_strength": 0.66, "hook_repetition": 0.56}

    a.energy["beat_rms"] = [0.08, 0.10, 0.18, 0.22, 0.34, 0.38, 0.48, 0.54, 0.64, 0.70, 0.84, 0.88, 0.96, 1.00]
    b.energy["beat_rms"] = [0.09, 0.11, 0.16, 0.20, 0.30, 0.34, 0.42, 0.48, 0.60, 0.66, 0.80, 0.86, 0.94, 0.98]

    plan = build_stub_arrangement_plan(a, b).to_dict()

    build_section = next(section for section in plan["sections"] if section["label"] == "build")
    late_major_sections = [section for section in plan["sections"] if section["label"] in {"payoff", "bridge"}]

    assert build_section["source_parent"] == "B"
    assert len(late_major_sections) >= 1
    assert all(section["source_parent"] == "A" for section in late_major_sections)



def test_enumerate_section_choices_penalizes_rewinding_backward_in_same_parent_timeline():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.20)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]

    a.energy["beat_rms"] = [0.08, 0.10, 0.12, 0.14, 0.18, 0.20, 0.24, 0.28, 0.34, 0.38, 0.44, 0.50, 0.56, 0.62]
    b.energy["beat_rms"] = [0.06, 0.08, 0.12, 0.16, 0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.76]

    previous = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.55, role="build"),
        blended_error=0.0,
        score_breakdown={},
    )
    assert previous.candidate.label == "phrase_3_5"

    spec = _SectionSpec(label="payoff", start_bar=16, bar_count=8, target_energy=0.62, source_parent_preference=None, transition_in="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous, prior_selections=[previous])

    rewind_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_4")
    forward_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_6")

    assert rewind_b.score_breakdown["reuse_source_rewind"] > 0.0
    assert rewind_b.score_breakdown["selection_reuse"] > forward_b.score_breakdown["selection_reuse"]
    assert rewind_b.blended_error > forward_b.blended_error


def test_enumerate_section_choices_penalizes_overstretched_phrase_windows_before_resolver():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 4, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 4, 0.20)

    # Parent A has normal 8s phrases for an 8-bar target at 128 BPM (~7.5s).
    a.duration_seconds = 32.0
    a.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0]
    a.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0]
    a.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]
    a.energy["beat_rms"] = [0.10, 0.12, 0.22, 0.26, 0.46, 0.50, 0.74, 0.78]

    # Parent B exposes only 12s phrases. That makes any 8-bar pickup land at ~1.6x stretch,
    # which previously could still win on role/energy shape.
    b.duration_seconds = 48.0
    b.structure["phrase_boundaries_seconds"] = [0.0, 12.0, 24.0, 36.0, 48.0]
    b.structure["section_boundaries_seconds"] = [12.0, 24.0, 36.0]
    b.energy["beat_times"] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, 39.0, 45.0]
    b.energy["beat_rms"] = [0.08, 0.10, 0.24, 0.30, 0.54, 0.60, 0.92, 0.96]

    spec = _SectionSpec(label="build", start_bar=16, bar_count=4, target_energy=0.58, source_parent_preference=None, transition_in="blend", transition_out="swap")
    ranked = _enumerate_section_choices(spec, a, b, previous=None)

    best = ranked[0]
    overstretched_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_3")
    normal_a = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_2_3")

    assert overstretched_b.score_breakdown["stretch_ratio"] > 1.25
    assert overstretched_b.score_breakdown["stretch_gate"] > 0.0
    assert overstretched_b.score_breakdown["stretch_penalty"] > normal_a.score_breakdown["stretch_penalty"]
    assert normal_a.blended_error < overstretched_b.blended_error
    assert best.parent_id == "A"



def test_bridge_selection_prefers_real_reset_after_hot_payoff_over_staying_stuck_in_plateau():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.18, 0.24, 0.28, 0.34, 0.38, 0.44, 0.48, 0.40, 0.36, 0.52, 0.56, 0.26, 0.22]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.32, 0.36, 0.48, 0.52, 0.84, 0.88, 0.50, 0.46, 0.86, 0.90, 0.20, 0.18]

    previous = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="late", bar_count=16, target_energy=0.86, role="payoff"),
        blended_error=0.0,
        score_breakdown={},
    )

    spec = _SectionSpec(label="bridge", start_bar=40, bar_count=8, target_energy=0.52, source_parent_preference="A", transition_in="swap", transition_out="lift")
    ranked = _enumerate_section_choices(spec, a, b, previous, prior_selections=[previous])

    reset_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_6_8")
    hot_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_6")

    assert reset_b.score_breakdown["energy_arc"] < hot_b.score_breakdown["energy_arc"]
    assert reset_b.blended_error < hot_b.blended_error


def test_final_payoff_delivery_prefers_post_bridge_window_that_finishes_strong_without_rewinding():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.energy["derived"] = {
            "energy_confidence": 0.95,
            "payoff_windows": [{"start": 40.0, "end": 64.0, "score": 0.95}],
            "hook_windows": [{"start": 40.0, "end": 64.0, "score": 0.72}],
        }

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.18, 0.24, 0.28, 0.34, 0.38, 0.44, 0.48, 0.40, 0.36, 0.52, 0.56, 0.72, 0.76]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.32, 0.36, 0.48, 0.52, 0.84, 0.88, 0.48, 0.42, 0.94, 0.56, 0.74, 0.96]

    previous_bridge = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="late", bar_count=8, target_energy=0.52, role="bridge"),
        blended_error=0.0,
        score_breakdown={},
        section_label="bridge",
    )
    assert previous_bridge.candidate.label in {"phrase_4_6", "phrase_5_7"}

    spec = _SectionSpec(label="payoff", start_bar=48, bar_count=16, target_energy=0.90, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous_bridge, prior_selections=[previous_bridge])

    rewind_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_8")
    forward_a = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_4_8")

    assert rewind_b.score_breakdown["final_payoff_delivery"] > 0.0
    assert (
        rewind_b.score_breakdown["reuse_source_rewind"] > 0.0
        or rewind_b.score_breakdown["reuse_source_containment"] > 0.0
    )
    assert forward_a.score_breakdown["final_payoff_delivery"] < rewind_b.score_breakdown["final_payoff_delivery"]
    assert forward_a.score_breakdown["selection_reuse"] < rewind_b.score_breakdown["selection_reuse"]
    assert ranked[0].candidate.label == "phrase_4_8"


def test_intro_selection_penalizes_front_loaded_hotspot_window():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}

    a.energy["beat_rms"] = [0.12, 0.16, 0.70, 0.84, 0.28, 0.24, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54]
    a.energy["derived"] = {
        "energy_confidence": 0.95,
        "payoff_windows": [{"start": 8.0, "end": 24.0, "score": 0.96}],
        "hook_windows": [{"start": 8.0, "end": 24.0, "score": 0.88}],
        "payoff_strength": 0.96,
        "hook_strength": 0.88,
        "hook_repetition": 0.72,
    }
    b.energy["derived"] = {
        "energy_confidence": 0.85,
        "payoff_strength": 0.42,
        "hook_strength": 0.38,
        "hook_repetition": 0.30,
    }

    spec = _SectionSpec(label="intro", start_bar=0, bar_count=8, target_energy=0.22, source_parent_preference=None, transition_in=None, transition_out="lift")
    ranked = _enumerate_section_choices(spec, a, b, previous=None)

    hot_a = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_0_2")
    steadier_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_0_2")

    assert hot_a.score_breakdown["shape_intro_hotspot"] > 0.0
    assert hot_a.score_breakdown["section_shape"] > steadier_b.score_breakdown["section_shape"]
    assert hot_a.blended_error > steadier_b.blended_error
    assert ranked[0].parent_id == "B"


def test_intro_selection_penalizes_late_source_window_even_if_energy_shape_is_plausible():
    song = make_song("late_intro.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    song.duration_seconds = 48.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    song.energy["beat_rms"] = [0.14, 0.16, 0.18, 0.20, 0.24, 0.26, 0.28, 0.30, 0.20, 0.18, 0.16, 0.14]

    spec = _SectionSpec(label="intro", start_bar=0, bar_count=8, target_energy=0.22, source_parent_preference=None, transition_in=None, transition_out="lift")
    ranked = _enumerate_section_choices(spec, song, song, previous=None)

    early_pick = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_0_2")
    late_pick = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_4_6")

    assert late_pick.candidate.start > early_pick.candidate.start
    assert late_pick.score_breakdown["source_role_too_late_gap"] > 0.0
    assert late_pick.score_breakdown["source_role_position"] > early_pick.score_breakdown["source_role_position"]
    assert early_pick.blended_error < late_pick.blended_error


def test_intro_selection_prefers_headroom_over_pre_lit_mid_song_window_when_position_is_still_legal():
    song = make_song("headroom_intro.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    song.duration_seconds = 64.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
    song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
    song.energy["derived"] = {
        "energy_confidence": 0.95,
        "payoff_windows": [{"start": 28.0, "end": 48.0, "score": 0.94}],
        "hook_windows": [{"start": 28.0, "end": 48.0, "score": 0.88}],
        "payoff_strength": 0.92,
        "hook_strength": 0.84,
        "hook_repetition": 0.70,
    }
    song.energy["beat_rms"] = [
        0.08, 0.10,
        0.12, 0.14,
        0.18, 0.20,
        0.60, 0.72,
        0.68, 0.80,
        0.32, 0.36,
        0.28, 0.24,
        0.20, 0.18,
    ]

    spec = _SectionSpec(label="intro", start_bar=0, bar_count=8, target_energy=0.22, source_parent_preference=None, transition_in=None, transition_out="lift")
    ranked = _enumerate_section_choices(spec, song, song, previous=None)

    early_pick = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_0_2")
    mid_hot_pick = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_2_4")

    assert mid_hot_pick.score_breakdown["section_shape"] > early_pick.score_breakdown["section_shape"]
    assert mid_hot_pick.score_breakdown["role_prior"] > early_pick.score_breakdown["role_prior"]
    assert early_pick.blended_error < mid_hot_pick.blended_error


def test_outro_selection_penalizes_early_source_window_even_if_it_is_safe():
    song = make_song("outro.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    song.duration_seconds = 48.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
    song.energy["beat_rms"] = [0.18, 0.16, 0.26, 0.24, 0.34, 0.32, 0.42, 0.40, 0.30, 0.28, 0.20, 0.18]

    spec = _SectionSpec(label="outro", start_bar=40, bar_count=8, target_energy=0.30, source_parent_preference=None, transition_in="blend")
    ranked = _enumerate_section_choices(spec, song, song, previous=None)

    early_pick = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_0_2")
    late_pick = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_4_6")

    assert early_pick.score_breakdown["source_role_too_early_gap"] > 0.0
    assert early_pick.score_breakdown["source_role_position"] > late_pick.score_breakdown["source_role_position"]
    assert late_pick.blended_error < early_pick.blended_error


def test_intro_selection_hard_blocks_fake_intro_when_true_intro_option_exists():
    a = make_song("fake_intro_a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("fake_intro_b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]

    a.energy["beat_rms"] = [
        0.08, 0.10,
        0.20, 0.24,
        0.48, 0.52,
        0.54, 0.58,
        0.62, 0.66,
        0.70, 0.74,
    ]
    b.energy["beat_rms"] = [0.12, 0.14, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36]

    spec = _SectionSpec(label="intro", start_bar=0, bar_count=8, target_energy=0.24, source_parent_preference="A", transition_out="lift")
    ranked = _enumerate_section_choices(spec, a, b, previous=None, prior_selections=[], backbone_parent="A", donor_parent="B")

    assert any(item.parent_id == "A" and item.candidate.label == "phrase_0_2" for item in ranked)
    assert not any(item.parent_id == "A" and item.candidate.label == "phrase_2_4" for item in ranked)



def test_backbone_verse_selection_penalizes_opening_lane_jump_after_intro():
    a = make_song("opening_a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("opening_b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]

    a.energy["beat_rms"] = [
        0.10, 0.12,
        0.26, 0.30,
        0.34, 0.38,
        0.20, 0.24,
        0.42, 0.46,
        0.50, 0.54,
    ]
    b.energy["beat_rms"] = [0.12, 0.14, 0.18, 0.20, 0.24, 0.26, 0.30, 0.32, 0.36, 0.38, 0.42, 0.44]

    intro_previous = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=next(candidate for candidate in _phrase_window_candidates(a, 8) if candidate.label == "phrase_0_2"),
        blended_error=0.0,
        score_breakdown={},
        section_label="intro",
    )
    spec = _SectionSpec(label="verse", start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference="A", transition_in="blend", transition_out="lift")
    ranked = _enumerate_section_choices(spec, a, b, previous=intro_previous, prior_selections=[intro_previous], backbone_parent="A", donor_parent="B")

    near_pick = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_2_4")
    late_candidate = next(candidate for candidate in _phrase_window_candidates(a, 8) if candidate.label == "phrase_4_6")
    late_penalty, late_metrics = _selection_opening_continuity_penalty(spec, "A", late_candidate, [intro_previous], backbone_parent="A")

    assert late_metrics["opening_phrase_jump_gap"] > 0.0
    assert late_penalty > near_pick.score_breakdown["opening_continuity"]
    assert all(not (item.parent_id == "A" and item.candidate.label == "phrase_4_6") for item in ranked)


def test_backbone_verse_selection_joint_opening_lane_scoring_prefers_contiguous_intro_plus_verse():
    a = make_song("opening_joint_a.wav", 128.0, "A", "minor", "8A", 7, 0.20)
    b = make_song("opening_joint_b.wav", 128.0, "A", "minor", "8A", 7, 0.20)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]

    a.energy["beat_rms"] = [
        0.08, 0.10,
        0.24, 0.28,
        0.34, 0.38,
        0.12, 0.16,
        0.18, 0.22,
        0.46, 0.50,
        0.56, 0.60,
    ]
    b.energy["beat_rms"] = [0.12, 0.14, 0.18, 0.20, 0.22, 0.24, 0.28, 0.30, 0.34, 0.36, 0.40, 0.42, 0.46, 0.48]

    intro_previous = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=next(candidate for candidate in _phrase_window_candidates(a, 8) if candidate.label == "phrase_0_2"),
        blended_error=0.0,
        score_breakdown={},
        section_label="intro",
    )
    spec = _SectionSpec(label="verse", start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference="A", transition_in="blend", transition_out="lift")
    ranked = _enumerate_section_choices(spec, a, b, previous=intro_previous, prior_selections=[intro_previous], backbone_parent="A", donor_parent="B")

    near_pick = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_2_4")

    assert near_pick.score_breakdown["opening_joint_lane_gap"] == 0.0
    assert near_pick.score_breakdown["opening_joint_identity_gap"] >= 0.0
    assert not any(item.parent_id == "A" and item.candidate.label == "phrase_4_6" for item in ranked)



def test_backbone_verse_selection_hard_blocks_severe_opening_lane_jump_when_nearby_backbone_option_exists():
    a = make_song("opening_guard_a.wav", 128.0, "A", "minor", "8A", 7, 0.20)
    b = make_song("opening_guard_b.wav", 128.0, "A", "minor", "8A", 7, 0.20)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]

    a.energy["beat_rms"] = [
        0.10, 0.12,
        0.24, 0.28,
        0.30, 0.34,
        0.18, 0.22,
        0.26, 0.30,
        0.42, 0.46,
        0.50, 0.54,
    ]
    b.energy["beat_rms"] = [0.12, 0.14, 0.18, 0.20, 0.22, 0.24, 0.28, 0.30, 0.34, 0.36, 0.40, 0.42, 0.46, 0.48]

    intro_previous = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=next(candidate for candidate in _phrase_window_candidates(a, 8) if candidate.label == "phrase_0_2"),
        blended_error=0.0,
        score_breakdown={},
        section_label="intro",
    )
    spec = _SectionSpec(label="verse", start_bar=8, bar_count=8, target_energy=0.42, source_parent_preference="A", transition_in="blend", transition_out="lift")

    severe_candidate = next(candidate for candidate in _phrase_window_candidates(a, 8) if candidate.label == "phrase_5_7")
    penalty, metrics = _selection_opening_continuity_penalty(spec, "A", severe_candidate, [intro_previous], backbone_parent="A")
    assert metrics["opening_phrase_jump_gap"] > 0.50
    assert penalty > 0.50

    ranked = _enumerate_section_choices(spec, a, b, previous=intro_previous, prior_selections=[intro_previous], backbone_parent="A", donor_parent="B")

    assert any(item.parent_id == "A" and item.candidate.label == "phrase_2_4" for item in ranked)
    assert not any(item.parent_id == "A" and item.candidate.label == "phrase_5_7" for item in ranked)


def test_backbone_continuity_penalizes_forward_rewind_for_structural_backbone_lane():
    song = make_song("backbone_forward.wav", 128.0, "A", "minor", "8A", 7, 0.20)
    song.duration_seconds = 56.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
    song.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.28, 0.32, 0.40, 0.44, 0.52, 0.56, 0.64, 0.68, 0.76, 0.80]

    prior = [
        _WindowSelection(
            parent_id="A",
            song=song,
            candidate=next(candidate for candidate in _phrase_window_candidates(song, 8) if candidate.label == "phrase_2_4"),
            blended_error=0.0,
            score_breakdown={},
            section_label="verse",
        )
    ]
    rewind_candidate = next(candidate for candidate in _phrase_window_candidates(song, 8) if candidate.label == "phrase_1_3")
    forward_candidate = next(candidate for candidate in _phrase_window_candidates(song, 8) if candidate.label == "phrase_4_6")
    spec = _SectionSpec(label="bridge", start_bar=32, bar_count=8, target_energy=0.50, source_parent_preference="A")

    rewind_penalty, rewind_metrics = _selection_backbone_continuity_penalty(spec, "A", prior, backbone_parent="A", donor_parent="B", candidate=rewind_candidate)
    forward_penalty, forward_metrics = _selection_backbone_continuity_penalty(spec, "A", prior, backbone_parent="A", donor_parent="B", candidate=forward_candidate)

    assert rewind_metrics["forward_backbone_rewind"] > 0.0
    assert rewind_penalty > forward_penalty
    assert forward_metrics["forward_backbone_rewind"] == 0.0



def test_payoff_selection_penalizes_underhit_drop_even_when_seam_is_safe():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}

    a.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.28, 0.32, 0.38, 0.42, 0.46, 0.50, 0.56, 0.60, 0.66, 0.74, 0.86, 0.92]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.28, 0.32, 0.38, 0.42, 0.46, 0.50, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
    a.energy["derived"] = {
        "energy_confidence": 0.92,
        "payoff_windows": [{"start": 32.0, "end": 64.0, "score": 0.92}],
        "hook_windows": [{"start": 40.0, "end": 64.0, "score": 0.72}],
        "payoff_strength": 0.92,
        "hook_strength": 0.72,
        "hook_repetition": 0.58,
    }
    b.energy["derived"] = {
        "energy_confidence": 0.88,
        "payoff_windows": [{"start": 32.0, "end": 64.0, "score": 0.30}],
        "hook_windows": [{"start": 32.0, "end": 64.0, "score": 0.26}],
        "payoff_strength": 0.30,
        "hook_strength": 0.26,
        "hook_repetition": 0.24,
    }

    previous_build = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )

    spec = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous_build, prior_selections=[previous_build])

    strong_a = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_4_8")
    weak_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_8")

    assert weak_b.score_breakdown["shape_payoff_underhit"] > 0.0
    assert weak_b.score_breakdown["section_shape"] > strong_a.score_breakdown["section_shape"]
    assert weak_b.blended_error > strong_a.blended_error
    assert ranked[0].parent_id == "A"


def test_payoff_selection_hard_blocks_early_and_weak_candidate_when_late_payoff_exists():
    song = make_song("payoff_guard.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    song.duration_seconds = 64.0
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
    song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
    song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
    song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
    song.energy["beat_rms"] = [
        0.10, 0.12,
        0.72, 0.80,
        0.18, 0.22,
        0.30, 0.34,
        0.42, 0.48,
        0.54, 0.60,
        0.82, 0.90,
        0.90, 0.94,
    ]
    song.energy["derived"] = {
        "energy_confidence": 0.95,
        "payoff_windows": [{"start": 48.0, "end": 64.0, "score": 0.94}],
        "hook_windows": [{"start": 48.0, "end": 64.0, "score": 0.74}],
        "payoff_strength": 0.94,
        "hook_strength": 0.74,
        "hook_repetition": 0.60,
    }

    previous_build = _WindowSelection(
        parent_id="A",
        song=song,
        candidate=next(candidate for candidate in _phrase_window_candidates(song, 8) if candidate.label == "phrase_4_6"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )
    spec = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")

    ranked = _enumerate_section_choices(spec, song, song, previous_build, prior_selections=[previous_build])

    assert any(item.candidate.label == "phrase_4_8" for item in ranked)
    assert not any(item.candidate.label == "phrase_1_3" for item in ranked)




def test_payoff_selection_penalizes_same_parent_carryover_when_it_does_not_land_harder_than_build():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {
            "energy_confidence": 0.92,
            "payoff_strength": 0.82,
            "hook_strength": 0.68,
            "hook_repetition": 0.54,
            "payoff_windows": [{"start": 32.0, "end": 64.0, "score": 0.84}],
            "hook_windows": [{"start": 32.0, "end": 64.0, "score": 0.68}],
        }

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.26, 0.30, 0.38, 0.44, 0.54, 0.60, 0.68, 0.74, 0.82, 0.88, 0.94, 0.98]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.28, 0.32, 0.42, 0.48, 0.58, 0.64, 0.70, 0.76, 0.72, 0.78, 0.80, 0.84]

    previous_build = _WindowSelection(
        parent_id="B",
        song=b,
        candidate=_pick_candidate(b, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )
    assert previous_build.candidate.label == "phrase_3_5"

    spec = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous_build, prior_selections=[previous_build])

    carryover_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_8")
    cleaner_a = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_4_8")

    assert carryover_b.score_breakdown["final_payoff_delivery"] > 0.0
    assert carryover_b.blended_error > cleaner_a.blended_error
    assert ranked[0].parent_id == "A"


def test_payoff_selection_prefers_sustained_high_conviction_late_window_over_same_start_spiky_option():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {
            "energy_confidence": 0.95,
            "payoff_strength": 0.88,
            "hook_strength": 0.72,
            "hook_repetition": 0.60,
            "payoff_windows": [{"start": 32.0, "end": 64.0, "score": 0.94}],
            "hook_windows": [{"start": 32.0, "end": 64.0, "score": 0.76}],
        }

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.28, 0.34, 0.42, 0.48, 0.56, 0.62, 0.66, 0.70, 0.72, 0.96, 0.60, 0.54]
    b.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.28, 0.34, 0.42, 0.48, 0.58, 0.64, 0.72, 0.78, 0.84, 0.88, 0.92, 0.96]

    previous_build = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )

    spec = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous_build, prior_selections=[previous_build])

    spiky_a = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_4_8")
    sustained_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_8")

    assert spiky_a.score_breakdown["final_payoff_delivery"] > sustained_b.score_breakdown["final_payoff_delivery"]
    assert spiky_a.blended_error > sustained_b.blended_error
    assert ranked[0].parent_id == "B"
    assert ranked[0].candidate.label == "phrase_4_8"


def test_payoff_selection_prefers_stronger_build_to_payoff_contrast_over_flatter_payoff_followup():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {
            "energy_confidence": 0.94,
            "payoff_strength": 0.84,
            "hook_strength": 0.68,
            "hook_repetition": 0.56,
            "payoff_windows": [{"start": 32.0, "end": 64.0, "score": 0.90}],
            "hook_windows": [{"start": 32.0, "end": 64.0, "score": 0.72}],
        }

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.28, 0.34, 0.44, 0.50, 0.60, 0.66, 0.70, 0.74, 0.70, 0.72, 0.74, 0.76]
    b.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.28, 0.34, 0.44, 0.50, 0.60, 0.66, 0.70, 0.74, 0.82, 0.88, 0.92, 0.96]

    previous_build = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )

    spec = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous_build, prior_selections=[previous_build])

    flatter_a = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_4_8")
    stronger_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_8")

    assert flatter_a.score_breakdown["build_to_payoff_contrast"] > stronger_b.score_breakdown["build_to_payoff_contrast"]
    assert flatter_a.score_breakdown["contrast_tail_dominance_gap"] >= 0.0
    assert flatter_a.score_breakdown["contrast_payoff_conviction_gap"] >= 0.0
    assert stronger_b.blended_error < flatter_a.blended_error
    assert ranked[0].parent_id == "B"



def test_payoff_selection_penalizes_build_to_payoff_window_that_starts_too_early_for_late_arrival():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {
            "energy_confidence": 0.95,
            "payoff_strength": 0.86,
            "hook_strength": 0.70,
            "hook_repetition": 0.58,
            "payoff_windows": [{"start": 32.0, "end": 64.0, "score": 0.92}],
            "hook_windows": [{"start": 32.0, "end": 64.0, "score": 0.74}],
        }

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.26, 0.30, 0.40, 0.46, 0.56, 0.62, 0.72, 0.78, 0.86, 0.90, 0.94, 0.98]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.30, 0.34, 0.46, 0.52, 0.62, 0.68, 0.80, 0.86, 0.88, 0.92, 0.94, 0.98]

    previous_build = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.58, role="build"),
        blended_error=0.0,
        score_breakdown={},
        section_label="build",
    )

    spec = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous_build, prior_selections=[previous_build])

    early_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_3_7")
    late_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_8")

    assert early_b.score_breakdown["final_payoff_delivery"] > late_b.score_breakdown["final_payoff_delivery"]
    assert early_b.blended_error > late_b.blended_error
    assert ranked[0].candidate.label == "phrase_4_8"



def test_payoff_selection_penalizes_bridge_to_payoff_window_that_reaches_back_too_far_before_finale():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 8, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 8, 0.20)

    for song in (a, b):
        song.duration_seconds = 64.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {
            "energy_confidence": 0.95,
            "payoff_strength": 0.90,
            "hook_strength": 0.74,
            "hook_repetition": 0.60,
            "payoff_windows": [{"start": 40.0, "end": 64.0, "score": 0.95}],
            "hook_windows": [{"start": 40.0, "end": 64.0, "score": 0.76}],
        }

    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.18, 0.26, 0.30, 0.38, 0.44, 0.52, 0.58, 0.40, 0.34, 0.62, 0.70, 0.84, 0.92]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.20, 0.30, 0.34, 0.46, 0.52, 0.82, 0.88, 0.46, 0.40, 0.90, 0.94, 0.96, 1.00]

    previous_bridge = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="late", bar_count=8, target_energy=0.52, role="bridge"),
        blended_error=0.0,
        score_breakdown={},
        section_label="bridge",
    )

    spec = _SectionSpec(label="payoff", start_bar=48, bar_count=16, target_energy=0.90, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous_bridge, prior_selections=[previous_bridge])

    early_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_3_7")
    late_b = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_4_8")

    assert early_b.score_breakdown["final_payoff_delivery"] > late_b.score_breakdown["final_payoff_delivery"]
    assert early_b.blended_error > late_b.blended_error
    assert ranked[0].candidate.label == "phrase_4_8"



def test_planner_listen_feedback_reads_existing_analysis_signals():
    song = make_song("feedback.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    song.energy["derived"] = {
        "energy_confidence": 0.9,
        "payoff_strength": 0.8,
        "hook_strength": 0.6,
        "hook_repetition": 0.5,
    }
    song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]}

    feedback = _planner_listen_feedback(song)

    assert feedback.groove_confidence > 0.9
    assert feedback.energy_arc_strength > 0.7
    assert feedback.transition_readiness > 0.7
    assert feedback.payoff_readiness > 0.6



def test_enumerate_section_choices_penalizes_stretch_heavy_build_window_to_protect_groove():
    a = make_song("a.wav", 120.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 120.0, "A", "minor", "8A", 6, 0.20)

    a.duration_seconds = 63.0
    a.structure["phrase_boundaries_seconds"] = [0.0, 10.5, 21.0, 31.5, 42.0, 52.5, 63.0]
    a.structure["section_boundaries_seconds"] = [10.5, 21.0, 31.5, 42.0, 52.5]
    b.duration_seconds = 48.0
    b.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    b.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]

    a.energy["beat_times"] = [2.5, 7.5, 13.0, 18.0, 23.5, 28.5, 34.0, 39.0, 44.5, 49.5, 55.0, 60.0]
    b.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]

    # A has the flashier mid-build energy profile, but its phrase windows are ~21s for an 8-bar target
    # (~16s at 120 BPM), which caused the real run's groove-damaging stretch warnings.
    a.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.30, 0.34, 0.56, 0.60, 0.68, 0.72, 0.78, 0.82]
    b.energy["beat_rms"] = [0.08, 0.10, 0.16, 0.20, 0.28, 0.32, 0.46, 0.50, 0.58, 0.62, 0.68, 0.72]

    previous = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="early", bar_count=8, target_energy=0.25, role="intro"),
        blended_error=0.0,
        score_breakdown={},
    )
    spec = _SectionSpec(label="build", start_bar=8, bar_count=8, target_energy=0.58, source_parent_preference=None, transition_in="blend", transition_out="swap")
    ranked = _enumerate_section_choices(spec, a, b, previous)

    a_build = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_2_4")
    b_build = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_4")

    assert a_build.score_breakdown["stretch_ratio"] > 1.25
    assert a_build.score_breakdown["stretch_penalty"] > b_build.score_breakdown["stretch_penalty"]
    assert a_build.blended_error > b_build.blended_error
    assert ranked[0].parent_id == "B"



def test_enumerate_section_choices_penalizes_same_parent_streak_when_other_parent_has_better_groove_continuity():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 7, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 7, 0.20)

    for song in (a, b):
        song.duration_seconds = 56.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0]
        song.energy["derived"] = {"energy_confidence": 0.9, "payoff_strength": 0.7, "hook_strength": 0.6, "hook_repetition": 0.5}

    a.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.30, 0.34, 0.42, 0.46, 0.54, 0.58, 0.66, 0.70, 0.78, 0.82]
    b.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.28, 0.32, 0.40, 0.44, 0.52, 0.56, 0.64, 0.68, 0.76, 0.80]

    # A has visibly shakier beat spacing, so keeping the section streak on A should now pay a
    # groove-continuity cost once B is still plausible for the next blend handoff.
    a.metadata["tempo"] = {"beat_times": [0.0, 0.52, 0.97, 1.57, 2.01, 2.63, 3.05, 3.71]}
    b.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}

    prior_intro = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="early", bar_count=8, target_energy=0.24, role="intro"),
        blended_error=0.0,
        score_breakdown={},
    )
    prior_verse = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.40, role="verse"),
        blended_error=0.0,
        score_breakdown={},
    )

    spec = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference=None, transition_in="blend", transition_out="swap")
    ranked = _enumerate_section_choices(spec, a, b, previous=prior_verse, prior_selections=[prior_intro, prior_verse])

    best = ranked[0]
    best_a = next(item for item in ranked if item.parent_id == "A")
    best_b = next(item for item in ranked if item.parent_id == "B")

    assert best.parent_id == "B"
    assert best_a.score_breakdown["groove_continuity"] > 0.0
    assert best_a.score_breakdown["groove_same_parent_streak"] >= 2.0
    assert best_a.score_breakdown["groove_alternate_groove_edge"] > 0.0
    assert best_a.blended_error > best_b.blended_error



def test_enumerate_section_choices_penalizes_parent_with_weak_listen_feedback_for_payoff():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}

    # Same window-energy shape, but only A advertises strong evaluator-style readiness.
    a.energy["beat_rms"] = [0.10, 0.12, 0.18, 0.22, 0.30, 0.34, 0.42, 0.46, 0.60, 0.64, 0.92, 0.96]
    b.energy["beat_rms"] = list(a.energy["beat_rms"])
    a.energy["derived"] = {"energy_confidence": 0.92, "payoff_strength": 0.90, "hook_strength": 0.72, "hook_repetition": 0.68}
    b.energy["derived"] = {"energy_confidence": 0.18, "payoff_strength": 0.08, "hook_strength": 0.10, "hook_repetition": 0.05}

    spec = _SectionSpec(label="payoff", start_bar=24, bar_count=16, target_energy=0.86, source_parent_preference=None, transition_in="drop", transition_out="blend")
    ranked = _enumerate_section_choices(spec, a, b, previous=None)

    best = ranked[0]
    a_payoff = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_2_6")
    b_payoff = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_6")

    assert a_payoff.score_breakdown["listen_feedback"] < b_payoff.score_breakdown["listen_feedback"]
    assert a_payoff.blended_error < b_payoff.blended_error
    assert best.parent_id == "A"
    assert best.score_breakdown["listen_payoff_readiness"] > b_payoff.score_breakdown["listen_payoff_readiness"]


def test_build_selection_prefers_phrase_window_with_stronger_rhythmic_drive_and_stability_when_energy_shape_is_close():
    a = make_song("a.wav", 128.0, "A", "minor", "8A", 6, 0.20)
    b = make_song("b.wav", 128.0, "A", "minor", "8A", 6, 0.20)

    for song in (a, b):
        song.duration_seconds = 48.0
        song.structure["phrase_boundaries_seconds"] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
        song.structure["section_boundaries_seconds"] = [8.0, 16.0, 24.0, 32.0, 40.0]
        song.energy["beat_times"] = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0]
        song.metadata["tempo"] = {"beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
        song.energy["derived"] = {"energy_confidence": 0.88, "payoff_strength": 0.72, "hook_strength": 0.58, "hook_repetition": 0.46}

    a.energy["beat_rms"] = [0.08, 0.10, 0.18, 0.22, 0.30, 0.34, 0.44, 0.48, 0.58, 0.62, 0.70, 0.74]
    b.energy["beat_rms"] = list(a.energy["beat_rms"])
    # B phrase_2_4 is rhythmically stronger but still musically similar in energy shape.
    a.energy["onset_density"] = [0.10, 0.50, 0.14, 0.54, 0.18, 0.58, 0.22, 0.62, 0.20, 0.64, 0.18, 0.60]
    b.energy["onset_density"] = [0.12, 0.16, 0.22, 0.26, 0.30, 0.34, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64]

    previous = _WindowSelection(
        parent_id="A",
        song=a,
        candidate=_pick_candidate(a, target_position="mid", bar_count=8, target_energy=0.42, role="verse"),
        blended_error=0.0,
        score_breakdown={},
        section_label="verse",
    )
    spec = _SectionSpec(label="build", start_bar=16, bar_count=8, target_energy=0.58, source_parent_preference=None, transition_in="blend", transition_out="swap")
    ranked = _enumerate_section_choices(spec, a, b, previous=previous)

    a_build = next(item for item in ranked if item.parent_id == "A" and item.candidate.label == "phrase_2_4")
    b_build = next(item for item in ranked if item.parent_id == "B" and item.candidate.label == "phrase_2_4")

    assert a_build.score_breakdown["phrase_groove"] > b_build.score_breakdown["phrase_groove"]
    assert a_build.score_breakdown["phrase_groove_stability_gap"] > 0.0
    assert b_build.score_breakdown["phrase_groove"] < 0.01
    assert b_build.score_breakdown["phrase_groove_drive_gap"] < 0.01
    assert b_build.score_breakdown["phrase_groove_stability_gap"] < a_build.score_breakdown["phrase_groove_stability_gap"]
