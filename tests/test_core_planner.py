from src.core.analysis.models import SongDNA
from src.core.planner import build_compatibility_report, build_stub_arrangement_plan
from src.core.planner.arrangement import _SectionSpec, _WindowSelection, _enumerate_section_choices, _pick_candidate


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

    assert len(plan["sections"]) == 3
    assert plan["sections"][0]["label"] == "intro"
    assert plan["sections"][0]["source_section_label"].startswith("phrase_")
    assert plan["sections"][1]["source_section_label"].startswith("phrase_")
    assert plan["sections"][2]["source_section_label"].startswith("phrase_")
    assert "compatibility" in plan
    assert any("boundary confidence" in note for note in plan["planning_notes"])


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
    assert payoff["source_section_label"] == "phrase_2_6"


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


def test_build_plan_penalizes_energy_backslide_in_build_when_intro_is_already_hot():
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

    assert intro["source_parent"] == "B"
    assert intro["source_section_label"] == "phrase_2_4"
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
