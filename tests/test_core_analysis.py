from pathlib import Path

import numpy as np

from src.core.analysis import analyzer
from src.core.analysis.energy import _derive_phrase_signals, compute_energy_profile
from src.core.analysis.structure import _infer_section_role_hints, _merge_boundary_evidence, _select_phrase_boundaries


def test_analyze_audio_file_returns_song_dna(monkeypatch):
    monkeypatch.setattr(analyzer, "load_audio", lambda path: (np.zeros(44100), 44100))
    monkeypatch.setattr(analyzer, "duration_seconds", lambda audio, sr: 1.0)
    monkeypatch.setattr(analyzer, "detect_tempo", lambda audio, sr: {"bpm": 128.0, "confidence": 0.9, "beat_times": [], "method": "librosa"})
    monkeypatch.setattr(analyzer, "detect_key", lambda audio, sr: {"tonic": "A", "mode": "minor", "camelot": "8A", "confidence": 0.91, "method": "krumhansl_correlation", "chroma": []})
    monkeypatch.setattr(analyzer, "estimate_structure", lambda audio, sr: {"sections": [{"start": 0.0, "end": 1.0, "label": "section_0"}]})
    monkeypatch.setattr(analyzer, "compute_energy_profile", lambda audio, sr: {"summary": {"mean_rms": 0.1}})
    monkeypatch.setattr(analyzer, "analyze_musical_intelligence", lambda audio, sr: {"summary": {"melodic_identity_strength": 0.6, "rhythmic_confidence": 0.7}})

    result = analyzer.analyze_audio_file(Path("song.wav")).to_dict()

    assert result["sample_rate"] == 44100
    assert result["tempo_bpm"] == 128.0
    assert result["key"]["camelot"] == "8A"
    assert result["structure"]["sections"][0]["label"] == "section_0"
    assert result["musical_intelligence"]["summary"]["melodic_identity_strength"] == 0.6
    assert result["analysis_version"] == "0.1.0"


def test_compute_energy_profile_emits_bar_and_derived_signals():
    sr = 22050
    duration = 8.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    envelope = np.concatenate([
        np.linspace(0.15, 0.35, int(t.size * 0.25), endpoint=False),
        np.linspace(0.35, 0.85, int(t.size * 0.25), endpoint=False),
        np.linspace(0.20, 0.30, int(t.size * 0.25), endpoint=False),
        np.linspace(0.70, 0.95, t.size - int(t.size * 0.75), endpoint=False),
    ])
    audio = envelope * np.sin(2 * np.pi * 220 * t)

    profile = compute_energy_profile(audio.astype(float), sr)

    assert profile["bar_rms"]
    assert profile["bar_onset_density"]
    assert "derived" in profile
    assert profile["derived"]["energy_confidence"] >= 0.0
    assert "hook_strength" in profile["summary"]
    assert "build_strength" in profile["summary"]
    assert "ramp_consistency" in profile["summary"]
    assert "payoff_strength" in profile["summary"]
    assert "late_lift" in profile["summary"]
    assert "climax_strength" in profile["summary"]
    assert "plateau_strength" in profile["summary"]
    assert "hook_spend" in profile["summary"]


def test_compute_energy_profile_prefers_late_sustained_payoff_window_over_early_spike():
    bar_rms = np.asarray([0.08, 0.10, 0.88, 0.24, 0.22, 0.26, 0.74, 0.82], dtype=float)
    bar_onset = np.asarray([0.12, 0.14, 0.78, 0.20, 0.22, 0.24, 0.68, 0.72], dtype=float)
    bar_low = np.asarray([0.18, 0.20, 0.40, 0.22, 0.24, 0.28, 0.62, 0.68], dtype=float)
    bar_flatness = np.asarray([0.48, 0.44, 0.52, 0.46, 0.44, 0.42, 0.24, 0.20], dtype=float)
    bar_times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    assert derived["payoff_windows"]
    top = max(derived["payoff_windows"], key=lambda item: item["score"])
    assert top["start_bar"] == 4
    assert top["end_bar"] == 8
    assert derived["climax_windows"]
    top_climax = max(derived["climax_windows"], key=lambda item: item["score"])
    assert top_climax["start_bar"] == 4
    assert top_climax["end_bar"] == 8
    assert derived["late_lift"] > 0.2
    assert derived["plateau_strength"] > 0.5


def test_compute_energy_profile_flags_early_hook_spend_when_late_payoff_never_arrives():
    bar_rms = np.asarray([
        0.30, 0.34, 0.32, 0.35,
        0.31, 0.35, 0.33, 0.36,
        0.20, 0.22, 0.24, 0.23,
        0.18, 0.20, 0.21, 0.19,
    ], dtype=float)
    bar_onset = np.asarray([
        0.58, 0.60, 0.56, 0.59,
        0.57, 0.61, 0.58, 0.60,
        0.22, 0.24, 0.25, 0.23,
        0.18, 0.20, 0.19, 0.18,
    ], dtype=float)
    bar_low = np.asarray([
        0.42, 0.44, 0.40, 0.43,
        0.41, 0.45, 0.42, 0.44,
        0.22, 0.24, 0.23, 0.22,
        0.18, 0.19, 0.20, 0.18,
    ], dtype=float)
    bar_flatness = np.asarray([
        0.24, 0.22, 0.25, 0.23,
        0.24, 0.22, 0.23, 0.22,
        0.40, 0.42, 0.41, 0.43,
        0.46, 0.48, 0.47, 0.49,
    ], dtype=float)
    bar_times = np.arange(bar_rms.size, dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    assert derived["hook_windows"]
    assert derived["hook_spend"] > 0.2
    assert derived["early_hook_strength"] >= derived["late_hook_strength"]
    assert derived["late_payoff_strength"] < 0.35
    assert derived["late_lift"] == 0.0
    assert derived["climax_strength"] < derived["hook_strength"]


def test_compute_energy_profile_penalizes_repeated_early_hooks_even_when_one_late_hook_exists():
    bar_rms = np.asarray([
        0.30, 0.33, 0.31, 0.34,
        0.29, 0.32, 0.30, 0.33,
        0.18, 0.20, 0.19, 0.21,
        0.27, 0.30, 0.28, 0.31,
    ], dtype=float)
    bar_onset = np.asarray([
        0.58, 0.60, 0.57, 0.60,
        0.56, 0.59, 0.57, 0.59,
        0.20, 0.22, 0.21, 0.23,
        0.42, 0.45, 0.43, 0.46,
    ], dtype=float)
    bar_low = np.asarray([
        0.42, 0.44, 0.41, 0.43,
        0.40, 0.43, 0.41, 0.43,
        0.18, 0.20, 0.19, 0.21,
        0.31, 0.34, 0.32, 0.35,
    ], dtype=float)
    bar_flatness = np.asarray([
        0.24, 0.22, 0.24, 0.23,
        0.25, 0.23, 0.24, 0.23,
        0.44, 0.46, 0.45, 0.46,
        0.30, 0.28, 0.29, 0.28,
    ], dtype=float)
    bar_times = np.arange(bar_rms.size, dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    assert derived["hook_windows"]
    assert derived["late_hook_strength"] > 0.0
    assert derived["hook_spend"] > 0.3
    assert derived["early_hook_density"] >= 0.5
    assert derived["early_hook_strength"] > derived["late_payoff_strength"]


def test_compute_energy_profile_detects_real_build_without_confusing_flat_late_plateau_for_ramp():
    bar_times = np.arange(8, dtype=float)
    build_rms = np.asarray([0.10, 0.10, 0.10, 0.10, 0.20, 0.35, 0.55, 0.85], dtype=float)
    build_onset = np.asarray([0.10, 0.10, 0.10, 0.10, 0.18, 0.28, 0.45, 0.75], dtype=float)
    build_low = np.asarray([0.20, 0.20, 0.20, 0.20, 0.25, 0.35, 0.50, 0.70], dtype=float)
    build_flatness = np.asarray([0.50, 0.50, 0.50, 0.50, 0.35, 0.28, 0.22, 0.16], dtype=float)

    plateau_rms = np.asarray([0.10, 0.10, 0.10, 0.10, 0.82, 0.82, 0.82, 0.82], dtype=float)
    plateau_onset = np.asarray([0.10, 0.10, 0.10, 0.10, 0.58, 0.58, 0.58, 0.58], dtype=float)
    plateau_low = np.asarray([0.20, 0.20, 0.20, 0.20, 0.68, 0.68, 0.68, 0.68], dtype=float)
    plateau_flatness = np.asarray([0.50, 0.50, 0.50, 0.50, 0.16, 0.16, 0.16, 0.16], dtype=float)

    build = _derive_phrase_signals(build_rms, build_onset, build_low, build_flatness, bar_times)
    plateau = _derive_phrase_signals(plateau_rms, plateau_onset, plateau_low, plateau_flatness, bar_times)

    assert build["build_strength"] > plateau["build_strength"] + 0.20
    assert build["build_windows"]
    top_build = max(build["build_windows"], key=lambda item: item["score"])
    assert top_build["start_bar"] == 4
    assert top_build["end_bar"] == 8
    assert build["ramp_consistency"] >= 0.95
    assert plateau["plateau_strength"] > build["plateau_strength"]
    assert not plateau["build_windows"]


def test_compute_energy_profile_detects_late_plateau_and_climax_windows():
    bar_rms = np.asarray([
        0.10, 0.12, 0.16, 0.20,
        0.24, 0.30, 0.40, 0.52,
        0.66, 0.72, 0.74, 0.75,
        0.73, 0.74, 0.75, 0.74,
    ], dtype=float)
    bar_onset = np.asarray([
        0.12, 0.14, 0.16, 0.18,
        0.24, 0.28, 0.36, 0.44,
        0.58, 0.62, 0.60, 0.58,
        0.54, 0.52, 0.50, 0.48,
    ], dtype=float)
    bar_low = np.asarray([
        0.18, 0.18, 0.20, 0.22,
        0.26, 0.30, 0.36, 0.44,
        0.56, 0.60, 0.62, 0.64,
        0.63, 0.64, 0.64, 0.63,
    ], dtype=float)
    bar_flatness = np.asarray([
        0.52, 0.50, 0.48, 0.46,
        0.40, 0.36, 0.30, 0.24,
        0.18, 0.16, 0.14, 0.14,
        0.15, 0.15, 0.14, 0.15,
    ], dtype=float)
    bar_times = np.arange(bar_rms.size, dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    assert derived["plateau_windows"]
    assert derived["climax_windows"]
    top_plateau = max(derived["plateau_windows"], key=lambda item: item["score"])
    top_climax = max(derived["climax_windows"], key=lambda item: item["score"])
    assert top_plateau["start_bar"] >= 8
    assert top_climax["start_bar"] >= 8
    assert derived["late_lift"] > 0.4
    assert derived["climax_strength"] >= derived["payoff_strength"] - 0.1


def test_compute_energy_profile_emits_merged_late_payoff_plateau_windows_for_sustained_climax():
    bar_rms = np.asarray([
        0.10, 0.12, 0.16, 0.20,
        0.24, 0.30, 0.38, 0.48,
        0.64, 0.70, 0.74, 0.76,
        0.72, 0.74, 0.76, 0.75,
        0.74, 0.75, 0.76, 0.74,
    ], dtype=float)
    bar_onset = np.asarray([
        0.12, 0.14, 0.16, 0.18,
        0.22, 0.26, 0.34, 0.42,
        0.56, 0.60, 0.58, 0.56,
        0.52, 0.50, 0.48, 0.47,
        0.46, 0.45, 0.44, 0.43,
    ], dtype=float)
    bar_low = np.asarray([
        0.18, 0.18, 0.20, 0.22,
        0.24, 0.28, 0.34, 0.42,
        0.54, 0.60, 0.62, 0.63,
        0.62, 0.63, 0.63, 0.62,
        0.62, 0.63, 0.63, 0.62,
    ], dtype=float)
    bar_flatness = np.asarray([
        0.54, 0.52, 0.50, 0.48,
        0.42, 0.38, 0.30, 0.24,
        0.18, 0.16, 0.14, 0.14,
        0.15, 0.15, 0.14, 0.15,
        0.15, 0.15, 0.14, 0.15,
    ], dtype=float)
    bar_times = np.arange(bar_rms.size, dtype=float)

    derived = _derive_phrase_signals(bar_rms, bar_onset, bar_low, bar_flatness, bar_times)

    merged_payoff = [item for item in derived["payoff_windows"] if item["end_bar"] - item["start_bar"] >= 8]
    merged_climax = [item for item in derived["climax_windows"] if item["end_bar"] - item["start_bar"] >= 8]
    merged_plateau = [item for item in derived["plateau_windows"] if item["end_bar"] - item["start_bar"] >= 8]

    assert merged_payoff
    assert merged_climax
    assert merged_plateau
    assert any(item["start_bar"] >= 8 and item["span_phrases"] >= 2 for item in merged_payoff)
    assert max(item["score"] for item in merged_climax) >= derived["climax_strength"] - 0.05


def test_structure_role_hints_promote_repeated_high_energy_sections_to_chorus_like():
    section_features = [
        {
            "mean_rms": 0.12,
            "mean_flatness": 0.58,
            "chroma": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "mfcc": np.asarray([0.1, -0.1, 0.0, 0.0]),
        },
        {
            "mean_rms": 0.82,
            "mean_flatness": 0.16,
            "chroma": np.asarray([0.0, 1.0, 0.0, 0.0]),
            "mfcc": np.asarray([0.8, 0.2, -0.1, 0.0]),
        },
        {
            "mean_rms": 0.32,
            "mean_flatness": 0.42,
            "chroma": np.asarray([0.0, 0.0, 1.0, 0.0]),
            "mfcc": np.asarray([-0.4, 0.2, 0.1, 0.0]),
        },
        {
            "mean_rms": 0.78,
            "mean_flatness": 0.18,
            "chroma": np.asarray([0.0, 1.0, 0.0, 0.0]),
            "mfcc": np.asarray([0.79, 0.19, -0.08, 0.02]),
        },
    ]

    hints = _infer_section_role_hints(section_features)

    chorus_like = [idx for idx, hint in enumerate(hints) if hint["role_hint"] == "chorus_like"]
    assert chorus_like == [1, 3]
    assert hints[0]["role_hint"] == "intro_like"
    assert hints[1]["chorus_likelihood"] >= hints[2]["chorus_likelihood"]
    assert hints[3]["repetition"] >= 0.9


def test_structure_role_hints_avoid_false_chorus_when_similarity_is_only_adjacent():
    section_features = [
        {
            "mean_rms": 0.22,
            "mean_flatness": 0.36,
            "chroma": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "mfcc": np.asarray([0.30, 0.10, 0.0, 0.0]),
        },
        {
            "mean_rms": 0.28,
            "mean_flatness": 0.34,
            "chroma": np.asarray([0.98, 0.02, 0.0, 0.0]),
            "mfcc": np.asarray([0.29, 0.11, 0.0, 0.0]),
        },
        {
            "mean_rms": 0.35,
            "mean_flatness": 0.32,
            "chroma": np.asarray([0.0, 1.0, 0.0, 0.0]),
            "mfcc": np.asarray([-0.2, 0.3, 0.1, 0.0]),
        },
    ]

    hints = _infer_section_role_hints(section_features)

    assert not any(hint["is_chorus_candidate"] for hint in hints)
    assert hints[0]["adjacent_repeat_penalty"] > 0.0
    assert hints[1]["adjacent_repeat_penalty"] > 0.0


def test_select_phrase_boundaries_falls_back_to_coarse_regularized_grid_when_beat_grid_is_weak():
    beat_times = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=float)

    phrase_boundaries, method = _select_phrase_boundaries(
        beat_times=beat_times,
        duration=32.0,
        beats_per_phrase=16,
        tempo_bpm=120.0,
        beat_grid_confidence=0.18,
    )

    assert method == "coarse_regularized_grid"
    assert phrase_boundaries[0] == 0.0
    assert phrase_boundaries[1:4] == [8.0, 16.0, 24.0]


def test_merge_boundary_evidence_combines_phrase_and_novelty_support_into_confidence():
    merged = _merge_boundary_evidence(
        duration=32.0,
        phrase_boundaries=[0.0, 8.0, 16.0, 24.0],
        tempogram_boundaries=[7.85, 16.25],
        self_similarity_boundaries=[8.1, 23.9],
        beat_grid_confidence=0.82,
        phrase_boundary_method="beat_phrase_grid",
    )

    assert merged
    first = min(merged, key=lambda item: abs(float(item["time"]) - 8.0))
    assert abs(float(first["time"]) - 8.0) < 0.2
    assert float(first["confidence"]) > 0.6
    assert "phrase_grid" in first["sources"]
    assert any(source in first["sources"] for source in ("tempogram_novelty", "self_similarity"))
