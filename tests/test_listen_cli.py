from __future__ import annotations

import json
from pathlib import Path

import ai_dj
from src.core.evaluation.listen import evaluate_song


class DummySong:
    def __init__(self, path: str):
        self.source_path = path
        self.duration_seconds = 120.0
        self.tempo_bpm = 120.0
        self.key = {"tonic": "C", "mode": "major"}
        self.structure = {
            "sections": [
                {"label": "section_0", "start": 0.0, "end": 20.0},
                {"label": "section_1", "start": 20.0, "end": 40.0},
                {"label": "section_2", "start": 40.0, "end": 60.0},
                {"label": "section_3", "start": 60.0, "end": 80.0},
                {"label": "section_4", "start": 80.0, "end": 100.0},
                {"label": "section_5", "start": 100.0, "end": 120.0},
            ],
            "phrase_boundaries_seconds": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "novelty_boundaries_seconds": [18, 38, 58, 78, 98],
        }
        self.energy = {
            "rms": [0.04, 0.05, 0.05, 0.06, 0.08, 0.09, 0.11, 0.12, 0.12, 0.13, 0.14, 0.15],
            "spectral_centroid": [1600, 1700, 1800, 1900, 2000, 2200, 2400, 2300, 2100, 2000, 2100, 2200],
            "spectral_rolloff": [3200, 3400, 3500, 3600, 3800, 4100, 4300, 4200, 4000, 3900, 3950, 4050],
            "onset_density": [0.20, 0.22, 0.21, 0.24, 0.27, 0.30, 0.34, 0.35, 0.33, 0.31, 0.30, 0.32],
            "low_band_ratio": [0.33, 0.34, 0.35, 0.35, 0.36, 0.38, 0.39, 0.39, 0.37, 0.36, 0.36, 0.37],
            "spectral_flatness": [0.12, 0.12, 0.13, 0.14, 0.15, 0.15, 0.16, 0.16, 0.15, 0.15, 0.14, 0.14],
            "bar_rms": [0.10, 0.11, 0.12, 0.13, 0.16, 0.18, 0.21, 0.24, 0.28, 0.31, 0.35, 0.38, 0.42, 0.44, 0.45, 0.44],
            "bar_onset_density": [0.15, 0.16, 0.16, 0.17, 0.20, 0.22, 0.24, 0.26, 0.28, 0.31, 0.34, 0.36, 0.40, 0.42, 0.42, 0.41],
            "bar_low_band_ratio": [0.25, 0.25, 0.26, 0.26, 0.28, 0.29, 0.30, 0.31, 0.33, 0.34, 0.35, 0.36, 0.38, 0.39, 0.39, 0.38],
            "bar_spectral_flatness": [0.22, 0.22, 0.21, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.14, 0.13, 0.13, 0.13, 0.14],
            "derived": {
                "payoff_strength": 0.82,
                "hook_strength": 0.70,
                "hook_repetition": 0.66,
                "energy_confidence": 0.88,
                "payoff_windows": [{"start_bar": 12, "end_bar": 16, "score": 0.82}],
                "hook_windows": [{"start_bar": 8, "end_bar": 12, "score": 0.70}],
            },
        }
        self.metadata = {
            "tempo": {
                "beat_times": [x * 0.5 for x in range(240)],
            }
        }


def test_listen_command_writes_json(monkeypatch, tmp_path: Path):
    audio = tmp_path / 'song.mp3'
    audio.write_bytes(b'fake')

    monkeypatch.setattr(ai_dj, 'analyze_audio_file', lambda path, stems_dir=None: DummySong(str(path)))

    out = tmp_path / 'listen.json'
    rc = ai_dj.listen(str(audio), str(out))
    assert rc == 0
    payload = out.read_text()
    assert 'overall_score' in payload
    assert 'structure' in payload
    assert 'coherence' in payload
    assert 'mix_sanity' in payload
    assert 'song_likeness' in payload
    assert 'gating' in payload


def test_groove_score_penalizes_bar_level_pocket_collapse_even_when_beat_grid_is_regular():
    stable_song = DummySong('stable_pocket.wav')
    collapse_song = DummySong('collapsed_pocket.wav')

    stable_song.metadata['tempo']['beat_times'] = [x * 0.5 for x in range(240)]
    collapse_song.metadata['tempo']['beat_times'] = [x * 0.5 for x in range(240)]

    stable_song.energy['bar_onset_density'] = [0.22, 0.23, 0.24, 0.24, 0.26, 0.27, 0.29, 0.30, 0.31, 0.32, 0.33, 0.33, 0.34, 0.35, 0.35, 0.34]
    stable_song.energy['bar_low_band_ratio'] = [0.30, 0.30, 0.31, 0.31, 0.32, 0.33, 0.33, 0.34, 0.35, 0.35, 0.36, 0.36, 0.37, 0.37, 0.37, 0.36]

    collapse_song.energy['bar_onset_density'] = [0.24, 0.25, 0.26, 0.26, 0.28, 0.30, 0.31, 0.32, 0.10, 0.09, 0.08, 0.09, 0.30, 0.31, 0.32, 0.33]
    collapse_song.energy['bar_low_band_ratio'] = [0.32, 0.32, 0.33, 0.33, 0.34, 0.35, 0.35, 0.36, 0.14, 0.13, 0.12, 0.13, 0.34, 0.35, 0.35, 0.36]

    stable_report = evaluate_song(stable_song)
    collapse_report = evaluate_song(collapse_song)

    assert stable_report.groove.score > collapse_report.groove.score
    assert stable_report.groove.details['collapse_severity'] < 0.2
    assert collapse_report.groove.details['collapse_severity'] > 0.5
    assert collapse_report.groove.details['pocket_stability'] < stable_report.groove.details['pocket_stability']
    assert 'pocket collapses abruptly' in ' '.join(collapse_report.groove.fixes).lower()



def test_groove_score_uses_beat_level_pulse_when_bar_means_hide_intra_bar_collapse():
    stable_song = DummySong('stable_beat_pocket.wav')
    collapse_song = DummySong('collapsed_beat_pocket.wav')

    stable_song.metadata['tempo']['beat_times'] = [x * 0.5 for x in range(64)]
    collapse_song.metadata['tempo']['beat_times'] = [x * 0.5 for x in range(64)]

    shared_bar_onset = [0.25] * 16
    shared_bar_low = [0.33] * 16
    stable_song.energy['bar_onset_density'] = shared_bar_onset[:]
    collapse_song.energy['bar_onset_density'] = shared_bar_onset[:]
    stable_song.energy['bar_low_band_ratio'] = shared_bar_low[:]
    collapse_song.energy['bar_low_band_ratio'] = shared_bar_low[:]

    stable_song.energy['beat_onset_density'] = [0.25] * 32
    stable_song.energy['beat_low_band_ratio'] = [0.33] * 32

    collapse_song.energy['beat_onset_density'] = [0.25] * 16 + [0.08] * 8 + [0.25] * 8
    collapse_song.energy['beat_low_band_ratio'] = [0.33] * 16 + [0.12] * 8 + [0.33] * 8

    stable_report = evaluate_song(stable_song)
    collapse_report = evaluate_song(collapse_song)

    assert stable_report.groove.score > collapse_report.groove.score
    assert stable_report.groove.details['bar_collapse_severity'] == collapse_report.groove.details['bar_collapse_severity']
    assert collapse_report.groove.details['beat_collapse_severity'] > stable_report.groove.details['beat_collapse_severity']
    assert collapse_report.groove.details['beat_pulse_stability'] < stable_report.groove.details['beat_pulse_stability']
    assert any('beat-pocket stability' in line for line in collapse_report.groove.evidence)



def test_groove_soft_gate_marks_borderline_pocket_collapse_for_review():
    song = DummySong('borderline_groove_gate.wav')
    song.energy['bar_onset_density'] = [0.24, 0.25, 0.26, 0.27, 0.27, 0.26, 0.25, 0.24, 0.17, 0.16, 0.17, 0.18, 0.23, 0.24, 0.25, 0.26]
    song.energy['bar_low_band_ratio'] = [0.33, 0.34, 0.35, 0.35, 0.34, 0.34, 0.33, 0.32, 0.22, 0.21, 0.22, 0.23, 0.31, 0.32, 0.33, 0.33]
    song.energy.pop('beat_onset_density', None)
    song.energy.pop('beat_low_band_ratio', None)

    report = evaluate_song(song)

    assert report.groove.score < 55.0
    assert report.gating['groove_floor_triggered'] is True
    assert report.gating['status'] == 'review'
    assert 'groove grid is not stable enough' in report.gating['soft_fail_reasons']
    assert report.overall_score <= 59.0



def test_energy_arc_prefers_late_sustained_payoff_over_flat_profile():
    strong_song = DummySong('strong.wav')
    flat_song = DummySong('flat.wav')
    flat_song.energy['bar_rms'] = [0.20] * 16
    flat_song.energy['bar_onset_density'] = [0.22] * 16
    flat_song.energy['bar_low_band_ratio'] = [0.30] * 16
    flat_song.energy['bar_spectral_flatness'] = [0.18] * 16
    flat_song.energy['derived'] = {
        'payoff_strength': 0.05,
        'hook_strength': 0.08,
        'hook_repetition': 0.10,
        'hook_spend': 0.0,
        'early_hook_strength': 0.0,
        'late_hook_strength': 0.0,
        'late_payoff_strength': 0.0,
        'energy_confidence': 0.85,
        'payoff_windows': [],
        'hook_windows': [],
    }

    strong_report = evaluate_song(strong_song)
    flat_report = evaluate_song(flat_song)

    assert strong_report.energy_arc.score > flat_report.energy_arc.score
    assert strong_report.energy_arc.score >= 75.0
    assert flat_report.energy_arc.score < 60.0
    assert strong_report.energy_arc.details['aggregate_metrics']['late_lift'] > 0.08
    assert flat_report.energy_arc.details['aggregate_metrics']['contrast'] < 0.05
    assert any('macro-dynamic contrast' in fix.lower() for fix in flat_report.energy_arc.fixes)


def test_energy_arc_flags_early_hook_spend_even_with_some_hook_repetition():
    strong_song = DummySong('strong.wav')
    early_spend_song = DummySong('early_spend.wav')
    early_spend_song.energy['bar_rms'] = [0.38, 0.40, 0.39, 0.41, 0.36, 0.37, 0.36, 0.38, 0.20, 0.22, 0.21, 0.23, 0.27, 0.29, 0.28, 0.30]
    early_spend_song.energy['bar_onset_density'] = [0.42, 0.43, 0.42, 0.44, 0.40, 0.41, 0.40, 0.42, 0.18, 0.19, 0.18, 0.20, 0.28, 0.30, 0.29, 0.31]
    early_spend_song.energy['bar_low_band_ratio'] = [0.34, 0.35, 0.34, 0.35, 0.33, 0.34, 0.33, 0.34, 0.20, 0.21, 0.20, 0.21, 0.26, 0.27, 0.26, 0.27]
    early_spend_song.energy['bar_spectral_flatness'] = [0.16, 0.16, 0.17, 0.16, 0.17, 0.17, 0.18, 0.17, 0.28, 0.29, 0.28, 0.29, 0.22, 0.22, 0.23, 0.22]
    early_spend_song.energy['derived'] = {
        'payoff_strength': 0.34,
        'hook_strength': 0.82,
        'hook_repetition': 0.78,
        'hook_spend': 0.61,
        'early_hook_strength': 0.82,
        'late_hook_strength': 0.46,
        'late_payoff_strength': 0.28,
        'energy_confidence': 0.88,
        'payoff_windows': [{'start_bar': 12, 'end_bar': 16, 'score': 0.28}],
        'hook_windows': [
            {'start_bar': 0, 'end_bar': 4, 'score': 0.82},
            {'start_bar': 4, 'end_bar': 8, 'score': 0.79},
            {'start_bar': 12, 'end_bar': 16, 'score': 0.46},
        ],
    }

    strong_report = evaluate_song(strong_song)
    early_spend_report = evaluate_song(early_spend_song)

    assert strong_report.energy_arc.score > early_spend_report.energy_arc.score
    assert early_spend_report.energy_arc.details['aggregate_metrics']['hook_spend'] > 0.5
    assert any('spend its hook too early' in fix.lower() for fix in early_spend_report.energy_arc.fixes)


def test_structure_score_rewards_section_span_quality_and_coverage():
    strong_song = DummySong('strong_structure.wav')
    weak_song = DummySong('weak_structure.wav')

    weak_song.structure['sections'] = [
        {'label': 'micro_0', 'start': 0.0, 'end': 4.0},
        {'label': 'micro_1', 'start': 4.0, 'end': 8.0},
        {'label': 'micro_2', 'start': 8.0, 'end': 12.0},
        {'label': 'micro_3', 'start': 12.0, 'end': 16.0},
        {'label': 'mega', 'start': 16.0, 'end': 96.0},
    ]
    weak_song.structure['phrase_boundaries_seconds'] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
    weak_song.structure['novelty_boundaries_seconds'] = [16, 56, 88]

    strong_report = evaluate_song(strong_song)
    weak_report = evaluate_song(weak_song)

    strong_metrics = strong_report.structure.details['aggregate_metrics']
    weak_metrics = weak_report.structure.details['aggregate_metrics']

    assert strong_report.structure.score > weak_report.structure.score
    assert strong_metrics['section_span_quality'] > weak_metrics['section_span_quality']
    assert strong_metrics['coverage_ratio'] > 0.95
    assert weak_metrics['largest_section_ratio'] > 0.6
    assert any('dominant mega-section' in fix.lower() for fix in weak_report.structure.fixes)


def test_structure_score_penalizes_sparse_section_coverage():
    covered_song = DummySong('covered.wav')
    sparse_song = DummySong('sparse.wav')

    sparse_song.structure['sections'] = [
        {'label': 'intro', 'start': 0.0, 'end': 12.0},
        {'label': 'hook', 'start': 54.0, 'end': 66.0},
        {'label': 'outro', 'start': 108.0, 'end': 120.0},
    ]
    sparse_song.structure['phrase_boundaries_seconds'] = covered_song.structure['phrase_boundaries_seconds']
    sparse_song.structure['novelty_boundaries_seconds'] = covered_song.structure['novelty_boundaries_seconds']

    covered_report = evaluate_song(covered_song)
    sparse_report = evaluate_song(sparse_song)

    covered_metrics = covered_report.structure.details['aggregate_metrics']
    sparse_metrics = sparse_report.structure.details['aggregate_metrics']

    assert covered_report.structure.score > sparse_report.structure.score
    assert sparse_metrics['coverage_ratio'] < 0.4
    assert covered_metrics['section_span_quality'] > sparse_metrics['section_span_quality']
    assert any('section coverage' in fix.lower() for fix in sparse_report.structure.fixes)


def test_transition_score_emits_boundary_seam_metrics():
    song = DummySong('dummy.wav')
    song.energy['rms'] = [0.04, 0.04, 0.04, 0.04, 0.16, 0.17, 0.17, 0.17, 0.07, 0.07, 0.07, 0.07]
    song.energy['spectral_centroid'] = [1500, 1520, 1490, 1510, 3600, 3650, 3700, 3600, 1700, 1680, 1710, 1690]
    song.energy['spectral_rolloff'] = [3200, 3250, 3180, 3220, 7600, 7700, 7800, 7650, 3600, 3550, 3650, 3620]
    song.energy['onset_density'] = [0.18, 0.19, 0.18, 0.18, 0.62, 0.65, 0.64, 0.63, 0.22, 0.21, 0.22, 0.21]
    song.energy['low_band_ratio'] = [0.32, 0.33, 0.33, 0.32, 0.72, 0.74, 0.73, 0.72, 0.48, 0.47, 0.48, 0.47]
    song.energy['spectral_flatness'] = [0.10, 0.10, 0.11, 0.10, 0.33, 0.34, 0.33, 0.34, 0.14, 0.14, 0.15, 0.14]

    report = evaluate_song(song)

    metrics = report.transition.details['aggregate_metrics']
    assert metrics['avg_energy_jump'] > 0.2
    assert metrics['avg_spectral_jump'] > 0.2
    assert metrics['avg_onset_jump'] > 0.2
    assert metrics['avg_low_end_crowding_risk'] > 0.3
    assert metrics['avg_foreground_collision_risk'] > 0.3
    assert metrics['avg_vocal_competition_risk'] > 0.3
    assert metrics['avg_texture_shift'] > 0.2
    assert report.transition.details['worst_boundaries']
    assert any('boundary' in line for line in report.transition.evidence)


def test_transition_score_uses_seam_local_windows_not_whole_sections():
    song = DummySong('seam_local.wav')
    song.structure['sections'] = [
        {'label': 'build', 'start': 0.0, 'end': 20.0, 'transition_out': 'lift'},
        {'label': 'payoff', 'start': 20.0, 'end': 40.0, 'transition_in': 'drop'},
        {'label': 'outro', 'start': 40.0, 'end': 60.0},
    ]
    song.energy['rms'] = [0.04, 0.04, 0.04, 0.10, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]
    song.energy['spectral_centroid'] = [1200, 1200, 1200, 2200, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250]
    song.energy['spectral_rolloff'] = [2600, 2600, 2600, 4800, 4850, 4850, 4850, 4850, 4850, 4850, 4850, 4850]
    song.energy['onset_density'] = [0.16, 0.16, 0.16, 0.28, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29]
    song.energy['low_band_ratio'] = [0.28, 0.28, 0.28, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34]
    song.energy['spectral_flatness'] = [0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]

    report = evaluate_song(song)
    metrics = report.transition.details['aggregate_metrics']

    assert metrics['seam_window_seconds'] <= 8.0
    assert metrics['avg_energy_jump'] < 0.2
    assert metrics['avg_intent_mismatch'] < 0.1
    assert report.transition.details['worst_boundaries'][0]['intent'] == 'drop'


def test_transition_score_respects_intended_lift_vs_accidental_drop():
    lift_song = DummySong('lift.wav')
    bad_song = DummySong('bad.wav')
    for current in (lift_song, bad_song):
        current.structure['sections'] = [
            {'label': 'intro', 'start': 0.0, 'end': 20.0, 'transition_out': 'lift'},
            {'label': 'build', 'start': 20.0, 'end': 40.0, 'transition_in': 'lift'},
            {'label': 'payoff', 'start': 40.0, 'end': 60.0, 'transition_in': 'drop'},
        ]

    lift_song.energy['rms'] = [0.06, 0.06, 0.06, 0.06, 0.10, 0.12, 0.12, 0.12, 0.16, 0.18, 0.18, 0.18]
    lift_song.energy['spectral_centroid'] = [1500, 1500, 1520, 1520, 1900, 2200, 2200, 2200, 2600, 2900, 2900, 2900]
    lift_song.energy['spectral_rolloff'] = [3000, 3000, 3050, 3050, 3800, 4500, 4500, 4500, 5400, 6200, 6200, 6200]
    lift_song.energy['onset_density'] = [0.18, 0.18, 0.18, 0.18, 0.26, 0.31, 0.31, 0.31, 0.40, 0.46, 0.46, 0.46]
    lift_song.energy['low_band_ratio'] = [0.30, 0.30, 0.30, 0.30, 0.34, 0.36, 0.36, 0.36, 0.40, 0.43, 0.43, 0.43]
    lift_song.energy['spectral_flatness'] = [0.14, 0.14, 0.14, 0.14, 0.13, 0.12, 0.12, 0.12, 0.11, 0.10, 0.10, 0.10]

    bad_song.energy['rms'] = [0.06, 0.06, 0.06, 0.06, 0.03, 0.02, 0.02, 0.02, 0.16, 0.18, 0.18, 0.18]
    bad_song.energy['spectral_centroid'] = [1500, 1500, 1520, 1520, 1100, 1000, 1000, 1000, 2600, 2900, 2900, 2900]
    bad_song.energy['spectral_rolloff'] = [3000, 3000, 3050, 3050, 2100, 1800, 1800, 1800, 5400, 6200, 6200, 6200]
    bad_song.energy['onset_density'] = [0.18, 0.18, 0.18, 0.18, 0.08, 0.06, 0.06, 0.06, 0.40, 0.46, 0.46, 0.46]
    bad_song.energy['low_band_ratio'] = [0.30, 0.30, 0.30, 0.30, 0.22, 0.20, 0.20, 0.20, 0.40, 0.43, 0.43, 0.43]
    bad_song.energy['spectral_flatness'] = [0.14, 0.14, 0.14, 0.14, 0.18, 0.19, 0.19, 0.19, 0.11, 0.10, 0.10, 0.10]

    lift_report = evaluate_song(lift_song)
    bad_report = evaluate_song(bad_song)

    assert lift_report.transition.score > bad_report.transition.score
    assert lift_report.transition.details['aggregate_metrics']['avg_intent_mismatch'] < bad_report.transition.details['aggregate_metrics']['avg_intent_mismatch']
    assert any('intent mismatch' in line.lower() or '(lift)' in line.lower() for line in lift_report.transition.evidence + lift_report.transition.details['transition_diagnostics'])


def test_transition_score_catches_boundary_edge_cliffs_even_when_window_means_match():
    smooth_song = DummySong('smooth_edge.wav')
    cliff_song = DummySong('cliff_edge.wav')
    for current in (smooth_song, cliff_song):
        current.duration_seconds = 60.0
        current.metadata['tempo']['beat_times'] = [x * 0.5 for x in range(120)]
        current.structure['sections'] = [
            {'label': 'build', 'start': 0.0, 'end': 20.0, 'transition_out': 'lift'},
            {'label': 'payoff', 'start': 20.0, 'end': 40.0, 'transition_in': 'drop'},
            {'label': 'outro', 'start': 40.0, 'end': 60.0},
        ]

    base_rms = [0.12] * 60
    base_onset = [0.24] * 60
    base_centroid = [2200.0] * 60
    base_rolloff = [4800.0] * 60
    base_low = [0.34] * 60
    base_flat = [0.12] * 60

    smooth_song.energy['rms'] = list(base_rms)
    smooth_song.energy['onset_density'] = list(base_onset)
    smooth_song.energy['spectral_centroid'] = list(base_centroid)
    smooth_song.energy['spectral_rolloff'] = list(base_rolloff)
    smooth_song.energy['low_band_ratio'] = list(base_low)
    smooth_song.energy['spectral_flatness'] = list(base_flat)

    cliff_song.energy['rms'] = list(base_rms)
    cliff_song.energy['onset_density'] = list(base_onset)
    cliff_song.energy['spectral_centroid'] = list(base_centroid)
    cliff_song.energy['spectral_rolloff'] = list(base_rolloff)
    cliff_song.energy['low_band_ratio'] = list(base_low)
    cliff_song.energy['spectral_flatness'] = list(base_flat)

    # Around the 20s seam, keep the 4s window means roughly matched but force a sharp edge cliff exactly at the handoff.
    for idx, value in zip([16, 17, 18, 19, 20, 21, 22, 23], [0.30, 0.30, 0.30, 0.10, 0.50, 0.30, 0.10, 0.10]):
        cliff_song.energy['rms'][idx] = value
    for idx, value in zip([16, 17, 18, 19, 20, 21, 22, 23], [0.40, 0.40, 0.40, 0.14, 0.62, 0.40, 0.18, 0.18]):
        cliff_song.energy['onset_density'][idx] = value
    for idx, value in zip([16, 17, 18, 19, 20, 21, 22, 23], [2200, 2200, 2200, 1500, 3600, 2200, 1800, 1800]):
        cliff_song.energy['spectral_centroid'][idx] = value
    for idx, value in zip([16, 17, 18, 19, 20, 21, 22, 23], [4800, 4800, 4800, 3100, 7000, 4800, 3600, 3600]):
        cliff_song.energy['spectral_rolloff'][idx] = value

    smooth_report = evaluate_song(smooth_song)
    cliff_report = evaluate_song(cliff_song)

    smooth_metrics = smooth_report.transition.details['aggregate_metrics']
    cliff_metrics = cliff_report.transition.details['aggregate_metrics']

    assert cliff_metrics['avg_energy_jump'] < 0.05
    assert cliff_metrics['avg_edge_cliff_risk'] > 0.35
    assert cliff_metrics['avg_edge_energy_jump'] > cliff_metrics['avg_energy_jump']
    assert cliff_report.transition.details['worst_boundaries'][0]['edge_cliff_risk'] > 0.75
    assert cliff_report.transition.score < smooth_report.transition.score
    assert any('edge_cliff' in line for line in cliff_report.transition.details['transition_diagnostics'])


def test_mix_sanity_emits_ownership_clutter_penalties(tmp_path: Path):
    audio = tmp_path / 'dense.wav'
    audio.write_bytes(b'fake')
    song = DummySong(str(audio))
    song.energy['rms'] = [0.16] * 12
    song.energy['spectral_centroid'] = [3400] * 12
    song.energy['spectral_rolloff'] = [7200] * 12
    song.energy['onset_density'] = [0.58] * 12
    song.energy['low_band_ratio'] = [0.64] * 12
    song.energy['spectral_flatness'] = [0.36] * 12

    report = evaluate_song(song)
    clutter = report.mix_sanity.details['ownership_clutter_metrics']

    assert clutter['overcrowded_overlap_risk'] > 0.5
    assert clutter['low_end_conflict_risk'] > 0.5
    assert clutter['foreground_overload_risk'] > 0.5
    assert clutter['overcompressed_flatness_risk'] > 0.5
    assert clutter['vocal_competition_risk'] > 0.5
    assert clutter['crowding_burst_risk'] > 0.5
    assert clutter['crowding_sustained_ratio'] > 0.5
    assert any('low-end owner' in fix.lower() or 'foreground owner' in fix.lower() or 'lead-vocal ownership' in fix.lower() or 'competing lead' in fix.lower() or 'crowding' in fix.lower() for fix in report.mix_sanity.fixes)


def test_mix_sanity_detects_time_local_crowding_bursts_even_when_global_means_stay_reasonable():
    controlled_song = DummySong('controlled_mix.wav')
    crowded_song = DummySong('bursty_mix.wav')

    controlled_song.energy['rms'] = [0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.09, 0.09]
    controlled_song.energy['spectral_centroid'] = [1700, 1750, 1800, 1850, 1900, 1950, 1900, 1850, 1800, 1750, 1700, 1680]
    controlled_song.energy['spectral_rolloff'] = [3200, 3300, 3350, 3400, 3500, 3550, 3500, 3450, 3380, 3320, 3280, 3250]
    controlled_song.energy['onset_density'] = [0.18, 0.19, 0.20, 0.20, 0.22, 0.22, 0.21, 0.20, 0.19, 0.19, 0.20, 0.20]
    controlled_song.energy['low_band_ratio'] = [0.28, 0.28, 0.29, 0.29, 0.30, 0.30, 0.29, 0.29, 0.28, 0.28, 0.29, 0.29]
    controlled_song.energy['spectral_flatness'] = [0.14, 0.14, 0.15, 0.15, 0.16, 0.16, 0.15, 0.15, 0.14, 0.14, 0.15, 0.15]

    crowded_song.energy['rms'] = [0.08, 0.08, 0.09, 0.18, 0.19, 0.18, 0.09, 0.09, 0.17, 0.18, 0.17, 0.09]
    crowded_song.energy['spectral_centroid'] = [1700, 1750, 1800, 3350, 3450, 3380, 1850, 1800, 3300, 3400, 3320, 1780]
    crowded_song.energy['spectral_rolloff'] = [3200, 3300, 3350, 7000, 7200, 7050, 3450, 3380, 6900, 7100, 6950, 3320]
    crowded_song.energy['onset_density'] = [0.18, 0.19, 0.20, 0.55, 0.58, 0.56, 0.20, 0.19, 0.53, 0.56, 0.54, 0.20]
    crowded_song.energy['low_band_ratio'] = [0.28, 0.28, 0.29, 0.62, 0.64, 0.63, 0.29, 0.28, 0.60, 0.62, 0.61, 0.29]
    crowded_song.energy['spectral_flatness'] = [0.14, 0.14, 0.15, 0.34, 0.35, 0.34, 0.15, 0.14, 0.33, 0.34, 0.33, 0.15]

    controlled_report = evaluate_song(controlled_song)
    crowded_report = evaluate_song(crowded_song)

    controlled_clutter = controlled_report.mix_sanity.details['ownership_clutter_metrics']
    crowded_clutter = crowded_report.mix_sanity.details['ownership_clutter_metrics']

    assert crowded_clutter['crowding_burst_count'] >= 2
    assert crowded_clutter['crowding_burst_risk'] > controlled_clutter['crowding_burst_risk']
    assert crowded_clutter['crowding_sustained_ratio'] > controlled_clutter['crowding_sustained_ratio']
    assert crowded_report.mix_sanity.score < controlled_report.mix_sanity.score
    assert any('crowding' in fix.lower() for fix in crowded_report.mix_sanity.fixes)


def test_compare_listen_reports_writes_delta_json(tmp_path: Path):
    left = {
        'source_path': 'left.wav',
        'duration_seconds': 60.0,
        'overall_score': 82.0,
        'structure': {'score': 85.0, 'summary': 'strong', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': 80.0, 'summary': 'stable', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': 79.0, 'summary': 'good', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': 81.0, 'summary': 'clean', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': 83.0, 'summary': 'cohesive', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 84.0, 'summary': 'clear', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': 83.0, 'summary': 'coherent child song', 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': 'promising',
        'top_reasons': [],
        'top_fixes': [],
        'gating': {'status': 'pass', 'raw_overall_score': 82.0},
        'analysis_version': '0.5.0',
    }
    right = {
        **left,
        'source_path': 'right.wav',
        'overall_score': 74.0,
        'energy_arc': {'score': 66.0, 'summary': 'flat', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 70.0, 'summary': 'crowded', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': 68.0, 'summary': 'stitched', 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': 'mixed',
    }

    left_path = tmp_path / 'left_listen.json'
    right_path = tmp_path / 'right_listen.json'
    out = tmp_path / 'compare.json'
    left_path.write_text(json.dumps(left), encoding='utf-8')
    right_path.write_text(json.dumps(right), encoding='utf-8')

    rc = ai_dj.compare_listen(str(left_path), str(right_path), str(out))
    assert rc == 0

    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['winner']['overall'] == 'left'
    assert payload['comparison_id'] == f"{payload['left']['case_id']}__vs__{payload['right']['case_id']}"
    assert payload['deltas']['overall_score_delta'] == 8.0
    assert payload['deltas']['component_score_deltas']['energy_arc'] == 13.0
    assert payload['winner']['components']['mix_sanity'] == 'left'
    assert payload['left']['report_origin'] == 'listen_report'
    assert payload['right']['report_origin'] == 'listen_report'
    assert payload['summary']
    assert payload['diagnostics']['ranked_component_swings'][0]['component'] == 'song_likeness'
    assert payload['diagnostics']['left_profile']['strengths'][0]['component'] == 'structure'
    assert payload['diagnostics']['right_profile']['weaknesses'][0]['component'] == 'energy_arc'
    assert payload['decision']['winner'] == 'left'
    assert payload['decision']['winner_label'] == 'left_listen.json'
    assert payload['decision']['loser_label'] == 'right_listen.json'
    assert payload['decision']['confidence'] in {'leaning', 'clear'}
    deciding_components = {item['component']: item for item in payload['decision']['deciding_components']}
    assert {'energy_arc', 'mix_sanity'}.issubset(deciding_components)
    assert deciding_components['energy_arc']['winner_summary'] == 'good'
    assert any('wins overall by 8.0 listen points' in line for line in payload['decision']['why'])


def test_compare_listen_decision_preserves_tie_tradeoffs(tmp_path: Path):
    left = {
        'source_path': 'left.wav',
        'duration_seconds': 60.0,
        'overall_score': 80.0,
        'structure': {'score': 84.0, 'summary': 'more segmented', 'evidence': ['clear section turns'], 'fixes': [], 'details': {}},
        'groove': {'score': 78.0, 'summary': 'slightly looser', 'evidence': [], 'fixes': ['tighten beat grid'], 'details': {}},
        'energy_arc': {'score': 79.0, 'summary': 'balanced', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': 80.0, 'summary': 'controlled', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': 80.0, 'summary': 'steady', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 79.0, 'summary': 'slightly dense', 'evidence': [], 'fixes': ['thin overlapping mids'], 'details': {}},
        'song_likeness': {'score': 80.0, 'summary': 'reads like one song', 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': 'promising',
        'top_reasons': ['Clearer structure helps the arrangement read faster.'],
        'top_fixes': ['Stabilize groove and reduce midrange crowding.'],
        'gating': {'status': 'pass', 'raw_overall_score': 80.0},
        'analysis_version': '0.5.0',
    }
    right = {
        **left,
        'source_path': 'right.wav',
        'structure': {'score': 78.0, 'summary': 'less segmented', 'evidence': [], 'fixes': ['sharpen section contrast'], 'details': {}},
        'groove': {'score': 84.0, 'summary': 'tighter pocket', 'evidence': ['beat grid is more stable'], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 81.0, 'summary': 'cleaner spacing', 'evidence': ['less foreground masking'], 'fixes': [], 'details': {}},
        'song_likeness': {'score': 80.0, 'summary': 'reads like one song', 'evidence': [], 'fixes': [], 'details': {}},
        'top_reasons': ['Groove pocket is tighter and mix spacing is cleaner.'],
        'top_fixes': ['Sharpen section contrast.'],
    }

    left_path = tmp_path / 'left_tie.json'
    right_path = tmp_path / 'right_tie.json'
    out = tmp_path / 'compare_tie.json'
    left_path.write_text(json.dumps(left), encoding='utf-8')
    right_path.write_text(json.dumps(right), encoding='utf-8')

    rc = ai_dj.compare_listen(str(left_path), str(right_path), str(out))
    assert rc == 0

    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['winner']['overall'] == 'tie'
    assert payload['decision']['winner'] == 'tie'
    assert payload['decision']['confidence'] == 'tie'
    assert payload['decision']['deciding_components'][0]['component'] == 'structure'
    assert any('Largest tradeoff' in line for line in payload['decision']['why'])


def test_compare_listen_disambiguates_duplicate_basenames_in_explanations(tmp_path: Path):
    left_report = {
        'source_path': 'left.wav',
        'duration_seconds': 60.0,
        'overall_score': 82.0,
        'structure': {'score': 85.0, 'summary': 'strong', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': 81.0, 'summary': 'stable', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': 80.0, 'summary': 'good rise', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': 79.0, 'summary': 'clean', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': 80.0, 'summary': 'cohesive', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 80.0, 'summary': 'clear', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': 81.0, 'summary': 'reads like one child song', 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': 'promising',
        'top_reasons': [],
        'top_fixes': [],
        'gating': {'status': 'pass', 'raw_overall_score': 82.0},
        'analysis_version': '0.5.0',
    }
    right_report = {
        **left_report,
        'source_path': 'right.wav',
        'overall_score': 74.0,
        'energy_arc': {'score': 69.0, 'summary': 'flat', 'evidence': [], 'fixes': ['build a later payoff'], 'details': {}},
        'mix_sanity': {'score': 72.0, 'summary': 'crowded', 'evidence': [], 'fixes': ['thin overlaps'], 'details': {}},
        'song_likeness': {'score': 70.0, 'summary': 'more stitched', 'evidence': [], 'fixes': ['stabilize the backbone'], 'details': {}},
        'verdict': 'mixed',
    }

    left_dir = tmp_path / 'pass_a'
    right_dir = tmp_path / 'pass_b'
    left_dir.mkdir()
    right_dir.mkdir()
    left_path = left_dir / 'fusion_rerun.json'
    right_path = right_dir / 'fusion_rerun.json'
    left_path.write_text(json.dumps(left_report), encoding='utf-8')
    right_path.write_text(json.dumps(right_report), encoding='utf-8')

    comparison = ai_dj._build_listen_comparison(str(left_path), str(right_path))

    assert comparison['left']['input_label'] == 'fusion_rerun.json'
    assert comparison['right']['input_label'] == 'fusion_rerun.json'
    assert comparison['left']['display_label'] == 'pass_a/fusion_rerun.json'
    assert comparison['right']['display_label'] == 'pass_b/fusion_rerun.json'
    assert comparison['decision']['winner_label'] == 'pass_a/fusion_rerun.json'
    assert comparison['decision']['loser_label'] == 'pass_b/fusion_rerun.json'
    assert any('pass_a/fusion_rerun.json wins overall by 8.0 listen points over pass_b/fusion_rerun.json.' == line for line in comparison['decision']['why'])


def test_compare_listen_respects_explicit_case_labels(monkeypatch, tmp_path: Path):
    left = tmp_path / 'left.wav'
    right = tmp_path / 'right.wav'
    left.write_bytes(b'fake')
    right.write_bytes(b'fake')

    monkeypatch.setattr(ai_dj, 'analyze_audio_file', lambda path, stems_dir=None: DummySong(str(path)))

    comparison = ai_dj._build_listen_comparison(f'baseline={left}', f'experimental={right}')

    assert comparison['left']['input_label'] == 'baseline'
    assert comparison['right']['input_label'] == 'experimental'
    assert comparison['left']['display_label'] == 'baseline'
    assert comparison['right']['display_label'] == 'experimental'
    assert comparison['left']['case_id'] != comparison['right']['case_id']


def test_compare_listen_can_resolve_render_output_directories(monkeypatch, tmp_path: Path):
    left_dir = tmp_path / 'render_a'
    right_dir = tmp_path / 'render_b'
    left_dir.mkdir()
    right_dir.mkdir()
    left_audio = left_dir / 'child_master.wav'
    right_audio = right_dir / 'child_master.wav'
    left_audio.write_bytes(b'fake')
    right_audio.write_bytes(b'fake')
    (left_dir / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(left_audio)}}), encoding='utf-8')
    (right_dir / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(right_audio)}}), encoding='utf-8')

    monkeypatch.setattr(ai_dj, 'analyze_audio_file', lambda path, stems_dir=None: DummySong(str(path)))

    out = tmp_path / 'render_compare.json'
    rc = ai_dj.compare_listen(str(left_dir), str(right_dir), str(out))
    assert rc == 0

    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['left']['report_origin'] == 'render_output'
    assert payload['right']['report_origin'] == 'render_output'
    assert payload['left']['resolved_audio_path'] == str(left_audio)
    assert payload['right']['resolved_audio_path'] == str(right_audio)
    assert payload['winner']['overall'] == 'tie'
    assert payload['left']['input_label'] == 'render_a'
    assert payload['right']['input_label'] == 'render_b'
    assert payload['summary'][0].startswith('Overall: tie')


def test_compare_listen_writes_stable_default_artifact(monkeypatch, tmp_path: Path):
    left = tmp_path / 'left.wav'
    right = tmp_path / 'right.wav'
    left.write_bytes(b'fake')
    right.write_bytes(b'fake')

    monkeypatch.setattr(ai_dj, 'analyze_audio_file', lambda path, stems_dir=None: DummySong(str(path)))

    rc = ai_dj.compare_listen(str(left), str(right), None)
    assert rc == 0

    out = ai_dj._stable_compare_output_path(str(left), str(right))
    assert out.exists()
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['left']['input_label'] == 'left.wav'
    assert payload['right']['input_label'] == 'right.wav'


def test_compare_listen_accepts_directory_output(monkeypatch, tmp_path: Path):
    left = tmp_path / 'left.wav'
    right = tmp_path / 'right.wav'
    left.write_bytes(b'fake')
    right.write_bytes(b'fake')

    monkeypatch.setattr(ai_dj, 'analyze_audio_file', lambda path, stems_dir=None: DummySong(str(path)))

    out_dir = tmp_path / 'compare_dir'
    out_dir.mkdir()
    rc = ai_dj.compare_listen(str(left), str(right), str(out_dir))
    assert rc == 0

    out = out_dir / 'listen_compare.json'
    assert out.exists()


def test_manifest_identity_metrics_distinguish_true_two_parent_ownership_from_background_only_presence(tmp_path: Path):
    run_dir = tmp_path / 'identity_run'
    run_dir.mkdir()
    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    manifest = run_dir / 'render_manifest.json'
    manifest.write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "intro", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0},
    {"index": 1, "label": "verse", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0},
    {"index": 2, "label": "build", "source_parent": "A", "allowed_overlap": false, "overlap_beats_max": 0.0, "foreground_owner": "A", "background_owner": null, "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 3, "label": "payoff", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0}
  ],
  "work_orders": []
}""" % audio.as_posix())

    song = DummySong(str(audio))
    report = evaluate_song(song)
    manifest_metrics = report.mix_sanity.details['manifest_metrics']
    aggregate = manifest_metrics['aggregate_metrics']
    identity = manifest_metrics['fusion_identity']

    assert aggregate['true_two_parent_section_ratio'] == 0.25
    assert aggregate['true_two_parent_major_section_ratio'] == 0.25
    assert aggregate['background_only_presence_ratio'] == 0.75
    assert aggregate['background_only_identity_gap'] == 0.5
    assert identity['section_primary_counts'] == {'A': 1, 'B': 3}
    assert identity['major_section_primary_counts'] == {'A': 1, 'B': 2}
    assert identity['background_only_presence_counts'] == {'A': 3, 'B': 0}
    assert identity['minority_parent'] == 'A'



def test_manifest_metrics_do_not_count_redundant_same_owner_background_as_multi_owner_conflict(tmp_path: Path):
    run_dir = tmp_path / 'redundant_owner_run'
    run_dir.mkdir()
    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    manifest = run_dir / 'render_manifest.json'
    manifest.write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "intro", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "A", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0}
  ],
  "work_orders": []
}""" % audio.as_posix())

    song = DummySong(str(audio))
    report = evaluate_song(song)
    metrics = report.mix_sanity.details['manifest_metrics']['aggregate_metrics']
    identity = report.mix_sanity.details['manifest_metrics']['fusion_identity']

    assert metrics['multi_owner_conflict_ratio'] == 0.0
    assert identity['background_only_presence_counts'] == {'A': 0, 'B': 0}



def test_song_likeness_gate_rejects_non_song_like_render(tmp_path: Path):
    run_dir = tmp_path / 'non_song_like'
    run_dir.mkdir()
    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    (run_dir / 'render_manifest.json').write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "section_0", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 6.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "both", "stretch_ratio": 1.25, "transition_out": "blend"},
    {"index": 1, "label": "section_1", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 6.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "both", "stretch_ratio": 1.22, "transition_in": "swap", "transition_out": "blend"},
    {"index": 2, "label": "section_2", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 5.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "both", "stretch_ratio": 1.18, "transition_in": "swap", "transition_out": "blend"},
    {"index": 3, "label": "section_3", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 5.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "both", "stretch_ratio": 1.16, "transition_in": "swap"}
  ],
  "work_orders": []
}""" % audio.as_posix(), encoding='utf-8')

    song = DummySong(str(audio))
    song.structure['sections'] = [
        {'label': 'section_0', 'start': 0.0, 'end': 30.0},
        {'label': 'section_1', 'start': 30.0, 'end': 60.0},
        {'label': 'section_2', 'start': 60.0, 'end': 90.0},
        {'label': 'section_3', 'start': 90.0, 'end': 120.0},
    ]
    song.structure['phrase_boundaries_seconds'] = [0, 30, 60, 90]
    song.structure['novelty_boundaries_seconds'] = [30, 60, 90]
    song.energy['rms'] = [0.18, 0.18, 0.18, 0.18, 0.06, 0.06, 0.06, 0.06, 0.17, 0.17, 0.17, 0.17]
    song.energy['spectral_centroid'] = [3400, 3400, 3400, 3400, 1200, 1200, 1200, 1200, 3300, 3300, 3300, 3300]
    song.energy['spectral_rolloff'] = [7600, 7600, 7600, 7600, 2400, 2400, 2400, 2400, 7200, 7200, 7200, 7200]
    song.energy['onset_density'] = [0.56, 0.56, 0.56, 0.56, 0.10, 0.10, 0.10, 0.10, 0.52, 0.52, 0.52, 0.52]
    song.energy['low_band_ratio'] = [0.66, 0.66, 0.66, 0.66, 0.24, 0.24, 0.24, 0.24, 0.62, 0.62, 0.62, 0.62]
    song.energy['spectral_flatness'] = [0.34, 0.34, 0.34, 0.34, 0.12, 0.12, 0.12, 0.12, 0.32, 0.32, 0.32, 0.32]

    report = evaluate_song(song)

    assert report.song_likeness.score < 45.0
    assert report.gating['status'] == 'reject'
    assert report.overall_score <= 49.0
    assert report.song_likeness.details['aggregate_metrics']['composite_song_risk'] > 0.30
    assert 'composite_song_risk' in report.gating
    assert report.gating['track_switch_seam_hard_reject_triggered'] is True
    assert 'obvious track-switch seams detected' in report.gating['hard_fail_reasons']
    assert any('backbone continuity' in fix.lower() or 'cluttered donor carryover' in fix.lower() for fix in report.top_fixes)



def test_song_likeness_hard_rejects_full_mix_section_switch_medley(tmp_path: Path):
    run_dir = tmp_path / 'medley_like'
    run_dir.mkdir()
    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    (run_dir / 'render_manifest.json').write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "verse", "source_parent": "A", "allowed_overlap": false, "overlap_beats_max": 0.0, "foreground_owner": "A", "background_owner": null, "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 1, "label": "build", "source_parent": "B", "allowed_overlap": false, "overlap_beats_max": 0.0, "foreground_owner": "B", "background_owner": null, "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0},
    {"index": 2, "label": "payoff", "source_parent": "A", "allowed_overlap": false, "overlap_beats_max": 0.0, "foreground_owner": "A", "background_owner": null, "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 3, "label": "bridge", "source_parent": "B", "allowed_overlap": false, "overlap_beats_max": 0.0, "foreground_owner": "B", "background_owner": null, "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0}
  ],
  "work_orders": [
    {"section_index": 0, "parent_id": "A", "role": "full_mix", "foreground_state": "owner", "low_end_state": "owner", "vocal_state": "lead_only"},
    {"section_index": 1, "parent_id": "B", "role": "full_mix", "foreground_state": "owner", "low_end_state": "owner", "vocal_state": "lead_only"},
    {"section_index": 2, "parent_id": "A", "role": "full_mix", "foreground_state": "owner", "low_end_state": "owner", "vocal_state": "lead_only"},
    {"section_index": 3, "parent_id": "B", "role": "full_mix", "foreground_state": "owner", "low_end_state": "owner", "vocal_state": "lead_only"}
  ]
}""" % audio.as_posix(), encoding='utf-8')

    song = DummySong(str(audio))
    song.structure['sections'] = [
        {'label': 'verse', 'start': 0.0, 'end': 24.0},
        {'label': 'build', 'start': 24.0, 'end': 48.0, 'transition_in': 'lift'},
        {'label': 'payoff', 'start': 48.0, 'end': 72.0, 'transition_in': 'drop'},
        {'label': 'bridge', 'start': 72.0, 'end': 96.0},
    ]
    song.structure['phrase_boundaries_seconds'] = [0, 24, 48, 72, 96]
    song.structure['novelty_boundaries_seconds'] = [24, 48, 72]

    report = evaluate_song(song)
    metrics = report.song_likeness.details['aggregate_metrics']

    assert metrics['integrated_two_parent_section_ratio'] == 0.0
    assert metrics['support_layer_section_ratio'] == 0.0
    assert metrics['full_mix_medley_risk'] > 0.65
    assert report.gating['status'] == 'reject'
    assert 'render is still mostly alternating full-mix parent sections instead of integrated child sections' in report.gating['hard_fail_reasons']
    assert any('full-mix swaps' in fix.lower() or 'medley-like' in fix.lower() for fix in report.top_fixes)



def test_song_likeness_composite_detector_rejects_stitched_macro_arc_without_manifest_track_switch_trigger():
    song = DummySong('stitched_macro_arc.wav')
    song.structure['sections'] = [
        {'label': 'section_0', 'start': 0.0, 'end': 24.0},
        {'label': 'section_1', 'start': 24.0, 'end': 48.0},
        {'label': 'section_2', 'start': 48.0, 'end': 72.0},
        {'label': 'section_3', 'start': 72.0, 'end': 96.0},
        {'label': 'section_4', 'start': 96.0, 'end': 120.0},
    ]
    song.structure['phrase_boundaries_seconds'] = [0, 30, 60, 90, 120]
    song.structure['novelty_boundaries_seconds'] = [26, 54, 88]
    song.energy['bar_rms'] = [0.32, 0.33, 0.12, 0.11, 0.34, 0.35, 0.13, 0.12, 0.36, 0.37, 0.14, 0.13, 0.35, 0.36, 0.15, 0.14]
    song.energy['bar_onset_density'] = [0.34, 0.35, 0.12, 0.12, 0.35, 0.36, 0.13, 0.13, 0.36, 0.37, 0.14, 0.14, 0.35, 0.36, 0.15, 0.15]
    song.energy['bar_low_band_ratio'] = [0.34, 0.34, 0.22, 0.22, 0.35, 0.35, 0.23, 0.23, 0.36, 0.36, 0.24, 0.24, 0.35, 0.35, 0.24, 0.24]
    song.energy['bar_spectral_flatness'] = [0.16, 0.16, 0.19, 0.19, 0.15, 0.15, 0.19, 0.19, 0.15, 0.15, 0.20, 0.20, 0.15, 0.15, 0.20, 0.20]

    report = evaluate_song(song)

    assert report.song_likeness.score >= 45.0
    assert report.song_likeness.details['aggregate_metrics']['composite_song_risk'] > 0.50
    assert report.gating['composite_detector_triggered'] is True
    assert report.gating['track_switch_seam_hard_reject_triggered'] is False
    assert report.gating['status'] == 'reject'
    assert 'composite detector says the render still sounds like multiple pasted songs' in report.gating['hard_fail_reasons']



def test_song_likeness_rewards_clear_backbone_and_major_sections(tmp_path: Path):
    run_dir = tmp_path / 'song_like'
    run_dir.mkdir()
    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    (run_dir / 'render_manifest.json').write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "intro", "source_parent": "A", "allowed_overlap": false, "overlap_beats_max": 0.0, "foreground_owner": "A", "background_owner": null, "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 1, "label": "verse", "source_parent": "A", "allowed_overlap": false, "overlap_beats_max": 0.0, "foreground_owner": "A", "background_owner": null, "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 2, "label": "build", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 1.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0, "transition_in": "lift"},
    {"index": 3, "label": "payoff", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 1.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0, "transition_in": "drop"},
    {"index": 4, "label": "outro", "source_parent": "A", "allowed_overlap": false, "overlap_beats_max": 0.0, "foreground_owner": "A", "background_owner": null, "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0}
  ],
  "work_orders": []
}""" % audio.as_posix(), encoding='utf-8')

    song = DummySong(str(audio))
    song.structure['sections'] = [
        {'label': 'intro', 'start': 0.0, 'end': 16.0},
        {'label': 'verse', 'start': 16.0, 'end': 40.0},
        {'label': 'build', 'start': 40.0, 'end': 64.0, 'transition_in': 'lift'},
        {'label': 'payoff', 'start': 64.0, 'end': 96.0, 'transition_in': 'drop'},
        {'label': 'outro', 'start': 96.0, 'end': 120.0},
    ]
    song.structure['phrase_boundaries_seconds'] = [0, 16, 40, 64, 96, 120]
    song.structure['novelty_boundaries_seconds'] = [16, 40, 64, 96]

    report = evaluate_song(song)
    metrics = report.song_likeness.details['aggregate_metrics']

    assert report.song_likeness.score >= 60.0
    assert report.gating['status'] == 'pass'
    assert metrics['backbone_continuity'] > 0.55
    assert metrics['recognizable_section_ratio'] >= 0.8
    assert metrics['boundary_recovery'] >= 0.7
    assert metrics['role_plausibility'] >= 0.55



def test_song_likeness_uses_boundary_and_audio_readability_even_with_generic_labels():
    readable_song = DummySong('generic_readable.wav')
    weak_song = DummySong('generic_weak.wav')

    for current in (readable_song, weak_song):
        current.structure['sections'] = [
            {'label': 'section_0', 'start': 0.0, 'end': 16.0},
            {'label': 'section_1', 'start': 16.0, 'end': 40.0},
            {'label': 'section_2', 'start': 40.0, 'end': 64.0, 'transition_in': 'lift'},
            {'label': 'section_3', 'start': 64.0, 'end': 96.0, 'transition_in': 'drop'},
            {'label': 'section_4', 'start': 96.0, 'end': 120.0},
        ]

    readable_song.structure['phrase_boundaries_seconds'] = [0, 16, 40, 64, 96, 120]
    readable_song.structure['novelty_boundaries_seconds'] = [16, 40, 64, 96]
    readable_song.energy['bar_rms'] = [0.08, 0.09, 0.10, 0.11, 0.16, 0.18, 0.19, 0.20, 0.24, 0.27, 0.31, 0.35, 0.42, 0.45, 0.40, 0.28]
    readable_song.energy['bar_onset_density'] = [0.14, 0.15, 0.16, 0.16, 0.18, 0.20, 0.21, 0.22, 0.26, 0.28, 0.30, 0.32, 0.40, 0.43, 0.36, 0.24]
    readable_song.energy['bar_low_band_ratio'] = [0.24, 0.25, 0.25, 0.26, 0.27, 0.28, 0.29, 0.29, 0.31, 0.32, 0.34, 0.35, 0.38, 0.40, 0.36, 0.30]
    readable_song.energy['bar_spectral_flatness'] = [0.22, 0.22, 0.21, 0.21, 0.20, 0.19, 0.19, 0.18, 0.17, 0.16, 0.15, 0.15, 0.13, 0.13, 0.14, 0.18]

    weak_song.structure['phrase_boundaries_seconds'] = [0, 30, 60, 90, 120]
    weak_song.structure['novelty_boundaries_seconds'] = [28, 88]
    weak_song.energy['bar_rms'] = [0.30, 0.31, 0.30, 0.31, 0.10, 0.11, 0.10, 0.11, 0.33, 0.34, 0.33, 0.34, 0.12, 0.12, 0.13, 0.12]
    weak_song.energy['bar_onset_density'] = [0.34, 0.35, 0.34, 0.35, 0.12, 0.12, 0.13, 0.12, 0.36, 0.36, 0.35, 0.36, 0.14, 0.14, 0.15, 0.14]
    weak_song.energy['bar_low_band_ratio'] = [0.35, 0.35, 0.35, 0.35, 0.22, 0.22, 0.22, 0.22, 0.36, 0.36, 0.36, 0.36, 0.24, 0.24, 0.24, 0.24]
    weak_song.energy['bar_spectral_flatness'] = [0.16, 0.16, 0.16, 0.16, 0.18, 0.18, 0.18, 0.18, 0.15, 0.15, 0.15, 0.15, 0.19, 0.19, 0.19, 0.19]

    readable_report = evaluate_song(readable_song)
    weak_report = evaluate_song(weak_song)
    readable_metrics = readable_report.song_likeness.details['aggregate_metrics']
    weak_metrics = weak_report.song_likeness.details['aggregate_metrics']

    assert readable_metrics['label_support_ratio'] == 0.0
    assert weak_metrics['label_support_ratio'] == 0.0
    assert readable_metrics['boundary_recovery'] > weak_metrics['boundary_recovery']
    assert readable_metrics['role_plausibility'] > weak_metrics['role_plausibility']
    assert readable_metrics['planner_audio_climax_conviction'] > weak_metrics['planner_audio_climax_conviction']
    assert readable_metrics['climax_conviction'] > weak_metrics['climax_conviction']
    assert readable_metrics['recognizable_section_ratio'] > weak_metrics['recognizable_section_ratio']
    assert readable_report.song_likeness.score > weak_report.song_likeness.score



def test_song_likeness_does_not_overreward_section_label_tokens_without_audio_support():
    readable_generic = DummySong('readable_generic.wav')
    weak_labeled = DummySong('weak_labeled.wav')

    readable_generic.structure['sections'] = [
        {'label': 'section_0', 'start': 0.0, 'end': 16.0},
        {'label': 'section_1', 'start': 16.0, 'end': 40.0},
        {'label': 'section_2', 'start': 40.0, 'end': 64.0, 'transition_in': 'lift'},
        {'label': 'section_3', 'start': 64.0, 'end': 96.0, 'transition_in': 'drop'},
        {'label': 'section_4', 'start': 96.0, 'end': 120.0},
    ]
    readable_generic.structure['phrase_boundaries_seconds'] = [0, 16, 40, 64, 96, 120]
    readable_generic.structure['novelty_boundaries_seconds'] = [16, 40, 64, 96]
    readable_generic.energy['bar_rms'] = [0.08, 0.09, 0.10, 0.11, 0.15, 0.17, 0.18, 0.19, 0.24, 0.27, 0.31, 0.35, 0.42, 0.45, 0.40, 0.26]
    readable_generic.energy['bar_onset_density'] = [0.14, 0.15, 0.16, 0.16, 0.18, 0.19, 0.20, 0.21, 0.25, 0.27, 0.29, 0.31, 0.38, 0.42, 0.35, 0.22]
    readable_generic.energy['bar_low_band_ratio'] = [0.24, 0.24, 0.25, 0.25, 0.27, 0.28, 0.28, 0.29, 0.31, 0.32, 0.33, 0.35, 0.38, 0.40, 0.36, 0.29]
    readable_generic.energy['bar_spectral_flatness'] = [0.22, 0.22, 0.21, 0.21, 0.20, 0.19, 0.19, 0.18, 0.17, 0.16, 0.15, 0.15, 0.13, 0.13, 0.14, 0.18]

    weak_labeled.structure['sections'] = [
        {'label': 'intro', 'start': 0.0, 'end': 30.0},
        {'label': 'verse', 'start': 30.0, 'end': 60.0},
        {'label': 'build', 'start': 60.0, 'end': 90.0, 'transition_in': 'lift'},
        {'label': 'payoff', 'start': 90.0, 'end': 120.0, 'transition_in': 'drop'},
    ]
    weak_labeled.structure['phrase_boundaries_seconds'] = [0, 30, 60, 90, 120]
    weak_labeled.structure['novelty_boundaries_seconds'] = [28, 88]
    weak_labeled.energy['bar_rms'] = [0.30, 0.31, 0.30, 0.31, 0.10, 0.11, 0.10, 0.11, 0.33, 0.34, 0.33, 0.34, 0.12, 0.12, 0.13, 0.12]
    weak_labeled.energy['bar_onset_density'] = [0.34, 0.35, 0.34, 0.35, 0.12, 0.12, 0.13, 0.12, 0.36, 0.36, 0.35, 0.36, 0.14, 0.14, 0.15, 0.14]
    weak_labeled.energy['bar_low_band_ratio'] = [0.35, 0.35, 0.35, 0.35, 0.22, 0.22, 0.22, 0.22, 0.36, 0.36, 0.36, 0.36, 0.24, 0.24, 0.24, 0.24]
    weak_labeled.energy['bar_spectral_flatness'] = [0.16, 0.16, 0.16, 0.16, 0.18, 0.18, 0.18, 0.18, 0.15, 0.15, 0.15, 0.15, 0.19, 0.19, 0.19, 0.19]

    readable_report = evaluate_song(readable_generic)
    weak_report = evaluate_song(weak_labeled)

    readable_metrics = readable_report.song_likeness.details['aggregate_metrics']
    weak_metrics = weak_report.song_likeness.details['aggregate_metrics']

    assert weak_metrics['label_support_ratio'] > readable_metrics['label_support_ratio']
    assert readable_metrics['role_plausibility'] > weak_metrics['role_plausibility']
    assert readable_metrics['planner_audio_climax_conviction'] > weak_metrics['planner_audio_climax_conviction']
    assert readable_report.song_likeness.score > weak_report.song_likeness.score
    assert readable_metrics['recognizable_section_ratio'] > 0.75
    assert weak_report.song_likeness.score < 82.0



def test_song_likeness_prefers_true_build_and_payoff_shapes_from_audio_evidence():
    readable_song = DummySong('readable_roles.wav')
    weak_song = DummySong('weak_roles.wav')

    for current in (readable_song, weak_song):
        current.structure['sections'] = [
            {'label': 'intro', 'start': 0.0, 'end': 24.0},
            {'label': 'verse', 'start': 24.0, 'end': 48.0},
            {'label': 'build', 'start': 48.0, 'end': 72.0, 'transition_in': 'lift'},
            {'label': 'payoff', 'start': 72.0, 'end': 96.0, 'transition_in': 'drop'},
            {'label': 'outro', 'start': 96.0, 'end': 120.0},
        ]
        current.structure['phrase_boundaries_seconds'] = [0, 24, 48, 72, 96, 120]
        current.structure['novelty_boundaries_seconds'] = [24, 48, 72, 96]
        current.energy['bar_onset_density'] = [0.12, 0.13, 0.14, 0.14, 0.18, 0.19, 0.20, 0.20, 0.22, 0.24, 0.28, 0.30, 0.34, 0.37, 0.40, 0.41]
        current.energy['bar_low_band_ratio'] = [0.22, 0.22, 0.23, 0.23, 0.25, 0.26, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.34, 0.35, 0.36, 0.35]
        current.energy['bar_spectral_flatness'] = [0.24, 0.24, 0.23, 0.23, 0.21, 0.21, 0.20, 0.20, 0.18, 0.18, 0.17, 0.17, 0.14, 0.14, 0.14, 0.16]

    readable_song.energy['bar_rms'] = [0.07, 0.08, 0.10, 0.11, 0.16, 0.17, 0.18, 0.18, 0.20, 0.23, 0.28, 0.33, 0.39, 0.41, 0.41, 0.34]
    weak_song.energy['bar_rms'] = [0.07, 0.08, 0.10, 0.11, 0.16, 0.17, 0.18, 0.18, 0.33, 0.20, 0.21, 0.22, 0.42, 0.24, 0.23, 0.20]

    readable_report = evaluate_song(readable_song)
    weak_report = evaluate_song(weak_song)

    readable_metrics = readable_report.song_likeness.details['aggregate_metrics']
    weak_metrics = weak_report.song_likeness.details['aggregate_metrics']

    assert readable_metrics['narrative_flow'] > weak_metrics['narrative_flow']
    assert readable_metrics['recognizable_section_ratio'] > weak_metrics['recognizable_section_ratio']
    assert readable_report.song_likeness.score > weak_report.song_likeness.score



def test_song_likeness_penalizes_section_energy_ping_pong_even_with_clean_boundaries():
    stable_song = DummySong('stable_arc.wav')
    ping_pong_song = DummySong('ping_pong_arc.wav')

    for current in (stable_song, ping_pong_song):
        current.structure['sections'] = [
            {'label': 'section_0', 'start': 0.0, 'end': 24.0},
            {'label': 'section_1', 'start': 24.0, 'end': 48.0},
            {'label': 'section_2', 'start': 48.0, 'end': 72.0, 'transition_in': 'lift'},
            {'label': 'section_3', 'start': 72.0, 'end': 96.0, 'transition_in': 'drop'},
            {'label': 'section_4', 'start': 96.0, 'end': 120.0},
        ]
        current.structure['phrase_boundaries_seconds'] = [0, 24, 48, 72, 96, 120]
        current.structure['novelty_boundaries_seconds'] = [24, 48, 72, 96]
        current.energy['bar_onset_density'] = [0.18, 0.18, 0.20, 0.20, 0.22, 0.22, 0.28, 0.28, 0.35, 0.35, 0.42, 0.42]
        current.energy['bar_low_band_ratio'] = [0.24, 0.24, 0.26, 0.26, 0.28, 0.28, 0.31, 0.31, 0.35, 0.35, 0.33, 0.33]
        current.energy['bar_spectral_flatness'] = [0.24, 0.24, 0.22, 0.22, 0.20, 0.20, 0.18, 0.18, 0.15, 0.15, 0.17, 0.17]

    stable_song.energy['bar_rms'] = [0.08, 0.09, 0.15, 0.17, 0.22, 0.24, 0.31, 0.34, 0.42, 0.45, 0.28, 0.24]
    ping_pong_song.energy['bar_rms'] = [0.08, 0.09, 0.34, 0.36, 0.10, 0.11, 0.38, 0.40, 0.12, 0.13, 0.35, 0.37]

    stable_report = evaluate_song(stable_song)
    ping_pong_report = evaluate_song(ping_pong_song)

    stable_metrics = stable_report.song_likeness.details['aggregate_metrics']
    ping_pong_metrics = ping_pong_report.song_likeness.details['aggregate_metrics']

    assert stable_metrics['boundary_recovery'] == ping_pong_metrics['boundary_recovery']
    assert ping_pong_metrics['direction_flip_ratio'] > stable_metrics['direction_flip_ratio']
    assert ping_pong_metrics['narrative_flow'] < stable_metrics['narrative_flow']
    assert ping_pong_report.song_likeness.score < stable_report.song_likeness.score
    assert any('ping-pong' in fix.lower() for fix in ping_pong_report.song_likeness.fixes)



def test_transition_score_detects_manifest_track_switching_patterns(tmp_path: Path):
    run_dir = tmp_path / 'switch_run'
    run_dir.mkdir()
    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    manifest = run_dir / 'render_manifest.json'
    manifest.write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "intro", "source_parent": "A", "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "transition_out": "swap"},
    {"index": 1, "label": "verse", "source_parent": "B", "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "transition_in": "swap", "transition_out": "swap"},
    {"index": 2, "label": "build", "source_parent": "A", "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "transition_in": "swap", "transition_out": "swap"},
    {"index": 3, "label": "payoff", "source_parent": "B", "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "transition_in": "swap"}
  ],
  "work_orders": []
}""" % audio.as_posix())

    song = DummySong(str(audio))
    report = evaluate_song(song)
    metrics = report.transition.details['aggregate_metrics']

    assert metrics['manifest_owner_switch_ratio'] >= 0.99
    assert metrics['manifest_alternating_triplet_ratio'] >= 0.99
    assert metrics['manifest_swap_density'] >= 0.99
    assert metrics['manifest_switch_detector_risk'] > 0.7
    assert report.gating['status'] == 'reject'
    assert 'obvious track-switch seams detected' in report.gating['hard_fail_reasons']
    assert report.gating['track_switch_seam_hard_reject_triggered'] is True
    assert any('switch detector' in line.lower() for line in report.transition.evidence)
    assert any('track switching' in fix.lower() for fix in report.transition.fixes)



def test_manifest_aware_transition_and_mix_penalties(tmp_path: Path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    manifest = run_dir / 'render_manifest.json'
    manifest.write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "intro", "allowed_overlap": true, "overlap_beats_max": 6.0, "foreground_owner": "A", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "both", "stretch_ratio": 1.28, "collapse_if_conflict": true, "transition_out": "blend"},
    {"index": 1, "label": "payoff", "allowed_overlap": true, "overlap_beats_max": 4.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "A", "vocal_policy": "both", "stretch_ratio": 1.0, "collapse_if_conflict": true, "transition_in": "swap"}
  ],
  "work_orders": [
    {"section_index": 0, "parent_id": "A", "foreground_state": "owner", "low_end_state": "owner", "vocal_state": "lead_only"},
    {"section_index": 0, "parent_id": "B", "foreground_state": "owner", "low_end_state": "owner", "vocal_state": "lead_only"},
    {"section_index": 1, "parent_id": "A", "foreground_state": "owner", "low_end_state": "owner", "vocal_state": "lead_only"},
    {"section_index": 1, "parent_id": "B", "foreground_state": "owner", "low_end_state": "owner", "vocal_state": "lead_only"}
  ],
  "warnings": ["section 0 stretch ratio warning"],
  "fallbacks": ["collapse to single source if conflict remains"]
}""" % audio.as_posix())

    song = DummySong(str(audio))
    report = evaluate_song(song)

    transition_manifest = report.transition.details['manifest_metrics']['aggregate_metrics']
    mix_manifest = report.mix_sanity.details['manifest_metrics']['aggregate_metrics']
    assert transition_manifest['multi_owner_conflict_ratio'] > 0.5
    assert transition_manifest['lead_conflict_ratio'] > 0.5
    assert transition_manifest['avg_overlap_beats'] >= 4.0
    assert transition_manifest['seam_risk_ratio'] > 0.0
    assert transition_manifest['low_end_owner_stability_risk'] > 0.0
    assert mix_manifest['crowding_ratio'] > 0.5
    assert any('manifest' in line.lower() for line in report.transition.evidence)
    assert any('manifest' in line.lower() for line in report.mix_sanity.evidence)
    assert any('manifest' in fix.lower() or 'lead-vocal ownership' in fix.lower() or 'overlap' in fix.lower() for fix in report.top_fixes + report.transition.fixes + report.mix_sanity.fixes)


def test_manifest_low_end_owner_stability_detector_penalizes_overlap_flips(tmp_path: Path):
    run_dir = tmp_path / 'low_end_owner_flip_run'
    run_dir.mkdir()
    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    manifest = run_dir / 'render_manifest.json'
    manifest.write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "intro", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0, "transition_out": "blend"},
    {"index": 1, "label": "verse", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0, "transition_in": "swap", "transition_out": "blend"},
    {"index": 2, "label": "build", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0, "transition_in": "swap", "transition_out": "lift"},
    {"index": 3, "label": "payoff", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0, "transition_in": "drop"}
  ],
  "work_orders": []
}""" % audio.as_posix(), encoding='utf-8')

    song = DummySong(str(audio))
    report = evaluate_song(song)

    transition_metrics = report.transition.details['aggregate_metrics']
    mix_manifest = report.mix_sanity.details['manifest_metrics']['aggregate_metrics']

    assert transition_metrics['manifest_low_end_owner_switch_ratio'] == 1.0
    assert transition_metrics['manifest_low_end_overlap_switch_ratio'] == 1.0
    assert transition_metrics['manifest_low_end_owner_stability_risk'] > 0.8
    assert mix_manifest['low_end_owner_switch_ratio'] == 1.0
    assert mix_manifest['low_end_overlap_switch_ratio'] == 1.0
    assert mix_manifest['low_end_owner_stability_risk'] > 0.8
    assert any('low-end ownership' in fix.lower() for fix in report.transition.fixes + report.mix_sanity.fixes)



def test_manifest_low_end_owner_stability_detector_surfaces_ping_pong_intrusions_even_without_constant_flips(tmp_path: Path):
    steady_dir = tmp_path / 'steady_low_end_run'
    steady_dir.mkdir()
    steady_audio = steady_dir / 'child_master.wav'
    steady_audio.write_bytes(b'fake')
    (steady_dir / 'render_manifest.json').write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "intro", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 1, "label": "verse", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 2, "label": "build", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 3, "label": "payoff", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0},
    {"index": 4, "label": "outro", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0}
  ],
  "work_orders": []
}""" % steady_audio.as_posix(), encoding='utf-8')

    ping_pong_dir = tmp_path / 'ping_pong_low_end_run'
    ping_pong_dir.mkdir()
    ping_pong_audio = ping_pong_dir / 'child_master.wav'
    ping_pong_audio.write_bytes(b'fake')
    (ping_pong_dir / 'render_manifest.json').write_text("""{
  "outputs": {"master_wav": "%s"},
  "sections": [
    {"index": 0, "label": "intro", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 1, "label": "verse", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 2, "label": "break", "source_parent": "B", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "B", "background_owner": "A", "low_end_owner": "B", "vocal_policy": "B_only", "stretch_ratio": 1.0},
    {"index": 3, "label": "build", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0},
    {"index": 4, "label": "outro", "source_parent": "A", "allowed_overlap": true, "overlap_beats_max": 2.0, "foreground_owner": "A", "background_owner": "B", "low_end_owner": "A", "vocal_policy": "A_only", "stretch_ratio": 1.0}
  ],
  "work_orders": []
}""" % ping_pong_audio.as_posix(), encoding='utf-8')

    steady_report = evaluate_song(DummySong(str(steady_audio)))
    ping_pong_report = evaluate_song(DummySong(str(ping_pong_audio)))

    steady_manifest = steady_report.mix_sanity.details['manifest_metrics']['aggregate_metrics']
    ping_pong_manifest = ping_pong_report.mix_sanity.details['manifest_metrics']['aggregate_metrics']

    assert steady_manifest['low_end_ping_pong_ratio'] == 0.0
    assert steady_manifest['low_end_longest_run_ratio'] >= 0.4
    assert steady_manifest['low_end_owner_majority_ratio'] >= 0.6
    assert ping_pong_manifest['low_end_ping_pong_ratio'] > 0.3
    assert ping_pong_manifest['low_end_longest_run_ratio'] < steady_manifest['low_end_longest_run_ratio']
    assert ping_pong_manifest['low_end_owner_stability_risk'] > steady_manifest['low_end_owner_stability_risk']



def test_listener_agent_rejects_non_songs_and_only_recommends_survivors(tmp_path: Path):
    strong = {
        'source_path': 'strong.wav',
        'duration_seconds': 60.0,
        'overall_score': 82.0,
        'structure': {'score': 85.0, 'summary': 'strong', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': 78.0, 'summary': 'stable pocket', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': 79.0, 'summary': 'late payoff lands', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': 77.0, 'summary': 'transitions feel musical', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': 83.0, 'summary': 'cohesive', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 80.0, 'summary': 'clear enough', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': 81.0, 'summary': 'reads like one song', 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': 'promising',
        'top_reasons': ['The render reads like one coherent song.'],
        'top_fixes': [],
        'gating': {'status': 'pass', 'raw_overall_score': 82.0},
        'analysis_version': '0.5.0',
    }
    poor = {
        'source_path': 'poor.wav',
        'duration_seconds': 60.0,
        'overall_score': 34.0,
        'structure': {'score': 52.0, 'summary': 'generic blocks', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': 31.0, 'summary': 'unstable', 'evidence': [], 'fixes': ['stabilize groove'], 'details': {}},
        'energy_arc': {'score': 28.0, 'summary': 'flat and front-loaded', 'evidence': [], 'fixes': ['build a real payoff'], 'details': {}},
        'transition': {'score': 33.0, 'summary': 'track switching', 'evidence': [], 'fixes': ['reduce obvious switches'], 'details': {}},
        'coherence': {'score': 40.0, 'summary': 'stitched', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 39.0, 'summary': 'messy', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': 20.0, 'summary': 'not one song', 'evidence': [], 'fixes': ['make it sound like one song'], 'details': {}},
        'verdict': 'poor',
        'top_reasons': ['The render does not read like one song.'],
        'top_fixes': ['Reject this output.'],
        'gating': {'status': 'reject', 'raw_overall_score': 34.0},
        'analysis_version': '0.5.0',
    }

    strong_path = tmp_path / 'strong.json'
    poor_path = tmp_path / 'poor.json'
    out = tmp_path / 'listener_agent.json'
    strong_path.write_text(json.dumps(strong), encoding='utf-8')
    poor_path.write_text(json.dumps(poor), encoding='utf-8')

    rc = ai_dj.listener_agent([str(strong_path), str(poor_path)], str(out), shortlist=1)
    assert rc == 0

    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['counts']['total'] == 2
    assert payload['counts']['survivors'] == 1
    assert payload['counts']['rejected'] == 1
    assert payload['listener_agent']['acceptance_criteria']['survivor_minimums']['song_likeness'] == 60.0
    assert payload['listener_agent']['acceptance_criteria']['hard_reject_component_floors']['transition'] == 45.0
    assert payload['recommended_for_human_review'][0]['label'] == 'strong.json'
    assert payload['recommended_for_human_review'][0]['decision'] == 'survivor'
    assert payload['recommended_for_human_review'][0]['acceptance_checks']['survivor_minimums']['song_likeness']['passed'] is True
    assert payload['recommended_for_human_review'][0]['acceptance_checks']['listen_gate']['passed'] is True
    assert payload['rejected'][0]['label'] == 'poor.json'
    assert payload['rejected'][0]['decision'] == 'reject'
    assert payload['rejected'][0]['acceptance_checks']['hard_reject_component_floors']['song_likeness']['passed'] is False
    assert payload['rejected'][0]['acceptance_checks']['listen_gate']['passed'] is False
    assert 'does not sound like one real song' in payload['rejected'][0]['hard_fail_reasons']
    assert payload['summary'][0].startswith('Listener agent kept 1 of 2')



def test_listener_agent_survivor_ranking_penalizes_critical_bottlenecks(tmp_path: Path):
    bottlenecked = {
        'source_path': 'bottlenecked.wav',
        'duration_seconds': 60.0,
        'overall_score': 86.0,
        'structure': {'score': 84.0, 'summary': 'clear sections', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': 90.0, 'summary': 'huge pocket', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': 90.0, 'summary': 'strong payoff', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': 90.0, 'summary': 'smooth swaps', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': 85.0, 'summary': 'cohesive', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 80.0, 'summary': 'clear enough', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {
            'score': 60.0,
            'summary': 'just enough to pass but still stitched',
            'evidence': [],
            'fixes': [],
            'details': {'aggregate_metrics': {'backbone_continuity': 0.62, 'recognizable_section_ratio': 0.7, 'boundary_recovery': 0.65, 'role_plausibility': 0.66, 'background_only_identity_gap': 0.18, 'owner_switch_ratio': 0.35}},
        },
        'verdict': 'promising',
        'top_reasons': ['Energetic and polished.'],
        'top_fixes': ['Still make it feel more like one song.'],
        'gating': {'status': 'pass', 'raw_overall_score': 86.0},
        'analysis_version': '0.5.0',
    }
    balanced = {
        'source_path': 'balanced.wav',
        'duration_seconds': 60.0,
        'overall_score': 82.0,
        'structure': {'score': 82.0, 'summary': 'clear sections', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': 78.0, 'summary': 'steady pocket', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': 78.0, 'summary': 'good late lift', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': 78.0, 'summary': 'musical handoffs', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': 78.0, 'summary': 'consistent', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 78.0, 'summary': 'clear enough', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {
            'score': 78.0,
            'summary': 'reads like one song',
            'evidence': [],
            'fixes': [],
            'details': {'aggregate_metrics': {'backbone_continuity': 0.73, 'recognizable_section_ratio': 0.74, 'boundary_recovery': 0.71, 'role_plausibility': 0.7, 'background_only_identity_gap': 0.12, 'owner_switch_ratio': 0.28}},
        },
        'verdict': 'promising',
        'top_reasons': ['Balanced and convincing.'],
        'top_fixes': [],
        'gating': {'status': 'pass', 'raw_overall_score': 82.0},
        'analysis_version': '0.5.0',
    }

    bottlenecked_path = tmp_path / 'bottlenecked.json'
    balanced_path = tmp_path / 'balanced.json'
    out = tmp_path / 'listener_agent_ranked.json'
    bottlenecked_path.write_text(json.dumps(bottlenecked), encoding='utf-8')
    balanced_path.write_text(json.dumps(balanced), encoding='utf-8')

    rc = ai_dj.listener_agent([str(bottlenecked_path), str(balanced_path)], str(out), shortlist=2)
    assert rc == 0

    payload = json.loads(out.read_text(encoding='utf-8'))
    recommended = payload['recommended_for_human_review']
    assert [row['label'] for row in recommended] == ['balanced.json', 'bottlenecked.json']
    assert recommended[0]['rank_diagnostics']['critical_floor'] == 78.0
    assert recommended[1]['rank_diagnostics']['weakest_critical_component'] == 'song_likeness'
    assert recommended[1]['rank_diagnostics']['imbalance_penalty'] > 0.0
    assert recommended[1]['rank_diagnostics']['weighted_rank'] > recommended[0]['rank_diagnostics']['weighted_rank']
    assert recommended[1]['listener_rank'] < recommended[0]['listener_rank']
    assert payload['listener_agent']['ranking_policy']['critical_rank_targets']['song_likeness'] == 70.0



def test_listener_agent_returns_no_human_shortlist_when_everything_is_bad(tmp_path: Path):
    weak = {
        'source_path': 'weak.wav',
        'duration_seconds': 60.0,
        'overall_score': 48.0,
        'structure': {'score': 60.0, 'summary': 'okay', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': 42.0, 'summary': 'loose', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': 40.0, 'summary': 'weak arc', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': 41.0, 'summary': 'switchy', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': 49.0, 'summary': 'uneven', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 45.0, 'summary': 'rough', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': 38.0, 'summary': 'still stitched', 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': 'weak',
        'top_reasons': [],
        'top_fixes': ['Reject this version.'],
        'gating': {'status': 'reject', 'raw_overall_score': 48.0},
        'analysis_version': '0.5.0',
    }

    weak_path = tmp_path / 'weak.json'
    weak_path.write_text(json.dumps(weak), encoding='utf-8')
    out = tmp_path / 'listener_agent_none.json'

    rc = ai_dj.listener_agent([str(weak_path)], str(out), shortlist=3)
    assert rc == 0

    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['recommended_for_human_review'] == []
    assert payload['counts']['survivors'] == 0
    assert payload['counts']['rejected'] == 1
    assert payload['summary'][0] == 'Listener agent found no outputs good enough for human review.'
