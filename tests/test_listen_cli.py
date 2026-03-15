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
    assert any('low-end owner' in fix.lower() or 'foreground owner' in fix.lower() or 'lead-vocal ownership' in fix.lower() or 'competing lead' in fix.lower() for fix in report.mix_sanity.fixes)


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
    assert any('backbone continuity' in fix.lower() or 'cluttered donor carryover' in fix.lower() for fix in report.top_fixes)



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
    assert mix_manifest['crowding_ratio'] > 0.5
    assert any('manifest' in line.lower() for line in report.transition.evidence)
    assert any('manifest' in line.lower() for line in report.mix_sanity.evidence)
    assert any('manifest' in fix.lower() or 'lead-vocal ownership' in fix.lower() or 'overlap' in fix.lower() for fix in report.top_fixes + report.transition.fixes + report.mix_sanity.fixes)


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
    assert payload['recommended_for_human_review'][0]['label'] == 'strong.json'
    assert payload['recommended_for_human_review'][0]['decision'] == 'survivor'
    assert payload['rejected'][0]['label'] == 'poor.json'
    assert payload['rejected'][0]['decision'] == 'reject'
    assert 'does not sound like one real song' in payload['rejected'][0]['hard_fail_reasons']
    assert payload['summary'][0].startswith('Listener agent kept 1 of 2')



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
