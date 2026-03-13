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
        'verdict': 'promising',
        'top_reasons': [],
        'top_fixes': [],
        'analysis_version': '0.3.0',
    }
    right = {
        **left,
        'source_path': 'right.wav',
        'overall_score': 74.0,
        'energy_arc': {'score': 66.0, 'summary': 'flat', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 70.0, 'summary': 'crowded', 'evidence': [], 'fixes': [], 'details': {}},
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
    assert payload['deltas']['overall_score_delta'] == 8.0
    assert payload['deltas']['component_score_deltas']['energy_arc'] == 13.0
    assert payload['winner']['components']['mix_sanity'] == 'left'
    assert payload['left']['report_origin'] == 'listen_report'
    assert payload['right']['report_origin'] == 'listen_report'
    assert payload['summary']


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
