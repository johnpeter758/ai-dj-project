from __future__ import annotations

import json
from pathlib import Path

from scripts import listen_feedback_loop as loop
from scripts.listen_feedback_loop import build_feedback_brief


def _report(overall: float, *, verdict: str = 'promising', song_likeness: float = 80.0, groove: float = 80.0, energy_arc: float = 80.0, transition: float = 80.0, structure: float = 80.0, coherence: float = 80.0, mix_sanity: float = 80.0, climax_position: float | None = None) -> dict:
    song_likeness_details = {}
    if climax_position is not None:
        song_likeness_details = {'aggregate_metrics': {'climax_section_relative_center': climax_position}}
    return {
        'source_path': 'dummy.wav',
        'duration_seconds': 120.0,
        'overall_score': overall,
        'structure': {'score': structure, 'summary': 'structure', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': groove, 'summary': 'groove', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': energy_arc, 'summary': 'energy_arc', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': transition, 'summary': 'transition', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': coherence, 'summary': 'coherence', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': mix_sanity, 'summary': 'mix', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': song_likeness, 'summary': 'song_like', 'evidence': [], 'fixes': [], 'details': song_likeness_details},
        'verdict': verdict,
        'top_reasons': ['good thing'],
        'top_fixes': ['fix thing'],
        'gating': {'status': 'pass', 'raw_overall_score': overall},
        'analysis_version': '0.5.0',
    }


def _write_manifest(path: Path, labels: list[str]) -> None:
    sections = []
    start_bar = 0
    for index, label in enumerate(labels):
        sections.append({'index': index, 'label': label, 'start_bar': start_bar, 'bar_count': 4, 'source_parent': 'A'})
        start_bar += 4
    path.write_text(json.dumps({'outputs': {'master_wav': str(path.with_name('child_master.wav'))}, 'sections': sections}), encoding='utf-8')


def test_build_feedback_brief_ranks_biggest_gaps_and_code_targets(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref_a = tmp_path / 'ref_a.json'
    ref_b = tmp_path / 'ref_b.json'

    candidate.write_text(json.dumps(_report(58.0, verdict='weak', song_likeness=35.0, groove=42.0, energy_arc=40.0, transition=44.0)), encoding='utf-8')
    ref_a.write_text(json.dumps(_report(92.0, song_likeness=90.0, groove=89.0, energy_arc=91.0, transition=86.0)), encoding='utf-8')
    ref_b.write_text(json.dumps(_report(88.0, song_likeness=86.0, groove=84.0, energy_arc=87.0, transition=82.0)), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref_a), str(ref_b)], target_score=99.0)

    assert brief['goal']['current_overall_score'] == 58.0
    assert brief['goal']['gap_to_target'] == 41.0
    assert brief['gap_summary']['overall_vs_references'] < 0.0
    assert brief['ranked_interventions']
    top = brief['ranked_interventions'][0]
    assert top['component'] in {'song_likeness', 'energy_arc', 'groove'}
    assert brief['next_code_targets']
    assert any(path.endswith('src/core/planner/arrangement.py') for path in brief['next_code_targets'])
    assert brief['automation_loop'][0].startswith('Compare candidate against references')


def test_build_feedback_brief_collects_climax_position_alignment_vs_references(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref_a = tmp_path / 'ref_a.json'
    ref_b = tmp_path / 'ref_b.json'

    candidate.write_text(json.dumps(_report(70.0, energy_arc=66.0, climax_position=0.38)), encoding='utf-8')
    ref_a.write_text(json.dumps(_report(95.0, energy_arc=94.0, climax_position=0.78)), encoding='utf-8')
    ref_b.write_text(json.dumps(_report(93.0, energy_arc=92.0, climax_position=0.74)), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref_a), str(ref_b)])

    alignment = brief['reference_alignment']['climax_position']
    assert alignment['candidate_climax_position'] == 0.38
    assert alignment['avg_reference_climax_position'] == 0.76
    assert alignment['mean_absolute_delta'] == 0.38
    assert alignment['mean_position_similarity'] == 0.0
    failure_modes = {item['failure_mode'] for item in brief['planner_feedback_map']}
    assert 'climax_position_alignment' in failure_modes


def test_build_feedback_brief_collects_section_program_alignment_vs_references(tmp_path: Path, monkeypatch):
    candidate_manifest = tmp_path / 'candidate_render_manifest.json'
    ref_a_manifest = tmp_path / 'ref_a_render_manifest.json'
    ref_b_manifest = tmp_path / 'ref_b_render_manifest.json'
    _write_manifest(candidate_manifest, ['build', 'bridge', 'outro'])
    _write_manifest(ref_a_manifest, ['intro', 'verse', 'build', 'payoff', 'outro'])
    _write_manifest(ref_b_manifest, ['intro', 'verse', 'build', 'payoff', 'outro'])

    reports = {
        'candidate': {
            'input_label': 'candidate',
            'case_id': 'candidate',
            'input_path': 'candidate',
            'report_origin': 'render_output',
            'resolved_audio_path': 'candidate.wav',
            'render_manifest_path': str(candidate_manifest),
            'report': _report(70.0, structure=62.0),
        },
        'ref_a': {
            'input_label': 'ref_a',
            'case_id': 'ref_a',
            'input_path': 'ref_a',
            'report_origin': 'render_output',
            'resolved_audio_path': 'ref_a.wav',
            'render_manifest_path': str(ref_a_manifest),
            'report': _report(93.0, structure=92.0),
        },
        'ref_b': {
            'input_label': 'ref_b',
            'case_id': 'ref_b',
            'input_path': 'ref_b',
            'report_origin': 'render_output',
            'resolved_audio_path': 'ref_b.wav',
            'render_manifest_path': str(ref_b_manifest),
            'report': _report(91.0, structure=90.0),
        },
    }

    monkeypatch.setattr(loop, '_load_report', lambda path: reports[Path(path).name])
    monkeypatch.setattr(
        loop,
        '_component_gap_summary',
        lambda candidate_path, reference_paths: ({'overall_vs_references': -22.0, 'structure': -29.0, 'song_likeness': 0.0, 'groove': 0.0, 'energy_arc': 0.0, 'transition': 0.0, 'coherence': 0.0, 'mix_sanity': 0.0}, []),
    )

    brief = build_feedback_brief('candidate', ['ref_a', 'ref_b'])

    alignment = brief['reference_alignment']['section_program']
    assert alignment['candidate_program_signature'] == 'build -> bridge -> outro'
    assert alignment['mean_program_similarity'] < 0.67
    assert alignment['per_reference'][0]['reference_program_signature'] == 'intro -> verse -> build -> payoff -> outro'
    failure_modes = {item['failure_mode'] for item in brief['planner_feedback_map']}
    assert 'section_program_reference_alignment' in failure_modes


def test_build_feedback_brief_collects_reference_strengths(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref = tmp_path / 'ref.json'
    candidate.write_text(json.dumps(_report(70.0, song_likeness=68.0, groove=67.0, energy_arc=66.0)), encoding='utf-8')
    ref.write_text(json.dumps(_report(95.0, song_likeness=97.0, groove=96.0, energy_arc=94.0, transition=93.0, structure=92.0)), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref)])

    assert brief['references'][0]['overall_score'] == 95.0
    strengths = {item['component']: item['avg_reference_score'] for item in brief['reference_strengths']}
    assert strengths
    assert max(strengths.values()) >= 93.0


def test_build_feedback_brief_accepts_reference_collection_file(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref_a = tmp_path / 'ref_a.json'
    ref_b = tmp_path / 'ref_b.json'
    refs = tmp_path / 'refs.json'

    candidate.write_text(json.dumps(_report(70.0, song_likeness=68.0, groove=67.0, energy_arc=66.0)), encoding='utf-8')
    ref_a.write_text(json.dumps(_report(95.0, song_likeness=97.0, groove=96.0, energy_arc=94.0)), encoding='utf-8')
    ref_b.write_text(json.dumps(_report(90.0, song_likeness=92.0, groove=91.0, energy_arc=89.0)), encoding='utf-8')
    refs.write_text(json.dumps({'references': [ref_a.name, ref_b.name, ref_a.name]}), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(refs)])

    assert [item['overall_score'] for item in brief['references']] == [95.0, 90.0]
    assert len(brief['pairwise_comparisons']) == 2


def test_build_feedback_brief_exposes_reference_weighted_quality_gate_diagnostics(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref = tmp_path / 'ref.json'

    candidate.write_text(
        json.dumps(
            _report(
                91.0,
                structure=95.0,
                groove=95.0,
                energy_arc=95.0,
                transition=95.0,
                coherence=95.0,
                mix_sanity=95.0,
                song_likeness=40.0,
            )
        ),
        encoding='utf-8',
    )
    ref.write_text(
        json.dumps(
            _report(
                92.0,
                structure=80.0,
                groove=80.0,
                energy_arc=80.0,
                transition=80.0,
                coherence=80.0,
                mix_sanity=80.0,
                song_likeness=100.0,
            )
        ),
        encoding='utf-8',
    )

    brief = build_feedback_brief(str(candidate), [str(ref)])

    diagnostics = brief['quality_gate_diagnostics']['reference_weighted']
    assert diagnostics['candidate_weighted_score'] == 85.5
    assert diagnostics['reference_weighted_score'] == 83.4
    assert diagnostics['weighted_gap_vs_references'] == 2.1
    assert diagnostics['top_blockers'][0]['component'] == 'song_likeness'
    assert diagnostics['top_blockers'][0]['weighted_gap_contribution'] == -10.34



def test_build_feedback_brief_averages_mixed_reference_gaps_per_component(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    stronger_ref = tmp_path / 'stronger_ref.json'
    weaker_ref = tmp_path / 'weaker_ref.json'

    candidate.write_text(
        json.dumps(
            _report(
                74.0,
                verdict='mixed',
                structure=70.0,
                groove=72.0,
                energy_arc=69.0,
                transition=73.0,
                coherence=71.0,
                mix_sanity=68.0,
                song_likeness=75.0,
            )
        ),
        encoding='utf-8',
    )
    stronger_ref.write_text(
        json.dumps(
            _report(
                86.0,
                verdict='promising',
                structure=84.0,
                groove=88.0,
                energy_arc=87.0,
                transition=85.0,
                coherence=83.0,
                mix_sanity=86.0,
                song_likeness=89.0,
            )
        ),
        encoding='utf-8',
    )
    weaker_ref.write_text(
        json.dumps(
            _report(
                70.0,
                verdict='mixed',
                structure=68.0,
                groove=65.0,
                energy_arc=66.0,
                transition=69.0,
                coherence=67.0,
                mix_sanity=62.0,
                song_likeness=72.0,
            )
        ),
        encoding='utf-8',
    )

    brief = build_feedback_brief(str(candidate), [str(stronger_ref), str(weaker_ref)])

    assert brief['gap_summary']['overall_vs_references'] == -4.0
    assert brief['gap_summary']['groove'] == -4.5
    assert brief['gap_summary']['energy_arc'] == -7.5
    assert brief['gap_summary']['mix_sanity'] == -6.0
    assert brief['gap_summary']['song_likeness'] == -5.5
    assert brief['gap_summary']['transition'] == -4.0
    assert brief['gap_summary']['structure'] == -6.0
    assert brief['gap_summary']['coherence'] == -4.0

    assert [item['decision']['winner'] for item in brief['pairwise_comparisons']] == ['right', 'left']
    assert brief['ranked_interventions'][0]['component'] == 'energy_arc'
    assert all(item['gap_vs_references'] < 0.0 for item in brief['ranked_interventions'])


def test_build_feedback_brief_exposes_reference_groove_similarity(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref = tmp_path / 'ref.json'

    candidate_payload = _report(74.0, groove=72.0)
    candidate_payload['groove']['details'] = {
        'beat_stability': 0.72,
        'pocket_stability': 0.68,
        'collapse_severity': 0.21,
    }
    reference_payload = _report(90.0, groove=88.0)
    reference_payload['groove']['details'] = {
        'beat_stability': 0.89,
        'pocket_stability': 0.86,
        'collapse_severity': 0.08,
    }

    candidate.write_text(json.dumps(candidate_payload), encoding='utf-8')
    ref.write_text(json.dumps(reference_payload), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref)])

    groove_similarity = brief['reference_groove_similarity']
    assert groove_similarity['comparison_count'] == 1
    assert groove_similarity['avg_similarity'] is not None
    assert groove_similarity['strongest_mismatches']
    assert groove_similarity['strongest_mismatches'][0]['reference_label'] == 'ref.json'



def test_build_feedback_brief_exposes_reference_dynamic_contour_similarity(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref = tmp_path / 'ref.json'

    candidate_payload = _report(72.0, energy_arc=64.0)
    candidate_payload['energy_arc']['details'] = {
        'macro_profile': {
            'late_peak_ratio': 0.33,
            'payoff_contrast': 0.28,
        },
        'dynamic_range_rms': 0.17,
    }
    reference_payload = _report(91.0, energy_arc=90.0)
    reference_payload['energy_arc']['details'] = {
        'macro_profile': {
            'late_peak_ratio': 0.82,
            'payoff_contrast': 0.74,
        },
        'dynamic_range_rms': 0.43,
    }

    candidate.write_text(json.dumps(candidate_payload), encoding='utf-8')
    ref.write_text(json.dumps(reference_payload), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref)])

    contour_similarity = brief['reference_dynamic_contour_similarity']
    assert contour_similarity['comparison_count'] == 1
    assert contour_similarity['avg_similarity'] is not None
    assert contour_similarity['strongest_mismatches']
    assert contour_similarity['strongest_mismatches'][0]['reference_label'] == 'ref.json'
    assert brief['reference_alignment']['dynamic_contour']['avg_similarity'] == contour_similarity['avg_similarity']
    failure_modes = {item['failure_mode'] for item in brief['planner_feedback_map']}
    assert 'dynamic_contour_alignment' in failure_modes



def test_build_feedback_brief_emits_structured_planner_feedback_map(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref = tmp_path / 'ref.json'

    payload = _report(52.0, verdict='weak', song_likeness=34.0, energy_arc=38.0, transition=36.0, mix_sanity=40.0)
    payload['song_likeness']['summary'] = 'stitched and not one song'
    payload['song_likeness']['fixes'] = ['Improve backbone continuity and reduce cluttered donor carryover.']
    payload['song_likeness']['details'] = {'aggregate_metrics': {'backbone_continuity': 0.31}}
    payload['energy_arc']['fixes'] = ['The fusion appears to spend its hook too early; build a real late payoff.']
    payload['transition']['fixes'] = ['Manifest ownership is flipping often enough to read like track switching.']
    payload['transition']['details'] = {'aggregate_metrics': {'manifest_switch_detector_risk': 0.82}, 'transition_diagnostics': ['switch detector: owner switch ratio 1.00']}
    payload['mix_sanity']['fixes'] = ['Low-end ownership is not staying anchored through adjacent sections.']
    payload['mix_sanity']['details'] = {'manifest_metrics': {'aggregate_metrics': {'low_end_owner_stability_risk': 0.91}}}
    payload['top_fixes'] = ['Make it feel like one song with a steadier backbone.']

    candidate.write_text(json.dumps(payload), encoding='utf-8')
    ref.write_text(json.dumps(_report(93.0, song_likeness=92.0, energy_arc=90.0, transition=89.0, mix_sanity=88.0)), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref)])

    failure_modes = {item['failure_mode'] for item in brief['planner_feedback_map']}
    assert 'backbone_continuity' in failure_modes
    assert 'late_payoff_mapping' in failure_modes
    assert 'ownership_switching' in failure_modes
    assert 'low_end_ownership' in failure_modes
    backbone = next(item for item in brief['planner_feedback_map'] if item['failure_mode'] == 'backbone_continuity')
    assert any('src/core/planner/arrangement.py' == path for path in backbone['planner_code_targets'])
    assert any('backbone continuity' in text.lower() for text in backbone['matched_feedback'])
    assert any(path.endswith('src/core/planner/arrangement.py') for path in brief['next_code_targets'])


def test_build_feedback_brief_maps_structure_and_groove_feedback_to_planner_targets(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref = tmp_path / 'ref.json'

    payload = _report(60.0, verdict='mixed', structure=49.0, groove=44.0)
    payload['structure']['fixes'] = ['Increase structural certainty so the planner is not forced into coarse whole-song windows.']
    payload['groove']['fixes'] = ['Tighten bar-to-bar rhythmic continuity; the pocket collapses abruptly across adjacent windows.']
    payload['top_fixes'] = ['Sharpen readable sections and stabilize groove handoffs.']

    candidate.write_text(json.dumps(payload), encoding='utf-8')
    ref.write_text(json.dumps(_report(91.0, structure=92.0, groove=90.0)), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref)])

    failure_modes = {item['failure_mode'] for item in brief['planner_feedback_map']}
    assert 'section_readability' in failure_modes
    assert 'groove_handoff_stability' in failure_modes
    section_map = next(item for item in brief['planner_feedback_map'] if item['failure_mode'] == 'section_readability')
    assert section_map['component'] == 'structure'
    assert any('coarse whole-song windows' in text.lower() for text in section_map['matched_feedback'])



def test_build_feedback_brief_emits_structured_render_feedback_map(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref = tmp_path / 'ref.json'

    payload = _report(55.0, verdict='weak', transition=38.0, mix_sanity=41.0)
    payload['transition']['fixes'] = [
        'Smooth energy handoffs at section seams so transitions do not feel pasted or cliff-like.',
        'Control brightness and timbre swaps at boundaries; filtered handoffs or better source-window matching should reduce spectral shock.',
    ]
    payload['transition']['details'] = {
        'aggregate_metrics': {
            'avg_edge_cliff_risk': 0.44,
            'avg_vocal_competition_risk': 0.37,
        }
    }
    payload['mix_sanity']['fixes'] = [
        'Reduce full-spectrum overlap; too many simultaneous elements are making the render feel crowded instead of arranged.',
        'Resolve lead-vocal ownership more clearly; the current texture suggests competing lead material.',
        'Low-end ownership is flipping too often across adjacent sections, especially through overlaps; keep the kick/sub anchor on one parent longer.',
    ]
    payload['mix_sanity']['details'] = {
        'ownership_clutter_metrics': {'crowding_burst_risk': 0.52},
        'manifest_metrics': {'aggregate_metrics': {'low_end_owner_stability_risk': 0.88}},
    }
    payload['top_fixes'] = ['Trim overlap entries/exits and keep one clear low-end owner through seam windows.']

    candidate.write_text(json.dumps(payload), encoding='utf-8')
    ref.write_text(json.dumps(_report(92.0, transition=90.0, mix_sanity=89.0)), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref)])

    failure_modes = {item['failure_mode'] for item in brief['render_feedback_map']}
    assert 'seam_handoff_envelope' in failure_modes
    assert 'overlap_density_control' in failure_modes
    assert 'foreground_vocal_singularity' in failure_modes
    assert 'low_end_seam_control' in failure_modes

    seam_map = next(item for item in brief['render_feedback_map'] if item['failure_mode'] == 'seam_handoff_envelope')
    assert seam_map['component'] == 'transition'
    assert any(path.endswith('src/core/render/renderer.py') for path in seam_map['render_code_targets'])
    assert any('cliff-like' in text.lower() or 'spectral shock' in text.lower() for text in seam_map['matched_feedback'])
    assert any(path.endswith('src/core/render/renderer.py') for path in brief['next_code_targets'])


def test_build_feedback_brief_extracts_stable_reference_features(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref_a = tmp_path / 'ref_a.json'
    ref_b = tmp_path / 'ref_b.json'

    candidate.write_text(json.dumps(_report(72.0, transition=68.0, mix_sanity=66.0)), encoding='utf-8')

    ref_a_payload = _report(94.0, transition=92.0, mix_sanity=90.0)
    ref_a_payload['transition']['details'] = {
        'aggregate_metrics': {
            'avg_edge_cliff_risk': 0.11,
            'manifest_switch_detector_risk': 0.18,
        }
    }
    ref_a_payload['mix_sanity']['details'] = {
        'manifest_metrics': {'aggregate_metrics': {'low_end_owner_stability_risk': 0.14}}
    }

    ref_b_payload = _report(91.0, transition=89.0, mix_sanity=88.0)
    ref_b_payload['transition']['details'] = {
        'aggregate_metrics': {
            'avg_edge_cliff_risk': 0.09,
            'manifest_switch_detector_risk': 0.21,
        }
    }
    ref_b_payload['mix_sanity']['details'] = {
        'manifest_metrics': {'aggregate_metrics': {'low_end_owner_stability_risk': 0.16}}
    }

    ref_a.write_text(json.dumps(ref_a_payload), encoding='utf-8')
    ref_b.write_text(json.dumps(ref_b_payload), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref_a), str(ref_b)])

    transition_summary = next(item for item in brief['reference_feature_summary'] if item['component'] == 'transition')
    assert transition_summary['reference_score_floor'] == 89.0
    assert transition_summary['reference_score_avg'] == 90.5
    assert transition_summary['reference_score_ceiling'] == 92.0
    metric_names = {item['metric'] for item in transition_summary['stable_metrics']}
    assert 'aggregate_metrics.avg_edge_cliff_risk' in metric_names
    assert 'aggregate_metrics.manifest_switch_detector_risk' in metric_names

    mix_summary = next(item for item in brief['reference_feature_summary'] if item['component'] == 'mix_sanity')
    assert mix_summary['reference_score_avg'] == 89.0
    low_end_metric = next(item for item in mix_summary['stable_metrics'] if item['metric'] == 'manifest_metrics.aggregate_metrics.low_end_owner_stability_risk')
    assert low_end_metric['avg'] == 0.15
    assert low_end_metric['spread'] == 0.02
