from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ai_dj


def _listen_report(*, source_path: str, overall: float, song_likeness: float, groove: float, energy_arc: float, transition: float, verdict: str, gate_status: str) -> dict:
    return {
        'source_path': source_path,
        'duration_seconds': 60.0,
        'overall_score': overall,
        'structure': {'score': 80.0, 'summary': 'clear sections', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': groove, 'summary': 'groove', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': energy_arc, 'summary': 'arc', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': transition, 'summary': 'transition', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': 78.0, 'summary': 'coherent', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': 77.0, 'summary': 'clean enough', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {
            'score': song_likeness,
            'summary': 'song-like' if song_likeness >= 60.0 else 'borderline song-likeness' if song_likeness >= 55.0 else 'not one song',
            'evidence': [],
            'fixes': [],
            'details': {
                'aggregate_metrics': {
                    'backbone_continuity': 0.72 if song_likeness >= 60.0 else 0.54 if song_likeness >= 55.0 else 0.22,
                    'recognizable_section_ratio': 0.70 if song_likeness >= 60.0 else 0.52 if song_likeness >= 55.0 else 0.20,
                    'boundary_recovery': 0.68 if song_likeness >= 60.0 else 0.50 if song_likeness >= 55.0 else 0.18,
                    'role_plausibility': 0.69 if song_likeness >= 60.0 else 0.51 if song_likeness >= 55.0 else 0.24,
                    'background_only_identity_gap': 0.12 if song_likeness >= 60.0 else 0.28 if song_likeness >= 55.0 else 0.58,
                    'owner_switch_ratio': 0.22 if song_likeness >= 60.0 else 0.44 if song_likeness >= 55.0 else 0.88,
                }
            },
        },
        'verdict': verdict,
        'top_reasons': ['good output'] if verdict == 'promising' else ['bad output'],
        'top_fixes': [] if verdict == 'promising' else ['reject this output'],
        'gating': {'status': gate_status, 'raw_overall_score': overall},
        'analysis_version': '0.5.0',
    }



def _candidate_dir(tmp_path: Path, candidate_id: str, report: dict, variant_id: str) -> dict:
    run_dir = tmp_path / candidate_id
    run_dir.mkdir(parents=True)
    audio_path = run_dir / 'child_master.wav'
    audio_path.write_bytes(b'fake')
    report = dict(report)
    report['source_path'] = str(audio_path)
    (run_dir / 'listen_report.json').write_text(json.dumps(report), encoding='utf-8')
    (run_dir / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(audio_path)}}), encoding='utf-8')
    (run_dir / 'arrangement_plan.json').write_text(json.dumps({'sections': [], 'planning_diagnostics': {'variant': {'variant_id': variant_id}}}), encoding='utf-8')
    return {
        'candidate_id': candidate_id,
        'outdir': str(run_dir),
        'render_manifest_path': str(run_dir / 'render_manifest.json'),
        'arrangement_plan_path': str(run_dir / 'arrangement_plan.json'),
        'master_wav_path': str(audio_path),
        'master_mp3_path': None,
        'variant_config': {'variant_id': variant_id, 'strategy': 'single_section_alternate' if variant_id != 'baseline' else 'baseline'},
    }



def test_build_auto_shortlist_report_keeps_only_survivor_shortlist(tmp_path: Path):
    strong = _candidate_dir(
        tmp_path,
        'candidate_001',
        _listen_report(source_path='strong.wav', overall=82.0, song_likeness=79.0, groove=76.0, energy_arc=78.0, transition=77.0, verdict='promising', gate_status='pass'),
        'baseline',
    )
    solid = _candidate_dir(
        tmp_path,
        'candidate_002',
        _listen_report(source_path='solid.wav', overall=79.0, song_likeness=74.0, groove=73.0, energy_arc=72.0, transition=74.0, verdict='promising', gate_status='pass'),
        'swap_01_payoff_B',
    )
    weak = _candidate_dir(
        tmp_path,
        'candidate_003',
        _listen_report(source_path='weak.wav', overall=42.0, song_likeness=30.0, groove=41.0, energy_arc=40.0, transition=39.0, verdict='poor', gate_status='reject'),
        'swap_02_build_A',
    )

    candidates = [ai_dj._evaluate_auto_shortlist_candidate(candidate) for candidate in [strong, solid, weak]]
    report = ai_dj._build_auto_shortlist_report(
        song_a='song_a.wav',
        song_b='song_b.wav',
        output_root=tmp_path,
        batch_size=3,
        shortlist=2,
        variant_mode='safe',
        candidates=candidates,
    )

    assert report['listener_agent_report']['counts']['survivors'] == 2
    assert report['listener_agent_report']['counts']['rejected'] == 1
    assert [row['candidate_id'] for row in report['recommended_shortlist']] == ['candidate_001', 'candidate_002']
    assert report['closest_misses'] == []
    assert report['pairwise_pool']['winner'] == 'candidate_001'
    assert 'No candidates survived' not in ' '.join(report['summary'])



def test_build_auto_shortlist_report_preserves_closest_misses_when_no_survivors(tmp_path: Path):
    borderline_a = _candidate_dir(
        tmp_path,
        'candidate_001',
        _listen_report(source_path='a.wav', overall=64.0, song_likeness=58.0, groove=57.0, energy_arc=55.0, transition=60.0, verdict='mixed', gate_status='review'),
        'baseline',
    )
    borderline_b = _candidate_dir(
        tmp_path,
        'candidate_002',
        _listen_report(source_path='b.wav', overall=62.0, song_likeness=56.0, groove=55.0, energy_arc=54.0, transition=58.0, verdict='mixed', gate_status='review'),
        'swap_01_bridge_B',
    )

    candidates = [ai_dj._evaluate_auto_shortlist_candidate(candidate) for candidate in [borderline_a, borderline_b]]
    report = ai_dj._build_auto_shortlist_report(
        song_a='song_a.wav',
        song_b='song_b.wav',
        output_root=tmp_path,
        batch_size=2,
        shortlist=2,
        variant_mode='safe',
        candidates=candidates,
    )

    assert report['recommended_shortlist'] == []
    assert [row['candidate_id'] for row in report['closest_misses']] == ['candidate_001', 'candidate_002']
    assert 'No candidates survived the automatic gate' in ' '.join(report['summary'])



def test_apply_auto_shortlist_pruning_deletes_non_survivor_run_dirs(tmp_path: Path):
    keep_dir = tmp_path / 'candidate_001'
    drop_dir = tmp_path / 'candidate_002'
    keep_dir.mkdir(parents=True)
    drop_dir.mkdir(parents=True)
    report = {
        'recommended_shortlist': [{'candidate_id': 'candidate_001'}],
        'closest_misses': [{'candidate_id': 'candidate_002', 'run_dir': str(drop_dir), 'audio_path': str(drop_dir / 'child_master.wav')}],
        'candidates': [
            {'candidate_id': 'candidate_001', 'run_dir': str(keep_dir), 'audio_path': str(keep_dir / 'child_master.wav')},
            {'candidate_id': 'candidate_002', 'run_dir': str(drop_dir), 'audio_path': str(drop_dir / 'child_master.wav')},
        ],
        'summary': [],
        'pruning': {'enabled': False, 'kept_candidate_ids': ['candidate_001'], 'deleted_candidate_ids': [], 'deleted_candidate_count': 0},
    }

    pruned = ai_dj._apply_auto_shortlist_pruning(tmp_path, report, delete_non_survivors=True)
    assert keep_dir.exists()
    assert not drop_dir.exists()
    assert pruned['pruning']['enabled'] is True
    assert pruned['pruning']['deleted_candidate_ids'] == ['candidate_002']
    assert pruned['closest_misses'][0]['artifacts_pruned'] is True



def test_build_auto_shortlist_variant_configs_emits_baseline_plus_safe_alternates():
    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [
                {
                    'label': 'payoff',
                    'selected_parent': 'A',
                    'selected_window_label': 'phrase_1_2',
                    'selection_rank': 1,
                    'candidate_shortlist': [
                        {
                            'rank': 1,
                            'parent_id': 'A',
                            'window_label': 'phrase_1_2',
                            'selected': True,
                            'planner_error': 0.20,
                            'error_delta_vs_selected': 0.0,
                            'score_breakdown': {'stretch_ratio': 1.0, 'stretch_gate': 0.0, 'seam_risk': 0.25, 'transition_viability': 0.30},
                        },
                        {
                            'rank': 2,
                            'parent_id': 'B',
                            'window_label': 'phrase_5_6',
                            'selected': False,
                            'planner_error': 0.34,
                            'error_delta_vs_selected': 0.14,
                            'score_breakdown': {'stretch_ratio': 1.06, 'stretch_gate': 0.0, 'seam_risk': 0.34, 'transition_viability': 0.42},
                        },
                    ],
                    'cross_parent_best_alternate': {
                        'rank': 2,
                        'parent_id': 'B',
                        'window_label': 'phrase_5_6',
                        'selected': False,
                        'planner_error': 0.34,
                        'error_delta_vs_selected': 0.14,
                        'score_breakdown': {'stretch_ratio': 1.06, 'stretch_gate': 0.0, 'seam_risk': 0.34, 'transition_viability': 0.42},
                    },
                }
            ],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    assert configs[0]['variant_id'] == 'baseline'
    assert any(config['strategy'] == 'single_section_alternate' for config in configs[1:])
    assert len(configs) == 2


def _make_section_with_alternate(label: str, selected_parent: str, selected_label: str, alt_parent: str, alt_label: str):
    return {
        'label': label,
        'selected_parent': selected_parent,
        'selected_window_label': selected_label,
        'selection_rank': 1,
        'candidate_shortlist': [
            {
                'rank': 1,
                'parent_id': selected_parent,
                'window_label': selected_label,
                'selected': True,
                'planner_error': 0.22,
                'error_delta_vs_selected': 0.0,
                'score_breakdown': {'stretch_ratio': 1.0, 'stretch_gate': 0.0, 'seam_risk': 0.2, 'transition_viability': 0.3},
            },
            {
                'rank': 2,
                'parent_id': alt_parent,
                'window_label': alt_label,
                'selected': False,
                'planner_error': 0.35,
                'error_delta_vs_selected': 0.13,
                'score_breakdown': {'stretch_ratio': 1.05, 'stretch_gate': 0.0, 'seam_risk': 0.3, 'transition_viability': 0.4},
            },
        ],
        'cross_parent_best_alternate': {
            'rank': 2,
            'parent_id': alt_parent,
            'window_label': alt_label,
            'selected': False,
            'planner_error': 0.35,
            'error_delta_vs_selected': 0.13,
            'score_breakdown': {'stretch_ratio': 1.05, 'stretch_gate': 0.0, 'seam_risk': 0.3, 'transition_viability': 0.4},
        },
    }



def test_build_auto_shortlist_variant_configs_can_emit_dual_section_variants_when_room_exists():
    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [
                _make_section_with_alternate('intro', 'A', 'phrase_0_2', 'B', 'phrase_6_8'),
                _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10'),
                _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12'),
            ],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=6, variant_mode='safe')
    assert configs[0]['variant_id'] == 'baseline'
    assert any(config['strategy'] == 'single_section_alternate' for config in configs[1:])
    assert any(config['strategy'] == 'dual_section_alternate' for config in configs[1:])



def test_build_auto_shortlist_variant_configs_reserves_room_for_combo_under_tight_budget():
    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [
                _make_section_with_alternate('intro', 'A', 'phrase_0_2', 'B', 'phrase_6_8'),
                _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10'),
                _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12'),
            ],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')

    assert len(configs) == 3
    assert configs[0]['variant_id'] == 'baseline'
    assert any(config['strategy'] == 'single_section_alternate' for config in configs[1:])
    assert any(config['strategy'] == 'dual_section_alternate' for config in configs[1:])



def test_build_auto_shortlist_variant_configs_prioritizes_payoff_combo_when_budget_is_one_combo_slot():
    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [
                _make_section_with_alternate('intro', 'A', 'phrase_0_2', 'B', 'phrase_6_8'),
                _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10'),
                _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12'),
            ],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}

    assert 'payoff' in combo_labels



def test_build_auto_shortlist_variant_configs_avoids_intro_payoff_combo_when_core_payoff_combo_exists():
    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [
                _make_section_with_alternate('intro', 'A', 'phrase_0_2', 'B', 'phrase_6_8'),
                _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'A', 'phrase_8_10'),
                _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'A', 'phrase_10_12'),
            ],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}

    assert 'payoff' in combo_labels
    assert 'intro' not in combo_labels



def test_build_auto_shortlist_variant_configs_prefers_payoff_build_combo_over_payoff_verse_when_single_combo_slot():
    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [
                _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10'),
                _make_section_with_alternate('build', 'A', 'phrase_3_5', 'B', 'phrase_9_11'),
                _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12'),
            ],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}

    assert combo_labels == {'build', 'payoff'}


def test_build_auto_shortlist_variant_configs_forces_core_donor_single_before_same_parent_fallback():
    payoff_same_parent = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'A', 'phrase_10_12')
    verse_donor = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    intro_same_parent = _make_section_with_alternate('intro', 'A', 'phrase_0_2', 'A', 'phrase_6_8')
    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [payoff_same_parent, verse_donor, intro_same_parent],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    single = next(config for config in configs if config['strategy'] == 'single_section_alternate')
    swap = single['swaps'][0]

    assert str(swap.get('section_label') or '').strip().lower() == 'verse'
    assert swap.get('alternate_parent') == 'B'



def test_build_auto_shortlist_variant_configs_treats_suffixed_core_labels_as_core_for_donor_priority():
    payoff_same_parent = _make_section_with_alternate('payoff_a', 'A', 'phrase_4_6', 'A', 'phrase_10_12')
    verse_donor = _make_section_with_alternate('verse_b', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    intro_same_parent = _make_section_with_alternate('intro', 'A', 'phrase_0_2', 'A', 'phrase_6_8')
    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [payoff_same_parent, verse_donor, intro_same_parent],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    single = next(config for config in configs if config['strategy'] == 'single_section_alternate')
    swap = single['swaps'][0]

    assert str(swap.get('section_label') or '').strip().lower() == 'verse_b'
    assert swap.get('alternate_parent') == 'B'



def test_build_auto_shortlist_variant_configs_baseline_combo_forces_core_donor_when_singles_have_none():
    payoff_same_parent = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'A', 'phrase_10_12')
    build_same_parent = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'A', 'phrase_9_11')
    verse_donor = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    bridge_donor = _make_section_with_alternate('bridge', 'A', 'phrase_5_7', 'B', 'phrase_11_13')
    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'baseline',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [payoff_same_parent, build_same_parent, verse_donor, bridge_donor],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}
    combo_parents = {str(swap.get('alternate_parent') or '') for swap in combo.get('swaps', [])}

    assert combo_labels & {'verse', 'build', 'payoff', 'bridge'}
    assert 'B' in combo_parents
