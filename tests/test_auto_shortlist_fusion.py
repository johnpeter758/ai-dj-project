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


def test_build_auto_shortlist_variant_configs_prefers_contiguous_handoff_combo_over_lower_error_noncontiguous_combo():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    bridge = _make_section_with_alternate('bridge', 'A', 'phrase_3_5', 'B', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12')

    # Make payoff+verse slightly higher error than payoff+bridge so pure error ranking would pick bridge.
    verse['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.18
    verse['cross_parent_best_alternate']['score_breakdown']['seam_risk'] = 0.36
    verse['candidate_shortlist'][1]['error_delta_vs_selected'] = 0.18
    verse['candidate_shortlist'][1]['score_breakdown']['seam_risk'] = 0.36
    bridge['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.09
    bridge['cross_parent_best_alternate']['score_breakdown']['seam_risk'] = 0.24
    bridge['candidate_shortlist'][1]['error_delta_vs_selected'] = 0.09
    bridge['candidate_shortlist'][1]['score_breakdown']['seam_risk'] = 0.24

    # Mark payoff and verse as explicit handoff-bearing sections to increase structure/planner pressure.
    payoff['transition_mode'] = 'arrival_handoff'
    verse['transition_mode'] = 'single_owner_handoff'
    bridge['transition_mode'] = 'same_parent_flow'

    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, bridge, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}

    assert combo_labels == {'bridge', 'payoff'}


def test_build_auto_shortlist_variant_configs_prefers_contiguous_same_owner_handoff_combo_over_lower_error_split_combo():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    bridge = _make_section_with_alternate('bridge', 'A', 'phrase_3_5', 'B', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12')

    # Lower raw error on verse+payoff; contiguous bridge+payoff should still win because it
    # preserves same-owner handoff continuity across adjacent sections.
    verse['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.06
    verse['candidate_shortlist'][1]['error_delta_vs_selected'] = 0.06
    bridge['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.14
    bridge['candidate_shortlist'][1]['error_delta_vs_selected'] = 0.14
    payoff['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.12
    payoff['candidate_shortlist'][1]['error_delta_vs_selected'] = 0.12

    verse['transition_mode'] = 'same_parent_flow'
    bridge['transition_mode'] = 'single_owner_handoff'
    payoff['transition_mode'] = 'arrival_handoff'

    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, bridge, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}

    assert combo_labels == {'bridge', 'payoff'}


def test_build_auto_shortlist_variant_configs_prefers_generated_same_owner_handoff_chain_candidate_over_lower_error_split_handoff_combo():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'C', 'phrase_8_10')
    build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'C', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12')

    verse['transition_mode'] = 'single_owner_handoff'
    build['transition_mode'] = 'arrival_handoff'
    payoff['transition_mode'] = 'arrival_handoff'

    # Build primary alternate points at C with slightly lower error, but a B alternate exists in shortlist.
    build['cross_parent_best_alternate']['parent_id'] = 'C'
    build['cross_parent_best_alternate']['window_label'] = 'phrase_9_11_c'
    build['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.08
    build['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.42, 'transition_viability': 0.38})
    build['candidate_shortlist'][-1] = dict(build['cross_parent_best_alternate'])
    build['candidate_shortlist'].append(
        {
            'rank': 3,
            'parent_id': 'B',
            'window_label': 'phrase_9_11_b',
            'selected': False,
            'planner_error': 0.39,
            'error_delta_vs_selected': 0.11,
            'score_breakdown': {
                'stretch_ratio': 1.04,
                'stretch_gate': 0.0,
                'seam_risk': 0.44,
                'transition_viability': 0.40,
            },
        }
    )

    payoff['cross_parent_best_alternate']['parent_id'] = 'B'
    payoff['cross_parent_best_alternate']['window_label'] = 'phrase_10_12_b'
    payoff['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.10
    payoff['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.45, 'transition_viability': 0.41})
    payoff['candidate_shortlist'][-1] = dict(payoff['cross_parent_best_alternate'])

    # Keep verse as a viable but less relevant option so build+payoff is still the intended combo family.
    verse['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.13
    verse['candidate_shortlist'][-1] = dict(verse['cross_parent_best_alternate'])

    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, build, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}
    combo_parents = {str(swap.get('alternate_parent') or '').strip() for swap in combo.get('swaps', [])}

    assert combo_labels == {'build', 'payoff'}
    assert combo_parents == {'B'}


def test_build_auto_shortlist_variant_configs_prefers_lower_crowding_handoff_chain_candidate_within_error_budget():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'C', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12')

    verse['transition_mode'] = 'single_owner_handoff'
    build['transition_mode'] = 'arrival_handoff'
    payoff['transition_mode'] = 'arrival_handoff'

    # Primary build alternate has lower error but worse handoff crowding pressure.
    build['cross_parent_best_alternate']['parent_id'] = 'C'
    build['cross_parent_best_alternate']['window_label'] = 'phrase_9_11_c'
    build['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.07
    build['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.60, 'transition_viability': 0.57})
    build['candidate_shortlist'][-1] = dict(build['cross_parent_best_alternate'])

    # Chain candidate is slightly higher error but much less crowded and same-owner with payoff.
    build['candidate_shortlist'].append(
        {
            'rank': 3,
            'parent_id': 'B',
            'window_label': 'phrase_9_11_b_lowcrowd',
            'selected': False,
            'planner_error': 0.40,
            'error_delta_vs_selected': 0.14,
            'score_breakdown': {
                'stretch_ratio': 1.02,
                'stretch_gate': 0.0,
                'seam_risk': 0.30,
                'transition_viability': 0.28,
            },
        }
    )

    payoff['cross_parent_best_alternate']['parent_id'] = 'B'
    payoff['cross_parent_best_alternate']['window_label'] = 'phrase_10_12_b'
    payoff['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.10
    payoff['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.34, 'transition_viability': 0.33})
    payoff['candidate_shortlist'][-1] = dict(payoff['cross_parent_best_alternate'])

    # Keep verse available but less competitive than build+payoff family.
    verse['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.13
    verse['candidate_shortlist'][-1] = dict(verse['cross_parent_best_alternate'])

    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, build, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}
    build_swap = next(swap for swap in combo.get('swaps', []) if str(swap.get('section_label') or '').strip().lower() == 'build')

    assert combo_labels == {'build', 'payoff'}
    assert build_swap.get('alternate_parent') == 'B'



def test_build_auto_shortlist_variant_configs_admits_section_local_transition_relief_candidate_within_error_budget():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'C', 'phrase_8_10')
    build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'C', 'phrase_9_11_c')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'D', 'phrase_10_12_d')

    verse['transition_mode'] = 'same_parent_flow'
    build['transition_mode'] = 'arrival_handoff'
    payoff['transition_mode'] = 'single_owner_handoff'

    # Primary build alternate is lower error but crowded.
    build['cross_parent_best_alternate']['parent_id'] = 'C'
    build['cross_parent_best_alternate']['window_label'] = 'phrase_9_11_c'
    build['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.08
    build['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.66, 'transition_viability': 0.62})
    build['candidate_shortlist'][-1] = dict(build['cross_parent_best_alternate'])

    # Section-local transition-relief candidate is higher error but materially cleaner.
    build['candidate_shortlist'].append(
        {
            'rank': 3,
            'parent_id': 'D',
            'window_label': 'phrase_9_11_d_relief',
            'selected': False,
            'planner_error': 0.41,
            'error_delta_vs_selected': 0.26,
            'score_breakdown': {
                'stretch_ratio': 1.02,
                'stretch_gate': 0.0,
                'seam_risk': 0.30,
                'transition_viability': 0.27,
            },
        }
    )

    payoff['cross_parent_best_alternate']['parent_id'] = 'D'
    payoff['cross_parent_best_alternate']['window_label'] = 'phrase_10_12_d'
    payoff['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.17
    payoff['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.32, 'transition_viability': 0.29})
    payoff['candidate_shortlist'][-1] = dict(payoff['cross_parent_best_alternate'])

    # Keep verse viable but not dominant.
    verse['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.20
    verse['candidate_shortlist'][-1] = dict(verse['cross_parent_best_alternate'])

    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, build, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}
    build_swap = next(swap for swap in combo.get('swaps', []) if str(swap.get('section_label') or '').strip().lower() == 'build')

    assert combo_labels == {'build', 'payoff'}
    assert build_swap.get('alternate_parent') == 'D'



def test_build_auto_shortlist_variant_configs_prefers_two_hop_handoff_owner_bridge_over_higher_crowding_primary_parent():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'C', 'phrase_8_10_c')
    build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'A', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12_b')

    verse['transition_mode'] = 'arrival_handoff'
    build['transition_mode'] = 'same_parent_flow'
    payoff['transition_mode'] = 'single_owner_handoff'

    # Verse primary alternate is lower error but crowded (C).
    verse['cross_parent_best_alternate']['parent_id'] = 'C'
    verse['cross_parent_best_alternate']['window_label'] = 'phrase_8_10_c'
    verse['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.10
    verse['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.62, 'transition_viability': 0.60})
    verse['candidate_shortlist'][-1] = dict(verse['cross_parent_best_alternate'])

    # Two-hop handoff neighbor parent (B) is slightly higher error but much cleaner.
    verse['candidate_shortlist'].append(
        {
            'rank': 3,
            'parent_id': 'B',
            'window_label': 'phrase_8_10_b_bridge',
            'selected': False,
            'planner_error': 0.38,
            'error_delta_vs_selected': 0.22,
            'score_breakdown': {
                'stretch_ratio': 1.03,
                'stretch_gate': 0.0,
                'seam_risk': 0.22,
                'transition_viability': 0.20,
            },
        }
    )

    # Make build section ineligible so combo search focuses on verse+payoff ownership bridge.
    build['cross_parent_best_alternate']['error_delta_vs_selected'] = 1.20
    build['candidate_shortlist'][-1] = dict(build['cross_parent_best_alternate'])

    payoff['cross_parent_best_alternate']['parent_id'] = 'B'
    payoff['cross_parent_best_alternate']['window_label'] = 'phrase_10_12_b'
    payoff['cross_parent_best_alternate']['error_delta_vs_selected'] = 0.20
    payoff['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.26, 'transition_viability': 0.24})
    payoff['candidate_shortlist'][-1] = dict(payoff['cross_parent_best_alternate'])

    plan = SimpleNamespace(
        planning_diagnostics={
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, build, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    verse_swap = next(swap for swap in combo.get('swaps', []) if str(swap.get('section_label') or '').strip().lower() == 'verse')

    assert verse_swap.get('alternate_parent') == 'B'



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


def test_build_auto_shortlist_variant_configs_baseline_combo_keeps_donor_paths_when_same_parent_options_rank_higher_per_section():
    def _section_with_ranked_alternates(label: str, donor_window: str, same_parent_window: str):
        return {
            'label': label,
            'selected_parent': 'A',
            'selected_window_label': f'{label}_selected',
            'selection_rank': 1,
            'candidate_shortlist': [
                {
                    'rank': 1,
                    'parent_id': 'A',
                    'window_label': f'{label}_selected',
                    'selected': True,
                    'planner_error': 0.20,
                    'error_delta_vs_selected': 0.0,
                    'score_breakdown': {'stretch_ratio': 1.0, 'stretch_gate': 0.0, 'seam_risk': 0.2, 'transition_viability': 0.3},
                },
                {
                    'rank': 2,
                    'parent_id': 'A',
                    'window_label': same_parent_window,
                    'selected': False,
                    'planner_error': 0.24,
                    'error_delta_vs_selected': 0.09,
                    'score_breakdown': {'stretch_ratio': 1.02, 'stretch_gate': 0.0, 'seam_risk': 0.22, 'transition_viability': 0.31},
                },
                {
                    'rank': 3,
                    'parent_id': 'B',
                    'window_label': donor_window,
                    'selected': False,
                    'planner_error': 0.28,
                    'error_delta_vs_selected': 0.14,
                    'score_breakdown': {'stretch_ratio': 1.05, 'stretch_gate': 0.0, 'seam_risk': 0.25, 'transition_viability': 0.35},
                },
            ],
            'cross_parent_best_alternate': {
                'rank': 3,
                'parent_id': 'B',
                'window_label': donor_window,
                'selected': False,
                'planner_error': 0.28,
                'error_delta_vs_selected': 0.14,
                'score_breakdown': {'stretch_ratio': 1.05, 'stretch_gate': 0.0, 'seam_risk': 0.25, 'transition_viability': 0.35},
            },
        }

    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'baseline',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [
                _section_with_ranked_alternates('payoff', 'payoff_donor', 'payoff_same_parent'),
                _section_with_ranked_alternates('build', 'build_donor', 'build_same_parent'),
            ],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_parents = {str(swap.get('alternate_parent') or '') for swap in combo.get('swaps', [])}

    assert 'B' in combo_parents


def test_build_auto_shortlist_variant_configs_baseline_combo_prefers_core_shape_over_intro_donor_fallback_when_core_donor_is_unavailable():
    intro_donor = _make_section_with_alternate('intro', 'A', 'phrase_0_2', 'B', 'phrase_6_8')
    verse_same_parent = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'A', 'phrase_8_10')
    payoff_same_parent = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'A', 'phrase_10_12')

    # Remove core donor opportunities so baseline has only intro donor support.
    verse_same_parent['cross_parent_best_alternate'] = None
    payoff_same_parent['cross_parent_best_alternate'] = None

    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'baseline',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [intro_donor, verse_same_parent, payoff_same_parent],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    combo = next(config for config in configs if config['strategy'] == 'dual_section_alternate')
    combo_labels = {str(swap.get('section_label') or '').strip().lower() for swap in combo.get('swaps', [])}
    combo_parents = {str(swap.get('alternate_parent') or '') for swap in combo.get('swaps', [])}

    assert 'intro' not in combo_labels
    assert combo_labels == {'verse', 'payoff'}
    assert combo_parents == {'A'}


def test_build_auto_shortlist_variant_configs_baseline_reserves_support_variant_for_relaxed_core_donor_overlay():
    intro_donor = _make_section_with_alternate('intro', 'A', 'phrase_0_2', 'B', 'phrase_6_8')
    verse_same_parent = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'A', 'phrase_8_10')
    payoff_same_parent = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'A', 'phrase_10_12')

    relaxed_payoff_donor = {
        'rank': 3,
        'parent_id': 'B',
        'window_label': 'phrase_5_7',
        'selected': False,
        'planner_error': 1.12,
        'error_delta_vs_selected': 2.2,
        'score_breakdown': {
            'stretch_ratio': 1.06,
            'stretch_gate': 0.0,
            'seam_risk': 0.34,
            'transition_viability': 0.41,
        },
    }
    payoff_same_parent['cross_parent_best_alternate'] = dict(relaxed_payoff_donor)
    payoff_same_parent['candidate_shortlist'].append(dict(relaxed_payoff_donor))

    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'baseline',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [intro_donor, verse_same_parent, payoff_same_parent],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')

    support = next(config for config in configs if config['strategy'] == 'single_section_support')
    assert support['swaps'] == []
    assert len(support['supports']) == 1
    support_payload = support['supports'][0]
    assert str(support_payload.get('section_label') or '').strip().lower() == 'payoff'
    assert support_payload.get('support_parent') == 'B'


def test_build_auto_shortlist_variant_configs_adaptive_reserves_support_variant_for_integration():
    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'adaptive',
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
    assert any(config['strategy'] in {'single_section_support', 'dual_section_support'} for config in configs)
    assert any(config['strategy'] == 'dual_section_alternate' for config in configs)


def test_build_auto_shortlist_variant_configs_baseline_keeps_support_variant_even_with_core_donor_swaps():
    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'baseline',
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

    assert len(configs) == 3
    assert any(config['strategy'] in {'single_section_support', 'dual_section_support'} for config in configs)
    assert any(config['strategy'] == 'dual_section_alternate' for config in configs)


def test_build_auto_shortlist_variant_configs_adaptive_dual_support_avoids_extreme_risk_payoff_pair():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'B', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12')

    verse['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.22, 'transition_viability': 0.25})
    build['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.24, 'transition_viability': 0.28})
    payoff['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.96, 'transition_viability': 0.94})

    for sec in (verse, build, payoff):
        sec['candidate_shortlist'][-1] = dict(sec['cross_parent_best_alternate'])

    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'adaptive',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, build, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    dual_support = next(config for config in configs if config['strategy'] == 'dual_section_support')
    labels = {str(item.get('section_label') or '').strip().lower() for item in dual_support['supports']}

    assert labels == {'verse', 'build'}


def test_build_auto_shortlist_variant_configs_adaptive_dual_support_prioritizes_handoff_sections():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'B', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12')

    verse['transition_mode'] = 'same_parent_flow'
    build['transition_mode'] = 'arrival_handoff'
    payoff['transition_mode'] = 'same_parent_flow'

    verse['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.18, 'transition_viability': 0.22})
    build['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.74, 'transition_viability': 0.70})
    payoff['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.22, 'transition_viability': 0.26})

    for sec in (verse, build, payoff):
        sec['candidate_shortlist'][-1] = dict(sec['cross_parent_best_alternate'])

    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'adaptive',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, build, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    dual_support = next(config for config in configs if config['strategy'] == 'dual_section_support')
    labels = {str(item.get('section_label') or '').strip().lower() for item in dual_support['supports']}

    assert 'build' in labels


def test_build_auto_shortlist_variant_configs_adaptive_dual_support_avoids_mixed_handoff_chain_for_contiguous_build_payoff():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'B', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12')

    # Keep all risks in-range so pairing choice is driven by chain smoothness.
    verse['transition_mode'] = 'same_parent_flow'
    build['transition_mode'] = 'single_owner_handoff'
    payoff['transition_mode'] = 'same_parent_flow'

    verse['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.22, 'transition_viability': 0.26})
    build['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.24, 'transition_viability': 0.30})
    payoff['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.23, 'transition_viability': 0.27})

    for sec in (verse, build, payoff):
        sec['candidate_shortlist'][-1] = dict(sec['cross_parent_best_alternate'])

    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'adaptive',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, build, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    dual_support = next(config for config in configs if config['strategy'] == 'dual_section_support')
    labels = [str(item.get('section_label') or '').strip().lower() for item in dual_support['supports']]

    assert set(labels) != {'build', 'payoff'}


def test_build_auto_shortlist_variant_configs_adaptive_dual_support_penalizes_high_risk_mixed_handoff_chain():
    verse = _make_section_with_alternate('verse', 'A', 'phrase_2_4', 'B', 'phrase_8_10')
    build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'B', 'phrase_9_11')
    payoff = _make_section_with_alternate('payoff', 'A', 'phrase_4_6', 'B', 'phrase_10_12')

    build['transition_mode'] = 'single_owner_handoff'
    payoff['transition_mode'] = 'same_parent_flow'

    # Keep the build/payoff pair viable but above the high-risk mismatch threshold.
    build['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.58, 'transition_viability': 0.58})
    payoff['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.53, 'transition_viability': 0.53})
    verse['cross_parent_best_alternate']['score_breakdown'].update({'seam_risk': 0.28, 'transition_viability': 0.29})

    for sec in (verse, build, payoff):
        sec['candidate_shortlist'][-1] = dict(sec['cross_parent_best_alternate'])

    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'adaptive',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [verse, build, payoff],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    dual_support = next(config for config in configs if config['strategy'] == 'dual_section_support')
    labels = {str(item.get('section_label') or '').strip().lower() for item in dual_support['supports']}

    assert labels != {'build', 'payoff'}


def test_build_auto_shortlist_variant_configs_support_policy_adapts_to_transition_risk():
    def _build_plan_with_risk(*, seam_risk: float, transition_viability: float):
        build = _make_section_with_alternate('build', 'A', 'phrase_3_5', 'B', 'phrase_9_11')
        build['cross_parent_best_alternate'] = {
            'rank': 2,
            'parent_id': 'B',
            'window_label': 'phrase_9_11',
            'selected': False,
            'planner_error': 0.31,
            'error_delta_vs_selected': 0.12,
            'score_breakdown': {
                'stretch_ratio': 1.02,
                'stretch_gate': 0.0,
                'seam_risk': seam_risk,
                'transition_viability': transition_viability,
            },
        }
        build['candidate_shortlist'][-1] = dict(build['cross_parent_best_alternate'])
        return SimpleNamespace(
            planning_diagnostics={
                'arrangement_mode': 'baseline',
                'backbone_plan': {'backbone_parent': 'A'},
                'selected_sections': [build],
            },
            sections=[],
            planning_notes=[],
        )

    high_risk_configs = ai_dj._build_auto_shortlist_variant_configs(
        _build_plan_with_risk(seam_risk=0.84, transition_viability=0.78),
        batch_size=3,
        variant_mode='safe',
    )
    high_risk_support = next(config for config in high_risk_configs if config['strategy'] == 'single_section_support')['supports'][0]

    low_risk_configs = ai_dj._build_auto_shortlist_variant_configs(
        _build_plan_with_risk(seam_risk=0.16, transition_viability=0.22),
        batch_size=3,
        variant_mode='safe',
    )
    low_risk_support = next(config for config in low_risk_configs if config['strategy'] == 'single_section_support')['supports'][0]

    assert high_risk_support['support_policy']['risk'] > low_risk_support['support_policy']['risk']
    assert high_risk_support['support_mode'] == 'filtered_counterlayer'
    assert low_risk_support['support_mode'] == 'foreground_counterlayer'
    assert high_risk_support['support_gain_db'] < low_risk_support['support_gain_db']
    assert high_risk_support['support_policy']['transition_viability'] < low_risk_support['support_policy']['transition_viability']
    assert high_risk_support['support_policy']['foreground_collision_risk'] > low_risk_support['support_policy']['foreground_collision_risk']
    assert high_risk_support['support_policy']['transition_error'] > low_risk_support['support_policy']['transition_error']


def test_build_auto_shortlist_variant_configs_adaptive_synthesizes_counterparent_support_when_core_options_are_same_parent_only():
    plan = SimpleNamespace(
        planning_diagnostics={
            'arrangement_mode': 'adaptive',
            'backbone_plan': {'backbone_parent': 'A'},
            'selected_sections': [
                _make_section_with_alternate('intro', 'A', 'phrase_1_3', 'B', 'phrase_1_3'),
                _make_section_with_alternate('verse', 'A', 'phrase_3_5', 'A', 'phrase_2_4'),
                _make_section_with_alternate('build', 'B', 'phrase_5_7', 'B', 'phrase_4_6'),
                _make_section_with_alternate('payoff', 'B', 'phrase_7_11', 'B', 'phrase_6_10'),
            ],
        },
        sections=[],
        planning_notes=[],
    )

    configs = ai_dj._build_auto_shortlist_variant_configs(plan, batch_size=3, variant_mode='safe')
    support = next(config for config in configs if config['strategy'] in {'single_section_support', 'dual_section_support'})
    assert len(support['supports']) >= 1

    for payload in support['supports']:
        sec_idx = int(payload['section_index'])
        selected_section = plan.planning_diagnostics['selected_sections'][sec_idx]
        assert payload['kind'] == 'support_overlay_counterparent'
        assert payload['support_parent'] == 'A'
        assert payload['support_parent'] != selected_section['selected_parent']
        assert payload['support_section_label'] == selected_section['selected_window_label']


def test_apply_auto_shortlist_variant_applies_support_overlay_to_section_and_diagnostics():
    plan = SimpleNamespace(
        sections=[
            SimpleNamespace(
                label='payoff',
                source_parent='A',
                source_section_label='phrase_4_6',
                support_parent=None,
                support_section_label=None,
                support_gain_db=None,
                support_mode=None,
            )
        ],
        planning_diagnostics={
            'selected_sections': [
                {
                    'label': 'payoff',
                    'selected_parent': 'A',
                    'selected_window_label': 'phrase_4_6',
                }
            ]
        },
        planning_notes=[],
    )

    variant = {
        'variant_id': 'support_01_payoff_B',
        'label': 'payoff + B support',
        'strategy': 'single_section_support',
        'variant_mode': 'safe',
        'swaps': [],
        'supports': [
            {
                'section_index': 0,
                'section_label': 'payoff',
                'support_parent': 'B',
                'support_section_label': 'phrase_5_7',
                'support_gain_db': -10.5,
                'support_mode': 'filtered_counterlayer',
                'kind': 'support_overlay',
                'error_delta': 2.2,
                'support_policy': {
                    'risk': 0.68,
                    'foreground_collision_risk': 0.41,
                    'transition_viability': 0.57,
                },
            }
        ],
    }

    updated = ai_dj._apply_auto_shortlist_variant(plan, variant)

    section = updated.sections[0]
    assert section.support_parent == 'B'
    assert section.support_section_label == 'phrase_5_7'
    assert section.support_gain_db == -10.5
    assert section.support_mode == 'filtered_counterlayer'
    assert section.support_transition_risk == 0.68
    assert section.support_foreground_collision_risk == 0.41
    assert section.support_transition_viability == 0.57

    diag = updated.planning_diagnostics['selected_sections'][0]
    assert diag['support_recipe']['parent_id'] == 'B'
    assert diag['support_recipe']['window_label'] == 'phrase_5_7'
    assert diag['support_recipe']['mode'] == 'filtered_counterlayer'
    assert isinstance(diag['support_recipe']['policy'], dict)
