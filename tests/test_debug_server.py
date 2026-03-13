import json
from pathlib import Path

import server
from server import app


def test_health_route():
    client = app.test_client()
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'healthy'


def test_index_route_serves_main_ui():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'VocalFusion' in response.data
    assert b'Latest benchmark-listen' in response.data


def test_status_includes_listen_compare_benchmark_and_manifest_summaries(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    run_dir = runs_dir / 'fusion_case'
    run_dir.mkdir(parents=True)

    audio = run_dir / 'child_master.wav'
    audio.write_bytes(b'fake')
    manifest = run_dir / 'render_manifest.json'
    manifest.write_text(json.dumps({
        'outputs': {'master_wav': str(audio)},
        'sections': [
            {'index': 0, 'allowed_overlap': True, 'stretch_ratio': 1.32},
            {'index': 1, 'allowed_overlap': False, 'stretch_ratio': 1.0},
        ],
        'work_orders': [{'section_index': 0}, {'section_index': 1}],
        'warnings': ['warning 1'],
        'fallbacks': ['fallback 1'],
    }), encoding='utf-8')

    listen = runs_dir / 'listen_latest.json'
    listen.write_text(json.dumps({
        'overall_score': 78.0,
        'verdict': 'promising',
        'top_reasons': ['Strong groove'],
        'top_fixes': ['Tighten transitions'],
        'source_path': str(audio),
    }), encoding='utf-8')

    compare = runs_dir / 'listen_compare_latest.json'
    compare.write_text(json.dumps({
        'summary': 'Left wins on groove and mix.',
        'winner': {'overall': 'left', 'components': {'groove': 'left'}},
        'deltas': {'overall_score_delta': 6.0, 'component_score_deltas': {'groove': 8.0, 'mix_sanity': 2.0}},
    }), encoding='utf-8')

    benchmark = runs_dir / 'listen_benchmark_latest.json'
    benchmark.write_text(json.dumps({
        'winner': 'fusion_case',
        'ranking': [
            {
                'label': 'fusion_case',
                'wins': 2,
                'ties': 0,
                'losses': 0,
                'net_score_delta': 15.0,
                'overall_score': 78.0,
                'verdict': 'promising',
            },
            {
                'label': 'baseline_case',
                'wins': 1,
                'ties': 0,
                'losses': 1,
                'net_score_delta': 2.0,
                'overall_score': 72.0,
                'verdict': 'mixed',
            },
        ],
        'comparisons': [
            {'left': 'fusion_case', 'right': 'baseline_case', 'winner': 'left', 'overall_score_delta': 6.0, 'decision': {}},
        ],
    }), encoding='utf-8')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, '_extract_current_task', lambda: {'summary': 'Evaluator surfacing', 'source': None, 'details': []})
    monkeypatch.setattr(server, '_latest_commit', lambda: {})
    monkeypatch.setattr(server, '_changed_files', lambda: [])

    payload = server._workloop_status()

    assert payload['latest_evaluator_result']['overall_score'] == 78.0
    assert payload['latest_compare_listen_result']['overall_winner'] == 'left'
    assert payload['latest_compare_listen_result']['biggest_component_delta']['component'] == 'groove'
    assert payload['latest_benchmark_listen_result']['winner'] == 'fusion_case'
    assert payload['latest_benchmark_listen_result']['top_entry']['wins'] == 2
    assert payload['latest_benchmark_listen_result']['leader_gap'] == 13.0
    assert payload['latest_manifest']['diagnostics']['warning_count'] == 1
    assert payload['latest_manifest']['diagnostics']['overlap_section_count'] == 1
    assert payload['latest_manifest']['diagnostics']['stretch_risk_count'] == 1
    assert payload['latest_run_summary']['run_dir'] == 'fusion_case'
    assert payload['latest_run_summary']['manifest']['name'] == 'render_manifest.json'
