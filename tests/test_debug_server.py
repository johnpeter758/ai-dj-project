import json
from pathlib import Path
from types import SimpleNamespace

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

    listener_agent = runs_dir / 'listener_agent_latest.json'
    listener_agent.write_text(json.dumps({
        'listener_agent': {'purpose': 'Reject weak outputs'},
        'summary': ['Listener agent kept 1 of 2 candidates for human review.'],
        'recommended_for_human_review': [
            {'label': 'fusion_case', 'listener_rank': 81.0, 'overall_score': 78.0, 'verdict': 'promising'}
        ],
        'rejected': [
            {'label': 'baseline_case', 'hard_fail_reasons': ['does not sound like one real song']}
        ],
    }), encoding='utf-8')

    active_sprint = runs_dir / 'active_sprint_status.json'
    active_sprint.write_text(json.dumps({
        'active': True,
        'task': 'Close the listen/eval loop in UI',
        'started_at': '2026-03-15 11:00 EDT',
        'last_heartbeat': '2026-03-15 11:05 EDT',
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
    assert payload['latest_benchmark_listen_result']['ranking_preview'][0]['label'] == 'fusion_case'
    assert payload['latest_benchmark_listen_result']['ranking_preview'][1]['label'] == 'baseline_case'
    assert payload['latest_listener_agent_result']['recommended_count'] == 1
    assert payload['latest_listener_agent_result']['winner'] == 'fusion_case'
    assert payload['latest_listener_agent_result']['top_reject_reason'] == 'does not sound like one real song'
    assert payload['active_sprint']['status'] == 'active'
    assert payload['active_sprint']['task'] == 'Close the listen/eval loop in UI'
    assert payload['workloop_visualization']['progress_percent'] == 86
    assert payload['workloop_visualization']['current_stage']['key'] == 'code'
    assert payload['workloop_visualization']['stages'][0]['key'] == 'code'
    assert payload['latest_manifest']['diagnostics']['warning_count'] == 1
    assert payload['latest_manifest']['diagnostics']['overlap_section_count'] == 1
    assert payload['latest_manifest']['diagnostics']['stretch_risk_count'] == 1
    assert payload['latest_run_summary']['run_dir'] == 'fusion_case'
    assert payload['latest_run_summary']['manifest']['name'] == 'render_manifest.json'


def test_listener_agent_api_runs_on_recent_render_outputs(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    first = runs_dir / 'fusion_a'
    second = runs_dir / 'fusion_b'
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / 'child_master.wav').write_bytes(b'fake')
    (second / 'child_master.wav').write_bytes(b'fake')
    (first / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(first / 'child_master.wav')}}), encoding='utf-8')
    (second / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(second / 'child_master.wav')}}), encoding='utf-8')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        out_index = cmd.index('--output') + 1
        out_path = Path(cmd[out_index])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            'listener_agent': {'purpose': 'Reject non-song outputs'},
            'summary': ['Listener agent kept 1 of 2 candidates for human review.'],
            'recommended_for_human_review': [
                {'label': 'fusion_a', 'listener_rank': 80.0, 'overall_score': 76.0, 'verdict': 'promising'}
            ],
            'rejected': [
                {'label': 'fusion_b', 'hard_fail_reasons': ['transitions still read like track switching']}
            ],
        }), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post('/api/listener-agent', json={'shortlist': 1})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['input_count'] == 2
    assert payload['shortlist'] == 1
    assert payload['report']['recommended_for_human_review'][0]['label'] == 'fusion_a'
    assert payload['report']['rejected'][0]['hard_fail_reasons'][0] == 'transitions still read like track switching'


def test_listener_agent_api_uses_explicit_run_scoped_inputs(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    first = runs_dir / 'fusion_a'
    second = runs_dir / 'fusion_b'
    ignored = runs_dir / 'fusion_c'
    for path in (first, second, ignored):
        path.mkdir(parents=True)
        (path / 'child_master.wav').write_bytes(b'fake')
        (path / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(path / 'child_master.wav')}}), encoding='utf-8')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    seen_cmd: list[str] = []

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        seen_cmd[:] = cmd
        out_index = cmd.index('--output') + 1
        out_path = Path(cmd[out_index])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            'listener_agent': {'purpose': 'Reject non-song outputs'},
            'summary': ['Listener agent kept 1 of 2 explicit candidates for human review.'],
            'recommended_for_human_review': [
                {'label': 'fusion_b', 'listener_rank': 82.0, 'overall_score': 78.0, 'verdict': 'promising'}
            ],
            'rejected': [
                {'label': 'fusion_a', 'hard_fail_reasons': ['intro still reads like track switching']}
            ],
        }), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post('/api/listener-agent', json={
        'inputs': [str(second), str(first)],
        'shortlist': 2,
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['inputs'] == [str(second.resolve()), str(first.resolve())]
    input_index = seen_cmd.index('listener-agent') + 1
    shortlist_index = seen_cmd.index('--shortlist')
    assert seen_cmd[input_index:shortlist_index] == [str(second.resolve()), str(first.resolve())]
    assert str(ignored.resolve()) not in seen_cmd
    assert payload['report']['recommended_for_human_review'][0]['label'] == 'fusion_b'


def test_listener_agent_api_rejects_non_numeric_shortlist(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir(parents=True)
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)

    client = app.test_client()
    response = client.post('/api/listener-agent', json={'shortlist': 'top-two'})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload['status'] == 'error'
    assert 'shortlist must be an integer' in payload['error']


def test_closed_loop_api_runs_bounded_loop_and_returns_report(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    music_dir = tmp_path / 'music'
    runs_dir.mkdir(parents=True)
    music_dir.mkdir(parents=True)

    song_a = music_dir / 'song_a.wav'
    song_b = music_dir / 'song_b.wav'
    reference = runs_dir / 'ref.wav'
    song_a.write_bytes(b'a')
    song_b.write_bytes(b'b')
    reference.write_bytes(b'r')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'MUSIC_DIR', music_dir)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        assert 'closed-loop' in cmd
        out_index = cmd.index('--output') + 1
        out_dir = Path(cmd[out_index])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'closed_loop_report.json').write_text(json.dumps({
            'best_iteration': {'iteration': 2, 'candidate_overall_score': 81.0, 'candidate_verdict': 'promising'},
            'stop_reason': 'plateau:1',
            'summary': ['Best iteration was 2 with overall score 81.0.'],
        }), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='closed-loop ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post('/api/closed-loop', json={
        'song_a': str(song_a),
        'song_b': str(song_b),
        'references': [str(reference)],
        'max_iterations': 2,
        'plateau_limit': 1,
        'quality_gate': 88,
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['song_a'] == str(song_a.resolve())
    assert payload['song_b'] == str(song_b.resolve())
    assert payload['references'] == [str(reference.resolve())]
    assert payload['config']['max_iterations'] == 2
    assert payload['config']['plateau_limit'] == 1
    assert payload['report']['best_iteration']['iteration'] == 2
    assert payload['report']['stop_reason'] == 'plateau:1'


def test_closed_loop_api_rejects_paths_outside_music_or_runs(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    music_dir = tmp_path / 'music'
    runs_dir.mkdir(parents=True)
    music_dir.mkdir(parents=True)
    reference = runs_dir / 'ref.wav'
    reference.write_bytes(b'r')
    outsider = tmp_path / 'elsewhere.wav'
    outsider.write_bytes(b'x')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'MUSIC_DIR', music_dir)

    client = app.test_client()
    response = client.post('/api/closed-loop', json={
        'song_a': str(outsider),
        'song_b': str(outsider),
        'references': [str(reference)],
    })

    assert response.status_code == 403
    payload = response.get_json()
    assert payload['status'] == 'error'
    assert 'must stay inside music/ or runs/' in payload['error']
