import io
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

    closed_loop_dir = runs_dir / 'closed_loop_case'
    closed_loop_dir.mkdir(parents=True)
    (closed_loop_dir / 'closed_loop_report.json').write_text(json.dumps({
        'best_iteration': {
            'iteration': 2,
            'candidate_overall_score': 81.0,
            'candidate_verdict': 'promising',
            'candidate_listener_decision': 'survivor',
            'candidate_input': str(closed_loop_dir / 'iteration_002'),
        },
        'stop_reason': 'plateau:1',
        'summary': ['Best iteration was 2 with overall score 81.0.'],
        'loop_summary': {
            'total_iterations': 2,
            'net_improvement': 5.0,
            'score_trajectory': [76.0, 81.0],
            'best_iteration': {
                'iteration': 2,
                'candidate_overall_score': 81.0,
                'candidate_verdict': 'promising',
                'candidate_listener_decision': 'survivor',
                'candidate_input': str(closed_loop_dir / 'iteration_002'),
            },
            'best_top_intervention': {
                'component': 'song_likeness',
                'problem': 'Opening still reads like track switching.',
            },
            'iteration_summaries': [
                {
                    'iteration': 1,
                    'candidate_overall_score': 76.0,
                    'candidate_verdict': 'mixed',
                    'candidate_listener_decision': 'borderline',
                    'gap_to_target': 12.0,
                    'plateau_count': 0,
                    'render_dir': str(closed_loop_dir / 'iteration_001'),
                },
                {
                    'iteration': 2,
                    'candidate_overall_score': 81.0,
                    'candidate_verdict': 'promising',
                    'candidate_listener_decision': 'survivor',
                    'gap_to_target': 7.0,
                    'plateau_count': 1,
                    'render_dir': str(closed_loop_dir / 'iteration_002'),
                },
            ],
        },
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
    assert payload['latest_closed_loop_result']['stop_reason'] == 'plateau:1'
    assert payload['latest_closed_loop_result']['total_iterations'] == 2
    assert payload['latest_closed_loop_result']['net_improvement'] == 5.0
    assert payload['latest_closed_loop_result']['best_iteration']['iteration'] == 2
    assert payload['latest_closed_loop_result']['latest_iteration']['candidate_listener_decision'] == 'survivor'
    assert payload['latest_closed_loop_result']['best_top_intervention']['component'] == 'song_likeness'
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


def test_status_page_mentions_closed_loop_summary_card():
    client = app.test_client()
    response = client.get('/status')
    assert response.status_code == 200
    assert b'Latest closed-loop' in response.data


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


def test_benchmark_spec_api_builds_and_persists_spec_from_runs_inputs(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir(parents=True)
    strong = runs_dir / 'strong.json'
    weak = runs_dir / 'weak.json'
    strong.write_text('{}', encoding='utf-8')
    weak.write_text('{}', encoding='utf-8')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)

    client = app.test_client()
    response = client.post('/api/benchmark-spec', json={
        'good_cases': [{'label': 'strong_case', 'path': str(strong)}],
        'bad_cases': [{'label': 'weak_case', 'path': str(weak)}],
        'expected_order': ['strong_case', 'weak_case'],
        'overall_at_least': ['strong_case:overall_score_at_least=80'],
        'overall_at_most': ['weak_case:overall_score_at_most=70'],
        'better_than': ['strong_case>weak_case:overall=10'],
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['case_count'] == 2
    assert payload['expected_order'] == ['strong_case', 'weak_case']
    assert payload['report']['cases'][0]['curation_tier'] == 'good'
    assert payload['report']['cases'][1]['curation_tier'] == 'bad'
    assert payload['report']['cases'][0]['expect']['gating_status'] == 'pass'
    assert payload['report']['cases'][1]['expect']['gating_status'] == 'reject'
    assert payload['report']['cases'][0]['expect']['better_than'][0]['overall_score_delta_at_least'] == 10.0
    report_path = Path(payload['report_path'])
    assert report_path.exists()
    assert report_path.parent == runs_dir / 'benchmark_specs'


def test_benchmark_spec_api_rejects_paths_outside_runs(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    outsider = tmp_path / 'elsewhere.json'
    runs_dir.mkdir(parents=True)
    outsider.write_text('{}', encoding='utf-8')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)

    client = app.test_client()
    response = client.post('/api/benchmark-spec', json={
        'cases': [{'label': 'outside_case', 'path': str(outsider)}],
        'good_cases': [{'label': 'inside_case', 'path': str(outsider)}],
    })

    assert response.status_code == 403
    payload = response.get_json()
    assert payload['status'] == 'error'
    assert 'must stay inside runs/' in payload['error']


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


def test_closed_loop_api_flattens_reference_bundle_before_cli(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    music_dir = tmp_path / 'music'
    runs_dir.mkdir(parents=True)
    music_dir.mkdir(parents=True)

    song_a = music_dir / 'song_a.wav'
    song_b = music_dir / 'song_b.wav'
    render_a = runs_dir / 'fusion_a'
    render_b = runs_dir / 'fusion_b'
    render_a.mkdir()
    render_b.mkdir()
    bundle = runs_dir / 'listener_agent_report.json'
    song_a.write_bytes(b'a')
    song_b.write_bytes(b'b')
    bundle.write_text(json.dumps({
        'recommended_for_human_review': [{'input_path': str(render_a)}],
        'borderline': [{'input_path': str(render_b)}],
        'rejected': [{'input_path': str(render_a)}],
    }), encoding='utf-8')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'MUSIC_DIR', music_dir)
    seen_cmd: list[str] = []

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        seen_cmd[:] = cmd
        out_index = cmd.index('--output') + 1
        out_dir = Path(cmd[out_index])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'closed_loop_report.json').write_text(json.dumps({'best_iteration': None, 'stop_reason': 'max_iterations:1', 'summary': []}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='closed-loop ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post('/api/closed-loop', json={
        'song_a': str(song_a),
        'song_b': str(song_b),
        'references': [str(bundle)],
        'max_iterations': 1,
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['references'] == [str(render_a.resolve()), str(render_b.resolve())]
    reference_args = seen_cmd[seen_cmd.index(str(song_b.resolve())) + 1:seen_cmd.index('--output')]
    assert reference_args == [str(render_a.resolve()), str(render_b.resolve())]



def test_closed_loop_api_rejects_reference_bundle_entries_outside_music_or_runs(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    music_dir = tmp_path / 'music'
    runs_dir.mkdir(parents=True)
    music_dir.mkdir(parents=True)

    song_a = music_dir / 'song_a.wav'
    song_b = music_dir / 'song_b.wav'
    outsider = tmp_path / 'elsewhere' / 'bad_ref.wav'
    outsider.parent.mkdir(parents=True)
    bundle = runs_dir / 'listener_agent_report.json'
    song_a.write_bytes(b'a')
    song_b.write_bytes(b'b')
    outsider.write_bytes(b'x')
    bundle.write_text(json.dumps({'recommended_for_human_review': [{'input_path': str(outsider)}]}), encoding='utf-8')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'MUSIC_DIR', music_dir)

    client = app.test_client()
    response = client.post('/api/closed-loop', json={
        'song_a': str(song_a),
        'song_b': str(song_b),
        'references': [str(bundle)],
    })

    assert response.status_code == 403
    payload = response.get_json()
    assert payload['status'] == 'error'
    assert 'must stay inside music/ or runs/' in payload['error']



def test_compare_listen_api_runs_with_two_render_dirs(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    left = runs_dir / 'fusion_left'
    right = runs_dir / 'fusion_right'
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    (left / 'render_manifest.json').write_text(json.dumps({'outputs': {}}), encoding='utf-8')
    (right / 'render_manifest.json').write_text(json.dumps({'outputs': {}}), encoding='utf-8')

    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        out_index = cmd.index('--output') + 1
        out_path = Path(cmd[out_index])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            'summary': 'left wins',
            'winner': {'overall': 'left'},
            'deltas': {'overall_score_delta': 3.0, 'component_score_deltas': {'groove': 5.0}},
        }), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)
    client = app.test_client()
    response = client.post('/api/compare-listen', json={'left': str(left), 'right': str(right)})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['report']['winner']['overall'] == 'left'
    assert payload['report']['deltas']['component_score_deltas']['groove'] == 5.0



def test_compare_listen_api_rejects_paths_outside_runs(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    outsider = tmp_path / 'outsider'
    runs_dir.mkdir(parents=True)
    outsider.mkdir(parents=True)
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    client = app.test_client()
    response = client.post('/api/compare-listen', json={'left': str(outsider), 'right': str(outsider)})
    assert response.status_code == 403
    assert 'inside runs/' in response.get_json()['error']



def test_human_feedback_render_and_pairwise_persist(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    data_dir = tmp_path / 'data' / 'human_feedback'
    run_dir = runs_dir / 'fusion_case'
    other_dir = runs_dir / 'fusion_other'
    run_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    (run_dir / 'child_master.mp3').write_bytes(b'fake')
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'HUMAN_FEEDBACK_DIR', data_dir)

    client = app.test_client()
    render_response = client.post('/api/human-feedback/render', json={
        'run_dir': str(run_dir),
        'overall_label': 'reject',
        'tags': ['bad groove', 'not one song'],
        'note': 'still stitched',
        'timestamp_sec': 31.2,
    })
    assert render_response.status_code == 200
    render_payload = render_response.get_json()
    assert render_payload['status'] == 'success'
    assert (run_dir / 'human_feedback.json').exists()

    pairwise_response = client.post('/api/human-feedback/pairwise', json={
        'left_run_dir': str(run_dir),
        'right_run_dir': str(other_dir),
        'winner': 'right',
        'tags': ['better groove'],
    })
    assert pairwise_response.status_code == 200
    pairwise_payload = pairwise_response.get_json()
    assert pairwise_payload['feedback']['winner'] == 'right'

    list_response = client.get('/api/human-feedback')
    assert list_response.status_code == 200
    list_payload = list_response.get_json()
    assert list_payload['count'] == 2
    assert list_payload['summary']['type_counts']['render'] == 1
    assert list_payload['summary']['type_counts']['pairwise'] == 1



def test_human_feedback_learning_endpoint_returns_derived_priors(tmp_path, monkeypatch):
    feedback_dir = tmp_path / 'data' / 'human_feedback'
    feedback_dir.mkdir(parents=True)
    events = feedback_dir / 'events.jsonl'
    events.write_text(
        '\n'.join([
            json.dumps({'type': 'render', 'overall_label': 'reject', 'tags': ['bad groove', 'not one song'], 'run_dir': 'runs/a', 'note': 'bad', 'timestamp_sec': 12.5}),
            json.dumps({'type': 'render', 'overall_label': 'promising', 'tags': ['good backbone'], 'run_dir': 'runs/b'}),
            json.dumps({'type': 'pairwise', 'winner': 'right', 'tags': ['track-switch seam']}),
        ]) + '\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(server, 'HUMAN_FEEDBACK_DIR', feedback_dir)
    client = app.test_client()
    response = client.get('/api/human-feedback/learning')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    learning = payload['learning']
    assert learning['summary']['render_event_count'] == 2
    assert learning['derived_priors']['groove_rejection_pressure'] > 0.0
    assert learning['derived_priors']['medley_rejection_pressure'] > 0.0

    distill = client.post('/api/human-feedback/learning/distill', json={})
    assert distill.status_code == 200
    distill_payload = distill.get_json()
    assert Path(distill_payload['output_path']).exists()



def test_run_diagnostics_returns_worst_moments(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    run_dir = runs_dir / 'fusion_case'
    run_dir.mkdir(parents=True)
    (run_dir / 'render_manifest.json').write_text(json.dumps({
        'sections': [
            {'index': 0, 'label': 'intro', 'target': {'start_sec': 0.0, 'end_sec': 8.0}},
            {'index': 1, 'label': 'verse', 'target': {'start_sec': 8.0, 'end_sec': 16.0}},
        ],
        'outputs': {},
    }), encoding='utf-8')
    (run_dir / 'listen_report.json').write_text(json.dumps({
        'overall_score': 60.0,
        'transition': {
            'details': {
                'worst_moments': [
                    {'kind': 'boundary_transition', 'center_time': 12.0, 'start_time': 9.0, 'end_time': 15.0, 'severity': 0.88, 'summary': 'bad seam'}
                ]
            }
        },
        'mix_sanity': {
            'details': {
                'manifest_metrics': {
                    'risky_sections': [{'section_index': 1, 'label': 'verse', 'risk': 0.55}]
                }
            }
        }
    }), encoding='utf-8')
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    client = app.test_client()
    response = client.get('/api/run-diagnostics', query_string={'run_dir': str(run_dir)})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert len(payload['worst_moments']) >= 1
    assert payload['worst_moments'][0]['severity'] >= 0.55



def test_auto_shortlist_fusion_api_returns_survivors(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    uploads_dir = runs_dir / 'ui_uploads'
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'UPLOADS_DIR', uploads_dir)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        outdir = Path(cmd[cmd.index('--output') + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        survivor_dir = outdir / 'candidate_001'
        survivor_dir.mkdir(parents=True, exist_ok=True)
        audio = survivor_dir / 'child_master.wav'
        audio.write_bytes(b'fake')
        (survivor_dir / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(audio)}}), encoding='utf-8')
        report = {
            'summary': ['Generated 4 candidates.', 'Shortlisted 1 survivor for human review.'],
            'recommended_shortlist': [
                {
                    'candidate_id': 'candidate_001',
                    'label': 'candidate_001',
                    'decision': 'survivor',
                    'listener_rank': 80.0,
                    'overall_score': 78.0,
                    'verdict': 'promising',
                    'run_dir': str(survivor_dir),
                    'audio_path': str(audio),
                    'top_reasons': ['good output'],
                    'top_fixes': [],
                }
            ],
            'closest_misses': [],
            'listener_agent_report': {'counts': {'survivors': 1, 'borderline': 0, 'rejected': 3}},
            'candidates': [],
            'pairwise_pool': {'winner': 'candidate_001'},
            'pruning': {'enabled': True, 'deleted_candidate_ids': ['candidate_002'], 'deleted_candidate_count': 1},
        }
        (outdir / 'auto_shortlist_report.json').write_text(json.dumps(report), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)
    client = app.test_client()
    response = client.post(
        '/api/auto-shortlist-fusion',
        data={
            'song_a': (io.BytesIO(b'aaa'), 'song_a.mp3'),
            'song_b': (io.BytesIO(b'bbb'), 'song_b.mp3'),
            'batch_size': '4',
            'shortlist': '2',
        },
        content_type='multipart/form-data',
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['recommended_shortlist'][0]['candidate_id'] == 'candidate_001'
    assert payload['recommended_shortlist'][0]['audio_url'].startswith('/api/artifact?path=')
    assert payload['pruning']['enabled'] is True
    assert payload['pruning']['deleted_candidate_count'] == 1



def test_fuse_upload_api_promotes_top_survivor_to_single_playable_output(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    uploads_dir = runs_dir / 'ui_uploads'
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'UPLOADS_DIR', uploads_dir)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        assert 'auto-shortlist-fusion' in cmd
        assert '--keep-non-survivors' in cmd
        outdir = Path(cmd[cmd.index('--output') + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        survivor_dir = outdir / 'candidate_001'
        survivor_dir.mkdir(parents=True, exist_ok=True)
        audio = survivor_dir / 'child_master.wav'
        audio.write_bytes(b'fake-wave')
        report = {
            'summary': ['Generated 4 candidates.', 'Shortlisted 1 survivor for human review.'],
            'recommended_shortlist': [
                {
                    'candidate_id': 'candidate_001',
                    'label': 'candidate_001',
                    'decision': 'survivor',
                    'listener_rank': 80.0,
                    'overall_score': 78.0,
                    'verdict': 'promising',
                    'run_dir': str(survivor_dir),
                    'audio_path': str(audio),
                    'top_reasons': ['good output'],
                    'top_fixes': [],
                }
            ],
            'closest_misses': [],
            'candidates': [],
        }
        (outdir / 'auto_shortlist_report.json').write_text(json.dumps(report), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='fuse ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post(
        '/api/fuse-upload',
        data={
            'song_a': (io.BytesIO(b'aaa'), 'song_a.mp3'),
            'song_b': (io.BytesIO(b'bbb'), 'song_b.mp3'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['result']['delivery_mode'] == 'survivor'
    assert payload['result']['selected_candidate_id'] == 'candidate_001'
    promoted = Path(payload['result']['audio_path'])
    assert promoted.exists()
    assert promoted.read_bytes() == b'fake-wave'
    assert payload['result']['audio_url'].startswith('/api/artifact?path=')



def test_fuse_upload_api_returns_best_playable_fallback_when_no_survivor_exists(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    uploads_dir = runs_dir / 'ui_uploads'
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'UPLOADS_DIR', uploads_dir)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        outdir = Path(cmd[cmd.index('--output') + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        miss_dir = outdir / 'candidate_002'
        miss_dir.mkdir(parents=True, exist_ok=True)
        audio = miss_dir / 'child_master.wav'
        audio.write_bytes(b'fallback-wave')
        report = {
            'summary': ['Generated 4 candidates.', 'No candidates survived the automatic gate.'],
            'recommended_shortlist': [],
            'closest_misses': [
                {
                    'candidate_id': 'candidate_002',
                    'label': 'candidate_002',
                    'decision': 'borderline',
                    'listener_rank': 59.0,
                    'overall_score': 59.0,
                    'verdict': 'mixed',
                    'run_dir': str(miss_dir),
                    'audio_path': str(audio),
                    'top_reasons': ['groove grid is not stable enough'],
                    'top_fixes': ['make it feel more like one song'],
                }
            ],
            'candidates': [],
        }
        (outdir / 'auto_shortlist_report.json').write_text(json.dumps(report), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='fuse ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post(
        '/api/fuse-upload',
        data={
            'song_a': (io.BytesIO(b'aaa'), 'song_a.mp3'),
            'song_b': (io.BytesIO(b'bbb'), 'song_b.mp3'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['result']['delivery_mode'] == 'fallback'
    assert payload['result']['selection_source'] == 'closest_miss'
    assert payload['result']['selected_candidate_id'] == 'candidate_002'
    promoted = Path(payload['result']['audio_path'])
    assert promoted.exists()
    assert promoted.read_bytes() == b'fallback-wave'



def test_fuse_upload_api_drops_to_deterministic_fallback_if_advanced_path_fails(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    uploads_dir = runs_dir / 'ui_uploads'
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'UPLOADS_DIR', uploads_dir)
    calls = []

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        calls.append((cmd, timeout))
        outdir = Path(cmd[cmd.index('--output') + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        if 'auto-shortlist-fusion' in cmd:
            return SimpleNamespace(returncode=2, stdout='advanced fail', stderr='advanced bad')
        assert 'fusion' in cmd
        audio = outdir / 'child_master.wav'
        audio.write_bytes(b'direct-wave')
        (outdir / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(audio)}}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='direct ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post(
        '/api/fuse-upload',
        data={
            'song_a': (io.BytesIO(b'aaa'), 'song_a.mp3'),
            'song_b': (io.BytesIO(b'bbb'), 'song_b.mp3'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['result']['delivery_mode'] == 'deterministic_fallback'
    assert payload['result']['selection_source'] == 'direct_fusion'
    assert payload['result']['selected_candidate_id'] == 'direct_fusion'
    assert 'advanced fail' in payload['stdout']
    assert 'direct ok' in payload['stdout']
    promoted = Path(payload['result']['audio_path'])
    assert promoted.exists()
    assert promoted.read_bytes() == b'direct-wave'
    assert any('auto-shortlist-fusion' in call for call, _ in calls)
    assert any('fusion' in call for call, _ in calls)
    advanced_timeout = next(timeout for call, timeout in calls if 'auto-shortlist-fusion' in call)
    direct_timeout = next(timeout for call, timeout in calls if 'fusion' in call)
    assert advanced_timeout == server.SIMPLE_FUSE_ADVANCED_TIMEOUT_SECONDS
    assert direct_timeout == server.SIMPLE_FUSE_DIRECT_TIMEOUT_SECONDS



def test_fuse_upload_api_drops_to_deterministic_fallback_if_advanced_path_times_out(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    uploads_dir = runs_dir / 'ui_uploads'
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'UPLOADS_DIR', uploads_dir)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        outdir = Path(cmd[cmd.index('--output') + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        if 'auto-shortlist-fusion' in cmd:
            raise server.subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        audio = outdir / 'child_master.wav'
        audio.write_bytes(b'direct-after-timeout')
        (outdir / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(audio)}}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='direct ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post(
        '/api/fuse-upload',
        data={
            'song_a': (io.BytesIO(b'aaa'), 'song_a.mp3'),
            'song_b': (io.BytesIO(b'bbb'), 'song_b.mp3'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['result']['delivery_mode'] == 'deterministic_fallback'
    assert payload['result']['selection_source'] == 'direct_fusion'
    assert 'timed out' in payload['result']['fallback_reason']
    promoted = Path(payload['result']['audio_path'])
    assert promoted.exists()
    assert promoted.read_bytes() == b'direct-after-timeout'



def test_fuse_upload_api_drops_to_deterministic_fallback_if_advanced_path_has_no_playable_audio(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    uploads_dir = runs_dir / 'ui_uploads'
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    monkeypatch.setattr(server, 'UPLOADS_DIR', uploads_dir)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        outdir = Path(cmd[cmd.index('--output') + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        if 'auto-shortlist-fusion' in cmd:
            report = {
                'summary': ['Generated 4 candidates.', 'No playable shortlist candidate could be promoted.'],
                'recommended_shortlist': [],
                'closest_misses': [],
                'candidates': [],
            }
            (outdir / 'auto_shortlist_report.json').write_text(json.dumps(report), encoding='utf-8')
            return SimpleNamespace(returncode=0, stdout='advanced empty', stderr='')
        audio = outdir / 'child_master.wav'
        audio.write_bytes(b'direct-from-empty')
        (outdir / 'render_manifest.json').write_text(json.dumps({'outputs': {'master_wav': str(audio)}}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='direct ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post(
        '/api/fuse-upload',
        data={
            'song_a': (io.BytesIO(b'aaa'), 'song_a.mp3'),
            'song_b': (io.BytesIO(b'bbb'), 'song_b.mp3'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['result']['delivery_mode'] == 'deterministic_fallback'
    assert payload['result']['selection_source'] == 'direct_fusion'
    assert payload['result']['fallback_reason']
    promoted = Path(payload['result']['audio_path'])
    assert promoted.exists()
    assert promoted.read_bytes() == b'direct-from-empty'



def test_status_includes_latest_auto_shortlist_summary(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir(parents=True)
    report_dir = runs_dir / 'auto_shortlist_case'
    report_dir.mkdir(parents=True)
    (report_dir / 'auto_shortlist_report.json').write_text(json.dumps({
        'summary': ['Generated 4 candidates.', 'Shortlisted 1 survivor for human review.'],
        'recommended_shortlist': [{'candidate_id': 'candidate_001', 'label': 'candidate_001'}],
        'closest_misses': [],
        'listener_agent_report': {'counts': {'survivors': 1, 'borderline': 1, 'rejected': 2}},
        'candidates': [{}, {}, {}, {}],
        'pairwise_pool': {'winner': 'candidate_001'},
    }), encoding='utf-8')
    monkeypatch.setattr(server, 'RUNS_DIR', runs_dir)
    client = app.test_client()
    response = client.get('/api/status')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['latest_auto_shortlist_result']['winner'] == 'candidate_001'
    assert payload['latest_auto_shortlist_result']['survivor_count'] == 1



def test_closed_loop_api_hydrates_dispatch_specs_to_cli_files(tmp_path, monkeypatch):
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
    seen_cmd: list[str] = []

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        seen_cmd[:] = cmd
        out_index = cmd.index('--output') + 1
        out_dir = Path(cmd[out_index])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'closed_loop_report.json').write_text(json.dumps({'best_iteration': None, 'stop_reason': 'no_change_command_configured', 'summary': []}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='closed-loop ok', stderr='')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)

    client = app.test_client()
    response = client.post('/api/closed-loop', json={
        'song_a': str(song_a),
        'song_b': str(song_b),
        'references': [str(reference)],
        'change_dispatch': {'command': 'echo patch {iteration}', 'timeout': 21},
        'test_dispatch': {'command': 'pytest -q tests/test_closed_loop_listener_runner.py', 'timeout': 55},
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'success'
    assert payload['config']['change_dispatch']['timeout'] == 21
    assert payload['config']['test_dispatch']['timeout'] == 55
    change_index = seen_cmd.index('--change-dispatch') + 1
    test_index = seen_cmd.index('--test-dispatch') + 1
    change_dispatch_path = Path(seen_cmd[change_index])
    test_dispatch_path = Path(seen_cmd[test_index])
    assert json.loads(change_dispatch_path.read_text(encoding='utf-8'))['command'] == 'echo patch {iteration}'
    assert json.loads(test_dispatch_path.read_text(encoding='utf-8'))['timeout'] == 55
