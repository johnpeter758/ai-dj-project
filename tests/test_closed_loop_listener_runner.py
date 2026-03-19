from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import closed_loop_listener_runner as loop


BASE_REPORT = {
    'source_path': 'dummy.wav',
    'duration_seconds': 120.0,
    'structure': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'groove': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'energy_arc': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'transition': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'coherence': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'mix_sanity': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'song_likeness': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'verdict': 'mixed',
    'top_reasons': [],
    'top_fixes': [],
    'gating': {'status': 'pass', 'raw_overall_score': 0.0},
    'analysis_version': '0.5.0',
}


def test_closed_loop_stops_on_quality_gate(monkeypatch, tmp_path: Path):
    scores = iter([72.0, 86.0])
    decisions = iter(["borderline", "survivor"])

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        score = next(scores)
        payload = dict(BASE_REPORT)
        payload['overall_score'] = score
        payload['verdict'] = 'promising' if score >= 85.0 else 'mixed'
        payload['gating'] = {'status': 'pass', 'raw_overall_score': score}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'schema_version': '0.2.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 72.0, 'gap_to_target': 27.0},
            'ranked_interventions': [{'component': 'song_likeness', 'code_targets': ['src/core/planner/arrangement.py']}],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/planner/arrangement.py'],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': next(decisions), 'listener_rank': 80.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)
    monkeypatch.setattr(loop, '_run_command_template', lambda *args, **kwargs: {'returncode': 0, 'stdout': '', 'stderr': '', 'command': ['noop'], 'command_text': 'noop'})

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop'),
        max_iterations=3,
        quality_gate=85.0,
        change_command='echo patch {iteration}',
    )

    assert report['stop_reason'] == 'quality_gate_reached:85.0'
    assert report['best_iteration']['iteration'] == 2
    assert report['best_iteration']['candidate_listener_decision'] == 'survivor'
    assert report['schema_version'] == loop.CLOSED_LOOP_SCHEMA_VERSION
    assert report['artifact_schema']['iteration_artifacts']['listen_feedback_brief']['kind'] == 'listen_feedback_brief'
    iter_one_artifacts = report['iterations'][0]['artifacts']
    assert iter_one_artifacts['render_output']['kind'] == 'render_output_dir'
    assert iter_one_artifacts['listen_feedback_brief']['schema_version']
    assert iter_one_artifacts['listener_assessment']['kind'] == 'listener_agent_case_assessment'
    assert Path(tmp_path / 'loop' / 'closed_loop_report.json').exists()



def test_closed_loop_does_not_stop_when_reference_weighted_gate_fails(monkeypatch, tmp_path: Path):
    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 91.0
        payload['verdict'] = 'promising'
        payload['gating'] = {'status': 'pass', 'raw_overall_score': 91.0}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'schema_version': '0.5.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 91.0, 'gap_to_target': 8.0},
            'quality_gate_diagnostics': {
                'reference_weighted': {
                    'candidate_weighted_score': 85.5,
                    'reference_weighted_score': 83.4,
                    'weighted_gap_vs_references': 2.1,
                    'top_blockers': [
                        {'component': 'song_likeness', 'weighted_gap_contribution': -10.34},
                    ],
                },
            },
            'ranked_interventions': [{'component': 'song_likeness', 'code_targets': ['src/core/planner/arrangement.py']}],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/planner/arrangement.py'],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'survivor', 'listener_rank': 92.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop_ref_gate'),
        max_iterations=2,
        quality_gate=90.0,
        change_command=None,
    )

    assert report['stop_reason'] == 'no_change_command_configured'
    gate_status = report['iterations'][0]['quality_gate_status']
    assert gate_status['passes_overall_gate'] is True
    assert gate_status['passes_reference_weighted_gate'] is False
    assert gate_status['reason'] == 'reference_weighted_score_below_gate'
    assert gate_status['blocking_components'] == ['song_likeness']



def test_closed_loop_stops_on_plateau_without_change_command(monkeypatch, tmp_path: Path):
    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 58.0
        payload['verdict'] = 'weak'
        payload['gating'] = {'status': 'reject', 'raw_overall_score': 58.0}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'schema_version': '0.2.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 58.0, 'gap_to_target': 41.0},
            'ranked_interventions': [{'component': 'song_likeness', 'code_targets': ['src/core/evaluation/listen.py']}],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/evaluation/listen.py'],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'reject', 'listener_rank': 40.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop'),
        max_iterations=4,
        quality_gate=90.0,
        plateau_limit=2,
        change_command=None,
    )

    assert report['stop_reason'] == 'no_change_command_configured'
    assert report['best_iteration']['candidate_overall_score'] == 58.0
    saved = json.loads((tmp_path / 'loop' / 'closed_loop_report.json').read_text(encoding='utf-8'))
    assert saved['stop_reason'] == 'no_change_command_configured'



def test_closed_loop_normalizes_reference_collection_before_feedback(monkeypatch, tmp_path: Path):
    ref_a = tmp_path / 'ref_a.wav'
    ref_b = tmp_path / 'ref_b.wav'
    ref_a.write_text('a', encoding='utf-8')
    ref_b.write_text('b', encoding='utf-8')
    refs = tmp_path / 'refs.txt'
    refs.write_text(f'{ref_a.name}\n{ref_b.name}\n{ref_a.name}\n', encoding='utf-8')

    seen_refs: list[str] = []

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 58.0
        payload['verdict'] = 'weak'
        payload['gating'] = {'status': 'reject', 'raw_overall_score': 58.0}
        return payload

    def fake_feedback(_candidate: str, refs_in: list[str], target_score: float = 99.0):
        seenRefs = refs_in
        seen_refs[:] = seenRefs
        return {
            'schema_version': '0.2.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 58.0, 'gap_to_target': 41.0},
            'ranked_interventions': [{'component': 'song_likeness', 'code_targets': ['src/core/evaluation/listen.py']}],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/evaluation/listen.py'],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'reject', 'listener_rank': 40.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=[str(refs)],
        output_root=str(tmp_path / 'loop_norm'),
        max_iterations=1,
        quality_gate=90.0,
        change_command=None,
    )

    assert report['stop_reason'] == 'max_iterations:1'
    assert seen_refs == [str(ref_a.resolve()), str(ref_b.resolve())]



def test_closed_loop_writes_change_packet_and_exposes_template_fields(monkeypatch, tmp_path: Path):
    commands: list[str] = []

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 61.0
        payload['verdict'] = 'weak'
        payload['top_reasons'] = ['not one song']
        payload['top_fixes'] = ['fix the opening lane']
        payload['gating'] = {'status': 'reject', 'raw_overall_score': 61.0}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'schema_version': '0.2.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 61.0, 'gap_to_target': 38.0},
            'next_code_targets': ['src/core/planner/arrangement.py', 'src/core/evaluation/listen.py'],
            'planner_feedback_map': [
                {
                    'failure_mode': 'backbone_continuity',
                    'component': 'song_likeness',
                    'confidence': 0.91,
                    'planner_code_targets': ['src/core/planner/arrangement.py', 'src/core/evaluation/listen.py'],
                    'matched_feedback': ['Improve backbone continuity and reduce cluttered donor carryover.'],
                    'actions': ['Strengthen backbone continuity guards.'],
                }
            ],
            'ranked_interventions': [
                {
                    'component': 'song_likeness',
                    'gap_vs_references': -24.5,
                    'problem': 'The render does not yet read like one intentional song.',
                    'code_targets': ['src/core/planner/arrangement.py'],
                    'actions': ['Keep one readable backbone lane instead of stitched strong chunks.'],
                }
            ],
            'render_feedback_map': [],
        }

    def fake_shell(command_template: str, context: dict[str, object], *, timeout: int = 3600, label: str):
        rendered = command_template.format(**context)
        commands.append(rendered)
        return {'returncode': 0, 'stdout': '', 'stderr': '', 'command': rendered.split(), 'command_text': rendered}

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'reject', 'listener_rank': 43.5})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)
    monkeypatch.setattr(loop, '_run_command_template', fake_shell)

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop_packet'),
        max_iterations=3,
        quality_gate=90.0,
        plateau_limit=1,
        change_command='echo {top_component}::{change_context_json}::{change_request_md}::{candidate_score}',
    )

    assert report['stop_reason'] == 'plateau:1'
    assert len(commands) == 1
    assert commands[0].startswith('echo song_likeness::')
    assert '::61.0' in commands[0]

    iteration = report['iterations'][0]
    context_path = Path(iteration['change_context_json'])
    request_path = Path(iteration['change_request_md'])
    assert context_path.exists()
    assert request_path.exists()

    payload = json.loads(context_path.read_text(encoding='utf-8'))
    assert payload['schema_version'] == loop.CHANGE_COMMAND_CONTEXT_SCHEMA_VERSION
    assert payload['top_intervention']['component'] == 'song_likeness'
    assert payload['candidate']['overall_score'] == 61.0
    assert payload['next_code_targets'] == ['src/core/planner/arrangement.py', 'src/core/evaluation/listen.py']
    assert payload['planner_feedback_map'][0]['failure_mode'] == 'backbone_continuity'
    assert payload['prioritized_execution_plan'] == []
    assert iteration['candidate_listener_decision'] == 'reject'
    assert iteration['candidate_listener_rank'] == 43.5
    assert Path(iteration['listener_assessment_path']).exists()

    request_text = request_path.read_text(encoding='utf-8')
    assert iteration['artifacts']['change_context']['kind'] == 'change_command_context'
    assert iteration['artifacts']['change_request']['kind'] == 'change_request_markdown'
    assert iteration['artifacts']['change_context']['exists'] is True
    assert 'Closed-loop change request' in request_text
    assert 'song_likeness' in request_text
    assert 'src/core/planner/arrangement.py' in request_text
    assert 'Planner feedback routes' in request_text
    assert 'backbone_continuity' in request_text


def test_closed_loop_writes_structured_loop_summary(monkeypatch, tmp_path: Path):
    scores = iter([61.0, 66.5])

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        score = next(scores)
        payload = dict(BASE_REPORT)
        payload['overall_score'] = score
        payload['verdict'] = 'mixed' if score < 65.0 else 'promising'
        payload['gating'] = {'status': 'review', 'raw_overall_score': score}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'schema_version': '0.2.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 61.0, 'gap_to_target': 38.0},
            'ranked_interventions': [
                {
                    'component': 'song_likeness',
                    'gap_vs_references': -18.0,
                    'problem': 'Still not one coherent song.',
                    'code_targets': ['src/core/planner/arrangement.py'],
                    'actions': ['Keep one readable backbone lane.'],
                }
            ],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/planner/arrangement.py'],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'borderline', 'listener_rank': 61.5})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)
    monkeypatch.setattr(loop, '_run_command_template', lambda *args, **kwargs: {'returncode': 0, 'stdout': '', 'stderr': '', 'command': ['noop'], 'command_text': 'noop'})

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop_summary'),
        max_iterations=2,
        quality_gate=90.0,
        change_command='echo patch {iteration}',
    )

    loop_summary = report['loop_summary']
    assert loop_summary['total_iterations'] == 2
    assert loop_summary['score_trajectory'] == [61.0, 66.5]
    assert loop_summary['net_improvement'] == 5.5
    assert loop_summary['best_iteration']['iteration'] == 2
    assert loop_summary['best_iteration']['candidate_listener_decision'] == 'borderline'
    assert loop_summary['best_top_intervention']['component'] == 'song_likeness'
    assert loop_summary['iteration_summaries'][0]['top_intervention']['component'] == 'song_likeness'
    assert loop_summary['iteration_summaries'][0]['candidate_listener_decision'] == 'borderline'
    assert any('Net improvement across the run was 5.5 points.' == line for line in report['summary'])

    saved = json.loads((tmp_path / 'loop_summary' / 'closed_loop_report.json').read_text(encoding='utf-8'))
    assert saved['loop_summary']['iteration_summaries'][1]['candidate_verdict'] == 'promising'



def test_closed_loop_runs_test_command_after_successful_change(monkeypatch, tmp_path: Path):
    command_log: list[tuple[str, int, dict[str, object]]] = []
    scores = iter([62.0, 91.0])
    decisions = iter(['reject', 'survivor'])

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        score = next(scores)
        payload = dict(BASE_REPORT)
        payload['overall_score'] = score
        payload['verdict'] = 'promising' if score >= 90.0 else 'weak'
        payload['gating'] = {'status': 'pass' if score >= 90.0 else 'reject', 'raw_overall_score': score}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'schema_version': '0.2.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 62.0, 'gap_to_target': 37.0},
            'ranked_interventions': [
                {
                    'component': 'song_likeness',
                    'gap_vs_references': -20.0,
                    'problem': 'Still not reading like one coherent song.',
                    'code_targets': ['src/core/planner/arrangement.py'],
                    'actions': ['Strengthen the backbone opening lane.'],
                }
            ],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/planner/arrangement.py'],
        }

    def fake_run(command_template: str, context: dict[str, object], *, timeout: int = 3600, label: str):
        command_log.append((label, timeout, dict(context)))
        return {
            'returncode': 0,
            'stdout': '',
            'stderr': '',
            'command': [label],
            'command_text': f'{label} ok',
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': next(decisions), 'listener_rank': 88.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)
    monkeypatch.setattr(loop, '_run_command_template', fake_run)

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop_test_command'),
        max_iterations=2,
        quality_gate=90.0,
        change_command='echo patch {iteration}',
        test_command='pytest {change_context_json}',
    )

    assert report['stop_reason'] == 'quality_gate_reached:90.0'
    assert [entry[0] for entry in command_log] == ['change', 'test']
    change_context = command_log[0][2]
    test_context = command_log[1][2]
    assert test_context['change_context_json'] == change_context['change_context_json']
    assert test_context['change_request_md'] == change_context['change_request_md']

    iteration = report['iterations'][0]
    assert iteration['change_command']['command_text'] == 'change ok'
    assert iteration['test_command']['command_text'] == 'test ok'
    assert iteration['artifacts']['change_context']['exists'] is True
    assert iteration['artifacts']['change_request']['exists'] is True



def test_closed_loop_stops_on_test_command_failure(monkeypatch, tmp_path: Path):
    command_log: list[str] = []

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 64.0
        payload['verdict'] = 'mixed'
        payload['gating'] = {'status': 'review', 'raw_overall_score': 64.0}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'schema_version': '0.2.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 64.0, 'gap_to_target': 35.0},
            'ranked_interventions': [
                {
                    'component': 'transition',
                    'gap_vs_references': -14.0,
                    'problem': 'Transitions still expose audible track switches.',
                    'code_targets': ['src/core/render/renderer.py'],
                    'actions': ['Tighten handoff cleanup and arrival filtering.'],
                }
            ],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/render/renderer.py'],
        }

    def fake_run(command_template: str, context: dict[str, object], *, timeout: int = 3600, label: str):
        command_log.append(label)
        if label == 'change':
            return {
                'returncode': 0,
                'stdout': '',
                'stderr': '',
                'command': ['change'],
                'command_text': 'change ok',
            }
        return {
            'returncode': 3,
            'stdout': '',
            'stderr': 'tests failed',
            'command': ['test'],
            'command_text': 'test failed',
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'borderline', 'listener_rank': 52.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)
    monkeypatch.setattr(loop, '_run_command_template', fake_run)

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop_test_failure'),
        max_iterations=3,
        quality_gate=90.0,
        change_command='echo patch {iteration}',
        test_command='pytest tests/test_closed_loop_listener_runner.py',
    )

    assert report['stop_reason'] == 'test_command_failed:1'
    assert command_log == ['change', 'test']
    assert len(report['iterations']) == 1
    iteration = report['iterations'][0]
    assert iteration['change_command']['returncode'] == 0
    assert iteration['test_command']['returncode'] == 3
    saved = json.loads((tmp_path / 'loop_test_failure' / 'closed_loop_report.json').read_text(encoding='utf-8'))
    assert saved['stop_reason'] == 'test_command_failed:1'



def test_closed_loop_plateau_resets_on_listener_decision_progress_even_without_overall_gain(monkeypatch, tmp_path: Path):
    scores = iter([64.0, 64.0, 64.0, 64.0])
    decisions = iter(['reject', 'borderline', 'borderline', 'borderline'])
    ranks = iter([40.0, 40.0, 40.0, 40.0])

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        score = next(scores)
        payload = dict(BASE_REPORT)
        payload['overall_score'] = score
        payload['verdict'] = 'mixed'
        payload['gating'] = {'status': 'review', 'raw_overall_score': score}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'schema_version': '0.2.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 64.0, 'gap_to_target': 35.0},
            'ranked_interventions': [{'component': 'song_likeness', 'code_targets': ['src/core/planner/arrangement.py']}],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/planner/arrangement.py'],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': next(decisions), 'listener_rank': next(ranks)})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)
    monkeypatch.setattr(loop, '_run_command_template', lambda *args, **kwargs: {'returncode': 0, 'stdout': '', 'stderr': '', 'command': ['noop'], 'command_text': 'noop'})

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop_listener_progress'),
        max_iterations=4,
        quality_gate=90.0,
        plateau_limit=2,
        change_command='echo patch {iteration}',
    )

    assert report['stop_reason'] == 'plateau:2'
    assert len(report['iterations']) == 4
    assert report['iterations'][0]['plateau_count'] == 0
    assert report['iterations'][1]['plateau_count'] == 0
    assert report['iterations'][2]['plateau_count'] == 1
    assert report['iterations'][3]['plateau_count'] == 2
    assert report['iterations'][1]['improved_vs_best_before'] is True
    assert report['best_iteration']['iteration'] == 2



def test_closed_loop_plateau_resets_on_reference_weighted_gain_even_when_overall_is_flat(monkeypatch, tmp_path: Path):
    weighted_scores = iter([70.0, 71.0, 71.0, 71.0])

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 64.0
        payload['verdict'] = 'mixed'
        payload['gating'] = {'status': 'review', 'raw_overall_score': 64.0}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        weighted = next(weighted_scores)
        return {
            'schema_version': '0.5.0',
            'goal': {'target_listener_score': target_score, 'current_overall_score': 64.0, 'gap_to_target': 35.0},
            'quality_gate_diagnostics': {
                'reference_weighted': {
                    'candidate_weighted_score': weighted,
                    'reference_weighted_score': 80.0,
                    'weighted_gap_vs_references': weighted - 80.0,
                    'top_blockers': [{'component': 'song_likeness', 'weighted_gap_contribution': -9.0}],
                },
            },
            'ranked_interventions': [{'component': 'song_likeness', 'code_targets': ['src/core/planner/arrangement.py']}],
            'planner_feedback_map': [],
            'render_feedback_map': [],
            'next_code_targets': ['src/core/planner/arrangement.py'],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'reject', 'listener_rank': 40.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)
    monkeypatch.setattr(loop, '_run_command_template', lambda *args, **kwargs: {'returncode': 0, 'stdout': '', 'stderr': '', 'command': ['noop'], 'command_text': 'noop'})

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop_weighted_progress'),
        max_iterations=4,
        quality_gate=90.0,
        plateau_limit=2,
        min_improvement=0.5,
        change_command='echo patch {iteration}',
    )

    assert report['stop_reason'] == 'plateau:2'
    assert len(report['iterations']) == 4
    assert report['iterations'][1]['plateau_count'] == 0
    assert report['iterations'][2]['plateau_count'] == 1
    assert report['iterations'][3]['plateau_count'] == 2
    assert report['iterations'][1]['improved_vs_best_before'] is True
    assert report['best_iteration']['iteration'] == 2



def test_closed_loop_rejects_malformed_feedback_brief(monkeypatch, tmp_path: Path):
    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 58.0
        payload['verdict'] = 'weak'
        payload['gating'] = {'status': 'reject', 'raw_overall_score': 58.0}
        return payload

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'reject', 'listener_rank': 40.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', lambda *_args, **_kwargs: {'schema_version': '0.2.0', 'ranked_interventions': []})

    with pytest.raises(loop.LoopError, match='feedback brief missing goal dict'):
        loop.run_closed_loop(
            song_a='a.wav',
            song_b='b.wav',
            references=['ref.wav'],
            output_root=str(tmp_path / 'loop_bad_feedback'),
            max_iterations=1,
            quality_gate=90.0,
        )



def test_run_command_template_formats_placeholders_without_shell_splitting():
    result = loop._run_command_template(
        "python3 -c {script} {top_actions}",
        {
            'script': "import sys; print(sys.argv[1])",
            'top_actions': 'keep one readable backbone lane',
        },
        label='change',
    )

    assert result['returncode'] == 0
    assert result['command'][3] == 'keep one readable backbone lane'
    assert result['stdout'].splitlines() == ['keep one readable backbone lane']


def test_command_template_fields_text_documents_change_artifacts_and_examples():
    text = loop._command_template_fields_text()

    assert '{feedback_json}' in text
    assert '{change_context_json}' in text
    assert '{change_request_md}' in text
    assert '{top_actions}' in text
    assert '--change-command "python scripts/your_patch_step.py --context {change_context_json}"' in text


def test_main_prints_template_fields_without_required_positionals(monkeypatch, capsys):
    monkeypatch.setattr(loop.sys, 'argv', ['closed_loop_listener_runner.py', '--print-template-fields'])

    rc = loop.main()

    assert rc == 0
    output = capsys.readouterr().out
    assert 'Closed-loop command template fields' in output
    assert '{candidate_score}' in output
    assert '{top_code_targets}' in output


def test_closed_loop_rejects_feedback_brief_missing_schema_version(monkeypatch, tmp_path: Path):
    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 58.0
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'goal': {'target_listener_score': target_score, 'current_overall_score': 58.0, 'gap_to_target': 41.0},
            'ranked_interventions': [],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, '_candidate_listener_assessment', lambda _candidate_input: {'decision': 'reject', 'listener_rank': 40.0})
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)

    with pytest.raises(loop.LoopError, match='schema_version'):
        loop.run_closed_loop(
            song_a='a.wav',
            song_b='b.wav',
            references=['ref.wav'],
            output_root=str(tmp_path / 'loop_bad_brief'),
            max_iterations=1,
        )


@pytest.mark.parametrize('template', [
    'python3 -c pass && echo unsafe',
    'python3 -c pass; echo unsafe',
    'python3 -c pass | cat',
    'python3 -c $(echo unsafe)',
    'python3 -c `echo unsafe`',
])
def test_run_command_template_rejects_shell_operators(template: str):
    with pytest.raises(loop.LoopError):
        loop._run_command_template(template, {}, label='change')


def test_hydrate_dispatch_spec_uses_command_and_timeout():
    hydrated = loop._hydrate_dispatch_spec({'command': 'echo {iteration}', 'timeout': 17}, label='change')

    assert hydrated['command'] == 'echo {iteration}'
    assert hydrated['timeout'] == 17
    assert hydrated['schema_version'] == loop.DISPATCH_SPEC_SCHEMA_VERSION


def test_main_reads_change_dispatch_spec_and_passes_hydrated_payload(tmp_path: Path, monkeypatch):
    dispatch_path = tmp_path / 'change_dispatch.json'
    dispatch_path.write_text(json.dumps({'command': 'echo patch {iteration}', 'timeout': 19}), encoding='utf-8')
    seen: dict[str, object] = {}

    def fake_run_closed_loop(**kwargs):
        seen.update(kwargs)
        return {'best_iteration': None, 'stop_reason': 'no_change_command_configured', 'summary': []}

    monkeypatch.setattr(loop, 'run_closed_loop', fake_run_closed_loop)
    monkeypatch.setattr(loop.sys, 'argv', [
        'closed_loop_listener_runner.py',
        'song_a.wav',
        'song_b.wav',
        'ref.wav',
        '--output-root',
        str(tmp_path / 'out'),
        '--change-dispatch',
        str(dispatch_path),
    ])

    rc = loop.main()

    assert rc == 0
    assert seen['change_dispatch']['command'] == 'echo patch {iteration}'
    assert seen['change_dispatch']['timeout'] == 19
    assert seen['change_command'] is None
